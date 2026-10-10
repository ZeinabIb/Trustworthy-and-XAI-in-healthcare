"""Image index (one row per image) with study time order, and patient-level pilot sampling (Phase 1.2)."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS, POSITIVE

STUDY_DIR = re.compile(r"^study(\d+)$")
INDEX_COLUMNS = ["patient_id", "study_id", "study_rank", "image_path", "view", "source_split"]


class StudyOrderError(ValueError):
    """Study time order is missing or ambiguous (escalate to Zeinab: priors depend on it)."""


def parse_study(image_path: str) -> tuple[str, int]:
    """'train/patient00001/study3/view1_frontal.png' -> ('patient00001/study3', 3)."""
    parts = PurePosixPath(str(image_path).replace("\\", "/")).parts
    if len(parts) < 3 or not STUDY_DIR.match(parts[-2]):
        raise ValueError(f"cannot parse study from image path {image_path!r}")
    return f"{parts[-3]}/{parts[-2]}", int(STUDY_DIR.match(parts[-2]).group(1))


def build_image_table(meta: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Normalised image table from the raw metadata (non-text columns only)."""
    cols = cfg["data"]["columns"]
    parsed = meta[cols["image_path"]].map(parse_study)
    return pd.DataFrame(
        {
            "patient_id": meta[cols["patient_id"]].astype(str).to_numpy(),
            "study_id": [p[0] for p in parsed],
            "study_num": [p[1] for p in parsed],
            "study_order": meta[cols["study_order"]].to_numpy(),
            "image_path": meta[cols["image_path"]].astype(str).str.replace("\\", "/").to_numpy(),
            "view": meta[cols["view"]].astype(str).to_numpy(),
            "source_split": meta[cols["source_split"]].astype(str).to_numpy(),
        }
    )


OVERLAP_POLICIES = ("drop_patients", "error")


def overlapping_patients(images: pd.DataFrame) -> set[str]:
    """Patients with a study whose [min, max] order interval overlaps an earlier study's interval."""
    iv = images.groupby("study_id").agg(
        patient_id=("patient_id", "first"), lo=("study_order", "min"), hi=("study_order", "max")
    )
    iv = iv.sort_values(["patient_id", "lo"])
    prev_hi = iv.groupby("patient_id")["hi"].transform(lambda s: s.cummax().shift())
    return set(iv.loc[prev_hi.notna() & (iv["lo"] <= prev_hi), "patient_id"])


def assign_study_rank(images: pd.DataFrame, overlap_policy: str) -> tuple[pd.DataFrame, set[str]]:
    """Add study_rank (0 = earliest) per patient. Returns (images, dropped patient ids).

    The order column may rank images rather than studies (CheXpert Plus does): a study's time is the
    order of its earliest image. If two studies of a patient interleave in time, their order is
    ambiguous: `drop_patients` removes those patients, `error` raises (escalate).
    """
    if overlap_policy not in OVERLAP_POLICIES:
        raise ValueError(f"unknown overlap policy {overlap_policy!r}; expected one of {OVERLAP_POLICIES}")
    if images["study_order"].isna().any():
        raise StudyOrderError("study_order has missing values")
    ambiguous = overlapping_patients(images)
    if ambiguous and overlap_policy == "error":
        raise StudyOrderError(f"{len(ambiguous)} patients have studies interleaved in time")
    images = images[~images["patient_id"].isin(ambiguous)]
    studies = images.groupby("study_id").agg(patient_id=("patient_id", "first"), start=("study_order", "min"))
    dup = studies.duplicated(["patient_id", "start"], keep=False)
    if dup.any():
        raise StudyOrderError(f"ties in study_order within patients: {list(studies.index[dup][:5])}")
    studies = studies.sort_values(["patient_id", "start"])
    studies["study_rank"] = studies.groupby("patient_id").cumcount()
    ranked = images.merge(studies[["study_rank"]], left_on="study_id", right_index=True, how="left")
    return ranked, ambiguous


def sample_patients(
    images: pd.DataFrame, labels: pd.DataFrame, n: int | None, seed: int, min_pos: int
) -> list[str]:
    """Seeded patient sample keeping ALL studies of each patient.

    First, rarest finding first, add patients with a positive label until each finding has `min_pos`
    positive patients (using at most half the budget). Then fill the rest, keeping the share of
    patients with >=2 studies equal to the full population.
    """
    patients = np.array(sorted(images["patient_id"].unique()))
    if n is None or n >= len(patients):
        return patients.tolist()
    rng = np.random.default_rng(seed)
    study_patient = images.drop_duplicates("study_id").set_index("study_id")["patient_id"]
    pos = labels.set_index("study_id")[list(FINDINGS)].eq(POSITIVE)
    pos_patients = {f: set(study_patient[pos.index[pos[f]]]) for f in FINDINGS}

    chosen: list[str] = []
    budget = n // 2
    for finding in sorted(FINDINGS, key=lambda f: len(pos_patients[f])):
        have = len(pos_patients[finding] & set(chosen))
        candidates = sorted(pos_patients[finding] - set(chosen))
        for pid in rng.permutation(candidates) if candidates else []:
            if have >= min_pos or len(chosen) >= budget:
                break
            chosen.append(str(pid))
            have += 1

    n_studies = images.groupby("patient_id")["study_id"].nunique()
    remaining = [p for p in patients if p not in set(chosen)]
    multi = [p for p in remaining if n_studies[p] >= 2]
    single = [p for p in remaining if n_studies[p] < 2]
    to_fill = n - len(chosen)
    multi_share = float((n_studies >= 2).mean())
    n_multi = min(len(multi), round(to_fill * multi_share))
    n_single = min(len(single), to_fill - n_multi)
    n_multi = min(len(multi), to_fill - n_single)
    chosen += [str(p) for p in rng.permutation(multi)[:n_multi]]
    chosen += [str(p) for p in rng.permutation(single)[:n_single]]
    return sorted(chosen)


def finalise_index(images: pd.DataFrame) -> pd.DataFrame:
    ordered = images[INDEX_COLUMNS].sort_values(["patient_id", "study_rank", "image_path"])
    return ordered.reset_index(drop=True)
