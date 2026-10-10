"""Per-study labels for the 14 findings, normalised to {1, 0, -1, NaN} (Phase 1.3).

Label files (checked on the real data): JSON lines, one record per image, `path_to_image` (.jpg) plus one
key per finding. Paths are matched without extension, so labels keyed by .jpg join .png image paths.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS, LABEL_VALUES, NEGATIVE, POSITIVE, SPLITS, UNCERTAIN
from mfsi_cxr.utils.logging import get_logger

log = get_logger("labels")
# How labels of one study's images are combined when they differ (their reports can differ slightly).
AGGREGATIONS = ("priority", "error")
LABEL_PRIORITY = (POSITIVE, UNCERTAIN, NEGATIVE)


class LabelError(ValueError):
    """Unexpected label value or labels that disagree between images of one study."""


def read_label_records(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return pd.DataFrame(json.loads(text))
    return pd.DataFrame([json.loads(line) for line in text.splitlines() if line.strip()])


def path_key(image_path: str) -> str:
    return str(PurePosixPath(str(image_path).replace("\\", "/")).with_suffix(""))


def normalise_value(value: Any) -> float:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return np.nan
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelError(f"unexpected label value {value!r}") from exc
    if np.isnan(out):
        return np.nan
    if out not in LABEL_VALUES:
        raise LabelError(f"unexpected label value {value!r}")
    return out


def labels_per_study(
    records: pd.DataFrame, images: pd.DataFrame, image_path_col: str, aggregation: str
) -> pd.DataFrame:
    """One row per study_id with the 14 finding columns in FINDINGS order."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown label aggregation {aggregation!r}; expected one of {AGGREGATIONS}")
    missing = [f for f in FINDINGS if f not in records.columns]
    if missing:
        raise LabelError(f"label file lacks findings: {missing}")
    recs = records[[image_path_col, *FINDINGS]].copy()
    for f in FINDINGS:
        recs[f] = recs[f].map(normalise_value).astype("float64")
    recs["key"] = recs[image_path_col].map(path_key)
    img = images[["image_path", "study_id"]].assign(key=images["image_path"].map(path_key))
    joined = img.merge(recs.drop(columns=[image_path_col]), on="key", how="inner")
    disagree = joined.groupby("study_id")[list(FINDINGS)].nunique(dropna=False).gt(1).any(axis=1)
    if disagree.any():
        if aggregation == "error":
            raise LabelError(f"labels differ between images of studies {list(disagree[disagree].index[:5])}")
        log.warning("%d studies have labels that differ between images; combined by %r",
                    int(disagree.sum()), aggregation)
    out = _combine(joined, list(FINDINGS))
    return out.sort_values("study_id").reset_index(drop=True)


def _combine(joined: pd.DataFrame, findings: list[str]) -> pd.DataFrame:
    """Per-study label from its images' labels: positive > uncertain > negative > blank (vectorised)."""
    rank_of = {state: len(LABEL_PRIORITY) - i for i, state in enumerate(LABEL_PRIORITY)}  # blank -> 0
    state_of = {rank: state for state, rank in rank_of.items()} | {0: np.nan}
    ranks = joined[findings].apply(lambda col: col.map(rank_of).fillna(0))
    best = ranks.groupby(joined["study_id"]).max()
    return best.apply(lambda col: col.map(state_of)).astype("float64").reset_index()


def prevalence_table(
    labels: pd.DataFrame, index: pd.DataFrame, splits: pd.DataFrame, split: str
) -> pd.DataFrame:
    """n_pos / n_neg / n_uncertain / n_blank per finding over the studies of one split."""
    if split not in SPLITS:
        raise ValueError(split)
    patients = set(splits.loc[splits["split"] == split, "patient_id"])
    studies = set(index.loc[index["patient_id"].isin(patients), "study_id"])
    sub = labels[labels["study_id"].isin(studies)]
    return pd.DataFrame(
        {
            "finding": list(FINDINGS),
            "n_pos": [int((sub[f] == POSITIVE).sum()) for f in FINDINGS],
            "n_neg": [int((sub[f] == NEGATIVE).sum()) for f in FINDINGS],
            "n_uncertain": [int((sub[f] == UNCERTAIN).sum()) for f in FINDINGS],
            "n_blank": [int(sub[f].isna().sum()) for f in FINDINGS],
        }
    )
