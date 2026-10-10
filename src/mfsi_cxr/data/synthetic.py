"""Synthetic CheXpert-Plus-like raw data for tests and the smoke profile. No real patient data.

Every forbidden-for-index text section of every study carries a unique canary token built by
`canary(section, study_id)`, so a leak can be traced to the exact study and section. Study folder
numbers are deliberately NOT chronological: time order comes only from the order column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS

# Sections that may never reach the index text. `section_new_unreviewed` is not in the config at all:
# it checks that an unknown column is forbidden by default.
CANARY_SECTIONS: tuple[str, ...] = (
    "section_findings",
    "section_impression",
    "section_narrative",
    "section_summary",
    "section_end_of_impression",
    "section_comparison",
    "section_procedure_comments",
    "section_new_unreviewed",
)
PRE_READ_SECTIONS: tuple[str, ...] = ("section_clinical_history", "section_history", "section_technique")
# Probability of each label state when drawing synthetic labels: positive, negative, uncertain, blank.
LABEL_STATE_PROBS = (0.3, 0.4, 0.1, 0.2)


def study_tag(study_id: str) -> str:
    """'patient00003/study2' -> 'patient00003_study2' (a token-safe id)."""
    return study_id.replace("/", "_")


def canary(section: str, study_id: str) -> str:
    return f"CANARY_{section.removeprefix('section_').upper()}_{study_tag(study_id)}"


def pre_read_text(section: str, study_id: str) -> str:
    return f"pre-read {section.removeprefix('section_')} of {study_tag(study_id)}"


def make_synthetic_dataset(root: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    """Write metadata, labels and tiny PNGs under `root`. Returns the written file paths."""
    syn = cfg["synthetic"]
    rng = np.random.default_rng(syn["seed"])
    cols = cfg["data"]["columns"]
    rows: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []

    for p in range(syn["n_patients"]):
        patient = f"patient{p:05d}"
        n_studies = int(rng.integers(1, syn["max_studies"] + 1))
        folder_numbers = rng.permutation(np.arange(1, n_studies + 1))  # not chronological on purpose
        is_radiologist = p % syn["radiologist_every"] == syn["radiologist_every"] - 1
        age = int(rng.integers(20, 90))
        sex = str(rng.choice(["F", "M"]))
        for order, number in enumerate(folder_numbers, start=1):
            study_id = f"{patient}/study{number}"
            no_pre_read = rng.random() < syn["p_no_pre_read"]
            sections = {s: (None if no_pre_read else pre_read_text(s, study_id)) for s in PRE_READ_SECTIONS}
            sections |= {s: canary(s, study_id) for s in CANARY_SECTIONS}
            report = " ".join(v for v in sections.values() if v)
            labels = _draw_labels(rng)
            n_images = int(rng.integers(1, syn["max_images"] + 1))
            for k in range(1, n_images + 1):
                view = "Frontal" if k == 1 or rng.random() < 0.5 else "Lateral"
                rel = f"{'valid' if is_radiologist else 'train'}/{study_id}/view{k}_{view.lower()}.png"
                rows.append(
                    {
                        cols["image_path"]: rel,
                        cols["patient_id"]: patient,
                        cols["study_order"]: order,
                        cols["view"]: view,
                        cols["source_split"]: "valid" if is_radiologist else "train",
                        "age": age,
                        "sex": sex,
                        "race": "synthetic",
                        **sections,
                        "report": report,
                    }
                )
                label_records.append({cols["image_path"]: rel, **labels})

    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / cfg["data"]["metadata_file"]
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    labels_path = root / cfg["data"]["labels_file"]
    with labels_path.open("w", encoding="utf-8") as f:
        for rec in label_records:
            f.write(json.dumps(rec) + "\n")
    _write_images(root, [r[cols["image_path"]] for r in rows], syn["image_size"], rng)
    return {"metadata": meta_path, "labels": labels_path}


def _draw_labels(rng: np.random.Generator) -> dict[str, float | None]:
    states = [1.0, 0.0, -1.0, None]
    return {f: states[int(rng.choice(len(states), p=LABEL_STATE_PROBS))] for f in FINDINGS}


def _write_images(root: Path, rel_paths: list[str], size: int, rng: np.random.Generator) -> None:
    from PIL import Image

    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (size, size), dtype=np.uint8), mode="L").save(path)
