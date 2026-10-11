"""Fixed project constants (from 00_OVERVIEW.md). Order of FINDINGS is the label column order everywhere.

The parent/child tree was checked on 2026-10-11 against the CheXpert label tree (Pham et al.,
arXiv 1911.06475, Fig. 1, "proposed by Irvin et al.") and approved by Zeinab (docs/DECISIONS.md). The figure
also draws Lung Opacity -> Pneumonia; that edge is implied here through Consolidation, so hierarchy checks
must use ancestors(), never only the direct parent. The classifier head stays flat (hard rule 5).
"""

from __future__ import annotations

FINDINGS: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

# Child -> parent. Findings not listed are roots.
PARENT: dict[str, str] = {
    "Cardiomegaly": "Enlarged Cardiomediastinum",
    "Lung Lesion": "Lung Opacity",
    "Edema": "Lung Opacity",
    "Consolidation": "Lung Opacity",
    "Atelectasis": "Lung Opacity",
    "Pneumonia": "Consolidation",
}
NO_FINDING = "No Finding"
# CheXpert: No Finding is positive only when no pathology is positive or uncertain. Support Devices is
# not a pathology, so it may co-occur with No Finding.
NON_PATHOLOGY: frozenset[str] = frozenset({NO_FINDING, "Support Devices"})
PATHOLOGIES: tuple[str, ...] = tuple(f for f in FINDINGS if f not in NON_PATHOLOGY)


def ancestors(finding: str) -> tuple[str, ...]:
    """All ancestors of a finding, nearest first ('Pneumonia' -> ('Consolidation', 'Lung Opacity'))."""
    out: list[str] = []
    while finding in PARENT:
        finding = PARENT[finding]
        out.append(finding)
    return tuple(out)


# Label states in the CheXpert files; blank / missing stays NaN ("not mentioned").
POSITIVE, NEGATIVE, UNCERTAIN = 1.0, 0.0, -1.0
LABEL_VALUES: frozenset[float] = frozenset({POSITIVE, NEGATIVE, UNCERTAIN})

SPLITS: tuple[str, ...] = ("train", "val", "test")
HOLDOUT = "holdout"  # radiologist-read subset, never used for training or tuning
