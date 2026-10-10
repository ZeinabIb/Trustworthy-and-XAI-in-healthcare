"""Fixed project constants (from 00_OVERVIEW.md). Order of FINDINGS is the label column order everywhere.

The parent/child tree is NOT encoded yet: it must first be checked against the original CheXpert paper
(STOP-AND-ASK in 00_OVERVIEW.md). Phase 1 does not need it.
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

# Label states in the CheXpert files; blank / missing stays NaN ("not mentioned").
POSITIVE, NEGATIVE, UNCERTAIN = 1.0, 0.0, -1.0
LABEL_VALUES: frozenset[float] = frozenset({POSITIVE, NEGATIVE, UNCERTAIN})

SPLITS: tuple[str, ...] = ("train", "val", "test")
HOLDOUT = "holdout"  # radiologist-read subset, never used for training or tuning
