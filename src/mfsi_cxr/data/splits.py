"""Patient-level train/val/test split, created once, seeded (Phase 1.4).

Patients in the dataset's radiologist-read split are flagged and, under policy `holdout`, kept out of
train/val/test entirely (they are an extra evaluation only; never used for tuning).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import HOLDOUT, SPLITS

POLICIES = ("holdout",)


def radiologist_patients(index: pd.DataFrame, cfg: dict[str, Any]) -> set[str]:
    value = cfg["data"]["radiologist_split_value"]
    return set(index.loc[index["source_split"] == value, "patient_id"])


def make_splits(
    patients: list[str], radiologist: set[str], fractions: dict[str, float], seed: int, policy: str
) -> pd.DataFrame:
    if policy not in POLICIES:
        raise ValueError(f"unknown radiologist policy {policy!r}; expected one of {POLICIES}")
    if abs(sum(fractions[s] for s in SPLITS) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1: {fractions}")
    pool = np.array(sorted(set(patients) - radiologist))
    order = np.random.default_rng(seed).permutation(pool)
    n_train = int(round(fractions["train"] * len(order)))
    n_val = int(round(fractions["val"] * len(order)))
    assigned = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    rows = [(str(p), s, False) for s, ps in assigned.items() for p in ps]
    rows += [(p, HOLDOUT, True) for p in sorted(set(patients) & radiologist)]
    out = pd.DataFrame(rows, columns=["patient_id", "split", "radiologist_subset"])
    return out.sort_values("patient_id").reset_index(drop=True)
