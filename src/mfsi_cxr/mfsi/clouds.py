"""Class clouds per finding (Phase 3, step 1). Built from TRAIN embeddings only.

P_l = {i : y_il = 1}, N_l = {i : y_il in {0, NaN}} (blank = not mentioned = negative). Uncertain (-1) is
in neither cloud under `ignore`; under `ones` (ablation) it joins the positive cloud. Every sample is scored.
"""

from __future__ import annotations

import numpy as np

from mfsi_cxr.constants import NEGATIVE, POSITIVE, UNCERTAIN

UNCERTAIN_POLICIES = ("ignore", "ones")


def cloud_masks(y: np.ndarray, policy: str) -> tuple[np.ndarray, np.ndarray]:
    """y (N, L) in {1, 0, -1, NaN} -> boolean (positive, negative) cloud membership, each (N, L)."""
    if policy not in UNCERTAIN_POLICIES:
        raise ValueError(f"mfsi.uncertain_policy must be one of {UNCERTAIN_POLICIES}, got {policy!r}")
    pos = y == POSITIVE
    if policy == "ones":
        pos = pos | (y == UNCERTAIN)
    neg = (y == NEGATIVE) | np.isnan(y)
    return pos, neg


def prototype(z: np.ndarray) -> np.ndarray:
    """Cloud mean in float64."""
    return z.astype(np.float64).mean(0)
