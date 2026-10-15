"""Dimension reduction before MFSI (Zeinab 2026-10-15; answers the plan 3 STOP-AND-ASK on Sigma conditioning).

The 1024-d embeddings are rank-deficient (z_txt rank 768 by construction; ~99% of variance in ~100-170 dims),
so 1024-d class covariances are singular and Mahalanobis distances are dominated by near-empty directions.
Each modality is projected onto its top-k principal components, fitted on TRAIN only (float64, centred SVD);
k is the smallest number of components keeping >= `variance` of the train variance. The MFSI formulas are
unchanged, in k dimensions. Mahalanobis distance is invariant to rotation/sign of the basis, so the SVD sign
convention does not affect any score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mfsi_cxr.utils.logging import get_logger

log = get_logger("mfsi")


@dataclass
class Projection:
    """z -> (z - mean) @ basis, basis (d, k). Method "none" is the identity (the plan as written)."""

    mean: np.ndarray
    basis: np.ndarray | None
    variance_kept: float

    @property
    def k(self) -> int:
        return len(self.mean) if self.basis is None else self.basis.shape[1]

    def __call__(self, z: np.ndarray) -> np.ndarray:
        if self.basis is None:
            return z
        return (z.astype(np.float64) - self.mean) @ self.basis

    def info(self) -> dict[str, Any]:
        return {"d_in": int(len(self.mean)), "k": int(self.k), "variance_kept": float(self.variance_kept)}


def n_components(explained: np.ndarray, variance: float) -> int:
    """Smallest k with cumulative explained-variance share >= variance."""
    share = np.cumsum(explained) / explained.sum()
    return int(min(np.searchsorted(share, variance - 1e-12) + 1, len(share)))


def fit_projection(z_train: np.ndarray, cfg_reduce: dict[str, Any], name: str) -> Projection:
    z = z_train.astype(np.float64)
    mean = z.mean(0)
    method = cfg_reduce["method"]
    if method == "none":
        return Projection(mean, None, 1.0)
    if method != "pca":
        raise ValueError(f"mfsi.reduce.method must be pca or none, got {method!r}")
    _, s, vt = np.linalg.svd(z - mean, full_matrices=False)
    explained = s**2
    k = n_components(explained, float(cfg_reduce["variance"]))
    kept = float(explained[:k].sum() / explained.sum())
    log.info("%s: PCA %d -> %d dims (%.4f of train variance)", name, z.shape[1], k, kept)
    return Projection(mean, vt[:k].T.copy(), kept)
