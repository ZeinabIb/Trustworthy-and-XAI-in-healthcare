"""Typicality term (sensitivity variant; docs/PREREG_TYPICALITY.md, approved 2026-10-15).

t(z) = Mahalanobis distance of the projected embedding to the TRAIN mean of all train samples of the modality,
with the TRAIN total covariance (float64, Cholesky; ridge only if not PD). Reference: t of all VAL samples.
S(z) = (1 + #{val: t >= t(z)}) / (1 + n_val); T(z) = min(1, S(z) / q). The variant uses r' = r * T; T = 1 for
cases inside the central (1 - q) of val, so typical cases are unchanged.
"""

from __future__ import annotations

import numpy as np

from mfsi_cxr.mfsi.clouds import prototype
from mfsi_cxr.mfsi.covariance import make_pd, raw_covariance
from mfsi_cxr.mfsi.distance import Gaussian


def fit_model(z_train: np.ndarray, ridge_factor: float, name: str) -> Gaussian:
    cov, _ = make_pd(raw_covariance(z_train), ridge_factor, f"{name}:typicality")
    return Gaussian(prototype(z_train), cov)


def survival(t: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(1 + #{reference >= t}) / (1 + n) for each t."""
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    at_least = len(ref) - np.searchsorted(ref, np.asarray(t, dtype=np.float64), side="left")
    return (1.0 + at_least) / (1.0 + len(ref))


def factor(t: np.ndarray, reference: np.ndarray, q: float) -> np.ndarray:
    return np.minimum(1.0, survival(t, reference) / q)
