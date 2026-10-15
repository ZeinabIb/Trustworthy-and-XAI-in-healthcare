"""Mahalanobis distance (Phase 3, step 6) with a Cholesky solve, never an explicit inverse. float64."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


class Gaussian:
    """Prototype mu and covariance S of one cloud, factorised once."""

    def __init__(self, mu: np.ndarray, cov: np.ndarray) -> None:
        self.mu = mu.astype(np.float64)
        self.chol = np.linalg.cholesky(cov.astype(np.float64))

    def distance(self, z: np.ndarray) -> np.ndarray:
        """d(z) = sqrt((z - mu)^T S^-1 (z - mu)) for every row of z (N, d) -> (N,)."""
        x = (z.astype(np.float64) - self.mu).T
        y = solve_triangular(self.chol, x, lower=True, check_finite=False)
        return np.sqrt(np.clip((y * y).sum(0), 0.0, None))

    def log_likelihood(self, z: np.ndarray) -> np.ndarray:
        """Gaussian log-density per row (used to choose alpha by cross-validation)."""
        d = len(self.mu)
        logdet = 2.0 * np.log(np.diag(self.chol)).sum()
        return -0.5 * (self.distance(z) ** 2 + logdet + d * np.log(2 * np.pi))


class EuclideanCloud:
    """Sensitivity variant (`mfsi.distance: euclidean`): d(z) = ||z - mu||."""

    def __init__(self, mu: np.ndarray) -> None:
        self.mu = mu.astype(np.float64)

    def distance(self, z: np.ndarray) -> np.ndarray:
        return np.linalg.norm(z.astype(np.float64) - self.mu, axis=1)


class LOFCloud:
    """Sensitivity variant (`mfsi.distance: lof`): d(z) = -LOF score_samples against the class cloud
    (>= ~1, larger = more outlying), LocalOutlierFactor(novelty=True) fit on the TRAIN cloud."""

    def __init__(self, z_cloud: np.ndarray, n_neighbors: int) -> None:
        from sklearn.neighbors import LocalOutlierFactor

        k = max(1, min(n_neighbors, len(z_cloud) - 1))
        self.model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(z_cloud.astype(np.float64))

    def distance(self, z: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(z.astype(np.float64))
