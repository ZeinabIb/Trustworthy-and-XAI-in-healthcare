"""Covariances (Phase 3, steps 3-5), all float64.

Raw:      S_c = 1/(|C|-1) sum (z - mu)(z - mu)^T
Negative: S_-, plus a ridge lambda I (lambda = ridge_factor * trace/d) only if not PD (logged).
Positive: S~_+ = alpha S_+ + (1 - alpha) T, with T = S~_parent+ (already shrunk; findings in topological
          order) or the pooled within-class covariance. Only the shape is borrowed; the mean stays the
          finding's own. A badly conditioned target chains to the pooled covariance. cond() is logged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mfsi_cxr.constants import FINDINGS, PARENT
from mfsi_cxr.utils.logging import get_logger

log = get_logger("mfsi")


def raw_covariance(z: np.ndarray, mu: np.ndarray | None = None) -> np.ndarray:
    z = z.astype(np.float64)
    mu = z.mean(0) if mu is None else mu
    x = z - mu
    return x.T @ x / max(len(z) - 1, 1)


def pooled_covariance(clouds: Sequence[np.ndarray]) -> np.ndarray:
    """(1/N) sum over all clouds of sum_i (z_i - mu_C)(z_i - mu_C)^T, N = total cloud sizes."""
    d = clouds[0].shape[1]
    total, n = np.zeros((d, d)), 0
    for z in clouds:
        if len(z):
            x = z.astype(np.float64) - z.astype(np.float64).mean(0)
            total += x.T @ x
            n += len(z)
    return total / max(n, 1)


def is_pd(m: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(m)
        return True
    except np.linalg.LinAlgError:
        return False


def make_pd(m: np.ndarray, ridge_factor: float, name: str) -> tuple[np.ndarray, float]:
    """Add lambda I (growing x10) until Cholesky succeeds; returns the matrix and the ridge used (0: none)."""
    if is_pd(m):
        return m, 0.0
    lam = ridge_factor * float(np.trace(m)) / len(m)
    lam = lam if lam > 0 else ridge_factor
    for _ in range(12):
        fixed = m + lam * np.eye(len(m))
        if is_pd(fixed):
            log.info("%s: not PD, ridge lambda=%.3g added", name, lam)
            return fixed, lam
        lam *= 10
    raise np.linalg.LinAlgError(f"{name}: could not make the covariance positive definite")


def shrink(raw: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * raw + (1.0 - alpha) * target


def topological_order(findings: Sequence[str] = FINDINGS) -> list[str]:
    """Roots first; every child after its parent (Pneumonia after Consolidation after Lung Opacity)."""
    order: list[str] = []

    def visit(f: str) -> None:
        if f in order:
            return
        if f in PARENT:
            visit(PARENT[f])
        order.append(f)

    for f in findings:
        visit(f)
    return order


@dataclass
class CovarianceSet:
    """Final per-finding class covariances of one modality, with what was done to them."""

    pos: dict[str, np.ndarray] = field(default_factory=dict)
    neg: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, dict] = field(default_factory=dict)


AlphaChooser = Callable[[str, np.ndarray, np.ndarray], float]


def fit_covariances(z: np.ndarray, pos: np.ndarray, neg: np.ndarray,
                    alphas: dict[str, float] | AlphaChooser, ridge_factor: float, cond_max: float,
                    modality: str, pooled: np.ndarray | None = None) -> CovarianceSet:
    """z (N, d) TRAIN embeddings; pos/neg (N, L) cloud masks. `alphas` is a fixed alpha per finding, or a
    chooser (finding, positive embeddings, shrinkage target) -> alpha, called in topological order so a
    child's target is its parent's final matrix."""
    if pooled is None:
        pooled = pooled_covariance([z[pos[:, j]] for j in range(len(FINDINGS))]
                                   + [z[neg[:, j]] for j in range(len(FINDINGS))])
    pooled, pooled_ridge = make_pd(pooled, ridge_factor, f"{modality}:pooled")
    out = CovarianceSet()
    for f in topological_order():
        j = FINDINGS.index(f)
        zn = z[neg[:, j]]
        s_neg, ridge_neg = make_pd(raw_covariance(zn), ridge_factor, f"{modality}:{f}:neg")
        target, target_name = pooled, "pooled"
        if f in PARENT:
            parent = out.pos[PARENT[f]]
            if np.linalg.cond(parent) <= cond_max:
                target, target_name = parent, f"parent:{PARENT[f]}"
            else:
                log.info("%s:%s: parent target badly conditioned; chained to pooled", modality, f)
        zp = z[pos[:, j]]
        raw_pos = raw_covariance(zp) if len(zp) >= 2 else np.zeros_like(pooled)
        if len(zp) < 2:
            alpha = 0.0
        elif callable(alphas):
            alpha = float(alphas(f, zp, target))
        else:
            alpha = float(alphas.get(f, 0.0))
        s_pos, ridge_pos = make_pd(shrink(raw_pos, target, alpha), ridge_factor, f"{modality}:{f}:pos")
        out.pos[f], out.neg[f] = s_pos, s_neg
        out.info[f] = {"n_pos": int(len(zp)), "n_neg": int(len(zn)), "alpha": float(alpha),
                       "target": target_name, "ridge_pos": ridge_pos, "ridge_neg": ridge_neg,
                       "cond_pos": float(np.linalg.cond(s_pos)), "cond_neg": float(np.linalg.cond(s_neg))}
    out.info["_pooled"] = {"ridge": pooled_ridge, "cond": float(np.linalg.cond(pooled))}
    return out
