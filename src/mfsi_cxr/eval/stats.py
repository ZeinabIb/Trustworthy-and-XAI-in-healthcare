"""Statistics for Phase 4: patient-level bootstrap, DeLong, paired AURC bootstrap, Benjamini-Hochberg, and
the decision rule for the cross-modal claim (written before any run, plan 4).

Bootstrap resamples PATIENTS with replacement and takes all rows of each drawn patient together, so samples
of one patient never split between draws. DeLong treats samples as independent (it does not model patient
clustering); the bootstrap CIs are the headline uncertainty, DeLong p-values the paired AUROC test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def patient_groups(patient_id: np.ndarray) -> list[np.ndarray]:
    """Row indices of each patient, in first-appearance order."""
    import pandas as pd

    codes, _ = pd.factorize(np.asarray(patient_id))
    rows = np.argsort(codes, kind="stable")
    return np.split(rows, np.cumsum(np.bincount(codes))[:-1])


def bootstrap(patient_id: np.ndarray, stat: Callable[[np.ndarray], float | np.ndarray], n_boot: int,
              seed: int) -> np.ndarray:
    """n_boot values of stat(rows) on patient-resampled rows (shape (B,) or (B, ...))."""
    groups = patient_groups(np.asarray(patient_id))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), len(groups))
        out.append(stat(np.concatenate([groups[g] for g in draw])))
    return np.asarray(out, dtype=float)


def percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Percentile interval of bootstrap values, NaN draws dropped (e.g. a draw with one class only)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks
    return out


def delong_cov(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUCs (k,) and their DeLong covariance (k, k) for k scores (k, n) against binary y (fast DeLong,
    Sun & Xu 2014, with midranks for ties)."""
    y = np.asarray(y).astype(bool)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        k = len(scores)
        return np.full(k, np.nan), np.full((k, k), np.nan)
    tx = np.stack([_midrank(r) for r in pos])
    ty = np.stack([_midrank(r) for r in neg])
    tz = np.stack([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = (tz[:, :m].sum(1) - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:, :m] - tx) / n          # placement of each positive among the negatives
    v01 = 1.0 - (tz[:, m:] - ty) / m    # placement of each negative among the positives
    cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return aucs, cov


def delong_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict[str, float]:
    """Paired two-sided DeLong test of AUC(s1) = AUC(s2)."""
    from scipy.stats import norm

    aucs, cov = delong_cov(y, np.stack([s1, s2]))
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if np.isfinite(var) and var > 0:
        z = diff / np.sqrt(var)
        p = float(2 * norm.sf(abs(z)))
    else:  # identical scores (diff 0, var 0) -> p = 1; undefined otherwise
        z = float("nan")
        p = 1.0 if diff == 0 else float("nan")
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": diff, "z": float(z), "p": p}


def benjamini_hochberg(p: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """BH-adjusted q-values and rejections; NaN p-values stay NaN and are never rejected."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m:
        order = np.argsort(pv)
        ranked = pv[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(adj, 1.0)
        q[ok] = out
    return q, np.where(ok, q <= alpha, False)


def cross_modal_claim_holds(diff_ci_low: float) -> bool:
    """Pre-stated rule (plan 4): the cross-modal claim holds for a finding only if the (1 - alpha) CI of
    AURC(best single-modality MFSI) - AURC(MFSI) excludes 0 in MFSI's favour, i.e. its lower end is > 0."""
    return bool(np.isfinite(diff_ci_low) and diff_ci_low > 0)
