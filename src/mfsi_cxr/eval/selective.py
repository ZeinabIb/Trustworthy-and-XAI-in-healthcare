"""Selective prediction metrics (Phase 4): correctness, risk-coverage, AURC, accuracy at coverage, AUROC.

Trust scores are oriented higher = more reliable. Cases are withheld from the lowest trust upwards; risk@k is
the error rate among the k most trusted cases. Ties in trust are broken in expectation (errors inside a tied
group are spread uniformly over it), so a score's AURC does not depend on the row order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BLANK_POLICIES = ("negative", "exclude")


def correctness(y: np.ndarray, y_hat: np.ndarray, blank_policy: str) -> tuple[np.ndarray, np.ndarray]:
    """correct = [y_hat == y] on scored labels. y in {1, 0} always scored; blank (NaN) scored as 0 under
    'negative', dropped under 'exclude'; uncertain (-1) never scored. Returns (correct, used), same shape."""
    if blank_policy not in BLANK_POLICIES:
        raise ValueError(f"eval.blank_policy must be one of {BLANK_POLICIES}, got {blank_policy!r}")
    y = np.asarray(y, dtype=float)
    blank = np.isnan(y)
    y_eff = np.where(blank, 0.0, y) if blank_policy == "negative" else y
    used = (y == 1) | (y == 0) | (blank & (blank_policy == "negative"))
    correct = used & (np.asarray(y_hat).astype(bool) == (y_eff == 1))
    return correct, used


def cumulative_errors(trust: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Expected number of errors among the k most trusted cases, k = 0..n (ties broken in expectation)."""
    trust = np.asarray(trust, dtype=float)
    err = 1.0 - np.asarray(correct, dtype=float)
    order = np.argsort(-trust, kind="stable")
    t, e = trust[order], err[order]
    n = len(t)
    out = np.zeros(n + 1)
    if n == 0:
        return out
    ends = np.r_[np.flatnonzero(np.diff(t) != 0) + 1, n]  # exclusive end of each tied group
    starts = np.r_[0, ends[:-1]]
    for s, end in zip(starts, ends, strict=True):
        rate = e[s:end].mean()
        out[s + 1:end + 1] = out[s] + rate * np.arange(1, end - s + 1)
    return out


def risk_coverage(trust: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coverage k/n, risk@k) for k = 1..n."""
    n = len(trust)
    k = np.arange(1, n + 1)
    return k / n, cumulative_errors(trust, correct)[1:] / k


def aurc(trust: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve: mean of risk@k over k = 1..n. Lower is better."""
    if len(trust) == 0:
        return float("nan")
    return float(risk_coverage(trust, correct)[1].mean())


def accuracy_at_coverage(trust: np.ndarray, correct: np.ndarray,
                         coverages: Sequence[float]) -> dict[str, float]:
    """1 - risk among the ceil(c * n) most trusted cases, per coverage c."""
    n = len(trust)
    cum = cumulative_errors(trust, correct)
    out = {}
    for c in coverages:
        k = int(np.ceil(c * n))
        out[str(c)] = float(1.0 - cum[k] / k) if k else float("nan")
    return out


def auroc_trust(trust: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of the trust score for separating correct (positive) from wrong cases; NaN if one class only."""
    from sklearn.metrics import roc_auc_score

    correct = np.asarray(correct).astype(int)
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, trust))
