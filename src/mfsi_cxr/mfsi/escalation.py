"""Thresholds and escalation (Phase 3, step 12). All thresholds are chosen on VAL.

escalate_l = [R_l < theta_R] or flag_l. theta_R escalates a fixed share of val cases (10/25/50%), per
finding (mfsi.theta_scope: finding; Zeinab 2026-10-15: a pooled threshold escalated by prevalence) or per
stratum (the plan as written); one-witness and two-witness R are never thresholded on the same scale. The
classifier call y_hat uses a per-finding threshold fixed on val (Youden's J by default), not 0.5.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

YHAT_RULES = ("youden", "f1")


def yhat_threshold(prob: np.ndarray, y: np.ndarray, rule: str) -> float:
    """Threshold on one finding's classifier probabilities (y binary, scored labels only)."""
    from sklearn.metrics import roc_curve

    if rule not in YHAT_RULES:
        raise ValueError(f"mfsi.yhat_rule must be one of {YHAT_RULES}, got {rule!r}")
    if len(np.unique(y)) < 2:
        return 0.5
    if rule == "youden":
        fpr, tpr, thr = roc_curve(y, prob)
        best = int(np.argmax(tpr - fpr))
        return float(min(thr[best], 1.0))
    candidates = np.unique(prob)
    f1 = []
    for t in candidates:
        pred = prob >= t
        tp = float((pred & (y == 1)).sum())
        f1.append(2 * tp / max(pred.sum() + (y == 1).sum(), 1))
    return float(candidates[int(np.argmax(f1))])


def stratum_thresholds(r: np.ndarray, strata: np.ndarray, fractions: Sequence[float]) -> dict[str, dict]:
    """theta_R per stratum and fraction: the f-quantile of that stratum's val R (cells: sample x finding)."""
    out: dict[str, dict] = {}
    for s in np.unique(strata):
        values = r[:, strata == s].ravel()
        out[str(s)] = {str(f): float(np.quantile(values, f)) for f in fractions} if values.size else {}
    return out


def finding_thresholds(r: np.ndarray, findings: Sequence[str], fractions: Sequence[float]) -> dict[str, dict]:
    """theta_R per finding and fraction: the f-quantile of that finding's val R over samples."""
    return {f: {str(q): float(np.quantile(r[:, j], q)) for q in fractions} if len(r) else {}
            for j, f in enumerate(findings)}


def escalate(r: np.ndarray, theta_r: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """r (N, L), theta_r (L,) each finding's threshold (own or its stratum's), flags (N, L) -> (N, L) bool."""
    return (r < theta_r[None, :]) | flags.astype(bool)
