"""Classifier decision rule and MSP, y_hat v2 (Zeinab 2026-10-15; docs/PROPOSAL_YHAT_V2.md).

Every rule maps the raw probability p to a decision-centred probability q, y_hat and MSP:
- platt (v2 primary): q = p_cal = sigmoid(a logit(p) + b); (a, b) by maximum likelihood with Platt's smoothed
  targets t+ = (N+ + 1)/(N+ + 2), t- = 1/(N- + 2); a >= 0 (never reverses the classifier's ranking).
  y_hat = 1[q >= 0.5], MSP = max(q, 1 - q).
- youden_coherent (sensitivity): t = Youden threshold on p; q = sigmoid(logit(p) - logit(t)), so y_hat =
  1[q >= 0.5] = 1[p >= t] and MSP_t = max(q, 1 - q) = sigmoid(|logit p - logit t|).
- youden / f1 (old rules, reproduce earlier runs): y_hat = 1[p >= t], MSP = max(p, 1 - p) (incoherent; kept
  only for reproduction).
Fitting uses scored rows only: y = 1 positive, 0 or blank negative, uncertain excluded. A fit with 0 or 1
positives is still made and flagged (`sparse`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, logit

from mfsi_cxr.mfsi.escalation import yhat_threshold

RULES = ("platt", "youden_coherent", "youden", "f1")
VERSION = "yhat_v2"


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps))


def fit_platt(p: np.ndarray, label: np.ndarray, eps: float) -> dict[str, Any]:
    """(a, b) of sigmoid(a logit(p) + b) by smoothed-target maximum likelihood, a >= 0 (L-BFGS-B)."""
    from scipy.optimize import minimize

    x = _logit(p, eps)
    label = np.asarray(label, dtype=bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    t = np.where(label, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        s = w[0] * x + w[1]
        loss = np.mean(np.logaddexp(0.0, s) - t * s)
        resid = expit(s) - t
        return float(loss), np.array([np.mean(resid * x), np.mean(resid)])

    if len(x) == 0:
        return {"a": 0.0, "b": 0.0, "n_pos": 0, "n_neg": 0, "sparse": True}
    res = minimize(objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None), (None, None)])
    return {"a": float(res.x[0]), "b": float(res.x[1]), "n_pos": n_pos, "n_neg": n_neg,
            "sparse": n_pos <= 1, "converged": bool(res.success)}


def fit_rule(rule: str, p: np.ndarray, label: np.ndarray, eps: float) -> dict[str, Any]:
    """Parameters of `rule` for one finding from its scored rows (p raw, label 1/0)."""
    if rule not in RULES:
        raise ValueError(f"mfsi.yhat_rule must be one of {RULES}, got {rule!r}")
    if rule == "platt":
        return fit_platt(p, label, eps)
    n_pos = int(np.asarray(label).sum())
    return {"t": yhat_threshold(p, np.asarray(label, dtype=float), "f1" if rule == "f1" else "youden"),
            "n_pos": n_pos, "n_neg": int(len(label) - n_pos), "sparse": n_pos <= 1}


def apply_rule(rule: str, params: dict[str, Any], p: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray,
                                                                                      np.ndarray]:
    """(q, y_hat, msp) for raw probabilities p of one finding."""
    p = np.asarray(p, dtype=np.float64)
    if rule == "platt":
        q = expit(params["a"] * _logit(p, eps) + params["b"])
        return q, q >= 0.5, np.maximum(q, 1 - q)
    if rule == "youden_coherent":
        q = expit(_logit(p, eps) - _logit(np.array(params["t"]), eps))
        return q, p >= params["t"], np.maximum(q, 1 - q)
    return p, p >= params["t"], np.maximum(p, 1 - p)


def apply_all(rule: str, per_finding: list[dict[str, Any]], probs: np.ndarray,
              eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(q, y_hat, msp), each (N, L), with one parameter set per finding (FINDINGS order)."""
    out = [apply_rule(rule, prm, probs[:, j], eps) for j, prm in enumerate(per_finding)]
    return tuple(np.stack([o[k] for o in out], 1) for k in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------- out-of-fold (dev path) and nested


def make_folds(patient_id: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold index per row; patients assigned to folds by a seeded permutation (round robin). Shared by the
    out-of-fold calibration and the added-value (a) dev CV, so both use the same folds."""
    uniq, inv = np.unique(np.asarray(patient_id), return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_patient = np.empty(len(uniq), int)
    fold_of_patient[order] = np.arange(len(uniq)) % n_folds
    return fold_of_patient[inv]


def out_of_fold(rule: str, probs: np.ndarray, label: np.ndarray, used: np.ndarray, folds: np.ndarray,
                eps: float, exclude: int | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold (q, y_hat, msp) (N, L): rows of fold k use a rule fitted on the scored rows of the other
    folds (and never on fold `exclude`, for nested calibration). Also (n_folds, L) sparse-fold flags. Rows of
    fold `exclude` are left NaN / False."""
    n, n_find = probs.shape
    q = np.full((n, n_find), np.nan)
    y_hat = np.zeros((n, n_find), bool)
    msp = np.full((n, n_find), np.nan)
    ks = [k for k in np.unique(folds) if k != exclude]
    sparse = np.zeros((int(folds.max()) + 1, n_find), bool)
    for k in ks:
        test_rows = folds == k
        train_rows = (folds != k) & (folds != (-1 if exclude is None else exclude))
        for j in range(n_find):
            tr = train_rows & used[:, j]
            prm = fit_rule(rule, probs[tr, j], label[tr, j], eps)
            sparse[k, j] = prm["sparse"]
            res = apply_rule(rule, prm, probs[test_rows, j], eps)
            q[test_rows, j], y_hat[test_rows, j], msp[test_rows, j] = res
    return q, y_hat, msp, sparse
