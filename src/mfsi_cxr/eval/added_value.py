"""Secondary RQ1 analysis "added value beyond MSP", as pre-registered in docs/PREREG_ADDED_VALUE.md (approved
2026-10-15; thresholds in configs/eval.yaml `added_value`). Inputs are the primary analysis' arrays (one per
seed): msp and mfsi trust (N, L), correct and used (N, L), and the patient ids. No test logic here; the
caller decides the split (val dev path in 44_added_value_dev, test only behind the freeze guard).

(a) combined score: per finding and seed, LogisticRegression(correct ~ standardised [MSP, MFSI]) on val;
    delta = AURC(MSP) - AURC(combined), seed mean inside each paired patient-bootstrap draw.
(b) stratified AUROC of MFSI for correctness within MSP quintiles (edges from val, merged when they coincide),
    pair-weighted over eligible strata; eligibility fixed once on the evaluated sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval.rq1 import _fast_auroc, _metrics, _nan
from mfsi_cxr.eval.stats import bootstrap, percentile_ci
from mfsi_cxr.utils.logging import get_logger

log = get_logger("eval")


@dataclass
class SeedArrays:
    """One seed of one split: trust scores and correctness (N, L)."""

    msp: np.ndarray
    mfsi: np.ndarray
    correct: np.ndarray
    used: np.ndarray


# ---------------------------------------------------------------- (a) combined score


@dataclass
class Combined:
    """Per finding: (mean, sd, coef, intercept) of the val-fitted model, or None when not eligible."""

    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float] | None] = field(default_factory=dict)


def _features(msp: np.ndarray, mfsi: np.ndarray) -> np.ndarray:
    return np.c_[msp, mfsi].astype(np.float64)


def _fit_one(x: np.ndarray, c: np.ndarray, cfg_av: dict[str, Any]) -> tuple[np.ndarray, np.ndarray,
                                                                               np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    mean, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    model = LogisticRegression(C=cfg_av["C"], solver="lbfgs", max_iter=cfg_av["max_iter"])
    model.fit((x - mean) / sd, c.astype(int))
    return mean, sd, model.coef_[0].copy(), float(model.intercept_[0])


def _predict(m: tuple[np.ndarray, np.ndarray, np.ndarray, float], x: np.ndarray) -> np.ndarray:
    mean, sd, coef, b = m
    return 1.0 / (1.0 + np.exp(-(((x - mean) / sd) @ coef + b)))


def eligible_a(seed: SeedArrays, cfg_av: dict[str, Any]) -> np.ndarray:
    """(L,) eligible on this seed's val rows: >= min_correct correct and >= min_wrong wrong scored rows."""
    c = np.where(seed.used, seed.correct, False).sum(0)
    w = np.where(seed.used, ~seed.correct.astype(bool), False).sum(0)
    return (c >= cfg_av["min_correct"]) & (w >= cfg_av["min_wrong"])


def fit_combined(val: SeedArrays, cfg_av: dict[str, Any], eligible: np.ndarray) -> Combined:
    out = Combined()
    for j, f in enumerate(FINDINGS):
        if not eligible[j]:
            out.models[f] = None
            continue
        u = val.used[:, j]
        out.models[f] = _fit_one(_features(val.msp[u, j], val.mfsi[u, j]), val.correct[u, j], cfg_av)
    return out


def score_combined(model: Combined, data: SeedArrays) -> np.ndarray:
    """Combined trust (N, L); NaN for findings without a model."""
    out = np.full(data.msp.shape, np.nan)
    for j, f in enumerate(FINDINGS):
        if model.models[f] is not None:
            out[:, j] = _predict(model.models[f], _features(data.msp[:, j], data.mfsi[:, j]))
    return out


def crossval_combined(val: SeedArrays, folds: np.ndarray, cfg_av: dict[str, Any], eligible: np.ndarray,
                      inner: Callable[[int], SeedArrays]) -> tuple[np.ndarray, np.ndarray]:
    """Dev path: out-of-fold combined trust on val with NESTED calibration (y_hat v2 amendment 4). For outer
    fold k the model is trained on the other folds' rows with inner(k) arrays (their MSP / correct from a
    calibration that never saw fold k) and applied to fold k with `val` (outer out-of-fold arrays). A training
    part lacking a class makes the finding n/a. Returns (trust (N, L), per-finding ok)."""
    out = np.full(val.msp.shape, np.nan)
    ok = eligible.copy()
    ks = np.unique(folds)
    if len(ks) < cfg_av["cv_folds"] or not ok.any():
        log.info("added value (a) dev: %d folds < %d or nothing eligible -> all n/a", len(ks),
                 cfg_av["cv_folds"])
        return out, np.zeros_like(ok)
    for k in ks:
        tr_arr = inner(int(k))
        tr_rows, te_rows = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        for j, f in enumerate(FINDINGS):
            if not ok[j]:
                continue
            tr = tr_rows[tr_arr.used[tr_rows, j]]
            te = te_rows[val.used[te_rows, j]]
            if len(np.unique(tr_arr.correct[tr, j])) < 2:
                log.info("added value (a) dev: %s training folds lack a class -> n/a", f)
                ok[j] = False
                continue
            m = _fit_one(_features(tr_arr.msp[tr, j], tr_arr.mfsi[tr, j]), tr_arr.correct[tr, j], cfg_av)
            out[te, j] = _predict(m, _features(val.msp[te, j], val.mfsi[te, j]))
    out[:, ~ok] = np.nan
    return out, ok


def coefficient_table(models_by_seed: list[Combined], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed, model in zip(seeds, models_by_seed, strict=True):
        for f in FINDINGS:
            m = model.models[f]
            rows.append({"seed": seed, "finding": f, "coef_msp": m[2][0] if m else np.nan,
                         "coef_mfsi": m[2][1] if m else np.nan, "intercept": m[3] if m else np.nan,
                         "mfsi_coef_negative": bool(m is not None and m[2][1] < 0)})
    return pd.DataFrame(rows)


def evaluate_a(combined: list[np.ndarray], data: list[SeedArrays], patient_id: np.ndarray,
               eligible: np.ndarray, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """delta = AURC(MSP) - AURC(combined) per finding (seed mean inside each draw) and macro over eligible."""
    e = cfg["eval"]

    def stat(rows: np.ndarray) -> np.ndarray:
        per_seed = []
        for comb, d in zip(combined, data, strict=True):
            vals = np.full((len(FINDINGS), 2), np.nan)
            for j in np.flatnonzero(eligible):
                r = rows[d.used[rows, j]]
                c = d.correct[r, j]
                vals[j] = [_metrics(d.msp[r, j], c, [])[0], _metrics(comb[r, j], c, [])[0]]
            per_seed.append(vals)
        return _nan(np.nanmean, np.array(per_seed), axis=0) if len(per_seed) > 1 else per_seed[0]

    point = stat(np.arange(len(patient_id)))
    boot = bootstrap(patient_id, stat, e["bootstrap_B"], e["bootstrap_seed"])
    rows = []
    for j, f in enumerate(FINDINGS):
        if not eligible[j]:
            rows.append({"finding": f, "eligible": False})
            continue
        lo, hi = percentile_ci(boot[:, j, 0] - boot[:, j, 1], e["alpha"])
        rows.append({"finding": f, "eligible": True, "aurc_msp": point[j, 0], "aurc_combined": point[j, 1],
                     "delta": point[j, 0] - point[j, 1], "delta_lo": lo, "delta_hi": hi,
                     "added_value": bool(lo > 0)})
    per = pd.DataFrame(rows)
    idx = np.flatnonzero(eligible)
    if len(idx):
        d_boot = (boot[:, idx, 0] - boot[:, idx, 1]).mean(1)
        lo, hi = percentile_ci(d_boot, e["alpha"])
        macro = pd.DataFrame([{"n_findings": len(idx), "findings": ";".join(FINDINGS[j] for j in idx),
                               "delta": float((point[idx, 0] - point[idx, 1]).mean()), "delta_lo": lo,
                               "delta_hi": hi, "added_value": bool(lo > 0)}])
    else:
        macro = pd.DataFrame([{"n_findings": 0, "findings": "none", "added_value": False}])
    return per, macro


def delong_a(combined: list[np.ndarray], data: list[SeedArrays], eligible: np.ndarray, seeds: list[int],
             alpha: float) -> pd.DataFrame:
    from mfsi_cxr.eval.stats import benjamini_hochberg, delong_test

    rows = []
    for seed, comb, d in zip(seeds, combined, data, strict=True):
        block = []
        for j in np.flatnonzero(eligible):
            u = d.used[:, j]
            t = delong_test(d.correct[u, j].astype(int), comb[u, j], d.msp[u, j])
            block.append({"seed": seed, "finding": FINDINGS[j], "auroc_combined": t["auc1"],
                          "auroc_msp": t["auc2"], "diff": t["diff"], "p": t["p"]})
        q, rej = benjamini_hochberg(np.array([b["p"] for b in block]), alpha)
        for b, qq, rr in zip(block, q, rej, strict=True):
            rows.append({**b, "q_bh": qq, "reject_bh": bool(rr)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (b) stratified AUROC within MSP quintiles


def quintile_edges(msp_val: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Interior edges from val MSP, coinciding edges merged (np.unique) -> len(edges) + 1 strata."""
    return np.unique(np.quantile(msp_val, quantiles)) if len(msp_val) else np.array([])


def assign(msp: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Stratum index per row; bins closed on the right (x <= edge goes to the lower bin)."""
    return np.searchsorted(edges, msp, side="left")


@dataclass
class StrataPlan:
    """Per seed and finding: interior edges, number of strata, eligible stratum ids (fixed on the original
    evaluated sample), and whether the (finding, seed) has a value (>= min_strata eligible)."""

    edges: dict[tuple[int, int], np.ndarray]
    eligible: dict[tuple[int, int], np.ndarray]
    counts: pd.DataFrame
    finding_ok: np.ndarray  # (L,) has a value for every seed


def plan_strata(val: list[SeedArrays], data: list[SeedArrays], cfg_av: dict[str, Any]) -> StrataPlan:
    edges, elig, rows = {}, {}, []
    ok = np.ones(len(FINDINGS), bool)
    for s, (v, d) in enumerate(zip(val, data, strict=True)):
        for j, f in enumerate(FINDINGS):
            e = quintile_edges(v.msp[v.used[:, j], j], cfg_av["quantiles"])
            u = d.used[:, j]
            q = assign(d.msp[u, j], e)
            c = d.correct[u, j].astype(bool)
            good = []
            for k in range(len(e) + 1):
                nc, nw = int((c & (q == k)).sum()), int((~c & (q == k)).sum())
                is_ok = nc >= cfg_av["min_stratum_correct"] and nw >= cfg_av["min_stratum_wrong"]
                good.append(is_ok)
                rows.append({"seed_index": s, "finding": f, "stratum": k, "n_strata": len(e) + 1,
                             "merged_edges": len(cfg_av["quantiles"]) - len(e), "n_correct": nc,
                             "n_wrong": nw,
                             "eligible": is_ok})
            edges[(s, j)], elig[(s, j)] = e, np.flatnonzero(good)
            ok[j] &= len(elig[(s, j)]) >= cfg_av["min_strata"]
    return StrataPlan(edges, elig, pd.DataFrame(rows), ok)


def _stratified(trust: np.ndarray, correct: np.ndarray, q: np.ndarray,
                strata: np.ndarray) -> tuple[float, int]:
    """Pair-weighted AUROC over the given strata on these rows; a stratum lacking a class is dropped and
    counted."""
    num = den = 0.0
    dropped = 0
    for k in strata:
        sel = q == k
        c = correct[sel].astype(bool)
        w = float(c.sum()) * float((~c).sum())
        if w == 0:
            dropped += 1
            continue
        num += w * _fast_auroc(trust[sel], c)
        den += w
    return (num / den if den else np.nan), dropped


def evaluate_b(data: list[SeedArrays], patient_id: np.ndarray, plan: StrataPlan,
               cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = cfg["eval"]
    n_seeds = len(data)

    def stat(rows: np.ndarray) -> np.ndarray:
        vals = np.full((n_seeds, len(FINDINGS)), np.nan)
        drops = np.zeros((n_seeds, len(FINDINGS)))
        for s, d in enumerate(data):
            for j in np.flatnonzero(plan.finding_ok):
                r = rows[d.used[rows, j]]
                q = assign(d.msp[r, j], plan.edges[(s, j)])
                vals[s, j], drops[s, j] = _stratified(d.mfsi[r, j], d.correct[r, j], q, plan.eligible[(s, j)])
        return np.r_[_nan(np.nanmean, vals, axis=0) if n_seeds > 1 else vals[0], drops.sum(0)]

    n_find = len(FINDINGS)
    point = stat(np.arange(len(patient_id)))
    boot = bootstrap(patient_id, stat, e["bootstrap_B"], e["bootstrap_seed"])
    n_elig = np.array([sum(len(plan.eligible[(s, j)]) for s in range(n_seeds)) for j in range(n_find)])
    rows = []
    for j, f in enumerate(FINDINGS):
        if not plan.finding_ok[j]:
            rows.append({"finding": f, "enters": False})
            continue
        lo, hi = percentile_ci(boot[:, j], e["alpha"])
        rows.append({"finding": f, "enters": True, "stratified_auroc": point[j], "lo": lo, "hi": hi,
                     "adds_information": bool(lo > 0.5),
                     "drop_share": float(boot[:, n_find + j].sum() / (len(boot) * n_elig[j]))})
    per = pd.DataFrame(rows)
    idx = np.flatnonzero(plan.finding_ok)
    if len(idx):
        lo, hi = percentile_ci(_nan(np.nanmean, boot[:, idx], axis=1), e["alpha"])
        macro = pd.DataFrame([{"n_findings": len(idx), "findings": ";".join(FINDINGS[j] for j in idx),
                               "stratified_auroc": float(_nan(np.nanmean, point[idx])), "lo": lo, "hi": hi,
                               "adds_information": bool(lo > 0.5)}])
    else:
        macro = pd.DataFrame([{"n_findings": 0, "findings": "none", "adds_information": False}])
    return per, macro
