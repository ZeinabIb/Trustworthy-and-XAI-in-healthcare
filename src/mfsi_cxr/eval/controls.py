"""Validity controls (Phase 4, plan 4 "Validity controls"), computed from the MFSI table of a split and the
correctness of the RQ1 analysis (dev path: out-of-fold y_hat on val). Pure functions; scripts write files.

1. agreement spread + 4. modality dominance: spread_table (also used by scripts/32_agreement_spread.py).
2. data-volume confound: Spearman / Pearson of R and of escalation with n_studies, n_images, text_len,
   n_priors (= n_studies - 1); quantile-bin tables; if |Spearman| >= eval.controls.confound_strong for a
   variable in any finding, R rank-normalised within that variable's bins is evaluated next to R.
5. hierarchy: coherence-flag rate per child label and error rate flagged vs unflagged.
6. calibration of R: accuracy within deciles of R per finding, with a monotonicity summary.
Free check: R of uncertain (y = -1) vs certain-label rows per finding (Mann-Whitney) and the macro difference
with a patient-bootstrap CI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from mfsi_cxr.constants import FINDINGS, ancestors
from mfsi_cxr.eval.stats import bootstrap, percentile_ci

VOLUME_VARS = ("n_studies", "n_images", "text_len", "n_priors")


def _corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(fn(a, b)[0])


def spread_table(val: pd.DataFrame) -> pd.DataFrame:
    """Agreement spread (first experiment) and corr(p_img+, p_txt+) per finding."""
    rows = []
    for f in FINDINGS:
        g = val[val["finding"] == f]
        a = g["A_cos"].to_numpy()
        rows.append({"finding": f, "stratum": g["stratum"].iloc[0], "n": len(g), "mean": a.mean(),
                     "sd": a.std(ddof=1), "p5": np.quantile(a, 0.05), "p95": np.quantile(a, 0.95),
                     "corr_A_min_margin": _corr(pearsonr, a, np.minimum(g["r_img"], g["r_txt"]).to_numpy()),
                     "corr_p_img_p_txt": _corr(pearsonr, g["p_img_pos"].to_numpy(),
                                               g["p_txt_pos"].to_numpy())})
    return pd.DataFrame(rows)


def wide(table: pd.DataFrame, col: str) -> np.ndarray:
    """(N, L) array of a long MFSI-table column (rows sample-major, FINDINGS order)."""
    return table[col].to_numpy().reshape(-1, len(FINDINGS))


def volume_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Per-sample volume variables (n_priors = n_studies - 1: the index study plus its priors)."""
    s = table.iloc[:: len(FINDINGS)][["n_studies", "n_images", "text_len"]].reset_index(drop=True)
    return s.assign(n_priors=s["n_studies"] - 1)


def _bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin codes (ties merged); one bin when x is constant."""
    if np.unique(x).size < 2:
        return np.zeros(len(x), int)
    return np.asarray(pd.qcut(x, n_bins, labels=False, duplicates="drop"))


def volume_confound(table: pd.DataFrame, escalate: np.ndarray,
                    n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(correlation table, bin table). R from the table; escalate (N, L) is the escalation used."""
    vol = volume_frame(table)
    r = wide(table, "R")
    corr, bins = [], []
    for v in VOLUME_VARS:
        x = vol[v].to_numpy(dtype=float)
        codes = _bins(x, n_bins)
        for j, f in enumerate(FINDINGS):
            corr.append({"variable": v, "finding": f,
                         "spearman_R": _corr(spearmanr, x, r[:, j]),
                         "pearson_R": _corr(pearsonr, x, r[:, j]),
                         "spearman_escalate": _corr(spearmanr, x, escalate[:, j].astype(float))})
            for b in np.unique(codes):
                sel = codes == b
                bins.append({"variable": v, "finding": f, "bin": int(b), "n": int(sel.sum()),
                             "x_min": float(x[sel].min()), "x_max": float(x[sel].max()),
                             "mean_R": float(r[sel, j].mean()),
                             "escalation_rate": float(escalate[sel, j].mean())})
    return pd.DataFrame(corr), pd.DataFrame(bins)


def strong_variables(corr: pd.DataFrame, threshold: float) -> list[str]:
    return sorted(corr.loc[corr["spearman_R"].abs() >= threshold, "variable"].unique())


def rank_normalised(table: pd.DataFrame, variable: str, n_bins: int) -> np.ndarray:
    """R rank-normalised to [0, 1] within the variable's quantile bins, per finding (N, L)."""
    x = volume_frame(table)[variable].to_numpy(dtype=float)
    codes = _bins(x, n_bins)
    r = wide(table, "R")
    out = np.zeros_like(r)
    for b in np.unique(codes):
        sel = codes == b
        ranks = pd.DataFrame(r[sel]).rank(pct=True).to_numpy()
        out[sel] = ranks
    return out


def hierarchy_check(y_hat: np.ndarray, flags: np.ndarray, correct: np.ndarray,
                    used: np.ndarray) -> pd.DataFrame:
    """Per finding (children via the label tree; any pathology or No Finding via the approved No Finding
    rule): flag rate and error rate among flagged vs unflagged rows."""
    rows = []
    for j, f in enumerate(FINDINGS):
        u = used[:, j]
        fl = flags[u, j].astype(bool)
        err = ~correct[u, j].astype(bool)
        rows.append({"finding": f, "has_ancestor": bool(ancestors(f)), "n_used": int(u.sum()),
                     "flag_rate": float(fl.mean()) if u.any() else np.nan,
                     "n_flagged": int(fl.sum()),
                     "error_flagged": float(err[fl].mean()) if fl.any() else np.nan,
                     "error_unflagged": float(err[~fl].mean()) if (~fl).any() else np.nan,
                     "predicted_positive_rate": float(y_hat[u, j].mean()) if u.any() else np.nan})
    return pd.DataFrame(rows)


def r_deciles(r: np.ndarray, correct: np.ndarray, used: np.ndarray,
              n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(accuracy per R decile per finding, monotonicity summary per finding)."""
    rows, summary = [], []
    for j, f in enumerate(FINDINGS):
        u = used[:, j]
        x, c = r[u, j], correct[u, j].astype(float)
        if np.unique(x).size < 2:
            summary.append({"finding": f, "n_bins": 0, "spearman_bin_accuracy": np.nan, "inversions": np.nan})
            continue
        codes = pd.qcut(pd.Series(x).rank(method="first"), n_bins, labels=False, duplicates="drop").to_numpy()
        acc = []
        for b in np.unique(codes):
            sel = codes == b
            acc.append(c[sel].mean())
            rows.append({"finding": f, "decile": int(b), "n": int(sel.sum()), "R_min": float(x[sel].min()),
                         "R_max": float(x[sel].max()), "accuracy": float(c[sel].mean())})
        acc = np.array(acc)
        summary.append({"finding": f, "n_bins": len(acc),
                        "spearman_bin_accuracy": _corr(spearmanr, np.arange(len(acc)), acc),
                        "inversions": int((np.diff(acc) < 0).sum())})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def uncertain_check(r: np.ndarray, y: np.ndarray, patient_id: np.ndarray,
                    cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R of uncertain (y = -1) vs certain-label rows (1, 0 or blank) per finding; macro difference over
    findings with >= eval.controls.uncertain_min uncertain rows, patient-bootstrap CI."""
    e = cfg["eval"]
    min_n = e["controls"]["uncertain_min"]
    unc = y == -1
    rows, eligible = [], []
    for j, f in enumerate(FINDINGS):
        a, b = r[unc[:, j], j], r[~unc[:, j], j]
        ok = len(a) >= min_n
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if ok and len(b) else np.nan
        rows.append({"finding": f, "n_uncertain": int(len(a)),
                     "mean_R_uncertain": float(a.mean()) if len(a) else np.nan,
                     "mean_R_certain": float(b.mean()),
                     "diff": float(a.mean() - b.mean()) if len(a) else np.nan,
                     "mannwhitney_p": p, "eligible": ok})
        if ok:
            eligible.append(j)

    def stat(rows_: np.ndarray) -> float:
        d = [r[rows_][unc[rows_, j], j].mean() - r[rows_][~unc[rows_, j], j].mean()
             for j in eligible if unc[rows_, j].any() and (~unc[rows_, j]).any()]
        return float(np.mean(d)) if d else np.nan

    if eligible:
        point = stat(np.arange(len(y)))
        lo, hi = percentile_ci(bootstrap(patient_id, stat, e["bootstrap_B"], e["bootstrap_seed"]), e["alpha"])
    else:
        point, lo, hi = np.nan, np.nan, np.nan
    names = ";".join(FINDINGS[j] for j in eligible) or "none"
    macro = pd.DataFrame([{"n_findings": len(eligible), "findings": names,
                           "macro_diff": point, "lo": lo, "hi": hi, "lower_for_uncertain": bool(hi < 0)}])
    return pd.DataFrame(rows), macro
