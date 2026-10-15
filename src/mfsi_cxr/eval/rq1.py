"""RQ1 evaluation engine (Phase 4): every trust score, per finding and macro-averaged, with patient-bootstrap
CIs.

- correct = [y_hat == y] per seed (each seed's own classifier and val-fixed thresholds); scored labels by
  eval.blank_policy. Metrics per (score, finding): AURC, accuracy at the coverages, AUROC(trust vs correct).
- Seeds: each bootstrap draw computes the metric per seed and averages over seeds (same patients in every
  seed), so CIs are for the seed mean; the across-seed range of the point estimates is reported too.
- Headline = macro average over findings of the per-finding metric. Trust scales differ between findings, so
  scores are never ranked across findings (DECISIONS 2026-10-15).
- One bootstrap pass evaluates all scores on the same draws, so differences against the reference (MFSI) are
  paired. DeLong per seed and finding, BH across findings within a seed and score.
- Cross-modal rule (pre-stated, plan 4): per finding, diff = AURC(best of mfsi_img, mfsi_txt) - AURC(mfsi);
  the claim holds only if the lower CI end of diff is > 0. The best single modality is picked on the same data
  by point AURC, which favours the baseline. One-witness findings: R = r_img, so diff <= 0 (point) and the
  claim cannot hold.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval.selective import correctness, cumulative_errors
from mfsi_cxr.eval.stats import (
    benjamini_hochberg,
    bootstrap,
    cross_modal_claim_holds,
    delong_test,
    percentile_ci,
)

SINGLE = ("mfsi_img", "mfsi_txt")


def _nan(fn, a: np.ndarray, **kw) -> np.ndarray | float:
    """nanmean / nanmin / nanmax without the all-NaN warning (a finding with no scored rows is NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, **kw)


def metric_names(coverages: list[float]) -> list[str]:
    """Per-score metrics, then the score-independent sanity anchors (Zeinab 2026-10-15): random AURC (the
    expected AURC of a random order = the error rate) and oracle AURC (all correct cases first)."""
    return ["aurc", *[f"acc@{c}" for c in coverages], "auroc", "aurc_random", "aurc_oracle"]


def oracle_aurc(n: int, n_err: float) -> float:
    """AURC of a perfect trust order: risk@k = max(0, k - (n - n_err)) / k."""
    if n == 0:
        return float("nan")
    k = np.arange(1, n + 1)
    return float((np.maximum(0.0, k - (n - n_err)) / k).mean())


def _fast_auroc(score: np.ndarray, positive: np.ndarray) -> float:
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _metrics(trust: np.ndarray, correct: np.ndarray, coverages: list[float]) -> np.ndarray:
    n = len(trust)
    if n == 0:
        return np.full(len(coverages) + 4, np.nan)
    cum = cumulative_errors(trust, correct)
    out = [float((cum[1:] / np.arange(1, n + 1)).mean())]
    for c in coverages:
        k = int(np.ceil(c * n))
        out.append(1.0 - cum[k] / k if k else np.nan)
    out.append(_fast_auroc(trust, correct.astype(bool)))
    n_err = float(n - correct.sum())
    out += [n_err / n, oracle_aurc(n, n_err)]
    return np.array(out)


class Inputs:
    """Aligned per-seed arrays: trust[name] (S, N, L), correct and used (S, N, L), patient_id (N,)."""

    def __init__(self, scores_by_seed: list[dict[str, np.ndarray]], y: np.ndarray,
                 y_hat_by_seed: list[np.ndarray], patient_id: np.ndarray, blank_policy: str,
                 finding_mask: np.ndarray | None = None) -> None:
        names = [k for k in scores_by_seed[0] if all(k in s for s in scores_by_seed)]
        self.names = names
        self.trust = {k: np.stack([s[k] for s in scores_by_seed]) for k in names}
        cu = [correctness(y, yh, blank_policy) for yh in y_hat_by_seed]
        self.correct = np.stack([c for c, _ in cu])
        self.used = np.stack([u for _, u in cu])
        if finding_mask is not None:  # findings outside the mask are not scored at all (metrics n/a)
            self.used &= np.asarray(finding_mask, dtype=bool)[None, None, :]
        self.patient_id = np.asarray(patient_id)
        self.n_seeds = len(scores_by_seed)


def _block(inp: Inputs, rows: np.ndarray, coverages: list[float]) -> np.ndarray:
    """(n_scores, L, M) seed-mean metrics on `rows` (patient-resampled row indices, repeats allowed)."""
    out = np.full((len(inp.names), len(FINDINGS), len(coverages) + 4), np.nan)
    for j in range(len(FINDINGS)):
        per_seed = []
        for s in range(inp.n_seeds):
            r = rows[inp.used[s, rows, j]]
            c = inp.correct[s, r, j]
            per_seed.append([_metrics(inp.trust[k][s, r, j], c, coverages) for k in inp.names])
        out[:, j, :] = _nan(np.nanmean, np.array(per_seed), axis=0) if inp.n_seeds > 1 else per_seed[0]
    return out


def evaluate(inp: Inputs, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    e = cfg["eval"]
    cov = list(e["coverages"])
    mnames = metric_names(cov)
    alpha, ref = e["alpha"], e["reference"]
    all_rows = np.arange(len(inp.patient_id))
    point = _block(inp, all_rows, cov)
    boot = bootstrap(inp.patient_id, lambda rows: _block(inp, rows, cov), e["bootstrap_B"],
                     e["bootstrap_seed"])
    seed_points = [_block(_one_seed(inp, s), all_rows, cov) for s in range(inp.n_seeds)]
    ri = inp.names.index(ref)

    per_finding = []
    for i, name in enumerate(inp.names):
        for j, f in enumerate(FINDINGS):
            row: dict[str, Any] = {"score": name, "finding": f,
                                   "n_used": int(inp.used[0, :, j].sum()),
                                   "error_rate": float(1 - inp.correct[0, inp.used[0, :, j], j].mean())
                                   if inp.used[0, :, j].any() else np.nan}
            for m, mn in enumerate(mnames):
                row[mn] = point[i, j, m]
                row[f"{mn}_lo"], row[f"{mn}_hi"] = percentile_ci(boot[:, i, j, m], alpha)
            sp = [p[i, j, 0] for p in seed_points]
            row["aurc_seed_min"] = float(_nan(np.nanmin, sp))
            row["aurc_seed_max"] = float(_nan(np.nanmax, sp))
            row["aurc_minus_ref"] = point[i, j, 0] - point[ri, j, 0]
            diff = boot[:, i, j, 0] - boot[:, ri, j, 0]
            row["aurc_minus_ref_lo"], row["aurc_minus_ref_hi"] = percentile_ci(diff, alpha)
            _vs_random(row, point[i, j], boot[:, i, j], mnames, alpha)
            per_finding.append(row)

    macro_point = _nan(np.nanmean, point, axis=1)        # (n_scores, M)
    macro_boot = _nan(np.nanmean, boot, axis=2)          # (B, n_scores, M)
    main = []
    for i, name in enumerate(inp.names):
        row = {"score": name, "n_findings": int(np.isfinite(point[i, :, 0]).sum())}
        for m, mn in enumerate(mnames):
            row[mn] = macro_point[i, m]
            row[f"{mn}_lo"], row[f"{mn}_hi"] = percentile_ci(macro_boot[:, i, m], alpha)
        sp = [_nan(np.nanmean, p[i, :, 0]) for p in seed_points]
        row["aurc_seed_min"], row["aurc_seed_max"] = float(np.min(sp)), float(np.max(sp))
        row["aurc_minus_ref"] = macro_point[i, 0] - macro_point[ri, 0]
        diff = macro_boot[:, i, 0] - macro_boot[:, ri, 0]
        row["aurc_minus_ref_lo"], row["aurc_minus_ref_hi"] = percentile_ci(diff, alpha)
        _vs_random(row, macro_point[i], macro_boot[:, i], mnames, alpha)
        row["findings_below_random"] = ";".join(
            f for j, f in enumerate(FINDINGS) if point[i, j, 0] > point[i, j, mnames.index("aurc_random")])
        main.append(row)

    return {"main": pd.DataFrame(main), "per_finding": pd.DataFrame(per_finding),
            "delong": _delong(inp, ref, alpha), "cross_modal": _cross_modal(inp, point, boot, alpha)}


def _vs_random(row: dict[str, Any], point: np.ndarray, boot: np.ndarray, mnames: list[str],
               alpha: float) -> None:
    """AURC minus random AURC (> 0 = worse than a random order), with its paired bootstrap CI."""
    k = mnames.index("aurc_random")
    row["aurc_minus_random"] = point[0] - point[k]
    row["aurc_minus_random_lo"], row["aurc_minus_random_hi"] = percentile_ci(boot[:, 0] - boot[:, k], alpha)
    row["below_random"] = bool(row["aurc_minus_random"] > 0)
    row["below_random_ci"] = bool(row["aurc_minus_random_lo"] > 0)


def _one_seed(inp: Inputs, s: int) -> Inputs:
    one = object.__new__(Inputs)
    one.names, one.patient_id, one.n_seeds = inp.names, inp.patient_id, 1
    one.trust = {k: v[s:s + 1] for k, v in inp.trust.items()}
    one.correct, one.used = inp.correct[s:s + 1], inp.used[s:s + 1]
    return one


def _delong(inp: Inputs, ref: str, alpha: float) -> pd.DataFrame:
    rows = []
    for s in range(inp.n_seeds):
        for name in inp.names:
            if name == ref:
                continue
            ps = []
            for j, f in enumerate(FINDINGS):
                u = inp.used[s, :, j]
                c = inp.correct[s, u, j].astype(int)
                if len(np.unique(c)) < 2:
                    t = {"auc1": np.nan, "auc2": np.nan, "diff": np.nan, "z": np.nan, "p": np.nan}
                else:
                    t = delong_test(c, inp.trust[ref][s, u, j], inp.trust[name][s, u, j])
                rows.append({"seed_index": s, "score": name, "finding": f, "auroc_ref": t["auc1"],
                             "auroc_score": t["auc2"], "diff_ref_minus_score": t["diff"], "z": t["z"],
                             "p": t["p"]})
                ps.append(t["p"])
            q, rej = benjamini_hochberg(np.array(ps), alpha)
            for k in range(len(FINDINGS)):
                rows[-len(FINDINGS) + k].update(q_bh=q[k], reject_bh=bool(rej[k]))
    return pd.DataFrame(rows)


def _cross_modal(inp: Inputs, point: np.ndarray, boot: np.ndarray, alpha: float) -> pd.DataFrame:
    if not all(k in inp.names for k in (*SINGLE, "mfsi")):
        return pd.DataFrame()
    ri = inp.names.index("mfsi")
    rows = []
    for j, f in enumerate(FINDINGS):
        single = {k: point[inp.names.index(k), j, 0] for k in SINGLE}
        finite = {k: v for k, v in single.items() if np.isfinite(v)}
        if not finite:
            rows.append({"finding": f, "best_single": None, "diff": np.nan, "diff_lo": np.nan,
                         "diff_hi": np.nan, "claim_holds": False})
            continue
        best = min(finite, key=finite.get)
        bi = inp.names.index(best)
        lo, hi = percentile_ci(boot[:, bi, j, 0] - boot[:, ri, j, 0], alpha)
        rows.append({"finding": f, "best_single": best, "aurc_best_single": point[bi, j, 0],
                     "aurc_mfsi": point[ri, j, 0], "diff": point[bi, j, 0] - point[ri, j, 0],
                     "diff_lo": lo, "diff_hi": hi, "claim_holds": cross_modal_claim_holds(lo)})
    return pd.DataFrame(rows)
