"""Load one split for all seeds, build every trust score and run the RQ1 engine (Phase 4).

Shared by scripts/40_eval_rq1_dev.py (val) and scripts/41_run_rq1_final.py (test, behind the freeze guard).
Rows must be the same samples in the same order for every seed (checked).

y_hat v2 (docs/PROPOSAL_YHAT_V2.md): on the dev path (`oof=True`, val) y_hat, the decision-centred probability
q and MSP = max(q, 1 - q) are OUT-OF-FOLD (rule fitted on the other val folds, patient-grouped), so dev error
rates are not in-sample; on the final path they come from the parquet (rule fitted on all val, frozen).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval.baselines import all_scores
from mfsi_cxr.eval.calibration import ece_mce
from mfsi_cxr.eval.rq1 import Inputs, evaluate
from mfsi_cxr.mfsi import yhat
from mfsi_cxr.mfsi.fit import load_split, tau_targets
from mfsi_cxr.utils.io import artifact_dir, check_artifact, read_table


@dataclass
class SplitData:
    split: str
    seeds: list[int]
    scores: list[dict[str, np.ndarray]]
    y_hat: list[np.ndarray]
    tables: list[pd.DataFrame]
    y: np.ndarray
    patient_id: np.ndarray
    strata: dict[str, str]
    probs: list[np.ndarray] = field(default_factory=list)       # raw classifier p per seed
    q: list[np.ndarray] = field(default_factory=list)           # decision-centred probability per seed
    folds: np.ndarray | None = None                             # dev path: fold per row
    sparse: list[np.ndarray] = field(default_factory=list)      # dev path: (n_folds, L) sparse-fold flags


def collect(cfg: dict[str, Any], split: str, seeds: list[int], oof: bool = False) -> SplitData:
    """All seeds of a split. oof=True (dev, val only): out-of-fold y_hat / q / MSP."""
    if oof and split != "val":
        raise ValueError("the out-of-fold dev path is for val only")
    raw, tables, y_hat, params_all = [], [], [], []
    for seed in seeds:
        path = artifact_dir(cfg) / f"mfsi_params_seed{seed}.json"
        check_artifact(path, cfg)
        params = json.loads(path.read_text(encoding="utf-8"))
        train, data = load_split(cfg, "train", seed), load_split(cfg, split, seed)
        table = read_table(artifact_dir(cfg) / f"mfsi_{split}_seed{seed}.parquet", cfg)
        n = len(data["study_id"])
        if list(table["study_id"].iloc[:: len(FINDINGS)]) != list(data["study_id"]):
            raise ValueError(f"seed {seed}: MFSI table rows do not match the {split} export")
        raw.append((train, data, table, params))
        tables.append(table)
        y_hat.append(table["y_hat"].to_numpy().reshape(n, len(FINDINGS)).astype(bool))
        params_all.append(params)
    ref = raw[0][1]
    for seed, (_, data, _, _) in zip(seeds, raw, strict=True):
        if not np.array_equal(data["study_id"], ref["study_id"]):
            raise ValueError(f"seed {seed}: {split} samples differ from seed {seeds[0]} (order or content)")
    probs = [data["probs"] for _, data, _, _ in raw]
    n = len(ref["study_id"])
    folds, sparse = None, []
    if oof:
        m, e = cfg["mfsi"], cfg["eval"]
        label, used = tau_targets(ref["y"], m["uncertain_policy"])
        folds = yhat.make_folds(ref["patient_id"], e["calib_cv_folds"], e["calib_cv_seed"])
        q, y_hat = [], []
        for p in probs:
            qq, yh, _, sp = yhat.out_of_fold(m["yhat_rule"], p, label, used, folds, m["platt_eps"])
            q.append(qq)
            y_hat.append(yh)
            sparse.append(sp)
    else:
        q = [t["prob_cal"].to_numpy().reshape(n, len(FINDINGS)) for t in tables]
    scores = [all_scores(train, data, table, cfg, params, qq, q)
              for (train, data, table, params), qq in zip(raw, q, strict=True)]
    return SplitData(split, list(seeds), scores, y_hat, tables, ref["y"], ref["patient_id"],
                     params_all[0]["strata"], probs, q, folds, sparse)


def sparse_fold_table(d: SplitData) -> pd.DataFrame:
    """Dev path: calibration folds whose training part had 0 or 1 positives, per seed and finding."""
    return pd.DataFrame([{"seed": seed, "finding": f, "n_folds": int(sp.shape[0]),
                          "flagged_folds": int(sp[:, j].sum())}
                         for seed, sp in zip(d.seeds, d.sparse, strict=True) for j, f in enumerate(FINDINGS)])


def run(cfg: dict[str, Any], d: SplitData, blank_policy: str | None = None,
        finding_mask: np.ndarray | None = None) -> dict[str, pd.DataFrame]:
    policy = blank_policy or cfg["eval"]["blank_policy"]
    out = evaluate(Inputs(d.scores, d.y, d.y_hat, d.patient_id, policy, finding_mask), cfg)
    out["per_finding"].insert(2, "stratum", out["per_finding"]["finding"].map(d.strata))
    if not out["cross_modal"].empty:
        out["cross_modal"].insert(1, "stratum", out["cross_modal"]["finding"].map(d.strata))
    for t in out.values():
        t.insert(0, "split", d.split)
        t.insert(1, "blank_policy", policy)
    return out


def exclude_eligibility(cfg: dict[str, Any], y: np.ndarray) -> pd.DataFrame:
    """Per finding: explicit positives / negatives / blanks / uncertain in the split, and whether the
    'exclude blanks' sensitivity is computed (>= exclude_min_pos positives, >= exclude_min_neg negatives)."""
    e = cfg["eval"]
    n_pos, n_neg = (y == 1).sum(0), (y == 0).sum(0)
    return pd.DataFrame({"finding": list(FINDINGS), "n_explicit_pos": n_pos, "n_explicit_neg": n_neg,
                         "n_blank": np.isnan(y).sum(0), "n_uncertain": (y == -1).sum(0),
                         "eligible": (n_pos >= e["exclude_min_pos"]) & (n_neg >= e["exclude_min_neg"])})


def blank_sensitivity(cfg: dict[str, Any], d: SplitData,
                      main_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """'Exclude blanks' vs the headline policy on the SAME eligible findings (macro over eligible only), next
    to the headline macro over all findings. Returns (per-score table, per-finding eligibility table)."""
    elig = exclude_eligibility(cfg, d.y)
    mask = elig["eligible"].to_numpy()
    eligible = ";".join(elig.loc[mask, "finding"]) or "none"
    head = cfg["eval"]["blank_policy"]
    other = "exclude" if head == "negative" else "negative"
    if not mask.any():
        out = main_all[["score"]].assign(**{f"aurc_{head}_all": main_all["aurc"]}, eligible_findings=eligible)
        return out, elig
    a = run(cfg, d, head, mask)["main"].set_index("score")
    b = run(cfg, d, other, mask)["main"].set_index("score")
    out = pd.DataFrame({
        f"aurc_{head}_all": main_all.set_index("score")["aurc"],
        f"aurc_{head}_eligible": a["aurc"], f"aurc_{other}_eligible": b["aurc"],
        f"aurc_{other}_eligible_lo": b["aurc_lo"], f"aurc_{other}_eligible_hi": b["aurc_hi"],
        "shift_eligible": b["aurc"] - a["aurc"], "n_eligible_findings": int(mask.sum()),
        "eligible_findings": eligible})
    out["exceeds_warn"] = out["shift_eligible"].abs() > cfg["eval"]["blank_shift_warn"]
    return out.reset_index(), elig


def profile_calibration(cfg: dict[str, Any], d: SplitData) -> pd.DataFrame:
    """ECE / MCE (equal mass) of p^m_+ against the class label, before (tau = 1, b = 0) and after the fitted
    tau and offset; per seed, modality and finding. Uncertain excluded as in the cloud policy."""
    bins = cfg["eval"]["calibration_bins"]
    label, used = tau_targets(d.y, cfg["mfsi"]["uncertain_policy"])
    rows = []
    for seed, table in zip(d.seeds, d.tables, strict=True):
        n = len(d.y)
        for m in ("img", "txt"):
            before = expit(table[f"d_{m}_neg"].to_numpy() - table[f"d_{m}_pos"].to_numpy()).reshape(n, -1)
            after = table[f"p_{m}_pos"].to_numpy().reshape(n, -1)
            for j, f in enumerate(FINDINGS):
                u = used[:, j]
                eb, mb = ece_mce(before[u, j], label[u, j], bins)
                ea, ma = ece_mce(after[u, j], label[u, j], bins)
                rows.append({"split": d.split, "seed": seed, "modality": m, "finding": f, "n": int(u.sum()),
                             "ece_before": eb, "mce_before": mb, "ece_after": ea, "mce_after": ma})
    return pd.DataFrame(rows)


def decision_table(cfg: dict[str, Any], d: SplitData) -> pd.DataFrame:
    """Per seed and finding on the scored rows: prevalence (blank policy), predicted-positive rate, error rate
    and the share of errors that are false negatives, plus raw vs decision-centred ECE (equal mass)."""
    from mfsi_cxr.eval.selective import correctness

    e = cfg["eval"]
    rows = []
    for seed, yh, q, p in zip(d.seeds, d.y_hat, d.q, d.probs, strict=True):
        c, u = correctness(d.y, yh, e["blank_policy"])
        for j, f in enumerate(FINDINGS):
            uj = u[:, j]
            pos = d.y[uj, j] == 1
            pred = yh[uj, j].astype(bool)
            err = ~c[uj, j]
            rows.append({"seed": seed, "finding": f, "n_used": int(uj.sum()), "prevalence": float(pos.mean()),
                         "pred_pos_rate": float(pred.mean()), "error_rate": float(err.mean()),
                         "fn_share_of_errors": float((err & pos).sum() / max(err.sum(), 1)),
                         "ece_raw": ece_mce(p[uj, j], pos.astype(float), e["calibration_bins"])[0],
                         "ece_decision_prob": ece_mce(q[uj, j], pos.astype(float), e["calibration_bins"])[0]})
    return pd.DataFrame(rows)
