"""Load one split for all seeds, build every trust score and run the RQ1 engine (Phase 4).

Shared by scripts/40_eval_rq1_dev.py (val) and scripts/41_run_rq1_final.py (test, behind the freeze guard).
Rows must be the same samples in the same order for every seed (checked).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval.baselines import all_scores
from mfsi_cxr.eval.calibration import ece_mce
from mfsi_cxr.eval.rq1 import Inputs, evaluate
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


def collect(cfg: dict[str, Any], split: str, seeds: list[int]) -> SplitData:
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
    probs_by_seed = [data["probs"] for _, data, _, _ in raw]
    scores = [all_scores(train, data, table, cfg, params, probs_by_seed)
              for train, data, table, params in raw]
    return SplitData(split, list(seeds), scores, y_hat, tables, ref["y"], ref["patient_id"],
                     params_all[0]["strata"])


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
