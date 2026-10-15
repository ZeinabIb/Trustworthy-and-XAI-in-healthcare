"""Split-generic Phase 4 runners shared by the dev scripts (val: 40, 42, 43, 45) and the one-time final run on
test (41). Each returns {table name: DataFrame}; write_tables() writes them with sidecars. A SplitData of the
test split can only exist behind the freeze guard (pipeline.collect needs the TestAccess token).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.eval import ablations as ab
from mfsi_cxr.eval import controls as ct
from mfsi_cxr.eval import pipeline
from mfsi_cxr.eval.added_value_dev import run_dev
from mfsi_cxr.eval.figures import deciles_figure, reliability_figure, risk_coverage_figure, volume_figure
from mfsi_cxr.eval.freeze import TestAccess
from mfsi_cxr.eval.rq1 import Inputs, evaluate
from mfsi_cxr.eval.selective import correctness
from mfsi_cxr.mfsi.escalation import escalate
from mfsi_cxr.mfsi.fit import fit_mfsi, load_split, score_split, tau_targets, theta_vector, write_params
from mfsi_cxr.mfsi.hierarchy import coherence_flags
from mfsi_cxr.utils.io import artifact_dir, check_artifact, output_dir, write_meta

FIG_SCORES = ["mfsi", "mfsi_img", "mfsi_txt", "msp"]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write_tables(cfg: dict[str, Any], tables: dict[str, pd.DataFrame], prefix: str, split: str) -> list[Path]:
    out_dir = output_dir(cfg) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = out_dir / f"{prefix}{name}.csv"
        table.to_csv(path, index=False)
        write_meta(path, cfg, None, split=split, throwaway=cfg["profile"] != "full")
        paths.append(path)
    return paths


def fig_path(cfg: dict[str, Any], name: str) -> Path:
    d = output_dir(cfg) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


# ---------------------------------------------------------------- RQ1 main analysis


def rq1(cfg: dict[str, Any], d: pipeline.SplitData) -> dict[str, pd.DataFrame]:
    res = pipeline.run(cfg, d)
    sens, elig = pipeline.blank_sensitivity(cfg, d, res["main"])
    out = {**res, "blank_sensitivity": sens, "blank_eligibility": elig,
           "decisions": pipeline.decision_table(cfg, d),
           "profile_calibration": pipeline.profile_calibration(cfg, d)}
    if d.sparse:
        out["calibration_sparse_folds"] = pipeline.sparse_fold_table(d)
    return out


def rq1_figures(cfg: dict[str, Any], d: pipeline.SplitData, suffix: str) -> list[Path]:
    e = cfg["eval"]
    correct, used = correctness(d.y, d.y_hat[0], e["blank_policy"])
    rc = fig_path(cfg, f"risk_coverage{suffix}.png")
    risk_coverage_figure(d.scores[0], correct, used, FIG_SCORES, rc,
                         f"Risk-coverage on {d.split} (seed {d.seeds[0]}, {cfg['profile']})")
    label, lab_used = tau_targets(d.y, cfg["mfsi"]["uncertain_policy"])
    n = len(d.y)
    probs = {f"p_{m}+": d.tables[0][f"p_{m}_pos"].to_numpy().reshape(n, -1) for m in ("img", "txt")}
    rd = fig_path(cfg, f"reliability_diagrams{suffix}.png")
    reliability_figure(probs, label, lab_used, e["calibration_bins"], rd,
                       f"Profile reliability on {d.split} after tau and offset (seed {d.seeds[0]})")
    for p in (rc, rd):
        write_meta(p, cfg, None, split=d.split, throwaway=cfg["profile"] != "full")
    return [rc, rd]


# ---------------------------------------------------------------- validity controls


def controls(cfg: dict[str, Any], d: pipeline.SplitData) -> dict[str, pd.DataFrame]:
    e, c = cfg["eval"], cfg["eval"]["controls"]
    out: dict[str, list[pd.DataFrame]] = {}

    def add(name: str, table: pd.DataFrame, seed: int) -> None:
        out.setdefault(name, []).append(table.assign(seed=seed))

    for k, (seed, table, y_hat) in enumerate(zip(d.seeds, d.tables, d.y_hat, strict=True)):
        params = json.loads((artifact_dir(cfg) / f"mfsi_params_seed{seed}.json").read_text(encoding="utf-8"))
        r = ct.wide(table, "R")
        theta = theta_vector(params, cfg["mfsi"]["operating_fraction"])
        flags = coherence_flags(y_hat, r, theta)
        esc = escalate(r, theta, flags)
        correct, used = correctness(d.y, y_hat, e["blank_policy"])
        add("agreement_spread", ct.spread_table(table), seed)
        corr, bins = ct.volume_confound(table, esc, c["confound_bins"])
        add("volume_correlation", corr, seed)
        add("volume_bins", bins, seed)
        strong = ct.strong_variables(corr, c["confound_strong"])
        if strong:
            scores = {"mfsi": r, "msp": d.scores[k]["msp"],
                      **{f"mfsi_norm_{v}": ct.rank_normalised(table, v, c["confound_bins"]) for v in strong}}
            add("volume_normalised", evaluate(Inputs([scores], d.y, [y_hat], d.patient_id, e["blank_policy"]),
                                              cfg)["main"], seed)
        add("hierarchy", ct.hierarchy_check(y_hat, flags, correct, used), seed)
        dec, mono = ct.r_deciles(r, correct, used, e["r_deciles"])
        add("r_deciles", dec, seed)
        add("r_deciles_monotonicity", mono, seed)
        per, macro = ct.uncertain_check(r, d.y, d.patient_id, cfg)
        add("uncertain", per, seed)
        add("uncertain_macro", macro, seed)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def controls_figures(cfg: dict[str, Any], tables: dict[str, pd.DataFrame], split: str, seed: int,
                     suffix: str) -> list[Path]:
    vb = tables["volume_bins"]
    vc = fig_path(cfg, f"volume_confound{suffix}.png")
    volume_figure(vb[vb["seed"] == seed], vc, f"Mean R by data volume on {split} (seed {seed})")
    dd = tables["r_deciles"]
    rd = fig_path(cfg, f"r_deciles{suffix}.png")
    deciles_figure(dd[dd["seed"] == seed], rd, f"Accuracy within deciles of R on {split} (seed {seed})")
    for p in (vc, rd):
        write_meta(p, cfg, None, split=split, throwaway=cfg["profile"] != "full")
    return [vc, rd]


# ---------------------------------------------------------------- sensitivity table


def sensitivity(cfg: dict[str, Any], d: pipeline.SplitData, access: TestAccess | None = None,
                oof: bool = True) -> dict[str, pd.DataFrame]:
    """Plan 4 sensitivity rows on d's split (variants fitted on train / val, as the default)."""
    seeds = list(d.seeds)
    policy = cfg["eval"]["blank_policy"]
    dist_name = ab.best_distance(pipeline.run(cfg, d)["main"])
    data = {s: (load_split(cfg, "train", s), load_split(cfg, "val", s), load_split(cfg, d.split, s))
            for s in seeds}
    informative = {}
    for s in seeds:
        probe = artifact_dir(cfg) / f"probe_text_seed{s}.json"
        check_artifact(probe, cfg)
        informative[s] = json.loads(probe.read_text(encoding="utf-8"))["text_informative"]
    sens_dir = artifact_dir(cfg) / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)
    rows, per_rows, extra = [], [], {}
    variants = [{"variable": "(default)", "variant": "default", "set": {}}] + list(cfg["eval"]["sensitivity"])
    default = None
    for v in variants:
        label = {"variable": v["variable"], "variant": v["variant"]}
        if "not_run" in v:
            rows.append({**label, "status": "not run", "note": v["not_run"]})
            continue
        cfg_v = ab.with_overrides(cfg, v["set"])
        if v.get("changes_correctness"):
            d_v = pipeline.collect(cfg_v, d.split, seeds, oof=oof, access=access)
            out = pipeline.run(cfg_v, d_v)
            rule = slug(cfg_v["mfsi"]["yhat_rule"])
            for k in ("main", "per_finding", "cross_modal"):
                extra[f"yhat_{rule}_{k}"] = out[k]
            res = ab.row_result(out, dist_name, None)
            rows.append({**label, "status": "run (own correctness; not paired with default)", **res,
                         "changes_conclusion": _changes(res, default),
                         "note": f"MSP and y_hat from the {rule} rule; see the yhat_{rule}_* tables"})
            continue
        tables = []
        for s in seeds:
            train, val, split_data = data[s]
            params = fit_mfsi(train, val, cfg_v, informative[s], s)
            write_params(params, sens_dir / f"mfsi_params_{slug(v['variant'])}_seed{s}.json", cfg_v, s)
            tables.append(score_split(train, split_data, params, cfg_v))
        out = evaluate(ab.paired_inputs(tables, d.scores, dist_name, d.y, d.y_hat, d.patient_id, policy), cfg)
        res = ab.row_result(out, dist_name, "mfsi_default")
        same = all(np.array_equal(t["R"].to_numpy(), dt["R"].to_numpy())
                   for t, dt in zip(tables, d.tables, strict=True))
        if v["variant"] == "default":
            default = res
        rows.append({**label, "status": "run", **res, "changes_conclusion": _changes(res, default),
                     "escalation_share_range": ab.escalation_range(tables),
                     "note": "R identical to default (escalation only)" if same and v["set"] else ""})
        per_rows.append(ab.per_finding_rows(out, label, d.strata))
    table = pd.DataFrame(rows).assign(best_distance_baseline=dist_name)
    return {"ablations": table, "ablations_per_finding": pd.concat(per_rows, ignore_index=True), **extra}


def _changes(res: dict, default: dict | None) -> bool:
    if default is None:
        return False
    return any(res[k] != default[k] for k in ("vs_msp", "vs_best_distance", "cross_modal_holds"))


def sensitivity_yhat_added_value(cfg: dict[str, Any], d_val: pipeline.SplitData) -> dict[str, pd.DataFrame]:
    """Dev path only: added-value tables under each y_hat sensitivity rule (val, out of fold)."""
    out = {}
    for v in cfg["eval"]["sensitivity"]:
        if v.get("changes_correctness"):
            cfg_v = ab.with_overrides(cfg, v["set"])
            rule = slug(cfg_v["mfsi"]["yhat_rule"])
            d_v = pipeline.collect(cfg_v, "val", list(d_val.seeds), oof=True)
            out.update({f"{rule}_{k}": t for k, t in run_dev(cfg_v, d_v).items()})
    return out
