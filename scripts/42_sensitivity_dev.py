"""Sensitivity analysis on VAL (Phase 4 part C, dev path): one change per row from the default, Phase 3 re-run
in memory (train for clouds, val for tau / thresholds), RQ1 metrics with the default out-of-fold correctness.

Writes outputs/<profile>/tables/rq1_dev_ablations.csv (one row per variant, CIs, conclusions, flag),
rq1_dev_ablations_per_finding.csv, and for the y_hat-rule row its own
rq1_dev_yhat_<rule>_{main,per_finding}.csv and added_value_dev_<rule>_*.csv. Variant MFSI parameters:
artifacts/<profile>/sensitivity/. Never reads test.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from mfsi_cxr.eval import ablations as ab
from mfsi_cxr.eval import pipeline
from mfsi_cxr.eval.added_value_dev import run_dev
from mfsi_cxr.eval.rq1 import evaluate
from mfsi_cxr.mfsi.fit import fit_mfsi, load_split, score_split, write_params
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import artifact_dir, check_artifact, output_dir, write_meta

SPLIT = "val"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write(table: pd.DataFrame, name: str, cfg: dict) -> None:
    out = output_dir(cfg) / "tables" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    write_meta(out, cfg, None, split=SPLIT)


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    seeds = list(cfg["train"]["seeds"])
    policy = cfg["eval"]["blank_policy"]
    d = pipeline.collect(cfg, SPLIT, seeds, oof=True)
    dist_name = ab.best_distance(pipeline.run(cfg, d)["main"])
    data = {s: (load_split(cfg, "train", s), load_split(cfg, SPLIT, s)) for s in seeds}
    informative = {}
    for s in seeds:
        probe = artifact_dir(cfg) / f"probe_text_seed{s}.json"
        check_artifact(probe, cfg)
        informative[s] = json.loads(probe.read_text(encoding="utf-8"))["text_informative"]
    sens_dir = artifact_dir(cfg) / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    rows, per_rows = [], []
    variants = [{"variable": "(default)", "variant": "default", "set": {}}] + list(cfg["eval"]["sensitivity"])
    default_tables = d.tables
    default_conclusions = None
    for v in variants:
        label = {"variable": v["variable"], "variant": v["variant"]}
        if "not_run" in v:
            rows.append({**label, "status": "not run", "note": v["not_run"]})
            continue
        cfg_v = ab.with_overrides(cfg, v["set"])
        if v.get("changes_correctness"):
            d_v = pipeline.collect(cfg_v, SPLIT, seeds, oof=True)
            out = pipeline.run(cfg_v, d_v)
            rule = slug(cfg_v["mfsi"]["yhat_rule"])
            for k in ("main", "per_finding", "cross_modal"):
                write(out[k], f"rq1_dev_yhat_{rule}_{k}", cfg_v)
            for name, table in run_dev(cfg_v, d_v).items():
                write(table, f"added_value_dev_{rule}_{name}", cfg_v)
            res = ab.row_result(out, dist_name, None)
            rows.append({**label, "status": "run (own correctness; not paired with default)", **res,
                         "changes_conclusion": res_changes(res, default_conclusions),
                         "note": "MSP and y_hat from the Youden rule with coherent MSP; see rq1_dev_yhat_*"})
            continue
        tables = []
        for s in seeds:
            train, val = data[s]
            params = fit_mfsi(train, val, cfg_v, informative[s], s)
            write_params(params, sens_dir / f"mfsi_params_{slug(v['variant'])}_seed{s}.json", cfg_v, s)
            tables.append(score_split(train, val, params, cfg_v))
        inp = ab.paired_inputs(tables, d.scores, dist_name, d.y, d.y_hat, d.patient_id, policy)
        out = evaluate(inp, cfg)
        res = ab.row_result(out, dist_name, "mfsi_default")
        same = all(np.array_equal(t["R"].to_numpy(), dt["R"].to_numpy())
                   for t, dt in zip(tables, default_tables, strict=True))
        note = "R identical to default (row changes escalation only)" if same and v["set"] else ""
        if v["variant"] == "default":
            default_conclusions = res
        rows.append({**label, "status": "run", **res,
                     "changes_conclusion": res_changes(res, default_conclusions),
                     "escalation_share_range": ab.escalation_range(tables), "note": note})
        per_rows.append(ab.per_finding_rows(out, label, d.strata))

    table = pd.DataFrame(rows)
    write(table, "rq1_dev_ablations", cfg)
    write(pd.concat(per_rows, ignore_index=True), "rq1_dev_ablations_per_finding", cfg)
    cols = ["variable", "variant", "aurc", "aurc_lo", "aurc_hi", "minus_default", "minus_default_lo",
            "minus_default_hi", "vs_msp", "cross_modal_holds", "vs_best_distance", "changes_conclusion"]
    print(f"best distance baseline (default, macro AURC): {dist_name}")
    print(table[[c for c in cols if c in table]].round(4).to_string(index=False))
    flagged = table[table.get("changes_conclusion", False) == True]  # noqa: E712
    if len(flagged):
        print("FLAG: variants that change a headline conclusion:", flagged["variant"].tolist())
    return 0


def res_changes(res: dict, default: dict | None) -> bool:
    if default is None:
        return False
    return any(res[k] != default[k] for k in ("vs_msp", "vs_best_distance", "cross_modal_holds"))


if __name__ == "__main__":
    sys.exit(main())
