"""Validity controls on VAL (Phase 4 part D, dev path; out-of-fold y_hat for correctness, flags, escalation).

Per seed: agreement spread + modality dominance, data-volume confound (with rank-normalised R where a
correlation is strong), hierarchy flags, accuracy within deciles of R, and the uncertain-label check. Writes
outputs/<profile>/tables/controls_dev_*.csv and figures/{volume_confound,r_deciles}_dev.png. Never reads test.
The degrade-one-modality test is scripts/45_degrade_dev.py (it re-extracts embeddings).
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from mfsi_cxr.eval import controls as ct
from mfsi_cxr.eval import pipeline
from mfsi_cxr.eval.figures import deciles_figure, volume_figure
from mfsi_cxr.eval.rq1 import Inputs, evaluate
from mfsi_cxr.eval.selective import correctness
from mfsi_cxr.mfsi.escalation import escalate
from mfsi_cxr.mfsi.fit import theta_vector
from mfsi_cxr.mfsi.hierarchy import coherence_flags
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import artifact_dir, output_dir, write_meta

SPLIT = "val"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    e, c = cfg["eval"], cfg["eval"]["controls"]
    d = pipeline.collect(cfg, SPLIT, list(cfg["train"]["seeds"]), oof=True)
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
            res = evaluate(Inputs([scores], d.y, [y_hat], d.patient_id, e["blank_policy"]), cfg)["main"]
            add("volume_normalised", res, seed)
        add("hierarchy", ct.hierarchy_check(y_hat, flags, correct, used), seed)
        dec, mono = ct.r_deciles(r, correct, used, e["r_deciles"])
        add("r_deciles", dec, seed)
        add("r_deciles_monotonicity", mono, seed)
        per, macro = ct.uncertain_check(r, d.y, d.patient_id, cfg)
        add("uncertain", per, seed)
        add("uncertain_macro", macro, seed)
        print(f"seed {seed}: strong volume correlations (|rho| >= {c['confound_strong']}):", strong or "none")
        print(macro.round(4).to_string(index=False))

    tables, figs = output_dir(cfg) / "tables", output_dir(cfg) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    for name, parts in out.items():
        path = tables / f"controls_dev_{name}.csv"
        pd.concat(parts, ignore_index=True).to_csv(path, index=False)
        write_meta(path, cfg, None, split=SPLIT)
    first = d.seeds[0]
    bins = pd.concat(out["volume_bins"])
    volume_figure(bins[bins["seed"] == first], figs / "volume_confound_dev.png",
                  f"Mean R by data volume on {SPLIT} (seed {first})")
    dec = pd.concat(out["r_deciles"])
    deciles_figure(dec[dec["seed"] == first], figs / "r_deciles_dev.png",
                   f"Accuracy within deciles of R on {SPLIT} (seed {first})")
    for name in ("volume_confound_dev.png", "r_deciles_dev.png"):
        write_meta(figs / name, cfg, None, split=SPLIT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
