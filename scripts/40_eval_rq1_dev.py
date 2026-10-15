"""RQ1 development evaluation on VAL (Phase 4): MFSI vs every baseline, per finding and macro, with
patient-bootstrap CIs; blank-label sensitivity; calibration of the profiles. Never reads test.

Writes outputs/<profile>/tables/rq1_dev_{main,per_finding,delong,cross_modal,blank_sensitivity,
profile_calibration}.csv and figures/{risk_coverage,reliability_diagrams}_dev.png. Prints the plan's
STOP-AND-ASK conditions: a baseline beating MFSI (macro AURC CI entirely below MFSI's) and a large
blank-policy shift.
"""

from __future__ import annotations

import sys

import pandas as pd

from mfsi_cxr.eval import pipeline
from mfsi_cxr.eval.figures import reliability_figure, risk_coverage_figure
from mfsi_cxr.eval.selective import correctness
from mfsi_cxr.mfsi.fit import tau_targets
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta

SPLIT = "val"
FIG_SCORES = ["mfsi", "mfsi_img", "mfsi_txt", "msp"]


def write(table: pd.DataFrame, name: str, cfg: dict) -> None:
    out = output_dir(cfg) / "tables" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    write_meta(out, cfg, None, split=SPLIT)


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    e = cfg["eval"]
    d = pipeline.collect(cfg, SPLIT, list(cfg["train"]["seeds"]))
    res = pipeline.run(cfg, d)
    other = "exclude" if e["blank_policy"] == "negative" else "negative"
    res_other = pipeline.run(cfg, d, other)
    for key, table in res.items():
        write(table, f"rq1_dev_{key}", cfg)
    sens = pipeline.blank_sensitivity(cfg, res["main"], res_other["main"])
    write(sens, "rq1_dev_blank_sensitivity", cfg)
    calib = pipeline.profile_calibration(cfg, d)
    write(calib, "rq1_dev_profile_calibration", cfg)

    figs = output_dir(cfg) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    correct, used = correctness(d.y, d.y_hat[0], e["blank_policy"])
    risk_coverage_figure(d.scores[0], correct, used, FIG_SCORES, figs / "risk_coverage_dev.png",
                         f"Risk-coverage on {SPLIT} (seed {d.seeds[0]}, {cfg['profile']})")
    label, lab_used = tau_targets(d.y, cfg["mfsi"]["uncertain_policy"])
    n = len(d.y)
    probs = {f"p_{m}+": d.tables[0][f"p_{m}_pos"].to_numpy().reshape(n, -1) for m in ("img", "txt")}
    reliability_figure(probs, label, lab_used, e["calibration_bins"], figs / "reliability_diagrams_dev.png",
                       f"Profile reliability on {SPLIT} after tau and offset (seed {d.seeds[0]})")
    for name in ("risk_coverage_dev.png", "reliability_diagrams_dev.png"):
        write_meta(figs / name, cfg, None, split=SPLIT)

    cols = ["score", "n_findings", "aurc", "aurc_lo", "aurc_hi", "aurc_minus_ref", "aurc_minus_ref_lo",
            "aurc_minus_ref_hi", "auroc"]
    print(res["main"][cols].sort_values("aurc").round(4).to_string(index=False))
    cm = res["cross_modal"]
    if not cm.empty:
        print("cross-modal claim holds for:", sorted(cm.loc[cm["claim_holds"], "finding"]) or "none")
    better = res["main"][(res["main"]["score"] != e["reference"]) & (res["main"]["aurc_minus_ref_hi"] < 0)]
    if len(better):
        print(f"STOP-AND-ASK: macro AURC CI below {e['reference']} for {sorted(better['score'])}")
    big = sens[sens["exceeds_warn"]]
    if len(big):
        print(f"STOP-AND-ASK: blank policy shifts macro AURC by > {e['blank_shift_warn']} for "
              f"{sorted(big['score'])}")
    print("profile ECE (equal mass, mean over findings):",
          calib.groupby("modality")[["ece_before", "ece_after"]].mean().round(4).to_dict("index"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
