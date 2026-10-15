"""RQ1 development evaluation on VAL (Phase 4): MFSI vs every baseline, per finding and macro, with
patient-bootstrap CIs and sanity anchors; blank-label sensitivity; decisions; calibration of the profiles.
Out-of-fold y_hat / MSP (y_hat v2). Never reads test.

Writes outputs/<profile>/tables/rq1_dev_*.csv and figures/{risk_coverage,reliability_diagrams}_dev.png. Prints
the plan's STOP-AND-ASK conditions: a baseline beating MFSI (macro AURC CI entirely below MFSI's) and a large
blank-policy shift.
"""

from __future__ import annotations

import sys

from mfsi_cxr.eval import pipeline, runs
from mfsi_cxr.utils.cli import base_parser, config_from_args

SPLIT = "val"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    e = cfg["eval"]
    d = pipeline.collect(cfg, SPLIT, list(cfg["train"]["seeds"]), oof=True)
    t = runs.rq1(cfg, d)
    runs.write_tables(cfg, t, "rq1_dev_", SPLIT)
    runs.rq1_figures(cfg, d, "_dev")

    cols = ["score", "n_findings", "aurc", "aurc_lo", "aurc_hi", "aurc_minus_ref", "aurc_minus_ref_lo",
            "aurc_minus_ref_hi", "auroc"]
    print(t["main"][cols].sort_values("aurc").round(4).to_string(index=False))
    cm = t["cross_modal"]
    if not cm.empty:
        print("cross-modal claim holds for:", sorted(cm.loc[cm["claim_holds"], "finding"]) or "none")
    better = t["main"][(t["main"]["score"] != e["reference"]) & (t["main"]["aurc_minus_ref_hi"] < 0)]
    if len(better):
        print(f"STOP-AND-ASK: macro AURC CI below {e['reference']} for {sorted(better['score'])}")
    sens = t["blank_sensitivity"]
    print("exclude-blanks sensitivity eligible findings:", sens["eligible_findings"].iloc[0])
    big = sens[sens["exceeds_warn"]] if "exceeds_warn" in sens else sens.iloc[:0]
    if len(big):
        print(f"STOP-AND-ASK: blank policy shifts macro AURC (eligible findings) by > "
              f"{e['blank_shift_warn']} for {sorted(big['score'])}")
    calib = t["profile_calibration"]
    print("profile ECE (equal mass, mean over findings):",
          calib.groupby("modality")[["ece_before", "ece_after"]].mean().round(4).to_dict("index"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
