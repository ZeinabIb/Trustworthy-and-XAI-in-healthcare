"""Sensitivity analysis on VAL (Phase 4 part C, dev path): one change per row from the default, Phase 3 re-run
in memory (train for clouds, val for tau / thresholds), RQ1 metrics with the default out-of-fold correctness.

Writes outputs/<profile>/tables/rq1_dev_ablations{,_per_finding}.csv, the y_hat-rule rows' own
rq1_dev_yhat_<rule>_*.csv and added_value_dev_<rule>_*.csv. Variant MFSI parameters:
artifacts/<profile>/sensitivity/. Never reads test.
"""

from __future__ import annotations

import sys

from mfsi_cxr.eval import pipeline, runs
from mfsi_cxr.utils.cli import base_parser, config_from_args

SPLIT = "val"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    d = pipeline.collect(cfg, SPLIT, list(cfg["train"]["seeds"]), oof=True)
    t = runs.sensitivity(cfg, d)
    runs.write_tables(cfg, t, "rq1_dev_", SPLIT)
    runs.write_tables(cfg, runs.sensitivity_yhat_added_value(cfg, d), "added_value_dev_", SPLIT)
    table = t["ablations"]
    cols = ["variable", "variant", "aurc", "aurc_lo", "aurc_hi", "minus_default", "minus_default_lo",
            "minus_default_hi", "vs_msp", "cross_modal_holds", "vs_best_distance", "changes_conclusion"]
    print(f"best distance baseline (default, macro AURC): {table['best_distance_baseline'].iloc[0]}")
    print(table[[c for c in cols if c in table]].round(4).to_string(index=False))
    flagged = table[table["changes_conclusion"].fillna(False).astype(bool)]
    if len(flagged):
        print("FLAG: variants that change a headline conclusion:", flagged["variant"].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
