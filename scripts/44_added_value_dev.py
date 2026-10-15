"""Added value beyond MSP, development path on VAL (pre-registered secondary RQ1 analysis; Phase 4).

(a) out-of-fold combined score [MSP, MFSI] by patient-grouped CV on val (nested calibration: y_hat v2) vs
    MSP (delta-AURC, paired bootstrap), standardised coefficients from the full-val fit (negative MFSI
    coefficients flagged), DeLong + BH;
(b) stratified AUROC of MFSI within val MSP quintiles. Development only: never used to change the spec
(docs/PREREG_ADDED_VALUE.md). Never reads test. Writes outputs/<profile>/tables/added_value_dev_*.csv.
"""

from __future__ import annotations

import sys

import pandas as pd

from mfsi_cxr.eval import pipeline
from mfsi_cxr.eval.added_value_dev import run_dev
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta

SPLIT = "val"


def write_tables(tables: dict[str, pd.DataFrame], prefix: str, cfg: dict) -> None:
    for name, table in tables.items():
        out = output_dir(cfg) / "tables" / f"{prefix}_{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        write_meta(out, cfg, None, split=SPLIT)


def report(t: dict[str, pd.DataFrame]) -> None:
    print("(a) eligible findings:", t["a_macro"]["findings"].iloc[0])
    print(t["a_macro"].round(4).to_string(index=False))
    neg = t["a_coefficients"][t["a_coefficients"]["mfsi_coef_negative"]]
    if len(neg):
        print("FLAG: negative standardised MFSI coefficient for", sorted(set(neg["finding"])))
    print("(b) findings entering:", t["b_macro"]["findings"].iloc[0])
    print(t["b_macro"].round(4).to_string(index=False))
    merged = t["b_strata"].groupby(["seed_index", "finding"])["merged_edges"].first()
    if (merged > 0).any():
        print("merged quintile edges:", merged[merged > 0].to_dict())


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    d = pipeline.collect(cfg, SPLIT, list(cfg["train"]["seeds"]), oof=True)
    tables = run_dev(cfg, d)
    write_tables(tables, "added_value_dev", cfg)
    report(tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
