"""Added value beyond MSP, development path on VAL (pre-registered secondary RQ1 analysis; Phase 4).

(a) out-of-fold combined score [MSP, MFSI] by patient-grouped CV on val (nested calibration: y_hat v2) vs MSP
    (delta-AURC, paired bootstrap),
    standardised coefficients from the full-val fit (negative MFSI coefficients flagged), DeLong + BH;
(b) stratified AUROC of MFSI within val MSP quintiles. Development only: never used to change the spec
(docs/PREREG_ADDED_VALUE.md). Never reads test. Writes outputs/<profile>/tables/added_value_dev_*.csv.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval import added_value as av
from mfsi_cxr.eval import pipeline
from mfsi_cxr.eval.selective import correctness
from mfsi_cxr.mfsi import yhat
from mfsi_cxr.mfsi.fit import tau_targets
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta

SPLIT = "val"


def write(table: pd.DataFrame, name: str, cfg: dict) -> None:
    out = output_dir(cfg) / "tables" / f"added_value_dev_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    write_meta(out, cfg, None, split=SPLIT)


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    a_cfg = cfg["eval"]["added_value"]
    d = pipeline.collect(cfg, SPLIT, list(cfg["train"]["seeds"]), oof=True)
    m = cfg["mfsi"]
    label, lab_used = tau_targets(d.y, m["uncertain_policy"])
    policy = cfg["eval"]["blank_policy"]
    seeds_arr = []
    for scores, y_hat in zip(d.scores, d.y_hat, strict=True):
        c, u = correctness(d.y, y_hat, policy)
        seeds_arr.append(av.SeedArrays(scores["msp"], scores["mfsi"], c, u))

    def inner_for(s: int):
        """Nested calibration for outer fold k: rule fitted on folds other than {row's fold, k}."""
        def inner(k: int) -> av.SeedArrays:
            _, yh, msp, _ = yhat.out_of_fold(m["yhat_rule"], d.probs[s], label, lab_used, d.folds,
                                             m["platt_eps"], exclude=k)
            c, u = correctness(d.y, yh, policy)
            return av.SeedArrays(msp, seeds_arr[s].mfsi, c, u & (d.folds != k)[:, None])
        return inner

    # (a): eligible for every seed, then the CV fold check (dev only)
    eligible = np.logical_and.reduce([av.eligible_a(s, a_cfg) for s in seeds_arr])
    oof, ok = [], eligible.copy()
    for k, s in enumerate(seeds_arr):
        pred, ok_s = av.crossval_combined(s, d.folds, a_cfg, eligible, inner_for(k))
        oof.append(pred)
        ok &= ok_s
    per_a, macro_a = av.evaluate_a(oof, seeds_arr, d.patient_id, ok, cfg)
    counts = pd.DataFrame([{"seed": seed, "finding": f, "n_correct": int(s.correct[s.used[:, j], j].sum()),
                            "n_wrong": int((~s.correct[s.used[:, j], j].astype(bool)).sum())}
                           for seed, s in zip(d.seeds, seeds_arr, strict=True)
                           for j, f in enumerate(FINDINGS)])
    coefs = av.coefficient_table([av.fit_combined(s, a_cfg, eligible) for s in seeds_arr], d.seeds)
    write(per_a, "a_per_finding", cfg)
    write(macro_a, "a_macro", cfg)
    write(coefs, "a_coefficients", cfg)
    write(counts, "a_eligibility_counts", cfg)
    write(av.delong_a(oof, seeds_arr, ok, d.seeds, cfg["eval"]["alpha"]), "a_delong", cfg)

    # (b): edges from val, evaluated on val (dev)
    plan = av.plan_strata(seeds_arr, seeds_arr, a_cfg)
    per_b, macro_b = av.evaluate_b(seeds_arr, d.patient_id, plan, cfg)
    write(per_b, "b_per_finding", cfg)
    write(macro_b, "b_macro", cfg)
    write(plan.counts, "b_strata", cfg)
    write(pipeline.sparse_fold_table(d), "calibration_sparse_folds", cfg)

    print("(a) eligible findings:", macro_a["findings"].iloc[0])
    print(macro_a.round(4).to_string(index=False))
    neg = coefs[coefs["mfsi_coef_negative"]]
    if len(neg):
        print("FLAG: negative standardised MFSI coefficient for", sorted(set(neg["finding"])))
    print("(b) findings entering:", macro_b["findings"].iloc[0])
    print(macro_b.round(4).to_string(index=False))
    merged = plan.counts.groupby(["seed_index", "finding"])["merged_edges"].first()
    if (merged > 0).any():
        print("merged quintile edges:", merged[merged > 0].to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
