"""Added value beyond MSP (pre-registered; docs/PREREG_ADDED_VALUE.md).

run_dev: VAL dev path (scripts/44, and the y_hat sensitivity rows of 42): (a) out-of-fold combined score with
nested calibration (y_hat v2 amendment 4); (b) stratified AUROC within val MSP quintiles. `d` must come from
pipeline.collect(..., oof=True) under the same cfg.
run_final: final path (scripts/41): (a) one model per finding and seed fitted on all val scored rows, applied
to test, eligibility decided on val; (b) quintile edges from val, eligibility fixed on test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval import added_value as av
from mfsi_cxr.eval import pipeline
from mfsi_cxr.eval.selective import correctness
from mfsi_cxr.mfsi import yhat
from mfsi_cxr.mfsi.fit import tau_targets


def run_dev(cfg: dict[str, Any], d: pipeline.SplitData) -> dict[str, pd.DataFrame]:
    a_cfg = cfg["eval"]["added_value"]
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

    # (b): edges from val, evaluated on val (dev)
    plan = av.plan_strata(seeds_arr, seeds_arr, a_cfg)
    per_b, macro_b = av.evaluate_b(seeds_arr, d.patient_id, plan, cfg)
    return {"a_per_finding": per_a, "a_macro": macro_a, "a_coefficients": coefs,
            "a_eligibility_counts": counts,
            "a_delong": av.delong_a(oof, seeds_arr, ok, d.seeds, cfg["eval"]["alpha"]),
            "b_per_finding": per_b, "b_macro": macro_b, "b_strata": plan.counts,
            "calibration_sparse_folds": pipeline.sparse_fold_table(d)}


def _arrays(cfg: dict[str, Any], d: pipeline.SplitData) -> list[av.SeedArrays]:
    out = []
    for scores, y_hat in zip(d.scores, d.y_hat, strict=True):
        c, u = correctness(d.y, y_hat, cfg["eval"]["blank_policy"])
        out.append(av.SeedArrays(scores["msp"], scores["mfsi"], c, u))
    return out


def run_final(cfg: dict[str, Any], d_val: pipeline.SplitData,
              d_test: pipeline.SplitData) -> dict[str, pd.DataFrame]:
    """Pre-registered final path; d_val from the frozen rule (oof=False), d_test behind the freeze guard."""
    a_cfg = cfg["eval"]["added_value"]
    val, test = _arrays(cfg, d_val), _arrays(cfg, d_test)
    eligible = np.logical_and.reduce([av.eligible_a(s, a_cfg) for s in val])
    models = [av.fit_combined(s, a_cfg, eligible) for s in val]
    comb = [av.score_combined(m, t) for m, t in zip(models, test, strict=True)]
    per_a, macro_a = av.evaluate_a(comb, test, d_test.patient_id, eligible, cfg)
    counts = pd.DataFrame([{"seed": seed, "finding": f,
                            "n_correct_val": int(s.correct[s.used[:, j], j].sum()),
                            "n_wrong_val": int((~s.correct[s.used[:, j], j].astype(bool)).sum())}
                           for seed, s in zip(d_val.seeds, val, strict=True) for j, f in enumerate(FINDINGS)])
    plan = av.plan_strata(val, test, a_cfg)
    per_b, macro_b = av.evaluate_b(test, d_test.patient_id, plan, cfg)
    delong = av.delong_a(comb, test, eligible, d_val.seeds, cfg["eval"]["alpha"])
    return {"a_per_finding": per_a, "a_macro": macro_a,
            "a_coefficients": av.coefficient_table(models, d_val.seeds),
            "a_eligibility_counts": counts, "a_delong": delong,
            "b_per_finding": per_b, "b_macro": macro_b, "b_strata": plan.counts}
