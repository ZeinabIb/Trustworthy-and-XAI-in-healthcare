"""Sensitivity analysis (Phase 4, plan 4 table): each row re-runs Phase 3 with exactly one change from the
default (configs/eval.yaml `eval.sensitivity`), then the RQ1 metrics with the DEFAULT correctness, so a row's
MFSI is compared with the default MFSI on the same rows and bootstrap draws (paired).

Per row: macro AURC of the variant's MFSI with its CI, the variant - default difference with its CI, and the
three headline conclusions recomputed with the variant (MFSI vs MSP; cross-modal claim count with the
variant's own single-modality margins; MFSI vs the best distance baseline of the default run). A row is
flagged when any conclusion differs from the default row. Rows whose R equals the default R everywhere
(they only change escalation) are run and labelled. A row with `changes_correctness` (the y_hat rule)
changes `correct` itself and is evaluated on its own (not paired with the default). A `not_run` row is
reported as not run.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval.rq1 import Inputs

DISTANCE_PREFIXES = ("tied_maha_", "euclid_", "lof_")


def with_overrides(cfg: dict[str, Any], sets: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of cfg with dotted keys set (e.g. {'mfsi.reduce.method': 'none'}); unknown keys refused."""
    out = copy.deepcopy(cfg)
    for dotted, value in sets.items():
        node = out
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        if last not in node:
            raise KeyError(f"sensitivity override of an unknown key: {dotted}")
        node[last] = value
    return out


def classify(lo: float, hi: float) -> str:
    """Sign class of (score - MFSI) AURC: > 0 = MFSI better, < 0 = score better, else no difference."""
    if lo > 0:
        return "MFSI better"
    if hi < 0:
        return "other better"
    return "no difference"


def best_distance(main: pd.DataFrame) -> str:
    rows = main[main["score"].str.startswith(DISTANCE_PREFIXES)]
    return str(rows.sort_values("aurc")["score"].iloc[0])


def row_result(out: dict[str, pd.DataFrame], dist_name: str, default_name: str | None) -> dict[str, Any]:
    """Headline numbers and conclusions of one evaluated row (reference score = 'mfsi')."""
    m = out["main"].set_index("score")
    r: dict[str, Any] = {"aurc": m.loc["mfsi", "aurc"], "aurc_lo": m.loc["mfsi", "aurc_lo"],
                         "aurc_hi": m.loc["mfsi", "aurc_hi"], "auroc": m.loc["mfsi", "auroc"],
                         "aurc_random": m.loc["mfsi", "aurc_random"]}
    if default_name is not None:  # variant - default = -(default - variant)
        r["minus_default"] = -m.loc[default_name, "aurc_minus_ref"]
        r["minus_default_lo"] = -m.loc[default_name, "aurc_minus_ref_hi"]
        r["minus_default_hi"] = -m.loc[default_name, "aurc_minus_ref_lo"]
    r["vs_msp"] = classify(m.loc["msp", "aurc_minus_ref_lo"], m.loc["msp", "aurc_minus_ref_hi"])
    r["vs_best_distance"] = classify(m.loc[dist_name, "aurc_minus_ref_lo"],
                                     m.loc[dist_name, "aurc_minus_ref_hi"])
    cm = out["cross_modal"]
    r["cross_modal_holds"] = int(cm["claim_holds"].sum()) if not cm.empty else 0
    return r


def paired_inputs(variant_tables: list[pd.DataFrame], default_scores: list[dict[str, np.ndarray]],
                  dist_name: str, y: np.ndarray, y_hat: list[np.ndarray], patient_id: np.ndarray,
                  blank_policy: str) -> Inputs:
    """Inputs with the variant's mfsi / mfsi_img / mfsi_txt next to the default MFSI, MSP and the distance
    baseline, all on the default correctness."""
    scores = []
    n = len(y)
    for t, ds in zip(variant_tables, default_scores, strict=True):
        v = {"mfsi": t["R"].to_numpy().reshape(n, len(FINDINGS)),
             "mfsi_img": t["r_img"].to_numpy().reshape(n, len(FINDINGS)),
             "mfsi_txt": t["r_txt"].to_numpy().reshape(n, len(FINDINGS))}
        scores.append({**v, "mfsi_default": ds["mfsi"], "msp": ds["msp"], dist_name: ds[dist_name]})
    return Inputs(scores, y, y_hat, patient_id, blank_policy)


def per_finding_rows(out: dict[str, pd.DataFrame], label: dict[str, str],
                     strata: dict[str, str]) -> pd.DataFrame:
    pf = out["per_finding"]
    v = pf[pf["score"] == "mfsi"].set_index("finding")
    dflt = pf[pf["score"] == "mfsi_default"].set_index("finding")
    res = pd.DataFrame({"aurc": v["aurc"], "aurc_lo": v["aurc_lo"], "aurc_hi": v["aurc_hi"],
                        "minus_default": -dflt["aurc_minus_ref"],
                        "minus_default_lo": -dflt["aurc_minus_ref_hi"],
                        "minus_default_hi": -dflt["aurc_minus_ref_lo"],
                        "stratum": v.index.map(strata)}).reset_index()
    for k, val in label.items():
        res.insert(0, k, val)
    return res


def escalation_range(tables: list[pd.DataFrame]) -> str:
    """Min-max over findings of the val escalation share at the operating point (seed 0)."""
    share = tables[0].groupby("finding")["escalate"].mean()
    return f"{share.min():.3f}-{share.max():.3f}"
