"""Sensitivity-table helpers: overrides, conclusion classes, pairing with the default."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from tests.test_mfsi_fit import INFORMATIVE, _export

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval import ablations as ab
from mfsi_cxr.eval.baselines import all_scores
from mfsi_cxr.eval.rq1 import evaluate
from mfsi_cxr.mfsi.fit import fit_mfsi, score_split


def test_with_overrides_sets_dotted_keys_and_refuses_unknown() -> None:
    cfg = {"mfsi": {"reduce": {"method": "pca"}, "agreement": "cos"}}
    out = ab.with_overrides(cfg, {"mfsi.reduce.method": "none", "mfsi.agreement": "js"})
    assert out["mfsi"]["reduce"]["method"] == "none" and cfg["mfsi"]["reduce"]["method"] == "pca"
    with pytest.raises(KeyError):
        ab.with_overrides(cfg, {"mfsi.agreemnt": "js"})


def test_classify() -> None:
    assert ab.classify(0.01, 0.05) == "MFSI better"
    assert ab.classify(-0.05, -0.01) == "other better"
    assert ab.classify(-0.01, 0.02) == "no difference"


def test_default_row_pairs_with_itself(smoke_cfg: dict[str, Any]) -> None:
    cfg = copy.deepcopy(smoke_cfg)
    cfg["eval"]["bootstrap_B"] = 20
    train, val = _export(300, "train", 0), _export(150, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    table = score_split(train, val, params, cfg)
    n = len(val["study_id"])
    q = table["prob_cal"].to_numpy().reshape(n, len(FINDINGS))
    scores = all_scores(train, val, table, cfg, params, q, [q])
    y_hat = table["y_hat"].to_numpy().reshape(n, len(FINDINGS))
    inp = ab.paired_inputs([table], [scores], "euclid_img", val["y"], [y_hat], val["patient_id"], "negative")
    res = ab.row_result(evaluate(inp, cfg), "euclid_img", "mfsi_default")
    assert res["minus_default"] == 0 and res["minus_default_lo"] == 0 and res["minus_default_hi"] == 0
    cfg_v = ab.with_overrides(cfg, {"mfsi.covariance": "tied"})
    t_v = score_split(train, val, fit_mfsi(train, val, cfg_v, INFORMATIVE, seed=0), cfg_v)
    inp_v = ab.paired_inputs([t_v], [scores], "euclid_img", val["y"], [y_hat], val["patient_id"], "negative")
    res_v = ab.row_result(evaluate(inp_v, cfg), "euclid_img", "mfsi_default")
    assert res_v["minus_default_lo"] <= res_v["minus_default"] <= res_v["minus_default_hi"]
    assert not np.array_equal(t_v["R"], table["R"])
