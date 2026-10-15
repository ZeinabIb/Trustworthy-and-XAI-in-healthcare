"""Typicality variant (docs/PREREG_TYPICALITY.md): tail survival, factor, wiring into MFSI."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from tests.test_mfsi_fit import INFORMATIVE, _export

from mfsi_cxr.mfsi import typicality
from mfsi_cxr.mfsi.fit import fit_mfsi, score_split


def test_survival_and_factor() -> None:
    ref = np.arange(1.0, 101.0)                       # 100 reference distances 1..100
    assert typicality.survival(np.array([0.0]), ref)[0] == pytest.approx(101 / 101)
    assert typicality.survival(np.array([100.0]), ref)[0] == pytest.approx(2 / 101)
    assert typicality.survival(np.array([1e9]), ref)[0] == pytest.approx(1 / 101)
    t = typicality.factor(np.array([10.0, 96.0, 98.0, 100.0, 1e9]), ref, 0.05)
    assert t[0] == 1.0 and t[1] == 1.0  # S = 6/101 > q: inside the central 95%
    assert t[2] == pytest.approx((1 + 3) / 101 / 0.05)
    assert t[3] == pytest.approx(2 / 101 / 0.05) and t[4] == pytest.approx(1 / 101 / 0.05)
    assert np.all(np.diff(t) <= 0)                     # farther = never more typical


def test_far_cases_get_low_t_and_variant_scales_r(smoke_cfg: dict[str, Any]) -> None:
    cfg = copy.deepcopy(smoke_cfg)
    train, val = _export(300, "train", 0), _export(150, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    assert len(params["typicality_img"]["reference"]) == 150
    far = dict(val, z_img=val["z_img"] * 25)           # degraded: far from all image data
    base, moved = score_split(train, val, params, cfg), score_split(train, far, params, cfg)
    assert base["T_img"].mean() > 0.9 and moved["T_img"].max() < 0.2
    assert np.array_equal(moved["T_txt"], base["T_txt"])   # text untouched
    cfg["mfsi"]["typicality"]["enabled"] = True
    on = score_split(train, far, params, cfg)
    assert np.allclose(on["r_img"], moved["r_img"] * moved["T_img"])
    assert np.allclose(score_split(train, val, params, smoke_cfg)["R"], base["R"])  # default unchanged
