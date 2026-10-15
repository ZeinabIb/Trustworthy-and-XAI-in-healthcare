"""Validity controls on synthetic tables."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval import controls as ct

N, L = 400, len(FINDINGS)
RNG = np.random.default_rng(0)


def _table(r: np.ndarray, n_images: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"finding": np.tile(FINDINGS, N), "R": r.ravel(),
                         "n_studies": np.repeat(1 + n_images // 3, L), "n_images": np.repeat(n_images, L),
                         "text_len": np.repeat(np.full(N, 7), L)})


def test_volume_frame_and_strong_correlation() -> None:
    n_images = RNG.integers(1, 12, N)
    r = np.clip(n_images[:, None] / 12 + RNG.normal(0, 0.05, (N, L)), 0, 1)  # R driven by volume
    t = _table(r, n_images)
    vol = ct.volume_frame(t)
    assert (vol["n_priors"] == vol["n_studies"] - 1).all()
    corr, bins = ct.volume_confound(t, np.zeros((N, L), bool), 4)
    assert "n_images" in ct.strong_variables(corr, 0.3)
    assert corr.loc[corr["variable"] == "text_len", "spearman_R"].isna().all()  # constant -> undefined
    assert set(bins.loc[bins["variable"] == "text_len", "bin"]) == {0}


def test_rank_normalised_removes_between_bin_level() -> None:
    n_images = RNG.integers(1, 12, N)
    r = np.clip(n_images[:, None] / 12 + RNG.normal(0, 0.05, (N, L)), 0, 1)
    norm = ct.rank_normalised(_table(r, n_images), "n_images", 4)
    assert norm.min() > 0 and norm.max() <= 1
    corr = np.corrcoef(n_images, norm[:, 0])[0, 1]
    assert abs(corr) < abs(np.corrcoef(n_images, r[:, 0])[0, 1])


def test_r_deciles_monotone_when_r_predicts_correctness() -> None:
    r = RNG.uniform(size=(N, L))
    correct = RNG.random((N, L)) < r
    dec, mono = ct.r_deciles(r, correct, np.ones((N, L), bool), 10)
    assert (mono["n_bins"] == 10).all() and (mono["spearman_bin_accuracy"] > 0.5).all()
    assert dec.groupby("finding")["n"].sum().eq(N).all()


def test_hierarchy_check_rates() -> None:
    y_hat = np.zeros((4, L), bool)
    flags = np.zeros((4, L), bool)
    correct = np.ones((4, L), bool)
    j = FINDINGS.index("Edema")
    flags[:2, j] = True
    correct[0, j] = False
    t = ct.hierarchy_check(y_hat, flags, correct, np.ones((4, L), bool)).set_index("finding")
    assert t.loc["Edema", "flag_rate"] == 0.5 and t.loc["Edema", "error_flagged"] == 0.5
    assert t.loc["Edema", "error_unflagged"] == 0.0 and t.loc["Edema", "has_ancestor"]
    assert not t.loc["Fracture", "has_ancestor"]


def test_uncertain_check_detects_lower_r(smoke_cfg: dict[str, Any]) -> None:
    cfg = copy.deepcopy(smoke_cfg)
    cfg["eval"]["bootstrap_B"] = 50
    y = RNG.choice([1.0, 0.0, -1.0], p=[0.4, 0.4, 0.2], size=(N, L))
    r = np.where(y == -1, 0.3, 0.7) + RNG.normal(0, 0.05, (N, L))
    per, macro = ct.uncertain_check(r, y, np.array([f"p{i}" for i in range(N)]), cfg)
    assert per["eligible"].all() and (per["diff"] < 0).all() and (per["mannwhitney_p"] < 0.01).all()
    assert macro["lower_for_uncertain"].iloc[0]
    assert macro["macro_diff"].iloc[0] == pytest.approx(-0.4, abs=0.02)
    y[:, 0] = np.where(y[:, 0] == -1, 0.0, y[:, 0])  # no uncertain rows -> not eligible
    per, _ = ct.uncertain_check(r, y, np.array([f"p{i}" for i in range(N)]), cfg)
    assert not per.loc[0, "eligible"]
