"""PCA projection before MFSI (mfsi.reduce): k rule, train-only fit, conditioning on rank-deficient input."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from tests.test_mfsi_fit import INFORMATIVE, _export

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.mfsi.fit import fit_mfsi, score_split
from mfsi_cxr.mfsi.reduce import fit_projection, n_components

PCA = {"method": "pca", "variance": 0.99}


@pytest.fixture
def cfg(smoke_cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(smoke_cfg)


def test_n_components_is_smallest_k_reaching_the_share() -> None:
    explained = np.array([50.0, 30.0, 15.0, 4.0, 1.0])  # cumulative 0.5 0.8 0.95 0.99 1.0
    assert n_components(explained, 0.5) == 1
    assert n_components(explained, 0.9) == 3
    assert n_components(explained, 0.99) == 4
    assert n_components(explained, 1.0) == 5


def test_projection_keeps_requested_variance_and_is_orthonormal() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 40)) * np.geomspace(10, 0.01, 40)
    proj = fit_projection(z, PCA, "t")
    assert proj.variance_kept >= 0.99 and proj.k < 40
    assert np.allclose(proj.basis.T @ proj.basis, np.eye(proj.k))
    smaller = fit_projection(z, {"method": "pca", "variance": 0.9}, "t")
    assert smaller.k < proj.k


def test_none_is_identity() -> None:
    z = np.random.default_rng(1).normal(size=(50, 6))
    proj = fit_projection(z, {"method": "none", "variance": 0.99}, "t")
    assert proj(z) is z and proj.k == 6


def test_unknown_method_fails() -> None:
    with pytest.raises(ValueError):
        fit_projection(np.zeros((5, 3)), {"method": "umap", "variance": 0.99}, "t")


def _rank_deficient(data: dict[str, np.ndarray], d_out: int = 40) -> dict[str, np.ndarray]:
    """Lift 12-d embeddings to d_out dims by a fixed linear map (like the 768 -> 1024 text projection)."""
    lift = np.random.default_rng(7).normal(size=(data["z_img"].shape[1], d_out))
    return dict(data, z_img=data["z_img"] @ lift, z_txt=data["z_txt"] @ lift)


def test_rank_deficient_embeddings_need_no_ridge_after_pca(cfg: dict[str, Any]) -> None:
    train, val = _rank_deficient(_export(400, "train", 0)), _rank_deficient(_export(200, "val", 1))
    cfg["mfsi"]["reduce"] = {"method": "none", "variance": 0.99}
    raw = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    assert any(raw["cov_txt"][f]["ridge_neg"] > 0 for f in FINDINGS)  # singular in 40-d: rank 12
    cfg["mfsi"]["reduce"] = PCA
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    for name in ("img", "txt"):
        assert params[f"reduce_{name}"]["k"] <= 12 and params[f"reduce_{name}"]["variance_kept"] >= 0.99
        assert all(params[f"cov_{name}"][f]["ridge_neg"] == 0 for f in FINDINGS)


def test_projection_is_fitted_on_train_only(cfg: dict[str, Any]) -> None:
    cfg["mfsi"]["reduce"] = PCA
    train, val = _export(300, "train", 0), _export(150, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    shifted = dict(val, z_img=val["z_img"] * 100 + 5)  # val geometry must not change the train PCA
    again = fit_mfsi(train, shifted, cfg, INFORMATIVE, seed=0)
    assert params["reduce_img"] == again["reduce_img"]


def test_scoring_refuses_a_different_k(cfg: dict[str, Any]) -> None:
    cfg["mfsi"]["reduce"] = PCA
    train, val = _export(300, "train", 0), _export(150, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    params["reduce_txt"] = dict(params["reduce_txt"], k=params["reduce_txt"]["k"] + 1)
    with pytest.raises(ValueError, match="projection rebuilt"):
        score_split(train, val, params, cfg)
