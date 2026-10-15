"""Phase 4 sensitivity variants of Phase 3 (covariance, distance, tau = 1, one-witness): each runs, defaults
are unchanged, bad values are refused."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from tests.test_mfsi_fit import INFORMATIVE, _export

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.mfsi.covariance import fit_covariances, is_pd
from mfsi_cxr.mfsi.fit import fit_mfsi, score_split


@pytest.fixture
def cfg(smoke_cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(smoke_cfg)


def _run(cfg: dict[str, Any], **mfsi: Any):
    c = copy.deepcopy(cfg)
    c["mfsi"].update(mfsi)
    train, val = _export(300, "train", 0), _export(150, "val", 1)
    params = fit_mfsi(train, val, c, INFORMATIVE, seed=0)
    return params, score_split(train, val, params, c)


@pytest.mark.parametrize("mode", ["parent", "pooled_target", "ledoit_wolf", "tied", "sample_ridge"])
def test_covariance_modes_are_pd(mode: str) -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 6))
    y = rng.random((200, len(FINDINGS))) < 0.3
    covs = fit_covariances(z, y, ~y, {f: 0.5 for f in FINDINGS}, 1e-3, 1e10, "t", mode=mode)
    assert all(is_pd(covs.pos[f]) and is_pd(covs.neg[f]) for f in FINDINGS)
    if mode == "tied":
        assert np.allclose(covs.pos["Edema"], covs.neg["Fracture"])
    if mode == "pooled_target":
        assert all(covs.info[f]["target"] == "pooled" for f in FINDINGS)
    with pytest.raises(ValueError):
        fit_covariances(z, y, ~y, {}, 1e-3, 1e10, "t", mode="banana")


@pytest.mark.parametrize("override", [{"covariance": "ledoit_wolf"}, {"covariance": "tied"},
                                      {"distance": "euclidean"}, {"distance": "lof"},
                                      {"tau_objective": "none"}, {"one_witness": "naive"}])
def test_variants_score_end_to_end(cfg: dict[str, Any], override: dict[str, Any]) -> None:
    params, table = _run(cfg, **override)
    assert table["R"].between(0, 1).all() and np.isfinite(table["R"]).all()
    if override.get("tau_objective") == "none":
        assert all(r["tau"] == 1.0 and r["b"] == 0.0 for r in params["tau_img"]["per_finding"].values())
    if override.get("one_witness") == "naive":
        assert set(params["strata"].values()) == {"two_witness"}
        one = table[table["finding"] == "Fracture"]  # text-uninformed: still R = r_img
        assert np.allclose(one["R"], one["r_img"])


def test_default_matches_explicit_defaults(cfg: dict[str, Any]) -> None:
    _, a = _run(cfg)
    _, b = _run(cfg, covariance="parent", distance="mahalanobis", one_witness="separate")
    assert np.array_equal(a["R"].to_numpy(), b["R"].to_numpy())


def test_bad_variant_values_are_refused(cfg: dict[str, Any]) -> None:
    for key, value in (("distance", "cosine"), ("one_witness", "ignore"), ("tau_objective", "brier")):
        with pytest.raises(ValueError):
            _run(cfg, **{key: value})
