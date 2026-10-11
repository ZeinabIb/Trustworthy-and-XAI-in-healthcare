"""MFSI fit / tune / score end to end on synthetic exports; leakage guard; output contract."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.mfsi.fit import SplitError, fit_mfsi, score_split

COLUMNS = ["patient_id", "study_id", "finding", "y", "y_hat", "prob", "p_img_pos", "p_txt_pos", "d_img_pos",
           "d_img_neg", "d_txt_pos", "d_txt_neg", "r_img", "r_txt", "A_cos", "A_js", "R", "R_mean", "stratum",
           "flag", "escalate", "n_studies", "n_images", "text_len"]  # plan 3 "Outputs", spelled out


def _export(n: int, split: str, seed: int, d: int = 12) -> dict[str, np.ndarray]:
    """Embeddings that carry the labels (+ noise) in both modalities; a few uncertain and blank labels."""
    rng = np.random.default_rng(seed)
    proj = np.random.default_rng(99).normal(size=(len(FINDINGS), d))
    y = rng.choice([1.0, 0.0, np.nan, -1.0], p=[0.3, 0.3, 0.35, 0.05], size=(n, len(FINDINGS)))
    signal = (y == 1).astype(float) @ proj
    probs = 1 / (1 + np.exp(-(2 * (y == 1) - 1 + rng.normal(scale=1.0, size=y.shape))))
    return {
        "z_img": signal + rng.normal(scale=1.0, size=(n, d)),
        "z_txt": signal + rng.normal(scale=1.5, size=(n, d)),
        "y": y, "probs": probs, "patient_id": np.array([f"p{i // 2}" for i in range(n)]),
        "study_id": np.array([f"p{i // 2}/s{i % 2}" for i in range(n)]), "n_studies": np.ones(n, int),
        "n_images": np.ones(n, int), "text_len": np.full(n, 5), "_split": np.array(split),
    }


@pytest.fixture
def cfg(smoke_cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(smoke_cfg)


INFORMATIVE = {f: f in ("Pleural Effusion", "Edema", "Cardiomegaly") for f in FINDINGS}


def test_fit_refuses_wrong_splits(cfg: dict[str, Any]) -> None:
    train, val = _export(300, "train", 0), _export(150, "val", 1)
    with pytest.raises(SplitError):
        fit_mfsi(val, val, cfg, INFORMATIVE, seed=0)
    with pytest.raises(SplitError):
        fit_mfsi(train, _export(150, "test", 2), cfg, INFORMATIVE, seed=0)


def test_fit_and_score(cfg: dict[str, Any]) -> None:
    train, val = _export(400, "train", 0), _export(200, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    for name in ("img", "txt"):
        assert params[f"tau_{name}"]["tau"] > 0 and set(params[f"alpha_{name}"]) == set(FINDINGS)
    assert params["strata"]["Pleural Effusion"] == "two_witness"
    assert params["strata"]["Fracture"] == "one_witness"
    table = score_split(train, val, params, cfg)
    assert list(table.columns) == COLUMNS and len(table) == 200 * len(FINDINGS)
    for col in ("R", "R_mean", "r_img", "r_txt", "A_cos", "A_js"):
        assert table[col].between(0, 1).all(), col
    one = table[table["stratum"] == "one_witness"]
    assert np.allclose(one["R"], one["r_img"])  # R_one = r_img, its own scale
    op = cfg["mfsi"]["operating_fraction"]
    for s, rows in table.groupby("stratum"):
        theta = params["theta_R"][s][str(op)]
        assert (rows["R"] < theta).mean() == pytest.approx(op, abs=0.02)
    assert (table["escalate"] >= (table["R"] < 0)).all()


def test_scoring_is_deterministic_and_ignores_labels(cfg: dict[str, Any]) -> None:
    train, val = _export(300, "train", 0), _export(120, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    test = _export(100, "test", 3)
    first = score_split(train, test, params, cfg)
    blind = dict(test, y=np.full_like(test["y"], np.nan))  # labels removed: scores must not change
    second = score_split(train, blind, params, cfg)
    for col in ("R", "p_img_pos", "d_txt_neg", "escalate", "flag"):
        assert np.array_equal(first[col].to_numpy(), second[col].to_numpy()), col


def test_informative_embeddings_give_confident_profiles(cfg: dict[str, Any]) -> None:
    train, val = _export(500, "train", 0), _export(200, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    table = score_split(train, val, params, cfg)
    pos, neg = table[table["y"] == 1], table[table["y"] == 0]
    assert pos["p_img_pos"].mean() > neg["p_img_pos"].mean() + 0.2


def test_ablation_switches(cfg: dict[str, Any]) -> None:
    train, val = _export(300, "train", 0), _export(120, "val", 1)
    cfg["mfsi"]["agreement"] = "none"
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    table = score_split(train, val, params, cfg)
    two = table[table["stratum"] == "two_witness"]
    assert np.allclose(two["R"], np.minimum(two["r_img"], two["r_txt"]))
    cfg["mfsi"]["uncertain_policy"] = "ones"
    assert fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)["cov_img"]["Edema"]["n_pos"] > \
        params["cov_img"]["Edema"]["n_pos"]
