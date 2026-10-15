"""MFSI fit / tune / score end to end on synthetic exports; leakage guard; output contract."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.mfsi.fit import SplitError, _platt, fit_mfsi, fit_tau, score_split
from mfsi_cxr.mfsi.profile import profile_pos

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
    assert params["theta_scope"] == "finding"
    for f, rows in table.groupby("finding"):
        theta = params["theta_R_finding"][f][str(op)]
        assert (rows["R"] < theta).mean() == pytest.approx(op, abs=0.02), f
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


def test_tau_per_finding_with_global_fallback(cfg: dict[str, Any]) -> None:
    train, val = _export(300, "train", 0), _export(200, "val", 1)
    val["y"][:, FINDINGS.index("Fracture")] = np.where(np.arange(200) < 3, 1.0, 0.0)  # 3 val positives
    cfg["mfsi"]["tau_scope"] = "finding"
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    per = params["tau_img"]["per_finding"]
    assert per["Fracture"]["source"] == "global" and per["Fracture"]["n_pos"] == 3
    assert per["Fracture"]["tau"] == params["tau_img"]["tau"]
    assert per["Fracture"]["b"] == params["tau_img"]["b"]
    assert per["Edema"]["source"] == "finding"
    rows = score_split(train, val, params, cfg).query("finding == 'Edema'")
    expected = profile_pos(rows["d_img_pos"].to_numpy(), rows["d_img_neg"].to_numpy(), per["Edema"]["tau"],
                           per["Edema"]["b"])
    assert np.allclose(rows["p_img_pos"], expected)


def test_tau_global_scope_uses_one_tau(cfg: dict[str, Any]) -> None:
    train, val = _export(300, "train", 0), _export(200, "val", 1)
    cfg["mfsi"]["tau_scope"] = "global"
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    taus = {r["tau"] for r in params["tau_txt"]["per_finding"].values()}
    assert taus == {params["tau_txt"]["tau"]}
    assert all(r["source"] == "global" for r in params["tau_txt"]["per_finding"].values())


def test_tau_scope_is_validated(cfg: dict[str, Any]) -> None:
    cfg["mfsi"]["tau_scope"] = "sample"
    with pytest.raises(ValueError, match="tau_scope"):
        fit_mfsi(_export(200, "train", 0), _export(100, "val", 1), cfg, INFORMATIVE, seed=0)


def _platt_cfg(cfg: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return dict(cfg["mfsi"], tau_objective="logloss", offset=True, **kw)


def test_platt_recovers_known_tau_and_offset(cfg: dict[str, Any]) -> None:
    rng = np.random.default_rng(5)
    gap = rng.normal(0, 4, 20000)
    target = (rng.random(20000) < 1 / (1 + np.exp(-(gap - 1.5) / 2.0))).astype(float)  # tau 2, b 1.5
    r = _platt(gap, target, _platt_cfg(cfg))
    assert r["tau"] == pytest.approx(2.0, rel=0.08) and r["b"] == pytest.approx(1.5, abs=0.2)
    assert r["converged"] and not r["at_grid_boundary"] and r["logloss_after"] < r["logloss_before_tau1"]
    no_off = _platt(gap, target, _platt_cfg(cfg) | {"offset": False})
    assert no_off["b"] == 0.0 and no_off["logloss_after"] > r["logloss_after"]


def test_platt_keeps_tau_positive_and_flags_the_bound(cfg: dict[str, Any]) -> None:
    rng = np.random.default_rng(6)
    gap = rng.normal(0, 1, 2000)
    target = (gap < 0).astype(float)  # anti-aligned: best fit would flip the sign
    r = _platt(gap, target, _platt_cfg(cfg))
    assert r["tau"] > 0 and r["at_grid_boundary"]
    assert r["tau"] == pytest.approx(cfg["mfsi"]["tau_grid"]["max"])


def test_offset_removes_a_constant_shift(cfg: dict[str, Any]) -> None:
    """A finding whose gap is shifted by a constant gets the same profile after calibration."""
    rng = np.random.default_rng(7)
    n = 3000
    y = np.full((n, len(FINDINGS)), np.nan)
    y[:, 0] = rng.random(n) < 0.3
    d_pos = rng.normal(5, 1, (n, len(FINDINGS)))
    d_neg = d_pos + np.where(y == 1, 2.0, -2.0) + rng.normal(0, 2, d_pos.shape)
    c = copy.deepcopy(cfg)
    c["mfsi"].update(tau_objective="logloss", offset=True, tau_scope="finding")
    a = fit_tau(d_pos, d_neg, y, c)["per_finding"][FINDINGS[0]]
    shifted = fit_tau(d_pos, d_neg - 7.0, y, c)["per_finding"][FINDINGS[0]]
    assert shifted["tau"] == pytest.approx(a["tau"], rel=1e-3)
    assert shifted["b"] == pytest.approx(a["b"] - 7, abs=1e-2)


def test_ece_with_offset_is_refused(cfg: dict[str, Any]) -> None:
    cfg["mfsi"].update(tau_objective="ece", offset=True)
    with pytest.raises(ValueError, match="offset"):
        fit_mfsi(_export(200, "train", 0), _export(100, "val", 1), cfg, INFORMATIVE, seed=0)


def test_ece_objective_without_offset_is_the_plan(cfg: dict[str, Any]) -> None:
    cfg["mfsi"].update(tau_objective="ece", offset=False, tau_scope="global")
    params = fit_mfsi(_export(300, "train", 0), _export(150, "val", 1), cfg, INFORMATIVE, seed=0)
    assert params["tau_img"]["b"] == 0.0 and params["tau_img"]["objective"] == "ece"


def test_theta_stratum_scope_is_the_plan(cfg: dict[str, Any]) -> None:
    cfg["mfsi"]["theta_scope"] = "stratum"
    train, val = _export(400, "train", 0), _export(200, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    table = score_split(train, val, params, cfg)
    op = cfg["mfsi"]["operating_fraction"]
    for s, rows in table.groupby("stratum"):
        theta = params["theta_R"][s][str(op)]
        assert (rows["R"] < theta).mean() == pytest.approx(op, abs=0.02)
        assert np.array_equal(rows["escalate"], (rows["R"] < theta) | rows["flag"])


def test_theta_scope_is_validated(cfg: dict[str, Any]) -> None:
    cfg["mfsi"]["theta_scope"] = "global"
    with pytest.raises(ValueError, match="theta_scope"):
        fit_mfsi(_export(200, "train", 0), _export(100, "val", 1), cfg, INFORMATIVE, seed=0)
