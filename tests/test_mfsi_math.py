"""MFSI math (Phase 3): distance, shrinkage, profile and margin forms, agreement, worked examples."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis

from mfsi_cxr.constants import FINDINGS, PARENT
from mfsi_cxr.mfsi.clouds import cloud_masks
from mfsi_cxr.mfsi.covariance import (
    fit_covariances,
    is_pd,
    make_pd,
    pooled_covariance,
    raw_covariance,
    shrink,
    topological_order,
)
from mfsi_cxr.mfsi.distance import Gaussian
from mfsi_cxr.mfsi.profile import margin, profile_pos, profile_pos_lse
from mfsi_cxr.mfsi.reliability import agreement, agreement_cos, agreement_js, reliability

RNG = np.random.default_rng(0)


def _spd(d: int) -> np.ndarray:
    a = RNG.normal(size=(d, d))
    return a @ a.T + d * np.eye(d)


def test_mahalanobis_matches_scipy() -> None:
    d = 6
    cov, mu = _spd(d), RNG.normal(size=d)
    z = RNG.normal(size=(20, d))
    ours = Gaussian(mu, cov).distance(z)
    ref = [mahalanobis(x, mu, np.linalg.inv(cov)) for x in z]
    assert np.allclose(ours, ref, atol=1e-10)


def test_distance_at_prototype_is_zero_not_nan() -> None:
    mu = RNG.normal(size=4)
    assert Gaussian(mu, _spd(4)).distance(mu[None, :])[0] == pytest.approx(0.0, abs=1e-12)


def test_shrinkage_endpoints_and_pd_blend() -> None:
    d = 8
    z = RNG.normal(size=(3, d))  # n < d: raw covariance is singular
    raw, target = raw_covariance(z), _spd(d)
    assert not is_pd(raw)
    assert np.allclose(shrink(raw, target, 1.0), raw)
    assert np.allclose(shrink(raw, target, 0.0), target)
    for alpha in (0.1, 0.5, 0.9):
        assert is_pd(shrink(raw, target, alpha))


def test_make_pd_adds_ridge_only_when_needed() -> None:
    good = _spd(5)
    same, lam = make_pd(good, 1e-3, "t")
    assert lam == 0 and np.array_equal(same, good)
    singular = raw_covariance(RNG.normal(size=(2, 5)))
    fixed, lam = make_pd(singular, 1e-3, "t")
    assert lam > 0 and is_pd(fixed)


def test_topological_order_puts_parents_first() -> None:
    order = topological_order()
    assert sorted(order) == sorted(FINDINGS)
    for child, parent in PARENT.items():
        assert order.index(parent) < order.index(child)
    assert order.index("Lung Opacity") < order.index("Consolidation") < order.index("Pneumonia")


def _labelled(n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    z = RNG.normal(size=(n, d))
    y = RNG.choice([1.0, 0.0, np.nan, -1.0], p=[0.15, 0.4, 0.4, 0.05], size=(n, len(FINDINGS)))
    return z, y


def test_pneumonia_target_is_the_shrunk_consolidation_matrix() -> None:
    z, y = _labelled(200, 10)
    pos, neg = cloud_masks(y, "ignore")
    covs = fit_covariances(z, pos, neg, {f: 0.0 for f in FINDINGS}, 1e-3, 1e10, "img")
    assert covs.info["Pneumonia"]["target"] == "parent:Consolidation"
    assert covs.info["Consolidation"]["target"] == "parent:Lung Opacity"
    assert np.allclose(covs.pos["Pneumonia"], covs.pos["Consolidation"])  # alpha = 0: the target itself
    assert covs.info["Pleural Effusion"]["target"] == "pooled"


def test_rare_class_does_not_crash_and_logs_condition() -> None:
    z, y = _labelled(60, 40)  # few positives per finding, n_pos << d
    pos, neg = cloud_masks(y, "ignore")
    covs = fit_covariances(z, pos, neg, {f: 0.5 for f in FINDINGS}, 1e-3, 1e10, "img")
    for f in FINDINGS:
        assert is_pd(covs.pos[f]) and is_pd(covs.neg[f])
        assert np.isfinite(covs.info[f]["cond_pos"]) and covs.info[f]["n_pos"] < 40


def test_clouds_exclude_uncertain_and_count_blank_negative() -> None:
    y = np.array([[1.0], [0.0], [np.nan], [-1.0]])
    pos, neg = cloud_masks(y, "ignore")
    assert pos[:, 0].tolist() == [True, False, False, False]
    assert neg[:, 0].tolist() == [False, True, True, False]
    pos, _ = cloud_masks(y, "ones")
    assert pos[:, 0].tolist() == [True, False, False, True]


def test_pooled_covariance() -> None:
    a, b = RNG.normal(size=(30, 3)), RNG.normal(size=(20, 3)) + 5
    expected = ((a - a.mean(0)).T @ (a - a.mean(0)) + (b - b.mean(0)).T @ (b - b.mean(0))) / 50
    assert np.allclose(pooled_covariance([a, b]), expected)


def test_profile_and_margin_forms_agree() -> None:
    d_pos, d_neg = RNG.uniform(0, 10, 50), RNG.uniform(0, 10, 50)
    for tau, b in ((0.3, 0.0), (1.0, 0.0), (4.0, 0.0), (1.0, 2.5), (4.0, -1.0)):
        p = profile_pos(d_pos, d_neg, tau, b)
        assert np.allclose(p, profile_pos_lse(d_pos, d_neg, tau, b))
        r = margin(d_pos, d_neg, tau, b)
        assert np.allclose(r, np.abs(2 * p - 1)) and np.allclose(r, np.abs(p - (1 - p)))
    assert np.allclose(profile_pos(d_pos, d_neg, 1.0, 3.0), profile_pos(d_pos, d_neg - 3.0, 1.0))


def test_separated_vs_overlapping_clouds() -> None:
    d = 5
    eye = np.eye(d)
    g_pos, g_neg = Gaussian(np.full(d, 6.0), eye), Gaussian(np.zeros(d), eye)
    far = RNG.normal(size=(100, d)) + 6.0
    assert margin(g_pos.distance(far), g_neg.distance(far), 1.0).mean() > 0.99
    g_pos2 = Gaussian(np.full(d, 0.05), eye)
    near = RNG.normal(size=(100, d))
    assert margin(g_pos2.distance(near), g_neg.distance(near), 1.0).mean() < 0.1


def test_agreement_same_and_opposite_direction() -> None:
    assert agreement_cos(np.array([0.9]), np.array([0.9]))[0] == pytest.approx(1.0)
    assert agreement_cos(np.array([0.99]), np.array([0.01]))[0] < 0.05
    js = agreement_js(RNG.uniform(size=200), RNG.uniform(size=200))
    assert np.all((js >= 0) & (js <= 1))
    assert np.all(agreement(np.array([0.2]), np.array([0.9]), "none") == 1)


def test_worked_example_agreement() -> None:
    """Proposal: p_img = (0.9, 0.1), p_txt = (0.5, 0.5) -> cosine ~0.781, 1 - JS ~0.853 (base-2 logs)."""
    assert agreement_cos(np.array([0.9]), np.array([0.5]))[0] == pytest.approx(0.781, abs=5e-4)
    assert agreement_js(np.array([0.9]), np.array([0.5]))[0] == pytest.approx(0.853, abs=5e-4)


def test_worked_example_reliability() -> None:
    """Proposal: d+ = 1.2, d- = 2.5, tau = 1 -> p = (0.79, 0.21), r = 0.57 (the plan's 0.58 is 2*0.79-1 after
    rounding p); with p_txt = (0.75, 0.25): A ~0.998, R ~0.50."""
    p = profile_pos(np.array([1.2]), np.array([2.5]), 1.0)[0]
    r = margin(np.array([1.2]), np.array([2.5]), 1.0)[0]
    assert p == pytest.approx(0.786, abs=1e-3) and r == pytest.approx(0.572, abs=1e-3)
    a = agreement_cos(np.array([p]), np.array([0.75]))[0]
    r_txt = abs(2 * 0.75 - 1)
    assert a == pytest.approx(0.998, abs=1e-3)
    big_r = reliability(np.array([a]), np.array([r]), np.array([r_txt]), "min")[0]
    assert big_r == pytest.approx(0.50, abs=5e-3)
