"""Pre-registered added-value analysis (a) and (b) on synthetic arrays."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval import added_value as av

N, L = 600, len(FINDINGS)


@pytest.fixture
def cfg(smoke_cfg: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(smoke_cfg)
    c["eval"]["bootstrap_B"] = 40
    return c


def _arrays(seed: int, mfsi_signal: float) -> tuple[av.SeedArrays, np.ndarray]:
    """correct depends on MSP and (optionally) on MFSI independently of MSP."""
    rng = np.random.default_rng(seed)
    msp = rng.uniform(0.5, 1.0, (N, L))
    mfsi = rng.uniform(0, 1, (N, L))
    logit = 4 * (msp - 0.75) + mfsi_signal * (mfsi - 0.5)
    correct = rng.random((N, L)) < 1 / (1 + np.exp(-logit))
    used = np.ones((N, L), bool)
    return av.SeedArrays(msp, mfsi, correct, used), np.array([f"p{i // 2}" for i in range(N)])


def test_eligibility_rule_a(cfg: dict[str, Any]) -> None:
    s, _ = _arrays(0, 0.0)
    s.correct[:, 0] = True
    s.correct[:9, 0] = False                      # 9 wrong -> not eligible
    s.correct[:, 1] = True
    s.correct[:10, 1] = False                     # 10 wrong -> eligible
    e = av.eligible_a(s, cfg["eval"]["added_value"])
    assert not e[0] and e[1]


def test_combined_gains_when_mfsi_adds_information(cfg: dict[str, Any]) -> None:
    a = cfg["eval"]["added_value"]
    for signal, expect_gain in ((6.0, True), (0.0, False)):
        val, pid = _arrays(1, signal)
        test, pid_t = _arrays(2, signal)
        elig = av.eligible_a(val, a)
        model = av.fit_combined(val, a, elig)
        comb = av.score_combined(model, test)
        per, macro = av.evaluate_a([comb], [test], pid_t, elig, cfg)
        coefs = av.coefficient_table([model], [0])
        if expect_gain:
            assert macro["added_value"].iloc[0] and (coefs["coef_mfsi"] > 0).all()
        else:
            assert abs(macro["delta"].iloc[0]) < 0.01 and not macro["added_value"].iloc[0]


def test_crossval_is_out_of_fold_and_flags_negative_coefficients(cfg: dict[str, Any]) -> None:
    a = cfg["eval"]["added_value"]
    val, pid = _arrays(3, -6.0)                   # MFSI anti-informative: coefficient must be negative
    from mfsi_cxr.mfsi.yhat import make_folds

    folds = make_folds(pid, a["cv_folds"], seed=0)
    calls = []
    oof, ok = av.crossval_combined(val, folds, a, av.eligible_a(val, a), lambda k: calls.append(k) or val)
    assert ok.all() and np.isfinite(oof).all() and sorted(calls) == list(range(a["cv_folds"]))
    coefs = av.coefficient_table([av.fit_combined(val, a, ok)], [0])
    assert coefs["mfsi_coef_negative"].all()


def test_quintile_edges_merge_and_right_closed_bins() -> None:
    msp = np.r_[np.full(80, 1.0), np.linspace(0.5, 0.9, 20)]  # 80% ties at 1.0
    e = av.quintile_edges(msp, [0.2, 0.4, 0.6, 0.8])
    assert len(e) < 4                                         # coinciding edges merged
    assert av.assign(np.array([e[0], e[0] + 1e-9]), e).tolist() == [0, 1]


def test_stratified_auroc_is_pair_weighted() -> None:
    rng = np.random.default_rng(4)
    trust, correct = rng.random(400), rng.random(400) < 0.6
    q = np.zeros(400, int)
    v, dropped = av._stratified(trust, correct, q, np.array([0]))
    assert v == pytest.approx(roc_auc_score(correct, trust)) and dropped == 0
    q = (np.arange(400) >= 100).astype(int)
    a0 = roc_auc_score(correct[q == 0], trust[q == 0])
    a1 = roc_auc_score(correct[q == 1], trust[q == 1])
    w0 = correct[q == 0].sum() * (~correct[q == 0]).sum()
    w1 = correct[q == 1].sum() * (~correct[q == 1]).sum()
    v, _ = av._stratified(trust, correct, q, np.array([0, 1]))
    assert v == pytest.approx((w0 * a0 + w1 * a1) / (w0 + w1))
    correct[q == 1] = True                                    # stratum 1 lacks a class -> dropped, counted
    v, dropped = av._stratified(trust, correct, q, np.array([0, 1]))
    assert dropped == 1 and v == pytest.approx(roc_auc_score(correct[q == 0], trust[q == 0]))


def test_b_end_to_end_and_fixed_eligibility(cfg: dict[str, Any]) -> None:
    a = cfg["eval"]["added_value"]
    val, pid = _arrays(5, 6.0)
    plan = av.plan_strata([val], [val], a)
    assert plan.finding_ok.all()
    per, macro = av.evaluate_b([val], pid, plan, cfg)
    assert macro["adds_information"].iloc[0] and (per["drop_share"] >= 0).all()
    val0, pid0 = _arrays(6, 0.0)
    _, macro0 = av.evaluate_b([val0], pid0, av.plan_strata([val0], [val0], a), cfg)
    assert abs(macro0["stratified_auroc"].iloc[0] - 0.5) < 0.05
