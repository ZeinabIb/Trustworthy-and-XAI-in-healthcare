"""Phase 4 metrics and statistics on toy and synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from mfsi_cxr.eval.calibration import ece_mce, reliability_table
from mfsi_cxr.eval.selective import (
    accuracy_at_coverage,
    aurc,
    auroc_trust,
    correctness,
    cumulative_errors,
    risk_coverage,
)
from mfsi_cxr.eval.stats import (
    benjamini_hochberg,
    bootstrap,
    cross_modal_claim_holds,
    delong_cov,
    delong_test,
    patient_groups,
    percentile_ci,
)

RNG = np.random.default_rng(0)


def test_aurc_hand_computed() -> None:
    trust = np.array([4.0, 3.0, 2.0, 1.0])
    correct = np.array([1, 1, 0, 0])
    cov, risk = risk_coverage(trust, correct)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0]) and np.allclose(risk, [0, 0, 1 / 3, 1 / 2])
    assert aurc(trust, correct) == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)
    assert accuracy_at_coverage(trust, correct, [0.5, 0.75]) == pytest.approx({"0.5": 1.0, "0.75": 2 / 3})


def test_ties_are_broken_in_expectation() -> None:
    trust = np.array([1.0, 1.0, 0.0])  # the two tied cases: one right, one wrong
    correct = np.array([1, 0, 1])
    assert np.allclose(cumulative_errors(trust, correct), [0, 0.5, 1.0, 1.0])
    perm = np.array([1, 0, 2])
    assert aurc(trust, correct) == pytest.approx(aurc(trust[perm], correct[perm]))
    assert aurc(np.zeros(4), np.array([1, 0, 1, 0])) == pytest.approx(0.5)  # constant score = error rate


def test_oracle_minimises_and_anti_oracle_maximises_aurc() -> None:
    correct = RNG.random(500) < 0.7
    noise = RNG.normal(size=500)
    oracle, anti = correct + 0.01 * noise, -(correct + 0.01 * noise)
    assert aurc(oracle, correct) < aurc(noise, correct) < aurc(anti, correct)
    assert auroc_trust(oracle, correct) == 1.0 and auroc_trust(anti, correct) == 0.0


def test_correctness_blank_policies() -> None:
    y = np.array([1.0, 0.0, np.nan, -1.0, np.nan])
    y_hat = np.array([1, 1, 0, 1, 1])
    c, used = correctness(y, y_hat, "negative")
    assert used.tolist() == [True, True, True, False, True]
    assert c.tolist() == [True, False, True, False, False]
    c, used = correctness(y, y_hat, "exclude")
    assert used.tolist() == [True, True, False, False, False]
    with pytest.raises(ValueError):
        correctness(y, y_hat, "positive")


def test_equal_mass_calibration() -> None:
    prob = RNG.random(3000)
    y = (RNG.random(3000) < prob).astype(float)  # calibrated by construction
    ece, mce = ece_mce(prob, y, 15)
    assert ece < 0.03 and mce < 0.08
    t = reliability_table(prob, y, 15)
    assert len(t) == 15 and t["n"].max() - t["n"].min() <= 1
    assert ece_mce(np.full(100, 0.9), np.zeros(100), 15)[0] == pytest.approx(0.9)


def test_bootstrap_draws_whole_patients() -> None:
    patient = np.array(["a", "a", "b", "c", "c", "c"])
    groups = patient_groups(patient)
    assert [g.tolist() for g in groups] == [[0, 1], [2], [3, 4, 5]]
    seen = []
    bootstrap(patient, lambda rows: seen.append(rows) or 0.0, n_boot=50, seed=1)
    for rows in seen:
        for g in groups:  # every patient is in or out as a block (possibly several times)
            counts = np.array([np.sum(rows == r) for r in g])
            assert len(set(counts)) == 1
    v = bootstrap(patient, lambda rows: float(len(rows)), n_boot=20, seed=1)
    assert np.array_equal(v, bootstrap(patient, lambda rows: float(len(rows)), n_boot=20, seed=1))  # seeded


def test_percentile_ci_drops_nan() -> None:
    assert percentile_ci(np.r_[np.arange(101.0), np.nan], 0.1) == pytest.approx((5.0, 95.0))
    assert np.isnan(percentile_ci(np.array([np.nan]), 0.05)[0])


def _delong_bruteforce(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Original DeLong (1988) with O(mn) placement values, as an independent reference."""
    x, z = s[y == 1], s[y == 0]
    psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
    v10, v01 = psi.mean(1), psi.mean(0)
    return float(psi.mean()), float(v10.var(ddof=1) / len(x) + v01.var(ddof=1) / len(z))


def test_delong_matches_reference_and_sklearn() -> None:
    y = (RNG.random(300) < 0.4).astype(int)
    s1 = y + RNG.normal(size=300)
    s2 = np.round(y * 0.5 + RNG.normal(size=300), 1)  # with ties
    aucs, cov = delong_cov(y, np.stack([s1, s2]))
    for k, s in enumerate((s1, s2)):
        auc_ref, var_ref = _delong_bruteforce(y, s)
        assert aucs[k] == pytest.approx(roc_auc_score(y, s)) == pytest.approx(auc_ref)
        assert cov[k, k] == pytest.approx(var_ref)
    assert delong_test(y, s1, s1)["p"] == 1.0


def test_delong_type_one_error_near_nominal() -> None:
    rng = np.random.default_rng(3)
    rejections = 0
    for _ in range(400):
        y = (rng.random(200) < 0.5).astype(int)
        base = y + rng.normal(size=200)
        s1, s2 = base + rng.normal(size=200), base + rng.normal(size=200)  # equal true AUC
        rejections += delong_test(y, s1, s2)["p"] < 0.05
    assert 0.02 < rejections / 400 < 0.09


def test_benjamini_hochberg_matches_statsmodels() -> None:
    p = np.array([0.001, 0.02, 0.03, 0.2, 0.5, 0.04, np.nan])
    q, rej = benjamini_hochberg(p, 0.05)
    ref_rej, ref_q, _, _ = multipletests(p[:-1], alpha=0.05, method="fdr_bh")
    assert np.allclose(q[:-1], ref_q) and np.array_equal(rej[:-1], ref_rej)
    assert np.isnan(q[-1]) and not rej[-1]


def test_cross_modal_decision_rule() -> None:
    assert cross_modal_claim_holds(0.001)
    assert not cross_modal_claim_holds(0.0) and not cross_modal_claim_holds(-0.01)
    assert not cross_modal_claim_holds(float("nan"))
