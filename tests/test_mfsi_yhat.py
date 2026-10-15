"""y_hat v2: Platt rule, coherent Youden sensitivity, out-of-fold and nested calibration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from mfsi_cxr.mfsi import yhat

EPS = 1e-6
RNG = np.random.default_rng(0)


def test_platt_recovers_known_map() -> None:
    p = RNG.uniform(0.01, 0.99, 20000)
    label = RNG.random(20000) < expit(0.6 * logit(p) - 0.8)
    prm = yhat.fit_platt(p, label, EPS)
    assert prm["a"] == pytest.approx(0.6, abs=0.05) and prm["b"] == pytest.approx(-0.8, abs=0.05)
    q, y_hat, msp = yhat.apply_rule("platt", prm, p, EPS)
    assert np.array_equal(y_hat, q >= 0.5) and np.allclose(msp, np.maximum(q, 1 - q))


def test_platt_never_reverses_and_stays_finite_when_sparse() -> None:
    p = RNG.uniform(0.01, 0.99, 500)
    anti = p < 0.3                                     # label falls as p rises
    prm = yhat.fit_platt(p, anti, EPS)
    assert prm["a"] == 0.0                             # bounded at 0: constant p_cal, ranking never flipped
    q, _, _ = yhat.apply_rule("platt", prm, p, EPS)
    assert np.ptp(q) == 0
    for n_pos in (0, 1):
        lab = np.zeros(500, bool)
        lab[:n_pos] = True
        prm = yhat.fit_platt(p, lab, EPS)
        assert np.isfinite([prm["a"], prm["b"]]).all() and prm["sparse"]
    assert not yhat.fit_platt(p, p > 0.5, EPS)["sparse"]


def test_youden_coherent_formula() -> None:
    p = np.array([0.01, 0.1, 0.2, 0.5, 0.9])
    prm = {"t": 0.2}
    q, y_hat, msp = yhat.apply_rule("youden_coherent", prm, p, EPS)
    assert q[2] == pytest.approx(0.5) and np.array_equal(y_hat, p >= 0.2)
    assert np.allclose(msp, expit(np.abs(logit(p) - logit(0.2))))
    assert np.all(np.diff(q) > 0)                      # monotone in p
    _, _, msp_old = yhat.apply_rule("youden", prm, p, EPS)
    assert np.allclose(msp_old, np.maximum(p, 1 - p))  # old rule kept for reproduction only
    with pytest.raises(ValueError):
        yhat.fit_rule("accuracy", p, p > 0.5, EPS)


def test_folds_keep_patients_together() -> None:
    pid = np.repeat(np.arange(40), 3)
    folds = yhat.make_folds(pid, 5, seed=1)
    assert all(len(set(folds[pid == i])) == 1 for i in range(40))
    assert np.bincount(folds).tolist() == [24] * 5
    assert np.array_equal(folds, yhat.make_folds(pid, 5, seed=1))


def _setup(n: int = 300):
    probs = RNG.uniform(0.01, 0.99, (n, 2))
    label = RNG.random((n, 2)) < probs
    used = np.ones((n, 2), bool)
    folds = yhat.make_folds(np.arange(n), 5, seed=0)
    return probs, label, used, folds


def test_out_of_fold_ignores_own_fold_labels() -> None:
    probs, label, used, folds = _setup()
    q, _, _, sparse = yhat.out_of_fold("platt", probs, label, used, folds, EPS)
    assert np.isfinite(q).all() and sparse.shape == (5, 2)
    flipped = label.copy()
    flipped[folds == 2] = ~flipped[folds == 2]
    q2, _, _, _ = yhat.out_of_fold("platt", probs, flipped, used, folds, EPS)
    assert np.allclose(q[folds == 2], q2[folds == 2])        # fold 2 rows never see fold 2 labels
    assert not np.allclose(q[folds != 2], q2[folds != 2])


def test_nested_excludes_the_outer_fold() -> None:
    probs, label, used, folds = _setup()
    q, _, _, _ = yhat.out_of_fold("platt", probs, label, used, folds, EPS, exclude=3)
    assert np.isnan(q[folds == 3]).all() and np.isfinite(q[folds != 3]).all()
    flipped = label.copy()
    flipped[folds == 3] = ~flipped[folds == 3]
    q2, _, _, _ = yhat.out_of_fold("platt", probs, flipped, used, folds, EPS, exclude=3)
    assert np.allclose(q[folds != 3], q2[folds != 3])        # training rows never see outer-fold labels


def test_sparse_folds_are_flagged() -> None:
    probs, label, used, folds = _setup()
    label[:, 1] = False
    label[folds == 0, 1] = True                               # all positives in fold 0
    _, _, _, sparse = yhat.out_of_fold("youden_coherent", probs, label, used, folds, EPS)
    assert sparse[1:, 1].sum() == 0 and sparse[0, 1]          # only fold 0's training part lacks positives
