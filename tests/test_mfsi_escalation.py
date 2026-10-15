"""Coherence flags and escalation (Phase 3, step 12)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mfsi_cxr.constants import FINDINGS, NO_FINDING
from mfsi_cxr.mfsi.escalation import escalate, finding_thresholds, stratum_thresholds, yhat_threshold
from mfsi_cxr.mfsi.hierarchy import coherence_flags

CHILD, PARENT_ = FINDINGS.index("Cardiomegaly"), FINDINGS.index("Enlarged Cardiomediastinum")


@pytest.mark.parametrize(("child_pos", "parent_neg", "parent_reliable"),
                         list(itertools.product([False, True], repeat=3)))
def test_flag_truth_table(child_pos: bool, parent_neg: bool, parent_reliable: bool) -> None:
    """flag = [y_hat child = 1] and [y_hat parent = 0] and [R_parent >= theta] (all 8 combinations)."""
    y_hat = np.zeros((1, len(FINDINGS)), dtype=bool)
    y_hat[0, CHILD], y_hat[0, PARENT_] = child_pos, not parent_neg
    r = np.full((1, len(FINDINGS)), 0.1)
    r[0, PARENT_] = 0.9 if parent_reliable else 0.1
    flags = coherence_flags(y_hat, r, np.full(len(FINDINGS), 0.5))
    assert flags[0, CHILD] == (child_pos and parent_neg and parent_reliable)


@pytest.mark.parametrize(("low_r", "flag"), list(itertools.product([False, True], repeat=2)))
def test_escalate_is_low_r_or_flag(low_r: bool, flag: bool) -> None:
    r = np.array([[0.1 if low_r else 0.9]])
    assert escalate(r, np.array([0.5]), np.array([[flag]]))[0, 0] == (low_r or flag)


def test_grandparent_counts() -> None:
    """Pneumonia and Consolidation called while Lung Opacity (grandparent / parent) is confidently denied:
    both are flagged; with Lung Opacity also called, neither is."""
    y_hat = np.zeros((1, len(FINDINGS)), dtype=bool)
    y_hat[0, FINDINGS.index("Pneumonia")] = True
    y_hat[0, FINDINGS.index("Consolidation")] = True
    r = np.full((1, len(FINDINGS)), 0.9)
    flags = coherence_flags(y_hat, r, np.full(len(FINDINGS), 0.5))
    assert flags[0, FINDINGS.index("Pneumonia")] and flags[0, FINDINGS.index("Consolidation")]
    y_hat[0, FINDINGS.index("Lung Opacity")] = True
    flags = coherence_flags(y_hat, r, np.full(len(FINDINGS), 0.5))
    assert not flags[0, FINDINGS.index("Pneumonia")] and not flags[0, FINDINGS.index("Consolidation")]


def test_no_finding_conflict() -> None:
    y_hat = np.zeros((1, len(FINDINGS)), dtype=bool)
    nf, eff = FINDINGS.index(NO_FINDING), FINDINGS.index("Pleural Effusion")
    sd = FINDINGS.index("Support Devices")
    y_hat[0, [nf, eff, sd]] = True
    r = np.full((1, len(FINDINGS)), 0.9)
    flags = coherence_flags(y_hat, r, np.full(len(FINDINGS), 0.5))
    assert flags[0, eff] and flags[0, nf]
    assert not flags[0, sd]  # Support Devices is not a pathology


def test_stratum_thresholds_escalate_the_stated_share() -> None:
    rng = np.random.default_rng(0)
    r = rng.uniform(size=(1000, 3))
    strata = np.array(["one_witness", "two_witness", "two_witness"])
    thr = stratum_thresholds(r, strata, [0.1, 0.25])
    for s, cols in (("one_witness", [0]), ("two_witness", [1, 2])):
        for f in (0.1, 0.25):
            assert (r[:, cols] < thr[s][str(f)]).mean() == pytest.approx(f, abs=0.01)


def test_finding_thresholds_escalate_the_stated_share_per_finding() -> None:
    rng = np.random.default_rng(1)
    r = np.c_[rng.uniform(size=1000), rng.uniform(size=1000) * 0.1, 0.9 + rng.uniform(size=1000) * 0.1]
    thr = finding_thresholds(r, ["a", "b", "c"], [0.1, 0.25])
    for j, f in enumerate("abc"):
        for q in (0.1, 0.25):
            assert (r[:, j] < thr[f][str(q)]).mean() == pytest.approx(q, abs=0.01)
    pooled = stratum_thresholds(r, np.array(["one_witness"] * 3), [0.25])["one_witness"]["0.25"]
    assert (r[:, 2] < pooled).mean() == 0.0 and (r[:, 1] < pooled).mean() > 0.5  # pooled: escalates by scale


def test_youden_threshold_separates() -> None:
    prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1])
    t = yhat_threshold(prob, y, "youden")
    assert 0.3 < t <= 0.7 and np.array_equal(prob >= t, y.astype(bool))
    assert yhat_threshold(prob, np.zeros(6), "youden") == 0.5
    assert 0.3 < yhat_threshold(prob, y, "f1") <= 0.7
