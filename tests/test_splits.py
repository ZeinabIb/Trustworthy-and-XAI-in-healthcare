from __future__ import annotations

import itertools

import pytest

from mfsi_cxr.constants import HOLDOUT, SPLITS
from mfsi_cxr.data.splits import make_splits

FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
PATIENTS = [f"patient{i:05d}" for i in range(200)]
RADIOLOGIST = {"patient00007", "patient00100"}


def _split(seed: int = 0):
    return make_splits(PATIENTS, RADIOLOGIST, FRACTIONS, seed, "holdout")


def test_no_patient_overlap() -> None:
    s = _split()
    groups = {name: set(s.loc[s.split == name, "patient_id"]) for name in (*SPLITS, HOLDOUT)}
    for a, b in itertools.combinations(groups, 2):
        assert not groups[a] & groups[b], (a, b)
    assert set().union(*groups.values()) == set(PATIENTS)
    assert s["patient_id"].is_unique


def test_deterministic_with_seed() -> None:
    assert _split(3).equals(_split(3))
    assert not _split(3).equals(_split(4))


def test_fractions() -> None:
    counts = _split()["split"].value_counts()
    n = len(PATIENTS) - len(RADIOLOGIST)
    assert counts["train"] == round(0.7 * n)
    assert counts["val"] == round(0.1 * n)


def test_radiologist_subset_held_out() -> None:
    s = _split().set_index("patient_id")
    for p in RADIOLOGIST:
        assert s.loc[p, "split"] == HOLDOUT and s.loc[p, "radiologist_subset"]
    assert not s.loc[s.split != HOLDOUT, "radiologist_subset"].any()


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        make_splits(PATIENTS, set(), {"train": 0.5, "val": 0.1, "test": 0.1}, 0, "holdout")
    with pytest.raises(ValueError):
        make_splits(PATIENTS, set(), FRACTIONS, 0, "train_on_it")
