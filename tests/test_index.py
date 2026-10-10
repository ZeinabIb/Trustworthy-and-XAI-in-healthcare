from __future__ import annotations

from typing import Any

import pytest

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.data.index import INDEX_COLUMNS, parse_study, sample_patients


def test_parse_study() -> None:
    assert parse_study("train/patient00001/study3/view1_frontal.png") == ("patient00001/study3", 3)
    assert parse_study("train\\patient00001\\study12\\view2_lateral.jpg") == ("patient00001/study12", 12)
    with pytest.raises(ValueError):
        parse_study("patient00001/view1.png")


def test_index_contract(built: dict[str, Any]) -> None:
    index = built["index"]
    assert list(index.columns) == INDEX_COLUMNS
    assert not index["image_path"].str.contains(":").any(), "paths must be relative to data.root"
    assert (built["root"] / index["image_path"].iloc[0]).exists()


def _sample(built: dict[str, Any], n: int | None, seed: int = 0) -> list[str]:
    return sample_patients(built["images"], built["labels"], n, seed, min_pos=2)


def test_sample_size_and_determinism(built: dict[str, Any]) -> None:
    a = _sample(built, 12)
    assert len(a) == 12 and len(set(a)) == 12
    assert a == _sample(built, 12)
    assert a != _sample(built, 12, seed=1)


def test_null_means_all_patients(built: dict[str, Any]) -> None:
    assert len(_sample(built, None)) == built["images"]["patient_id"].nunique()


def test_sample_keeps_all_studies(built: dict[str, Any]) -> None:
    images = built["images"]
    chosen = set(_sample(built, 12))
    kept = images[images.patient_id.isin(chosen)]
    full = images.groupby("patient_id")["study_id"].nunique()
    assert (kept.groupby("patient_id")["study_id"].nunique() == full[list(kept.patient_id.unique())]).all()


def test_sample_covers_findings(built: dict[str, Any]) -> None:
    chosen = set(_sample(built, 12))
    images, labels = built["images"], built["labels"]
    studies = set(images[images.patient_id.isin(chosen)].study_id)
    sub = labels[labels.study_id.isin(studies)]
    assert all((sub[f] == 1.0).any() for f in FINDINGS)
