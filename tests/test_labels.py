from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.data.labels import LabelError, labels_per_study, normalise_value, prevalence_table
from mfsi_cxr.data.splits import make_splits


def test_values_and_column_order(built: dict[str, Any]) -> None:
    labels = built["labels"]
    assert list(labels.columns) == ["study_id", *FINDINGS]
    values = set(np.unique(labels[list(FINDINGS)].to_numpy()[~labels[list(FINDINGS)].isna().to_numpy()]))
    assert values <= {1.0, 0.0, -1.0}
    assert labels[list(FINDINGS)].isna().any().any(), "blank labels must stay NaN"
    assert labels["study_id"].is_unique


@pytest.mark.parametrize(("raw", "expected"), [(1, 1.0), ("0", 0.0), (-1.0, -1.0), ("1.0", 1.0)])
def test_normalise(raw: Any, expected: float) -> None:
    assert normalise_value(raw) == expected


@pytest.mark.parametrize("raw", [None, "", " ", float("nan")])
def test_blank_is_nan(raw: Any) -> None:
    assert np.isnan(normalise_value(raw))


@pytest.mark.parametrize("raw", [2, "yes", 0.5])
def test_unexpected_value(raw: Any) -> None:
    with pytest.raises(LabelError):
        normalise_value(raw)


def _records(paths: list[str], value: float) -> pd.DataFrame:
    return pd.DataFrame([{"path_to_image": p, **{f: value for f in FINDINGS}} for p in paths])


TWO_IMAGES = pd.DataFrame({"image_path": ["t/p1/study1/view1.png", "t/p1/study1/view2.png"],
                           "study_id": ["p1/study1"] * 2})


def _two(first: float | None, second: float | None) -> pd.DataFrame:
    return pd.concat(
        [_records(["t/p1/study1/view1.png"], first), _records(["t/p1/study1/view2.png"], second)]
    )


def test_disagreeing_images_error_policy() -> None:
    with pytest.raises(LabelError, match="differ"):
        labels_per_study(_two(1.0, 0.0), TWO_IMAGES, "path_to_image", "error")


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [(1.0, 0.0, 1.0), (0.0, -1.0, -1.0), (-1.0, 1.0, 1.0), (None, 0.0, 0.0), (None, None, None)],
)
def test_priority_combination(first: float | None, second: float | None, expected: float | None) -> None:
    out = labels_per_study(_two(first, second), TWO_IMAGES, "path_to_image", "priority")
    value = out.loc[0, "Cardiomegaly"]
    assert (np.isnan(value) if expected is None else value == expected)


def test_unknown_aggregation() -> None:
    with pytest.raises(ValueError):
        labels_per_study(_two(1.0, 1.0), TWO_IMAGES, "path_to_image", "vote")


def test_extension_independent_join() -> None:
    images = pd.DataFrame({"image_path": ["t/p1/study1/view1.png"], "study_id": ["p1/study1"]})
    out = labels_per_study(_records(["t/p1/study1/view1.jpg"], 1.0), images, "path_to_image", "priority")
    assert len(out) == 1 and out.loc[0, "Cardiomegaly"] == 1.0


def test_prevalence_counts(built: dict[str, Any]) -> None:
    index, labels = built["index"], built["labels"]
    splits = make_splits(sorted(index.patient_id.unique()), set(), {"train": 0.7, "val": 0.1, "test": 0.2},
                         0, "holdout")
    table = prevalence_table(labels, index, splits, "train")
    train_patients = splits[splits.split == "train"].patient_id
    train_studies = index[index.patient_id.isin(train_patients)].study_id.nunique()
    assert list(table["finding"]) == list(FINDINGS)
    assert (table[["n_pos", "n_neg", "n_uncertain", "n_blank"]].sum(axis=1) == train_studies).all()
