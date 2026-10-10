from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mfsi_cxr.data.index import StudyOrderError, assign_study_rank


def _images(rows: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["patient_id", "study_id", "study_num", "study_order"])


def test_rank_follows_order_column_not_folder_number() -> None:
    images = _images([("p1", "p1/study1", 1, 3), ("p1", "p1/study2", 2, 1), ("p1", "p1/study3", 3, 2)])
    ranks = assign_study_rank(images).set_index("study_id")["study_rank"]
    assert ranks.to_dict() == {"p1/study2": 0, "p1/study3": 1, "p1/study1": 2}


def test_ties_are_escalated() -> None:
    with pytest.raises(StudyOrderError, match="ties"):
        assign_study_rank(_images([("p1", "p1/study1", 1, 1), ("p1", "p1/study2", 2, 1)]))


def test_inconsistent_order_within_study() -> None:
    with pytest.raises(StudyOrderError, match="within studies"):
        assign_study_rank(_images([("p1", "p1/study1", 1, 1), ("p1", "p1/study1", 1, 2)]))


def test_missing_order() -> None:
    with pytest.raises(StudyOrderError, match="missing"):
        assign_study_rank(_images([("p1", "p1/study1", 1, float("nan"))]))


def test_priors_strictly_earlier(built: dict[str, Any]) -> None:
    index = built["index"].drop_duplicates("study_id").set_index("study_id")
    for r in index.itertuples():
        s = built["builder"].build(r.patient_id, r.Index)
        assert all(index.loc[p, "study_rank"] < r.study_rank for p in s.prior_study_ids)
        assert all(index.loc[p, "patient_id"] == r.patient_id for p in s.prior_study_ids)


def test_synthetic_folder_numbers_are_not_chronological(built: dict[str, Any]) -> None:
    """The fixture must exercise the order column, not the folder number."""
    studies = built["images"].drop_duplicates("study_id")
    assert (studies["study_num"] - 1 != studies["study_rank"]).any()
