from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mfsi_cxr.data.index import StudyOrderError, assign_study_rank, overlapping_patients


def _images(rows: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["patient_id", "study_id", "study_num", "study_order"])


def _ranks(rows: list[tuple[str, str, int, float]], policy: str = "drop_patients") -> dict[str, int]:
    ranked, _ = assign_study_rank(_images(rows), policy)
    return ranked.drop_duplicates("study_id").set_index("study_id")["study_rank"].to_dict()


def test_rank_follows_order_column_not_folder_number() -> None:
    rows = [("p1", "p1/study1", 1, 3), ("p1", "p1/study2", 2, 1), ("p1", "p1/study3", 3, 2)]
    assert _ranks(rows) == {"p1/study2": 0, "p1/study3": 1, "p1/study1": 2}


def test_order_ranks_images_study_starts_at_min() -> None:
    """CheXpert Plus ranks images: a 2-image study has two consecutive order values."""
    rows = [("p1", "p1/study1", 1, 3), ("p1", "p1/study1", 1, 4), ("p1", "p1/study2", 2, 1),
            ("p1", "p1/study2", 2, 2)]
    assert _ranks(rows) == {"p1/study2": 0, "p1/study1": 1}


def test_interleaved_studies_are_detected() -> None:
    rows = [("p1", "p1/study1", 1, 1), ("p1", "p1/study1", 1, 3), ("p1", "p1/study2", 2, 2),
            ("p2", "p2/study1", 1, 1), ("p2", "p2/study2", 2, 2)]
    assert overlapping_patients(_images(rows)) == {"p1"}
    ranked, dropped = assign_study_rank(_images(rows), "drop_patients")
    assert dropped == {"p1"} and set(ranked["patient_id"]) == {"p2"}
    with pytest.raises(StudyOrderError, match="interleaved"):
        assign_study_rank(_images(rows), "error")


def test_nested_interleaving_is_detected() -> None:
    """Study 3 overlaps study 1 even though study 2 sits between them."""
    rows = [("p1", "p1/study1", 1, 1), ("p1", "p1/study1", 1, 9), ("p1", "p1/study2", 2, 3),
            ("p1", "p1/study3", 3, 5)]
    assert overlapping_patients(_images(rows)) == {"p1"}


def test_missing_order() -> None:
    with pytest.raises(StudyOrderError, match="missing"):
        assign_study_rank(_images([("p1", "p1/study1", 1, float("nan"))]), "drop_patients")


def test_unknown_policy() -> None:
    with pytest.raises(ValueError):
        assign_study_rank(_images([("p1", "p1/study1", 1, 1)]), "guess")


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
