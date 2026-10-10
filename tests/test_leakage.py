"""Canary leakage test (mandatory, Phase 1.5). Must stay green on every data-loader change.

Every forbidden section of every synthetic study holds a unique canary naming its study. The index
text may never contain a canary of the index study (or of any study); prior texts may contain canaries
of earlier studies only. Pending: RadGraph facts (Phase 1.6, format not yet inspected) and the
tokenised model input (Phase 2) must be added to this test when they exist.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pytest

from mfsi_cxr.data.synthetic import CANARY_SECTIONS, canary
from mfsi_cxr.data.text_inputs import FORBIDDEN_INDEX_ALWAYS, LeakageConfigError, TextBuilder

ALLOWED_PRIOR_CANARIES = ("section_findings", "section_impression", "section_narrative")


def _samples(built: dict[str, Any]) -> list:
    idx = built["index"].drop_duplicates("study_id")
    return [built["builder"].build(r.patient_id, r.study_id) for r in idx.itertuples()]


def _all_text(sample) -> str:
    return " ".join([sample.index_text, *sample.prior_texts])


def test_no_index_canary_anywhere_in_its_sample(built: dict[str, Any]) -> None:
    for s in _samples(built):
        text = _all_text(s)
        for section in CANARY_SECTIONS:
            assert canary(section, s.index_study_id) not in text, (section, s.index_study_id)


def test_index_text_holds_no_canary_at_all(built: dict[str, Any]) -> None:
    for s in _samples(built):
        assert "CANARY" not in s.index_text, s.index_study_id


def test_prior_canaries_do_appear(built: dict[str, Any]) -> None:
    """Guards against a trivially passing test: earlier reports ARE read in full."""
    with_priors = [s for s in _samples(built) if s.n_priors]
    assert with_priors, "synthetic data must contain patients with priors"
    for s in with_priors:
        text = " ".join(s.prior_texts)
        for prior in s.prior_study_ids:
            for section in ALLOWED_PRIOR_CANARIES:
                assert canary(section, prior) in text


def test_later_studies_never_appear(built: dict[str, Any]) -> None:
    index = built["index"].drop_duplicates("study_id").set_index("study_id")
    for s in _samples(built):
        rank = index.loc[s.index_study_id, "study_rank"]
        later = index[(index.patient_id == s.patient_id) & (index.study_rank > rank)].index
        text = _all_text(s)
        for sid in later:
            assert sid not in s.prior_study_ids
            assert all(canary(sec, sid) not in text for sec in CANARY_SECTIONS)


def test_unlisted_column_is_forbidden_by_default(built: dict[str, Any]) -> None:
    for s in _samples(built):
        assert "NEW_UNREVIEWED" not in _all_text(s)


def test_artifact_table_has_no_canary(built: dict[str, Any]) -> None:
    table = built["builder"].build_all()
    for col in table.columns:
        values = table[col].astype(str)
        assert not values.str.contains("CANARY").any(), col


@pytest.mark.parametrize("column", sorted(FORBIDDEN_INDEX_ALWAYS))
def test_config_cannot_allow_forbidden_index_column(built: dict[str, Any], column: str) -> None:
    cfg = copy.deepcopy(built["cfg"])
    cfg["data"]["text_sections"]["allowed_index"].append(column)
    with pytest.raises(LeakageConfigError):
        TextBuilder.from_root(built["root"], built["index"], cfg)


def test_builder_rejects_extra_index_columns(built: dict[str, Any]) -> None:
    meta = pd.DataFrame({"image_path": ["x"], "section_findings": ["CANARY"]})
    with pytest.raises(LeakageConfigError):
        TextBuilder(built["index"], meta, pd.DataFrame({"image_path": []}), built["cfg"])


def test_missing_pre_read_text_gets_explicit_token(built: dict[str, Any]) -> None:
    token = built["cfg"]["data"]["no_pre_read_token"]
    samples = _samples(built)
    empty = [s for s in samples if not s.has_pre_read]
    assert empty, "synthetic data must contain studies without pre-read text"
    for s in empty:
        assert s.index_text.startswith(token)
        assert s.text_len == 0
    for s in samples:
        if s.has_pre_read:
            assert token not in s.index_text and s.text_len > 0
        assert "[DEMOGRAPHICS]" in s.index_text


def test_priors_truncated_most_recent_first(built: dict[str, Any]) -> None:
    cfg = copy.deepcopy(built["cfg"])
    cfg["data"]["max_priors"] = 1
    builder = TextBuilder.from_root(built["root"], built["index"], cfg)
    index = built["index"].drop_duplicates("study_id").set_index("study_id")
    for r in index.itertuples():
        s = builder.build(r.patient_id, r.Index)
        assert s.n_priors <= 1
        if r.study_rank > 0:
            assert index.loc[s.prior_study_ids[0], "study_rank"] == r.study_rank - 1
