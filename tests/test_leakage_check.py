"""Real-data leakage audit (scripts/16_check_leakage.py) on synthetic artifacts.

Clean inputs must report no leak; each kind of leak, injected on purpose, must be caught; shared
boilerplate must be counted as such and not as a leak.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import pandas as pd
import pytest

from mfsi_cxr.data.leakage_check import (
    LeakageInputs,
    check_leakage,
    load_inputs,
    normalise,
    run_leakage_check,
    snippets,
)
from mfsi_cxr.data.synthetic import canary
from mfsi_cxr.utils.io import output_dir, read_meta


@pytest.fixture
def inputs(smoke_artifacts: dict[str, Any]) -> LeakageInputs:
    return load_inputs(smoke_artifacts)


def _tampered(inp: LeakageInputs) -> LeakageInputs:
    return dataclasses.replace(inp, text_inputs=inp.text_inputs.copy(deep=True),
                               rebuilt_index_text=dict(inp.rebuilt_index_text),
                               prior_texts=dict(inp.prior_texts), studies=dict(inp.studies))


def _row_with_priors(inp: LeakageInputs, n: int = 1) -> int:
    rows = inp.text_inputs.index[inp.text_inputs["n_priors"] >= n]
    assert len(rows), f"synthetic data must contain a sample with >= {n} priors"
    return int(rows[0])


def _later_study(inp: LeakageInputs, sid: str) -> str:
    """A study of the same patient that is later than `sid` (searching all samples)."""
    studies = inp.index.drop_duplicates("study_id").set_index("study_id")
    pid, rank = studies.loc[sid, "patient_id"], studies.loc[sid, "study_rank"]
    later = studies[(studies.patient_id == pid) & (studies.study_rank > rank)].index
    assert len(later)
    return str(later[0])


def test_snippets_split_fold_and_drop_short() -> None:
    out = snippets("No acute  disease seen today. OK.\nHeart Size Is Normal Here;", 15)
    assert out == {"no acute disease seen today", "heart size is normal here"}
    assert normalise("  A\tB  ") == "a b"


def test_clean_synthetic_inputs_have_no_leak(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    rep = check_leakage(inputs, smoke_artifacts)
    assert rep.n_leaks == 0, rep.as_dict()
    assert rep.n_samples == len(inputs.text_inputs)
    assert rep.n_with_priors > 0


def test_index_canary_in_stored_text_is_caught(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    inp = _tampered(inputs)
    sid = inp.text_inputs.at[0, "study_id"]
    inp.text_inputs.at[0, "index_text"] += " " + canary("section_findings", sid)
    rep = check_leakage(inp, smoke_artifacts)
    assert rep.index_forbidden_leak == 1
    assert rep.index_text_mismatch == 1


def test_forbidden_tag_in_index_text_is_caught(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    inp = _tampered(inputs)
    sid = inp.text_inputs.at[0, "study_id"]
    inp.text_inputs.at[0, "index_text"] += " [IMPRESSION] x"
    inp.rebuilt_index_text[sid] = inp.text_inputs.at[0, "index_text"]
    rep = check_leakage(inp, smoke_artifacts)
    assert rep.index_forbidden_tag == 1
    assert rep.n_leaks == 1


def test_later_text_in_prior_texts_is_caught(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    inp = _tampered(inputs)
    row = _row_with_priors(inp)
    sid = inp.text_inputs.at[row, "study_id"]
    for leaked_from in (sid, _later_study(inp, inp.text_inputs.at[row, "prior_study_ids"][0])):
        inp.prior_texts = dict(inputs.prior_texts)
        inp.prior_texts[sid] = (*inp.prior_texts[sid], canary("section_impression", leaked_from))
        assert check_leakage(inp, smoke_artifacts).prior_leak == 1, leaked_from


def test_later_study_listed_as_prior_is_caught(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    inp = _tampered(inputs)
    row = _row_with_priors(inp)
    sid = inp.text_inputs.at[row, "study_id"]
    inp.text_inputs.at[row, "prior_study_ids"] = [_later_study(inp, sid)]
    assert check_leakage(inp, smoke_artifacts).prior_not_earlier == 1


def test_too_many_priors_is_caught(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    cfg = copy.deepcopy(smoke_artifacts)
    row = _row_with_priors(inputs, 2)
    cfg["data"]["max_priors"] = int(inputs.text_inputs.at[row, "n_priors"]) - 1
    assert check_leakage(inputs, cfg).prior_too_many >= 1


def test_copied_forward_text_is_boilerplate_not_leak(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    """A later report repeating an earlier sentence is not a leak when the prior shows that sentence."""
    inp = _tampered(inputs)
    row = _row_with_priors(inp)
    sid = inp.text_inputs.at[row, "study_id"]
    prior = inp.text_inputs.at[row, "prior_study_ids"][0]
    repeated = normalise(canary("section_findings", prior))
    own = inp.studies[sid]
    inp.studies[sid] = dataclasses.replace(own, forbidden_index=own.forbidden_index | {repeated})
    rep = check_leakage(inp, smoke_artifacts)
    assert rep.prior_boilerplate_hits >= 1
    assert rep.n_leaks == 0


def test_forbidden_text_equal_to_allowed_text_is_boilerplate(inputs: LeakageInputs,
                                                             smoke_artifacts: dict) -> None:
    inp = _tampered(inputs)
    row = int(inp.text_inputs.index[inp.text_inputs["has_pre_read"]][0])
    sid = inp.text_inputs.at[row, "study_id"]
    own = inp.studies[sid]
    shared = next(iter(own.allowed_index))
    inp.studies[sid] = dataclasses.replace(own, forbidden_index=own.forbidden_index | {shared})
    rep = check_leakage(inp, smoke_artifacts)
    assert rep.index_boilerplate_hits >= 1
    assert rep.n_leaks == 0


def test_run_writes_counts_with_sidecar(smoke_artifacts: dict[str, Any]) -> None:
    rep = run_leakage_check(smoke_artifacts)
    out = output_dir(smoke_artifacts) / "tables" / "leakage_check.json"
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == rep.as_dict() and saved["n_leaks"] == 0
    assert all(isinstance(v, int) for v in saved.values())  # counts only, never text
    assert read_meta(out)["profile"] == smoke_artifacts["profile"]


def test_stored_prior_refs_match_n_priors(inputs: LeakageInputs) -> None:
    assert all(len(p) == n for p, n in zip(inputs.text_inputs["prior_study_ids"],
                                           inputs.text_inputs["n_priors"], strict=True))


def test_later_canary_in_radgraph_facts_is_caught(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    inp = _tampered(inputs)
    row = _row_with_priors(inp)
    prior = inp.text_inputs.at[row, "prior_study_ids"][0]
    leaked = canary("section_impression", inp.text_inputs.at[row, "study_id"])
    facts = inp.facts.copy()
    facts.loc[facts.index[facts["study_id"] == prior][0], "entity_text"] = leaked
    inp.facts = facts
    rep = check_leakage(inp, smoke_artifacts)
    assert rep.radgraph_leak == 1 and rep.radgraph_foreign_fact == 1


def test_words_of_separate_facts_do_not_form_a_leak(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    """A later sentence split over two facts of a prior (each fine on its own) is not a leak."""
    inp = _tampered(inputs)
    row = _row_with_priors(inp)
    prior = inp.text_inputs.at[row, "prior_study_ids"][0]
    later = normalise(canary("section_impression", inp.text_inputs.at[row, "study_id"]))
    half = len(later) // 2
    facts = inp.facts.copy()
    at = facts.index[facts["study_id"] == prior][:2]
    facts.loc[at, "entity_text"] = [later[:half], later[half:]]
    inp.facts = facts
    assert check_leakage(inp, smoke_artifacts).radgraph_leak == 0


def test_facts_of_a_non_prior_study_are_caught(inputs: LeakageInputs, smoke_artifacts: dict) -> None:
    inp = _tampered(inputs)
    priors = {p for ids in inp.text_inputs["prior_study_ids"] for p in ids}
    outsider = next(s for s in inp.index["study_id"].unique() if s not in priors)
    extra = inp.facts.iloc[[0]].assign(study_id=outsider)
    inp.facts = pd.concat([inp.facts, extra], ignore_index=True)
    assert check_leakage(inp, smoke_artifacts).radgraph_not_prior == 1
