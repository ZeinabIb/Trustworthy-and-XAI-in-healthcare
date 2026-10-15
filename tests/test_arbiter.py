"""Phase 5 arbiter with a mock LLM (no GPU, no model): packet safety and determinism, controls, vote,
hierarchy rule, per-claim evidence removal, caching / resumability, local-only and pinned endpoints."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mfsi_cxr.arbiter import llm_client as lc
from mfsi_cxr.arbiter.cases import CaseBuilder
from mfsi_cxr.arbiter.packet import REMOVED, Modality, PacketInputs, build_packet
from mfsi_cxr.arbiter.rules import Context, violations
from mfsi_cxr.arbiter.run import (
    HUMAN_REVIEW_CANNOT_DECIDE,
    HUMAN_REVIEW_HIERARCHY,
    HUMAN_REVIEW_NO_MAJORITY,
    INVALID,
    SHOWN,
    run_case,
)
from mfsi_cxr.arbiter.vote import NO_MAJORITY, vote
from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.data.synthetic import CANARY_SECTIONS, canary
from mfsi_cxr.utils.io import artifact_dir

MOD = Modality(0.8, 0.3, 0.6, 0.4, 0.5, "two_witness", True)


def _inputs(**kw: Any) -> PacketInputs:
    base = dict(case_id="p1/s1|Edema", finding="Edema", index_text="Shortness of breath.",
                prior_facts=({"rank": 2, "entity": "edema", "certainty": "uncertain", "anatomy": "lung"},
                             {"rank": 1, "entity": "edema", "certainty": "definitely present",
                              "anatomy": ""}),
                prob_cal=0.71, threshold=0.5, y_hat=True, modality=MOD,
                parent=("Lung Opacity", True, 0.8), children=())
    base.update(kw)
    return PacketInputs(**base)


@pytest.fixture
def cfg(smoke_cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(smoke_cfg)


# ---------------------------------------------------------------- packet


def test_packet_is_deterministic_and_ordered(cfg: dict[str, Any]) -> None:
    a, b = build_packet(_inputs(), cfg), build_packet(_inputs(), cfg)
    assert a.hash == b.hash and a.items == b.items
    assert a.ids == [f"E{k}" for k in range(1, len(a.items) + 1)]
    roles = list(a.roles.values())
    assert roles[0] == "classifier" and roles[1] == "pre_read" and roles[-1] == "hierarchy"
    facts = [t for i, t in a.items if a.roles[i] == "fact"]
    assert "date-rank 1" in facts[0] and "date-rank 2" in facts[1]          # most recent prior first
    assert build_packet(_inputs(prob_cal=0.72), cfg).hash != a.hash


def test_packet_caps_facts_and_truncates_text(cfg: dict[str, Any]) -> None:
    facts = tuple({"rank": k, "entity": f"e{k}", "certainty": "uncertain", "anatomy": ""}
                  for k in range(1, 30))
    p = build_packet(_inputs(prior_facts=facts, index_text="x" * 5000), cfg)
    assert sum(r == "fact" for r in p.roles.values()) == cfg["arbiter"]["max_facts"]
    pre = next(t for i, t in p.items if p.roles[i] == "pre_read")
    assert len(pre) < cfg["arbiter"]["index_text_chars"] + 100


def test_forbidden_tokens_are_refused(cfg: dict[str, Any]) -> None:
    with pytest.raises(AssertionError, match="forbidden"):
        build_packet(_inputs(index_text="history CANARY_FINDINGS_X"), cfg,
                     forbidden_tokens=("CANARY_FINDINGS_X",))


def test_packets_from_synthetic_artifacts_carry_no_index_canary(smoke_artifacts: dict[str, Any]) -> None:
    """Built from the Phase 1 synthetic artifacts (whose forbidden sections carry canary tokens)."""
    cfg = smoke_artifacts
    adir = artifact_dir(cfg)
    texts = pd.read_parquet(adir / "text_inputs.parquet")
    facts = pd.read_parquet(adir / "radgraph_facts.parquet")
    rows = [{"patient_id": r.patient_id, "study_id": r.study_id, "finding": f, "y_hat": True,
             "prob_cal": 0.6, "p_img_pos": 0.7, "p_txt_pos": 0.6, "r_img": 0.4, "r_txt": 0.2, "R": 0.3,
             "stratum": "two_witness", "escalate": True} for r in texts.itertuples() for f in FINDINGS]
    params = {"yhat": {"rule": "platt"}}
    cb = CaseBuilder(pd.DataFrame(rows), texts, facts, params, {f: True for f in FINDINGS}, np.zeros(14))
    n_with_facts = 0
    for r in texts.itertuples():
        tokens = tuple(canary(s, r.study_id) for s in CANARY_SECTIONS)
        for f in ("Pleural Effusion", "Edema", "Pneumonia"):
            p = build_packet(cb.inputs(r.study_id, f), cfg, forbidden_tokens=tokens)
            n_with_facts += any("date-rank" in t for _, t in p.items)
            assert not any(tok in p.render() for tok in tokens)
    assert n_with_facts > 0  # prior facts do reach packets


def test_control_switches_change_only_their_item(cfg: dict[str, Any]) -> None:
    base = build_packet(_inputs(), cfg)
    absent = build_packet(_inputs(), cfg, reliability="absent")
    assert "modality" not in absent.roles.values()
    keep = [t for i, t in base.items if base.roles[i] != "modality"]
    assert [t for _, t in absent.items] == keep
    swap = build_packet(_inputs(), cfg, reliability="shuffled_swap_direction")
    mod = next(t for i, t in swap.items if swap.roles[i] == "modality")
    assert "image leans ABSENT" in mod and "P(present) = 0.20" in mod and "r_img = 0.60" in mod
    donor = Modality(0.1, 0.9, 0.9, 0.9, 0.9, "two_witness", True)
    shuf = build_packet(_inputs(), cfg, reliability="shuffled_across_patients", donor=donor)
    assert "P(present) = 0.10" in next(t for i, t in shuf.items if shuf.roles[i] == "modality")
    blind = build_packet(_inputs(), cfg, order="blinded")
    assert "classifier" not in blind.roles.values()
    assert [t for i, t in blind.items] == [t for i, t in base.items if base.roles[i] != "classifier"]
    with pytest.raises(ValueError):
        build_packet(_inputs(), cfg, reliability="shuffled_across_patients")


def test_removal_keeps_ids(cfg: dict[str, Any]) -> None:
    p = build_packet(_inputs(), cfg)
    r = p.removed({"E2", "E4"})
    assert r.ids == p.ids and dict(r.items)["E2"] == REMOVED and dict(r.items)["E3"] == dict(p.items)["E3"]


# ---------------------------------------------------------------- vote and hierarchy


@pytest.mark.parametrize("answers,rule,expected", [
    (["present"] * 3, "majority", "present"), (["present", "present", "absent"], "majority", "present"),
    (["present", "absent", "cannot_decide"], "majority", NO_MAJORITY),
    (["absent", None, "absent"], "majority", "absent"), ([None, None, "absent"], "majority", NO_MAJORITY),
    ([None, None, None], "majority", NO_MAJORITY),
    (["present", "present", "absent"], "unanimous", NO_MAJORITY), (["absent"] * 3, "unanimous", "absent"),
    (["cannot_decide"] * 2 + ["absent"], "majority", "cannot_decide"),
])
def test_vote(answers: list, rule: str, expected: str) -> None:
    assert vote(answers, rule) == expected


def _ctx(states: dict[str, bool], reliable: tuple[str, ...] = ()) -> Context:
    full = {f: False for f in FINDINGS} | states
    return Context(full, dict(full), {f: f in reliable for f in FINDINGS})


@pytest.mark.parametrize("finding,decision,states,expect", [
    ("Pneumonia", "present", {"Consolidation": True, "Lung Opacity": True}, False),
    ("Pneumonia", "present", {"Consolidation": False, "Lung Opacity": True}, True),     # parent absent
    ("Pneumonia", "present", {"Consolidation": True, "Lung Opacity": False}, True),     # grandparent absent
    ("Pneumonia", "absent", {}, False),
    ("Consolidation", "absent", {"Pneumonia": True}, True),                           # child present
    ("Consolidation", "absent", {"Pneumonia": False}, False),
    ("Fracture", "present", {}, False),                                               # a root
    ("No Finding", "present", {"Edema": True, "Lung Opacity": True}, True),
    ("Edema", "present", {"Lung Opacity": True, "No Finding": True}, True),
    ("Support Devices", "present", {"No Finding": True}, False),                     # not a pathology
    ("Edema", "cannot_decide", {}, False),
])
def test_hierarchy_rule(finding: str, decision: str, states: dict, expect: bool) -> None:
    assert bool(violations(finding, decision, _ctx(states), cited_hierarchy=True,
                           unaddressed_is_violation=True)) is expect


def test_unaddressed_reliable_parent() -> None:
    states = {"Lung Opacity": True}
    ctx = Context({f: False for f in FINDINGS} | states | {"Lung Opacity": True},
                  {f: False for f in FINDINGS}, {f: f == "Lung Opacity" for f in FINDINGS})
    assert violations("Edema", "present", ctx, cited_hierarchy=False, unaddressed_is_violation=True)
    assert not violations("Edema", "present", ctx, cited_hierarchy=True, unaddressed_is_violation=True)
    assert not violations("Edema", "present", ctx, cited_hierarchy=False, unaddressed_is_violation=False)


# ---------------------------------------------------------------- pipeline with a scripted mock LLM


def _client(cfg: dict[str, Any], tmp_path, fn) -> lc.CachedClient:
    a = cfg["arbiter"]
    return lc.CachedClient(lc.MockBackend(fn), tmp_path / "runs", a["temperature"], a["top_p"],
                           a["max_tokens"])


def _evidence(messages: list[dict[str, str]]) -> dict[str, str]:
    lines = messages[1]["content"].splitlines()
    return dict(line.split(": ", 1) for line in lines if line[:1] == "E" and ": " in line)


def depends_on_e2(messages: list[dict[str, str]], seed: int) -> str:
    """Decision depends on E2 only; C1 cites E2, C2 cites E3."""
    decision = "absent" if _evidence(messages).get("E2") == REMOVED else "present"
    return json.dumps({"decision": decision, "claims": [{"id": "C1", "text": "a", "cites": ["E2"]},
                                                        {"id": "C2", "text": "b", "cites": ["E3"]},
                                                        {"id": "C3", "text": "c", "cites": ["E2"]}]})


def _no_conflict() -> Context:
    return _ctx({"Lung Opacity": True})


def test_ablation_tags_claims(cfg: dict[str, Any], tmp_path) -> None:
    client = _client(cfg, tmp_path, depends_on_e2)
    res = run_case(client, cfg, build_packet(_inputs(), cfg), _no_conflict())
    assert res.status == SHOWN and res.decision == "present"
    tags = {c["id"]: c["tag"] for c in res.claims}
    assert tags == {"C1": "VERIFIED", "C2": "UNVERIFIED", "C3": "VERIFIED"}
    assert res.n_calls == 3 + 2 * 3            # vote + two distinct citation sets x 3 seeds


def test_no_majority_invalid_and_cannot_decide(cfg: dict[str, Any], tmp_path) -> None:
    by_seed = {1: "present", 2: "absent", 3: "cannot_decide"}
    split = _client(cfg, tmp_path / "a", lambda m, s: json.dumps({"decision": by_seed[s], "claims": []}))
    r = run_case(split, cfg, build_packet(_inputs(), cfg), _no_conflict())
    assert r.status == HUMAN_REVIEW_NO_MAJORITY and r.claims == []
    junk = _client(cfg, tmp_path / "b", lambda m, s: "not json")
    r = run_case(junk, cfg, build_packet(_inputs(), cfg), _no_conflict())
    assert r.status == INVALID and r.n_invalid_runs == 3
    assert r.n_calls == 3 * (1 + cfg["arbiter"]["retries_invalid"])
    unsure = _client(cfg, tmp_path / "c",
                     lambda m, s: json.dumps({"decision": "cannot_decide", "claims": []}))
    r = run_case(unsure, cfg, build_packet(_inputs(), cfg), _no_conflict())
    assert r.status == HUMAN_REVIEW_CANNOT_DECIDE


def test_repair_retry_recovers(cfg: dict[str, Any], tmp_path) -> None:
    def fn(messages: list[dict[str, str]], seed: int) -> str:
        if messages[-1]["role"] == "user" and "previous answer was not valid" in messages[-1]["content"]:
            return json.dumps({"decision": "present", "claims": [{"id": "C1", "text": "a", "cites": ["E1"]}]})
        return json.dumps({"decision": "present", "claims": [{"id": "C1", "text": "a", "cites": ["E99"]}]})
    r = run_case(_client(cfg, tmp_path, fn), cfg, build_packet(_inputs(), cfg), _no_conflict())
    assert r.status == SHOWN and r.n_invalid_runs == 0 and r.n_failed_parses == 3


def test_hierarchy_violation_hides_reasoning(cfg: dict[str, Any], tmp_path) -> None:
    client = _client(cfg, tmp_path, depends_on_e2)
    r = run_case(client, cfg, build_packet(_inputs(), cfg), _ctx({"Lung Opacity": False}))
    assert r.status == HUMAN_REVIEW_HIERARCHY and r.claims == [] and r.hierarchy_violations


def test_cache_makes_reruns_free_and_identical(cfg: dict[str, Any], tmp_path) -> None:
    first = run_case(_client(cfg, tmp_path, depends_on_e2), cfg, build_packet(_inputs(), cfg), _no_conflict())
    again = _client(cfg, tmp_path, lambda m, s: pytest.fail("cache miss"))
    second = run_case(again, cfg, build_packet(_inputs(), cfg), _no_conflict())
    assert second.to_json() == first.to_json() and again.n_new == 0 and again.n_hit == first.n_calls


def test_endpoint_must_be_local_and_weights_pinned(cfg: dict[str, Any], tmp_path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    for bad in ("https://api.example.com/v1", "http://10.0.0.5:11434/v1"):
        with pytest.raises(lc.NonLocalEndpointError):
            lc.check_local(bad)
    lc.check_local("http://127.0.0.1:11434/v1")
    c = copy.deepcopy(cfg)
    c["arbiter"].update(model="qwen2.5:7b-instruct-q4_K_M", endpoint="http://127.0.0.1:11434/v1",
                        model_digest=None)
    monkeypatch.setattr(lc, "served_digest", lambda *a, **k: "abc")
    with pytest.raises(lc.PinError, match="not set"):
        lc.make_client(c, tmp_path)
    c["arbiter"]["model_digest"] = "other"
    with pytest.raises(lc.PinError, match="digest"):
        lc.make_client(c, tmp_path)
    c["arbiter"]["model_digest"] = "abc"
    assert lc.make_client(c, tmp_path).backend.revision == "abc"
