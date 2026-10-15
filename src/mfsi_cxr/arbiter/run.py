"""Arbiter pipeline for one case (Phase 5): 3 voting runs -> vote -> hierarchy rule -> per-claim evidence
removal -> ArbiterResult. The model output is untrusted: nothing is shown unless every check passes, and
then every claim is shown with its tag (unverified claims are never hidden).

Check 3 (per-claim evidence removal, arbiter/ablate.py) uses the majority-consistent run: the first run
whose decision equals the majority. VERIFIED means the arbiter USED that evidence, not that the claim is
medically true; the words true / correct are never used for these tags.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mfsi_cxr.arbiter import prompts
from mfsi_cxr.arbiter.ablate import UNVERIFIED, VERIFIED, tag_claims
from mfsi_cxr.arbiter.llm_client import CachedClient
from mfsi_cxr.arbiter.packet import Packet
from mfsi_cxr.arbiter.rules import Context, violations
from mfsi_cxr.arbiter.vote import NO_MAJORITY, vote

SHOWN = "SHOWN"
HUMAN_REVIEW_NO_MAJORITY = "HUMAN_REVIEW_NO_MAJORITY"
HUMAN_REVIEW_CANNOT_DECIDE = "HUMAN_REVIEW_CANNOT_DECIDE"
HUMAN_REVIEW_HIERARCHY = "HUMAN_REVIEW_HIERARCHY"
INVALID = "INVALID"


@dataclass
class RunOutput:
    seed: int
    decision: str | None            # None = invalid after the repair retry
    claims: list[dict[str, Any]]
    n_calls: int
    error: str | None = None


@dataclass
class ArbiterResult:
    case_id: str
    finding: str
    packet_hash: str
    status: str
    decision: str | None
    claims: list[dict[str, Any]] = field(default_factory=list)   # {id, text, cites, tag}
    evidence: list[tuple[str, str]] = field(default_factory=list)
    run_decisions: list[str | None] = field(default_factory=list)
    hierarchy_violations: list[str] = field(default_factory=list)
    n_calls: int = 0
    n_invalid_runs: int = 0
    n_failed_parses: int = 0     # voting runs: calls whose output failed the schema (before or after repair)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def one_run(client: CachedClient, cfg: dict[str, Any], packet: Packet, seed: int) -> RunOutput:
    a = cfg["arbiter"]
    msgs = prompts.messages(cfg, packet)
    n, error = 0, None
    for attempt in range(1 + a["retries_invalid"]):
        rec = client.complete(msgs, seed, packet.hash)
        n += 1
        try:
            ans = prompts.parse(rec["raw"], packet, a["max_claims"])
            return RunOutput(seed, ans.decision, [c.model_dump() for c in ans.claims], n)
        except prompts.InvalidAnswerError as err:
            error = str(err)
            if attempt < a["retries_invalid"]:
                msgs = prompts.repair_messages(cfg, prompts.messages(cfg, packet), rec["raw"], error)
    return RunOutput(seed, None, [], n, error)


def voted(client: CachedClient, cfg: dict[str, Any], packet: Packet) -> tuple[str, list[RunOutput]]:
    runs = [one_run(client, cfg, packet, s) for s in cfg["arbiter"]["seeds"]]
    return vote([r.decision for r in runs], cfg["arbiter"]["vote"]), runs


def run_case(client: CachedClient, cfg: dict[str, Any], packet: Packet, ctx: Context) -> ArbiterResult:
    a = cfg["arbiter"]
    majority, runs = voted(client, cfg, packet)
    res = ArbiterResult(packet.case_id, packet.finding, packet.hash, "", None, evidence=list(packet.items),
                        run_decisions=[r.decision for r in runs], n_calls=sum(r.n_calls for r in runs),
                        n_invalid_runs=sum(r.decision is None for r in runs),
                        n_failed_parses=sum(r.n_calls - (r.decision is not None) for r in runs))
    if all(r.decision is None for r in runs):
        res.status = INVALID
        return res
    if majority == NO_MAJORITY:
        res.status = HUMAN_REVIEW_NO_MAJORITY
        return res
    if majority == "cannot_decide":
        res.status, res.decision = HUMAN_REVIEW_CANNOT_DECIDE, majority
        return res
    chosen = next(r for r in runs if r.decision == majority)
    hierarchy_ids = {i for i, role in packet.roles.items() if role == "hierarchy"}
    cited_tree = any(set(c["cites"]) & hierarchy_ids for c in chosen.claims)
    bad = violations(packet.finding, majority, ctx, cited_tree, a["unaddressed_reliable_parent_is_violation"])
    if bad:
        res.status, res.decision, res.hierarchy_violations = HUMAN_REVIEW_HIERARCHY, majority, bad
        return res
    def revote(p: Packet) -> tuple[str, int]:
        new, extra = voted(client, cfg, p)
        return new, sum(r.n_calls for r in extra)

    res.claims, extra_calls = tag_claims(chosen.claims, packet, majority, revote)
    res.n_calls += extra_calls
    res.status, res.decision = SHOWN, majority
    return res


TAG_TEXT = {VERIFIED: "VERIFIED - the decision changes without this evidence (the arbiter used it)",
            UNVERIFIED: "UNVERIFIED - the decision does not depend on this evidence"}


def render_text(res: ArbiterResult) -> str:
    """Plain-text clinician-facing view. Nothing but the status is shown unless all checks passed."""
    head = f"Finding: {res.finding}\nStatus: {res.status}"
    if res.status != SHOWN:
        return head + "\n(No reasoning is shown: the case goes to human review.)\n"
    lines = [head, f"Arbiter decision: {res.decision}", "Claims:"]
    for c in res.claims:
        lines.append(f"  {c['id']} [{TAG_TEXT[c['tag']]}] {c['text']} (cites {', '.join(c['cites'])})")
    lines.append("Evidence:")
    lines += [f"  {i}: {t}" for i, t in res.evidence]
    return "\n".join(lines) + "\n"
