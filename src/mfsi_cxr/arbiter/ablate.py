"""Check 3: per-claim evidence removal (Phase 5). For each claim of the majority-consistent run, the evidence it
cites is replaced by "[removed]" (ids unchanged) and the 3-seed vote is repeated: decision changed (including
to NO_MAJORITY) -> VERIFIED, unchanged -> UNVERIFIED. Claims with the same sorted cited set share one re-ask,
so the cost is at most (distinct cited sets) x len(seeds) extra runs. VERIFIED = the arbiter used the
evidence; it does not mean the claim is true."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mfsi_cxr.arbiter.packet import Packet

VERIFIED, UNVERIFIED = "VERIFIED", "UNVERIFIED"


def tag_claims(claims: list[dict[str, Any]], packet: Packet, majority: str,
               revote: Callable[[Packet], tuple[str, int]]) -> tuple[list[dict[str, Any]], int]:
    """(claims with a 'tag', extra calls). revote(packet) -> (majority label, calls used)."""
    verdict: dict[tuple[str, ...], str] = {}
    calls = 0
    out = []
    for c in claims:
        key = tuple(sorted(set(c["cites"])))
        if key not in verdict:
            new, n = revote(packet.removed(set(key)))
            calls += n
            verdict[key] = VERIFIED if new != majority else UNVERIFIED
        out.append({**c, "tag": verdict[key]})
    return out, calls
