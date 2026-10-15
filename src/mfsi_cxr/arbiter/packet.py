"""Evidence packet for one escalated (sample, finding) (Phase 5). A pure function of an allowlisted input record.

HARD RULE: the arbiter never sees the image, the full chart, the index study's Findings / Impression /
narrative, or any label. PacketInputs has no field that could carry them: the index text is the leakage-safe
pre-read text built in Phase 1 (text_inputs.parquet), prior facts come from RadGraph of PRIOR studies only, and
the rest are model outputs. build_packet asserts that no index-study canary token and no label appears.

Items (ids E1..En, deterministic order; order 'revealed' puts the classifier output first, 'blinded' hides it):
  classifier output for the finding (calibrated probability, decision threshold, called state);
  pre-read text of the index study (truncated);
  prior-report facts about the finding, "[prior study, date-rank k] <entity> - <certainty> - <anatomy>";
  modality information (profiles, leanings, margins, R and stratum) - reliability control switch;
  label-tree context (parent's and children's classifier states).
Removed items keep their id and read "[removed]" (ids never renumbered).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from mfsi_cxr.constants import FINDINGS, PARENT

RELIABILITY_MODES = ("true", "shuffled_across_patients", "shuffled_swap_direction", "absent")
ORDER_MODES = ("revealed", "blinded")
REMOVED = "[removed]"
CERTAINTY_RANK = {"definitely present": 0, "definitely absent": 1, "uncertain": 2}
FORBIDDEN_KEYS = frozenset({"y", "label", "labels", "y_true", "findings_text", "impression", "report", "image"})


@dataclass(frozen=True)
class Modality:
    p_img_pos: float
    p_txt_pos: float
    r_img: float
    r_txt: float
    R: float
    stratum: str
    text_informative: bool


@dataclass(frozen=True)
class PacketInputs:
    """Everything the arbiter may see for one case. No label, image, report or chart field exists."""

    case_id: str
    finding: str
    index_text: str
    prior_facts: tuple[dict[str, Any], ...]      # {rank, entity, certainty, anatomy}
    prob_cal: float
    threshold: float
    y_hat: bool
    modality: Modality
    parent: tuple[str, bool, float] | None       # (name, classifier state, calibrated probability)
    children: tuple[tuple[str, bool], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Packet:
    case_id: str
    finding: str
    items: tuple[tuple[str, str], ...]           # (id, text)
    roles: dict[str, str]                        # id -> role (classifier, pre_read, fact, modality, hierarchy)
    variant: dict[str, str]

    @property
    def ids(self) -> list[str]:
        return [i for i, _ in self.items]

    @property
    def hash(self) -> str:
        blob = json.dumps({"finding": self.finding, "items": self.items, "variant": self.variant},
                          sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def removed(self, ids: set[str]) -> Packet:
        """The same packet with the given items replaced by "[removed]" (ids unchanged)."""
        items = tuple((i, REMOVED if i in ids else t) for i, t in self.items)
        return Packet(self.case_id, self.finding, items, self.roles, self.variant)

    def render(self) -> str:
        return "\n".join(f"{i}: {t}" for i, t in self.items)


def _state(present: bool) -> str:
    return "PRESENT" if present else "ABSENT"


def _lean(p: float) -> str:
    return "leans PRESENT" if p >= 0.5 else "leans ABSENT"


def select_facts(facts: tuple[dict[str, Any], ...], max_facts: int) -> list[dict[str, Any]]:
    """Most recent prior first (rank 1 = most recent), then certainty, then entity text (deterministic)."""
    order = sorted(facts, key=lambda f: (int(f["rank"]), CERTAINTY_RANK.get(f["certainty"], 9), str(f["entity"])))
    return order[:max_facts]


def _modality_text(finding: str, m: Modality, swap: bool) -> str:
    p_img, p_txt = (1 - m.p_img_pos, 1 - m.p_txt_pos) if swap else (m.p_img_pos, m.p_txt_pos)
    txt = (f"text profile P(present) = {p_txt:.2f} (text {_lean(p_txt)}), margin r_txt = {m.r_txt:.2f}"
           if m.text_informative else "text has no signal for this finding")
    return (f"Modality reliability for {finding}: image profile P(present) = {p_img:.2f} (image {_lean(p_img)}), "
            f"margin r_img = {m.r_img:.2f}; {txt}; reliability R = {m.R:.2f} (stratum {m.stratum}).")


def build_packet(inp: PacketInputs, cfg: dict[str, Any], reliability: str = "true", order: str = "revealed",
                 donor: Modality | None = None, forbidden_tokens: tuple[str, ...] = ()) -> Packet:
    """Evidence packet. `donor` supplies another patient's modality info for 'shuffled_across_patients'.
    `forbidden_tokens` (e.g. the index study's canary tokens in tests) must not occur anywhere."""
    if reliability not in RELIABILITY_MODES:
        raise ValueError(f"packet reliability must be one of {RELIABILITY_MODES}, got {reliability!r}")
    if order not in ORDER_MODES:
        raise ValueError(f"packet order must be one of {ORDER_MODES}, got {order!r}")
    if inp.finding not in FINDINGS:
        raise ValueError(f"unknown finding {inp.finding!r}")
    a = cfg["arbiter"]
    entries: list[tuple[str, str]] = []
    if order == "revealed":
        entries.append(("classifier", f"Classifier for {inp.finding}: calibrated probability {inp.prob_cal:.2f}; "
                                      f"decision threshold {inp.threshold:.2f}; the classifier calls it "
                                      f"{_state(inp.y_hat)}."))
    text = (inp.index_text or "").strip()
    text = text[: a["index_text_chars"]] + (" ..." if len(text) > a["index_text_chars"] else "")
    entries.append(("pre_read", f"Pre-read text of the current study (indication / history / technique): "
                                f"{text or '[no pre-read text]'}"))
    facts = select_facts(inp.prior_facts, a["max_facts"])
    if facts:
        for f in facts:
            anatomy = f.get("anatomy") or "no anatomy given"
            entries.append(("fact", f"[prior study, date-rank {int(f['rank'])}] {f['entity']} - {f['certainty']} - "
                                    f"{anatomy}"))
    else:
        entries.append(("fact", f"No prior-report facts about {inp.finding}."))
    if reliability != "absent":
        m = donor if reliability == "shuffled_across_patients" else inp.modality
        if m is None:
            raise ValueError("shuffled_across_patients needs a donor case")
        entries.append(("modality", _modality_text(inp.finding, m, reliability == "shuffled_swap_direction")))
    parts = []
    if inp.parent is not None:
        name, state, prob = inp.parent
        parts.append(f"parent finding {name}: classifier calls it {_state(state)} "
                     f"(calibrated probability {prob:.2f})")
    elif inp.finding in PARENT:
        parts.append(f"parent finding {PARENT[inp.finding]}: not available")
    if inp.children:
        parts.append("child findings: " + ", ".join(f"{c} {_state(s)}" for c, s in inp.children))
    entries.append(("hierarchy", "Label tree: " + ("; ".join(parts) if parts else "no parent or child findings")
                    + "."))
    items = tuple((f"E{k}", t) for k, (_, t) in enumerate(entries, start=1))
    roles = {f"E{k}": r for k, (r, _) in enumerate(entries, start=1)}
    packet = Packet(inp.case_id, inp.finding, items, roles, {"reliability": reliability, "order": order})
    rendered = packet.render()
    leaked = [t for t in forbidden_tokens if t and t in rendered]
    if leaked:
        raise AssertionError(f"forbidden content in the evidence packet of {inp.case_id}: {leaked[:3]}")
    return packet
