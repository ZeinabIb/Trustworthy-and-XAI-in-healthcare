"""Check 2: hierarchy rule (Phase 5), no model. The arbiter's decision for finding f, combined with the other
findings' states of the same sample, must be coherent with the approved label tree (constants.PARENT, all
ancestors; No Finding rule approved 2026-10-11):
- f present while an ancestor is absent -> violation;
- f absent while a descendant is present -> violation (a parent denied under a present child);
- No Finding present with any pathology present, or a pathology present with No Finding present -> violation;
- (plan design choice, arbiter.unaddressed_reliable_parent_is_violation) f present while the CLASSIFIER calls an
  ancestor absent with R_ancestor >= theta_ancestor, and no claim of the run cites the label-tree item.
State of another finding = its final arbiter decision if it was also arbitrated, else the classifier's call.
"""

from __future__ import annotations

from dataclasses import dataclass

from mfsi_cxr.constants import FINDINGS, NO_FINDING, PATHOLOGIES, ancestors


@dataclass(frozen=True)
class Context:
    states: dict[str, bool]              # finding -> present (final decision if arbitrated, else classifier)
    classifier: dict[str, bool]          # finding -> classifier call
    reliable: dict[str, bool]            # finding -> R >= theta_R (operating point)


def descendants(finding: str) -> list[str]:
    return [f for f in FINDINGS if finding in ancestors(f)]


def violations(finding: str, decision: str, ctx: Context, cited_hierarchy: bool,
               unaddressed_is_violation: bool) -> list[str]:
    """Reasons the decision breaks the label tree (empty list = coherent). 'cannot_decide' never violates."""
    if decision not in ("present", "absent"):
        return []
    out = []
    if decision == "present":
        out += [f"{finding} present under absent ancestor {a}" for a in ancestors(finding) if not ctx.states[a]]
        if finding == NO_FINDING:
            out += [f"No Finding present with {p} present" for p in PATHOLOGIES if ctx.states[p]]
        elif finding in PATHOLOGIES and ctx.states[NO_FINDING]:
            out.append(f"{finding} present with No Finding present")
        if unaddressed_is_violation and not cited_hierarchy:
            out += [f"{finding} present, classifier denies reliable ancestor {a}, label tree not addressed"
                    for a in ancestors(finding) if not ctx.classifier[a] and ctx.reliable[a]]
    else:
        out += [f"{finding} absent under present descendant {d}" for d in descendants(finding) if ctx.states[d]]
    return out
