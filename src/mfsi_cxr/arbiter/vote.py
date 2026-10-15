"""Check 1: three-run vote (Phase 5). Answers are 'present' / 'absent' / 'cannot_decide' or None (invalid run,
counts as no answer). majority: a label given by >= 2 runs; unanimous: all runs agree. Otherwise NO_MAJORITY.
A majority on 'cannot_decide' goes to human review (the caller maps it)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

NO_MAJORITY = "NO_MAJORITY"
RULES = ("majority", "unanimous")


def vote(answers: Sequence[str | None], rule: str) -> str:
    if rule not in RULES:
        raise ValueError(f"arbiter.vote must be one of {RULES}, got {rule!r}")
    valid = [a for a in answers if a is not None]
    if not valid:
        return NO_MAJORITY
    label, count = Counter(valid).most_common(1)[0]
    need = len(answers) if rule == "unanimous" else len(answers) // 2 + 1
    return label if count >= need else NO_MAJORITY
