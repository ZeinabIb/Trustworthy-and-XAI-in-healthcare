"""Deterministic mock arbiter for the smoke profile and tests (no GPU, no model). It reads the rendered packet:
decision = the classifier's call if shown, else the image leaning, else cannot_decide; claims cite the items it
read. It is a stand-in for plumbing checks only, never for results."""

from __future__ import annotations

import json
import re


def policy(messages: list[dict[str, str]], seed: int) -> str:
    user = messages[1]["content"]
    items = dict(re.findall(r"^(E\d+): (.*)$", user, flags=re.M))
    clf = next((i for i, t in items.items() if t.startswith("Classifier for")), None)
    mod = next((i for i, t in items.items() if t.startswith("Modality reliability")), None)
    if clf:
        decision = "present" if "calls it PRESENT" in items[clf] else "absent"
        cites = [clf]
    elif mod:
        decision = "present" if "image leans PRESENT" in items[mod] else "absent"
        cites = [mod]
    else:
        return json.dumps({"decision": "cannot_decide", "claims": []})
    claims = [{"id": "C1", "text": "The main signal points this way.", "cites": cites}]
    if mod and mod not in cites:
        claims.append({"id": "C2", "text": "Modality information is consistent.", "cites": [mod]})
    return json.dumps({"decision": decision, "claims": claims})
