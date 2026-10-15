"""Prompt and output schema (Phase 5). The system prompt is a versioned, hashed file (configs/prompts/); the
user message is the finding plus the rendered evidence packet. The answer is UNTRUSTED: it is parsed strictly
(pydantic) and every cited id must exist in the packet."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from mfsi_cxr.arbiter.packet import Packet
from mfsi_cxr.utils.config import REPO_ROOT

DECISIONS = ("present", "absent", "cannot_decide")


class Claim(BaseModel):
    id: str = Field(pattern=r"^C[1-9][0-9]*$")
    text: str = Field(min_length=1, max_length=400)
    cites: list[str] = Field(min_length=1)

    @field_validator("cites")
    @classmethod
    def _ids(cls, v: list[str]) -> list[str]:
        for c in v:
            if not re.fullmatch(r"E[1-9][0-9]*", c):
                raise ValueError(f"not an evidence id: {c!r}")
        return v


class Answer(BaseModel):
    decision: Literal["present", "absent", "cannot_decide"]
    claims: list[Claim] = Field(default_factory=list)


class InvalidAnswerError(ValueError):
    """The model's output did not satisfy the schema or cited ids outside the packet."""


def load_prompt(cfg: dict[str, Any], key: str = "prompt_file") -> str:
    return (REPO_ROOT / cfg["arbiter"][key]).read_text(encoding="utf-8")


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def messages(cfg: dict[str, Any], packet: Packet) -> list[dict[str, str]]:
    user = f"Finding: {packet.finding}\nEvidence:\n{packet.render()}\n\nAnswer with JSON only."
    return [{"role": "system", "content": load_prompt(cfg)}, {"role": "user", "content": user}]


def repair_messages(cfg: dict[str, Any], base: list[dict[str, str]], raw: str,
                    error: str) -> list[dict[str, str]]:
    repair = load_prompt(cfg, "repair_file").replace("{error}", error[:300])
    return [*base, {"role": "assistant", "content": raw}, {"role": "user", "content": repair}]


def _extract_json(raw: str) -> str:
    """The first {...} block (models sometimes wrap JSON in prose or code fences)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise InvalidAnswerError("no JSON object in the answer")
    return raw[start:end + 1]


def parse(raw: str, packet: Packet, max_claims: int) -> Answer:
    try:
        answer = Answer.model_validate(json.loads(_extract_json(raw)))
    except (json.JSONDecodeError, ValidationError) as err:
        raise InvalidAnswerError(str(err).splitlines()[0]) from err
    if len(answer.claims) > max_claims:
        raise InvalidAnswerError(f"{len(answer.claims)} claims > {max_claims}")
    known = set(packet.ids)
    for c in answer.claims:
        unknown = [x for x in c.cites if x not in known]
        if unknown:
            raise InvalidAnswerError(f"claim {c.id} cites ids not in the evidence list: {unknown}")
    if len({c.id for c in answer.claims}) != len(answer.claims):
        raise InvalidAnswerError("duplicate claim ids")
    return answer


def file_digest(path: Path) -> str:
    return prompt_hash(path.read_text(encoding="utf-8"))
