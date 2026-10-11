"""RadGraph XL facts (Phase 1.6) -> radgraph_facts.parquet: evidence for the arbiter (Phase 5).

RadGraph XL was run per section, on `radgraph.files` (section_findings, section_impression) only. Both are
forbidden for the index study (HARD RULE 1), so a sample only ever gets facts of its PRIOR studies:
radgraph_facts.parquet holds the facts of studies that are a prior of some sample, and facts_for_sample()
selects them by the sample's prior study ids.

The records carry no image or study id (each is {"0": {text, entities, ...}}, in no particular order), so
they are joined to a study by section text, compared alphanumerically and case-folded. A study's text is
taken from its first image by path, like the text branch. Texts whose records disagree are dropped.
Decisions: docs/DECISIONS.md, 2026-10-11.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from mfsi_cxr.constants import FINDINGS, NO_FINDING
from mfsi_cxr.data.text_inputs import IMAGE_PATH, _per_study
from mfsi_cxr.utils.logging import get_logger

log = get_logger("radgraph")

FACT_COLUMNS: tuple[str, ...] = (
    "patient_id", "study_id", "fact_id", "entity_text", "entity_type", "certainty", "measurement",
    "relation", "anatomy", "source_section", "findings",
)
_NON_ALNUM = re.compile(r"[^0-9a-z]")
_WORD = re.compile(r"[a-z0-9]+")

Entities = dict[str, dict[str, Any]]
Matcher = Callable[[str, str], list[str]]


def text_key(text: Any) -> str:
    """Join key between a RadGraph record and a metadata section: alphanumeric characters, case-folded."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return _NON_ALNUM.sub("", str(text).casefold())


def _signature(entities: Entities) -> tuple:
    def target(tid: str) -> str:
        e = entities.get(tid)
        return f"{e['label']}|{text_key(e['tokens'])}" if e else ""

    return tuple(sorted(
        (e["label"], text_key(e["tokens"]), tuple(sorted((r[0], target(r[1])) for r in e["relations"])))
        for e in entities.values()
    ))


def load_records(path: Path) -> tuple[dict[str, Entities], set[str]]:
    """Text key -> entities, without the keys whose records disagree (returned separately)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    entities: dict[str, Entities] = {}
    signature: dict[str, tuple] = {}
    conflicting: set[str] = set()
    for record in data:
        (rec,) = record.values()
        key = text_key(rec["text"])
        if not key:
            continue
        sig = _signature(rec["entities"])
        if key not in signature:
            signature[key], entities[key] = sig, rec["entities"]
        elif signature[key] != sig:
            conflicting.add(key)
    for key in conflicting:
        del entities[key]
    return entities, conflicting


def parse_label(label: str) -> tuple[str, str, bool]:
    """'Observation::measurement::definitely present' -> ('Observation', 'definitely present', True)."""
    parts = label.split("::")
    return parts[0], parts[-1], "measurement" in parts[1:-1]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def finding_matcher(cfg: dict[str, Any]) -> Matcher:
    """(entity tokens, located_at anatomy) -> findings, from configs/finding_synonyms.yaml."""
    rules = cfg["finding_synonyms"]
    bad = (set(rules) - set(FINDINGS)) | ({NO_FINDING} & set(rules))
    if bad:
        raise ValueError(f"finding_synonyms: not mappable findings {sorted(bad)}")
    compiled = [
        (finding, {w.casefold() for w in r["observation"]}, {w.casefold() for w in r.get("anatomy", [])})
        for finding in FINDINGS if finding in rules for r in rules[finding]
    ]

    def match(tokens: str, anatomy: str) -> list[str]:
        own, near = _words(tokens), _words(tokens) | _words(anatomy)
        out = [f for f, obs, anat in compiled if own & obs and (not anat or near & anat)]
        return list(dict.fromkeys(out))

    return match


def entity_facts(entities: Entities, patient_id: str, study_id: str, section: str,
                 match: Matcher) -> list[dict[str, Any]]:
    rows = []
    for eid, e in entities.items():
        etype, certainty, measurement = parse_label(e["label"])
        located = [entities[t]["tokens"] for r, t in e["relations"] if r == "located_at" and t in entities]
        anatomy = "; ".join(located)
        rows.append({
            "patient_id": patient_id,
            "study_id": study_id,
            "fact_id": f"{study_id}#{section}#{eid}",
            "entity_text": e["tokens"],
            "entity_type": etype,
            "certainty": certainty,
            "measurement": measurement,
            "relation": ";".join(sorted({r for r, _ in e["relations"]})),
            "anatomy": anatomy,
            "source_section": section,
            "findings": match(e["tokens"], anatomy) if etype == "Observation" else [],
        })
    return rows


def facts_for_sample(facts: pd.DataFrame, prior_study_ids: Iterable[str]) -> pd.DataFrame:
    """The facts a sample may use: those of its prior studies only (never its index study)."""
    return facts[facts["study_id"].isin(set(prior_study_ids))]


def extract_facts(meta: pd.DataFrame, index: pd.DataFrame, prior_ids: Iterable[str], root: Path,
                  cfg: dict[str, Any]) -> tuple[pd.DataFrame, Counter]:
    """Facts of the given prior studies. `meta` holds IMAGE_PATH and the RadGraph section columns."""
    rows = _per_study(meta, index)  # first image by path per study
    patient_of = index.drop_duplicates("study_id").set_index("study_id")["patient_id"]
    match = finding_matcher(cfg)
    facts: list[dict[str, Any]] = []
    stats: Counter = Counter()
    prior_ids = sorted(set(prior_ids))
    for section, name in cfg["radgraph"]["files"].items():
        records, conflicting = load_records(root / cfg["data"]["radgraph_dir"] / name)
        stats[f"{section}:texts_dropped_conflicting"] = len(conflicting)
        for sid in prior_ids:
            key = text_key(rows.at[sid, section]) if sid in rows.index else ""
            if not key:
                stats[f"{section}:empty"] += 1
            elif key in records:
                stats[f"{section}:matched"] += 1
                facts += entity_facts(records[key], patient_of[sid], sid, section, match)
            elif key in conflicting:
                stats[f"{section}:dropped_conflicting"] += 1
            else:
                stats[f"{section}:unmatched"] += 1
    return pd.DataFrame(facts, columns=list(FACT_COLUMNS)), stats


def build_radgraph_facts(cfg: dict[str, Any]) -> pd.DataFrame:
    """radgraph_facts.parquet for the studies that are a prior of some sample."""
    from mfsi_cxr.data.schema import read_metadata
    from mfsi_cxr.utils.config import data_root
    from mfsi_cxr.utils.io import artifact_dir, read_table, write_table

    root, adir = data_root(cfg), artifact_dir(cfg)
    index = read_table(adir / "index.parquet", cfg)
    text = read_table(adir / "text_inputs.parquet", cfg)
    prior_ids = {s for ids in text["prior_study_ids"] for s in ids}
    path_col = cfg["data"]["columns"]["image_path"]
    meta = read_metadata(root, cfg, columns=[path_col, *cfg["radgraph"]["files"]])
    facts, stats = extract_facts(meta.rename(columns={path_col: IMAGE_PATH}), index, prior_ids, root, cfg)
    write_table(facts, adir / "radgraph_facts.parquet", cfg)
    for section in cfg["radgraph"]["files"]:
        done = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith(f"{section}:")}
        log.info("%s: %s", section, done)
    mapped = facts["findings"].map(len).gt(0)
    log.info("prior studies=%d facts=%d mapped to a finding=%.1f%%", len(prior_ids), len(facts),
             100 * mapped.mean() if len(facts) else 0.0)
    return facts
