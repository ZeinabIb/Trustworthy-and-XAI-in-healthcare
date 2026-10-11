"""RadGraph facts (Phase 1.6): text join, conflict drop, first-image rule, label parsing, finding mapping.

Leakage of facts (index/later canaries never reach a sample) is covered in test_leakage.py.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mfsi_cxr.data.radgraph import (
    FACT_COLUMNS,
    extract_facts,
    finding_matcher,
    load_records,
    parse_label,
    text_key,
)
from mfsi_cxr.data.schema import read_metadata
from mfsi_cxr.data.synthetic import radgraph_entities
from mfsi_cxr.data.text_inputs import IMAGE_PATH
from mfsi_cxr.utils.io import artifact_dir, read_table


def _record(text: str, entities: dict) -> dict:
    return {"0": {"text": text, "entities": entities, "data_source": None, "data_split": "inference"}}


def _meta(cfg: dict[str, Any]) -> pd.DataFrame:
    path_col = cfg["data"]["columns"]["image_path"]
    meta = read_metadata(Path(cfg["data"]["root"]), cfg, [path_col, *cfg["radgraph"]["files"]])
    return meta.rename(columns={path_col: IMAGE_PATH})


def _priors(cfg: dict[str, Any]) -> set[str]:
    text = read_table(artifact_dir(cfg) / "text_inputs.parquet", cfg)
    return {p for ids in text["prior_study_ids"] for p in ids}


def test_text_key_ignores_case_spacing_and_punctuation() -> None:
    assert text_key("No acute  process .") == text_key("no acute process") == "noacuteprocess"
    assert text_key(None) == text_key(float("nan")) == ""


def test_parse_label() -> None:
    assert parse_label("Observation::definitely absent") == ("Observation", "definitely absent", False)
    assert parse_label("Anatomy::measurement::definitely present") == ("Anatomy", "definitely present", True)


def test_conflicting_records_are_dropped_and_duplicates_kept(tmp_path: Path) -> None:
    other = radgraph_entities("x")
    other["2"]["label"] = "Observation::definitely absent"
    records = [_record("Same text.", radgraph_entities("x")), _record("same  text", radgraph_entities("x")),
               _record("Conflict.", radgraph_entities("x")), _record("conflict", other)]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    kept, conflicting = load_records(path)
    assert set(kept) == {"sametext"} and conflicting == {"conflict"}


@pytest.mark.parametrize(("tokens", "anatomy", "expected"), [
    ("effusion", "pleural", ["Pleural Effusion"]),
    ("effusions", "", ["Pleural Effusion"]),
    ("enlarged", "heart", ["Cardiomegaly"]),
    ("widened", "mediastinum", ["Enlarged Cardiomediastinum"]),
    ("nodule", "lung", ["Lung Lesion"]),
    ("tube", "", ["Support Devices"]),
    ("thickening", "", []),            # needs pleural anatomy
    ("thickening", "pleural", ["Pleural Other"]),
    ("normal", "heart", []),
])
def test_finding_mapping(smoke_cfg: dict[str, Any], tokens: str, anatomy: str, expected: list[str]) -> None:
    assert finding_matcher(smoke_cfg)(tokens, anatomy) == expected


def test_no_finding_cannot_be_mapped(smoke_cfg: dict[str, Any]) -> None:
    cfg = copy.deepcopy(smoke_cfg)
    cfg["finding_synonyms"]["No Finding"] = [{"observation": ["normal"]}]
    with pytest.raises(ValueError, match="No Finding"):
        finding_matcher(cfg)


def test_build_on_synthetic_artifacts(smoke_artifacts: dict[str, Any]) -> None:
    cfg = smoke_artifacts
    facts = read_table(artifact_dir(cfg) / "radgraph_facts.parquet", cfg)
    assert list(facts.columns) == list(FACT_COLUMNS)
    priors = _priors(cfg)
    assert set(facts["study_id"]) == priors  # every prior study matched, nothing else
    assert facts["fact_id"].is_unique
    assert set(facts["source_section"]) == set(cfg["radgraph"]["files"])
    effusion = facts[facts["entity_text"] == "effusion"]
    assert len(effusion) and all(list(f) == ["Pleural Effusion"] for f in effusion["findings"])
    assert set(effusion["anatomy"]) == {"pleural"} and set(effusion["relation"]) == {"located_at"}
    assert set(facts.loc[facts["entity_type"] == "Anatomy", "findings"].map(len)) == {0}


def test_unmatched_and_conflicting_texts_are_counted(smoke_artifacts: dict[str, Any]) -> None:
    cfg = smoke_artifacts
    root = Path(cfg["data"]["root"])
    index = read_table(artifact_dir(cfg) / "index.parquet", cfg)
    meta, priors = _meta(cfg), sorted(_priors(cfg))
    section = "section_findings"
    path = root / cfg["data"]["radgraph_dir"] / cfg["radgraph"]["files"][section]
    records = json.loads(path.read_text(encoding="utf-8"))
    by_study = meta.merge(index[[IMAGE_PATH, "study_id"]], on=IMAGE_PATH).groupby("study_id")[section].first()
    conflict_text, missing_text = by_study[priors[0]], by_study[priors[1]]
    changed = radgraph_entities(conflict_text)
    changed["2"]["label"] = "Observation::definitely absent"
    records = [r for r in records if text_key(r["0"]["text"]) != text_key(missing_text)]
    records.append(_record(conflict_text, changed))
    path.write_text(json.dumps(records), encoding="utf-8")

    facts, stats = extract_facts(meta, index, priors, root, cfg)
    assert stats[f"{section}:dropped_conflicting"] == 1 and stats[f"{section}:unmatched"] == 1
    assert stats[f"{section}:matched"] == len(priors) - 2
    found = set(facts.loc[facts["source_section"] == section, "study_id"])
    assert priors[0] not in found and priors[1] not in found


def test_study_text_comes_from_first_image_by_path(smoke_artifacts: dict[str, Any]) -> None:
    cfg = smoke_artifacts
    root = Path(cfg["data"]["root"])
    index = read_table(artifact_dir(cfg) / "index.parquet", cfg)
    meta, priors = _meta(cfg), sorted(_priors(cfg))
    multi = index[index["study_id"].isin(priors)].groupby("study_id")[IMAGE_PATH].agg(sorted)
    multi = multi[multi.map(len) > 1]
    assert len(multi), "synthetic data must contain a prior study with several images"
    paths = multi.iloc[0]
    meta.loc[meta[IMAGE_PATH].isin(paths[1:]), "section_findings"] = "a text with no record at all"
    _, stats = extract_facts(meta, index, priors, root, cfg)
    assert stats["section_findings:unmatched"] == 0  # later copies are ignored
    meta.loc[meta[IMAGE_PATH] == paths[0], "section_findings"] = "a text with no record at all"
    _, stats = extract_facts(meta, index, priors, root, cfg)
    assert stats["section_findings:unmatched"] == 1

