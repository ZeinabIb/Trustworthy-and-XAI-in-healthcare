"""Real-data leakage audit (Phase 1.5); complements the synthetic canary test (tests/test_leakage.py).

This is an AUDIT, not part of the text branch: it reads every report column of the raw metadata so it
can check that the built text inputs and RadGraph facts never contain the index study's forbidden text or
text of later studies. Nothing it reads is written out or passed to a model; it returns counts only.

A snippet is a sentence-like piece of a report (split at . ! ? ; and newlines), case-folded with
whitespace collapsed, at least `data.leakage_check.min_snippet_chars` long. A hit is a snippet found in a
built text. A hit is a leak unless the snippet also occurs in text the sample may read (the index study's
allowed sections for the index text; strictly earlier studies for the prior texts). Those hits are shared
boilerplate and are counted separately.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mfsi_cxr.data.text_inputs import FORBIDDEN_INDEX_ALWAYS, IMAGE_PATH, TextBuilder, _tag

_SPLIT = re.compile(r"[.!?;\n]+")
_SPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9a-z]")
_COPY_SEP = "\x00"  # never inside a snippet, so a snippet cannot span two report copies


def normalise(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def _alnum(text: str) -> str:
    return _NON_ALNUM.sub("", text.casefold())


def snippets(text: str, min_chars: int) -> set[str]:
    pieces = (normalise(p) for p in _SPLIT.split(text))
    return {p for p in pieces if len(p) >= min_chars}


def report_columns(columns: Sequence[str]) -> list[str]:
    """Every report-text column of the metadata: all section_* columns and the full report."""
    return [c for c in columns if c.startswith("section_") or c in FORBIDDEN_INDEX_ALWAYS]


def metadata_columns(root: Path, cfg: dict[str, Any]) -> list[str]:
    from mfsi_cxr.data.schema import metadata_path

    path = metadata_path(root, cfg)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    return list(pd.read_csv(path, nrows=0).columns)


@dataclass(frozen=True)
class StudyText:
    allowed_index: frozenset[str]    # snippets of the index-allowed sections (all image copies)
    forbidden_index: frozenset[str]  # snippets of every other report column
    full: str                        # normalised text of all report columns, copies joined


def study_texts(meta: pd.DataFrame, index: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, StudyText]:
    """Per study, the snippets of all its report columns over all image copies of the report."""
    min_chars = cfg["data"]["leakage_check"]["min_snippet_chars"]
    allowed = set(cfg["data"]["text_sections"]["allowed_index"])
    cols = report_columns([c for c in meta.columns if c != IMAGE_PATH])
    meta = meta.assign(**{IMAGE_PATH: meta[IMAGE_PATH].astype(str).str.replace("\\", "/")})
    joined = meta.merge(index[[IMAGE_PATH, "study_id"]], on=IMAGE_PATH, how="inner")
    out: dict[str, StudyText] = {}
    for sid, g in joined.groupby("study_id"):
        ok: set[str] = set()
        bad: set[str] = set()
        full: list[str] = []
        for c in cols:
            for value in g[c].dropna().astype(str):
                (ok if c in allowed else bad).update(snippets(value, min_chars))
                full.append(normalise(value))
        out[str(sid)] = StudyText(frozenset(ok), frozenset(bad), _COPY_SEP.join(full))
    return out


@dataclass
class LeakageInputs:
    text_inputs: pd.DataFrame                  # the stored text_inputs.parquet
    index: pd.DataFrame                        # the stored index.parquet
    rebuilt_index_text: dict[str, str]         # index study id -> index text from a fresh build
    prior_texts: dict[str, tuple[str, ...]]    # index study id -> prior texts from a fresh build
    studies: dict[str, StudyText]
    report_cols: list[str]
    facts: pd.DataFrame                        # the stored radgraph_facts.parquet


@dataclass
class LeakageReport:
    """Sample-level counts (one sample = one index study), except the *_hits fields (snippets)."""

    n_samples: int = 0
    n_with_priors: int = 0
    index_text_mismatch: int = 0     # stored index text differs from a fresh build
    index_forbidden_tag: int = 0     # index text carries the tag of a non-allowed section
    index_forbidden_leak: int = 0    # index text holds forbidden text of its own study
    index_boilerplate_hits: int = 0  # forbidden snippets also present in the allowed sections
    prior_not_earlier: int = 0       # a prior is not a strictly earlier study of the same patient
    prior_too_many: int = 0          # more priors than data.max_priors
    prior_leak: int = 0              # prior texts hold index/later text absent from all earlier studies
    prior_boilerplate_hits: int = 0  # index/later snippets that also occur in earlier studies
    radgraph_not_prior: int = 0      # facts of a study that is no sample's prior
    radgraph_foreign_fact: int = 0   # facts whose text is not part of their own study's report (bad join)
    radgraph_leak: int = 0           # a sample's fact holds index/later text absent from earlier studies

    LEAK_FIELDS = ("index_text_mismatch", "index_forbidden_tag", "index_forbidden_leak",
                   "prior_not_earlier", "prior_too_many", "prior_leak", "radgraph_not_prior",
                   "radgraph_foreign_fact", "radgraph_leak")

    @property
    def n_leaks(self) -> int:
        return sum(getattr(self, f) for f in self.LEAK_FIELDS)

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "n_leaks": self.n_leaks}


def load_inputs(cfg: dict[str, Any]) -> LeakageInputs:
    from mfsi_cxr.data.schema import read_metadata
    from mfsi_cxr.utils.config import data_root
    from mfsi_cxr.utils.io import artifact_dir, read_table

    root, adir = data_root(cfg), artifact_dir(cfg)
    index = read_table(adir / "index.parquet", cfg)
    text_inputs = read_table(adir / "text_inputs.parquet", cfg)
    facts = read_table(adir / "radgraph_facts.parquet", cfg)
    builder = TextBuilder.from_root(root, index, cfg)
    built = {r.study_id: builder.build(r.patient_id, r.study_id) for r in text_inputs.itertuples()}

    path_col = cfg["data"]["columns"]["image_path"]
    cols = report_columns(metadata_columns(root, cfg))
    meta = read_metadata(root, cfg, columns=[path_col, *cols]).rename(columns={path_col: IMAGE_PATH})
    return LeakageInputs(
        text_inputs=text_inputs,
        index=index,
        rebuilt_index_text={sid: s.index_text for sid, s in built.items()},
        prior_texts={sid: s.prior_texts for sid, s in built.items()},
        studies=study_texts(meta, index, cfg),
        report_cols=cols,
        facts=facts,
    )


def check_leakage(inp: LeakageInputs, cfg: dict[str, Any]) -> LeakageReport:
    allowed = set(cfg["data"]["text_sections"]["allowed_index"])
    forbidden_tags = [f"[{_tag(c)}]" for c in inp.report_cols if c not in allowed]
    max_priors = cfg["data"]["max_priors"]
    empty = StudyText(frozenset(), frozenset(), "")

    studies = inp.index.drop_duplicates("study_id")
    rank: Mapping[str, int] = studies.set_index("study_id")["study_rank"].to_dict()
    patient_of: Mapping[str, str] = studies.set_index("study_id")["patient_id"].to_dict()
    by_patient = {pid: g["study_id"].tolist() for pid, g in studies.groupby("patient_id")}

    all_priors = {p for ids in inp.text_inputs["prior_study_ids"] for p in ids}
    fact_studies = set(inp.facts["study_id"])
    # Each fact is checked on its own: joining facts would glue words of different entities into
    # sentences that occur in no report (43 such false hits on the pilot, 2026-10-11).
    fact_text = {s: [normalise(t) for t in g] for s, g in inp.facts.groupby("study_id")["entity_text"]}

    rep = LeakageReport()
    rep.radgraph_not_prior = len(fact_studies - all_priors)
    for s, texts in fact_text.items():
        own = _alnum(inp.studies.get(s, empty).full)
        rep.radgraph_foreign_fact += sum(_alnum(t) not in own for t in texts)
    for row in inp.text_inputs.itertuples():
        sid, pid, r = row.study_id, row.patient_id, rank[row.study_id]
        own = inp.studies.get(sid, empty)
        rep.n_samples += 1

        # Index text: the stored one (what training reads) is checked, and must equal a fresh build.
        text = str(row.index_text)
        if text != inp.rebuilt_index_text.get(sid):
            rep.index_text_mismatch += 1
        if any(tag in text for tag in forbidden_tags):
            rep.index_forbidden_tag += 1
        norm = normalise(text)
        hits = {s for s in own.forbidden_index if s in norm}
        rep.index_boilerplate_hits += len(hits & own.allowed_index)
        if hits - own.allowed_index:
            rep.index_forbidden_leak += 1

        # Priors: structure from the stored references, content from a fresh build.
        priors = list(row.prior_study_ids)
        rep.n_with_priors += bool(priors)
        if any(patient_of.get(p) != pid or rank[p] >= r for p in priors):
            rep.prior_not_earlier += 1
        if len(priors) > max_priors:
            rep.prior_too_many += 1
        hay = normalise(" ".join(inp.prior_texts.get(sid, ())))
        sample_facts = [t for p in priors for t in fact_text.get(p, ())]
        if not hay and not sample_facts:
            continue
        own_patient = by_patient[pid]
        needles: set[str] = set()
        for s in own_patient:
            if rank[s] >= r:
                st = inp.studies.get(s, empty)
                needles |= st.allowed_index | st.forbidden_index
        earlier = _COPY_SEP.join(inp.studies.get(s, empty).full for s in own_patient if rank[s] < r)
        hits = {n for n in needles if n in hay}
        leaks = {h for h in hits if h not in earlier}
        rep.prior_boilerplate_hits += len(hits) - len(leaks)
        rep.prior_leak += bool(leaks)
        rep.radgraph_leak += any(n in t and n not in earlier for n in needles for t in sample_facts)
    return rep


def run_leakage_check(cfg: dict[str, Any]) -> LeakageReport:
    """Check the stored text inputs; write counts to outputs/<profile>/tables/leakage_check.json."""
    import json

    from mfsi_cxr.utils.io import output_dir, write_meta

    rep = check_leakage(load_inputs(cfg), cfg)
    out = output_dir(cfg) / "tables" / "leakage_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep.as_dict(), indent=2) + "\n", encoding="utf-8")
    write_meta(out, cfg)
    return rep
