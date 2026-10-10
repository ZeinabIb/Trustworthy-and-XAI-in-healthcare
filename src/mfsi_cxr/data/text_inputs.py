"""Leakage-safe text inputs for the text branch (Phase 1.5). HARD RULE 1.

Index (current) study: ONLY the `allowed_index` sections + demographics. Prior studies (strictly earlier
in time): the `allowed_prior` sections. Everything is an allowlist: a column that is not listed is never
read, and the metadata is loaded with only the listed columns. The canary test (tests/test_leakage.py)
must stay green.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Never allowed for the index study, whatever the config says (labels come from these).
FORBIDDEN_INDEX_ALWAYS: frozenset[str] = frozenset(
    {
        "section_findings",
        "section_impression",
        "section_narrative",
        "section_summary",
        "section_end_of_impression",
        "report",
    }
)
IMAGE_PATH = "image_path"


class LeakageConfigError(ValueError):
    """The config or the caller tried to give the index text a column it may not read."""


@dataclass(frozen=True)
class TextSample:
    patient_id: str
    index_study_id: str
    index_text: str
    prior_study_ids: tuple[str, ...]
    prior_texts: tuple[str, ...]
    text_len: int  # whitespace tokens of the pre-read sections (0 when there are none)
    n_priors: int
    has_pre_read: bool


def validate_sections(cfg: dict[str, Any]) -> None:
    allowed = set(cfg["data"]["text_sections"]["allowed_index"])
    bad = allowed & FORBIDDEN_INDEX_ALWAYS
    if bad:
        raise LeakageConfigError(f"forbidden columns listed as allowed_index: {sorted(bad)}")


def index_columns(cfg: dict[str, Any]) -> list[str]:
    validate_sections(cfg)
    return [*cfg["data"]["text_sections"]["allowed_index"], *cfg["data"]["demographics"]]


def prior_columns(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["data"]["text_sections"]["allowed_prior"])


def _tag(column: str) -> str:
    return column.removeprefix("section_").upper()


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _join_sections(row: pd.Series, columns: list[str]) -> str:
    return " ".join(f"[{_tag(c)}] {_clean(row[c])}" for c in columns if _clean(row[c]))


def _demographics(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for c in columns:
        value = row[c]
        if _clean(value):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            parts.append(f"{c} {value}")
    return f"[DEMOGRAPHICS] {', '.join(parts)}" if parts else ""


def _per_study(meta: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Attach study_id via the image path and keep one row per study (sections are per report)."""
    paths = index[[IMAGE_PATH, "study_id"]]
    meta = meta.assign(**{IMAGE_PATH: meta[IMAGE_PATH].astype(str).str.replace("\\", "/")})
    return meta.merge(paths, on=IMAGE_PATH, how="inner").drop_duplicates("study_id").set_index("study_id")


class TextBuilder:
    """Builds TextSamples from separately loaded index-allowed and prior-allowed metadata columns."""

    def __init__(self, index: pd.DataFrame, index_meta: pd.DataFrame, prior_meta: pd.DataFrame,
                 cfg: dict[str, Any]) -> None:
        self.idx_cols = index_columns(cfg)
        self.sec_cols = list(cfg["data"]["text_sections"]["allowed_index"])
        self.demo_cols = list(cfg["data"]["demographics"])
        self.prior_cols = prior_columns(cfg)
        extra = set(index_meta.columns) - {IMAGE_PATH, *self.idx_cols}
        if extra:
            raise LeakageConfigError(f"index metadata carries non-allowlisted columns: {sorted(extra)}")
        extra = set(prior_meta.columns) - {IMAGE_PATH, *self.prior_cols}
        if extra:
            raise LeakageConfigError(f"prior metadata carries non-allowlisted columns: {sorted(extra)}")
        self.max_priors = cfg["data"]["max_priors"]
        self.no_text = cfg["data"]["no_pre_read_token"]
        self._index_rows = _per_study(index_meta, index)
        self._prior_rows = _per_study(prior_meta, index)
        studies = index.drop_duplicates("study_id")[["patient_id", "study_id", "study_rank"]]
        self._rank = studies.set_index("study_id")["study_rank"].to_dict()
        self._by_patient = {
            pid: g.sort_values("study_rank")["study_id"].tolist() for pid, g in studies.groupby("patient_id")
        }

    @classmethod
    def from_root(cls, root: Path, index: pd.DataFrame, cfg: dict[str, Any]) -> TextBuilder:
        """Load only allowlisted columns from the raw metadata (the IO-level guard)."""
        from mfsi_cxr.data.schema import read_metadata

        path_col = cfg["data"]["columns"]["image_path"]

        def load(columns: list[str]) -> pd.DataFrame:
            meta = read_metadata(root, cfg, columns=[path_col, *columns])
            return meta.rename(columns={path_col: IMAGE_PATH})

        return cls(index, load(index_columns(cfg)), load(prior_columns(cfg)), cfg)

    def build(self, patient_id: str, index_study_id: str) -> TextSample:
        if index_study_id not in self._by_patient.get(patient_id, []):
            raise KeyError(f"{index_study_id} is not a study of {patient_id}")
        row = self._index_rows.loc[index_study_id]
        sections = _join_sections(row, self.sec_cols)
        demo = _demographics(row, self.demo_cols)
        index_text = " ".join(t for t in (sections or self.no_text, demo) if t)

        rank = self._rank[index_study_id]
        earlier = [s for s in self._by_patient[patient_id] if self._rank[s] < rank]
        priors = list(reversed(earlier))[: self.max_priors]  # most recent first
        prior_texts = tuple(_join_sections(self._prior_rows.loc[s], self.prior_cols) for s in priors)
        return TextSample(
            patient_id=patient_id,
            index_study_id=index_study_id,
            index_text=index_text,
            prior_study_ids=tuple(priors),
            prior_texts=prior_texts,
            text_len=len(sections.split()),
            n_priors=len(priors),
            has_pre_read=bool(sections),
        )

    def build_all(self) -> pd.DataFrame:
        """One row per (patient, index study). Prior texts are stored as references (study ids)."""
        rows = []
        for pid, studies in sorted(self._by_patient.items()):
            for sid in studies:
                s = self.build(pid, sid)
                rows.append(
                    {
                        "patient_id": pid,
                        "study_id": sid,
                        "study_rank": self._rank[sid],
                        "index_text": s.index_text,
                        "prior_study_ids": list(s.prior_study_ids),
                        "n_priors": s.n_priors,
                        "text_len": s.text_len,
                        "has_pre_read": s.has_pre_read,
                    }
                )
        return pd.DataFrame(rows)
