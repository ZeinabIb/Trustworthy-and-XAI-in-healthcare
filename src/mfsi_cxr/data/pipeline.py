"""Phase 1 build steps as functions. The numbered scripts are thin CLIs over these, and tests call them
directly with a temp-dir config."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mfsi_cxr.constants import SPLITS
from mfsi_cxr.data.index import assign_study_rank, build_image_table, finalise_index, sample_patients
from mfsi_cxr.data.labels import labels_per_study, prevalence_table, read_label_records
from mfsi_cxr.data.schema import read_metadata, structural_columns
from mfsi_cxr.data.splits import make_splits, radiologist_patients
from mfsi_cxr.data.text_inputs import TextBuilder
from mfsi_cxr.utils.config import data_root
from mfsi_cxr.utils.io import artifact_dir, output_dir, read_table, write_meta, write_table
from mfsi_cxr.utils.logging import get_logger

log = get_logger("pipeline")
# Plan 07: log findings with fewer positives than this in train (never drop them).
LOW_POSITIVES = 5


def build_index(cfg: dict[str, Any]) -> pd.DataFrame:
    """index.parquet + labels.parquet + pilot_patients.txt for the configured patient subset."""
    root, dcfg = data_root(cfg), cfg["data"]
    meta = read_metadata(root, cfg, columns=structural_columns(cfg))
    images = assign_study_rank(build_image_table(meta, cfg))
    records = read_label_records(root / dcfg["labels_file"])
    labels = labels_per_study(records, images, dcfg["columns"]["image_path"])

    chosen = sample_patients(images, labels, dcfg["pilot_patients"], dcfg["split_seed"],
                             dcfg["pilot_min_pos_per_finding"])
    index = finalise_index(images[images["patient_id"].isin(chosen)])
    labels = labels[labels["study_id"].isin(set(index["study_id"]))].reset_index(drop=True)

    adir = artifact_dir(cfg)
    write_table(index, adir / "index.parquet", cfg)
    write_table(labels, adir / "labels.parquet", cfg)
    patients_file = adir / "pilot_patients.txt"
    patients_file.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    write_meta(patients_file, cfg)
    n_unlabelled = index["study_id"].nunique() - len(labels)
    if n_unlabelled:
        log.warning("%d studies have no label record", n_unlabelled)
    log.info("patients=%d studies=%d images=%d", len(chosen), index["study_id"].nunique(), len(index))
    return index


def build_splits(cfg: dict[str, Any]) -> pd.DataFrame:
    """splits.parquet + outputs/<profile>/tables/label_prevalence.csv (train)."""
    adir, dcfg = artifact_dir(cfg), cfg["data"]
    index = read_table(adir / "index.parquet", cfg)
    labels = read_table(adir / "labels.parquet", cfg)
    splits = make_splits(
        sorted(index["patient_id"].unique()),
        radiologist_patients(index, cfg),
        dcfg["split_fractions"],
        dcfg["split_seed"],
        dcfg["radiologist_policy"],
    )
    write_table(splits, adir / "splits.parquet", cfg, seed=dcfg["split_seed"])
    log.info("split sizes: %s", splits["split"].value_counts().to_dict())

    table = prevalence_table(labels, index, splits, SPLITS[0])
    out = output_dir(cfg) / "tables" / "label_prevalence.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    write_meta(out, cfg)
    for row in table.itertuples():
        if row.n_pos < LOW_POSITIVES:
            log.warning("finding %r has only %d positives in train", row.finding, row.n_pos)
    return splits


def build_text_inputs(cfg: dict[str, Any]) -> pd.DataFrame:
    """text_inputs.parquet: leakage-safe index text + prior study refs per (patient, index study)."""
    adir = artifact_dir(cfg)
    index = read_table(adir / "index.parquet", cfg)
    table = TextBuilder.from_root(data_root(cfg), index, cfg).build_all()
    write_table(table, adir / "text_inputs.parquet", cfg)
    log.info("samples=%d no_pre_read=%d mean_text_len=%.1f mean_priors=%.2f", len(table),
             int((~table["has_pre_read"]).sum()), table["text_len"].mean(), table["n_priors"].mean())
    return table
