"""Build index.parquet (one row per image, with study_rank) and labels.parquet for the patient subset.

The subset size comes from data.pilot_patients (null = all patients); the chosen patient list is saved
to artifacts/<profile>/pilot_patients.txt. Reads only non-text metadata columns.
"""

from __future__ import annotations

import sys

from mfsi_cxr.data.index import assign_study_rank, build_image_table, finalise_index, sample_patients
from mfsi_cxr.data.labels import labels_per_study, read_label_records
from mfsi_cxr.data.schema import read_metadata, structural_columns
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.config import data_root
from mfsi_cxr.utils.io import artifact_dir, write_meta, write_table
from mfsi_cxr.utils.logging import get_logger

log = get_logger("11_build_index")


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
