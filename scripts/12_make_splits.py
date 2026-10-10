"""Make the patient-level split (splits.parquet) and the train label prevalence table."""

from __future__ import annotations

import sys

from mfsi_cxr.constants import SPLITS
from mfsi_cxr.data.labels import prevalence_table
from mfsi_cxr.data.splits import make_splits, radiologist_patients
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import artifact_dir, output_dir, read_table, write_meta, write_table
from mfsi_cxr.utils.logging import get_logger

log = get_logger("12_make_splits")
# Plan 07: log findings with fewer positives than this in train (never drop them).
LOW_POSITIVES = 5


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__).parse_args(argv))
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
