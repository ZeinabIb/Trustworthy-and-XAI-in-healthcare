"""Build text_inputs.parquet: leakage-safe index text + prior study refs per (patient, index study)."""

from __future__ import annotations

import sys

from mfsi_cxr.data.text_inputs import TextBuilder
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.config import data_root
from mfsi_cxr.utils.io import artifact_dir, read_table, write_table
from mfsi_cxr.utils.logging import get_logger

log = get_logger("14_build_text_inputs")


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__).parse_args(argv))
    adir = artifact_dir(cfg)
    index = read_table(adir / "index.parquet", cfg)
    table = TextBuilder.from_root(data_root(cfg), index, cfg).build_all()
    write_table(table, adir / "text_inputs.parquet", cfg)
    log.info("samples=%d no_pre_read=%d mean_text_len=%.1f mean_priors=%.2f", len(table),
             int((~table["has_pre_read"]).sum()), table["text_len"].mean(), table["n_priors"].mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
