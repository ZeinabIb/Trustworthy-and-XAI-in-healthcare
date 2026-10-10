"""Build text_inputs.parquet: leakage-safe index text + prior study refs per (patient, index study)."""

from __future__ import annotations

import sys

from mfsi_cxr.data.pipeline import build_text_inputs
from mfsi_cxr.utils.cli import base_parser, config_from_args


def main(argv: list[str] | None = None) -> int:
    build_text_inputs(config_from_args(base_parser(__doc__).parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
