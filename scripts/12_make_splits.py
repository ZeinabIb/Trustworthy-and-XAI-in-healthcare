"""Make the patient-level split (splits.parquet) and the train label prevalence table."""

from __future__ import annotations

import sys

from mfsi_cxr.data.pipeline import build_splits
from mfsi_cxr.utils.cli import base_parser, config_from_args


def main(argv: list[str] | None = None) -> int:
    build_splits(config_from_args(base_parser(__doc__).parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
