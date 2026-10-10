"""Build index.parquet (one row per image, with study_rank) and labels.parquet for the patient subset.

The subset size comes from data.pilot_patients (null = all patients); the chosen patient list is saved
to artifacts/<profile>/pilot_patients.txt. Reads only non-text metadata columns.
"""

from __future__ import annotations

import sys

from mfsi_cxr.data.pipeline import build_index
from mfsi_cxr.utils.cli import base_parser, config_from_args


def main(argv: list[str] | None = None) -> int:
    build_index(config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
