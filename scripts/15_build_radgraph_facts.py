"""Build radgraph_facts.parquet: RadGraph XL facts of the studies that are a prior of some sample.

Index studies never get their own facts (RadGraph exists only for findings/impression, both forbidden for
the index study). Needs index.parquet and text_inputs.parquet (scripts 11 and 14) and the RadGraph files
under <data.root>/<data.radgraph_dir> (09_fetch_raw --radgraph).
"""

from __future__ import annotations

import sys

from mfsi_cxr.data.radgraph import build_radgraph_facts
from mfsi_cxr.utils.cli import base_parser, config_from_args


def main(argv: list[str] | None = None) -> int:
    build_radgraph_facts(config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
