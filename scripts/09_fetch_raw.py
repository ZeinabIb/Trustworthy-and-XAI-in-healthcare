"""Download the CheXpert Plus label files and metadata table from Redivis into the data root.

Only after the data use agreement allows the destination. No images are downloaded.
Usage:
  python scripts/09_fetch_raw.py --profile pilot --dest D:/chexpert_plus [--metadata-table <ref>]
--dest defaults to data.root (or MFSI_DATA_ROOT). The destination may not be inside the repo or in a
cloud-synced folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mfsi_cxr.data.fetch import UnsafeDestinationError, fetch_all
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.config import data_root


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--labels-table", default=None, help="override fetch.labels_table")
    parser.add_argument("--metadata-table", default=None, help="override fetch.metadata_table")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if cfg["profile"] == "smoke":
        print("09_fetch_raw: the smoke profile uses synthetic data only")
        return 1
    dest = args.dest or data_root(cfg)
    if dest is None:
        print("09_fetch_raw: give --dest or set MFSI_DATA_ROOT")
        return 1
    metadata_ref = args.metadata_table or cfg["fetch"]["metadata_table"]
    try:
        written = fetch_all(cfg, dest, args.labels_table or cfg["fetch"]["labels_table"], metadata_ref)
    except UnsafeDestinationError as exc:
        print(f"09_fetch_raw: {exc}")
        return 1
    for path in written:
        print(f"ok {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    if not metadata_ref:
        print("metadata table skipped: pass --metadata-table <ref> (copy it from the table's Redivis page)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
