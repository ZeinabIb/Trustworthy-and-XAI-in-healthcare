"""Download CheXpert Plus raw data from Redivis into the data root (after the agreement allows it).

Usage:
  python scripts/09_fetch_raw.py --profile pilot             label files + metadata table
  python scripts/09_fetch_raw.py --profile pilot --images    PNGs of the selected patients only
                                                             (needs artifacts/<profile>/index.parquet,
                                                             i.e. run 11_build_index first)
  python scripts/09_fetch_raw.py --profile pilot --radgraph  RadGraph XL files (all reports; ~850 MB)
--dest defaults to data.root (or MFSI_DATA_ROOT). The destination may not be inside the repo or in a
cloud-synced folder. Every step skips files already present, so an interrupted run can be restarted.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mfsi_cxr.data.fetch import UnsafeDestinationError, fetch_all, fetch_images, fetch_radgraph
from mfsi_cxr.data.images import select_study_images
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.config import data_root
from mfsi_cxr.utils.io import artifact_dir, read_table


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--labels-table", default=None, help="override fetch.labels_table")
    parser.add_argument("--metadata-table", default=None, help="override fetch.metadata_table")
    parser.add_argument("--images", action="store_true", help="download the selected patients' PNGs")
    parser.add_argument("--radgraph", action="store_true", help="download the RadGraph XL files")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if cfg["profile"] == "smoke":
        print("09_fetch_raw: the smoke profile uses synthetic data only")
        return 1
    dest = args.dest or data_root(cfg)
    if dest is None:
        print("09_fetch_raw: give --dest or set MFSI_DATA_ROOT")
        return 1
    try:
        if args.images:
            selected = select_study_images(read_table(artifact_dir(cfg) / "index.parquet", cfg), cfg)
            paths = selected["image_path"].tolist()
            print(f"{len(paths)} images selected ({selected['study_id'].nunique()} studies)")
            downloaded, skipped = fetch_images(cfg, dest, paths)
            print(f"ok images: {downloaded} downloaded, {skipped} already present")
            return 0
        if args.radgraph:
            for path in fetch_radgraph(cfg, dest):
                print(f"ok {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
            return 0
        metadata_ref = args.metadata_table or cfg["fetch"]["metadata_table"]
        written = fetch_all(cfg, dest, args.labels_table or cfg["fetch"]["labels_table"], metadata_ref)
    except UnsafeDestinationError as exc:
        print(f"09_fetch_raw: {exc}")
        return 1
    for path in written:
        print(f"ok {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    if not metadata_ref:
        print("metadata table skipped: set fetch.metadata_table in configs/data.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
