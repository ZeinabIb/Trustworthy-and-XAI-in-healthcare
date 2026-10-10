"""Audit the metadata table: dtype, null rate, class per column -> outputs/<profile>/DATA_SCHEMA.md.

No report text is written: text columns are summarised by length only.
"""

from __future__ import annotations

import sys

from mfsi_cxr.data.schema import audit, audit_markdown, metadata_path, read_metadata
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.config import data_root
from mfsi_cxr.utils.io import output_dir, write_meta


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    path = metadata_path(data_root(cfg), cfg)
    if not path.exists():
        print(f"10_audit_data: {path} not found; run scripts/09_fetch_raw.py first")
        return 1
    meta = read_metadata(data_root(cfg), cfg)
    table = audit(meta, cfg)
    out = output_dir(cfg) / "DATA_SCHEMA.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(audit_markdown(table, len(meta), cfg["profile"]), encoding="utf-8")
    write_meta(out, cfg)
    print(table[["column", "dtype", "null_rate", "class"]].to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
