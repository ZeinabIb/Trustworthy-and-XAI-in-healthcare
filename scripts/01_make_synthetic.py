"""Write synthetic raw data (with leakage canaries) into the smoke data root. Smoke profile only."""

from __future__ import annotations

import sys

from mfsi_cxr.data.synthetic import make_synthetic_dataset
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.config import data_root


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__).parse_args(argv))
    if cfg["profile"] != "smoke":
        print("01_make_synthetic: refusing to write synthetic data outside the smoke profile")
        return 1
    written = make_synthetic_dataset(data_root(cfg), cfg)
    print(f"synthetic data written: {', '.join(p.name for p in written.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
