"""Cache the selected images once, resized, as uint8 (artifacts/<profile>/img_cache/). Idempotent."""

from __future__ import annotations

import sys

from mfsi_cxr.data.images import build_image_cache
from mfsi_cxr.utils.cli import base_parser, config_from_args


def main(argv: list[str] | None = None) -> int:
    build_image_cache(config_from_args(base_parser(__doc__).parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
