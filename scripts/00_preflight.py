"""Preflight: check data root, artifacts, canary test, disk and GPU before running a stage.

Usage: python scripts/00_preflight.py --profile smoke [--stage data|train]
Exit code 0 if every check passes (skips allowed), 1 otherwise.
"""

from __future__ import annotations

import sys

from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.config import ConfigError
from mfsi_cxr.utils.preflight import STAGES, passed, run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=STAGES, default="train")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except ConfigError as exc:
        print(f"[fail] config: {exc}")
        return 1
    results = run_preflight(cfg, args.stage)
    for r in results:
        print(f"[{r.status}] {r.name}: {r.detail}")
    ok = passed(results)
    print(f"preflight {args.stage} ({cfg['profile']}): {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
