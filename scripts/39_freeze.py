"""Freeze configs, code and the frozen artifacts before the one-time test evaluation (Phase 4).

Writes artifacts/<profile>/rq1_frozen.json (commit, config-file hashes, hashes of the MFSI params, probes,
MFSI tables and embeddings). Refuses while configs/, src/ or scripts/ have uncommitted changes. Run it after
the dev scripts (30-45) and before scripts/41_run_rq1_final.py.
"""

from __future__ import annotations

import sys

from mfsi_cxr.eval.freeze import FreezeError, write_manifest
from mfsi_cxr.utils.cli import base_parser, config_from_args


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    try:
        path = write_manifest(cfg)
    except FreezeError as err:
        print(f"39_freeze: refused: {err}")
        return 1
    print(f"frozen: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
