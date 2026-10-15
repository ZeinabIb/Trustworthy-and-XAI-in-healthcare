"""The one-time RQ2 evaluation on TEST (Phase 6). The only script that runs the arbiter on test.

Refuses unless scripts/39_freeze.py froze configs, prompts, code and artifacts and nothing changed since; RQ2
opens the test split once (logged with purpose 'rq2' in artifacts/<profile>/test_opened.json; a rerun needs
--rerun-reason). Writes outputs/<profile>/tables/rq2_*.csv and figures/{flip_rates,cannot_decide_vs_R,
claims_verified}.png. On a non-full profile the outputs are throw-away (marked) and no metric is printed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mfsi_cxr.arbiter.llm_client import PinError
from mfsi_cxr.eval import rq2, runs
from mfsi_cxr.eval.freeze import FreezeError, check_frozen, record_opening
from mfsi_cxr.utils.cli import base_parser, config_from_args

SPLIT = "test"
PURPOSE = "rq2"


def _dev_figures():
    path = Path(__file__).with_name("60_arbiter_experiments_dev.py")
    spec = importlib.util.spec_from_file_location("dev60", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.figures


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--rerun-reason", default=None, help="required to open the test split again (logged)")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    try:
        access = check_frozen(cfg, args.rerun_reason, purpose=PURPOSE)
    except FreezeError as err:
        print(f"61_run_rq2_final: refused: {err}")
        return 1
    record_opening(cfg, access, args.rerun_reason, purpose=PURPOSE)
    try:
        t = rq2.run(cfg, SPLIT, access=access)
    except PinError as err:
        print(f"61_run_rq2_final: refused: {err}")
        return 1
    paths = runs.write_tables(cfg, t, "", SPLIT)
    figs = _dev_figures()(cfg, t, SPLIT, "")
    # No metric is printed: on the pilot the numbers are throw-away and must not be looked at.
    print(f"61_run_rq2_final ({cfg['profile']}): wrote {len(paths)} tables and {len(figs)} figures")
    for p in paths:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
