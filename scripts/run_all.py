"""Run every pipeline step in order for one profile; stop at the first failure.

Usage: python scripts/run_all.py --profile smoke
Steps are added here as each phase lands (10 -> ... -> 99).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mfsi_cxr.utils.cli import base_parser
from mfsi_cxr.utils.config import resolve_profile

SCRIPTS = Path(__file__).resolve().parent

ALL = ("smoke", "pilot", "full")
# (script, extra args, profiles it runs under). Each script except env_report also receives --profile.
STEPS: list[tuple[str, list[str], tuple[str, ...]]] = [
    ("env_report.py", [], ALL),
    ("01_make_synthetic.py", [], ("smoke",)),
    ("00_preflight.py", ["--stage", "data"], ALL),
    ("10_audit_data.py", [], ALL),
    ("11_build_index.py", [], ALL),
    ("12_make_splits.py", [], ALL),
    ("13_cache_images.py", [], ALL),
    ("14_build_text_inputs.py", [], ALL),
    ("15_build_radgraph_facts.py", [], ALL),
    ("16_check_leakage.py", [], ALL),
    ("20_train.py", [], ALL),
    ("21_extract.py", [], ALL),
    ("22_probe.py", [], ALL),
    ("23_masking_check.py", [], ALL),
    ("24_shuffled_text_check.py", [], ALL),
]


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    profile = resolve_profile(args.profile)
    steps = [(s, extra) for s, extra, profiles in STEPS if profile in profiles]
    for script, extra in steps:
        cmd = [sys.executable, str(SCRIPTS / script), *extra]
        if script != "env_report.py":
            cmd += ["--profile", profile]
        print(f"=== {script} {' '.join(extra)} ({profile})", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"run_all: {script} failed; stopping.")
            return 1
    print(f"run_all ({profile}): all {len(steps)} steps passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
