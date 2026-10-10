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

# (script, extra args). Each script also receives --profile.
STEPS: list[tuple[str, list[str]]] = [
    ("env_report.py", []),
    ("00_preflight.py", ["--stage", "data"]),
]


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    profile = resolve_profile(args.profile)
    for script, extra in STEPS:
        cmd = [sys.executable, str(SCRIPTS / script), *extra]
        if script != "env_report.py":
            cmd += ["--profile", profile]
        print(f"=== {script} {' '.join(extra)} ({profile})", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"run_all: {script} failed; stopping.")
            return 1
    print(f"run_all ({profile}): all {len(STEPS)} steps passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
