#!/usr/bin/env bash
# Run the whole ordered pipeline for one profile; stops at the first failure.
# Usage: scripts/run_all.sh --profile {smoke,pilot,full}
# The step list lives in scripts/run_all.py so Windows and Linux run the same thing.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/run_all.py "$@"
