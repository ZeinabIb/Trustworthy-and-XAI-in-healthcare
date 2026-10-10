"""Print Python, platform, PyTorch, CUDA and pinned package versions as JSON."""

from __future__ import annotations

import json
import sys

from mfsi_cxr.utils.logging import env_report, git_info


def main() -> int:
    print(json.dumps({**env_report(), "git": git_info()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
