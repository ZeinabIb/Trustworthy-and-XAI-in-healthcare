"""Audit the built text inputs for leakage on the real (or synthetic) data; counts only.

Checks the stored text_inputs.parquet: the index text holds no forbidden text of its own study and equals
a fresh build; priors are strictly earlier studies of the same patient, at most max_priors, and their texts
hold no index/later text that is absent from earlier studies. Writes
outputs/<profile>/tables/leakage_check.json and exits 1 on any leak.
"""

from __future__ import annotations

import sys

from mfsi_cxr.data.leakage_check import run_leakage_check
from mfsi_cxr.utils.cli import base_parser, config_from_args


def main(argv: list[str] | None = None) -> int:
    rep = run_leakage_check(config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv)))
    for key, value in rep.as_dict().items():
        print(f"{key:24s} {value}")
    if rep.n_leaks:
        print(f"15_check_leakage: {rep.n_leaks} leak(s) found")
        return 1
    print("15_check_leakage: no leaks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
