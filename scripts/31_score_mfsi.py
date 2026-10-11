"""Score MFSI for val and test with the fitted parameters (Phase 3).

Writes artifacts/<profile>/mfsi_{split}_seed{k}.parquet. Test is written for Phase 4's final run, never read.
"""

from __future__ import annotations

import json
import sys

from mfsi_cxr.mfsi.fit import load_split, score_split
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import artifact_dir, check_artifact, write_table

SPLITS_SCORED = ("val", "test")


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    for seed in cfg["train"]["seeds"]:
        path = artifact_dir(cfg) / f"mfsi_params_seed{seed}.json"
        check_artifact(path, cfg)
        params = json.loads(path.read_text(encoding="utf-8"))
        train = load_split(cfg, "train", seed)
        for split in SPLITS_SCORED:
            table = score_split(train, load_split(cfg, split, seed), params, cfg)
            out = artifact_dir(cfg) / f"mfsi_{split}_seed{seed}.parquet"
            write_table(table, out, cfg, seed, split=split)
            print(f"ok {out.name}: {len(table)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
