"""Export frozen embeddings and attention for train/val/test x branches x seeds (Phase 2.4).

Reads each run's best.pt. Writes artifacts/<profile>/emb_{split}_seed{k}[_branch].npz and attn_*.npz.
No metric is computed (the test split is exported, not opened).
"""

from __future__ import annotations

import sys

from mfsi_cxr.constants import SPLITS
from mfsi_cxr.models.extract import extract
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.device import device_from_config


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    device = device_from_config(cfg)
    for branch in cfg["train"]["branches"]:
        for seed in cfg["train"]["seeds"]:
            for split in SPLITS:
                emb, _ = extract(cfg, branch, seed, split, device)
                print(f"ok {emb.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
