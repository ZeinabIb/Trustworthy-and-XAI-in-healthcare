"""Build evidence packets for escalated cases (Phase 5).

Seeded sample of arbiter.n_cases escalated (sample, finding) rows of the VAL MFSI table (first seed),
packet variant from arbiter.packet. Writes artifacts/<profile>/arbiter/packets_val_seed{k}.jsonl (patient-
derived text: gitignored, never shared). Prints counts only.
"""

from __future__ import annotations

import json
import sys

from mfsi_cxr.arbiter.build import build_all
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import write_meta

SPLIT = "val"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    seed = cfg["train"]["seeds"][0]
    path, packets = build_all(cfg, SPLIT, seed)
    with path.open("w", encoding="utf-8") as f:
        for p in packets:
            f.write(json.dumps(p) + "\n")
    write_meta(path, cfg, seed, split=SPLIT, n=len(packets), variant=cfg["arbiter"]["packet"])
    n_items = [len(p["items"]) for p in packets]
    print(f"ok {path.name}: {len(packets)} packets, items per packet {min(n_items, default=0)}-"
          f"{max(n_items, default=0)}, findings {len({p['finding'] for p in packets})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
