"""Text-informativeness probe on the full model's z_txt (and z_img as reference), per seed (Phase 2.5).

Fit on train, evaluated on val only. Writes artifacts/<profile>/probe_text_seed{k}.json (the
text_informative mask Phase 3 reads) and outputs/<profile>/tables/probe_seed{k}.csv.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from mfsi_cxr.models.extract import export_paths, load_export
from mfsi_cxr.models.probe import probe_table, text_informative_mask
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import artifact_dir, output_dir, write_meta


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    status = 0
    for seed in cfg["train"]["seeds"]:
        train = load_export(export_paths(cfg, "full", "train", seed)[0], cfg)
        val = load_export(export_paths(cfg, "full", "val", seed)[0], cfg)
        table = pd.concat([probe_table(train, val, key, cfg, seed) for key in ("z_txt", "z_img")])
        mask = text_informative_mask(table)
        out = artifact_dir(cfg) / f"probe_text_seed{seed}.json"
        payload = {**mask, "cutoff": cfg["probe"]["text_cutoff"]}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_meta(out, cfg, seed)
        csv = output_dir(cfg) / "tables" / f"probe_seed{seed}.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv, index=False)
        write_meta(csv, cfg, seed)
        print(table.pivot_table(index="finding", columns="embedding", values="auroc").round(3).to_string())
        informative = [f for f, v in mask["text_informative"].items() if v]
        print(f"seed {seed}: text informative for {len(informative)} findings: {informative}")
        if mask["no_finding_text_informative"]:
            print("WARNING: z_txt informative for no finding (the MFSI agreement term would be meaningless)")
    return status


if __name__ == "__main__":
    sys.exit(main())
