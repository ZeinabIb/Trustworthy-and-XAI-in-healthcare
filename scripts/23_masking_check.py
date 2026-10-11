"""Masking-based causal check + attention entropy of the full model, per seed (Phase 2 safeguards).

Removes the top-attended image (or prior report) of val samples and reports how often a probability moves
by more than masking_check.margin. Attention entropy per branch and level comes from the val attn exports.
Writes outputs/<profile>/tables/masking_check.csv and attention_entropy.csv. Val only.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from mfsi_cxr.data.datasets import SampleDataset
from mfsi_cxr.models.extract import export_paths, load_export
from mfsi_cxr.models.mil import masking_check
from mfsi_cxr.models.train import load_model, make_loader, to_device
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.device import device_from_config
from mfsi_cxr.utils.io import output_dir, write_meta


def _entropy(w: np.ndarray, mask: np.ndarray) -> float:
    rows = mask.sum(-1) >= 2
    if not rows.any():
        return float("nan")
    h = -(w * np.log(np.clip(w, 1e-12, None))).sum(-1)
    return float(h[rows].mean())


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    cfg["train"]["precision"] = "fp32"
    device, mcfg = device_from_config(cfg), cfg["masking_check"]
    tables, entropies = [], []
    for seed in cfg["train"]["seeds"]:
        model = load_model(cfg, "full", seed, device)
        ds = SampleDataset(cfg, "val")
        sub = Subset(ds, range(min(len(ds), mcfg["n_samples"])))
        batches = [to_device(b, device) for b in make_loader(sub, cfg, shuffle=False, seed=seed)]
        tables.append(masking_check(model, batches, mcfg["margin"]).assign(seed=seed))
        att = load_export(export_paths(cfg, "full", "val", seed)[1], cfg)
        for name, label in (("img", "image"), ("txt", "text")):
            entropies.append({"seed": seed, "branch": label,
                              "level1": _entropy(att[f"{name}_a"], att[f"{name}_a_mask"]),
                              "level2": _entropy(att[f"{name}_b"], att[f"{name}_b_mask"])})
    table, ent = pd.concat(tables, ignore_index=True), pd.DataFrame(entropies)
    folder = output_dir(cfg) / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for df, name in ((table, "masking_check.csv"), (ent, "attention_entropy.csv")):
        df.to_csv(folder / name, index=False)
        write_meta(folder / name, cfg)
    print(table.round(4).to_string(index=False))
    print(ent.round(4).to_string(index=False))
    low = table[table["share_changed"] < mcfg["low_share"]]
    for row in low.itertuples():
        print(f"NOTE: {row.branch} seed {row.seed}: masking the top instance changes only "
              f"{row.share_changed:.1%} of samples: the attention is not doing real work (report it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
