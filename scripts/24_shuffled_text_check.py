"""Leakage sanity check (Phase 2.6): text-only model with the texts shuffled across samples vs correct texts.

Trains (or resumes) text_seed{k}_shuffled_text, where every sample gets the whole text of another sample,
and compares val AUROC with text_seed{k} (from 20_train). Shuffled should be near chance. If the correctly
assigned text-only model exceeds sanity.suspicious_auroc on a common finding, STOP: exit 1 and re-audit the
loader (pre-read text can mention findings, but not to that degree).
Writes outputs/<profile>/tables/shuffled_text_check.csv.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from mfsi_cxr.models.train import ckpt_dir, run_name, train_model
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    seed, scfg = cfg["train"]["seeds"][0], cfg["sanity"]
    shuffled = train_model(cfg, "text", seed, resume=True, text_shuffle_seed=scfg["shuffle_seed"])
    correct = json.loads((ckpt_dir(cfg, run_name("text", seed)) / "metrics.json").read_text(encoding="utf-8"))
    table = pd.DataFrame(correct["val_auroc"])[["finding", "auroc", "n_pos", "n_neg"]].rename(
        columns={"auroc": "auroc_correct_text"})
    table["auroc_shuffled_text"] = [r["auroc"] for r in shuffled["val_auroc"]]
    table["common"] = table["finding"].isin(scfg["common_findings"])
    enough = (table["n_pos"] >= scfg["min_class_count"]) & (table["n_neg"] >= scfg["min_class_count"])
    table["judged"] = table["common"] & enough
    table["suspicious"] = table["judged"] & (table["auroc_correct_text"] > scfg["suspicious_auroc"])
    out = output_dir(cfg) / "tables" / "shuffled_text_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    write_meta(out, cfg, seed)
    print(table.round(3).to_string(index=False))
    print(f"mean val AUROC: correct text {table['auroc_correct_text'].mean():.3f}, "
          f"shuffled text {table['auroc_shuffled_text'].mean():.3f}")
    if not table["judged"].any():
        print(f"no common finding has >= {scfg['min_class_count']} val positives and negatives: "
              "the suspicious-AUROC check could not be judged")
    if table["suspicious"].any():
        bad = table.loc[table["suspicious"], "finding"].tolist()
        print(f"STOP: text-only AUROC > {scfg['suspicious_auroc']} on {bad}; re-audit the loader for leakage")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
