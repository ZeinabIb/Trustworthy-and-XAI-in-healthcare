"""Train the backbone: full model + image-only + text-only, every configured seed (Phase 2.3).

Refuses to start unless `00_preflight --stage train` passes. Restartable with the same command: a run with
a last.pt resumes from it (refused if the recipe changed; then pass --fresh). Writes
outputs/<profile>/tables/backbone_auroc.csv (val AUROC per run and finding; val only, test never read).
"""

from __future__ import annotations

import json
import shutil
import sys

from mfsi_cxr.models.mil import BRANCHES
from mfsi_cxr.models.train import auroc_summary, ckpt_dir, run_name, train_model
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta
from mfsi_cxr.utils.preflight import passed, run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--branch", choices=BRANCHES, action="append", help="default: train.branches")
    parser.add_argument("--seed", type=int, action="append", help="default: train.seeds")
    parser.add_argument("--fresh", action="store_true", help="discard existing checkpoints of these runs")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    results = run_preflight(cfg, "train")
    for r in results:
        print(f"[{r.status}] {r.name}: {r.detail}")
    if not passed(results):
        print("20_train: preflight failed; not training")
        return 1
    branches, seeds = args.branch or cfg["train"]["branches"], args.seed or cfg["train"]["seeds"]
    for branch in branches:
        for seed in seeds:
            folder = ckpt_dir(cfg, run_name(branch, seed))
            if args.fresh and folder.exists():
                shutil.rmtree(folder)
            s = train_model(cfg, branch, seed, resume=True)
            print(f"{s['run']}: best epoch {s['best_epoch']}, val score {s['val_score']:.4f}")

    summaries = []
    for branch in cfg["train"]["branches"]:
        for seed in cfg["train"]["seeds"]:
            metrics = ckpt_dir(cfg, run_name(branch, seed)) / "metrics.json"
            if metrics.exists():
                summaries.append(json.loads(metrics.read_text(encoding="utf-8")))
    out = output_dir(cfg) / "tables" / "backbone_auroc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = auroc_summary(summaries)
    table.to_csv(out, index=False)
    write_meta(out, cfg)
    print(table.pivot_table(index="finding", columns="run", values="auroc").round(3).to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
