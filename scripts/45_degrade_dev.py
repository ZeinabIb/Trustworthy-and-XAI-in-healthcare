"""Degrade-one-modality test on VAL (Phase 4 control 3, dev path; face validity only).

For each seed: re-extract val embeddings through the frozen full model (fp32) once clean (checked against the
Phase 2 export) and once per degradation (configs/eval.yaml `eval.degrade`), score MFSI with the frozen
parameters, and report mean r_img, r_txt, A_cos and R before / after with patient-bootstrap CIs. Writes
artifacts/<profile>/degrade/emb_val_seed{k}_{variant}.npz, outputs/<profile>/tables/
controls_dev_degrade{,_per_finding}.csv and figures/degrade_test_dev.png. Never reads test.
"""

from __future__ import annotations

import sys

from mfsi_cxr.eval import degrade as dg
from mfsi_cxr.eval import runs
from mfsi_cxr.eval.figures import degrade_figure
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import write_meta

SPLIT = "val"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    summary, per_finding = dg.run(cfg, SPLIT)
    runs.write_tables(cfg, {"degrade": summary, "degrade_per_finding": per_finding}, "controls_dev_", SPLIT)
    first = cfg["train"]["seeds"][0]
    fig = runs.fig_path(cfg, "degrade_test_dev.png")
    degrade_figure(summary[summary["seed"] == first], fig, f"Degrade one modality on {SPLIT} (seed {first})")
    write_meta(fig, cfg, None, split=SPLIT)
    wide = summary.pivot_table(index=["variant", "modality", "own_margin_falls", "other_margin_unchanged"],
                               columns="metric", values="change", aggfunc="first", sort=False)
    print(wide.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
