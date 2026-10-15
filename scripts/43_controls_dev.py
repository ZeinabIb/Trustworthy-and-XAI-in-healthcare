"""Validity controls on VAL (Phase 4 part D, dev path; out-of-fold y_hat for correctness, flags, escalation).

Per seed: agreement spread + modality dominance, data-volume confound (with rank-normalised R where a
correlation is strong), hierarchy flags, accuracy within deciles of R, and the uncertain-label check. Writes
outputs/<profile>/tables/controls_dev_*.csv and figures/{volume_confound,r_deciles}_dev.png. Never reads test.
The degrade-one-modality test is scripts/45_degrade_dev.py (it re-extracts embeddings).
"""

from __future__ import annotations

import sys

from mfsi_cxr.eval import pipeline, runs
from mfsi_cxr.utils.cli import base_parser, config_from_args

SPLIT = "val"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    c = cfg["eval"]["controls"]
    d = pipeline.collect(cfg, SPLIT, list(cfg["train"]["seeds"]), oof=True)
    t = runs.controls(cfg, d)
    runs.write_tables(cfg, t, "controls_dev_", SPLIT)
    runs.controls_figures(cfg, t, SPLIT, d.seeds[0], "_dev")
    corr = t["volume_correlation"]
    for seed, g in corr.groupby("seed"):
        strong = sorted(g.loc[g["spearman_R"].abs() >= c["confound_strong"], "variable"].unique())
        print(f"seed {seed}: strong volume correlations (|rho| >= {c['confound_strong']}):", strong or "none")
    print(t["uncertain_macro"].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
