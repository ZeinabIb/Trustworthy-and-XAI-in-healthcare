"""RQ2 arbiter experiments A / B / C on VAL (Phase 6, dev path). Never reads test.

Pre-stated patient sample (arbiter.experiments.n_min_cases), main pipeline + six packet conditions, cached and
resumable (artifacts/<profile>/arbiter/exp_val_seed{k}.jsonl). Writes outputs/<profile>/tables/rq2_dev_*.csv
and figures/{flip_rates,cannot_decide_vs_R,claims_verified}_dev.png; prints the pre-stated primary test.
"""

from __future__ import annotations

import sys

from mfsi_cxr.arbiter.llm_client import PinError
from mfsi_cxr.eval import rq2, runs
from mfsi_cxr.eval.figures import cannot_decide_figure, claims_verified_figure, flip_rates_figure
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import write_meta

SPLIT = "val"


def figures(cfg: dict, t: dict, split: str, suffix: str) -> list:
    flips = t["rq2_listen"]
    noise = flips.loc[flips["condition"] == "rerun", "flip_rate"]
    names = ("flip_rates", "cannot_decide_vs_R", "claims_verified")
    paths = [runs.fig_path(cfg, f"{n}{suffix}.png") for n in names]
    flip_rates_figure(flips, float(noise.iloc[0]) if len(noise) else 0.0, paths[0],
                      f"Arbiter decision flips by packet condition ({split})")
    if not t["rq2_cannot_decide_vs_R"].empty:
        cannot_decide_figure(t["rq2_cannot_decide_vs_R"], paths[1], f"cannot_decide by R shown ({split})")
    if not t["rq2_holdup_per_finding"].empty:
        claims_verified_figure(t["rq2_holdup_per_finding"], paths[2],
                               f"Verified claims per finding ({split})")
    out = [p for p in paths if p.exists()]
    for p in out:
        write_meta(p, cfg, None, split=split, throwaway=cfg["profile"] != "full")
    return out


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    try:
        t = rq2.run(cfg, SPLIT)
    except PinError as err:
        print(f"60_arbiter_experiments_dev: refused: {err}")
        return 1
    runs.write_tables(cfg, {k.replace("rq2_", ""): v for k, v in t.items()}, "rq2_dev_", SPLIT)
    figures(cfg, t, SPLIT, "_dev")
    print(t["rq2_sample"].T.to_string(header=False))
    print(t["rq2_listen"].round(3).to_string(index=False))
    print(t["rq2_listen_primary"].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
