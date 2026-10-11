"""FIRST EXPERIMENT of Phases 3-4: spread of the agreement term A_cos on val, per finding (Phase 3).

Reports mean, sd, 5th/95th percentile of A_cos, corr(A, min margin) and corr(p_img+, p_txt+) per finding.
Flags an inert agreement term (sd < mfsi.inert_sd for most findings) and a possible shared-space artefact
(corr(p_img+, p_txt+) > mfsi.shared_space_corr); both are STOP-AND-ASK points. Val only. Writes
outputs/<profile>/tables/agreement_spread_seed{k}.csv and figures/agreement_spread_seed{k}.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import artifact_dir, output_dir, read_table, write_meta

SERIES, SURFACE, INK, MUTED = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e"  # dataviz reference palette


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def spread_table(val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FINDINGS:
        g = val[val["finding"] == f]
        a = g["A_cos"].to_numpy()
        rows.append({"finding": f, "stratum": g["stratum"].iloc[0], "n": len(g), "mean": a.mean(),
                     "sd": a.std(ddof=1), "p5": np.quantile(a, 0.05), "p95": np.quantile(a, 0.95),
                     "corr_A_min_margin": _corr(a, np.minimum(g["r_img"], g["r_txt"]).to_numpy()),
                     "corr_p_img_p_txt": _corr(g["p_img_pos"].to_numpy(), g["p_txt_pos"].to_numpy())})
    return pd.DataFrame(rows)


def plot(val: pd.DataFrame, table: pd.DataFrame, path: Path) -> None:
    """Small multiples: one histogram of A_cos per finding on a shared 0-1 axis; single series, no legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(15, 8), sharex=True, facecolor=SURFACE)
    bins = np.linspace(0, 1, 41)
    stats = table.set_index("finding")
    for ax, f in zip(axes.flat, FINDINGS, strict=False):
        ax.set_facecolor(SURFACE)
        ax.hist(val.loc[val["finding"] == f, "A_cos"], bins=bins, color=SERIES, edgecolor=SURFACE,
                linewidth=1.0)
        ax.set_title(f"{f}\nmean {stats.loc[f, 'mean']:.3f}  sd {stats.loc[f, 'sd']:.3f}", fontsize=9,
                     color=INK)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    for ax in list(axes.flat)[len(FINDINGS):]:
        ax.set_visible(False)
    fig.supxlabel("A_cos (agreement of the image and text profiles), val", color=INK)
    fig.supylabel("samples", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    m = cfg["mfsi"]
    for seed in cfg["train"]["seeds"]:
        val = read_table(artifact_dir(cfg) / f"mfsi_val_seed{seed}.parquet", cfg)
        table = spread_table(val)
        tables, figures = output_dir(cfg) / "tables", output_dir(cfg) / "figures"
        tables.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        csv = tables / f"agreement_spread_seed{seed}.csv"
        table.to_csv(csv, index=False)
        write_meta(csv, cfg, seed)
        png = figures / f"agreement_spread_seed{seed}.png"
        plot(val, table, png)
        write_meta(png, cfg, seed)
        print(table.round(3).to_string(index=False))
        inert = (table["sd"] < m["inert_sd"]).mean()
        if inert > 0.5:
            print(f"STOP-AND-ASK: A_cos sd < {m['inert_sd']} for {inert:.0%} of findings "
                  "(agreement term inert)")
        shared = table.loc[table["corr_p_img_p_txt"] > m["shared_space_corr"], "finding"].tolist()
        if shared:
            print(f"STOP-AND-ASK: corr(p_img+, p_txt+) > {m['shared_space_corr']} for {shared} "
                  "(shared-space artefact?)")
        print(f"wrote {csv} and {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
