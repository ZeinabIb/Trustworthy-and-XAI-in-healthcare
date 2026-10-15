"""Static figures for Phase 4 (matplotlib, PNG). Palette: the dataviz reference categorical slots 1-4 in fixed
order (blue, orange, aqua, yellow), recessive axes, 2 px lines, one legend per figure (identity never colour
alone). Small multiples share axes; every finding gets its own panel (never only an average).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval.calibration import reliability_table
from mfsi_cxr.eval.selective import risk_coverage

SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _grid(n_panels: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(16, 9.5), sharex=True, sharey=True, facecolor=SURFACE)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    for ax in axes.flat[:n_panels]:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    return plt, fig, axes


def risk_coverage_figure(scores: dict[str, np.ndarray], correct: np.ndarray, used: np.ndarray,
                         names: list[str], path: Path, title: str) -> None:
    """One panel per finding: risk vs coverage for up to four named scores (seed 0 arrays)."""
    plt, fig, axes = _grid(len(FINDINGS))
    for ax, (j, f) in zip(axes.flat, enumerate(FINDINGS), strict=False):
        u = used[:, j]
        for name, color in zip(names[: len(SLOTS)], SLOTS, strict=False):
            if u.sum() == 0:
                continue
            cov, risk = risk_coverage(scores[name][u, j], correct[u, j])
            ax.plot(cov, risk, color=color, linewidth=2, label=name)
        ax.set_title(f"{f} (n={int(u.sum())})", fontsize=9, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("coverage", color=MUTED, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("risk (error rate)", color=MUTED, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def reliability_figure(probs: dict[str, np.ndarray], label: np.ndarray, used: np.ndarray, bins: int,
                       path: Path, title: str) -> None:
    """One panel per finding: equal-mass reliability curve of each named profile (after calibration)."""
    plt, fig, axes = _grid(len(FINDINGS))
    for ax, (j, f) in zip(axes.flat, enumerate(FINDINGS), strict=False):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        u = used[:, j]
        for (name, p), color in zip(probs.items(), SLOTS, strict=False):
            if u.sum() == 0:
                continue
            t = reliability_table(p[u, j], label[u, j], bins)
            ax.plot(t["mean_prob"], t["frac_pos"], color=color, linewidth=2, marker="o", markersize=4,
                    label=name)
        ax.set_title(f"{f} (n={int(u.sum())})", fontsize=9, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("mean profile p+", color=MUTED, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction positive", color=MUTED, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
