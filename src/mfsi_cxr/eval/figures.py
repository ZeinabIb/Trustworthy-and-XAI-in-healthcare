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


def volume_figure(bins, path: Path, title: str) -> None:
    """One panel per volume variable: mean R per quantile bin, each finding a thin grey line, the mean over
    findings in the series colour (single highlighted series; no 14-colour legend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = list(dict.fromkeys(bins["variable"]))
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 3.6), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, v in zip(axes[0], variables, strict=True):
        g = bins[bins["variable"] == v]
        for _, gf in g.groupby("finding"):
            ax.plot(gf["bin"], gf["mean_R"], color=GRID, linewidth=1)
        mean = g.groupby("bin")["mean_R"].mean()
        ax.plot(mean.index, mean.to_numpy(), color=SLOTS[0], linewidth=2, marker="o", markersize=4,
                label="mean over findings")
        ax.set_title(v, fontsize=10, color=INK)
        ax.set_xlabel("quantile bin (low to high)", color=MUTED, fontsize=8)
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("mean R", color=MUTED, fontsize=8)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def deciles_figure(dec, path: Path, title: str) -> None:
    """One panel per finding: accuracy within each decile of R (calibration of R)."""
    plt, fig, axes = _grid(len(FINDINGS))
    for ax, f in zip(axes.flat, FINDINGS, strict=False):
        g = dec[dec["finding"] == f]
        ax.plot(g["decile"], g["accuracy"], color=SLOTS[0], linewidth=2, marker="o", markersize=4)
        ax.set_title(f, fontsize=9, color=INK)
        ax.set_ylim(0, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("decile of R (low to high)", color=MUTED, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy", color=MUTED, fontsize=8)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def degrade_figure(summary, path: Path, title: str) -> None:
    """Dot plot: change in mean r_img, r_txt and A_cos per degradation, with 95% CI whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(summary["variant"]))
    metrics = ["r_img", "r_txt", "A_cos"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k, (m, color) in enumerate(zip(metrics, SLOTS, strict=False)):
        g = summary[summary["metric"] == m].set_index("variant").loc[names]
        y = np.arange(len(names)) + (k - 1) * 0.22
        ax.errorbar(g["change"], y, xerr=[g["change"] - g["change_lo"], g["change_hi"] - g["change"]],
                    fmt="o", color=color, markersize=6, linewidth=1.5, capsize=0, label=m)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(np.arange(len(names)), names, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("change after degradation (mean over samples and findings)", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    ax.set_title(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def flip_rates_figure(flips, noise: float, path: Path, title: str) -> None:
    """Dot plot: post-vote decision flip rate per packet condition with 95% CI; dashed line = re-run noise
    floor (single series; the noise-floor row is the reference)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = flips[flips["condition"].isin(["rerun", "shuffle", "swap_modalities", "flip_leaning", "absent",
                                       "blinded"]) & flips["n"].gt(0)]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(f) + 1.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(f))
    lo = (f["flip_rate"] - f["lo"]).clip(lower=0)
    hi = (f["hi"] - f["flip_rate"]).clip(lower=0)
    ax.errorbar(f["flip_rate"], y, xerr=[lo, hi], fmt="o", color=SLOTS[0], markersize=6, capsize=0)
    ax.axvline(noise, color=MUTED, linestyle="--", linewidth=1)
    ax.text(noise, len(f) - 0.4, " re-run noise floor", color=MUTED, fontsize=8, va="bottom")
    ticks = [f"{c} (n={n})" for c, n in zip(f["condition"], f["n"], strict=True)]
    ax.set_yticks(y, ticks, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("decision flip rate vs the main run", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def cannot_decide_figure(table, path: Path, title: str) -> None:
    """cannot_decide rate by decile of the R shown, main packet vs shuffled packet (two series, legend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for (name, g), color in zip(table.groupby("packet", sort=True), SLOTS, strict=False):
        ax.plot(g["R_decile"], g["cannot_decide_rate"], color=color, linewidth=2, marker="o", markersize=4,
                label=f"{name} packet")
    ax.set_xlabel("decile of the R shown (low to high)", color=MUTED, fontsize=9)
    ax.set_ylabel("cannot_decide rate", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def claims_verified_figure(per_finding, path: Path, title: str) -> None:
    """Horizontal bars: share of claims tagged VERIFIED per finding (n claims in the label)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = per_finding.sort_values("verified_share")
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(t) + 1.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(np.arange(len(t)), t["verified_share"].fillna(0), color=SLOTS[0], height=0.6)
    ticks = [f"{f} ({int(c)} claims)" for f, c in zip(t["finding"], t["claims"], strict=True)]
    ax.set_yticks(np.arange(len(t)), ticks, fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of claims VERIFIED (the arbiter used the cited evidence)", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
