"""Calibration of the profiles p^m_+ against the label (Phase 4): ECE / MCE with equal-mass bins and the
reliability-diagram table. Phase 3 reports equal-width ECE when fitting tau/b; reporting here uses equal mass
(plan 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(prob: np.ndarray, y: np.ndarray, bins: int) -> pd.DataFrame:
    """Equal-mass bins (sorted by prob, split into `bins` near-equal parts): count, mean prob, mean label."""
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(prob, kind="stable")
    rows = [{"bin": i, "n": len(idx), "mean_prob": float(prob[idx].mean()), "frac_pos": float(y[idx].mean())}
            for i, idx in enumerate(np.array_split(order, min(bins, len(prob)))) if len(idx)]
    return pd.DataFrame(rows, columns=["bin", "n", "mean_prob", "frac_pos"])


def ece_mce(prob: np.ndarray, y: np.ndarray, bins: int) -> tuple[float, float]:
    """(ECE, MCE) over equal-mass bins; NaN for empty input."""
    t = reliability_table(prob, y, bins)
    if t.empty:
        return float("nan"), float("nan")
    gap = (t["frac_pos"] - t["mean_prob"]).abs()
    return float((gap * t["n"]).sum() / t["n"].sum()), float(gap.max())
