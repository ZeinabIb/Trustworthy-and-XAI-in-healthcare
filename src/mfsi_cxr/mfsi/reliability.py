"""Agreement and reliability (Phase 3, steps 9-11).

A_cos = cos(p_img, p_txt) with p = (p_+, p_-); A_js = 1 - JS(p_img, p_txt), base-2 logs so JS in [0, 1];
A_none = 1 (ablation). R = A * min(r_img, r_txt) (ablation: mean). One-witness findings (text not
informative) are a separate stratum with R = r_img, thresholded and calibrated on their own, never ranked
with two-witness R.
"""

from __future__ import annotations

import numpy as np

AGREEMENTS = ("cos", "js", "none")
COMBINES = ("min", "mean")
TWO_WITNESS, ONE_WITNESS = "two_witness", "one_witness"


def _pair(p_pos: np.ndarray) -> np.ndarray:
    return np.stack([p_pos, 1.0 - p_pos], -1)


def agreement_cos(p_img_pos: np.ndarray, p_txt_pos: np.ndarray) -> np.ndarray:
    a, b = _pair(p_img_pos), _pair(p_txt_pos)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def agreement_js(p_img_pos: np.ndarray, p_txt_pos: np.ndarray) -> np.ndarray:
    a, b = _pair(p_img_pos), _pair(p_txt_pos)
    m = (a + b) / 2

    def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(p > 0, p * np.log2(p / q), 0.0)
        return t.sum(-1)

    return 1.0 - 0.5 * (kl(a, m) + kl(b, m))


def agreement(p_img_pos: np.ndarray, p_txt_pos: np.ndarray, kind: str) -> np.ndarray:
    if kind == "cos":
        return agreement_cos(p_img_pos, p_txt_pos)
    if kind == "js":
        return agreement_js(p_img_pos, p_txt_pos)
    if kind == "none":
        return np.ones_like(p_img_pos)
    raise ValueError(f"mfsi.agreement must be one of {AGREEMENTS}, got {kind!r}")


def reliability(a: np.ndarray, r_img: np.ndarray, r_txt: np.ndarray, combine: str) -> np.ndarray:
    if combine == "min":
        return a * np.minimum(r_img, r_txt)
    if combine == "mean":
        return a * (r_img + r_txt) / 2
    raise ValueError(f"mfsi.combine must be one of {COMBINES}, got {combine!r}")
