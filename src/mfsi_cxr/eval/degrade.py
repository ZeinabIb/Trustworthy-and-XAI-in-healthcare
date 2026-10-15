"""Degrade-one-modality test (Phase 4 validity control 3; face validity only).

Only one modality of every sample is degraded (all of its images, or all of its texts); the frozen model
re-extracts embeddings (fp32, eval mode) and MFSI is re-scored with the frozen parameters. Expected: the
degraded side's margin r falls, the other side's does not (the branches are separate, so the other side's
embedding is unchanged by construction). Variants (configs/eval.yaml `eval.degrade`):
- image: Gaussian blur (sigma in pixels at the cached size), occlusion of a seeded random square covering a
  fraction of the image area, filled with that image's mean value;
- text: keep the first fraction of the words of every text, shuffle sentences, drop the history sections of
  the index text (sections are only removed from the leakage allowlist, never added).
Randomness is seeded per (degrade seed, sample index).
"""

from __future__ import annotations

import copy
import math
import re
from functools import partial
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _images(item: dict[str, Any], fn) -> dict[str, Any]:
    out = dict(item)
    out["images"] = fn(item["images"])
    out["prior_images"] = [fn(t) for t in item["prior_images"]]
    return out


def blur(t: torch.Tensor, sigma: float, rng: np.random.Generator) -> torch.Tensor:
    """t (K, C, H, W): Gaussian blur over H and W."""
    if t.numel() == 0:
        return t
    a = gaussian_filter(t.numpy().astype(np.float64), sigma=(0, 0, sigma, sigma))
    return torch.from_numpy(a.astype(np.float32))


def occlude(t: torch.Tensor, frac: float, rng: np.random.Generator) -> torch.Tensor:
    """Square covering `frac` of each image's area at a random position, filled with the image mean."""
    out = t.clone()
    _, _, h, w = t.shape
    side_h, side_w = max(1, round(math.sqrt(frac) * h)), max(1, round(math.sqrt(frac) * w))
    for k in range(t.shape[0]):
        y0, x0 = int(rng.integers(0, h - side_h + 1)), int(rng.integers(0, w - side_w + 1))
        out[k, :, y0:y0 + side_h, x0:x0 + side_w] = t[k].mean()
    return out


def truncate(text: str, frac: float, rng: np.random.Generator) -> str:
    words = text.split()
    return " ".join(words[: math.ceil(frac * len(words))]) if words else text


def shuffle_sentences(text: str, rng: np.random.Generator) -> str:
    parts = [p for p in SENTENCE.split(text) if p]
    return " ".join(parts[i] for i in rng.permutation(len(parts))) if len(parts) > 1 else text


def apply(kind: str, param: Any, item: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    if kind == "blur":
        return _images(item, lambda t: blur(t, float(param), rng))
    if kind == "occlude":
        return _images(item, lambda t: occlude(t, float(param), rng))
    if kind in ("truncate", "shuffle"):
        if kind == "truncate":
            def fn(s: str) -> str:
                return truncate(s, float(param), rng)
        else:
            def fn(s: str) -> str:
                return shuffle_sentences(s, rng)
        out = dict(item)
        out["index_text"] = fn(item["index_text"])
        out["prior_texts"] = [fn(s) for s in item["prior_texts"]]
        return out
    if kind in ("none", "drop_history"):  # drop_history acts on the dataset's text builder
        return item
    raise ValueError(f"unknown degradation {kind!r}")


class DegradedDataset(Dataset):
    """Wraps a SampleDataset; each item gets one degradation with an RNG seeded by (seed, index)."""

    def __init__(self, base: Dataset, kind: str, param: Any, seed: int,
                 history_sections: list[str] | None = None) -> None:
        self.base = base
        if kind == "drop_history":
            self.base = copy.copy(base)
            self.base.text = copy.copy(base.text)
            self.base.text.sec_cols = [c for c in base.text.sec_cols if c not in set(history_sections or [])]
        self.fn = partial(apply, kind, param)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.fn(self.base[i], np.random.default_rng([self.seed, i]))


METRICS = ("r_img", "r_txt", "A_cos", "R")


def summarise(clean, degraded, patient_id: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Mean of each metric before / after over all (sample, finding) rows, and the change with a patient-
    bootstrap CI (per-sample mean over findings, paired by sample)."""
    from mfsi_cxr.constants import FINDINGS
    from mfsi_cxr.eval.stats import bootstrap, percentile_ci

    e = cfg["eval"]
    rows = []
    for m in METRICS:
        before = clean[m].to_numpy().reshape(-1, len(FINDINGS)).mean(1)
        after = degraded[m].to_numpy().reshape(-1, len(FINDINGS)).mean(1)
        diff = after - before
        lo, hi = percentile_ci(bootstrap(patient_id, lambda r, d=diff: float(d[r].mean()), e["bootstrap_B"],
                                         e["bootstrap_seed"]), e["alpha"])
        rows.append({"metric": m, "before": float(before.mean()), "after": float(after.mean()),
                     "change": float(diff.mean()), "change_lo": lo, "change_hi": hi,
                     "max_abs_change": float(np.abs(diff).max())})
    return rows


def expectation(modality: str, summary: list[dict[str, Any]]) -> dict[str, Any]:
    """Face-validity expectation: the degraded side's margin falls (CI below 0), the other side's does not."""
    by = {r["metric"]: r for r in summary}
    own, other = ("r_img", "r_txt") if modality == "image" else ("r_txt", "r_img")
    return {"own_margin_falls": bool(by[own]["change_hi"] < 0),
            "other_margin_unchanged": bool(by[other]["change_lo"] <= 0 <= by[other]["change_hi"])}
