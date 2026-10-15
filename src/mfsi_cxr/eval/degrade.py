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


METRICS = ("r_img", "r_txt", "A_cos", "R", "r_img_typ", "r_txt_typ")


def with_typicality(table):
    """Typicality re-check (PREREG_TYPICALITY): r' = r * T with the frozen clean-val reference."""
    return table.assign(r_img_typ=table["r_img"] * table["T_img"], r_txt_typ=table["r_txt"] * table["T_txt"])


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
            "other_margin_unchanged": bool(by[other]["change_lo"] <= 0 <= by[other]["change_hi"]),
            "own_typ_margin_falls": bool(by[f"{own}_typ"]["change_hi"] < 0)}


def run(cfg: dict[str, Any], split: str) -> tuple[Any, Any]:
    """Re-extract `split` clean and per degradation through the frozen full model (fp32) for every seed, score
    MFSI with the frozen parameters; returns (summary, per-finding) DataFrames. Degraded exports are saved
    under artifacts/<profile>/degrade/. The clean pass is checked against the Phase 2 export."""
    import json

    import pandas as pd

    from mfsi_cxr.data.datasets import SampleDataset
    from mfsi_cxr.mfsi.fit import load_split, score_split
    from mfsi_cxr.models.extract import run_export
    from mfsi_cxr.models.train import load_model
    from mfsi_cxr.utils.device import device_from_config
    from mfsi_cxr.utils.io import artifact_dir, check_artifact, write_meta

    dcfg = cfg["eval"]["degrade"]
    device = device_from_config(cfg)
    run_cfg = copy.deepcopy(cfg)
    run_cfg["train"]["precision"] = "fp32"
    out_dir = artifact_dir(cfg) / "degrade"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, per_finding = [], []
    for seed in cfg["train"]["seeds"]:
        ppath = artifact_dir(cfg) / f"mfsi_params_seed{seed}.json"
        check_artifact(ppath, cfg)
        params = json.loads(ppath.read_text(encoding="utf-8"))
        train, export = load_split(cfg, "train", seed), load_split(cfg, split, seed)
        model = load_model(run_cfg, cfg["mfsi"]["embeddings"], seed, device)
        base = SampleDataset(run_cfg, split)
        variants = [{"name": "clean", "modality": "none", "kind": "none", "param": None}, *dcfg["variants"]]
        tables = {}
        for v in variants:
            ds = DegradedDataset(base, v["kind"], v["param"], dcfg["seed"], dcfg["history_sections"])
            emb, _ = run_export(model, ds, run_cfg, device)
            if not np.array_equal(emb["study_id"], export["study_id"]):
                raise RuntimeError(f"{v['name']}: re-extracted rows differ from the {split} export")
            path = out_dir / f"emb_{split}_seed{seed}_{v['name']}.npz"
            np.savez(path, **emb)
            write_meta(path, cfg, seed, split=split, degradation=v)
            if v["name"] == "clean":
                gap = max(float(np.abs(emb[k] - export[k]).max()) for k in ("z_img", "z_txt"))
                if gap > dcfg["clean_tolerance"]:
                    raise RuntimeError(f"clean re-extraction differs from the Phase 2 export "
                                       f"(max |dz| {gap:.2e})")
            tables[v["name"]] = with_typicality(score_split(train, emb, params, cfg))
        for v in variants[1:]:
            rows = summarise(tables["clean"], tables[v["name"]], export["patient_id"], cfg)
            exp = expectation(v["modality"], rows)
            summary += [{"seed": seed, "variant": v["name"], "modality": v["modality"], **r, **exp}
                        for r in rows]
            for f, g in tables[v["name"]].groupby("finding", sort=False):
                c = tables["clean"][tables["clean"]["finding"] == f]
                per_finding.append({"seed": seed, "variant": v["name"], "finding": f,
                                    **{f"change_{m}": float(g[m].mean() - c[m].mean()) for m in METRICS}})
    return pd.DataFrame(summary), pd.DataFrame(per_finding)
