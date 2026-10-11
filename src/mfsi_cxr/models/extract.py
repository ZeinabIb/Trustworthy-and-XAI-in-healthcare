"""Frozen exports (Phase 2.4): emb_{split}_seed{k}.npz and attn_{split}_seed{k}.npz. MFSI reads only these.

The model is loaded from best.pt, frozen, in eval mode (no dropout), and run in fp32 whatever the training
precision, so the exported vectors are deterministic and not bf16-rounded (MFSI fits covariances on them).
The full model writes the contract names; the branch ablations add a suffix (`..._image`, `..._text`).
No metric is computed here, so exporting the test split does not "open" it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mfsi_cxr.data.datasets import SampleDataset
from mfsi_cxr.models.mil import MILClassifier
from mfsi_cxr.models.train import load_model, make_loader, to_device
from mfsi_cxr.utils.io import artifact_dir, write_meta

ID_KEYS = ("patient_id", "study_id")
COUNT_KEYS = ("n_studies", "n_images", "text_len", "n_priors")


def export_paths(cfg: dict[str, Any], branch: str, split: str, seed: int) -> tuple[Path, Path]:
    suffix = "" if branch == "full" else f"_{branch}"
    adir = artifact_dir(cfg)
    return adir / f"emb_{split}_seed{seed}{suffix}.npz", adir / f"attn_{split}_seed{seed}{suffix}.npz"


def _pad_cat(arrays: list[np.ndarray], fill: float | bool) -> np.ndarray:
    """Concatenate (n_i, *dims_i) arrays along axis 0, padding the trailing dims to their maximum."""
    if not arrays:
        return np.zeros((0,))
    shape = [max(a.shape[d] for a in arrays) for d in range(1, arrays[0].ndim)]
    out = []
    for a in arrays:
        pad = [(0, 0)] + [(0, m - s) for m, s in zip(shape, a.shape[1:], strict=True)]
        out.append(np.pad(a, pad, constant_values=fill))
    return np.concatenate(out)


@torch.no_grad()
def run_export(model: MILClassifier, ds: SampleDataset, cfg: dict[str, Any],
               device: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    emb: dict[str, list] = {k: [] for k in ("z_img", "z_txt", "logits", "y", "label_mask",
                                            *ID_KEYS, *COUNT_KEYS)}
    attn: dict[str, list] = {}
    for batch in make_loader(ds, cfg, shuffle=False, seed=0):
        batch = to_device(batch, device)
        out = model(batch)
        for key in ("z_img", "z_txt"):
            if key in out:
                emb[key].append(out[key].float().cpu().numpy())
        emb["logits"].append(out["logits"].float().cpu().numpy())
        emb["y"].append(batch["label_values"].cpu().numpy())
        emb["label_mask"].append(batch["label_mask"].cpu().numpy())
        for k in ID_KEYS:
            emb[k] += list(batch[k])
        for k in COUNT_KEYS:
            emb[k].append(batch[k].cpu().numpy())
        for name in ("img", "txt"):
            if f"attn_{name}" not in out:
                continue
            for part in ("a", "a_mask", "b", "b_mask"):
                attn.setdefault(f"{name}_{part}", []).append(out[f"attn_{name}"][part].float().cpu().numpy()
                                                             if "mask" not in part
                                                             else out[f"attn_{name}"][part].cpu().numpy())
    res = {k: np.concatenate(v) for k, v in emb.items() if v and k not in ID_KEYS}
    res["probs"] = 1.0 / (1.0 + np.exp(-res["logits"]))
    for k in ID_KEYS:
        res[k] = np.array(emb[k], dtype=np.str_)
    att = {k: _pad_cat(v, False if k.endswith("mask") else 0.0) for k, v in attn.items()}
    att["study_id"] = res["study_id"]
    return res, att


def extract(cfg: dict[str, Any], branch: str, seed: int, split: str, device: str) -> tuple[Path, Path]:
    """Write the emb/attn exports of one (branch, seed, split) and their sidecars."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg["train"]["precision"] = "fp32"  # exports are never bf16-rounded
    model = load_model(run_cfg, branch, seed, device)
    ds = SampleDataset(run_cfg, split)
    emb, att = run_export(model, ds, run_cfg, device)
    emb_path, attn_path = export_paths(cfg, branch, split, seed)
    for path, arrays in ((emb_path, emb), (attn_path, att)):
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)
        write_meta(path, cfg, seed, branch=branch, split=split, n=len(emb["study_id"]))
    return emb_path, attn_path


def load_export(path: Path, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    from mfsi_cxr.utils.io import check_artifact

    check_artifact(path, cfg)
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
