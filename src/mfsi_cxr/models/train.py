"""Training loop (Phase 2.3): warm-up with frozen encoders, early stopping on val, resumable.

Checkpoints: artifacts/<profile>/ckpt/<run>/last.pt after every epoch and on interrupt (model, optimizer,
epoch, best score, patience, RNG states, history), best.pt for the best val selection score. `resume=True`
continues from last.pt with the same command. Each epoch's shuffle is seeded from (seed, epoch), so a resumed
run sees the same batches as an uninterrupted one. Val is used for early stopping only; test is never read.
"""

from __future__ import annotations

import copy
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mfsi_cxr.data.datasets import SampleDataset, collate_fn
from mfsi_cxr.models.classifier import auroc_table, pos_weight_from, selection_score, training_loss
from mfsi_cxr.models.mil import MILClassifier
from mfsi_cxr.utils.device import device_from_config
from mfsi_cxr.utils.io import artifact_dir, write_meta
from mfsi_cxr.utils.logging import get_logger
from mfsi_cxr.utils.seed import set_seed

log = get_logger("train")
PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
EPOCH_SEED_STRIDE = 1000  # shuffle seed of an epoch = seed * stride + epoch
RESUMABLE_KEYS = ("epochs", "num_workers", "device")  # may change between a run and its resume
RECIPE_SECTIONS = ("data", "image", "labels", "model", "train")  # what defines a trained model


class ResumeMismatchError(RuntimeError):
    """last.pt was written with another training recipe; resuming would mix two recipes."""


def recipe_hash(cfg: dict[str, Any]) -> str:
    """Hash of what defines a trained model (RECIPE_SECTIONS), except where the data lives and how long or
    fast it runs (more epochs may be added on resume). Probe, sanity and eval settings do not count."""
    from mfsi_cxr.utils.config import config_hash

    recipe = copy.deepcopy({k: cfg[k] for k in RECIPE_SECTIONS})
    recipe["data"].pop("root", None)
    for key in RESUMABLE_KEYS:
        recipe["train"].pop(key, None)
    return config_hash(recipe)


def run_name(branch: str, seed: int, shuffled_text: bool = False) -> str:
    return f"{branch}_seed{seed}" + ("_shuffled_text" if shuffled_text else "")


def ckpt_dir(cfg: dict[str, Any], name: str) -> Path:
    return artifact_dir(cfg) / "ckpt" / name


def autocast(cfg: dict[str, Any], device: str):
    dtype = PRECISIONS[cfg["train"]["precision"]]
    if dtype is None or device == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)


def make_loader(ds: SampleDataset, cfg: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    workers = cfg["train"]["num_workers"]
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle, generator=g,
                      collate_fn=collate_fn, num_workers=workers, persistent_workers=False,
                      pin_memory=False)


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(model: MILClassifier, loader: DataLoader, cfg: dict[str, Any], device: str) -> dict[str, Any]:
    """Eval-mode pass: logits, probs, targets, masks and ids as numpy arrays."""
    model.eval()
    acc: dict[str, list] = {"logits": [], "labels": [], "label_mask": [], "study_id": [], "patient_id": []}
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(cfg, device):
            out = model(batch)
        acc["logits"].append(out["logits"].float().cpu())
        acc["labels"].append(batch["labels"].cpu())
        acc["label_mask"].append(batch["label_mask"].cpu())
        acc["study_id"] += batch["study_id"]
        acc["patient_id"] += batch["patient_id"]
    res = {k: torch.cat(v).numpy() if k in ("logits", "labels", "label_mask") else np.array(v)
           for k, v in acc.items()}
    res["probs"] = 1.0 / (1.0 + np.exp(-res["logits"]))
    return res


def _rng_state() -> dict[str, Any]:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save(obj: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # a killed run never leaves a half-written checkpoint under the final name


def _optimizer(model: MILClassifier, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    t = cfg["train"]
    backbone = {id(p) for enc in model.encoders() for p in enc.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in backbone], "lr": t["lr"]},
        {"params": [p for p in model.parameters() if id(p) in backbone], "lr": t["lr_backbone"]},
    ]
    return torch.optim.AdamW(groups, weight_decay=t["weight_decay"])


def train_model(cfg: dict[str, Any], branch: str, seed: int, resume: bool = False,
                text_shuffle_seed: int | None = None) -> dict[str, Any]:
    """Train one run; returns its summary (val AUROC table of the best epoch, history)."""
    t = cfg["train"]
    name = run_name(branch, seed, text_shuffle_seed is not None)
    out_dir = ckpt_dir(cfg, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    last, best_path = out_dir / "last.pt", out_dir / "best.pt"
    device = device_from_config(cfg)
    set_seed(seed)

    train_ds = SampleDataset(cfg, "train", text_shuffle_seed=text_shuffle_seed)
    val_ds = SampleDataset(cfg, "val", text_builder=train_ds.text,
                           text_shuffle_seed=None if text_shuffle_seed is None else text_shuffle_seed + 1)
    val_loader = make_loader(val_ds, cfg, shuffle=False, seed=seed)
    model = MILClassifier(cfg, branch).to(device)
    optimizer = _optimizer(model, cfg)
    pos_weight = None
    if t["pos_weight"]:
        items = [train_ds[i] for i in range(len(train_ds))]
        pos_weight = pos_weight_from(np.stack([x["labels"].numpy() for x in items]),
                                     np.stack([x["label_mask"].numpy() for x in items]))

    start, best, patience, history = 0, -math.inf, 0, []
    if resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state.get("recipe") != recipe_hash(cfg):
            raise ResumeMismatchError(f"{last} was trained with another config; rerun with --fresh")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["epoch"] + 1, state["best"]
        patience, history = state["patience"], state["history"]
        _set_rng_state(state["rng"])
        log.info("%s: resumed after epoch %d", name, state["epoch"])

    def checkpoint(epoch: int) -> None:
        _save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
               "best": best, "patience": patience, "history": history, "rng": _rng_state(),
               "branch": branch, "seed": seed, "recipe": recipe_hash(cfg)}, last)

    epoch = start - 1
    try:
        for epoch in range(start, t["epochs"]):
            if patience >= t["early_stop_patience"]:
                break
            model.set_backbone_trainable(epoch >= t["warmup_epochs"])
            model.train()
            loader = make_loader(train_ds, cfg, shuffle=True, seed=seed * EPOCH_SEED_STRIDE + epoch)
            sums = {"loss": 0.0, "bce": 0.0, "entropy": 0.0}
            n = 0
            for batch in loader:
                batch = to_device(batch, device)
                with autocast(cfg, device):
                    out = model(batch)
                loss, parts = training_loss(out, batch, cfg, pos_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                optimizer.step()
                sums["loss"] += float(loss.detach())
                sums["bce"] += parts["bce"]
                sums["entropy"] += parts["entropy"]
                n += 1
            pred = predict(model, val_loader, cfg, device)
            table = auroc_table(pred["labels"], pred["label_mask"], pred["probs"])
            score = selection_score(table, t["early_stop_findings"])
            history.append({"epoch": epoch, **{k: v / max(n, 1) for k, v in sums.items()},
                            "val_score": score})
            log.info("%s epoch %d: loss %.4f entropy %.3f val score %.4f", name, epoch,
                     history[-1]["loss"], history[-1]["entropy"], score)
            improved = not math.isnan(score) and score > best
            if improved or not best_path.exists():
                best = score if improved else best
                patience = 0
                _save({"model": model.state_dict(), "epoch": epoch, "val_score": score}, best_path)
            else:
                patience += 1
            checkpoint(epoch)
    except KeyboardInterrupt:
        checkpoint(epoch)
        log.warning("%s: interrupted; saved last.pt (resume with --resume)", name)
        raise

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    pred = predict(model, val_loader, cfg, device)
    table = auroc_table(pred["labels"], pred["label_mask"], pred["probs"])
    summary = {"run": name, "branch": branch, "seed": seed, "best_epoch": state["epoch"],
               "val_score": state["val_score"], "history": history,
               "val_auroc": table.to_dict(orient="records")}
    metrics = out_dir / "metrics.json"
    metrics.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_meta(metrics, cfg, seed)
    write_meta(best_path, cfg, seed, branch=branch, epoch=state["epoch"])
    return summary


def load_model(cfg: dict[str, Any], branch: str, seed: int, device: str) -> MILClassifier:
    """The best checkpoint of a run, in eval mode (frozen for extraction)."""
    from mfsi_cxr.utils.io import check_artifact

    path = ckpt_dir(cfg, run_name(branch, seed)) / "best.pt"
    check_artifact(path, cfg)
    model = MILClassifier(cfg, branch).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def auroc_summary(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (run, finding) val AUROC of the best epoch."""
    return pd.DataFrame([{"run": s["run"], "branch": s["branch"], "seed": s["seed"], **r}
                         for s in summaries for r in s["val_auroc"]])
