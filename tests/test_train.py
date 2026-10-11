"""Trainer (Phase 2.3): overfitting, resume == uninterrupted, outputs. Tiny encoders on synthetic data."""

from __future__ import annotations

import copy
import json
import shutil
from typing import Any

import numpy as np
import pytest
import torch

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.models.classifier import auroc_table, masked_bce, pos_weight_from, selection_score
from mfsi_cxr.models.train import ckpt_dir, load_model, run_name, train_model
from mfsi_cxr.utils.io import read_meta


def _cfg(base: dict[str, Any], **train: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["train"].update({"batch_size": 4, "warmup_epochs": 1, "early_stop_patience": 100, **train})
    return cfg


def test_masked_bce_ignores_masked_entries() -> None:
    logits = torch.tensor([[10.0, -10.0]])
    target = torch.tensor([[1.0, 1.0]])
    assert masked_bce(logits, target, torch.tensor([[1.0, 0.0]])) < 1e-3
    assert masked_bce(logits, target, torch.tensor([[1.0, 1.0]])) > 1.0


def test_pos_weight() -> None:
    t = np.array([[1, 0], [0, 0], [0, 0], [1, 0]], dtype=float)
    m = np.ones_like(t)
    assert torch.allclose(pos_weight_from(t, m), torch.tensor([1.0, 1.0]))  # 2 neg / 2 pos; no pos -> 1


def test_auroc_table_and_selection_score() -> None:
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=float)
    p = np.array([[0.9, 0.1], [0.2, 0.2], [0.8, 0.3], [0.1, 0.4]])
    table = auroc_table(y, np.ones_like(y), p, findings=["A", "B"])
    assert table.loc[0, "auroc"] == 1.0 and np.isnan(table.loc[1, "auroc"])
    assert selection_score(table, ["A", "B"]) == 1.0
    assert np.isnan(selection_score(table, ["B"]))


def _learnable_batches(cfg: dict[str, Any], n: int = 64, per_batch: int = 8) -> list[dict[str, Any]]:
    """n samples whose index text names its positive labels and whose image has one bright row block per
    positive label (global pooling keeps only their number).

    The synthetic fixture's images are pure noise, so nothing could be learned from them; this set checks
    the model's capacity (plan 2.3: overfit 64 samples), not the data.
    """
    from mfsi_cxr.data.datasets import collate_fn

    gen = torch.Generator().manual_seed(0)
    size, lo, hi = cfg["image"]["size"], *cfg["image"]["value_range"]
    rows = size // len(FINDINGS)
    items = []
    for i in range(n):
        y = (torch.rand(len(FINDINGS), generator=gen) < 0.4).float()
        img = torch.full((1, cfg["image"]["channels"], size, size), float(lo))
        positive = torch.nonzero(y).flatten().tolist()
        for j in positive:
            img[0, :, j * rows:(j + 1) * rows] = hi
        text = " ".join(f"label{j}" for j in positive) or "none"
        items.append({"images": img, "prior_images": [], "index_text": text, "prior_texts": [],
                      "labels": y, "label_mask": torch.ones_like(y), "label_values": y,
                      "patient_id": f"p{i}", "study_id": f"p{i}/s1", "n_studies": 1, "n_images": 1,
                      "text_len": 2, "n_priors": 0})
    return [collate_fn(items[k:k + per_batch]) for k in range(0, n, per_batch)]


def test_full_model_can_overfit_64_samples(smoke_cfg: dict[str, Any]) -> None:
    """With tiny RANDOM encoders the loss falls steadily but slowly (a from-scratch 2-layer BERT learns
    token presence slowly), so the bound is a clear drop, not near zero. The near-zero overfit check
    with the pretrained encoders is part of the pilot run (DECISIONS 2026-10-11)."""
    from mfsi_cxr.models.classifier import training_loss
    from mfsi_cxr.models.mil import MILClassifier

    cfg = copy.deepcopy(smoke_cfg)
    cfg["model"]["dropout"] = 0.0
    torch.manual_seed(0)
    model = MILClassifier(cfg, "full")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batches = _learnable_batches(cfg)

    def epoch_loss(train: bool) -> float:
        total = 0.0
        for batch in batches:
            loss, parts = training_loss(model(batch), batch, cfg)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += parts["bce"]
        return total / len(batches)

    first = epoch_loss(train=False)
    for _ in range(60):
        epoch_loss(train=True)
    with torch.no_grad():
        last = epoch_loss(train=False)
    assert last < 0.3 * first


def test_resume_matches_an_uninterrupted_run(smoke_artifacts: dict[str, Any]) -> None:
    full = _cfg(smoke_artifacts, epochs=2)
    train_model(full, "full", seed=3)
    folder = ckpt_dir(full, run_name("full", 3))
    uninterrupted = torch.load(folder / "last.pt", weights_only=False)["model"]
    shutil.rmtree(folder)

    train_model(_cfg(smoke_artifacts, epochs=1), "full", seed=3)  # "killed" after epoch 0
    train_model(full, "full", seed=3, resume=True)
    resumed = torch.load(folder / "last.pt", weights_only=False)
    assert resumed["epoch"] == 1
    for k, v in uninterrupted.items():
        assert torch.allclose(v.float(), resumed["model"][k].float(), atol=1e-6), k


@pytest.mark.parametrize("branch", ["image", "text"])
def test_branch_runs_write_outputs(smoke_artifacts: dict[str, Any], branch: str) -> None:
    cfg = _cfg(smoke_artifacts, epochs=1)
    summary = train_model(cfg, branch, seed=0)
    folder = ckpt_dir(cfg, run_name(branch, 0))
    saved = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert saved["run"] == summary["run"] and len(saved["val_auroc"]) == len(FINDINGS)
    assert read_meta(folder / "best.pt")["profile"] == cfg["profile"]
    model = load_model(cfg, branch, 0, "cpu")
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_shuffled_text_run_has_its_own_name(smoke_artifacts: dict[str, Any]) -> None:
    cfg = _cfg(smoke_artifacts, epochs=1)
    summary = train_model(cfg, "text", seed=0, text_shuffle_seed=7)
    assert summary["run"] == "text_seed0_shuffled_text"
