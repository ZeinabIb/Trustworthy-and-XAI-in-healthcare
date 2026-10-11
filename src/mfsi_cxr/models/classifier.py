"""Loss and metrics for the flat 14-label head (Phase 2.3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from mfsi_cxr.constants import FINDINGS


def masked_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted BCE: per-label weights (0 = unused, e.g. uncertain under U-Ignore; blank-derived negatives
    get labels.blank_weight), normalised by the total weight."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target, reduction="none",
                                                          pos_weight=pos_weight)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def pos_weight_from(targets: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    """n_neg / n_pos per label over the scored train labels (blank negatives included when used;
    1 where a label has no positive)."""
    pos = (targets * masks).sum(0)
    neg = ((1 - targets) * masks).sum(0)
    return torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), dtype=torch.float32)


def training_loss(out: dict[str, Any], batch: dict[str, Any], cfg: dict[str, Any],
                  pos_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted BCE minus lambda * attention entropy (the penalty that keeps attention from collapsing)."""
    device = out["logits"].device
    bce = masked_bce(out["logits"], batch["labels"].to(device), batch["label_weight"].to(device),
                     None if pos_weight is None else pos_weight.to(device))
    loss = bce - cfg["train"]["entropy_lambda"] * out["entropy"]
    return loss, {"bce": float(bce.detach()), "entropy": float(out["entropy"].detach())}


def auroc_table(targets: np.ndarray, masks: np.ndarray, probs: np.ndarray,
                findings: Sequence[str] = FINDINGS) -> pd.DataFrame:
    """Per finding: AUROC over masked-in labels; NaN when a split lacks positives or negatives."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for j, name in enumerate(findings):
        keep = masks[:, j] > 0
        y, p = targets[keep, j], probs[keep, j]
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        auc = float(roc_auc_score(y, p)) if n_pos and n_neg else float("nan")
        rows.append({"finding": name, "auroc": auc, "n_pos": n_pos, "n_neg": n_neg})
    return pd.DataFrame(rows)


def selection_score(table: pd.DataFrame, findings: Sequence[str]) -> float:
    """Mean val AUROC over the early-stopping findings (NaN entries skipped; NaN if none is defined)."""
    values = table.set_index("finding").loc[list(findings), "auroc"].dropna()
    return float(values.mean()) if len(values) else float("nan")
