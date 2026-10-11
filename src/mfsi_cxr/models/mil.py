"""Two-level attention MIL (Phase 2.2) and the masking-based causal check.

Gated attention (Ilse et al. 2018): a_k = softmax_k(w^T (tanh(V h_k) * sigm(U h_k))), pooled = sum_k a_k h_k.
Image branch: level 1 images -> study vector (shared over studies), level 2 studies -> z_img (index study and
priors alike). Text branch: level 1 documents -> study vector (one document per study for now), level 2
ASYMMETRIC: z_txt = v_index + sum_priors b_s v_s with b_s = softmax_priors(w^T tanh(W [v_s ; v_index]))
(DECISIONS 2026-10-11), so z_txt always contains the index study in full and no priors gives v_index.
No positional encoding anywhere. Study 0 of every sample is the index study.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import nn

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.models.encoders import ImageEncoder, TextEncoder

BRANCHES = ("full", "image", "text")


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim where mask is True; rows without a valid entry get all zeros."""
    w = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    return torch.nan_to_num(w, nan=0.0) * mask


def attention_entropy(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean entropy over rows with at least two valid entries (a single instance has no choice)."""
    rows = mask.sum(-1) >= 2
    if not rows.any():
        return weights.new_zeros(())
    h = -(weights * torch.log(weights.clamp_min(1e-12))).sum(-1)
    return h[rows].mean()


class GatedAttentionPool(nn.Module):
    """(..., N, D) bags with (..., N) masks -> pooled (..., D) and weights (..., N)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = masked_softmax(scores, mask)
        return (a.unsqueeze(-1) * h).sum(-2), a


class AsymmetricPriorPool(nn.Module):
    """Index study as the query over its priors: z = v_index + sum_s b_s v_s."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.W = nn.Linear(2 * dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, v_index: torch.Tensor, v_priors: torch.Tensor,
                prior_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = v_index.unsqueeze(1).expand_as(v_priors)
        scores = self.w(torch.tanh(self.W(torch.cat([v_priors, query], -1)))).squeeze(-1)
        b = masked_softmax(scores, prior_mask)
        return v_index + (b.unsqueeze(-1) * v_priors).sum(1), b


def _study_bags(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Index + prior images as (B, S, K, C, H, W) with an image mask (B, S, K); study 0 = index."""
    img, prior = batch["images"], batch["prior_images"]
    b, k = img.shape[:2]
    kp = prior.shape[2] if prior.dim() > 2 else 0
    kmax = max(k, kp)
    s = 1 + prior.shape[1]
    bags = img.new_zeros(b, s, kmax, *img.shape[2:])
    mask = torch.zeros(b, s, kmax, dtype=torch.bool, device=img.device)
    bags[:, 0, :k], mask[:, 0, :k] = img, batch["image_mask"]
    if s > 1 and kp:
        bags[:, 1:, :kp], mask[:, 1:, :kp] = prior, batch["prior_image_mask"]
    return bags, mask


class ImageBranch(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        m = cfg["model"]
        self.encoder = ImageEncoder(cfg)
        d = m["emb_dim"]
        self.proj = nn.Identity() if self.encoder.feature_dim == d else nn.Linear(self.encoder.feature_dim, d)
        self.level1 = GatedAttentionPool(d, m["attention_hidden"])
        self.level2 = GatedAttentionPool(d, m["attention_hidden"])

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        bags, mask = _study_bags(batch)
        feats = bags.new_zeros(*mask.shape, self.level1.V.in_features)
        if mask.any():
            feats[mask] = self.proj(self.encoder(bags[mask])).to(feats.dtype)
        v, a = self.level1(feats, mask)
        study_mask = mask.any(-1)
        z, b = self.level2(v, study_mask)
        return {"z": z, "a": a, "a_mask": mask, "b": b, "b_mask": study_mask}


class TextBranch(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        m = cfg["model"]
        self.encoder = TextEncoder(cfg)
        d = m["emb_dim"]
        self.proj = nn.Linear(self.encoder.feature_dim, d)
        self.level1 = GatedAttentionPool(d, m["attention_hidden"])
        self.level2 = AsymmetricPriorPool(d, m["attention_hidden"])

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        prior_mask = batch.get("prior_text_mask", batch["prior_study_mask"])
        b, p = prior_mask.shape
        device = prior_mask.device
        index = self.proj(self.encoder(list(batch["index_text"]), "index"))
        priors = index.new_zeros(b, p, index.shape[-1])
        keep = prior_mask.cpu()
        texts = [t for i, docs in enumerate(batch["prior_texts"]) for j, t in enumerate(docs) if keep[i, j]]
        if texts:
            priors[prior_mask] = self.proj(self.encoder(texts, "prior")).to(priors.dtype)
        docs = torch.cat([index.unsqueeze(1), priors], 1).unsqueeze(2)  # (B, S, 1 doc, D)
        doc_mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=device), prior_mask], 1).unsqueeze(2)
        v, a = self.level1(docs, doc_mask)
        z, w = self.level2(v[:, 0], v[:, 1:], prior_mask)
        return {"z": z, "a": a, "a_mask": doc_mask, "b": w, "b_mask": prior_mask}


class MILClassifier(nn.Module):
    """Flat head over concat[z_img, z_txt] (or one of them for the branch ablations): 14 logits."""

    def __init__(self, cfg: dict[str, Any], branch: str = "full") -> None:
        super().__init__()
        if branch not in BRANCHES:
            raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}")
        self.branch = branch
        self.image = ImageBranch(cfg) if branch in ("full", "image") else None
        self.text = TextBranch(cfg) if branch in ("full", "text") else None
        n = (self.image is not None) + (self.text is not None)
        self.dropout = nn.Dropout(cfg["model"]["dropout"])
        self.head = nn.Linear(n * cfg["model"]["emb_dim"], len(FINDINGS))

    def encoders(self) -> list[nn.Module]:
        return [br.encoder for br in (self.image, self.text) if br is not None]

    def set_backbone_trainable(self, trainable: bool) -> None:
        for enc in self.encoders():
            enc.set_backbone_trainable(trainable)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        zs, ents = [], []
        for name, br in (("img", self.image), ("txt", self.text)):
            if br is None:
                continue
            r = br(batch)
            out[f"z_{name}"], out[f"attn_{name}"] = r["z"], r
            zs.append(r["z"])
            ents += [attention_entropy(r["a"], r["a_mask"]), attention_entropy(r["b"], r["b_mask"])]
        out["logits"] = self.head(self.dropout(torch.cat(zs, -1)))
        out["entropy"] = torch.stack(ents).mean()
        return out


def _without_top_image(batch: dict[str, Any], attn: dict[str, torch.Tensor]) -> tuple[dict, torch.Tensor]:
    """Remove each sample's image with the highest effective weight b_s * a_sk; eligible = >= 2 images.

    Only removable images compete: never padding, and never the index study's last image.
    """
    eff = (attn["b"].unsqueeze(-1) * attn["a"]).masked_fill(~attn["a_mask"], -1.0)
    single_index = attn["a_mask"][:, 0].sum(-1) == 1
    eff[:, 0] = eff[:, 0].masked_fill(single_index.unsqueeze(-1), -1.0)
    b = eff.shape[0]
    top = eff.reshape(b, -1).argmax(-1)
    s_idx, k_idx = top // eff.shape[2], top % eff.shape[2]
    eligible = attn["a_mask"].reshape(b, -1).sum(-1) >= 2
    new = dict(batch)
    image_mask, prior_image_mask = batch["image_mask"].clone(), batch["prior_image_mask"].clone()
    for i in range(b):
        if not eligible[i]:
            continue
        s, k = int(s_idx[i]), int(k_idx[i])
        if s == 0:
            image_mask[i, k] = False
        else:
            prior_image_mask[i, s - 1, k] = False
    new["image_mask"], new["prior_image_mask"] = image_mask, prior_image_mask
    if not image_mask.any(-1).all():  # an index study may not lose its last image
        raise RuntimeError("masking removed the last index image")
    return new, eligible


def _without_top_prior_text(batch: dict[str, Any],
                            attn: dict[str, torch.Tensor]) -> tuple[dict, torch.Tensor]:
    """Remove each sample's most-attended prior report (the index text is always kept in full)."""
    mask = batch.get("prior_text_mask", batch["prior_study_mask"]).clone()
    eligible = mask.any(-1)
    if mask.shape[1]:
        top = attn["b"].argmax(-1)
        mask[torch.arange(len(mask)), top] &= ~eligible
    new = dict(batch)
    new["prior_text_mask"] = mask
    return new, eligible


@torch.no_grad()
def masking_check(model: MILClassifier, batches: list[dict[str, Any]], margin: float) -> pd.DataFrame:
    """Per branch: share of eligible samples whose max |delta prob| over the 14 findings exceeds margin.

    Image: the image with the largest effective weight is removed (samples with >= 2 images). Text: the
    most-attended prior report is removed (samples with >= 1 prior); the index text is never removed.
    """
    was_training = model.training
    model.eval()
    rows = []
    for branch, remove in (("img", _without_top_image), ("txt", _without_top_prior_text)):
        if getattr(model, {"img": "image", "txt": "text"}[branch]) is None:
            continue
        n_eligible = n_changed = 0
        deltas = []
        for batch in batches:
            out = model(batch)
            masked, eligible = remove(batch, out[f"attn_{branch}"])
            if not eligible.any():
                continue
            p0, p1 = torch.sigmoid(out["logits"]), torch.sigmoid(model(masked)["logits"])
            d = (p1 - p0).abs().amax(-1)[eligible]
            n_eligible += int(eligible.sum())
            n_changed += int((d > margin).sum())
            deltas.append(d)
        mean_delta = float(torch.cat(deltas).mean()) if deltas else float("nan")
        rows.append({"branch": branch, "n_eligible": n_eligible, "n_changed": n_changed,
                     "share_changed": n_changed / n_eligible if n_eligible else float("nan"),
                     "mean_max_abs_delta": mean_delta, "margin": margin})
    model.train(was_training)
    return pd.DataFrame(rows)
