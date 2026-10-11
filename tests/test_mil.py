"""Two-level MIL (Phase 2.2) and the masking check, with tiny random encoders (no downloads)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.data.datasets import SampleDataset, collate_fn
from mfsi_cxr.models.encoders import HashTokenizer
from mfsi_cxr.models.mil import (
    AsymmetricPriorPool,
    GatedAttentionPool,
    MILClassifier,
    attention_entropy,
    masked_softmax,
    masking_check,
)


def _bags(b: int = 4, n: int = 6, d: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    h = torch.randn(b, n, d)
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i, k in enumerate([6, 3, 1, 4][:b]):
        mask[i, :k] = True
    return h, mask


def test_weights_sum_to_one_and_padding_gets_zero() -> None:
    h, mask = _bags()
    _, a = GatedAttentionPool(16, 8)(h, mask)
    assert torch.allclose(a.sum(-1), torch.ones(4))
    assert torch.all(a[~mask] == 0)


def test_padding_values_do_not_change_the_pooled_vector() -> None:
    h, mask = _bags()
    pool = GatedAttentionPool(16, 8)
    z1, _ = pool(h, mask)
    z2, _ = pool(h.masked_fill(~mask.unsqueeze(-1), 1e3), mask)
    assert torch.allclose(z1, z2, atol=1e-5)


def test_level1_is_permutation_invariant() -> None:
    h, mask = _bags()
    pool = GatedAttentionPool(16, 8)
    z, a = pool(h[:1], mask[:1])
    perm = torch.randperm(6)
    zp, ap = pool(h[:1, perm], mask[:1, perm])
    assert torch.allclose(z, zp, atol=1e-6) and torch.allclose(a[0, perm], ap[0], atol=1e-6)


def test_empty_bag_gives_zero_vector_not_nan() -> None:
    w = masked_softmax(torch.randn(2, 3), torch.zeros(2, 3, dtype=torch.bool))
    assert torch.equal(w, torch.zeros(2, 3))


def test_asymmetric_pool_keeps_index_in_full() -> None:
    torch.manual_seed(0)
    pool = AsymmetricPriorPool(8, 4)
    v_index, priors = torch.randn(3, 8), torch.randn(3, 2, 8)
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    z, b = pool(v_index, priors, mask)
    assert torch.allclose(z[2], v_index[2])                       # no priors -> v_index
    assert torch.allclose(b.sum(-1), torch.tensor([1.0, 1.0, 0.0]))
    assert torch.allclose(z[1], v_index[1] + priors[1, 0], atol=1e-6)  # one prior gets weight 1


def test_attention_entropy() -> None:
    uniform = torch.full((1, 4), 0.25)
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert torch.isclose(attention_entropy(uniform, mask), torch.log(torch.tensor(4.0)))
    single = torch.tensor([[1.0, 0.0]])
    assert attention_entropy(single, torch.tensor([[True, False]])) == 0  # no choice -> not counted


def test_hash_tokenizer_shapes_and_truncation() -> None:
    tok = HashTokenizer(100)
    enc = tok(["a b c d e f", "x"], max_length=4)
    assert enc["input_ids"].shape == (2, 4)
    assert enc["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert tok.token_ids("Effusion") == tok.token_ids("effusion")


@pytest.fixture
def batch(smoke_artifacts: dict[str, Any]) -> dict[str, Any]:
    ds = SampleDataset(smoke_artifacts, "train")
    return collate_fn([ds[i] for i in range(min(len(ds), 6))])


@pytest.mark.parametrize("branch", ["full", "image", "text"])
def test_model_shapes(smoke_artifacts: dict[str, Any], batch: dict[str, Any], branch: str) -> None:
    torch.manual_seed(0)
    model = MILClassifier(smoke_artifacts, branch)
    out = model(batch)
    b, d = len(batch["study_id"]), smoke_artifacts["model"]["emb_dim"]
    assert out["logits"].shape == (b, len(FINDINGS))
    assert ("z_img" in out) == (branch != "text") and ("z_txt" in out) == (branch != "image")
    for key in ("z_img", "z_txt"):
        if key in out:
            assert out[key].shape == (b, d) and torch.isfinite(out[key]).all()
    assert torch.isfinite(out["entropy"])


def test_backbone_freezing(smoke_artifacts: dict[str, Any]) -> None:
    model = MILClassifier(smoke_artifacts, "full")
    model.set_backbone_trainable(False)
    assert not any(p.requires_grad for e in model.encoders() for p in e.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    model.set_backbone_trainable(True)
    assert any(p.requires_grad for e in model.encoders() for p in e.parameters())



def _controlled_batch(values: list[list[float]], size: int = 8) -> dict[str, Any]:
    """Index-study images of known constant intensity; no priors; empty texts."""
    b, k = len(values), max(map(len, values))
    images = torch.zeros(b, k, 1, size, size)
    mask = torch.zeros(b, k, dtype=torch.bool)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            images[i, j], mask[i, j] = v, True
    return {"images": images, "image_mask": mask, "prior_images": torch.zeros(b, 0, 0, 1, size, size),
            "prior_image_mask": torch.zeros(b, 0, 0, dtype=torch.bool),
            "prior_study_mask": torch.zeros(b, 0, dtype=torch.bool),
            "index_text": [""] * b, "prior_texts": [[] for _ in range(b)]}


def _one_instance_model(cfg: dict[str, Any]) -> MILClassifier:
    """Image model whose level 1 attends (almost) only to the brightest image and whose output reads it."""
    model = MILClassifier(cfg, "image")
    d = cfg["model"]["emb_dim"]
    model.image.encoder.forward = lambda x: x.mean((1, 2, 3)).unsqueeze(-1).expand(-1, d)
    pool = model.image.level1
    with torch.no_grad():
        for lin in (pool.V, pool.U):
            lin.weight.zero_(), lin.bias.zero_()
            lin.weight[:, 0] = 1.0
        pool.w.weight.fill_(1e3)
        model.head.weight.zero_(), model.head.bias.zero_()
        model.head.weight[:, 0] = 5.0
    return model.eval()


def test_masking_check_detects_a_model_that_uses_one_instance(smoke_cfg: dict[str, Any]) -> None:
    model = _one_instance_model(smoke_cfg)
    batch = _controlled_batch([[0.1, 0.5, 0.9], [0.2, 0.8], [0.7]])  # last sample: one image, not eligible
    row = masking_check(model, [batch], margin=0.05).set_index("branch").loc["img"]
    assert row["n_eligible"] == 2 and row["n_changed"] == 2


def test_masking_check_on_a_constant_model_changes_nothing(smoke_artifacts: dict[str, Any],
                                                           batch: dict[str, Any]) -> None:
    model = MILClassifier(smoke_artifacts, "full")
    with torch.no_grad():
        model.head.weight.zero_()
    table = masking_check(model, [batch], margin=0.05).set_index("branch")
    assert set(table.index) == {"img", "txt"}
    assert (table["n_changed"] == 0).all()
