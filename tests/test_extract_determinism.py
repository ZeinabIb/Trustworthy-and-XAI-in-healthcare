"""Frozen exports (Phase 2.4): deterministic, right shapes, no NaN, ablation exports, attention masks."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.models.extract import export_paths, extract, load_export
from mfsi_cxr.models.train import train_model


@pytest.fixture
def trained(smoke_artifacts: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(smoke_artifacts)
    cfg["train"].update({"epochs": 1, "batch_size": 4})
    for branch in ("full", "image", "text"):
        train_model(cfg, branch, seed=0)
    return cfg


def test_extraction_is_deterministic(trained: dict[str, Any]) -> None:
    emb_path, attn_path = extract(trained, "full", 0, "train", "cpu")
    first = load_export(emb_path, trained)
    extract(trained, "full", 0, "train", "cpu")
    second = load_export(emb_path, trained)
    for k, v in first.items():
        if v.dtype.kind == "f":
            assert np.allclose(v, second[k], atol=1e-6, equal_nan=True), k
        else:
            assert np.array_equal(v, second[k]), k


def test_export_contents(trained: dict[str, Any]) -> None:
    emb_path, attn_path = extract(trained, "full", 0, "val", "cpu")
    emb, att = load_export(emb_path, trained), load_export(attn_path, trained)
    n, d = len(emb["study_id"]), trained["model"]["emb_dim"]
    assert emb["z_img"].shape == emb["z_txt"].shape == (n, d)
    assert np.isfinite(emb["z_img"]).all() and np.isfinite(emb["z_txt"]).all()
    assert emb["probs"].shape == emb["logits"].shape == emb["y"].shape == (n, len(FINDINGS))
    assert set(np.unique(emb["y"][~np.isnan(emb["y"])])) <= {-1.0, 0.0, 1.0}  # NaN kept
    for k in ("patient_id", "n_studies", "n_images", "text_len", "n_priors", "label_mask"):
        assert len(emb[k]) == n, k
    a, am = att["img_a"], att["img_a_mask"]
    assert np.allclose(a.sum(-1)[am.any(-1)], 1.0, atol=1e-5) and np.all(a[~am] == 0)
    assert np.array_equal(att["study_id"], emb["study_id"])


def test_ablation_exports_hold_one_embedding(trained: dict[str, Any]) -> None:
    for branch, present, absent in (("image", "z_img", "z_txt"), ("text", "z_txt", "z_img")):
        emb_path, _ = extract(trained, branch, 0, "val", "cpu")
        assert emb_path == export_paths(trained, branch, "val", 0)[0]
        assert emb_path.name.endswith(f"_{branch}.npz")
        emb = load_export(emb_path, trained)
        assert present in emb and absent not in emb
