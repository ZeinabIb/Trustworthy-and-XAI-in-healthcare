from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mfsi_cxr.constants import FINDINGS, SPLITS
from mfsi_cxr.data.datasets import SampleDataset, collate_fn, label_target_and_mask


def test_label_policies() -> None:
    v = np.array([1.0, 0.0, -1.0, np.nan])
    t, m = label_target_and_mask(v, "u_ignore")
    assert t.tolist() == [1, 0, 0, 0] and m.tolist() == [1, 1, 0, 0]
    t, m = label_target_and_mask(v, "u_ones")
    assert t.tolist() == [1, 0, 1, 0] and m.tolist() == [1, 1, 1, 0]
    with pytest.raises(ValueError):
        label_target_and_mask(v, "u_zeros_maybe")


def test_item_contents(smoke_artifacts: dict[str, Any]) -> None:
    cfg = smoke_artifacts
    ds = SampleDataset(cfg, "train")
    size, k_max = cfg["image"]["size"], cfg["data"]["max_images_per_study"]
    assert len(ds) > 0
    for i in range(len(ds)):
        x = ds[i]
        assert x["images"].ndim == 4 and x["images"].shape[1:] == (cfg["image"]["channels"], size, size)
        assert 1 <= x["images"].shape[0] <= k_max
        assert len(x["prior_images"]) == len(x["prior_texts"]) <= cfg["data"]["max_priors"]
        assert x["labels"].shape == (len(FINDINGS),) and x["label_mask"].shape == (len(FINDINGS),)
        assert x["n_studies"] == 1 + len(x["prior_images"])
        assert "CANARY" not in x["index_text"]


def test_split_datasets_are_patient_disjoint(smoke_artifacts: dict[str, Any]) -> None:
    patients = {s: set(SampleDataset(smoke_artifacts, s).samples["patient_id"]) for s in SPLITS}
    assert not patients["train"] & patients["val"]
    assert not patients["train"] & patients["test"]
    assert not patients["val"] & patients["test"]


def test_collate_shapes_and_masks(smoke_artifacts: dict[str, Any]) -> None:
    ds = SampleDataset(smoke_artifacts, "train")
    batch = next(iter(DataLoader(ds, batch_size=len(ds), collate_fn=collate_fn)))
    items = [ds[i] for i in range(len(ds))]
    b = len(ds)
    k = max(x["images"].shape[0] for x in items)
    p = max(len(x["prior_images"]) for x in items)
    assert batch["images"].shape[:2] == (b, k)
    assert batch["prior_images"].shape[:2] == (b, p)
    assert batch["labels"].shape == (b, len(FINDINGS))
    for i, x in enumerate(items):
        n = x["images"].shape[0]
        assert batch["image_mask"][i].sum() == n
        assert torch.equal(batch["images"][i, :n], x["images"])
        assert (batch["images"][i, n:] == 0).all()
        assert batch["prior_study_mask"][i].sum() == len(x["prior_images"])
        for j, t in enumerate(x["prior_images"]):
            assert batch["prior_image_mask"][i, j].sum() == t.shape[0]


def test_collate_without_priors(smoke_artifacts: dict[str, Any]) -> None:
    ds = SampleDataset(smoke_artifacts, "train")
    no_prior = [ds[i] for i in range(len(ds)) if not ds[i]["prior_images"]][:2]
    assert no_prior
    batch = collate_fn(no_prior)
    assert batch["prior_images"].shape[1] == 0 and batch["prior_study_mask"].shape == (len(no_prior), 0)
