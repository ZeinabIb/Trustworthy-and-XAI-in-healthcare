from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mfsi_cxr.data.images import (
    CACHE_ARRAY,
    CACHE_DIR,
    CACHE_TABLE,
    ImageCache,
    build_image_cache,
    load_gray,
    resize,
    select_study_images,
    to_tensor,
)
from mfsi_cxr.utils.io import ArtifactProfileError, artifact_dir, write_meta


def test_load_and_resize(synthetic_cfg: dict[str, Any]) -> None:
    root = Path(synthetic_cfg["data"]["root"])
    path = next(root.rglob("*.png"))
    arr = load_gray(path, "png")
    assert arr.dtype == np.uint8 and arr.ndim == 2
    out = resize(arr, 16)
    assert out.shape == (16, 16) and out.dtype == np.uint8


def test_dicom_not_silently_accepted(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        load_gray(tmp_path / "x.dcm", "dicom")


def test_to_tensor() -> None:
    u8 = np.array([[[0, 255], [255, 0]]], dtype=np.uint8)  # (1, 2, 2)
    x = to_tensor(u8, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert x.shape == (1, 3, 2, 2) and x.dtype == torch.float32
    assert torch.allclose(x[0, 0], torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))
    assert torch.equal(x[0, 0], x[0, 2])


def _index(views: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"study_id": [s for s, _ in views], "view": [v for _, v in views],
         "image_path": [f"{s}/view{i}.png" for i, (s, _) in enumerate(views)]}
    )


def test_selection_frontal_max_and_fallback(smoke_cfg: dict[str, Any]) -> None:
    cfg = copy.deepcopy(smoke_cfg)
    cfg["data"]["max_images_per_study"] = 2
    idx = _index([("a", "Frontal"), ("a", "Lateral"), ("a", "Frontal"), ("a", "Frontal"), ("b", "Lateral")])
    sel = select_study_images(idx, cfg)
    assert sel[sel.study_id == "a"]["view"].tolist() == ["Frontal", "Frontal"]
    assert sel[sel.study_id == "b"]["view"].tolist() == ["Lateral"]  # fallback to other views
    cfg["image"]["frontal_fallback"] = "none"
    assert "b" not in set(select_study_images(idx, cfg).study_id)


def test_cache_contents_and_idempotence(smoke_artifacts: dict[str, Any]) -> None:
    cfg = smoke_artifacts
    array_path = artifact_dir(cfg) / CACHE_DIR / CACHE_ARRAY
    arr = np.load(array_path, mmap_mode="r")
    size = cfg["image"]["size"]
    assert arr.shape[1:] == (size, size) and arr.dtype == np.uint8
    del arr  # Windows cannot replace a file that is still memory-mapped
    mtime = array_path.stat().st_mtime_ns
    build_image_cache(cfg)
    assert array_path.stat().st_mtime_ns == mtime, "unchanged inputs must not rebuild the cache"
    cfg = copy.deepcopy(cfg)
    cfg["image"]["size"] = size // 2
    build_image_cache(cfg)
    assert np.load(array_path, mmap_mode="r").shape[1:] == (size // 2, size // 2)


def test_cache_profile_guard(smoke_artifacts: dict[str, Any]) -> None:
    table = artifact_dir(smoke_artifacts) / CACHE_DIR / CACHE_TABLE
    write_meta(table, {**smoke_artifacts, "profile": "pilot"})  # pretend a pilot run wrote it
    with pytest.raises(ArtifactProfileError):
        ImageCache(smoke_artifacts)
