"""Image loading, per-study image selection, the uint8 cache and the tensor transform (Phase 1.7).

PNG (and synthetic PNG) only for now. DICOM (rescale, MONOCHROME1/2 polarity, windowing tuned on val)
waits for the PNG-vs-DICOM decision (question 9). Images are cached once, resized, as uint8 in one
memory-mapped array under artifacts/<profile>/img_cache/, so training never decodes PNGs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import torch

from mfsi_cxr.utils.config import data_root
from mfsi_cxr.utils.io import (
    artifact_dir,
    check_artifact,
    from_data_relative,
    read_meta,
    read_table,
    write_table,
)
from mfsi_cxr.utils.logging import get_logger

log = get_logger("images")
PIL_FORMATS = ("png", "synthetic")
FALLBACKS = ("all_views", "none")
CACHE_DIR = "img_cache"
CACHE_ARRAY = "images_u8.npy"
CACHE_TABLE = "img_cache_index.parquet"
# The index keeps the metadata's image path (.jpg); the files on disk carry the format's suffix.
FILE_SUFFIX = {"png": ".png", "synthetic": ".png"}


def local_image_path(rel: str, root: Path, image_format: str) -> Path:
    """Index image path (relative, any suffix) -> the image file under data.root."""
    if image_format not in FILE_SUFFIX:
        raise NotImplementedError(f"image format {image_format!r} not supported yet (PNG vs DICOM pending)")
    return from_data_relative(str(PurePosixPath(rel).with_suffix(FILE_SUFFIX[image_format])), root)


def load_gray(path: Path, image_format: str) -> np.ndarray:
    """Read one image as a 2-D uint8 array."""
    if image_format not in PIL_FORMATS:
        raise NotImplementedError(f"image format {image_format!r} not supported yet (PNG vs DICOM pending)")
    from PIL import Image

    with Image.open(path) as img:
        if img.mode == "I;16" or img.mode == "I":
            arr = np.asarray(img, dtype=np.float64)
            span = arr.max() - arr.min()
            return ((arr - arr.min()) / span * 255).astype(np.uint8) if span else np.zeros_like(arr, np.uint8)
        return np.asarray(img.convert("L"), dtype=np.uint8)


def resize(arr: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def to_tensor(u8: np.ndarray, channels: int, value_range: list[float]) -> torch.Tensor:
    """(..., H, W) uint8 -> (..., C, H, W) float32, [0, 255] mapped linearly to value_range.

    The XRV checkpoints take one channel in [-1024, 1024] (`image.channels`, `image.value_range`).
    """
    lo, hi = value_range
    x = torch.from_numpy(np.ascontiguousarray(u8)).float().mul_((hi - lo) / 255.0).add_(lo)
    return x.unsqueeze(-3).expand(*x.shape[:-2], channels, *x.shape[-2:]).contiguous()


def select_study_images(index: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Up to max_images_per_study images per study, frontal first when frontal_only (sorted by path)."""
    dcfg = cfg["data"]
    fallback = cfg["image"]["frontal_fallback"]
    if fallback not in FALLBACKS:
        raise ValueError(f"unknown frontal_fallback {fallback!r}")
    df = index.sort_values(["study_id", "image_path"])
    if dcfg["frontal_only"]:
        frontal = df["view"] == dcfg["frontal_value"]
        has_frontal = frontal.groupby(df["study_id"]).transform("any")
        keep = frontal | (~has_frontal if fallback == "all_views" else False)
        df = df[keep]
    return df.groupby("study_id", sort=False).head(dcfg["max_images_per_study"]).reset_index(drop=True)


def _cache_key(paths: list[str], size: int, image_format: str) -> str:
    h = hashlib.sha256(f"{size}|{image_format}|".encode())
    for p in paths:
        h.update(p.encode() + b"\n")
    return h.hexdigest()[:16]


def build_image_cache(cfg: dict[str, Any]) -> Path:
    """Idempotent: does nothing if a cache with the same image list, size and format exists."""
    adir = artifact_dir(cfg)
    cdir = adir / CACHE_DIR
    table_path, array_path = cdir / CACHE_TABLE, cdir / CACHE_ARRAY
    selected = select_study_images(read_table(adir / "index.parquet", cfg), cfg)
    size, fmt = cfg["image"]["size"], cfg["data"]["image_format"]
    key = _cache_key(selected["image_path"].tolist(), size, fmt)
    if table_path.exists() and array_path.exists():
        check_artifact(table_path, cfg)
        if read_meta(table_path).get("cache_key") == key:
            log.info("image cache up to date (%d images)", len(selected))
            return array_path

    cdir.mkdir(parents=True, exist_ok=True)
    tmp = cdir / (CACHE_ARRAY + ".tmp")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(selected), size, size))
    root = data_root(cfg)
    for row, rel in enumerate(selected["image_path"]):
        arr[row] = resize(load_gray(local_image_path(rel, root, fmt), fmt), size)
    arr.flush()
    del arr
    os.replace(tmp, array_path)  # atomic: a killed job never leaves a half-written cache in place
    table = selected[["image_path", "study_id"]].assign(row=np.arange(len(selected)))
    write_table(table, table_path, cfg, cache_key=key, size=size)
    log.info("cached %d images at %dx%d", len(selected), size, size)
    return array_path


class ImageCache:
    """Read-only view of the cache: study_id -> (K, S, S) uint8."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        cdir = artifact_dir(cfg) / CACHE_DIR
        table = read_table(cdir / CACHE_TABLE, cfg)
        self.path = cdir / CACHE_ARRAY
        self._array: np.ndarray | None = None
        if len(self.array) != len(table):
            raise RuntimeError("image cache array and index disagree; rerun 13_cache_images")
        self.rows = table.groupby("study_id", sort=False)["row"].apply(list).to_dict()

    @property
    def array(self) -> np.ndarray:
        if self._array is None:  # opened lazily, so DataLoader workers each map the file themselves
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_array": None}  # never pickle (copy) the whole cache into workers

    def study(self, study_id: str) -> np.ndarray:
        return np.asarray(self.array[self.rows.get(study_id, [])])
