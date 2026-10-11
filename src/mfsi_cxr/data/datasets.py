"""SampleDataset and collate_fn for ragged bags (Phase 1.8).

One sample = (patient, index study) (decision D1). Images: the index study's cached images plus the
images of the same prior studies the text branch uses (<= max_priors, most recent first). Text comes
from the leakage-safe TextBuilder only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from mfsi_cxr.constants import FINDINGS, NEGATIVE, POSITIVE, SPLITS, UNCERTAIN
from mfsi_cxr.data.images import ImageCache, to_tensor
from mfsi_cxr.data.text_inputs import TextBuilder
from mfsi_cxr.utils.config import data_root
from mfsi_cxr.utils.io import artifact_dir, read_table

UNCERTAIN_POLICIES = ("u_ignore", "u_ones")


def label_target_and_mask(values: np.ndarray, policy: str) -> tuple[np.ndarray, np.ndarray]:
    """values in {1,0,-1,NaN}. u_ignore: train on {0,1} only. u_ones: uncertain counts as positive."""
    if policy == "u_ignore":
        target = values == POSITIVE
        mask = (values == POSITIVE) | (values == NEGATIVE)
    elif policy == "u_ones":
        target = (values == POSITIVE) | (values == UNCERTAIN)
        mask = target | (values == NEGATIVE)
    else:
        raise ValueError(f"unknown uncertain policy {policy!r}; expected one of {UNCERTAIN_POLICIES}")
    return target.astype(np.float32), mask.astype(np.float32)


class SampleDataset(Dataset):
    def __init__(self, cfg: dict[str, Any], split: str, text_builder: TextBuilder | None = None,
                 text_shuffle_seed: int | None = None) -> None:
        """text_shuffle_seed: SANITY CHECK ONLY (Phase 2.6). Each sample gets the whole text (index text and
        prior reports) of another sample of the split, so text no longer belongs to its labels."""
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        adir = artifact_dir(cfg)
        index = read_table(adir / "index.parquet", cfg)
        labels = read_table(adir / "labels.parquet", cfg)
        splits = read_table(adir / "splits.parquet", cfg)
        texts = read_table(adir / "text_inputs.parquet", cfg)

        patients = set(splits.loc[splits["split"] == split, "patient_id"])
        self.samples = texts[texts["patient_id"].isin(patients)].sort_values(["patient_id", "study_rank"])
        self.samples = self.samples.reset_index(drop=True)
        self.labels = labels.set_index("study_id")[list(FINDINGS)]
        self.policy = cfg["labels"]["uncertain_policy"]
        self.channels, self.value_range = cfg["image"]["channels"], cfg["image"]["value_range"]
        self.cache = ImageCache(cfg)
        self.text = text_builder or TextBuilder.from_root(data_root(cfg), index, cfg)
        n = len(self.samples)
        self.text_source = (np.random.default_rng(text_shuffle_seed).permutation(n)
                            if text_shuffle_seed is not None else np.arange(n))

    def __len__(self) -> int:
        return len(self.samples)

    def _study_images(self, study_id: str) -> torch.Tensor:
        return to_tensor(self.cache.study(study_id), self.channels, self.value_range)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.samples.iloc[i]
        text = self.text.build(row["patient_id"], row["study_id"])
        if list(text.prior_study_ids) != list(row["prior_study_ids"]):
            raise RuntimeError(f"text_inputs.parquet is stale for {row['study_id']}; rerun 14")
        prior_images = [self._study_images(s) for s in text.prior_study_ids]
        source = int(self.text_source[i])
        if source != i:
            other = self.samples.iloc[source]
            text = self.text.build(other["patient_id"], other["study_id"])
        if row["study_id"] in self.labels.index:
            values = self.labels.loc[row["study_id"]].to_numpy(dtype=np.float64)
        else:
            values = np.full(len(FINDINGS), np.nan)
        target, mask = label_target_and_mask(values, self.policy)
        images = self._study_images(row["study_id"])
        return {
            "images": images,
            "prior_images": prior_images,
            "index_text": text.index_text,
            "prior_texts": list(text.prior_texts),
            "labels": torch.from_numpy(target),
            "label_values": torch.from_numpy(values.astype(np.float32)),  # {1,0,-1,NaN}, for the exports
            "label_mask": torch.from_numpy(mask),
            "patient_id": row["patient_id"],
            "study_id": row["study_id"],
            "n_studies": 1 + len(prior_images),
            "n_images": int(images.shape[0] + sum(p.shape[0] for p in prior_images)),
            "text_len": int(text.text_len),
            "n_priors": len(prior_images),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad ragged bags. Masks are True where a real image / prior study exists."""
    b = len(batch)
    shape = next((x["images"].shape[1:] for x in batch if x["images"].shape[0]), None)
    if shape is None:
        raise ValueError("batch has no images at all")
    k = max(x["images"].shape[0] for x in batch)
    p = max(len(x["prior_images"]) for x in batch)
    kp = max((t.shape[0] for x in batch for t in x["prior_images"]), default=0)

    images = torch.zeros(b, k, *shape)
    image_mask = torch.zeros(b, k, dtype=torch.bool)
    prior_images = torch.zeros(b, p, kp, *shape)
    prior_image_mask = torch.zeros(b, p, kp, dtype=torch.bool)
    prior_study_mask = torch.zeros(b, p, dtype=torch.bool)
    pt = max(len(x["prior_texts"]) for x in batch)
    prior_text_mask = torch.zeros(b, pt, dtype=torch.bool)
    for i, x in enumerate(batch):
        prior_text_mask[i, : len(x["prior_texts"])] = True
        n = x["images"].shape[0]
        images[i, :n] = x["images"]
        image_mask[i, :n] = True
        for j, t in enumerate(x["prior_images"]):
            prior_images[i, j, : t.shape[0]] = t
            prior_image_mask[i, j, : t.shape[0]] = True
            prior_study_mask[i, j] = True
    return {
        "images": images,
        "image_mask": image_mask,
        "prior_images": prior_images,
        "prior_image_mask": prior_image_mask,
        "prior_study_mask": prior_study_mask,
        "index_text": [x["index_text"] for x in batch],
        "prior_texts": [x["prior_texts"] for x in batch],
        "prior_text_mask": prior_text_mask,
        "labels": torch.stack([x["labels"] for x in batch]),
        "label_mask": torch.stack([x["label_mask"] for x in batch]),
        "label_values": torch.stack([x["label_values"] for x in batch]),
        "patient_id": [x["patient_id"] for x in batch],
        "study_id": [x["study_id"] for x in batch],
        "n_studies": torch.tensor([x["n_studies"] for x in batch]),
        "n_images": torch.tensor([x["n_images"] for x in batch]),
        "text_len": torch.tensor([x["text_len"] for x in batch]),
        "n_priors": torch.tensor([x["n_priors"] for x in batch]),
    }
