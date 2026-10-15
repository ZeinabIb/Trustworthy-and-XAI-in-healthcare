"""Degradations for the degrade-one-modality test (no model, no data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from mfsi_cxr.eval import degrade as dg

RNG = np.random.default_rng(0)


def _item() -> dict:
    return {"images": torch.randn(2, 1, 32, 32), "prior_images": [torch.randn(1, 1, 32, 32)],
            "index_text": "One two three. Four five six! Seven eight?", "prior_texts": ["a b c d", "e f"]}


def test_blur_smooths_only_images() -> None:
    item = _item()
    out = dg.apply("blur", 2, item, RNG)
    assert out["images"].std() < item["images"].std()
    assert out["prior_images"][0].std() < item["prior_images"][0].std()
    assert out["index_text"] == item["index_text"] and out["prior_texts"] == item["prior_texts"]


def test_occlusion_covers_the_fraction_with_the_mean() -> None:
    t = torch.randn(1, 1, 40, 40)
    out = dg.occlude(t, 0.25, np.random.default_rng(1))
    changed = (out != t).float().mean().item()
    assert abs(changed - 0.25) < 0.02 and torch.allclose(out[out != t], t.mean().expand_as(out[out != t]))


def test_text_degradations_leave_images_alone() -> None:
    item = _item()
    out = dg.apply("truncate", 0.5, item, RNG)
    assert out["index_text"] == "One two three. Four" and out["prior_texts"] == ["a b", "e"]
    assert torch.equal(out["images"], item["images"])
    sh = dg.apply("shuffle", None, item, np.random.default_rng(3))
    assert sorted(dg.SENTENCE.split(sh["index_text"])) == sorted(dg.SENTENCE.split(item["index_text"]))
    assert dg.truncate("", 0.25, RNG) == ""


class _Text:
    def __init__(self) -> None:
        self.sec_cols = ["section_clinical_history", "section_history", "section_technique"]


class _Base:
    def __init__(self) -> None:
        self.text = _Text()

    def __len__(self) -> int:
        return 3

    def __getitem__(self, i: int) -> dict:
        return {**_item(), "seen_sections": list(self.text.sec_cols)}


def test_drop_history_only_removes_allowlisted_history_sections() -> None:
    base = _Base()
    ds = dg.DegradedDataset(base, "drop_history", None, 0, ["section_clinical_history", "section_history"])
    assert ds[0]["seen_sections"] == ["section_technique"]
    assert base.text.sec_cols == ["section_clinical_history", "section_history", "section_technique"]


def test_degradation_is_seeded_per_sample() -> None:
    base = _Base()
    a = dg.DegradedDataset(base, "occlude", 0.5, seed=7)
    b = dg.DegradedDataset(base, "occlude", 0.5, seed=7)
    torch.manual_seed(0)
    item = _item()
    x = dg.apply("occlude", 0.5, item, np.random.default_rng([7, 1]))
    y = dg.apply("occlude", 0.5, item, np.random.default_rng([7, 1]))
    assert torch.equal(x["images"], y["images"]) and len(a) == len(b) == 3


def test_summary_and_expectation() -> None:
    n, n_find = 50, 14
    clean = pd.DataFrame({m: RNG.uniform(0.4, 0.6, n * n_find) for m in dg.METRICS})
    deg = clean.copy()
    deg["r_img"] -= 0.2
    deg["R"] -= 0.1
    cfg = {"eval": {"bootstrap_B": 50, "bootstrap_seed": 0, "alpha": 0.05}}
    rows = dg.summarise(clean, deg, np.arange(n).astype(str), cfg)
    by = {r["metric"]: r for r in rows}
    assert by["r_img"]["change"] < -0.19 and by["r_txt"]["change"] == 0
    assert dg.expectation("image", rows) == {"own_margin_falls": True, "other_margin_unchanged": True}
    assert dg.expectation("text", rows) == {"own_margin_falls": False, "other_margin_unchanged": False}
