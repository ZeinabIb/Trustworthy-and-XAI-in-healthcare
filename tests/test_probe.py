"""Text-informativeness probe (Phase 2.5) on synthetic embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.models.probe import bootstrap_auroc, probe_table, text_informative_mask


def _split(n: int, separable: bool, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random((n, len(FINDINGS))) < 0.4).astype(float)
    z = rng.normal(size=(n, 32))
    if separable:
        z[:, : len(FINDINGS)] += 4.0 * (2 * y - 1)  # each finding readable from one coordinate
    return {"z_txt": z, "z_img": rng.normal(size=(n, 32)), "y": y, "label_mask": np.ones_like(y),
            "patient_id": np.array([f"p{i // 2}" for i in range(n)])}


def test_separable_text_is_informative(smoke_cfg: dict[str, Any]) -> None:
    train, val = _split(400, True, 0), _split(200, True, 1)
    table = probe_table(train, val, "z_txt", smoke_cfg, seed=0)
    assert (table["auroc"] > 0.99).all() and table["informative"].all()
    out = text_informative_mask(table)
    assert all(out["text_informative"].values()) and not out["no_finding_text_informative"]


def test_noise_is_not_informative_and_is_flagged(smoke_cfg: dict[str, Any]) -> None:
    train, val = _split(400, False, 0), _split(200, False, 1)
    table = probe_table(train, val, "z_txt", smoke_cfg, seed=0)
    assert not table["informative"].any()
    assert text_informative_mask(table)["no_finding_text_informative"]


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = y + rng.normal(scale=1.0, size=300)
    point, lo, hi = bootstrap_auroc(y, score, np.arange(300) // 3, 200, 0.95, 0)
    assert lo < point < hi and 0.6 < point < 0.95


def test_single_class_gives_nan() -> None:
    point, lo, hi = bootstrap_auroc(np.zeros(10), np.arange(10.0), np.arange(10), 10, 0.95, 0)
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)
