"""Synthetic data is never written into a folder that may hold real data (incident 2026-10-15)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mfsi_cxr.data.synthetic import SYNTHETIC_MARKER, NotSyntheticRootError, make_synthetic_dataset


def test_refuses_a_non_empty_unmarked_folder(smoke_cfg: dict[str, Any], tmp_path: Path) -> None:
    real = tmp_path / "chexpert_plus"
    real.mkdir()
    (real / "df_chexpert_plus_240401.parquet").write_bytes(b"real data stand-in")
    with pytest.raises(NotSyntheticRootError):
        make_synthetic_dataset(real, smoke_cfg)
    assert (real / "df_chexpert_plus_240401.parquet").read_bytes() == b"real data stand-in"
    assert not (real / SYNTHETIC_MARKER).exists()


def test_writes_into_empty_folder_and_marks_it(smoke_cfg: dict[str, Any], tmp_path: Path) -> None:
    root = tmp_path / "raw"
    make_synthetic_dataset(root, smoke_cfg)
    assert (root / SYNTHETIC_MARKER).exists()
    make_synthetic_dataset(root, smoke_cfg)  # marked: may be regenerated

