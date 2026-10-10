from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from mfsi_cxr.utils.io import (
    ArtifactProfileError,
    artifact_dir,
    check_artifact,
    from_data_relative,
    read_meta,
    to_data_relative,
    write_meta,
)


def _write_artifact(cfg: dict[str, Any], name: str = "index.parquet") -> Path:
    path = artifact_dir(cfg) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic")
    write_meta(path, cfg, seed=0)
    return path


def test_artifacts_live_under_profile_folder(smoke_cfg: dict[str, Any]) -> None:
    assert artifact_dir(smoke_cfg).name == "smoke"


def test_sidecar_records_run_metadata(smoke_cfg: dict[str, Any]) -> None:
    meta = read_meta(_write_artifact(smoke_cfg))
    assert meta["profile"] == "smoke"
    assert meta["seed"] == 0
    assert {"config_hash", "git", "env"} <= meta.keys()


def test_guard_accepts_same_profile(smoke_cfg: dict[str, Any]) -> None:
    assert check_artifact(_write_artifact(smoke_cfg), smoke_cfg)["profile"] == "smoke"


def test_guard_rejects_other_profile(smoke_cfg: dict[str, Any]) -> None:
    path = _write_artifact(smoke_cfg)
    other = {**smoke_cfg, "profile": "full"}
    with pytest.raises(ArtifactProfileError, match="smoke"):
        check_artifact(path, other)


def test_guard_rejects_missing_sidecar(smoke_cfg: dict[str, Any]) -> None:
    path = artifact_dir(smoke_cfg) / "labels.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic")
    with pytest.raises(ArtifactProfileError, match="sidecar"):
        check_artifact(path, smoke_cfg)


def test_relative_paths_survive_moving_data_root(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    image = old_root / "png" / "patient00001" / "study1" / "view1_frontal.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic")
    rel = to_data_relative(image, old_root)

    new_root = tmp_path / "moved" / "elsewhere"
    shutil.copytree(old_root, new_root)
    assert rel == "png/patient00001/study1/view1_frontal.png"
    assert to_data_relative(new_root / "png/patient00001/study1/view1_frontal.png", new_root) == rel
    assert from_data_relative(rel, new_root).read_bytes() == b"synthetic"


def test_absolute_paths_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        from_data_relative(str(tmp_path / "x.png"), tmp_path)
