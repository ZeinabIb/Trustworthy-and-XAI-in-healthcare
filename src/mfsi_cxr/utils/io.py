"""Artifact paths, `*.meta.json` sidecars and the profile guard.

Every artifact lives under artifacts/<profile>/ and has a sidecar recording profile, config hash,
git hash and seed. Reading an artifact whose sidecar names another profile raises, so a `full` run
can never pick up pilot-tuned values. Data paths stored in artifacts are relative to data.root.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from mfsi_cxr.utils.config import REPO_ROOT
from mfsi_cxr.utils.logging import run_metadata

META_SUFFIX = ".meta.json"


class ArtifactProfileError(RuntimeError):
    """Raised when an artifact's sidecar is missing or belongs to a different profile."""


def _profile_dir(root_key: str, cfg: dict[str, Any]) -> Path:
    root = Path(cfg["paths"][root_key])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / cfg["profile"]


def artifact_dir(cfg: dict[str, Any]) -> Path:
    return _profile_dir("artifacts_root", cfg)


def output_dir(cfg: dict[str, Any]) -> Path:
    return _profile_dir("outputs_root", cfg)


def meta_path(path: Path) -> Path:
    return path.with_name(path.name + META_SUFFIX)


def write_meta(path: Path, cfg: dict[str, Any], seed: int | None = None, **extra: Any) -> Path:
    """Write the sidecar for an artifact that has just been written to `path`."""
    meta = {**run_metadata(cfg, seed), **extra}
    out = meta_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out


def read_meta(path: Path) -> dict[str, Any]:
    mp = meta_path(path)
    if not mp.exists():
        raise ArtifactProfileError(f"missing sidecar {mp.name} for artifact {path}")
    return json.loads(mp.read_text(encoding="utf-8"))


def check_artifact(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Refuse an artifact that is absent or was produced under another profile."""
    if not path.exists():
        raise FileNotFoundError(path)
    meta = read_meta(path)
    if meta.get("profile") != cfg["profile"]:
        raise ArtifactProfileError(
            f"{path} was written by profile {meta.get('profile')!r}, current profile is {cfg['profile']!r}"
        )
    return meta


def write_table(df: pd.DataFrame, path: Path, cfg: dict[str, Any], seed: int | None = None) -> Path:
    """Write a parquet artifact and its sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    write_meta(path, cfg, seed, n_rows=len(df))
    return path


def read_table(path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Read a parquet artifact after the profile guard."""
    check_artifact(path, cfg)
    return pd.read_parquet(path)


def to_data_relative(path: Path, root: Path) -> str:
    """Store a data path relative to data.root, with forward slashes, so the index survives a move."""
    rel = Path(path).resolve().relative_to(Path(root).resolve())
    return PurePosixPath(*rel.parts).as_posix()


def from_data_relative(rel: str, root: Path) -> Path:
    if Path(rel).is_absolute() or PurePosixPath(rel).is_absolute():
        raise ValueError(f"artifact paths must be relative to data.root, got {rel!r}")
    return Path(root) / PurePosixPath(rel)
