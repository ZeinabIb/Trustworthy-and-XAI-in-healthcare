"""Config loading: configs/base.yaml merged with configs/profiles/<profile>.yaml.

A profile may change sizes and speeds only. Any merged key that differs from base and is not in
ALLOWED_PROFILE_KEYS is an error, so method choices can never drift between smoke, pilot and full.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROFILES: tuple[str, ...] = ("smoke", "pilot", "full")
DEFAULT_PROFILE = "pilot"
PROFILE_ENV = "MFSI_PROFILE"
DATA_ROOT_ENV = "MFSI_DATA_ROOT"
SYNTHETIC_PROFILE = "smoke"  # the only profile whose data root is synthetic; it ignores DATA_ROOT_ENV

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
# Shared config files merged into base.yaml, in order, before the profile is applied.
COMPONENT_FILES: tuple[str, ...] = (
    "data.yaml", "finding_synonyms.yaml", "train.yaml", "mfsi.yaml", "eval.yaml",
)

# Dotted keys a profile is allowed to override. Everything else must equal base.yaml.
ALLOWED_PROFILE_KEYS: frozenset[str] = frozenset(
    {
        "data.root",
        "data.pilot_patients",
        "data.image_format",
        "model.weights",          # tiny_random in smoke only (no downloads in tests); DECISIONS 2026-10-11
        "image.size",
        "train.device",
        "train.batch_size",
        "train.num_workers",
        "train.epochs",
        "train.precision",
        "train.seeds",
        "arbiter.model",
        "arbiter.endpoint",
        "eval.bootstrap_B",
        "preflight.min_free_disk_gb",
        "preflight.min_free_gpu_gb",
    }
)


class ConfigError(ValueError):
    """Raised for an unknown profile or a profile that overrides a forbidden key."""


def resolve_profile(profile: str | None) -> str:
    """CLI value, else env MFSI_PROFILE, else 'pilot'."""
    name = profile or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ConfigError(f"unknown profile {name!r}; expected one of {PROFILES}")
    return name


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(flatten(value, dotted + "."))
        else:
            flat[dotted] = value
    return flat


def differing_keys(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    fa, fb = flatten(a), flatten(b)
    return {k for k in fa.keys() | fb.keys() if fa.get(k, _MISSING) != fb.get(k, _MISSING)}


_MISSING = object()


def check_profile_keys(base: dict[str, Any], merged: dict[str, Any]) -> None:
    bad = differing_keys(base, merged) - ALLOWED_PROFILE_KEYS
    if bad:
        raise ConfigError(f"profile overrides keys outside ALLOWED_PROFILE_KEYS: {sorted(bad)}")


def load_shared(config_dir: Path | None = None) -> dict[str, Any]:
    """base.yaml plus the component files (data.yaml, ...): everything every profile shares."""
    cdir = config_dir or CONFIG_DIR
    shared = _read_yaml(cdir / "base.yaml")
    for component in COMPONENT_FILES:
        path = cdir / component
        if path.exists():
            shared = deep_merge(shared, _read_yaml(path))
    return shared


def load_config(profile: str | None = None, config_dir: Path | None = None) -> dict[str, Any]:
    """Merged config for a profile, with `profile` set and `data.root` overridable by env. The smoke profile
    never takes MFSI_DATA_ROOT: it always uses its synthetic fixture root (incident 2026-10-15: a smoke run
    with the env var set wrote synthetic data into the real data root)."""
    name = resolve_profile(profile)
    cdir = config_dir or CONFIG_DIR
    base = load_shared(cdir)
    merged = deep_merge(base, _read_yaml(cdir / "profiles" / f"{name}.yaml"))
    check_profile_keys(base, merged)
    env_root = os.environ.get(DATA_ROOT_ENV)
    if env_root and name != SYNTHETIC_PROFILE:
        merged["data"]["root"] = env_root
    merged["profile"] = name
    return merged


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def data_root(cfg: dict[str, Any]) -> Path | None:
    """Absolute data root; relative roots are taken relative to the repo."""
    root = cfg["data"]["root"]
    if root is None:
        return None
    path = Path(root)
    return path if path.is_absolute() else REPO_ROOT / path
