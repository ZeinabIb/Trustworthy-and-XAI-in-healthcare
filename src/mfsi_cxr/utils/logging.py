"""Logging and run metadata (config hash, git hash, package versions) saved with every run."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

from mfsi_cxr.utils.config import REPO_ROOT, config_hash

TRACKED_PACKAGES = (
    "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "statsmodels", "matplotlib",
    "pyyaml", "pydicom", "pillow", "torch", "torchvision", "transformers",
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None}
    info: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def env_report() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        **torch_info(),
    }


def run_metadata(cfg: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    return {
        "profile": cfg["profile"],
        "config_hash": config_hash(cfg),
        "seed": seed,
        "git": git_info(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "env": env_report(),
    }
