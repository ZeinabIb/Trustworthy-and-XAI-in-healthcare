"""Preflight checks run before data scripts (stage 'data') and before training (stage 'train').

Training refuses to start unless `run_preflight(cfg, "train")` passes. Later phases extend the
artifact consistency checks; Step 0 provides the frame and the checks that need no data.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mfsi_cxr.utils.config import REPO_ROOT, data_root
from mfsi_cxr.utils.device import device_from_config
from mfsi_cxr.utils.io import ArtifactProfileError, artifact_dir, check_artifact

STAGES = ("data", "train")
# Phase 1 artifacts that training depends on (file names fixed by 00_OVERVIEW.md).
TRAIN_INPUT_ARTIFACTS = (
    "index.parquet", "labels.parquet", "splits.parquet", "text_inputs.parquet", "radgraph_facts.parquet",
)
CANARY_TEST = REPO_ROOT / "tests" / "test_leakage.py"
GB = 1024**3

OK, FAIL, SKIP = "ok", "fail", "skip"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str = ""


def check_data_root(cfg: dict[str, Any]) -> CheckResult:
    root = data_root(cfg)
    if root is None:
        return CheckResult("data_root", FAIL, "data.root is not set (profile yaml or MFSI_DATA_ROOT)")
    if not root.is_dir():
        return CheckResult("data_root", FAIL, f"{root} does not exist")
    return CheckResult("data_root", OK, str(root))


def check_disk(cfg: dict[str, Any]) -> CheckResult:
    target = artifact_dir(cfg)
    probe = next((p for p in (target, *target.parents) if p.exists()), REPO_ROOT)
    free_gb = shutil.disk_usage(probe).free / GB
    need = cfg["preflight"]["min_free_disk_gb"]
    status = OK if free_gb >= need else FAIL
    return CheckResult("disk", status, f"{free_gb:.1f} GB free at {probe}, need {need} GB")


def check_artifacts(cfg: dict[str, Any]) -> list[CheckResult]:
    results = []
    adir = artifact_dir(cfg)
    for name in TRAIN_INPUT_ARTIFACTS:
        path = adir / name
        try:
            check_artifact(path, cfg)
        except FileNotFoundError:
            results.append(CheckResult(f"artifact:{name}", FAIL, f"missing {path}"))
        except ArtifactProfileError as exc:
            results.append(CheckResult(f"artifact:{name}", FAIL, str(exc)))
        else:
            results.append(CheckResult(f"artifact:{name}", OK))
    return results


def check_canary(test_file: Path = CANARY_TEST) -> CheckResult:
    if not test_file.exists():
        return CheckResult("canary_leakage", FAIL, f"{test_file.name} not found")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_file)], cwd=REPO_ROOT, capture_output=True, text=True
    )
    status = OK if proc.returncode == 0 else FAIL
    return CheckResult("canary_leakage", status, proc.stdout.strip().splitlines()[-1] if proc.stdout else "")


def check_gpu(cfg: dict[str, Any]) -> CheckResult:
    device = device_from_config(cfg)
    if device != "cuda":
        return CheckResult("gpu_memory", SKIP, f"device is {device}")
    import torch

    free, _ = torch.cuda.mem_get_info()
    need = cfg["preflight"]["min_free_gpu_gb"]
    status = OK if free / GB >= need else FAIL
    return CheckResult("gpu_memory", status, f"{free / GB:.1f} GB free, need {need} GB")


def run_preflight(cfg: dict[str, Any], stage: str) -> list[CheckResult]:
    """Config validity is already enforced by load_config (profile-difference rule)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    results = [CheckResult("config", OK, f"profile {cfg['profile']}"), check_data_root(cfg), check_disk(cfg)]
    if stage == "train":
        results += check_artifacts(cfg)
        results.append(check_canary())
        results.append(check_gpu(cfg))
    return results


def passed(results: list[CheckResult]) -> bool:
    return all(r.status != FAIL for r in results)
