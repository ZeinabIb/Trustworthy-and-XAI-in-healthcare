from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from mfsi_cxr.utils.config import REPO_ROOT
from mfsi_cxr.utils.io import artifact_dir, write_meta
from mfsi_cxr.utils.preflight import (
    FAIL,
    OK,
    TRAIN_INPUT_ARTIFACTS,
    check_artifacts,
    check_canary,
    passed,
    run_preflight,
)

PREFLIGHT = REPO_ROOT / "scripts" / "00_preflight.py"


def _by_name(results: list) -> dict[str, Any]:
    return {r.name: r for r in results}


def _write_inputs(cfg: dict[str, Any], skip: str | None = None, profile: str | None = None) -> None:
    meta_cfg = {**cfg, "profile": profile or cfg["profile"]}
    adir = artifact_dir(cfg)
    adir.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_INPUT_ARTIFACTS:
        if name == skip:
            continue
        (adir / name).write_bytes(b"synthetic")
        write_meta(adir / name, meta_cfg)


def test_smoke_data_stage_passes(smoke_cfg: dict[str, Any]) -> None:
    assert passed(run_preflight(smoke_cfg, "data"))


def test_absent_data_root_fails(smoke_cfg: dict[str, Any], tmp_path: Path) -> None:
    smoke_cfg["data"]["root"] = str(tmp_path / "does_not_exist")
    results = _by_name(run_preflight(smoke_cfg, "data"))
    assert results["data_root"].status == FAIL


def test_unset_data_root_fails(smoke_cfg: dict[str, Any]) -> None:
    smoke_cfg["data"]["root"] = None
    assert not passed(run_preflight(smoke_cfg, "data"))


def test_missing_split_file_fails(smoke_cfg: dict[str, Any]) -> None:
    _write_inputs(smoke_cfg, skip="splits.parquet")
    results = _by_name(check_artifacts(smoke_cfg))
    assert results["artifact:splits.parquet"].status == FAIL
    assert results["artifact:index.parquet"].status == OK


def test_artifact_from_other_profile_fails(smoke_cfg: dict[str, Any]) -> None:
    _write_inputs(smoke_cfg, profile="pilot")
    assert all(r.status == FAIL for r in check_artifacts(smoke_cfg))


def test_missing_canary_test_fails(tmp_path: Path) -> None:
    assert check_canary(tmp_path / "test_leakage.py").status == FAIL


def test_train_stage_fails_without_phase1_artifacts(smoke_cfg: dict[str, Any]) -> None:
    assert not passed(run_preflight(smoke_cfg, "train"))


def test_cli_exit_codes() -> None:
    ok = subprocess.run([sys.executable, str(PREFLIGHT), "--profile", "smoke", "--stage", "data"],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "PASSED" in ok.stdout
