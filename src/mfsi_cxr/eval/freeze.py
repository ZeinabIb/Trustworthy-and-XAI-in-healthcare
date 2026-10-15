"""Freeze guard for the one-time test evaluation (Phase 4; HARD RULE 3: the test split is opened once).

scripts/39_freeze.py writes artifacts/<profile>/rq1_frozen.json (commit, config-file and frozen-input hashes)
only when configs/, src/ and scripts/ have no uncommitted changes. check_frozen() verifies that nothing has
changed since; it returns the TestAccess token that pipeline.collect() needs to read the test split, so the
test split is read only behind this check (scripts/41_run_rq1_final.py). A completed final run is recorded in
artifacts/<profile>/test_opened.json; a further run needs an explicit, logged reason.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mfsi_cxr.utils.config import CONFIG_DIR, REPO_ROOT, config_hash
from mfsi_cxr.utils.io import artifact_dir

MANIFEST = "rq1_frozen.json"
OPENED = "test_opened.json"
TRACKED = ("configs", "src", "scripts")


class FreezeError(RuntimeError):
    """The final run may not start (missing or stale freeze, uncommitted changes, repeated opening)."""


@dataclass(frozen=True)
class TestAccess:
    """Proof that check_frozen() passed; the only way to make pipeline.collect() read the test split."""

    manifest: dict[str, Any]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_inputs(cfg: dict[str, Any]) -> list[Path]:
    """Artifacts the final run reads; their hashes are frozen."""
    adir = artifact_dir(cfg)
    files = []
    for k in cfg["train"]["seeds"]:
        files += [adir / f"mfsi_params_seed{k}.json", adir / f"probe_text_seed{k}.json"]
        files += [adir / f"mfsi_{s}_seed{k}.parquet" for s in ("val", "test")]
        files += [adir / f"emb_{s}_seed{k}.npz" for s in ("train", "val", "test")]
    return files


def config_files() -> list[Path]:
    """Every config and prompt file (a changed prompt is a changed method)."""
    return sorted([*CONFIG_DIR.rglob("*.yaml"), *CONFIG_DIR.rglob("*.txt")])


def uncommitted() -> list[str]:
    return [line for line in _git("status", "--porcelain", "--", *TRACKED).splitlines() if line.strip()]


def write_manifest(cfg: dict[str, Any]) -> Path:
    dirty = uncommitted()
    if dirty:
        raise FreezeError(f"uncommitted changes in {TRACKED}: {dirty[:5]}")
    missing = [p.name for p in frozen_inputs(cfg) if not p.exists()]
    if missing:
        raise FreezeError(f"frozen inputs missing: {missing}")
    manifest = {"profile": cfg["profile"], "commit": _git("rev-parse", "HEAD").strip(),
                "config_hash": config_hash(cfg),
                "config_files": {p.relative_to(REPO_ROOT).as_posix(): sha256(p) for p in config_files()},
                "inputs": {p.name: sha256(p) for p in frozen_inputs(cfg)},
                "created_utc": datetime.datetime.now(datetime.UTC).isoformat()}
    path = artifact_dir(cfg) / MANIFEST
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def openings(cfg: dict[str, Any], purpose: str) -> list[dict[str, Any]]:
    """Recorded test openings for one final analysis ('rq1' = 41, 'rq2' = 61); old records count as rq1."""
    path = artifact_dir(cfg) / OPENED
    if not path.exists():
        return []
    log = json.loads(path.read_text(encoding="utf-8"))
    return [o for o in log["openings"] if o.get("purpose", "rq1") == purpose]


def check_frozen(cfg: dict[str, Any], rerun_reason: str | None = None, purpose: str = "rq1") -> TestAccess:
    """Verify the freeze; raise FreezeError listing every problem. Each final analysis (purpose) opens the
    test split once; a further opening for the same purpose needs a logged reason."""
    path = artifact_dir(cfg) / MANIFEST
    if not path.exists():
        raise FreezeError(f"{MANIFEST} missing: run scripts/39_freeze.py first")
    m = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if m["profile"] != cfg["profile"]:
        problems.append(f"manifest profile {m['profile']} != {cfg['profile']}")
    if uncommitted():
        problems.append(f"uncommitted changes in {TRACKED}")
    changed = _git("diff", "--name-only", m["commit"], "HEAD", "--", *TRACKED).split()
    if changed:
        problems.append(f"code or configs changed since the freeze commit: {changed[:5]}")
    if config_hash(cfg) != m["config_hash"]:
        problems.append("merged config differs from the frozen one")
    now = {p.relative_to(REPO_ROOT).as_posix(): sha256(p) for p in config_files()}
    if now != m["config_files"]:
        problems.append("config files differ from the frozen ones")
    for p in frozen_inputs(cfg):
        if not p.exists() or sha256(p) != m["inputs"].get(p.name):
            problems.append(f"frozen input changed or missing: {p.name}")
    if openings(cfg, purpose) and not rerun_reason:
        problems.append(f"the test split was already opened for {purpose} ({OPENED}); a rerun needs "
                        "--rerun-reason")
    if problems:
        raise FreezeError("; ".join(problems))
    return TestAccess(m)


def record_opening(cfg: dict[str, Any], access: TestAccess, rerun_reason: str | None,
                   purpose: str = "rq1") -> Path:
    path = artifact_dir(cfg) / OPENED
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"openings": []}
    log["openings"].append({"utc": datetime.datetime.now(datetime.UTC).isoformat(),
                            "commit": access.manifest["commit"], "purpose": purpose,
                            "reason": rerun_reason or "first opening"})
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return path
