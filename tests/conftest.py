from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mfsi_cxr.utils.config import DATA_ROOT_ENV, PROFILE_ENV, load_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never inherit a profile or data root from the shell."""
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)


@pytest.fixture
def smoke_cfg(tmp_path: Path) -> dict[str, Any]:
    """Smoke config with artifacts/outputs redirected into a temp dir."""
    cfg = load_config("smoke")
    cfg["paths"]["artifacts_root"] = str(tmp_path / "artifacts")
    cfg["paths"]["outputs_root"] = str(tmp_path / "outputs")
    return cfg
