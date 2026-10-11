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


@pytest.fixture
def synthetic_cfg(smoke_cfg: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Smoke config whose data root holds a freshly generated synthetic raw dataset."""
    from mfsi_cxr.data.synthetic import make_synthetic_dataset

    smoke_cfg["data"]["root"] = str(tmp_path / "raw")
    make_synthetic_dataset(Path(smoke_cfg["data"]["root"]), smoke_cfg)
    return smoke_cfg


@pytest.fixture
def smoke_artifacts(synthetic_cfg: dict[str, Any]) -> dict[str, Any]:
    """All Phase 1 artifacts built in a temp dir from synthetic data (index, labels, splits, text, RadGraph
    facts, image cache)."""
    from mfsi_cxr.data.images import build_image_cache
    from mfsi_cxr.data.pipeline import build_index, build_splits, build_text_inputs
    from mfsi_cxr.data.radgraph import build_radgraph_facts

    build_index(synthetic_cfg)
    build_splits(synthetic_cfg)
    build_text_inputs(synthetic_cfg)
    build_radgraph_facts(synthetic_cfg)
    build_image_cache(synthetic_cfg)
    return synthetic_cfg


@pytest.fixture
def built(synthetic_cfg: dict[str, Any]) -> dict[str, Any]:
    """Index (all synthetic patients), labels and a TextBuilder."""
    from mfsi_cxr.data.index import assign_study_rank, build_image_table, finalise_index
    from mfsi_cxr.data.labels import labels_per_study, read_label_records
    from mfsi_cxr.data.schema import read_metadata, structural_columns
    from mfsi_cxr.data.text_inputs import TextBuilder

    cfg = synthetic_cfg
    root = Path(cfg["data"]["root"])
    dcfg = cfg["data"]
    table = build_image_table(read_metadata(root, cfg, structural_columns(cfg)), cfg)
    images, _ = assign_study_rank(table, dcfg["study_overlap_policy"])
    index = finalise_index(images)
    labels = labels_per_study(read_label_records(root / dcfg["labels_file"]), index,
                              dcfg["columns"]["image_path"], dcfg["label_aggregation"])
    return {"cfg": cfg, "root": root, "images": images, "index": index, "labels": labels,
            "builder": TextBuilder.from_root(root, index, cfg)}
