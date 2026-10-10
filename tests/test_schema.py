from __future__ import annotations

from typing import Any

from mfsi_cxr.data.schema import (
    ALLOWED_INDEX,
    ALLOWED_PRIOR_ONLY,
    FORBIDDEN,
    NON_TEXT,
    audit,
    audit_markdown,
    read_metadata,
)


def test_audit_classes(synthetic_cfg: dict[str, Any]) -> None:
    from pathlib import Path

    meta = read_metadata(Path(synthetic_cfg["data"]["root"]), synthetic_cfg)
    table = audit(meta, synthetic_cfg).set_index("column")["class"]
    assert table["section_history"] == ALLOWED_INDEX
    assert table["section_findings"] == ALLOWED_PRIOR_ONLY
    assert table["section_new_unreviewed"] == FORBIDDEN  # unlisted text: forbidden by default
    assert table["report"] == FORBIDDEN
    assert table["path_to_image"] == NON_TEXT
    assert table["age"] == NON_TEXT


def test_audit_never_prints_text(synthetic_cfg: dict[str, Any]) -> None:
    from pathlib import Path

    meta = read_metadata(Path(synthetic_cfg["data"]["root"]), synthetic_cfg)
    md = audit_markdown(audit(meta, synthetic_cfg), len(meta), "smoke")
    assert "CANARY" not in md
    assert "pre-read" not in md
