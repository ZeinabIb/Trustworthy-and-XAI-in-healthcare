"""Final report: pre-stated outcome rules and loud failure on missing artifacts."""

from __future__ import annotations

import pandas as pd
import pytest

from mfsi_cxr.eval import report


def _main(diffs: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    rows = {"mfsi": (0, 0, 0), "mfsi_img": (0.01, -1, 1), **diffs}
    return pd.DataFrame([{"score": s, "aurc_minus_ref": d, "aurc_minus_ref_lo": lo, "aurc_minus_ref_hi": hi}
                         for s, (d, lo, hi) in rows.items()])


def test_h1_rule() -> None:
    better = {"msp": (0.02, 0.01, 0.03), "euclid_img": (0.05, 0.02, 0.07)}
    assert report.h1(_main(better))["outcome"] == "supported"
    assert report.h1(_main({**better, "lof_cat": (0.0, -0.01, 0.01)}))["outcome"] == "inconclusive"
    assert report.h1(_main({**better, "msp": (-0.03, -0.05, -0.01)}))["outcome"] == "not supported"
    assert "mfsi_img" not in report.h1(_main(better))["detail"]   # single-modality MFSI is H1b, not H1


def _cross(rows: list[tuple[str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["finding", "stratum", "claim_holds"])


def test_h1b_rule() -> None:
    assert report.h1b(_cross([("A", "one_witness", False)]))["outcome"].startswith("not assessable")
    both = _cross([("A", "two_witness", True), ("B", "two_witness", True)])
    assert report.h1b(both)["outcome"] == "supported"
    some = _cross([("A", "two_witness", True), ("B", "two_witness", False)])
    assert report.h1b(some)["outcome"] == "partially supported"
    assert report.h1b(_cross([("A", "two_witness", False)]))["outcome"] == "not supported"


def test_h2a_rule() -> None:
    row = {"diff": 0.1, "lo": 0.02, "hi": 0.2, "n": 50, "H2a_supported": True}
    assert report.h2a(pd.DataFrame([row]))["outcome"] == "supported"
    fails = pd.DataFrame([{**row, "lo": -0.01, "H2a_supported": False}])
    assert report.h2a(fails)["outcome"] == "not supported"


def test_missing_artifacts_fail_loudly(tmp_path) -> None:
    (tmp_path / "rq1_main.csv").write_text("score\nmfsi\n", encoding="utf-8")
    with pytest.raises(report.MissingArtifactError, match="rq2_listen"):
        report.load_tables(tmp_path, seed=0)
