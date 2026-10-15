"""Freeze guard, test-split read guard and the generated summary (no real data, no git state needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mfsi_cxr.eval import freeze, pipeline, summary
from mfsi_cxr.utils.io import artifact_dir


@pytest.fixture
def frozen_cfg(smoke_cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Smoke config with dummy frozen inputs and a fake, clean git."""
    adir = artifact_dir(smoke_cfg)
    adir.mkdir(parents=True, exist_ok=True)
    for p in freeze.frozen_inputs(smoke_cfg):
        p.write_text(f"content of {p.name}", encoding="utf-8")
    state = {"dirty": [], "changed": ""}

    def fake_git(*args: str) -> str:
        if args[0] == "status":
            return "\n".join(state["dirty"])
        if args[0] == "rev-parse":
            return "abc123\n"
        if args[0] == "diff":
            return state["changed"]
        raise AssertionError(args)

    monkeypatch.setattr(freeze, "_git", fake_git)
    smoke_cfg["_git_state"] = state
    return smoke_cfg


def test_collect_refuses_test_without_token(smoke_cfg: dict[str, Any]) -> None:
    with pytest.raises(PermissionError):
        pipeline.collect(smoke_cfg, "test", [0])
    with pytest.raises(PermissionError):
        pipeline.collect(smoke_cfg, "test", [0], access=object())  # type: ignore[arg-type]


def test_freeze_roundtrip_and_stale_detection(frozen_cfg: dict[str, Any]) -> None:
    cfg = {k: v for k, v in frozen_cfg.items() if k != "_git_state"}
    state = frozen_cfg["_git_state"]
    with pytest.raises(freeze.FreezeError, match="missing"):
        freeze.check_frozen(cfg)
    freeze.write_manifest(cfg)
    access = freeze.check_frozen(cfg)
    assert access.manifest["commit"] == "abc123"
    target = freeze.frozen_inputs(cfg)[0]
    target.write_text("changed", encoding="utf-8")
    with pytest.raises(freeze.FreezeError, match="frozen input changed"):
        freeze.check_frozen(cfg)
    target.write_text(f"content of {target.name}", encoding="utf-8")
    state["changed"] = "src/mfsi_cxr/mfsi/fit.py"
    with pytest.raises(freeze.FreezeError, match="changed since the freeze"):
        freeze.check_frozen(cfg)
    state["changed"] = ""
    cfg2 = {**cfg, "eval": {**cfg["eval"], "bootstrap_B": cfg["eval"]["bootstrap_B"] + 1}}
    with pytest.raises(freeze.FreezeError, match="merged config"):
        freeze.check_frozen(cfg2)


def test_freeze_refuses_uncommitted_and_second_opening(frozen_cfg: dict[str, Any]) -> None:
    cfg = {k: v for k, v in frozen_cfg.items() if k != "_git_state"}
    state = frozen_cfg["_git_state"]
    state["dirty"] = [" M configs/eval.yaml"]
    with pytest.raises(freeze.FreezeError, match="uncommitted"):
        freeze.write_manifest(cfg)
    state["dirty"] = []
    freeze.write_manifest(cfg)
    access = freeze.check_frozen(cfg)
    log = freeze.record_opening(cfg, access, None)
    with pytest.raises(freeze.FreezeError, match="already opened"):
        freeze.check_frozen(cfg)
    access = freeze.check_frozen(cfg, rerun_reason="bug fix in a figure")
    freeze.record_opening(cfg, access, "bug fix in a figure")
    assert "bug fix in a figure" in Path(log).read_text(encoding="utf-8")


def _tables() -> dict[str, pd.DataFrame]:
    def main_row(score: str, diff: float) -> dict[str, Any]:
        return {"score": score, "aurc": 0.1 + diff, "aurc_lo": 0.08 + diff, "aurc_hi": 0.12 + diff,
                "aurc_random": 0.14, "aurc_oracle": 0.02, "aurc_minus_ref": diff,
                "aurc_minus_ref_lo": diff - 0.01, "aurc_minus_ref_hi": diff + 0.01}
    main = pd.DataFrame([main_row("mfsi", 0.0), main_row("msp", 0.005), main_row("euclid_img", 0.05),
                         main_row("lof_cat", 0.04)])
    cross = pd.DataFrame({"finding": ["No Finding", "Edema"], "stratum": ["two_witness", "one_witness"],
                          "claim_holds": [True, False]})
    spread = pd.DataFrame({"finding": ["No Finding", "Edema"], "stratum": ["two_witness", "one_witness"],
                           "sd": [0.09, 0.001]})
    abl = pd.DataFrame({"variant": ["default", "LOF", "aligned"], "status": ["run", "run", "not run"],
                        "minus_default": [0.0, 0.02, None], "minus_default_lo": [0.0, 0.001, None],
                        "minus_default_hi": [0.0, 0.04, None], "changes_conclusion": [False, True, None]})
    am = pd.DataFrame([{"n_findings": 9, "delta": -0.004, "delta_lo": -0.02, "delta_hi": 0.01,
                        "added_value": False}])
    bm = pd.DataFrame([{"n_findings": 0, "adds_information": False}])
    return {"rq1_main": main, "rq1_cross_modal": cross, "controls_agreement_spread": spread,
            "rq1_ablations": abl, "added_value_a_macro": am, "added_value_b_macro": bm}


def test_summary_answers_the_three_conditions(smoke_cfg: dict[str, Any]) -> None:
    md = summary.build(smoke_cfg, _tables(), {"split": "test", "commit": "abc123", "seeds": [0]})
    assert "THROW-AWAY" in md                                     # smoke is not the full profile
    assert "The claim holds for 1 of 2 findings: No Finding" in md
    assert "- Answer: not inert." in md
    assert "yes, against every Euclidean / LOF baseline" in md
    assert "Rows changing a headline conclusion: LOF" in md and "Rows not run: aligned" in md
    assert "no added value" in md and "no finding enters" in md
    assert "not run in this final run" in md
