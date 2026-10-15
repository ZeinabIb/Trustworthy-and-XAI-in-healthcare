"""Final report (Phase 6, scripts/99_make_report.py): REPORT.md, the plan's table and figure set, and a
manifest, from the FINAL (test) artifacts only. Every number is read from a table; a missing artifact is an
error.

Pre-stated outcome rules (proposal; plans 4 and 6):
- H1  MFSI has lower macro AURC than each baseline: supported iff every baseline - MFSI CI is above 0;
      not supported iff any is below 0; else inconclusive.
- H1b cross-modal MFSI beats the best single modality (pre-stated per-finding rule): supported iff it holds
      for every two-witness finding; partially supported iff for some; not supported iff for none; not
      assessable without a two-witness finding.
- H2a flip(shuffle) - flip(re-run) CI above 0: supported; otherwise not supported (the arbiter was not
      shown to use the signal).
- H2b the verified-claim share is reported (descriptive; the proposal sets no threshold).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from mfsi_cxr.utils.config import config_hash
from mfsi_cxr.utils.logging import git_info, package_versions

BASELINE_PREFIXES = ("msp", "ensemble_var", "tied_maha_", "euclid_", "lof_")
REQUIRED_TABLES = (
    "rq1_main", "rq1_per_finding", "rq1_cross_modal", "rq1_ablations", "controls_agreement_spread",
    "added_value_a_macro", "added_value_b_macro", "rq2_help", "rq2_listen", "rq2_listen_primary",
    "rq2_holdup", "rq2_sample", "label_prevalence", "backbone_auroc", "masking_check",
)
REQUIRED_FIGURES = ("risk_coverage", "reliability_diagrams", "volume_confound", "r_deciles", "flip_rates")
PRODUCER = {  # first matching prefix wins; dev outputs (*_dev*, *_dev_*) are matched before
    "rq1_dev_": "40/42 (dev)", "controls_dev_": "43/45 (dev)", "added_value_dev_": "44/42 (dev)",
    "rq2_dev_": "60 (dev)", "arbiter_dev_": "52 (dev)", "arbiter_examples": "53_render_examples.py",
    "rq1_": "41_run_rq1_final.py", "controls_": "41_run_rq1_final.py", "added_value_": "41_run_rq1_final.py",
    "RQ1_summary": "41_run_rq1_final.py", "rq2_": "61_run_rq2_final.py",
    "label_prevalence": "11_build_index.py",
    "backbone_auroc": "20_train.py", "masking_check": "23_masking_check.py", "probe_": "22_probe.py",
    "agreement_spread": "32_agreement_spread.py", "mfsi_fit": "30_fit_mfsi.py", "REPORT": "99_make_report.py",
    "pipeline_overview": "99_make_report.py", "risk_coverage": "41 (40 for *_dev)",
    "reliability_diagrams": "41 (40 for *_dev)", "volume_confound": "41 (43 for *_dev)",
    "r_deciles": "41 (43 for *_dev)", "degrade_test": "41 (45 for *_dev)", "flip_rates": "61 (60 for *_dev)",
    "cannot_decide_vs_R": "61 (60 for *_dev)", "claims_verified": "61 (60 for *_dev)",
}
THREATS = (
    "Labels come from the CheXpert labeler on the reports (a reference standard, not ground truth); blanks "
    "are scored as negatives (pre-stated) and uncertain labels are excluded.",
    "Text is informative only for the findings the probe passed; for one-witness findings MFSI reduces to "
    "the image margin, so the cross-modal claim can only be tested on two-witness findings.",
    "The degrade test showed that the plain margin rises for out-of-distribution images; the pre-registered "
    "typicality variant is reported as a sensitivity row only.",
    "The arbiter is not bit-reproducible for a fixed seed on the GPU build; reruns are reproducible through "
    "the call cache, and the noise floor uses fresh calls with different seeds.",
    "VERIFIED means the arbiter used the cited evidence, not that a claim is true; whether tagging protects "
    "clinicians from over-reliance needs a human study and is not claimed.",
    "One local open-weight model and one prompt version; no second-LLM robustness check in this run.",
)
AI_USE = ("Code for this pipeline was written with an AI coding assistant (Claude, Anthropic) under the "
          "author's direction and review; every decision is logged in docs/DECISIONS.md. No AI system wrote "
          "or filled any result: all numbers above are read by scripts from the run's artifacts.")


class MissingArtifactError(FileNotFoundError):
    """A table or figure the report needs is missing (no silent empty tables)."""


def load_tables(tdir: Path, seed: int) -> dict[str, pd.DataFrame]:
    missing = [n for n in REQUIRED_TABLES if not (tdir / f"{n}.csv").exists()]
    probe = tdir / f"probe_seed{seed}.csv"
    if not probe.exists():
        missing.append(probe.name)
    if missing:
        raise MissingArtifactError(f"report artifacts missing in {tdir}: {missing}")
    t = {n: pd.read_csv(tdir / f"{n}.csv") for n in REQUIRED_TABLES}
    t["probe_text"] = pd.read_csv(probe)
    if (tdir / "controls_degrade.csv").exists():
        t["controls_degrade"] = pd.read_csv(tdir / "controls_degrade.csv")
    return t


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def h1(main: pd.DataFrame) -> dict[str, Any]:
    m = main.set_index("score")
    base = [s for s in m.index if s.startswith(BASELINE_PREFIXES)]
    above = [s for s in base if m.loc[s, "aurc_minus_ref_lo"] > 0]
    below = [s for s in base if m.loc[s, "aurc_minus_ref_hi"] < 0]
    if base and len(above) == len(base):
        outcome = "supported"
    else:
        outcome = "not supported" if below else "inconclusive"
    parts = [f"{s}: {m.loc[s, 'aurc_minus_ref']:.4f} "
             f"{_ci(m.loc[s, 'aurc_minus_ref_lo'], m.loc[s, 'aurc_minus_ref_hi'])}" for s in base]
    return {"outcome": outcome, "detail": "baseline - MFSI macro AURC: " + "; ".join(parts)}


def h1b(cross: pd.DataFrame) -> dict[str, Any]:
    two = cross[cross["stratum"] == "two_witness"]
    holds = two.loc[two["claim_holds"], "finding"].tolist()
    if not len(two):
        outcome = "not assessable (no two-witness finding)"
    elif len(holds) == len(two):
        outcome = "supported"
    else:
        outcome = "partially supported" if holds else "not supported"
    return {"outcome": outcome, "detail": f"claim holds for {len(holds)} of {len(two)} two-witness findings "
                                          f"({', '.join(holds) or 'none'})"}


def h2a(primary: pd.DataFrame) -> dict[str, Any]:
    p = primary.iloc[0]
    return {"outcome": "supported" if bool(p["H2a_supported"]) else "not supported",
            "detail": f"flip(shuffle) - flip(re-run) = {p['diff']:.4f} {_ci(p['lo'], p['hi'])}, "
                      f"n = {int(p['n'])}"}


def h2b(holdup: pd.DataFrame) -> dict[str, Any]:
    h = holdup.iloc[0]
    return {"outcome": "reported (descriptive)",
            "detail": f"verified-claim share {h['verified_claim_share']:.3f} over {int(h['n_shown'])} shown "
                      f"cases; cases with >= 1 verified claim {h['cases_ge1_verified']:.3f}"}


def _setup(cfg: dict[str, Any], seeds: list[int]) -> list[str]:
    from mfsi_cxr.arbiter.prompts import load_prompt, prompt_hash

    g, a = git_info(), cfg["arbiter"]
    image, text = cfg["model"]["image_encoder"], cfg["model"]["text_encoder"]
    pk = ", ".join(f"{k} {v}" for k, v in sorted(package_versions().items()) if v)
    return ["## Setup", "",
            f"- git {g.get('commit')} (dirty: {g.get('dirty')}); config hash {config_hash(cfg)}",
            f"- classifier seeds {seeds}; bootstrap B = {cfg['eval']['bootstrap_B']}",
            f"- weights: {cfg['model']['weights']}; image encoder {image['xrv_weights']} (torchxrayvision "
            f"{image['xrv_version']}, sha256 {image['sha256']})",
            f"- text encoder {text['repo']} @ {text['revision']}",
            f"- arbiter {a['model']} (weights {a['model_digest']})",
            f"- prompt v1 sha256 {prompt_hash(load_prompt(cfg))}",
            f"- packages: {pk}", ""]


def _rq2(t: dict[str, pd.DataFrame], hyp: dict[str, dict[str, Any]]) -> list[str]:
    help_ = t["rq2_help"].set_index("population")
    s0 = t["rq2_sample"].iloc[0]
    lines = ["## RQ2", "",
             f"- sample: {int(s0['n_cases'])} escalated cases of {int(s0['n_patients'])} patients",
             f"- sampling fractions: cases {s0['case_fraction']:.3f}, patients {s0['patient_fraction']:.3f}"]
    if "shown" in help_.index:
        s = help_.loc["shown"]
        lines += [f"- help (shown cases): classifier accuracy {s['classifier_accuracy']:.3f}, arbiter "
                  f"{s['arbiter_accuracy_decided']:.3f}",
                  f"- fixed {int(s['fixed'])}, broken {int(s['broken'])}, net per case "
                  f"{s['net_per_case']:.4f} {_ci(s['net_lo'], s['net_hi'])}, exact McNemar p "
                  f"{s['mcnemar_exact_p']:.4g}"]
    return [*lines, f"- listen: {hyp['H2a']['detail']}", f"- hold up: {hyp['H2b']['detail']}", ""]


def build_markdown(cfg: dict[str, Any], t: dict[str, pd.DataFrame], seeds: list[int]) -> str:
    from mfsi_cxr.eval import summary

    hyp = {"H1": h1(t["rq1_main"]), "H1b": h1b(t["rq1_cross_modal"]), "H2a": h2a(t["rq2_listen_primary"]),
           "H2b": h2b(t["rq2_holdup"])}
    lines = [f"# Report ({cfg['profile']})", ""]
    if cfg["profile"] != "full":
        lines += [f"**THROW-AWAY: {cfg['profile']} profile, plumbing check only. Never report.**", ""]
    lines += _setup(cfg, seeds)
    lines += ["## Pre-stated hypotheses", "", "| Hypothesis | Outcome | Evidence |", "|---|---|---|"]
    lines += [f"| {k} | {v['outcome']} | {v['detail']} |" for k, v in hyp.items()]
    rq1 = summary.build(cfg, t, {"split": "test", "commit": git_info().get("commit"), "seeds": seeds})
    lines += ["", "## RQ1 failure conditions", ""]
    lines += ["#" + ln if ln.startswith("#") else ln for ln in rq1.splitlines()[1:] if "THROW-AWAY" not in ln]
    lines += ["", *_rq2(t, hyp), "## Threats to validity", "", *[f"- {x}" for x in THREATS], "",
              "## AI-use statement", "", AI_USE, ""]
    return "\n".join(lines) + "\n"


def write_manifest(out_dir: Path, cfg: dict[str, Any]) -> Path:
    files = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name.endswith(".meta.json") or p.name == "manifest.json":
            continue
        producer = next((s for k, s in PRODUCER.items() if p.name.startswith(k)), "see sidecar")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        files.append({"path": p.relative_to(out_dir).as_posix(), "sha256": digest, "producer": producer})
    body = {"profile": cfg["profile"], "config_hash": config_hash(cfg), "git": git_info(),
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "files": files}
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return path


def copy_aliases(tdir: Path, fdir: Path, seed: int) -> None:
    """Plan-named copies: probe_text.csv, agreement_spread.png."""
    shutil.copyfile(tdir / f"probe_seed{seed}.csv", tdir / "probe_text.csv")
    src = fdir / f"agreement_spread_seed{seed}.png"
    if src.exists():
        shutil.copyfile(src, fdir / "agreement_spread.png")
