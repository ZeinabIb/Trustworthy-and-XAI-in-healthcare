"""RQ2 runner (Phase 6) shared by scripts/60_arbiter_experiments_dev.py (val) and 61_run_rq2_final.py (test,
behind the freeze guard). Draws the pre-stated patient sample, runs the arbiter experiments (cached,
resumable), and evaluates them; labels are read only here, on the evaluation side."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.arbiter.build import arbiter_dir, builder
from mfsi_cxr.arbiter.cases import select_cases_by_patient
from mfsi_cxr.arbiter.experiments import run_all
from mfsi_cxr.arbiter.llm_client import make_client
from mfsi_cxr.eval.arbiter_eval import evaluate
from mfsi_cxr.eval.freeze import TestAccess


def run(cfg: dict[str, Any], split: str, access: TestAccess | None = None) -> dict[str, pd.DataFrame]:
    if split == "test" and not isinstance(access, TestAccess):
        raise PermissionError("the test split is read only by 61_run_rq2_final after the freeze check")
    seed = cfg["train"]["seeds"][0]
    ex = cfg["arbiter"]["experiments"]
    cb, table = builder(cfg, split, seed)
    cases, info = select_cases_by_patient(table, ex["n_min_cases"], cfg["arbiter"]["case_seed"])
    client = make_client(cfg, arbiter_dir(cfg) / "runs")
    records = run_all(client, cfg, cb, cases, arbiter_dir(cfg) / f"exp_{split}_seed{seed}.jsonl")
    keys = list(zip(table["study_id"], table["finding"], strict=True))
    labels = dict(zip(keys, table["y"].astype(float), strict=True))
    y_hat = dict(zip(keys, table["y_hat"].astype(bool), strict=True))
    q = table["prob_cal"].to_numpy()
    msp = dict(zip(keys, np.maximum(q, 1 - q), strict=True))
    out = evaluate(records, labels, y_hat, msp, lambda study, final: cb.context(study, final), cfg)
    out["rq2_sample"] = pd.DataFrame([{**info, "split": split, "classifier_seed": seed,
                                       "model": client.backend.model_id, "revision": client.backend.revision,
                                       "calls_new_this_run": client.n_new}])
    return out
