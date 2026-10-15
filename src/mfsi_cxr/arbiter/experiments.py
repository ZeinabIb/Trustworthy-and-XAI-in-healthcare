"""Phase 6 arbiter experiments on a pre-stated case sample (arbiter side only; labels never enter here).

Per case: the MAIN run is the full pipeline (seeds arbiter.seeds; vote, label-tree rule, per-claim removal),
used by experiments A and C. Experiment B conditions are plain votes (no removal check) on modified packets:
  rerun            identical packet, seeds arbiter.experiments.noise_seeds (re-run noise floor)
  shuffle          modality item of another case of the same finding (seeded derangement within the finding)
  swap_modalities  image and text profiles / margins exchanged (two-witness findings only; else n/a)
  flip_leaning     each modality's leaning flipped (p -> 1 - p), margins kept (Phase 5 switch, extra row)
  absent           modality item removed
  blinded          classifier output hidden
The decision label of a condition is the post-vote label (present / absent / cannot_decide / NO_MAJORITY).
Records are written per case to a resumable jsonl.
"""

from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.arbiter.cases import CaseBuilder, derangement
from mfsi_cxr.arbiter.llm_client import CachedClient
from mfsi_cxr.arbiter.packet import Modality, NotApplicableError, build_packet
from mfsi_cxr.arbiter.run import SHOWN, run_case, voted
from mfsi_cxr.arbiter.vote import vote

CONDITIONS = ("rerun", "shuffle", "swap_modalities", "flip_leaning", "absent", "blinded")


def shuffle_donors(cases: pd.DataFrame, cb: CaseBuilder, seed: int) -> dict[str, Modality | None]:
    """case_id -> donor modality: a seeded derangement within each finding among the sampled cases (no case
    keeps its own; the marginal of R per finding is preserved). None if the finding has < 2 cases."""
    out: dict[str, Modality | None] = {}
    rng = np.random.default_rng(seed)
    for finding, g in cases.groupby("finding", sort=True):
        ids = [f"{s}|{finding}" for s in g["study_id"]]
        mods = [cb.inputs(s, finding).modality for s in g["study_id"]]
        perm = derangement(len(ids), rng)
        for k, cid in enumerate(ids):
            out[cid] = None if perm is None else mods[perm[k]]
    return out


def condition_label(client: CachedClient, cfg: dict[str, Any], packet) -> tuple[str, int]:
    label, runs = voted(client, cfg, packet)
    return label, sum(r.n_calls for r in runs)


def run_case_experiments(client: CachedClient, cfg: dict[str, Any], cb: CaseBuilder, patient_id: str,
                         study_id: str, finding: str, donor: Modality | None,
                         final: dict[str, bool]) -> dict[str, Any]:
    inp = cb.inputs(study_id, finding)
    base = build_packet(inp, cfg)
    main = run_case(client, cfg, base, cb.context(study_id, final))
    rec: dict[str, Any] = {"case_id": base.case_id, "patient_id": patient_id, "study_id": study_id,
                           "finding": finding,
                           "main": main.to_json(),
                           "main_label": vote(main.run_decisions, cfg["arbiter"]["vote"]),
                           "R_shown": {"main": inp.modality.R, "shuffle": donor.R if donor else None},
                           "two_witness": inp.modality.text_informative,
                           "image_more_reliable": inp.modality.r_img > inp.modality.r_txt,
                           "labels": {}, "calls": {}}
    noise_cfg = copy.deepcopy(cfg)
    noise_cfg["arbiter"]["seeds"] = list(cfg["arbiter"]["experiments"]["noise_seeds"])
    packets = {"rerun": (noise_cfg, base), "absent": (cfg, build_packet(inp, cfg, reliability="absent")),
               "blinded": (cfg, build_packet(inp, cfg, order="blinded")),
               "flip_leaning": (cfg, build_packet(inp, cfg, reliability="shuffled_swap_direction"))}
    if donor is not None:
        packets["shuffle"] = (cfg, build_packet(inp, cfg, reliability="shuffled_across_patients",
                                                donor=donor))
    with contextlib.suppress(NotApplicableError):  # one-witness finding: no text signal to swap (n/a)
        packets["swap_modalities"] = (cfg, build_packet(inp, cfg, reliability="swap_modalities"))
    for name in CONDITIONS:
        if name not in packets:
            rec["labels"][name] = None           # not applicable for this case
            continue
        c, p = packets[name]
        rec["labels"][name], rec["calls"][name] = condition_label(client, c, p)
    if main.status == SHOWN and main.decision in ("present", "absent"):
        final[finding] = main.decision == "present"
    return rec


def run_all(client: CachedClient, cfg: dict[str, Any], cb: CaseBuilder, cases: pd.DataFrame,
            out_path: Path) -> list[dict[str, Any]]:
    """Resumable: cases in out_path are kept; a study's final decisions feed its later label-tree checks."""
    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["case_id"]] = r
    donors = shuffle_donors(cases, cb, cfg["arbiter"]["experiments"]["shuffle_seed"])
    finals: dict[str, dict[str, bool]] = {}
    for r in done.values():
        m = r["main"]
        if m["status"] == SHOWN and m["decision"] in ("present", "absent"):
            finals.setdefault(r["study_id"], {})[r["finding"]] = m["decision"] == "present"
    with out_path.open("a", encoding="utf-8") as f:
        for row in cases.itertuples():
            cid = f"{row.study_id}|{row.finding}"
            if cid in done:
                continue
            rec = run_case_experiments(client, cfg, cb, row.patient_id, row.study_id, row.finding,
                                       donors[cid], finals.setdefault(row.study_id, {}))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[cid] = rec
    return [done[f"{r.study_id}|{r.finding}"] for r in cases.itertuples()]
