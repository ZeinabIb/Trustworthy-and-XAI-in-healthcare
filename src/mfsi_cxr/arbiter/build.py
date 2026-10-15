"""Packets for a split from the artifacts (scripts 51 and 52 share it)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mfsi_cxr.arbiter.cases import CaseBuilder, select_cases
from mfsi_cxr.arbiter.packet import Packet, build_packet
from mfsi_cxr.mfsi.fit import theta_vector
from mfsi_cxr.utils.io import artifact_dir, check_artifact, read_table


def arbiter_dir(cfg: dict[str, Any]) -> Path:
    d = artifact_dir(cfg) / "arbiter"
    d.mkdir(parents=True, exist_ok=True)
    return d


def builder(cfg: dict[str, Any], split: str, seed: int) -> tuple[CaseBuilder, Any]:
    adir = artifact_dir(cfg)
    ppath = adir / f"mfsi_params_seed{seed}.json"
    check_artifact(ppath, cfg)
    params = json.loads(ppath.read_text(encoding="utf-8"))
    table = read_table(adir / f"mfsi_{split}_seed{seed}.parquet", cfg)
    texts = read_table(adir / "text_inputs.parquet", cfg)
    facts = read_table(adir / "radgraph_facts.parquet", cfg)
    theta = theta_vector(params, cfg["mfsi"]["operating_fraction"])
    return CaseBuilder(table, texts, facts, params, params["text_informative"], theta), table


def packet_dict(p: Packet, patient_id: str, study_id: str) -> dict[str, Any]:
    return {"case_id": p.case_id, "patient_id": patient_id, "study_id": study_id, "finding": p.finding,
            "items": p.items, "roles": p.roles, "variant": p.variant, "hash": p.hash}


def from_dict(d: dict[str, Any]) -> Packet:
    return Packet(d["case_id"], d["finding"], tuple((i, t) for i, t in d["items"]), d["roles"], d["variant"])


def build_all(cfg: dict[str, Any], split: str, seed: int,
              forbidden: dict[str, tuple[str, ...]] | None = None) -> tuple[Path, list[dict[str, Any]]]:
    """All packets of the seeded case sample. `forbidden` maps study_id -> tokens that must not appear (tests:
    the index study's canary tokens)."""
    a = cfg["arbiter"]
    cb, table = builder(cfg, split, seed)
    cases = select_cases(table, a["n_cases"], a["case_seed"])
    v = a["packet"]
    out = []
    for r in cases.itertuples():
        inp = cb.inputs(r.study_id, r.finding)
        donor = (cb.donor(r.study_id, r.finding, r.patient_id, v["shuffle_seed"])
                 if v["reliability"] == "shuffled_across_patients" else None)
        p = build_packet(inp, cfg, reliability=v["reliability"], order=v["order"], donor=donor,
                         forbidden_tokens=(forbidden or {}).get(r.study_id, ()))
        out.append(packet_dict(p, r.patient_id, r.study_id))
    return arbiter_dir(cfg) / f"packets_{split}_seed{seed}.jsonl", out


def read_packets(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
