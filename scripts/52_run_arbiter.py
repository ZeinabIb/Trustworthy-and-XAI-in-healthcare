"""Run the verified arbiter on the packets of a split (Phase 5): 3-seed vote, hierarchy rule, per-claim
evidence removal. Resumable: cases with a result are skipped; every LLM call is cached. Refuses a non-local
endpoint and unpinned or mismatching weights.

Reads artifacts/<profile>/arbiter/packets_val_seed{k}.jsonl; writes results_val_seed{k}.jsonl (patient-
derived text, gitignored) and outputs/<profile>/tables/arbiter_dev_{cases,summary}.csv (no text). Prints
counts only.
"""

from __future__ import annotations

import json
import statistics
import sys

import pandas as pd

from mfsi_cxr.arbiter import prompts
from mfsi_cxr.arbiter.build import arbiter_dir, builder, from_dict, read_packets
from mfsi_cxr.arbiter.llm_client import PinError, make_client
from mfsi_cxr.arbiter.run import SHOWN, run_case
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta

SPLIT = "val"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    seed = cfg["train"]["seeds"][0]
    adir = arbiter_dir(cfg)
    try:
        client = make_client(cfg, adir / "runs")
    except PinError as err:
        print(f"52_run_arbiter: refused: {err}")
        return 1
    meta = {d["case_id"]: d for d in read_packets(adir / f"packets_{SPLIT}_seed{seed}.jsonl")}
    res_path = adir / f"results_{SPLIT}_seed{seed}.jsonl"
    done = {r["case_id"]: r for r in read_packets(res_path)} if res_path.exists() else {}
    cb, _ = builder(cfg, SPLIT, seed)
    final: dict[str, dict[str, bool]] = {}

    def remember(study: str, r: dict) -> None:
        if r["status"] == SHOWN and r["decision"] in ("present", "absent"):
            final.setdefault(study, {})[r["finding"]] = r["decision"] == "present"

    for r in done.values():
        remember(meta[r["case_id"]]["study_id"], r)
    with res_path.open("a", encoding="utf-8") as f:
        for case_id, d in meta.items():
            if case_id in done:
                continue
            study = d["study_id"]
            res = run_case(client, cfg, from_dict(d), cb.context(study, final.get(study, {}))).to_json()
            remember(study, res)
            f.write(json.dumps(res) + "\n")
            f.flush()
            done[case_id] = res
    cases = pd.DataFrame([{"case_id": r["case_id"], "finding": r["finding"], "status": r["status"],
                           "decision": r["decision"], "n_claims": len(r["claims"]),
                           "n_verified": sum(c["tag"] == "VERIFIED" for c in r["claims"]),
                           "n_calls": r["n_calls"], "n_invalid_runs": r["n_invalid_runs"],
                           "n_failed_parses": r.get("n_failed_parses", 0)} for r in done.values()])
    lat = [c["latency_s"] for c in client.cache.values() if c["model_id"] == client.backend.model_id]
    voting_runs = len(cases) * len(cfg["arbiter"]["seeds"])
    summary = pd.DataFrame([{
        "model": client.backend.model_id, "revision": client.backend.revision,
        "prompt_sha256": prompts.prompt_hash(prompts.load_prompt(cfg)), "n_cases": len(cases),
        **{f"status_{k}": int(v) for k, v in cases["status"].value_counts().items()},
        "invalid_run_rate": float(cases["n_invalid_runs"].sum() / max(voting_runs, 1)),
        "failed_parses_per_voting_run": float(cases["n_failed_parses"].sum() / max(voting_runs, 1)),
        "calls_total": int(cases["n_calls"].sum()), "calls_new_this_run": client.n_new,
        "latency_mean_s": statistics.fmean(lat) if lat else float("nan"),
        "latency_median_s": statistics.median(lat) if lat else float("nan")}])
    tdir = output_dir(cfg) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for name, t in (("arbiter_dev_cases", cases), ("arbiter_dev_summary", summary)):
        t.to_csv(tdir / f"{name}.csv", index=False)
        write_meta(tdir / f"{name}.csv", cfg, seed, split=SPLIT)
    print(summary.T.to_string(header=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
