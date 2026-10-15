"""Fit MFSI on the frozen full-model exports (Phase 3): train for clouds, covariances and alpha; val for tau,
y_hat thresholds and theta_R. Writes artifacts/<profile>/mfsi_params_seed{k}.json and
outputs/<profile>/tables/mfsi_fit_seed{k}.csv (alpha, n_pos, target, ridge, cond per finding and modality).
Prints the plan's STOP-AND-ASK conditions (tau at the grid boundary, poorly conditioned covariances).
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.mfsi.fit import MODALITIES, fit_mfsi, load_split, write_params
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import artifact_dir, check_artifact, output_dir, write_meta


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    m = cfg["mfsi"]
    for seed in cfg["train"]["seeds"]:
        probe = artifact_dir(cfg) / f"probe_text_seed{seed}.json"
        check_artifact(probe, cfg)
        informative = json.loads(probe.read_text(encoding="utf-8"))["text_informative"]
        train, val = load_split(cfg, "train", seed), load_split(cfg, "val", seed)
        params = fit_mfsi(train, val, cfg, informative, seed)
        write_params(params, artifact_dir(cfg) / f"mfsi_params_seed{seed}.json", cfg, seed)
        rows = [{"modality": name, "finding": f, **params[f"cov_{name}"][f],
                 **{k if k == "tau" else f"tau_{k}": v
                    for k, v in params[f"tau_{name}"]["per_finding"][f].items()}}
                for name, _ in MODALITIES for f in FINDINGS]
        table = pd.DataFrame(rows)
        out = output_dir(cfg) / "tables" / f"mfsi_fit_seed{seed}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        write_meta(out, cfg, seed)
        shown = ["n_pos", "alpha", "tau", "tau_source"]
        pivot = table.pivot_table(index="finding", columns="modality", values=shown, aggfunc="first")
        print(pivot.to_string())
        for name, _ in MODALITIES:
            r = params[f"reduce_{name}"]
            print(f"seed {seed} {name}: {m['reduce']['method']} {r['d_in']} -> {r['k']} dims "
                  f"({r['variance_kept']:.4f} of train variance)")
            t = params[f"tau_{name}"]
            note = "  STOP-AND-ASK: tau at the grid boundary" if t["at_grid_boundary"] else ""
            print(f"seed {seed} tau_{name} = {t['tau']:.3g}  ECE {t['ece_before_tau1']:.4f} (tau=1) -> "
                  f"{t['ece_after']:.4f}{note}")
            edge = sorted(f for f, r in t["per_finding"].items() if r["at_grid_boundary"])
            if edge:
                print(f"STOP-AND-ASK: per-finding tau_{name} at the grid boundary for {edge}")
        for cls in ("pos", "neg"):
            bad = table[(table[f"cond_{cls}"] > m["cond_max"]) | (table[f"ridge_{cls}"] > 0)]
            if len(bad):
                print(f"STOP-AND-ASK: {len(bad)} of {len(table)} {cls} covariances needed a ridge or "
                      f"are poorly conditioned ({sorted(set(bad['modality']))})")
        strata = {s: [f for f in FINDINGS if params["strata"][f] == s]
                  for s in set(params["strata"].values())}
        print("strata:", strata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
