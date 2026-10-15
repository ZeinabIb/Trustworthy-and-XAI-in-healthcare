"""Degrade-one-modality test on VAL (Phase 4 control 3, dev path; face validity only).

For each seed: re-extract val embeddings through the frozen full model (fp32) once clean (determinism check
against the Phase 2 export) and once per degradation (configs/eval.yaml `eval.degrade`), score MFSI with the
frozen parameters, and report mean r_img, r_txt, A_cos and R before / after with patient-bootstrap CIs.
Writes artifacts/<profile>/degrade/emb_val_seed{k}_{variant}.npz, outputs/<profile>/tables/
controls_dev_degrade{,_per_finding}.csv and figures/degrade_test_dev.png. Never reads test.
"""

from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pandas as pd

from mfsi_cxr.data.datasets import SampleDataset
from mfsi_cxr.eval import degrade as dg
from mfsi_cxr.eval.figures import degrade_figure
from mfsi_cxr.mfsi.fit import load_split, score_split
from mfsi_cxr.models.extract import run_export
from mfsi_cxr.models.train import load_model
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.device import device_from_config
from mfsi_cxr.utils.io import artifact_dir, check_artifact, output_dir, write_meta

SPLIT = "val"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    dcfg = cfg["eval"]["degrade"]
    device = device_from_config(cfg)
    run_cfg = copy.deepcopy(cfg)
    run_cfg["train"]["precision"] = "fp32"
    out_dir = artifact_dir(cfg) / "degrade"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, per_finding = [], []
    for seed in cfg["train"]["seeds"]:
        ppath = artifact_dir(cfg) / f"mfsi_params_seed{seed}.json"
        check_artifact(ppath, cfg)
        params = json.loads(ppath.read_text(encoding="utf-8"))
        train, export = load_split(cfg, "train", seed), load_split(cfg, SPLIT, seed)
        model = load_model(run_cfg, cfg["mfsi"]["embeddings"], seed, device)
        base = SampleDataset(run_cfg, SPLIT)
        variants = [{"name": "clean", "modality": "none", "kind": "none", "param": None}, *dcfg["variants"]]
        tables = {}
        for v in variants:
            ds = dg.DegradedDataset(base, v["kind"], v["param"], dcfg["seed"], dcfg["history_sections"])
            emb, _ = run_export(model, ds, run_cfg, device)
            if not np.array_equal(emb["study_id"], export["study_id"]):
                raise RuntimeError(f"{v['name']}: re-extracted rows differ from the {SPLIT} export")
            path = out_dir / f"emb_{SPLIT}_seed{seed}_{v['name']}.npz"
            np.savez(path, **emb)
            write_meta(path, cfg, seed, split=SPLIT, degradation=v)
            if v["name"] == "clean":
                gap = max(float(np.abs(emb[k] - export[k]).max()) for k in ("z_img", "z_txt"))
                print(f"seed {seed}: clean re-extraction vs Phase 2 export, max |dz| = {gap:.2e}")
            tables[v["name"]] = score_split(train, emb, params, cfg)
        for v in variants[1:]:
            rows = dg.summarise(tables["clean"], tables[v["name"]], export["patient_id"], cfg)
            exp = dg.expectation(v["modality"], rows)
            summary += [{"seed": seed, "variant": v["name"], "modality": v["modality"], **r, **exp}
                        for r in rows]
            for f, g in tables[v["name"]].groupby("finding", sort=False):
                c = tables["clean"][tables["clean"]["finding"] == f]
                per_finding.append({"seed": seed, "variant": v["name"], "finding": f,
                                    **{f"change_{m}": float(g[m].mean() - c[m].mean()) for m in dg.METRICS}})
    table = pd.DataFrame(summary)
    tdir, fdir = output_dir(cfg) / "tables", output_dir(cfg) / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    outputs = (("controls_dev_degrade", table),
               ("controls_dev_degrade_per_finding", pd.DataFrame(per_finding)))
    for name, t in outputs:
        t.to_csv(tdir / f"{name}.csv", index=False)
        write_meta(tdir / f"{name}.csv", cfg, None, split=SPLIT)
    first = cfg["train"]["seeds"][0]
    degrade_figure(table[table["seed"] == first], fdir / "degrade_test_dev.png",
                   f"Degrade one modality on {SPLIT} (seed {first})")
    write_meta(fdir / "degrade_test_dev.png", cfg, None, split=SPLIT)
    wide = table.pivot_table(index=["variant", "modality", "own_margin_falls", "other_margin_unchanged"],
                             columns="metric", values="change", aggfunc="first", sort=False)
    print(wide.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
