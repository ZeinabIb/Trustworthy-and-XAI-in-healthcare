"""The one-time RQ1 evaluation on TEST (Phase 4). The only script that reads the test split.

Refuses unless scripts/39_freeze.py has frozen configs, code and artifacts and nothing has changed since; a
second opening needs --rerun-reason (logged in artifacts/<profile>/test_opened.json). Runs, on test: RQ1 (all
scores, anchors, blank sensitivity), the sensitivity table, the validity controls, the degrade-one-modality
test, and the pre-registered added-value analysis; writes outputs/<profile>/tables/{rq1,controls,
added_value}_*.csv, figures, and outputs/<profile>/RQ1_summary.md (generated from the tables).
On a non-full profile the outputs are throw-away (marked) and no metric is printed.
"""

from __future__ import annotations

import sys

from mfsi_cxr.eval import ablations as ab
from mfsi_cxr.eval import degrade as dg
from mfsi_cxr.eval import pipeline, runs, summary
from mfsi_cxr.eval.added_value_dev import run_final
from mfsi_cxr.eval.figures import degrade_figure
from mfsi_cxr.eval.freeze import FreezeError, check_frozen, record_opening
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta

SPLIT = "test"


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--rerun-reason", default=None, help="required to open the test split again (logged)")
    parser.add_argument("--skip-degrade", action="store_true",
                        help="do not run the degrade test (needs the raw data); recorded in RQ1_summary.md")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    try:
        access = check_frozen(cfg, args.rerun_reason)
    except FreezeError as err:
        print(f"41_run_rq1_final: refused: {err}")
        return 1
    record_opening(cfg, access, args.rerun_reason)
    seeds = list(cfg["train"]["seeds"])
    d_test = pipeline.collect(cfg, SPLIT, seeds, access=access)
    d_val = pipeline.collect(cfg, "val", seeds)
    t: dict = {}
    t.update({f"rq1_{k}": v for k, v in runs.rq1(cfg, d_test).items()})
    t.update({f"rq1_{k}": v for k, v in runs.sensitivity(cfg, d_test, access=access, oof=False).items()})
    t.update({f"controls_{k}": v for k, v in runs.controls(cfg, d_test).items()})
    t.update({f"added_value_{k}": v for k, v in run_final(cfg, d_val, d_test).items()})
    for v in cfg["eval"]["sensitivity"]:  # added value under each y_hat sensitivity rule (pre-registered row)
        if v.get("changes_correctness"):
            cfg_v = ab.with_overrides(cfg, v["set"])
            rule = runs.slug(cfg_v["mfsi"]["yhat_rule"])
            res = run_final(cfg_v, pipeline.collect(cfg_v, "val", seeds),
                            pipeline.collect(cfg_v, SPLIT, seeds, access=access))
            t.update({f"added_value_{rule}_{k}": x for k, x in res.items()})
    figs = runs.rq1_figures(cfg, d_test, "")
    ctrl = {k[len("controls_"):]: v for k, v in t.items() if k.startswith("controls_")}
    figs += runs.controls_figures(cfg, ctrl, SPLIT, seeds[0], "")
    if not args.skip_degrade:
        summ, per = dg.run(cfg, SPLIT)
        t["controls_degrade"], t["controls_degrade_per_finding"] = summ, per
        fig = runs.fig_path(cfg, "degrade_test.png")
        degrade_figure(summ[summ["seed"] == seeds[0]], fig,
                       f"Degrade one modality on {SPLIT} (seed {seeds[0]})")
        write_meta(fig, cfg, None, split=SPLIT, throwaway=cfg["profile"] != "full")
        figs.append(fig)
    paths = runs.write_tables(cfg, t, "", SPLIT)
    md = output_dir(cfg) / "RQ1_summary.md"
    meta = {"split": SPLIT, "commit": access.manifest["commit"], "seeds": seeds}
    md.write_text(summary.build(cfg, t, meta), encoding="utf-8")
    write_meta(md, cfg, None, split=SPLIT, throwaway=cfg["profile"] != "full")
    # No metric is printed here: on the pilot the numbers are throw-away and must not be looked at.
    print(f"41_run_rq1_final ({cfg['profile']}): wrote {len(paths)} tables, {len(figs)} figures, {md.name}")
    for p in paths:
        print(f"  {p.name}")
    if args.skip_degrade:
        print("  degrade test skipped (--skip-degrade)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
