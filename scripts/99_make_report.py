"""Generate the final report from artifacts only (Phase 6): outputs/<profile>/REPORT.md, the plan's tables
and figures (incl. a static pipeline overview), and outputs/<profile>/manifest.json (sha256, producing script,
config hash of every output). Needs the final runs (41, 61); fails loudly if any required artifact is missing.
On a non-full profile the report is marked throw-away; no metric is printed here.
"""

from __future__ import annotations

import sys

from mfsi_cxr.eval import report
from mfsi_cxr.eval.figures import pipeline_figure
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    seeds = list(cfg["train"]["seeds"])
    out = output_dir(cfg)
    tdir, fdir = out / "tables", out / "figures"
    try:
        tables = report.load_tables(tdir, seeds[0])
    except report.MissingArtifactError as err:
        print(f"99_make_report: {err}")
        return 1
    missing = [f for f in report.REQUIRED_FIGURES if not (fdir / f"{f}.png").exists()]
    if missing:
        print(f"99_make_report: figures missing in {fdir}: {missing}")
        return 1
    report.copy_aliases(tdir, fdir, seeds[0])
    pipeline_figure(fdir / "pipeline_overview.png")
    md = out / "REPORT.md"
    md.write_text(report.build_markdown(cfg, tables, seeds), encoding="utf-8")
    write_meta(md, cfg, None, throwaway=cfg["profile"] != "full")
    man = report.write_manifest(out, cfg)
    print(f"99_make_report ({cfg['profile']}): wrote {md.name} and {man.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
