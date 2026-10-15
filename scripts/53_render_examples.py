"""Render arbiter examples for the report (Phase 5): SYNTHETIC ONLY.

Renders up to 20 results of the smoke profile (synthetic data, mock arbiter) as plain text and JSON into
outputs/smoke/arbiter_examples.{txt,json}. Any other profile is refused: putting real report text (or real
evidence packets) into any output is a plan STOP-AND-ASK, not a script option.
"""

from __future__ import annotations

import json
import sys

from mfsi_cxr.arbiter.build import arbiter_dir, read_packets
from mfsi_cxr.arbiter.run import ArbiterResult, render_text
from mfsi_cxr.utils.cli import base_parser, config_from_args
from mfsi_cxr.utils.io import output_dir, write_meta

MAX_EXAMPLES = 20
SYNTHETIC_PROFILE = "smoke"


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(base_parser(__doc__.splitlines()[0]).parse_args(argv))
    if cfg["profile"] != SYNTHETIC_PROFILE:
        print("53_render_examples: refused: only synthetic (smoke) examples may be rendered; real report "
              "text in any output needs Zeinab's decision first (plan 5 STOP-AND-ASK)")
        return 1
    seed = cfg["train"]["seeds"][0]
    results = read_packets(arbiter_dir(cfg) / f"results_val_seed{seed}.jsonl")[:MAX_EXAMPLES]
    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(render_text(ArbiterResult(**{**r, "evidence": [tuple(e) for e in r["evidence"]]}))
                     for r in results)
    (out / "arbiter_examples.txt").write_text(
        "SYNTHETIC DATA, MOCK ARBITER: illustration of the output format only.\n\n" + text, encoding="utf-8")
    (out / "arbiter_examples.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name in ("arbiter_examples.txt", "arbiter_examples.json"):
        write_meta(out / name, cfg, seed, synthetic=True)
    print(f"rendered {len(results)} synthetic examples to {out / 'arbiter_examples.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
