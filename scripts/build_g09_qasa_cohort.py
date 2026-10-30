#!/usr/bin/env python3
"""Build an offline, task-only G09 cohort from vendored ScholarQABench QASA.

The command does not call a model or network.  It writes the unannotated
dataset and evaluator-only reference sidecar separately; rubric authoring and
answer execution remain explicit later stages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.open_research_external import QASA_DEFAULT_SOURCE, build_qasa_unannotated_dataset


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=QASA_DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", default="g09-qasa-v1")
    parser.add_argument("--max-sources", type=int, default=30)
    parser.add_argument("--min-groups-per-split", type=int, default=12)
    args = parser.parse_args(argv)
    result = build_qasa_unannotated_dataset(args.source, split_seed=args.split_seed,
        max_sources=args.max_sources, min_groups_per_split=args.min_groups_per_split)
    out = args.output_dir.resolve()
    _write(out / "dataset.json", result["dataset"])
    _write(out / "reference-sidecar.evaluator-only.json", result["references"])
    _write(out / "report.json", result["report"])
    print(json.dumps({"output_dir": str(out), "report": result["report"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

