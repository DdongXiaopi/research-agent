#!/usr/bin/env python3
"""Build the offline G03 prompt+Skill candidate bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.clinifact_candidate_bundle import build_candidate_bundle  # noqa: E402
from src.util import read_json, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-report", type=Path, required=True)
    parser.add_argument("--skill-result", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--skill-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"candidate bundle already exists: {out}")
    bundle = build_candidate_bundle(
        prompt_report=read_json(args.prompt_report.resolve()),
        prompt_report_path=args.prompt_report.resolve(),
        skill_result=read_json(args.skill_result.resolve()),
        skill_result_path=args.skill_result.resolve(),
        release_dir=args.release_dir.resolve(),
        skill_name=args.skill_name,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, bundle)
    print(json.dumps({"status": "candidate_bundle_built",
                      "candidate_bundle_id": bundle["candidate_bundle_id"],
                      "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
                      "out": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
