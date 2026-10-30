"""Freeze the rank-121/122 G05 publication-status extension; makes no requests."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_publication_status import build_status_extension_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-sample-plan", type=Path, required=True)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    read_json = lambda path: json.loads(path.read_text(encoding="utf-8"))
    plan = build_status_extension_plan(
        read_json(args.source_pool), read_json(args.parent_plan),
        read_json(args.parent_report), read_json(args.parent_sample_plan),
        created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
    )
    output = args.output.resolve()
    try:
        output.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("G05 status extension plan output must stay under project .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen G05 status extension plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": "frozen_no_request_extension_plan",
        "output": str(output), "plan_sha256": plan["plan_sha256"],
        "doi_candidates": [row["doi"] for row in plan["doi_candidates"]],
        "physical_request_count": plan["physical_request_count"],
        "model_calls": plan["model_calls"], "target_scorer_calls": plan["target_scorer_calls"]},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
