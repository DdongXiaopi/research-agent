"""Build an immutable, target-blind G05 sample plan with empty task shells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_sampling import build_g05_sample_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--status-report", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--primary-size", type=int, default=60)
    parser.add_argument("--reserve-size", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pool = json.loads(args.source_pool.read_text(encoding="utf-8"))
    status = json.loads(args.status_report.read_text(encoding="utf-8"))
    plan = build_g05_sample_plan(
        pool, status, seed=args.seed,
        primary_size=args.primary_size, reserve_size=args.reserve_size,
    )
    output = args.output.resolve()
    try:
        output.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("G05 sample plan output must stay under project .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen G05 sample plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": "frozen_empty_task_shells",
        "output": str(output),
        "eligible_source_count": plan["eligible_source_count"],
        "primary_count": plan["primary_count"],
        "reserve_count": plan["reserve_count"],
        "class_capacity": plan["class_capacity"],
        "calibration_eligible": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
