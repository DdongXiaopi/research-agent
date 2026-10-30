"""Freeze a deterministic status-check request plan; this command makes no requests."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_publication_status import build_status_collection_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pool_path = args.source_pool.resolve()
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    plan = build_status_collection_plan(
        pool, seed=args.seed, sample_size=args.sample_size,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    output = args.output.resolve()
    try:
        output.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("status plan output must stay under project .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen status plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": "frozen_no_request_plan",
        "output": str(output),
        "source_pool_sha256": plan["source_pool_sha256"],
        "planned_count": plan["planned_count"],
        "seed": plan["seed"],
        "plan_sha256": plan["plan_sha256"],
        "model_calls": 0,
        "target_scorer_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
