#!/usr/bin/env python3
"""Prepare a history-excluded HealthFC exploratory cohort (zero model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.healthfc_cohort import build_healthfc_cohort  # noqa: E402
from src.util import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="HealthFC CSV source")
    parser.add_argument("--history-report", type=Path, required=True)
    parser.add_argument("--salt", default="healthfc-mipro-exploration-20261008-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        history = json.loads(args.history_report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read complete HealthFC history report: {exc}") from exc
    cohort = build_healthfc_cohort(args.source, history, salt=args.salt)
    if args.write:
        write_json(args.output, cohort)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output),
        "cohort_sha256": cohort["cohort_sha256"],
        "history_scan_complete": cohort["history_scan_complete"],
        "history_rows_excluded": cohort["history_rows_excluded"],
        "history_components_excluded": cohort["history_components_excluded"],
        "salt_attempts": cohort["salt_attempts"],
        "scopes": {name: {"n": item["task_count"],
                           "labels": item["class_counts"],
                           "components": item["component_count"]}
                   for name, item in cohort["scopes"].items()},
        "eligible_for_confirmation": cohort["eligible_for_confirmation"],
        "api_transfer_approved": cohort["api_transfer_approved"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
