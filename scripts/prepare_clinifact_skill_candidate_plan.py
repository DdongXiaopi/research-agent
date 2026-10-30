#!/usr/bin/env python3
"""Freeze a zero-call GEPA Skill-description candidate-generation plan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.split import load_tasks
from src.eval.clinifact_skill_candidate_plan import build_plan, validate_plan
from src.util import read_json, sha256_file, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--skill-name", default="statistical-interpretation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-limit", type=int, default=30)
    parser.add_argument("--max-metric-calls", type=int, default=120)
    parser.add_argument("--max-candidate-proposals", type=int, default=8)
    parser.add_argument("--reflection-minibatch-size", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.candidate_report.resolve()
    plan = build_plan(candidate_report=read_json(report_path),
                      candidate_report_path=report_path,
                      candidate_report_file_sha256=sha256_file(report_path),
                      release_dir=args.release_dir, skill_name=args.skill_name,
                      task_rows=load_tasks("clinifact_fixed_abstract", splits=["dev_optimize"]),
                      config_path=args.config, task_limit=args.task_limit,
                      max_metric_calls=args.max_metric_calls,
                      max_candidate_proposals=args.max_candidate_proposals,
                      reflection_minibatch_size=args.reflection_minibatch_size)
    validate_plan(plan)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {args.out}")
    write_json(args.out, plan)
    print(json.dumps({"status": plan["status"], "plan_sha256": plan["plan_sha256"],
                      "model_calls": 0, "network_calls": 0,
                      "dispatch_authorized": False,
                      "task_count": plan["task_scope"]["task_count"],
                      "skill_name": args.skill_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
