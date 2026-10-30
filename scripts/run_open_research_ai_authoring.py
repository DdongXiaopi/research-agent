#!/usr/bin/env python3
"""Run the isolated AI task-first rubric authoring job for G09."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.eval.open_research_ai_workflow import (  # noqa: E402
    build_author_plan, execute_author, write_run_artifacts,
)
from src.util import read_json, write_json  # noqa: E402


def _load_env() -> None:
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    values = dotenv_values(ROOT / ".env")
    for key, value in values.items():
        if value is not None and not os.environ.get(key):
            os.environ[key] = str(value)


def _config(path: str, model_id: str) -> dict:
    cfg = load_config(path)
    cfg.setdefault("models", {}).setdefault("task", {})["model_id"] = model_id
    cfg["models"]["task"]["provider"] = "http"
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-cost-usd", type=float, default=1.0,
                        help="explicit AI rubric-authoring spend ceiling")
    args = parser.parse_args(argv)
    _load_env()
    dataset = read_json(args.dataset.resolve())
    cfg = _config(args.config, args.model_id)
    plan = build_author_plan(dataset, cfg, max_tokens=args.max_tokens,
                             max_cost_usd=args.max_cost_usd)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    plan_path = (args.plan_path or (work / "author-plan.json")).resolve()
    if args.action == "plan":
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("existing G09 author plan differs")
        write_json(plan_path, plan)
        print(json.dumps({"plan_path": str(plan_path), "plan_sha256": plan["plan_sha256"],
                          "model_id": plan["model_id"], "max_calls": plan["max_calls"],
                          "model_calls": 0}, ensure_ascii=False))
        return 0
    if (not plan_path.exists() or read_json(plan_path) != plan
            or args.expected_plan_sha256 != plan["plan_sha256"]):
        raise ValueError("G09 author execution requires the exact frozen zero-call plan")
    if not os.environ.get("RSI_HOME") or Path(os.environ["RSI_HOME"]).resolve() == (ROOT / "artifacts").resolve():
        raise ValueError("G09 authoring requires an isolated RSI_HOME")
    result = execute_author(dataset, cfg, plan)
    pointer = write_run_artifacts(work, plan, result)
    print(json.dumps(pointer, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
