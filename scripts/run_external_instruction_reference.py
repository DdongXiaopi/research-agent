#!/usr/bin/env python3
"""Freeze or execute the diagnostic AI reference for an external G06 packet."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.eval.instruction_external_reference import reference_execute, reference_plan
from src.eval.instruction_reviewer_external_source import validate_packet
from src.util import read_json, write_json


def _load_env() -> None:
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None and not os.environ.get(key):
            os.environ[key] = str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--allow-paid", action="store_true",
                        help="allow a bounded paid model plan (required for official DeepSeek)")
    parser.add_argument("--max-cost-usd", type=float, default=0.5)
    args = parser.parse_args(argv)
    _load_env()
    packet = read_json(args.packet.resolve())
    validate_packet(packet)
    cfg = load_config(args.config)
    cfg.setdefault("models", {}).setdefault("task", {})["model_id"] = args.model_id
    cfg["models"]["task"]["provider"] = "http"
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    plan_path = (args.plan_path or work / "reference-plan.json").resolve()
    zero_price_only = not args.allow_paid
    max_cost_usd = 0.0 if zero_price_only else args.max_cost_usd
    plan = reference_plan(packet, cfg, max_cost_usd=max_cost_usd,
                          zero_price_only=zero_price_only)
    if args.action == "plan":
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("existing external reference plan differs")
        write_json(plan_path, plan)
        print(json.dumps({"plan_path": str(plan_path), "plan_sha256": plan["plan_sha256"],
                          "max_calls": plan["max_calls"], "model_calls": 0}, ensure_ascii=False))
        return 0
    if not plan_path.exists() or read_json(plan_path) != plan or args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("external reference execution requires the exact frozen plan")
    if not os.environ.get(cfg["models"]["task"]["api_key_env"]):
        raise RuntimeError("external reference execution credential is missing")
    result = reference_execute(packet, cfg, plan)
    write_json(work / "reference-pointer.json", {"run_dir": result["run_dir"],
              "plan_sha256": plan["plan_sha256"], "packet_sha256": packet["packet_sha256"]})
    print(json.dumps({"run_dir": result["run_dir"], "plan_sha256": plan["plan_sha256"],
                      "calls": result["receipt"]["call_count"],
                      "measured_usd": result["receipt"]["measured_usd"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
