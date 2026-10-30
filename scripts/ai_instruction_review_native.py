#!/usr/bin/env python3
"""Freeze two native AI labels, then calibrate the exact runtime instruction judge."""
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
from src.eval.native_instruction_reference import (  # noqa: E402
    calibration_execute, calibration_plan, freeze_reference, verify_calibration,
    verify_reference,
)
from src.util import artifacts_root, read_json, write_json  # noqa: E402


def _paths(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return tuple(root / name for name in (
        "packet.json", "RUBRIC.md", "reviewer_a.json", "reviewer_b.json",
        "frozen_reference.json"))


def _config(path: str, model: str) -> dict:
    cfg = load_config(path)
    cfg.setdefault("models", {}).setdefault("task", {})["model_id"] = model
    cfg["models"]["task"]["provider"] = "http"
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "plan", "execute", "report"))
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/experiment.task-input.healthver-exploration.yaml")
    parser.add_argument("--model-id", choices=("Qwen/Qwen3-8B", "Qwen/Qwen3-32B",
                        "Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414"),
                        default="Qwen/Qwen3-8B")
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    packet_path, rubric_path, a_path, b_path, frozen_path = _paths(args.packet_dir.resolve())
    if args.action == "freeze":
        if frozen_path.exists():
            raise ValueError("frozen native reference already exists")
        frozen = freeze_reference(packet_path, rubric_path, a_path, b_path)
        write_json(frozen_path, frozen)
        print(json.dumps({"reference_path": str(frozen_path),
                          "case_count": len(frozen["cases"])}, ensure_ascii=False))
        return 0
    packet, frozen = verify_reference(packet_path, rubric_path, a_path, b_path, frozen_path)
    if args.action == "report":
        if args.run_dir is None:
            parser.error("report requires --run-dir")
        job = verify_calibration(packet, frozen, frozen_path, args.run_dir)
        output = job["output"]
        print(json.dumps({"run_dir": str(job["root"]),
            "challenge_pass": output["challenge_pass"],
            "decision_challenge_pass": output["decision_challenge_pass"],
            "decision": output["decision"],
            "dimensions": output["dimensions"],
            "measured_usd": job["receipt"]["measured_usd"],
            "request_accounting": job["receipt"]["request_accounting"],
            "natural_distribution_calibrated": False,
            "production_gate_calibrated": False}, ensure_ascii=False))
        return 0
    if not os.environ.get("RSI_HOME") or artifacts_root().resolve() == (ROOT / "artifacts").resolve():
        raise ValueError("native instruction calibration requires isolated RSI_HOME")
    cfg = _config(args.config, args.model_id)
    plan = calibration_plan(packet, frozen, frozen_path, cfg,
                            max_cost_usd=args.max_cost_usd)
    if args.plan_path is None:
        parser.error("plan and execute require --plan-path")
    plan_path = args.plan_path.resolve()
    if args.action == "plan":
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("existing native instruction calibration plan differs")
        write_json(plan_path, plan)
        print(json.dumps({"plan_path": str(plan_path),
                          "plan_sha256": plan["plan_sha256"],
                          "model_id": plan["model_id"],
                          "max_calls": plan["max_calls"],
                          "max_cost_usd": plan["max_cost_usd"],
                          "model_calls": 0}, ensure_ascii=False))
        return 0
    if (not plan_path.exists() or read_json(plan_path) != plan
            or args.expected_plan_sha256 != plan["plan_sha256"]):
        raise ValueError("native instruction calibration requires the exact frozen zero-call plan")
    job = calibration_execute(packet, frozen, frozen_path, cfg, plan)
    verified = verify_calibration(packet, frozen, frozen_path, Path(job["run_dir"]))
    print(json.dumps({"run_dir": job["run_dir"], "challenge_pass": verified["output"]["challenge_pass"],
                      "decision_challenge_pass": verified["output"]["decision_challenge_pass"],
                      "decision": verified["output"]["decision"],
                      "dimensions": verified["output"]["dimensions"],
                      "measured_usd": verified["receipt"]["measured_usd"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
