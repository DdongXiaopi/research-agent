#!/usr/bin/env python3
"""Build, independently label, and measure the historical candidate census."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.eval import natural_instruction_calibration as calibration  # noqa: E402
from src.util import artifacts_root, read_json, write_json  # noqa: E402

FREE_MODELS = ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414")


def _config(path: str, model: str) -> dict:
    cfg = load_config(path)
    cfg.setdefault("models", {}).setdefault("task", {})["model_id"] = model
    cfg["models"]["task"]["provider"] = "http"
    return cfg


def _isolated() -> None:
    if not os.environ.get("RSI_HOME") or artifacts_root().resolve() == (ROOT / "artifacts").resolve():
        raise ValueError("natural instruction calibration requires isolated RSI_HOME")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("packet", "reference-plan", "reference-execute",
                                            "freeze-reference", "judge-plan", "judge-execute", "report"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--model-id", choices=FREE_MODELS, default="Qwen/Qwen3-8B")
    parser.add_argument("--slot", choices=("reviewer_a", "reviewer_b"))
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reference-a-run", type=Path)
    parser.add_argument("--reference-b-run", type=Path)
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    args = parser.parse_args(argv)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    packet_path, frozen_path = work / "packet.json", work / "frozen-reference.json"

    if args.action == "packet":
        packet = calibration.build_packet(ROOT)
        calibration.validate_packet(packet)
        if packet_path.exists() and read_json(packet_path) != packet:
            raise ValueError("existing natural instruction packet differs")
        write_json(packet_path, packet)
        print(json.dumps({"packet_path": str(packet_path), "packet_sha256": packet["packet_sha256"],
            "case_count": len(packet["cases"]), "case_count_by_domain": packet["case_count_by_domain"],
            "model_calls": 0}, ensure_ascii=False))
        return 0

    packet = read_json(packet_path)
    calibration.validate_packet(packet)
    if args.action in {"reference-plan", "reference-execute"}:
        if not args.slot or not args.plan_path:
            parser.error("reference plan/execute require --slot and --plan-path")
        _isolated()
        cfg = _config(args.config, args.model_id)
        plan = calibration.reference_plan(packet, args.slot, cfg, max_cost_usd=args.max_cost_usd)
        plan_path = args.plan_path.resolve()
        if args.action == "reference-plan":
            if plan_path.exists() and read_json(plan_path) != plan:
                raise ValueError("existing natural reference plan differs")
            write_json(plan_path, plan)
            print(json.dumps({"plan_path": str(plan_path), "plan_sha256": plan["plan_sha256"],
                              "slot": args.slot, "model_id": plan["model_id"],
                              "max_calls": plan["max_calls"], "model_calls": 0}, ensure_ascii=False))
            return 0
        if (not plan_path.exists() or read_json(plan_path) != plan
                or args.expected_plan_sha256 != plan["plan_sha256"]):
            raise ValueError("reference execution requires the exact frozen zero-call plan")
        result = calibration.reference_execute(packet, args.slot, cfg, plan)
        print(json.dumps({"run_dir": result["run_dir"], "slot": args.slot,
                          "model_id": plan["model_id"],
                          "measured_usd": result["receipt"]["measured_usd"]}, ensure_ascii=False))
        return 0

    if args.action == "freeze-reference":
        if not args.reference_a_run or not args.reference_b_run:
            parser.error("freeze-reference requires both reference run paths")
        frozen = calibration.freeze_reference(packet, args.reference_a_run, args.reference_b_run)
        if frozen_path.exists() and read_json(frozen_path) != frozen:
            raise ValueError("existing frozen natural reference differs")
        write_json(frozen_path, frozen)
        print(json.dumps({"frozen_reference_path": str(frozen_path),
            "case_count": len(frozen["cases"]), "disagreement_count": len(frozen["disagreements"]),
            "model_calls": 0}, ensure_ascii=False))
        return 0

    frozen = read_json(frozen_path)
    if args.action == "report":
        if not args.run_dir:
            parser.error("report requires --run-dir")
        job = calibration.verify_calibration(packet, frozen, args.run_dir)
        print(json.dumps({"run_dir": str(job["root"]), "challenge_pass": job["output"]["challenge_pass"],
            "dimensions": job["output"]["dimensions"],
            "natural_distribution_calibrated": job["output"]["natural_distribution_calibrated"],
            "production_gate_calibrated": job["output"]["production_gate_calibrated"],
            "measured_usd": job["receipt"]["measured_usd"],
            "request_accounting": job["receipt"]["request_accounting"]}, ensure_ascii=False))
        return 0

    if not args.plan_path:
        parser.error("judge plan/execute require --plan-path")
    _isolated()
    cfg = _config(args.config, args.model_id)
    plan = calibration.calibration_plan(packet, frozen, cfg, max_cost_usd=args.max_cost_usd)
    plan_path = args.plan_path.resolve()
    if args.action == "judge-plan":
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("existing natural judge plan differs")
        write_json(plan_path, plan)
        print(json.dumps({"plan_path": str(plan_path), "plan_sha256": plan["plan_sha256"],
            "model_id": plan["model_id"], "max_calls": plan["max_calls"], "model_calls": 0},
            ensure_ascii=False))
        return 0
    if (not plan_path.exists() or read_json(plan_path) != plan
            or args.expected_plan_sha256 != plan["plan_sha256"]):
        raise ValueError("judge execution requires the exact frozen zero-call plan")
    result = calibration.calibration_execute(packet, frozen, cfg, plan)
    verified = calibration.verify_calibration(packet, frozen, Path(result["run_dir"]))
    print(json.dumps({"run_dir": result["run_dir"],
        "challenge_pass": verified["output"]["challenge_pass"],
        "natural_distribution_calibrated": verified["output"]["natural_distribution_calibrated"],
        "dimensions": verified["output"]["dimensions"],
        "measured_usd": verified["receipt"]["measured_usd"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
