#!/usr/bin/env python3
"""Prepare and run two blind AI references for the frozen synthetic cohort."""
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
from src.eval import instruction_reviewer_cohort_reference as reference  # noqa: E402
from src.util import artifacts_root, read_json, write_json  # noqa: E402


FREE_MODELS = tuple(sorted(reference.ALLOWED_FREE_MODELS))


def _config(path: str, model_id: str) -> dict:
    cfg = load_config(path)
    cfg.setdefault("models", {}).setdefault("task", {})["model_id"] = model_id
    cfg["models"]["task"]["provider"] = "http"
    return cfg


def _isolated() -> None:
    if not os.environ.get("RSI_HOME") or artifacts_root().resolve() == (ROOT / "artifacts").resolve():
        raise ValueError("cohort reference jobs require isolated RSI_HOME")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-plan", "reference-plan", "reference-execute",
                                            "freeze-reference"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--model-id", choices=FREE_MODELS, default="THUDM/GLM-4-9B-0414")
    parser.add_argument("--model-id-a", choices=FREE_MODELS, default="THUDM/GLM-4-9B-0414")
    parser.add_argument("--model-id-b", choices=FREE_MODELS, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--slot", choices=("reviewer_a", "reviewer_b"))
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reference-a-run", type=Path)
    parser.add_argument("--reference-b-run", type=Path)
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    args = parser.parse_args(argv)
    packet = read_json(args.packet.resolve())
    if (args.action == "reference-execute"
            and reference.requires_reference_shards(packet)):
        raise ValueError("new synthetic reference cohorts must execute through the shard protocol")
    reference.validate_reference_packet(packet)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    if args.action == "freeze-plan":
        cfg_a = _config(args.config, args.model_id_a)
        cfg_b = _config(args.config, args.model_id_b)
        frozen_plan = reference.build_annotation_plan(packet, cfg_a, cfg_b, ROOT)
        plan_root = work / "reviewer_plans"
        for slot in ("reviewer_a", "reviewer_b"):
            slot_dir = plan_root / slot
            slot_dir.mkdir(parents=True, exist_ok=True)
            write_json(slot_dir / "plan.json", frozen_plan["plans"][slot])
        plan_path = work / "annotation-plan.json"
        if plan_path.exists() and read_json(plan_path) != frozen_plan:
            raise ValueError("existing cohort annotation plan differs")
        write_json(plan_path, frozen_plan)
        print(json.dumps({"annotation_plan_path": str(plan_path),
            "plan_sha256": frozen_plan["plan_sha256"],
            "packet_sha256": frozen_plan["packet_sha256"],
            "reviewer_a_model": frozen_plan["reviewer_a"]["model_id"],
            "reviewer_b_model": frozen_plan["reviewer_b"]["model_id"],
            "model_calls": 0, "paid_fallback": False}, ensure_ascii=False))
        return 0

    if args.action in {"reference-plan", "reference-execute"}:
        if not args.slot or not args.plan_path:
            parser.error("reference plan/execute require --slot and --plan-path")
        _isolated()
        cfg = _config(args.config, args.model_id)
        plan = reference.reference_plan(packet, args.slot, cfg, max_cost_usd=args.max_cost_usd)
        plan_path = args.plan_path.resolve()
        if args.action == "reference-plan":
            if plan_path.exists() and read_json(plan_path) != plan:
                raise ValueError("existing cohort reference plan differs")
            write_json(plan_path, plan)
            print(json.dumps({"plan_path": str(plan_path), "plan_sha256": plan["plan_sha256"],
                "slot": args.slot, "model_id": plan["model_id"],
                "max_calls": plan["max_calls"], "max_physical_attempts": plan["max_physical_attempts"],
                "model_calls": 0}, ensure_ascii=False))
            return 0
        if (not plan_path.exists() or read_json(plan_path) != plan
                or args.expected_plan_sha256 != plan["plan_sha256"]):
            raise ValueError("reference execution requires the exact frozen zero-call plan")
        result = reference.reference_execute(packet, args.slot, cfg, plan)
        run_pointer = work / f"{args.slot}-run-dir.json"
        write_json(run_pointer, {"run_dir": result["run_dir"], "plan_sha256": plan["plan_sha256"],
                                 "model_id": plan["model_id"], "slot": args.slot})
        print(json.dumps({"run_dir": result["run_dir"], "slot": args.slot,
            "model_id": plan["model_id"], "calls": result["receipt"]["call_count"],
            "measured_usd": result["receipt"]["measured_usd"],
            "usage_complete": result["receipt"]["usage_complete"]}, ensure_ascii=False))
        return 0

    if not args.reference_a_run or not args.reference_b_run:
        parser.error("freeze-reference requires both completed reference run paths")
    frozen = reference.freeze_reference(packet, args.reference_a_run, args.reference_b_run)
    frozen_path = work / "frozen-reference.json"
    if frozen_path.exists() and read_json(frozen_path) != frozen:
        raise ValueError("existing synthetic frozen reference differs")
    write_json(frozen_path, frozen)
    print(json.dumps({"frozen_reference_path": str(frozen_path),
        "case_count": len(frozen["cases"]), "disagreement_count": len(frozen["disagreements"]),
        "class_coverage": frozen["reference_class_coverage"],
        "class_coverage_sufficient": frozen["reference_class_coverage_sufficient"],
        "natural_distribution": frozen["natural_distribution"], "model_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
