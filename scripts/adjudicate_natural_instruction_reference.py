#!/usr/bin/env python3
"""Plan, execute, and freeze the natural instruction-reference v2 lineage.

The command never mutates the historical packet, reviewer runs, or v1 frozen
reference.  It dispatches one third-model call per case with at least one A/B
dimension disagreement; all calls are zero-price-only and require a separate
model family.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import natural_instruction_adjudication as adjudication  # noqa: E402
from src.eval.instruction_reviewer_pair_gate import load_g06_config  # noqa: E402


DEFAULT_ROOT = ROOT / ".workbuddy" / "instruction-natural-calibration-20261015"
DEFAULT_PACKET = DEFAULT_ROOT / "packet.json"
DEFAULT_RUN_A = DEFAULT_ROOT / "runtime" / "evaluation_only" / "ai_annotations" / "ai-7b384bb6a899476ea2ed2154e8f5c764"
DEFAULT_RUN_B = DEFAULT_ROOT / "runtime" / "evaluation_only" / "ai_annotations" / "ai-08619298c5b04a6da3ff4c53961490b7"
DEFAULT_PLAN = DEFAULT_ROOT / "adjudication-v2-plan.json"
DEFAULT_FROZEN = DEFAULT_ROOT / "frozen-reference-v2.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite immutable output: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _set_rsi_home(path: Path | None) -> None:
    if path is not None:
        os.environ["RSI_HOME"] = str(path.resolve())


def _load_project_key() -> None:
    """Read the local .env only when the launcher has not injected a key."""
    if os.environ.get("SILICONFLOW_API_KEY"):
        return
    try:
        from dotenv import dotenv_values
        value = dotenv_values(ROOT / ".env").get("SILICONFLOW_API_KEY")
    except Exception:
        value = None
    if value:
        os.environ["SILICONFLOW_API_KEY"] = str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute", "freeze"))
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--run-a", type=Path, default=DEFAULT_RUN_A)
    parser.add_argument("--run-b", type=Path, default=DEFAULT_RUN_B)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--adjudication-run", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--model-id", default="XingChenAGI/Xing4.0-29B")
    parser.add_argument("--rsi-home", type=Path,
                        help="isolated artifact root; required for plan/execute")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    _set_rsi_home(args.rsi_home)
    _load_project_key()

    packet = _read(args.packet)
    cfg, _provenance = load_g06_config(args.model_id, ROOT)
    if args.action == "plan":
        plan = adjudication.build_adjudication_plan(
            packet, args.run_a, args.run_b, cfg,
            target_model_id=args.model_id, max_cost_usd=0.0)
        _write_once(args.plan, plan)
        print(json.dumps({"status": "planned_zero_call", "plan": str(args.plan),
            "plan_sha256": plan["plan_sha256"], "planned_calls": plan["planned_calls"],
            "network_calls": 0, "model_calls": 0}, ensure_ascii=False))
        return 0

    plan = _read(args.plan)
    if args.action == "execute":
        if args.expected_plan_sha256 != plan.get("plan_sha256"):
            raise ValueError("--expected-plan-sha256 must match the immutable plan")
        result = adjudication.adjudication_execute(
            packet, args.run_a, args.run_b, cfg, plan)
        print(json.dumps({"status": "completed", "run_dir": result["run_dir"],
            "receipt_sha256": result["receipt"].get("output_sha256"),
            "calls": result["receipt"].get("call_count"),
            "measured_usd": result["receipt"].get("measured_usd")}, ensure_ascii=False))
        return 0

    if args.adjudication_run is None:
        parser.error("freeze requires --adjudication-run")
    frozen = adjudication.freeze_reference_v2(
        packet, args.run_a, args.run_b, args.adjudication_run, plan)
    _write_once(args.output, frozen)
    print(json.dumps({"status": "frozen", "output": str(args.output),
        "reference_sha256": frozen["reference_sha256"],
        "coverage": frozen["reference_class_coverage"],
        "coverage_sufficient": frozen["reference_class_coverage_sufficient"]},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
