#!/usr/bin/env python3
"""Plan, execute, and freeze the immutable natural-reference contract top-up v3."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import natural_instruction_contract_topup as topup  # noqa: E402
from src.eval.instruction_reviewer_pair_gate import load_g06_config  # noqa: E402

DEFAULT_ROOT = ROOT / ".workbuddy" / "instruction-natural-calibration-20261015"
DEFAULT_PACKET = DEFAULT_ROOT / "packet.json"
DEFAULT_RUN_A = DEFAULT_ROOT / "runtime" / "evaluation_only" / "ai_annotations" / "ai-7b384bb6a899476ea2ed2154e8f5c764"
DEFAULT_RUN_B = DEFAULT_ROOT / "runtime" / "evaluation_only" / "ai_annotations" / "ai-08619298c5b04a6da3ff4c53961490b7"
DEFAULT_V2 = ROOT / ".workbuddy" / "instruction-natural-calibration-20261015-v2-adjudication-r2" / "frozen-reference-v2.json"
DEFAULT_LINEAGE = ROOT / ".workbuddy" / "instruction-natural-calibration-20261015-v3-contract-topup"
DEFAULT_PLAN = DEFAULT_LINEAGE / "plan.json"
DEFAULT_FROZEN = DEFAULT_LINEAGE / "frozen-reference-v3.json"


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


def _load_key() -> None:
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
    parser.add_argument("--v2-reference", type=Path, default=DEFAULT_V2)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--topup-run", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--model-id", default="XingChenAGI/Xing4.0-29B")
    parser.add_argument("--rsi-home", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    if args.rsi_home:
        os.environ["RSI_HOME"] = str(args.rsi_home.resolve())
    _load_key()
    packet = _read(args.packet)
    cfg, _ = load_g06_config(args.model_id, ROOT)
    if args.action == "plan":
        plan = topup.build_plan(packet, args.run_a, args.run_b, cfg, model_id=args.model_id)
        _write_once(args.plan, plan)
        print(json.dumps({"status": "planned_zero_call", "plan": str(args.plan),
                          "plan_sha256": plan["plan_sha256"], "planned_calls": 3,
                          "network_calls": 0}, ensure_ascii=False))
        return 0
    plan = _read(args.plan)
    if args.action == "execute":
        if args.expected_plan_sha256 != plan.get("plan_sha256"):
            raise ValueError("--expected-plan-sha256 must match immutable top-up plan")
        result = topup.execute(packet, args.run_a, args.run_b, cfg, plan)
        print(json.dumps({"status": "completed", "run_dir": result["run_dir"],
                          "receipt_sha256": result["receipt"].get("output_sha256"),
                          "calls": result["receipt"].get("call_count"),
                          "measured_usd": result["receipt"].get("measured_usd")}, ensure_ascii=False))
        return 0
    if args.topup_run is None:
        parser.error("freeze requires --topup-run")
    frozen_v2 = _read(args.v2_reference)
    job = topup.verify(packet, args.run_a, args.run_b, plan, args.topup_run)
    frozen = topup.merge_v3(frozen_v2, job)
    _write_once(args.output, frozen)
    print(json.dumps({"status": "frozen", "output": str(args.output),
                      "reference_sha256": frozen["reference_sha256"],
                      "coverage": frozen["reference_class_coverage"],
                      "coverage_sufficient": frozen["reference_class_coverage_sufficient"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
