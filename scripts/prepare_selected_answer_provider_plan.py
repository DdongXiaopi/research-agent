#!/usr/bin/env python3
"""Freeze a provider-neutral selected-answer calibration plan and approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.grounding_selected_answer_calibration import build_selected_answer_plan
from src.eval.selected_answer_calibration_runner import _sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", default="2026-10-18T20:00:00Z")
    parser.add_argument("--model-id", default="deepseek-flash")
    parser.add_argument("--max-cost-usd", type=float, default=0.50)
    parser.add_argument("--max-tokens", type=int, default=1536,
                        choices=(512, 1024, 1536, 2048),
                        help="Frozen completion budget; reasoning models need more than the legacy 512.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("selected-answer provider plan output must be new")
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    protocol = {
        "provider": "deepseek_official",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "price_policy": "published_tariff",
        "paid_fallback": False,
        "response_format": {"type": "json_object"},
        # DeepSeek's official API uses this field for the thinking toggle;
        # enable_thinking is a provider-specific legacy flag and is not
        # reliable on deepseek-flash.
        "request_options": {"thinking": {"type": "disabled"}},
        "max_cost_usd": args.max_cost_usd,
        "max_physical_attempts_per_case": 1,
        "retries_per_case": 0,
    }
    plan = build_selected_answer_plan(
        packet, reference, model_id=args.model_id,
        model_protocol_sha256=_sha(protocol), created_at=args.created_at,
        model_protocol=protocol, max_tokens=args.max_tokens)
    approval_body = {
        "schema": "g05-selected-answer-calibration-dispatch-approval-v1",
        "plan_sha256": plan["plan_sha256"],
        "request_count": plan["request_count"],
        "max_physical_attempts": plan["max_physical_attempts"],
        "hard_cost_usd": f"{args.max_cost_usd:.2f}",
        "dispatch_authorized": True,
    }
    approval = {**approval_body, "approval_sha256": _sha(approval_body)}
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    (output / "approval.json").write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"plan": str(output / "plan.json"),
                      "approval": str(output / "approval.json"),
                      "plan_sha256": plan["plan_sha256"],
                      "request_count": plan["request_count"],
                      "max_cost_usd": args.max_cost_usd}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
