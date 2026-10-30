#!/usr/bin/env python3
"""Exhaustively reload and verify fresh v15 rubric-v7 A/B zero-price schedules."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_cohort_v15 as cohort  # noqa: E402
from src.eval import instruction_reviewer_pair_gate as pair_gate  # noqa: E402
from src.eval import instruction_reviewer_reference_shards as shards  # noqa: E402
from src.eval import ai_jobs  # noqa: E402
from src.runtime.provider import load_project_dotenv  # noqa: E402
from src.util import read_json, sha256_file  # noqa: E402


def run_preflight(packet_path: Path, schedule_a_path: Path,
                  schedule_b_path: Path, expected_a_sha256: str,
                  expected_b_sha256: str) -> dict:
    load_project_dotenv()
    packet_path = packet_path.resolve()
    packet = read_json(packet_path)
    cohort.validate_packet_integrity(packet, ROOT)
    if packet.get("schema") != cohort.SCHEMA:
        raise ValueError("only the fresh v15 packet is accepted")
    current = ai_jobs.source_tree_fingerprint()
    current_sha = current.get("sha256")
    if not isinstance(current_sha, str) or len(current_sha) != 64:
        raise ValueError("current source fingerprint is unavailable")
    inputs = (
        ("reviewer_a", schedule_a_path, expected_a_sha256, "Qwen/Qwen3-8B"),
        ("reviewer_b", schedule_b_path, expected_b_sha256, "XingChenAGI/Xing4.0-29B"),
    )
    checked_plan_hashes: list[str] = []
    evidence = []
    for slot, path, expected_sha, model_id in inputs:
        schedule_path = shards._controlled_work_path(path)
        schedule = read_json(schedule_path)
        if schedule.get("schedule_sha256") != expected_sha:
            raise ValueError(f"{slot} exact frozen schedule SHA mismatch")
        shards.validate_shard_schedule(packet, schedule, slot)
        cfg, _ = pair_gate.load_g06_config(model_id, ROOT)
        rebuilt = shards.build_reference_shard_schedule(packet, slot, cfg, ROOT)
        if (rebuilt.get("schedule_sha256") != schedule.get("schedule_sha256")
                or len(rebuilt.get("shards", [])) != 40):
            raise ValueError(f"{slot} schedule did not rebuild identically from current source")
        hashes = []
        for saved_row, rebuilt_row in zip(schedule["shards"], rebuilt["shards"], strict=True):
            plan = saved_row.get("plan")
            if (saved_row.get("inputs") != rebuilt_row.get("inputs")
                    or plan != rebuilt_row.get("plan")
                    or plan.get("source_tree_sha256") != current_sha
                    or plan.get("zero_price_only") is not True
                    or plan.get("max_cost_usd") != 0.0):
                raise ValueError(f"{slot} per-case plan is stale or violates the zero-price cap")
            hashes.append(plan["plan_sha256"])
            checked_plan_hashes.append(plan["plan_sha256"])
        if (schedule.get("model_id") != model_id
                or schedule.get("runtime_config", {}).get("models", {}).get(
                    "task", {}).get("base_url") != "https://api.siliconflow.cn/v1"
                or schedule.get("zero_price_only") is not True
                or schedule.get("max_cost_usd_per_model") != 0.0
                or schedule.get("paid_fallback") is not False
                or schedule.get("planned_calls") != 0):
            raise ValueError(f"{slot} schedule endpoint, model, cap, or fallback changed")
        evidence.append({
            "slot": slot,
            "model_id": model_id,
            "schedule_sha256": schedule["schedule_sha256"],
            "schedule_file_sha256": sha256_file(schedule_path),
            "plans_rebuilt_identically": len(hashes),
            "source_tree_sha256": current_sha,
            "pricing_evidence_sha256": schedule["price_evidence"][
                "current_official_pricing_evidence_sha256"],
            "zero_price_only": True,
            "max_cost_usd_per_model": 0.0,
            "paid_fallback": False,
        })
    if len(checked_plan_hashes) != 80 or len(set(checked_plan_hashes)) != 80:
        raise ValueError("expected 80 unique per-case plans across reviewers A/B")
    return {
        "schema": "g06-v15-reference-schedule-reload-preflight-v1",
        "status": "passed_zero_call_plan_integrity",
        "packet_sha256": packet["packet_sha256"],
        "packet_file_sha256": sha256_file(packet_path),
        "source_tree_sha256": current_sha,
        "source_file_count": current.get("file_count"),
        "plan_count": len(checked_plan_hashes),
        "plans_uniform_and_current": True,
        "schedules": evidence,
        "independent_reviewer_families": True,
        "project_dotenv_key_present": bool(os.environ.get("SILICONFLOW_API_KEY")),
        "endpoint": "https://api.siliconflow.cn/v1",
        "provider_calls": 0,
        "paid_fallback": False,
        "dispatch_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--schedule-a", type=Path, required=True)
    parser.add_argument("--schedule-b", type=Path, required=True)
    parser.add_argument("--schedule-a-sha256", required=True)
    parser.add_argument("--schedule-b-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = shards._controlled_work_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight report: {output}")
    result = run_preflight(args.packet, args.schedule_a, args.schedule_b,
                           args.schedule_a_sha256, args.schedule_b_sha256)
    shards._write_once(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
