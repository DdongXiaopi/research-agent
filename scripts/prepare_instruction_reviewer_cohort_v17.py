#!/usr/bin/env python3
"""Freeze a fresh G06 atomic-v8 reviewer challenge and a no-dispatch plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.instruction_reviewer_cohort_v17 import (  # noqa: E402
    OUTPUT_DIRECTORY_ID, build_packet, build_reference_plan, reference_payload,
    validate_packet_integrity, validate_reference_plan,
)
from src.eval.instruction_reviewer_rubric_v8 import RUBRIC_V8  # noqa: E402
from src.util import sha256_obj  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_once(path: Path, value: object) -> str:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
    return _sha(data)


def build(output: Path) -> dict[str, object]:
    output = output.resolve()
    workbuddy = (ROOT / ".workbuddy").resolve()
    if workbuddy not in output.parents or output.exists():
        raise ValueError("output must be a fresh child of project .workbuddy")
    packet = build_packet(ROOT)
    validate_packet_integrity(packet, ROOT)
    plan = build_reference_plan(packet)
    validate_reference_plan(packet, plan)
    output.mkdir(parents=True, exist_ok=False)
    packet_sha = _write_once(output / "packet.json", packet)
    reviewer_files = {}
    for slot in ("reviewer_a", "reviewer_b"):
        cases = []
        for case in packet["cases"]:
            payload = reference_payload(case)
            user = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            cases.append({"case_id": case["case_id"],
                "candidate_sha256": case["candidate_sha256"],
                "input_sha256": sha256_obj(payload),
                "system": RUBRIC_V8, "user": user,
                "response_format": payload["reference_spans"]["response_format"],
            })
        reviewer_files[f"{slot.replace('_', '-')}-blind.json"] = _write_once(
            output / f"{slot.replace('_', '-')}-blind.json", {
                "schema": "instruction-reviewer-v17-blind-reference-packet-v1",
                "packet_sha256": packet["packet_sha256"], "slot": slot,
                "rubric_version": "instruction-reviewer-reference-v8",
                "cases": cases,
            })
    plan_sha = _write_once(output / "reference-plan-only.json", plan)
    preflight = {
        "schema": "instruction-reviewer-v17-zero-call-preflight-v1",
        "status": "passed_zero_call_plan_integrity",
        "packet_sha256": packet["packet_sha256"], "packet_file_sha256": packet_sha,
        "plan_sha256": plan["plan_sha256"], "plan_file_sha256": plan_sha,
        "case_count": packet["case_count"],
        "exposure_overlap_count": packet["exposure_audit"]["overlap_count"],
        "prior_packet_count": packet["exposure_audit"]["prior_packet_count"],
        "reference_calls": 0, "target_calls": 0, "model_calls": 0,
        "dispatch_authorized": False,
        "source_tree_freeze": "deferred_until_G01_G09_source_work_finishes",
    }
    preflight_sha = _write_once(output / "zero-call-preflight.json", preflight)
    file_hashes = {path.name: _sha(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()}
    manifest = {"schema": "instruction-reviewer-v17-plan-only-bundle-v1",
        "files": file_hashes, "bundle_sha256": sha256_obj(file_hashes),
        "packet_sha256": packet["packet_sha256"], "plan_sha256": plan["plan_sha256"],
        "model_calls": 0, "network_calls": 0, "dispatch_authorized": False}
    manifest_sha = _write_once(output / "bundle-manifest.json", manifest)
    return {"output": str(output), "packet_sha256": packet["packet_sha256"],
        "plan_sha256": plan["plan_sha256"], "bundle_sha256": manifest["bundle_sha256"],
        "case_count": packet["case_count"], "prior_packet_count": packet["exposure_audit"]["prior_packet_count"],
        "reviewer_files": reviewer_files, "preflight_sha256": preflight_sha,
        "manifest_file_sha256": manifest_sha, "model_calls": 0, "dispatch_authorized": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / ".workbuddy" / OUTPUT_DIRECTORY_ID)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
