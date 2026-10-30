"""Derive a new calibration packet from never-started cases in a closed failed job.

Only completed HTTP operations and the failed call boundary are inspected. No
provider is constructed, and no completed or ambiguous case is replayed.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_healthver_same_claim_top3_reference import _case_id, _source_hashes
from scripts.score_task_input_grounding_human_reference import _validate_packet
from src.eval.ai_jobs import decode_json_response
from src.eval.grounding_v11 import evidence_selection_messages_v11
from src.eval.grounding_v14 import claim_relation_messages_v14
from src.eval.healthver_reference import build_reference
from src.runtime.call_ledger import summarize_ledger
from src.util import read_json, sha256_file, sha256_obj, write_json


def never_started_case_ids(packet: dict, calls: list[dict], *, calls_per_case: int) -> tuple[list[str], list[str]]:
    """Return untouched cases and all cases touched by a started call prefix."""
    cases = packet.get("cases")
    if not isinstance(cases, list) or not cases or type(calls_per_case) is not int or calls_per_case < 1:
        raise ValueError("partial calibration packet/call layout is malformed")
    if not isinstance(calls, list) or not calls:
        raise ValueError("partial calibration call ledger is empty")
    if any(row.get("index") != index for index, row in enumerate(calls)):
        raise ValueError("partial calibration calls are not a contiguous prefix")
    touched_count = math.ceil(len(calls) / calls_per_case)
    if touched_count >= len(cases):
        raise ValueError("failed calibration has no unmeasured cases left")
    touched = [case["case_id"] for case in cases[:touched_count]]
    untouched = [case["case_id"] for case in cases[touched_count:]]
    if len(untouched) < 30:
        raise ValueError("fewer than 30 wholly unmeasured cases remain for calibration")
    return untouched, touched


def _validate_closed_partial_job(run_dir: Path, packet: dict, reference: dict) -> tuple[dict, dict, list[dict], list[str], list[str]]:
    root = run_dir.resolve()
    base = (ROOT / "artifacts/evaluation_only/ai_annotations").resolve()
    if not root.is_relative_to(base):
        raise ValueError("partial recovery accepts only private evaluation-only job directories")
    plan, inputs, status, calls = (read_json(root / name) for name in
                                   ("plan.json", "inputs.json", "status.json", "calls.json"))
    if (status.get("status") != "failed" or status.get("error_type") != "ProviderError"
            or (root / "receipt.json").exists() or (root / "output.json").exists()
            or plan.get("kind") != "grounding_judge"
            or plan.get("plan_sha256") != sha256_obj({k: v for k, v in plan.items() if k != "plan_sha256"})
            or inputs != {"packet": packet, "reference": reference}
            or sha256_obj(inputs) != plan.get("inputs_sha256")):
        raise ValueError("partial job identity/status is not safe for remainder extraction")
    accounting = summarize_ledger(root / "call-ledger.jsonl")
    if (accounting != status.get("request_accounting")
            or accounting.get("logical_incomplete") != 0
            or accounting.get("physical_incomplete") != 0
            or accounting.get("physical_attempts_exact") is not True
            or accounting.get("logical_status_counts") != {"success": len(calls) - 1, "failed": 1}
            or accounting.get("physical_status_counts", {}).get("None", 0) < 1):
        raise ValueError("failed job ledger is not exactly closed with one terminal failed operation")
    if not calls or calls[-1].get("status") != "started" or any(
            row.get("status") != "completed" for row in calls[:-1]):
        raise ValueError("partial job does not end at one incomplete operation after a completed prefix")
    for index, row in enumerate(calls):
        case = packet["cases"][index // 2]
        presented = [{**doc, "text": str(doc.get("text", ""))[:6000]}
                     for doc in case["evidence"]]
        if index % 2 == 0:
            system, user = evidence_selection_messages_v11(case["question"], presented)
        else:
            prior = read_json(root / f"call-{index - 1}.json")
            try:
                selected = decode_json_response(prior["raw_output"])
            except (ValueError, TypeError):
                selected = prior["raw_output"]
            system, user = claim_relation_messages_v14(case["question"], presented, selected)
        expected_payload = {"runtime_user_message": user}
        if (row.get("index") != index or row.get("role") != "grounding_judge"
                or row.get("input_sha256") != sha256_obj(expected_payload)):
            raise ValueError("partial call is not bound to its exact frozen runtime prompt")
        if index == len(calls) - 1:
            continue
        record = read_json(root / f"call-{index}.json")
        if (record.get("role") != "grounding_judge" or record.get("payload") != expected_payload
                or record.get("system") != system or record.get("user") != user
                or row.get("response_sha256") is None):
            raise ValueError("completed call record identity is malformed")
        try:
            parsed = decode_json_response(record["raw_output"])
        except (ValueError, TypeError):
            parsed = record["raw_output"]
        if sha256_obj(parsed) != row["response_sha256"]:
            raise ValueError("completed response hash does not match its call ledger row")
    if (root / f"call-{len(calls) - 1}.json").exists():
        raise ValueError("failed request unexpectedly has a persisted successful response")
    untouched, touched = never_started_case_ids(packet, calls, calls_per_case=2)
    return plan, accounting, calls, untouched, touched


def derive(packet: dict, key: dict, reference: dict, run_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite derived calibration packet: {output_dir}")
    _validate_packet(packet, "reviewer_a")
    if (key.get("protocol_sha256") != packet.get("protocol_sha256")
            or key.get("key_sha256") != sha256_obj({k: v for k, v in key.items() if k != "key_sha256"})
            or reference != build_reference(packet)):
        raise ValueError("parent key/reference does not reproduce from the frozen HealthVer packet")
    plan, accounting, calls, untouched_ids, touched_ids = _validate_closed_partial_job(
        run_dir, packet, reference)
    remaining = [case for case in packet["cases"] if case["case_id"] in set(untouched_ids)]
    ref_by_id = {row["case_id"]: row for row in reference["per_case"]}
    key_by_hash = {row["input_sha256"]: row for row in key["cases"]}
    selected_key_rows = []
    for case in remaining:
        source = key_by_hash.get(case["input_sha256"])
        if source is None:
            raise ValueError("unmeasured case lacks its source identity in the private key")
        selected_key_rows.append(source)
    label_counts = Counter(ref_by_id[case["case_id"]]["resolved_verdict"] for case in remaining)
    if set(label_counts) != {"supported", "contradicted", "insufficient"} or min(label_counts.values()) < 3:
        raise ValueError("unmeasured remainder does not preserve the three-way calibration minimum")
    parent_run_id = Path(run_dir).name
    excluded_case_ids = sorted(touched_ids)
    derivation = {
        "method": "deterministic untouched suffix of a closed failed calibration job",
        "parent_packet_sha256": packet["packet_sha256"],
        "parent_protocol_sha256": packet["protocol_sha256"],
        "parent_plan_sha256": plan["plan_sha256"],
        "failed_run_id": parent_run_id,
        "excluded_case_ids": excluded_case_ids,
        "excluded_started_call_count": len(calls),
        "logical_operations_replayed": 0,
        "source_label_counts": dict(sorted(label_counts.items())),
    }
    context = dict(packet["protocol_context"])
    context.update({
        "schema": "task-input-grounding-reference-protocol-v7",
        "sampling": "parent class-balanced fresh claims; deterministic never-started suffix after closed failure",
        "per_label": min(label_counts.values()),
        "case_input_sha256s": sorted(case["input_sha256"] for case in remaining),
        "case_count": len(remaining),
        "source_hashes": _source_hashes(),
        "cohort_sha256": sha256_obj({"derivation": derivation,
                                      "case_input_sha256s": sorted(case["input_sha256"] for case in remaining)}),
        "label_counts": dict(sorted(label_counts.items())),
        "parent_protocol_sha256": packet["protocol_sha256"],
        "parent_packet_sha256": packet["packet_sha256"],
        "parent_plan_sha256": plan["plan_sha256"],
        "excluded_case_ids": excluded_case_ids,
        "derivation_source_sha256": sha256_file(Path(__file__)),
    })
    protocol_sha = sha256_obj(context)
    new_cases = []
    new_key_rows = []
    for case in remaining:
        key_row = key_by_hash[case["input_sha256"]]
        case_id = _case_id(protocol_sha, key_row["task_id"])
        new_cases.append({"case_id": case_id, "question": case["question"],
                          "evidence": case["evidence"], "input_sha256": case["input_sha256"]})
        new_key_rows.append({**key_row, "case_id": case_id})
    scope = {"dataset": "healthver", "split": "dev_optimize", "case_count": len(new_cases),
             "use": "source-native top-3 semantic calibration; no candidate confirmation",
             "derivation": {**derivation, "parent_protocol_sha256": packet["protocol_sha256"]},
             "limitations": ["same source dataset and parent fresh-claim cohort",
                             "all touched cases excluded; no completed request was replayed"]}
    derived = {"schema": "task-input-grounding-human-reference-v2",
               "protocol_sha256": protocol_sha, "protocol_context": context,
               "reviewer_slot": "reviewer_a", "scope": scope,
               "instructions": packet["instructions"], "cases": new_cases}
    derived["packet_sha256"] = sha256_obj(derived)
    _validate_packet(derived, "reviewer_a")
    derived_key = {"schema": key["schema"], "protocol_sha256": protocol_sha,
                   "cases": new_key_rows, "promotion_authorized": False}
    derived_key["key_sha256"] = sha256_obj(derived_key)
    derived_reference = build_reference(derived)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "reviewer_a_packet.json", derived)
    write_json(output_dir / "researcher_key_do_not_share.json", derived_key)
    write_json(output_dir / "source-reference.json", derived_reference)
    write_json(output_dir / "derivation-audit.json", {
        "schema": "healthver-grounding-unmeasured-remainder-v1",
        "derivation": derivation, "parent_run_request_accounting": accounting,
        "parent_packet_sha256": packet["packet_sha256"],
        "derived_packet_sha256": derived["packet_sha256"],
        "derived_protocol_sha256": protocol_sha,
        "derived_reference_sha256": derived_reference["reference_sha256"],
        "case_count": len(new_cases), "label_counts": dict(sorted(label_counts.items())),
        "physical_requests_during_derivation": 0,
    })
    return {"output_dir": str(output_dir), "case_count": len(new_cases),
            "label_counts": dict(sorted(label_counts.items())),
            "packet_sha256": derived["packet_sha256"], "protocol_sha256": protocol_sha}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--failed-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = derive(read_json(args.packet), read_json(args.key), read_json(args.reference),
                    args.failed_run_dir, args.output_dir)
    print(__import__("json").dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
