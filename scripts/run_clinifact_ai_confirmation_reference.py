#!/usr/bin/env python3
"""Freeze, annotate, and resolve a blind CliniFact confirmation reference."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.clinifact_cohort import read_source_rows
from src.eval import ai_jobs
from src.eval.clinifact_ai_reference import (
    ADJUDICATE_PROMPT, LABEL_PROMPT, apply_adjudications, build_packets,
    build_packets_from_source_bundle, confirmation_capacity, merge_annotations,
    validate_annotation,
)
from src.util import read_json, sha256_file, sha256_obj, write_json

OUT = ROOT / ".workbuddy/clinifact-ai-confirmation-20261016"
AUDIT = ROOT / "artifacts/probe/clinifact_source_audit_20261008.json"
HISTORY = ROOT / "artifacts/probe/clinifact_history_scan_20261014_v18.json"
SOURCE_COHORT = ROOT / "artifacts/probe/clinifact_mipro_exploratory_cohort_20261014_v18.json"
SOURCE_REPORT = ROOT / "artifacts/sandboxes/clinifact-mipro-v18-20261014/probe/clinifact_mipro/5b9b9cf61889430e/report.json"
CONFIG = ROOT / "artifacts/sandboxes/ei-mipro-live-v2-20261005/frozen-run-config.yaml"
REVIEW_MODELS = {"reviewer_a": "THUDM/GLM-4-9B-0414",
                 "reviewer_b": "Qwen/Qwen3-8B"}
ADJUDICATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def reviewer_model(slot: str) -> str:
    """Allow a fresh packet to select another verified free reviewer family.

    The model identity is written into the frozen plan and receipt; this is an
    execution choice for a new packet, never a mutation of an existing plan.
    """
    override = os.environ.get("RSI_REFERENCE_REVIEWER_B_MODEL") if slot == "reviewer_b" else None
    return str(override or REVIEW_MODELS[slot])


def config_for(model_id: str) -> dict:
    cfg = load_config(CONFIG)
    model = cfg.setdefault("models", {}).setdefault("task", {})
    if model.get("provider") != "http":
        raise ValueError("CliniFact AI annotation requires the frozen HTTP provider")
    model["model_id"] = model_id
    return cfg


def freeze_packets() -> dict:
    train = ROOT / "data/raw/clinifact/train_set.csv"
    validation = ROOT / "data/raw/clinifact/validation_set.csv"
    return build_packets(
        train, validation, read_json(AUDIT), read_json(HISTORY),
        read_json(SOURCE_COHORT), source_report_sha256=sha256_file(SOURCE_REPORT),
        seed=2026101601,
    )


def reviewer_plan(slot: str, packet_path: Path, plan_path: Path) -> dict:
    packet = read_json(packet_path)
    model_id = reviewer_model(slot)
    cfg = config_for(model_id)
    response_path = plan_path.parent / f"{slot}_responses.json"
    previous = read_json(response_path) if response_path.is_file() else None
    completed_ids = {row["case_id"] for row in (previous or {}).get("annotations", [])}
    case_ids = [case["case_id"] for case in packet["cases"] if case["case_id"] not in completed_ids]
    if not case_ids:
        raise ValueError("reviewer packet has no unreviewed cases remaining")
    inputs = {"packet": packet, "slot": slot, "case_ids": case_ids}
    plan = ai_jobs.build_plan("clinifact_reference", inputs, cfg,
                              max_calls=len(case_ids), tokens=500,
                              max_input_chars=25000, max_cost_usd=0.25)
    write_json(plan_path, {**plan, "packet_path": str(packet_path),
                           "slot": slot, "case_ids": case_ids,
                           "expected_model_id": model_id})
    return plan


def execute_reviewer(plan_path: Path, output_path: Path) -> dict:
    frozen = read_json(plan_path)
    slot = frozen["slot"]
    model_id = frozen["expected_model_id"]
    packet = read_json(frozen["packet_path"])
    case_ids = set(frozen["case_ids"])
    inputs = {"packet": packet, "slot": slot, "case_ids": frozen["case_ids"]}
    plan = {key: value for key, value in frozen.items()
            if key in {"schema", "kind", "inputs_sha256", "config_sha256", "source_tree_sha256",
                       "model_id", "model_role", "max_calls", "max_tokens", "max_input_chars",
                       "max_retries_per_call", "max_physical_attempts", "max_cost_usd",
                       "model_calls", "annotation_origin", "plan_sha256"}}
    if plan["model_id"] != model_id or plan["inputs_sha256"] != sha256_obj(inputs):
        raise ValueError("review plan model or packet identity mismatch")
    previous = read_json(output_path) if output_path.is_file() else None
    if previous and previous.get("status") != "partial":
        raise ValueError("AI reviewer response is already complete")
    ensure_one_shot_unused(plan, output_path)
    cfg = config_for(model_id)
    def worker(call):
        rows = []
        for case in packet["cases"]:
            if case["case_id"] not in case_ids:
                continue
            output, provenance = call("clinifact_label", {
                "case_id": case["case_id"], "input_sha256": case["input_sha256"],
                "claim": case["input"]["claim"], "title": case["input"]["title"],
                "abstract": case["input"]["abstract"],
            }, LABEL_PROMPT)
            try:
                normalized = validate_annotation(case, output)
                unusable_reason = ""
            except (TypeError, ValueError) as exc:
                normalized = {"case_id": case["case_id"], "input_sha256": case["input_sha256"],
                              "label": "unable_to_judge", "evidence_quote": "",
                              "rationale": "Output failed local schema or exact-source validation."}
                unusable_reason = str(exc)
            rows.append({**normalized, "reviewer_slot": slot, "model_id": model_id,
                         "reviewer_run_id": provenance["review_run_id"],
                         "unusable_reason": unusable_reason})
        return {"schema": "clinifact-blind-ai-review-v1",
                "protocol_sha256": packet["protocol_sha256"],
                "packet_sha256": packet["packet_sha256"], "reviewer_slot": slot,
                "model_id": model_id, "annotations": rows}
    # The parent task may advance unrelated modules while this long blind
    # annotation is in flight.  Keep the frozen plan's own source fingerprint
    # in the receipt, but use ai_jobs' explicit legacy-source allowance so an
    # unrelated dirty-tree edit cannot invalidate an otherwise exact packet,
    # model, config, and input binding.
    result = ai_jobs.execute(
        plan, inputs, cfg, worker,
        allowed_legacy_source_tree_sha256=plan["source_tree_sha256"])
    prior_rows = (previous or {}).get("annotations", [])
    combined = prior_rows + result["output"]["annotations"]
    all_ids = {row["case_id"] for row in combined}
    expected_ids = {case["case_id"] for case in packet["cases"]}
    if len(all_ids) != len(combined) or not all_ids.issubset(expected_ids):
        raise ValueError("review continuation duplicated or substituted case IDs")
    run_ids = sorted({row["reviewer_run_id"] for row in combined})
    session_id = "review-session-" + sha256_obj([slot, model_id, run_ids])[:24]
    output = {**result["output"], "annotations": combined,
              "status": "completed" if all_ids == expected_ids else "partial",
              "run_id": session_id, "run_ids": run_ids,
              "completed_case_count": len(all_ids),
              "receipt": result["receipt"], "run_dir": result["run_dir"],
              "continuation_count": int((previous or {}).get("continuation_count", 0)) + (1 if previous else 0)}
    write_json(output_path, output)
    return {"run_dir": result["run_dir"], "output_path": str(output_path),
            "calls": result["receipt"]["call_count"],
            "usage_complete": result["receipt"]["usage_complete"],
            "measured_usd": result["receipt"]["measured_usd"]}


def adjudication_plan(merged_path: Path, plan_path: Path) -> dict:
    merged = read_json(merged_path)
    disputes = merged["disagreements_for_adjudication"]
    all_cases = [{"case_id": row["case_id"], "input_sha256": row["input_sha256"],
              "input": row["input"], "reviewer_a": row["reviewer_a"],
              "reviewer_b": row["reviewer_b"]} for row in disputes]
    response_path = plan_path.parent / "adjudication_responses.json"
    previous = read_json(response_path) if response_path.is_file() else None
    completed_ids = {row["case_id"] for row in (previous or {}).get("annotations", [])}
    cases = [case for case in all_cases if case["case_id"] not in completed_ids]
    if not cases:
        write_json(plan_path, {"status": "no_adjudication_required",
                               "protocol_sha256": merged["protocol_sha256"], "case_count": 0})
        return {"status": "no_adjudication_required", "case_count": 0}
    model_id = ADJUDICATOR_MODEL
    cfg = config_for(model_id)
    inputs = {"schema": "clinifact-ai-adjudication-input-v1",
              "protocol_sha256": merged["protocol_sha256"], "cases": cases}
    plan = ai_jobs.build_plan("clinifact_reference", inputs, cfg,
                              max_calls=len(cases), tokens=500,
                              max_input_chars=25000, max_cost_usd=0.25)
    write_json(plan_path, {**plan, "inputs": inputs,
                           "expected_model_id": model_id})
    return plan


def execute_adjudication(plan_path: Path, output_path: Path) -> dict:
    frozen = read_json(plan_path)
    if frozen.get("status") == "no_adjudication_required":
        return {"status": frozen["status"], "case_count": 0}
    model_id = frozen["expected_model_id"]
    inputs = frozen["inputs"]
    plan = {key: value for key, value in frozen.items()
            if key in {"schema", "kind", "inputs_sha256", "config_sha256", "source_tree_sha256",
                       "model_id", "model_role", "max_calls", "max_tokens", "max_input_chars",
                       "max_retries_per_call", "max_physical_attempts", "max_cost_usd",
                       "model_calls", "annotation_origin", "plan_sha256"}}
    if plan["inputs_sha256"] != sha256_obj(inputs) or plan["model_id"] != model_id:
        raise ValueError("adjudication plan identity mismatch")
    ensure_one_shot_unused(plan, output_path)
    cfg = config_for(model_id)
    previous = read_json(output_path) if output_path.is_file() else None
    if previous and previous.get("status") != "partial":
        raise ValueError("adjudication response is already complete")
    def worker(call):
        rows = []
        for case in inputs["cases"]:
            output, provenance = call("clinifact_adjudicator", case, ADJUDICATE_PROMPT)
            try:
                normalized = validate_annotation(case, output)
                unusable_reason = ""
            except (TypeError, ValueError) as exc:
                normalized = {"case_id": case["case_id"], "input_sha256": case["input_sha256"],
                              "label": "unable_to_judge", "evidence_quote": "",
                              "rationale": "Output failed local schema or exact-source validation."}
                unusable_reason = str(exc)
            rows.append({**normalized, "case_id": case["case_id"],
                         "reviewer_run_id": provenance["review_run_id"],
                         "unusable_reason": unusable_reason})
        return {"schema": "clinifact-blind-ai-adjudication-v1",
                "protocol_sha256": inputs["protocol_sha256"],
                "model_id": model_id, "annotations": rows}
    result = ai_jobs.execute(
        plan, inputs, cfg, worker,
        allowed_legacy_source_tree_sha256=plan["source_tree_sha256"])
    combined = (previous or {}).get("annotations", []) + result["output"]["annotations"]
    case_ids = {row["case_id"] for row in combined}
    expected_ids = {case["case_id"] for case in inputs["cases"]}
    # A continuation plan contains only the remaining cases; validate against
    # the complete frozen adjudication input attached to the original plan.
    merge_path = output_path.parent / "reviewer_merge.json"
    all_expected = {row["case_id"] for row in read_json(merge_path)["disagreements_for_adjudication"]}
    if len(case_ids) != len(combined) or not case_ids.issubset(all_expected):
        raise ValueError("adjudication continuation duplicated or substituted case IDs")
    run_ids = sorted({row["reviewer_run_id"] for row in combined})
    output = {**result["output"], "annotations": combined,
              "status": "completed" if case_ids == all_expected else "partial",
              "run_id": "adjudication-session-" + sha256_obj([model_id, run_ids])[:24],
              "run_ids": run_ids,
              "receipt": result["receipt"], "run_dir": result["run_dir"]}
    write_json(output_path, output)
    return {"run_dir": result["run_dir"], "output_path": str(output_path),
            "calls": result["receipt"]["call_count"],
            "usage_complete": result["receipt"]["usage_complete"],
            "measured_usd": result["receipt"]["measured_usd"]}


def recover_failed_adjudication(plan_path: Path, output_path: Path,
                                expected_plan_sha256: str) -> dict:
    frozen = read_json(plan_path)
    runtime_root = Path(os.environ["RSI_HOME"]).resolve() / "evaluation_only/ai_annotations"
    matches = []
    for job in runtime_root.glob("ai-*"):
        try:
            saved_plan = read_json(job / "plan.json")
            status = read_json(job / "status.json")
            calls = read_json(job / "calls.json")
        except (OSError, ValueError):
            continue
        if (saved_plan.get("plan_sha256") == expected_plan_sha256
                and saved_plan.get("kind") == "clinifact_reference"
                and status.get("status") == "failed"):
            matches.append((job, saved_plan, status, calls))
    if len(matches) != 1:
        raise ValueError("expected exactly one failed adjudication job for the challenged plan")
    job, saved_plan, status, calls = matches[0]
    inputs = read_json(job / "inputs.json")
    model_id = saved_plan.get("model_id")
    if (saved_plan.get("plan_sha256") != expected_plan_sha256
            or saved_plan.get("inputs_sha256") != sha256_obj(inputs)
            or frozen.get("expected_model_id") != model_id):
        raise ValueError("failed adjudication inputs or model do not match the challenged run")
    if not calls or len(calls) != int(status.get("call_count", -1)) or any(
            row.get("status") != "completed" or row.get("index") != index
            for index, row in enumerate(calls)):
        raise ValueError("failed adjudication has incomplete call records; do not replay")
    from src.runtime.call_ledger import summarize_ledger
    accounting = summarize_ledger(job / "call-ledger.jsonl")
    statuses = accounting.get("physical_status_counts") or {}
    if (accounting.get("logical_incomplete") != 0 or accounting.get("physical_incomplete") != 0
            or accounting.get("logical_operations") != len(calls)
            or accounting.get("physical_attempts_exact") is not True
            or accounting.get("physical_http_attempts", 0) > 2 * len(calls)
            or set(statuses) - {"200", "None"}
            or statuses.get("200", 0) < len(calls)):
        raise ValueError("failed adjudication is not fully closed in the physical ledger")
    case_by_id = {row["case_id"]: row for row in inputs["cases"]}
    rows = []
    for index, call_record in enumerate(calls):
        raw = read_json(job / f"call-{index}.json")
        payload = raw.get("payload") or {}
        case_id = payload.get("case_id")
        case = case_by_id.get(case_id)
        if (raw.get("role") != "clinifact_adjudicator" or case is None
                or call_record.get("input_sha256") != sha256_obj(payload)
                or payload.get("input_sha256") != case.get("input_sha256")):
            raise ValueError("failed adjudication raw call does not bind an exact frozen dispute")
        try:
            normalized = validate_annotation(case, ai_jobs.decode_json_response(raw["raw_output"]))
            reason = ""
        except (KeyError, TypeError, ValueError) as exc:
            normalized = {"case_id": case_id, "input_sha256": case["input_sha256"],
                          "label": "unable_to_judge", "evidence_quote": "",
                          "rationale": "Output failed local schema or exact-source validation."}
            reason = str(exc)
        rows.append({**normalized, "reviewer_run_id": f"{job.name}-{index}",
                     "unusable_reason": reason})
    run_ids = sorted({row["reviewer_run_id"] for row in rows})
    partial = {"schema": "clinifact-blind-ai-adjudication-v1",
               "protocol_sha256": inputs["protocol_sha256"], "model_id": model_id,
               "annotations": rows, "status": "partial",
               "run_id": "adjudication-session-" + sha256_obj([model_id, run_ids])[:24],
               "run_ids": run_ids, "recovered_failed_job": job.name,
               "recovery_accounting": accounting}
    if output_path.exists():
        raise ValueError("refusing to overwrite existing adjudication response")
    write_json(output_path, partial)
    return {"recovered_calls": len(rows),
            "invalid_as_unable_to_judge": sum(row["label"] == "unable_to_judge" for row in rows),
            "partial_response": str(output_path)}


def ensure_one_shot_unused(plan: dict, output_path: Path) -> None:
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("AI reference execution requires an isolated RSI_HOME")
    root = Path(os.environ["RSI_HOME"]).resolve() / "evaluation_only/ai_annotations"
    if root.exists():
        for receipt in root.glob("ai-*/plan.json"):
            try:
                existing = read_json(receipt)
            except (OSError, ValueError):
                continue
            if existing.get("plan_sha256") == plan.get("plan_sha256"):
                raise ValueError("AI reference plan already has a run artifact; freeze a fresh packet")


def require_plan_challenge(frozen: dict, expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected or "") or frozen.get("plan_sha256") != expected:
        raise ValueError("execution requires the caller's exact frozen plan SHA-256")


def recover_failed_reviewer(plan_path: Path, output_path: Path, expected_plan_sha256: str) -> dict:
    """Recover closed case calls from a failed job without replaying any request."""
    frozen = read_json(plan_path)
    slot, model_id = frozen["slot"], frozen["expected_model_id"]
    packet = read_json(frozen["packet_path"])
    plan = {key: value for key, value in frozen.items()
            if key in {"schema", "kind", "inputs_sha256", "config_sha256", "source_tree_sha256",
                       "model_id", "model_role", "max_calls", "max_tokens", "max_input_chars",
                       "max_retries_per_call", "max_physical_attempts", "max_cost_usd",
                       "model_calls", "annotation_origin", "plan_sha256"}}
    runtime_root = Path(os.environ["RSI_HOME"]).resolve() / "evaluation_only/ai_annotations"
    matches = []
    for job in runtime_root.glob("ai-*"):
        try:
            saved_plan = read_json(job / "plan.json")
            status = read_json(job / "status.json")
            calls = read_json(job / "calls.json")
        except (OSError, ValueError):
            continue
        expected_old_inputs = {"packet": packet, "slot": slot}
        if (saved_plan.get("plan_sha256") == expected_plan_sha256
                and saved_plan.get("inputs_sha256") == sha256_obj(expected_old_inputs)
                and saved_plan.get("model_id") == model_id
                and saved_plan.get("kind") == "clinifact_reference"
                and status.get("status") == "failed"):
            matches.append((job, saved_plan, status, calls))
    if len(matches) != 1:
        raise ValueError("expected exactly one failed AI job with this frozen plan")
    job, saved_plan, status, calls = matches[0]
    if saved_plan.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("failed job plan differs from caller challenge")
    if not calls or len(calls) != int(status.get("call_count", -1)):
        raise ValueError("failed AI job has an incomplete or empty call index")
    if any(row.get("status") != "completed" or row.get("index") != index
           for index, row in enumerate(calls)):
        raise ValueError("failed AI job contains an open/invalid call; do not replay")
    from src.runtime.call_ledger import summarize_ledger
    accounting = summarize_ledger(job / "call-ledger.jsonl")
    if (accounting.get("logical_incomplete") != 0 or accounting.get("physical_incomplete") != 0
            or accounting.get("logical_operations") != len(calls)
            or accounting.get("physical_http_attempts") != len(calls)
            or accounting.get("physical_status_counts") != {"200": len(calls)}):
        raise ValueError("failed AI job is not fully closed in the request ledger")
    subset = [case for case in packet["cases"] if case["case_id"] in set(frozen["case_ids"])]
    if len(subset) != len(frozen["case_ids"]):
        raise ValueError("frozen plan contains unknown case identities")
    rows = []
    for index, call_record in enumerate(calls):
        raw = read_json(job / f"call-{index}.json")
        payload = raw.get("payload") or {}
        if (call_record.get("input_sha256") != sha256_obj(payload)
                or raw.get("role") != "clinifact_label"):
            raise ValueError("failed AI job raw call identity mismatch")
        case_id = payload.get("case_id")
        case = next((item for item in subset if item["case_id"] == case_id), None)
        if case is None or payload.get("input_sha256") != case["input_sha256"]:
            raise ValueError("failed AI job call does not map to an exact frozen case")
        try:
            output = ai_jobs.decode_json_response(raw["raw_output"])
            normalized = validate_annotation(case, output)
            reason = ""
        except (KeyError, TypeError, ValueError) as exc:
            normalized = {"case_id": case_id, "input_sha256": case["input_sha256"],
                          "label": "unable_to_judge", "evidence_quote": "",
                          "rationale": "Output failed local schema or exact-source validation."}
            reason = str(exc)
        rows.append({**normalized, "reviewer_slot": slot, "model_id": model_id,
                     "reviewer_run_id": f"{job.name}-{index}", "unusable_reason": reason})
    run_ids = sorted({row["reviewer_run_id"] for row in rows})
    partial = {"schema": "clinifact-blind-ai-review-v1",
               "protocol_sha256": packet["protocol_sha256"],
               "packet_sha256": packet["packet_sha256"], "reviewer_slot": slot,
               "model_id": model_id, "status": "partial", "annotations": rows,
               "run_id": "review-session-" + sha256_obj([slot, model_id, run_ids])[:24],
               "run_ids": run_ids, "completed_case_count": len(rows),
               "recovered_failed_job": job.name, "recovery_accounting": accounting,
               "recovered_response_sha256": sha256_file(job / "calls.json")}
    if output_path.exists():
        raise ValueError("refusing to overwrite an existing reviewer response")
    write_json(output_path, partial)
    return {"recovered_calls": len(rows), "invalid_as_abstain": sum(row["label"] == "unable_to_judge" for row in rows),
            "partial_response": str(output_path), "run_id": partial["run_id"]}


def finalize(bundle_dir: Path) -> dict:
    bundle = read_json(bundle_dir / "reference_packets.json")
    pa, pb = bundle["packets"]["reviewer_a"], bundle["packets"]["reviewer_b"]
    ra = read_json(bundle_dir / "reviewer_a_responses.json")
    rb = read_json(bundle_dir / "reviewer_b_responses.json")
    merged = merge_annotations(pa, ra, pb, rb)
    write_json(bundle_dir / "reviewer_merge.json", merged)
    adjudication_path = bundle_dir / "adjudication_responses.json"
    if merged["disagreements_for_adjudication"]:
        adjudication = read_json(adjudication_path)
        adjudications = adjudication["annotations"]
        reference = apply_adjudications(merged, adjudications,
                                        model_id=adjudication["model_id"], run_id=adjudication["run_id"])
    else:
        reference = apply_adjudications(merged, [], model_id=ADJUDICATOR_MODEL,
                                        run_id="unused-no-disagreements")
    # Preserve the exact candidate binding from the frozen packet.  Without
    # this field a later candidate-bound confirmation could only compare the
    # cohort hash and would have to reject an otherwise valid reference.
    reference["candidate_binding"] = bundle["candidate_binding"]
    capacity = confirmation_capacity(reference)
    reference["capacity_audit"] = capacity
    reference["private_source_label_crosstab_pending"] = True
    write_json(bundle_dir / "resolved_reference.json", reference)
    cohort = {
        "schema": "clinifact-ai-confirmation-cohort-v1",
        "status": "AI_REFERENCE_FROZEN" if capacity["eligible"] else "AI_REFERENCE_FROZEN_INSUFFICIENT_CLASS_CAPACITY",
        "candidate_binding": bundle["candidate_binding"],
        "protocol_sha256": bundle["protocol_sha256"],
        "reference_sha256": sha256_file(bundle_dir / "resolved_reference.json"),
        "source_hashes": bundle["source_hashes"],
        "minimum_annotation_votes": 2,
        "resolution_policy": "two-model agreement or third-model majority; unresolved cases excluded",
        "class_counts": reference["class_counts"], "task_count": reference["usable_count"],
        "task_ids": sorted(row["task_id"] for row in reference["consensus"]),
        "component_ids": sorted(row["component_id"] for row in reference["consensus"]),
        "eligible_for_confirmation": capacity["eligible"],
        "official_test_read": False, "selection_gate_read": False,
        "source_label_used_for_reference": False,
    }
    cohort["cohort_sha256"] = sha256_obj(cohort)
    write_json(bundle_dir / "confirmation_cohort.json", cohort)
    return {"reference": str(bundle_dir / "resolved_reference.json"),
            "cohort": str(bundle_dir / "confirmation_cohort.json"),
            "class_counts": reference["class_counts"], "usable_count": reference["usable_count"],
            "eligible_for_confirmation": capacity["eligible"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "plan-review", "execute-review", "recover-review",
                                            "merge", "plan-adjudication", "execute-adjudication",
                                            "recover-adjudication", "finalize"))
    parser.add_argument("--slot", choices=tuple(REVIEW_MODELS))
    parser.add_argument("--bundle-dir", type=Path, default=OUT)
    parser.add_argument("--source-bundle", type=Path,
                        help="freeze from an unlabeled CT.gov source bundle instead of the legacy CliniFact cohort")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args()
    root = args.bundle_dir
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "freeze":
        if any(root.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty bundle: {root}")
        if args.source_bundle:
            source_path = args.source_bundle.resolve()
            source = read_json(source_path)
            bundle = build_packets_from_source_bundle(
                source, source_bundle_sha256=sha256_file(source_path))
        else:
            bundle = freeze_packets()
        write_json(root / "reference_packets.json", bundle)
        for slot, suffix in (("reviewer_a", "a"), ("reviewer_b", "b")):
            write_json(root / f"reviewer_{suffix}_packet.json", bundle["packets"][slot])
        write_json(root / "private_source_key.json", bundle["private_key"])
        print(json.dumps({"protocol_sha256": bundle["protocol_sha256"],
                          "case_count": bundle["packets"]["reviewer_a"]["cases"].__len__(),
                          "reviewer_models": REVIEW_MODELS,
                          "adjudicator_model": ADJUDICATOR_MODEL,
                          "source_label_shown_to_reviewers": False}, ensure_ascii=False))
    elif args.action == "plan-review":
        if args.slot is None:
            parser.error("--slot is required")
        suffix = "a" if args.slot == "reviewer_a" else "b"
        plan = reviewer_plan(args.slot, root / f"reviewer_{suffix}_packet.json",
                             root / f"{args.slot}_plan.json")
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "model_id": plan["model_id"],
                          "calls": plan["max_calls"], "physical_attempts": plan["max_physical_attempts"],
                          "max_cost_usd": plan["max_cost_usd"]}, ensure_ascii=False))
    elif args.action == "execute-review":
        if args.slot is None:
            parser.error("--slot is required")
        require_plan_challenge(read_json(root / f"{args.slot}_plan.json"), args.expected_plan_sha256)
        print(json.dumps(execute_reviewer(root / f"{args.slot}_plan.json",
                                          root / f"{args.slot}_responses.json"), ensure_ascii=False))
    elif args.action == "recover-review":
        if args.slot is None:
            parser.error("--slot is required")
        print(json.dumps(recover_failed_reviewer(root / f"{args.slot}_plan.json",
                                                 root / f"{args.slot}_responses.json",
                                                 args.expected_plan_sha256), ensure_ascii=False))
    elif args.action == "merge":
        bundle = read_json(root / "reference_packets.json")
        merged = merge_annotations(bundle["packets"]["reviewer_a"],
                                   read_json(root / "reviewer_a_responses.json"),
                                   bundle["packets"]["reviewer_b"],
                                   read_json(root / "reviewer_b_responses.json"))
        write_json(root / "reviewer_merge.json", merged)
        print(json.dumps({"agreement_count": merged["agreement_count"],
                          "disagreement_count": merged["disagreement_count"],
                          "adjudicator_calls": len(merged["disagreements_for_adjudication"])}, ensure_ascii=False))
    elif args.action == "plan-adjudication":
        plan = adjudication_plan(root / "reviewer_merge.json", root / "adjudication_plan.json")
        print(json.dumps({"status": plan.get("status", "planned"),
                          "model_id": plan.get("model_id"), "calls": plan.get("max_calls", 0),
                          "physical_attempts": plan.get("max_physical_attempts", 0),
                          "plan_sha256": plan.get("plan_sha256")}, ensure_ascii=False))
    elif args.action == "execute-adjudication":
        require_plan_challenge(read_json(root / "adjudication_plan.json"), args.expected_plan_sha256)
        print(json.dumps(execute_adjudication(root / "adjudication_plan.json",
                                              root / "adjudication_responses.json"), ensure_ascii=False))
    elif args.action == "recover-adjudication":
        print(json.dumps(recover_failed_adjudication(root / "adjudication_plan.json",
                                                     root / "adjudication_responses.json",
                                                     args.expected_plan_sha256), ensure_ascii=False))
    else:
        print(json.dumps(finalize(root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
