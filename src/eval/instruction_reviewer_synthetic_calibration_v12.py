"""Target-v9 adapter for the fresh synthetic instruction-reviewer cohort v12.

This module is deliberately independent of historical calibration dispatch.
It binds the nested-JSON target prompt, per-case schema, source spans, and the
v8 host parser to a zero-price diagnostic plan. Results remain synthetic-only
and cannot authorize production promotion.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from . import ai_jobs
from . import instruction_reviewer_cohort_v12 as cohort_v12
from . import instruction_reviewer_cohort_v13 as cohort_v13
from . import instruction_reviewer_rubric_v8 as rubric_v8
from . import instruction_reviewer_sharded_freeze as freeze
from . import instruction_reviewer_pair_gate as pair_gate
from .instruction_reviewer_pair_gate import family_registry_sha256, load_g06_config, model_family
from ..evolution.instruction_review import parse_prompt_grounding_review_v8
from ..evolution.instruction_review_v9 import (
    PROMPT_GROUNDING_V9_VERSION, prepare_prompt_grounding_v9,
    prompt_grounding_protocol_v9,
)
from ..evolution.task_contracts import CLINIFACT_CONTRACT
from ..util import repo_root, sha256_obj


PLAN_SCHEMA_V12 = "g06-synthetic-instruction-reviewer-calibration-plan-v12-target-v9"
OUTPUT_SCHEMA_V12 = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v12-target-v9"
INPUT_SCHEMA_V12 = "g06-synthetic-instruction-reviewer-calibration-input-v12-target-v9"
TARGET_TRANSPORT_V12 = "json_schema_per_case_dynamic_v9_host_validated"
TARGET_PARSER_V12 = "parse_prompt_grounding_review_v8"
TARGET_HOST_VALIDATION_V12 = "per_case_exact_v8_schema_and_source_quote_validation"
TARGET_MAX_TOKENS_V12 = 2048
# Qwen free-tier access returned HTTP 429/TPM after the first successful case.
# Bind a one-minute per-success cooldown into the frozen effective config so
# the next per-case call starts after the provider's rolling minute window.
TARGET_REQUEST_COOLDOWN_SECONDS_V12 = 60.0
MAX_COST_USD = 0.05
MAX_INPUT_CHARS = 16000


def _cohort_for_packet(packet: dict):
    schema = packet.get("schema") if isinstance(packet, dict) else None
    if schema == cohort_v12.SCHEMA:
        return cohort_v12
    if schema == cohort_v13.SCHEMA:
        return cohort_v13
    raise ValueError("unsupported v12/v13 calibration packet")


def _protocol_v12(packet: dict) -> dict:
    """Return the exact packet-bound target transport protocol and schema manifest."""
    _cohort_for_packet(packet)
    target = prompt_grounding_protocol_v9(CLINIFACT_CONTRACT)
    if packet.get("target_protocol") != target:
        raise ValueError("v12 target prompt protocol differs from current v9")
    case_manifest = packet.get("reference_packet_manifest")
    if not isinstance(case_manifest, list) or len(case_manifest) != len(packet.get("cases", [])):
        raise ValueError("v12 reference per-case manifest is incomplete")
    return {
        "messages_protocol": target,
        "messages_protocol_sha256": sha256_obj(target),
        "reference_protocol": packet["reference_protocol"],
        "reference_protocol_sha256": sha256_obj(packet["reference_protocol"]),
        "parser": TARGET_PARSER_V12,
        "parser_candidate_evidence": "authored_strategy_only",
        "transport": TARGET_TRANSPORT_V12,
        "response_schema_binding": "case-specific-sha256-in-inputs-and-plan",
        "target_protocol_sha256": sha256_obj(target),
        "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "dynamic_schema_manifest_sha256": sha256_obj(case_manifest),
        "fail_closed_host_validation": True,
        "promotion_authorized": False,
    }


def _case_target_v12(case: dict, packet: dict) -> dict:
    """Rebuild v9 system/user/schema from candidate text; reject packet drift."""
    request = prepare_prompt_grounding_v9(case["strategy_text"], CLINIFACT_CONTRACT)
    response_sha = sha256_obj(request["response_format"])
    candidate_sha = sha256_obj(request["candidate_spans"])
    contract_sha = sha256_obj(request["contract_spans"])
    rows = packet.get("reference_packet_manifest", [])
    manifest = next((row for row in rows if row.get("case_id") == case.get("case_id")), None)
    reference = _cohort_for_packet(packet).reference_payload(case)["reference_spans"]
    if (manifest is None
            or response_sha != manifest.get("response_format_sha256")
            or candidate_sha != manifest.get("candidate_spans_sha256")
            or contract_sha != manifest.get("contract_spans_sha256")
            or response_sha != reference.get("response_format_sha256")):
        raise ValueError("v12 target per-case schema or span identity drifted")
    return {**request,
        "target_response_format_sha256": response_sha,
        "target_candidate_spans_sha256": candidate_sha,
        "target_contract_spans_sha256": contract_sha}


def _parse_v12_target_output(raw: str, item: dict, packet: dict) -> tuple[dict, dict]:
    """Validate target protocol and source-bound per-case schema before parsing."""
    _cohort_for_packet(packet)
    target = prompt_grounding_protocol_v9(CLINIFACT_CONTRACT)
    if (packet.get("target_protocol") != target
            or target.get("version") != PROMPT_GROUNDING_V9_VERSION):
        raise ValueError("v12 target protocol identity changed before parsing")
    case = {"case_id": item.get("case_id"), "strategy_text": item["candidate_strategy"]}
    request = _case_target_v12(case, packet)
    for key in ("target_response_format_sha256", "target_candidate_spans_sha256",
                "target_contract_spans_sha256"):
        if item.get(key) != request[key]:
            raise ValueError("v12 target per-case schema or span identity drifted")
    parsed = parse_prompt_grounding_review_v8(
        raw, item["candidate_strategy"], CLINIFACT_CONTRACT)
    provenance = {
        "target_protocol_version": target["version"],
        "target_protocol_sha256": sha256_obj(target),
        "response_format_sha256": request["target_response_format_sha256"],
        "candidate_spans_sha256": request["target_candidate_spans_sha256"],
        "contract_spans_sha256": request["target_contract_spans_sha256"],
        "parser": TARGET_PARSER_V12,
        "schema_valid": parsed.get("schema_valid") is True,
    }
    return parsed, provenance


def _inputs_v12(packet: dict, frozen: dict, protocol: dict) -> dict:
    """Build scorer-only inputs; blinded reference evidence never enters prompts."""
    left = {row["case_id"]: row for row in frozen["reviewer_a"]["case_evidence"]}
    right = {row["case_id"]: row for row in frozen["reviewer_b"]["case_evidence"]}
    consensus = {row["case_id"]: row for row in frozen["cases"]}
    cases = []
    schema_manifest = []
    for case in packet["cases"]:
        a, b = left[case["case_id"]], right[case["case_id"]]
        target = _case_target_v12(case, packet)
        cases.append({
            "case_id": case["case_id"],
            "candidate_strategy": case["strategy_text"],
            "reference": copy.deepcopy(consensus[case["case_id"]]["verdicts"]),
            "reviewer_evidence": {
                "reviewer_a": {"verdicts": copy.deepcopy(a["verdicts"]),
                    "normalization_issues": copy.deepcopy(a.get("normalization_issues", [])),
                    "receipt_file_sha256": a["receipt_file_sha256"]},
                "reviewer_b": {"verdicts": copy.deepcopy(b["verdicts"]),
                    "normalization_issues": copy.deepcopy(b.get("normalization_issues", [])),
                    "receipt_file_sha256": b["receipt_file_sha256"]}},
            "reviewer_disagreement": {dim: a["verdicts"][dim] != b["verdicts"][dim]
                for dim in ("assertion", "contract")},
            "target_response_format": target["response_format"],
            "target_response_format_sha256": target["target_response_format_sha256"],
            "target_candidate_spans_sha256": target["target_candidate_spans_sha256"],
            "target_contract_spans_sha256": target["target_contract_spans_sha256"],
        })
        schema_manifest.append({"case_id": case["case_id"],
            "response_format_sha256": target["target_response_format_sha256"],
            "candidate_spans_sha256": target["target_candidate_spans_sha256"],
            "contract_spans_sha256": target["target_contract_spans_sha256"]})
    manifest_sha = sha256_obj(schema_manifest)
    if manifest_sha != protocol["dynamic_schema_manifest_sha256"]:
        raise ValueError("v12 per-case target schema manifest differs from frozen protocol")
    target_sha = sha256_obj(packet["target_protocol"])
    reference_hash = packet["reference_protocol"]["protocol_hash"]
    return {
        "schema": INPUT_SCHEMA_V12,
        "packet_sha256": packet["packet_sha256"],
        "frozen_reference_sha256": frozen["frozen_reference_sha256"],
        "pair_gate_report_sha256": frozen["pair_gate_report_sha256"],
        "rubric_version": frozen["rubric_version"],
        "rubric_sha256": frozen["rubric_sha256"],
        "protocol": protocol,
        "protocol_sha256": sha256_obj(protocol),
        "target_protocol_version": packet["target_protocol"]["version"],
        "target_protocol_sha256": target_sha,
        "reference_protocol_hash": reference_hash,
        "target_transport": TARGET_TRANSPORT_V12,
        "dynamic_schema_manifest_sha256": manifest_sha,
        "thresholds": {"min_cases": 30, "min_approval_reference": 3,
            "min_rejection_reference": 3},
        "contract": copy.deepcopy(packet["contract"]),
        "cases": cases,
    }


def _job_protocol_metadata_v12(packet: dict, protocol: dict) -> dict:
    return {
        "protocol_version": packet["target_protocol"]["version"],
        "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
        "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "response_mode": TARGET_TRANSPORT_V12,
        "parser": TARGET_PARSER_V12,
        "dynamic_schema_manifest_sha256": protocol["dynamic_schema_manifest_sha256"],
        "host_validation": TARGET_HOST_VALIDATION_V12,
        "fail_closed": True,
    }


def _validate_packet_v12(packet: dict, workspace: Path) -> None:
    # Keep the historical adapter entry point for v12 callers, while routing
    # the fresh v13 cohort through its own immutable builder/source validator.
    # The v13 packet has the same target protocol and rubric, but a different
    # cohort identity and frozen-reference schema; accepting it through v12's
    # exact-identity validator would either reject valid work or weaken v12.
    schema = packet.get("schema") if isinstance(packet, dict) else None
    if schema == cohort_v12.SCHEMA:
        cohort = cohort_v12
        version = "v12"
    elif schema == cohort_v13.SCHEMA:
        cohort = cohort_v13
        version = "v13"
    else:
        raise ValueError("calibration requires a supported fresh v12/v13 packet")
    cohort.validate_packet_integrity(packet, workspace)
    cohort.validate_packet(packet, workspace)
    if (packet.get("case_count") != 40 or len(packet.get("cases", [])) != 40
            or packet.get("natural_distribution") is not False
            or packet.get("target_protocol") != prompt_grounding_protocol_v9(CLINIFACT_CONTRACT)
            or packet.get("reference_protocol", {}).get("protocol_hash")
                != rubric_v8.protocol_hash_v8()):
        raise ValueError(f"calibration requires the exact fresh {version}/v9/rubric-v8 packet")


def _validate_frozen_v12(packet: dict, frozen: dict, schedule_a: dict, state_a,
                         aggregate_a: dict, schedule_b: dict, state_b,
                         aggregate_b: dict) -> dict:
    """Delegate raw receipt reconstruction to freeze, then enforce v12 identity."""
    verified = freeze.verify_synthetic_diagnostic_reference(packet, frozen,
        schedule_a, state_a, aggregate_a, schedule_b, state_b, aggregate_b)
    bindings = freeze._protocol_bindings(packet)
    expected_frozen_schema = (freeze.FROZEN_SCHEMA_V11
        if packet.get("schema") == cohort_v13.SCHEMA else
        freeze.FROZEN_SCHEMA_V10)
    if (verified.get("schema") != expected_frozen_schema
            or verified.get("status") != "synthetic_diagnostic_frozen"
            or verified.get("annotation_origin") != "ai"
            or verified.get("synthetic_only") is not True
            or verified.get("natural_distribution") is not False
            or verified.get("promotion_authorized") is not False
            or verified.get("packet_sha256") != packet.get("packet_sha256")
            or any(verified.get(k) != v for k, v in bindings.items())
            or verified.get("reference_rubric_version") != rubric_v8.RUBRIC_VERSION
            or verified.get("pair_gate_report_sha256") != sha256_obj(verified.get("pair_gate_report"))):
        raise ValueError("frozen reference is stale or not bound to current source identities")
    rows = verified.get("cases")
    ids = [case["case_id"] for case in packet["cases"]]
    if not isinstance(rows, list) or [row.get("case_id") for row in rows] != ids:
        raise ValueError("v12 frozen consensus does not cover ordered packet cases")
    if verified.get("reference_class_coverage_sufficient") is not True:
        raise ValueError("v12 frozen reference class coverage is insufficient")
    for slot in ("reviewer_a", "reviewer_b"):
        reviewer = verified.get(slot, {})
        if ([row.get("case_id") for row in reviewer.get("case_evidence", [])] != ids
                or [row.get("case_id") for row in reviewer.get("case_receipts", [])] != ids):
            raise ValueError(f"v12 {slot} evidence and receipt coverage is incomplete")
    return verified


def _build_plan_v12(packet, frozen, schedule_a, state_a, aggregate_a,
                    schedule_b, state_b, aggregate_b, target_model_id: str,
                    workspace: Path):
    _validate_packet_v12(packet, workspace)
    verified = _validate_frozen_v12(packet, frozen, schedule_a, state_a,
        aggregate_a, schedule_b, state_b, aggregate_b)
    family = model_family(target_model_id)
    if family in {verified["reviewer_a"]["model_family"],
                  verified["reviewer_b"]["model_family"]}:
        raise ValueError("target scorer family must differ from both v12 reference families")
    # Reuse established target account/config gates, without changing historical schemas.
    from . import instruction_reviewer_synthetic_calibration as legacy_calibration
    cfg, config_provenance = load_g06_config(target_model_id, workspace)
    cfg.setdefault("models", {}).setdefault("task", {})[
        "cooldown"] = TARGET_REQUEST_COOLDOWN_SECONDS_V12
    access = legacy_calibration._zero_price_access(target_model_id, workspace, cfg)
    protocol = _protocol_v12(packet)
    inputs = _inputs_v12(packet, verified, protocol)
    metadata = _job_protocol_metadata_v12(packet, protocol)
    job_plan = ai_jobs.build_plan("instruction_judge", inputs, cfg,
        max_calls=len(packet["cases"]), tokens=TARGET_MAX_TOKENS_V12,
        max_input_chars=MAX_INPUT_CHARS, max_cost_usd=MAX_COST_USD,
        protocol_metadata=metadata)
    target_pair_gate_report = pair_gate.validate_g06_pair(
        packet, schedule_a, schedule_b, job_plan, workspace=workspace)
    if (not isinstance(target_pair_gate_report, dict)
            or target_pair_gate_report.get("status") != "verified_zero_call_pair"
            or target_pair_gate_report.get("planned_calls") != 0
            or target_pair_gate_report.get("promotion_authorized") is not False
            or target_pair_gate_report.get("target", {}).get("model_id") != target_model_id
            or target_pair_gate_report.get("target", {}).get("plan_sha256")
                != job_plan.get("plan_sha256")):
        raise ValueError("v12 target-family zero-call pair gate did not verify the exact job plan")
    target_pair_gate_report_sha256 = sha256_obj(target_pair_gate_report)
    binding = freeze._protocol_bindings(packet)
    body = {
        "schema": PLAN_SCHEMA_V12,
        "scope": "synthetic_diagnostic_only",
        "packet_sha256": packet["packet_sha256"],
        "frozen_reference_sha256": verified["frozen_reference_sha256"],
        "pair_gate_report_sha256": verified["pair_gate_report_sha256"],
        "family_registry_sha256": family_registry_sha256(),
        "source_artifacts": {slot: {
            "model_id": verified[slot]["model_id"],
            "model_family": verified[slot]["model_family"],
            "schedule_sha256": verified[slot]["schedule_sha256"],
            "aggregate_sha256": verified[slot]["aggregate_sha256"],
            "receipt_hashes": [row["receipt_file_sha256"]
                               for row in verified[slot]["case_receipts"]]}
            for slot in ("reviewer_a", "reviewer_b")},
        **binding,
        "protocol": protocol,
        "protocol_sha256": sha256_obj(protocol),
        "target_protocol_version": packet["target_protocol"]["version"],
        "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
        "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "target_transport": TARGET_TRANSPORT_V12,
        "dynamic_schema_manifest_sha256": protocol["dynamic_schema_manifest_sha256"],
        "ai_job_protocol_metadata": metadata,
        "thresholds": inputs["thresholds"],
        "config_provenance": config_provenance,
        "config_sha256": sha256_obj(cfg),
        "target_model_id": target_model_id,
        "target_model_family": family,
        "zero_price_access": access,
        "paid_fallback": False,
        "max_calls": len(packet["cases"]),
        "max_physical_attempts": len(packet["cases"]),
        "request_cooldown_seconds": TARGET_REQUEST_COOLDOWN_SECONDS_V12,
        "max_cost_usd": MAX_COST_USD,
        "max_tokens": TARGET_MAX_TOKENS_V12,
        "planned_calls": 0,
        "ai_job_plan": job_plan,
        "target_pair_gate_report": target_pair_gate_report,
        "target_pair_gate_report_sha256": target_pair_gate_report_sha256,
        "inputs_sha256": sha256_obj(inputs),
        "natural_distribution": False,
        "promotion_authorized": False,
        "production_gate_compatible": False,
    }
    return {**body, "plan_sha256": sha256_obj(body)}, inputs, cfg, verified


def calibration_plan_v12(packet: dict, frozen: dict,
        schedule_a: dict, state_a, aggregate_a: dict,
        schedule_b: dict, state_b, aggregate_b: dict,
        target_model_id: str, *, workspace: str | Path | None = None) -> dict:
    root = Path(workspace or repo_root()).resolve()
    plan, _inputs, _cfg, _verified = _build_plan_v12(packet, frozen,
        schedule_a, state_a, aggregate_a, schedule_b, state_b, aggregate_b,
        target_model_id, root)
    return plan


def _validate_target_result_output_v12(packet: dict, inputs: dict, output: dict,
        target_pair_gate_report: dict, target_pair_gate_report_sha256: str) -> None:
    protocol = _protocol_v12(packet)
    expected = {
        "schema": OUTPUT_SCHEMA_V12,
        "packet_sha256": packet["packet_sha256"],
        "target_protocol_version": packet["target_protocol"]["version"],
        "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
        "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "target_transport": TARGET_TRANSPORT_V12,
        "dynamic_schema_manifest_sha256": protocol["dynamic_schema_manifest_sha256"],
        "target_pair_gate_report_sha256": target_pair_gate_report_sha256,
    }
    if any(output.get(key) != value for key, value in expected.items()):
        raise ValueError("v12 target result is not bound to the frozen per-case v9 protocol")
    if (target_pair_gate_report_sha256 != sha256_obj(target_pair_gate_report)
            or output.get("target_pair_gate_report") != target_pair_gate_report):
        raise ValueError("v12 target-family pair-gate report drifted")
    cases = output.get("cases")
    expected_cases = inputs["cases"]
    if (not isinstance(cases, list)
            or [row.get("case_id") for row in cases]
                != [row.get("case_id") for row in expected_cases]):
        raise ValueError("v12 target result case order or coverage changed")
    for item, result in zip(expected_cases, cases):
        provenance = result.get("target_provenance", {})
        if (provenance.get("target_protocol_version") != packet["target_protocol"]["version"]
                or provenance.get("target_protocol_sha256") != sha256_obj(packet["target_protocol"])
                or provenance.get("response_format_sha256") != item["target_response_format_sha256"]
                or provenance.get("candidate_spans_sha256") != item["target_candidate_spans_sha256"]
                or provenance.get("contract_spans_sha256") != item["target_contract_spans_sha256"]
                or provenance.get("parser") != TARGET_PARSER_V12
                or type(result.get("parser_schema_valid")) is not bool
                or provenance.get("schema_valid") is not result.get("parser_schema_valid")):
            raise ValueError("v12 result case is not bound to its parser/schema identity")


def calibration_execute_v12(packet: dict, frozen: dict,
        schedule_a: dict, state_a, aggregate_a: dict,
        schedule_b: dict, state_b, aggregate_b: dict,
        target_model_id: str, plan: dict, *, workspace: str | Path | None = None,
        provider=None) -> dict:
    """Execute a prevalidated zero-price v12 plan with exact per-case schema calls."""
    root = Path(workspace or repo_root()).resolve()
    expected_plan, inputs, cfg, verified = _build_plan_v12(packet, frozen,
        schedule_a, state_a, aggregate_a, schedule_b, state_b, aggregate_b,
        target_model_id, root)
    if plan != expected_plan:
        raise ValueError("v12 target calibration plan/source/config identity changed")

    def worker(call):
        results = []
        for item in inputs["cases"]:
            case = {"case_id": item["case_id"], "strategy_text": item["candidate_strategy"]}
            target = _case_target_v12(case, packet)
            if (target["target_response_format_sha256"]
                    != item["target_response_format_sha256"]):
                raise ValueError("v12 target response schema changed before provider call")
            raw, provenance = call("instruction_judge",
                {"runtime_user_message": target["user"]}, target["system"],
                response_format=item["target_response_format"])
            parsed, parser_provenance = _parse_v12_target_output(raw, item, packet)
            provenance = {**provenance, **parser_provenance}
            assertion = parsed.get("assertion_review", parsed)
            contract = parsed.get("task_contract_review", {})
            results.append({"case_id": item["case_id"],
                "reference": copy.deepcopy(item["reference"]),
                "reviewer_evidence": copy.deepcopy(item["reviewer_evidence"]),
                "reviewer_disagreement": copy.deepcopy(item["reviewer_disagreement"]),
                "prediction": {
                    "assertion": assertion.get("verdict", "invalid")
                        if assertion.get("schema_valid") is True else "invalid",
                    "contract": contract.get("verdict", "invalid")
                        if contract.get("schema_valid") is True else "invalid"},
                "parser_schema_valid": parsed.get("schema_valid") is True,
                "target_provenance": provenance})
        return {"schema": OUTPUT_SCHEMA_V12,
            "packet_sha256": packet["packet_sha256"],
            "target_protocol_version": packet["target_protocol"]["version"],
            "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
            "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
            "target_transport": TARGET_TRANSPORT_V12,
            "dynamic_schema_manifest_sha256": _protocol_v12(packet)["dynamic_schema_manifest_sha256"],
            "target_pair_gate_report": copy.deepcopy(plan["target_pair_gate_report"]),
            "target_pair_gate_report_sha256": plan["target_pair_gate_report_sha256"],
            "frozen_reference_sha256": verified["frozen_reference_sha256"],
            "cases": results}

    job = ai_jobs.execute(plan["ai_job_plan"], inputs, cfg, worker, provider=provider)
    completed = ai_jobs.verify_completed(job["run_dir"], kind="instruction_judge")
    receipt, output = completed["receipt"], completed["output"]
    if (receipt.get("call_count") != len(packet["cases"])
            or receipt.get("usage_complete") is not True
            or receipt.get("measured_usd") != 0.0
            or len(output.get("cases", [])) != len(packet["cases"])):
        raise ValueError("v12 target result is incomplete, nonzero-cost, or usage-unverified")
    _validate_target_result_output_v12(packet, inputs, output,
        plan["target_pair_gate_report"], plan["target_pair_gate_report_sha256"])
    from ..runtime.provider import SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL
    calls = json.loads((Path(completed["root"]) / "calls.json").read_text(encoding="utf-8"))
    accounting = receipt.get("request_accounting", {})
    if (len(calls) != len(packet["cases"])
            or [row.get("index") for row in calls] != list(range(len(packet["cases"])))
            or any(row.get("status") != "completed"
                   or row.get("role") != "instruction_judge"
                   or row.get("response_format_sha256")
                        != item["target_response_format_sha256"]
                   for row, item in zip(calls, inputs["cases"]))
            or accounting.get("logical_operations") != len(packet["cases"])
            or accounting.get("logical_incomplete") != 0
            or accounting.get("physical_http_attempts") != len(packet["cases"])
            or accounting.get("physical_incomplete") != 0
            or accounting.get("physical_status_counts") != {"200": len(packet["cases"])}
            or accounting.get("physical_attempts_exact") is not True):
        raise ValueError("v12 scorer lacks exactly one successful physical request per case")
    for index, item in enumerate(inputs["cases"]):
        raw_call = json.loads((Path(completed["root"]) / f"call-{index}.json")
                              .read_text(encoding="utf-8"))
        cost = raw_call.get("cost", {})
        if (raw_call.get("role") != "instruction_judge"
                or raw_call.get("response_format_sha256")
                    != item["target_response_format_sha256"]
                or cost.get("usd") != 0.0
                or cost.get("usage_complete") is not True
                or cost.get("pricing_region") != "CN"
                or cost.get("pricing_is_account_invoice") is not False
                or cost.get("pricing_evidence_report_sha256")
                    != SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL.get(target_model_id)
                or cost.get("unknown_retry_attempts") != 0):
            raise ValueError("v12 scorer call lacks exact schema or zero-price CN usage evidence")
    from . import instruction_reviewer_synthetic_calibration as legacy_calibration
    dimensions = {dim: legacy_calibration._score(output["cases"], dim)
                  for dim in ("assertion", "contract")}
    return {"schema": OUTPUT_SCHEMA_V12,
        "packet_sha256": packet["packet_sha256"],
        "frozen_reference_sha256": verified["frozen_reference_sha256"],
        "pair_gate_report_sha256": verified["pair_gate_report_sha256"],
        "target_pair_gate_report": copy.deepcopy(plan["target_pair_gate_report"]),
        "target_pair_gate_report_sha256": plan["target_pair_gate_report_sha256"],
        "target_protocol_version": packet["target_protocol"]["version"],
        "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
        "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "dynamic_schema_manifest_sha256": _protocol_v12(packet)["dynamic_schema_manifest_sha256"],
        "target_model_id": target_model_id,
        "target_model_family": model_family(target_model_id),
        "target_plan_sha256": plan["plan_sha256"],
        "ai_job_plan_sha256": plan["ai_job_plan"]["plan_sha256"],
        "ai_job_receipt": receipt,
        "measurement_status": "measured_zero_price",
        "case_count": len(output["cases"]),
        "dimensions": dimensions,
        "cases": output["cases"],
        "synthetic_only": True,
        "natural_distribution": False,
        "promotion_authorized": False,
        "production_gate_compatible": False,
        "limitations": ["Synthetic diagnostic references do not establish natural-distribution validity."]}
