"""Offline request planning and verified reference collection for G06 v17.

The frozen v17 plan remains immutable. This versioned execution plan binds its
two independent AI reviewers, exact v8 schema/payloads, and the reference gate.
Disagreements resolve to uncertain; this protocol has no adjudicator role.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import instruction_reviewer_cohort_v17 as cohort
from . import instruction_reviewer_rubric_v8 as rubric
from ..runtime.provider import published_price_for_endpoint
from ..util import sha256_obj

SCHEMA = "instruction-reviewer-v17-role-execution-plan-v1"
REFERENCE_SCHEMA = "instruction-reviewer-v17-ai-reference-v1"
GATE_SCHEMA = "instruction-reviewer-v17-reference-validity-gate-v1"
BASE_URL = "https://api.siliconflow.cn/v1"
ROLES = ("reviewer_a", "reviewer_b")
MAX_TOKENS = rubric.REFERENCE_MAX_OUTPUT_TOKENS
THRESHOLDS = {"min_cases": 30, "min_approval_reference": 3,
              "min_rejection_reference": 3, "invalid_outputs": 0}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _blind_payload(case: dict[str, Any]) -> dict[str, Any]:
    return cohort.reference_payload(case)


def _request(packet: dict[str, Any], plan: dict[str, Any], role: str,
             case: dict[str, Any]) -> dict[str, Any]:
    model = plan["reviewers"][role]
    blind = _blind_payload(case)
    user = _canonical(blind).decode("utf-8")
    response_format = blind["reference_spans"]["response_format"]
    planned = next(row for row in plan["blind_case_plan"]
        if row["slot"] == role and row["case_id"] == case["case_id"])
    if (planned["candidate_sha256"] != case["candidate_sha256"]
            or planned["input_sha256"] != sha256_obj(blind)
            or planned["system_prompt_sha256"] != sha256_obj(rubric.RUBRIC_V8)
            or planned["user_payload_sha256"] != sha256_obj(user)
            or planned["response_format_sha256"] != sha256_obj(response_format)):
        raise ValueError("v17 frozen blind case plan does not match runtime request")
    body = {"schema": "instruction-reviewer-v17-call-v1", "packet_sha256": packet["packet_sha256"],
        "reference_plan_sha256": plan["plan_sha256"], "role": role,
        "case_id": case["case_id"], "model": model,
        "system": rubric.RUBRIC_V8, "user": user,
        "response_format": response_format, "temperature": 0.0,
        "max_tokens": MAX_TOKENS, "prompt_sha256": _sha_text(rubric.RUBRIC_V8 + "\0" + user)}
    return {**body, "call_sha256": sha256_obj(body)}


def build_execution_plan(packet: dict[str, Any], reference_plan: dict[str, Any],
                         zero_price_evidence: dict[str, Any]) -> dict[str, Any]:
    cohort.validate_packet_integrity(packet)
    cohort.validate_reference_plan(packet, reference_plan)
    if (zero_price_evidence.get("schema") != "g05-zero-price-role-evidence-v1"
            or zero_price_evidence.get("provider") != "siliconflow"
            or zero_price_evidence.get("actual_account_invoice_verified") is not False):
        raise ValueError("dated official zero-price role evidence is required")
    evidence = {row.get("model_id"): row for row in zero_price_evidence.get("models", [])
                if isinstance(row, dict)}
    for role, model in reference_plan["reviewers"].items():
        row = evidence.get(model["model_id"])
        if (row is None or row.get("family") != model["model_family"]
                or row.get("input_price_cny_per_million") != "0"
                or row.get("output_price_cny_per_million") != "0"):
            raise ValueError(f"zero-price evidence missing for {role}")
    requests = {role: [_request(packet, reference_plan, role, case)
                       for case in packet["cases"]] for role in ROLES}
    body = {"schema": SCHEMA, "packet_sha256": packet["packet_sha256"],
        "reference_plan_sha256": reference_plan["plan_sha256"],
        "zero_price_evidence_sha256": sha256_obj(zero_price_evidence),
        "endpoint": BASE_URL + "/chat/completions",
        "transport_policy": {"allowed_host": "api.siliconflow.cn", "trust_env": False,
            "timeout_seconds": 240, "retries_per_call": 0, "paid_fallback": False,
            "zero_price_only": True},
        "reviewers": reference_plan["reviewers"], "requests": requests,
        "planned_calls": {role: len(requests[role]) for role in ROLES},
        "max_physical_attempts": sum(len(requests[role]) for role in ROLES),
        "target_gate": {"requires_reference_status": "validity_and_class_coverage_passed",
            "class_thresholds": THRESHOLDS, "target_calls_before_gate": 0},
        "model_calls": 0, "network_calls": 0, "dispatch_authorized": False}
    return {**body, "execution_plan_sha256": sha256_obj(body)}


def validate_execution_plan(packet: dict[str, Any], reference_plan: dict[str, Any],
                            evidence: dict[str, Any], execution_plan: dict[str, Any]) -> None:
    expected = build_execution_plan(packet, reference_plan, evidence)
    if execution_plan != expected:
        raise ValueError("v17 execution plan is stale or altered")


def _parse(raw: str, case: dict[str, Any], contract_text: str) -> dict[str, Any]:
    normalized, issues = rubric.normalize_reference_v8(raw, case["strategy_text"], contract_text)
    if normalized is None:
        raise ValueError("invalid_v8_reference:" + ";".join(issues))
    return normalized


def _receipt(raw: Any, request: dict[str, Any]) -> str:
    required = {"case_id", "packet_sha256", "reference_plan_sha256", "call_sha256",
        "role", "provider", "model_id", "model_family", "provider_request_id",
        "http_status", "response_model", "finish_reason", "usage", "cost_usd",
        "raw_response", "raw_response_sha256"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("v17 receipt schema mismatch")
    model = request["model"]
    if (raw["case_id"] != request["case_id"]
            or raw["packet_sha256"] != request["packet_sha256"]
            or raw["reference_plan_sha256"] != request["reference_plan_sha256"]
            or raw["call_sha256"] != request["call_sha256"]
            or raw["role"] != request["role"] or raw["provider"] != model["provider"]
            or raw["model_id"] != model["model_id"] or raw["model_family"] != model["model_family"]
            or raw["response_model"] != model["model_id"] or raw["http_status"] != 200
            or raw["finish_reason"] != "stop" or raw["cost_usd"] != "0.00"
            or not isinstance(raw["provider_request_id"], str) or not raw["provider_request_id"].strip()
            or not isinstance(raw["raw_response"], str)
            or raw["raw_response_sha256"] != _sha_text(raw["raw_response"])):
        raise ValueError("v17 receipt is not bound to a successful zero-cost planned request")
    usage = raw["usage"]
    if (not isinstance(usage, dict) or set(usage) != {"prompt_tokens", "completion_tokens"}
            or any(type(value) is not int or value < 0 for value in usage.values())
            or usage["completion_tokens"] > MAX_TOKENS):
        raise ValueError("v17 receipt usage is invalid")
    return raw["raw_response"]


def collect_reference(packet: dict[str, Any], reference_plan: dict[str, Any],
                      execution_plan: dict[str, Any], evidence: dict[str, Any],
                      receipts: dict[str, dict[str, dict[str, Any]]], *,
                      correction_requests: dict[str, dict[str, dict[str, Any]]] | None = None,
                      correction_receipts: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    validate_execution_plan(packet, reference_plan, evidence, execution_plan)
    if not isinstance(receipts, dict) or set(receipts) != set(ROLES):
        raise ValueError("v17 reference needs exact reviewer_a/reviewer_b receipts")
    by_id = {case["case_id"]: case for case in packet["cases"]}
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ROLES:
        planned = {row["case_id"]: row for row in execution_plan["requests"][role]}
        if set(receipts[role]) != set(by_id):
            raise ValueError(f"{role} receipt coverage is incomplete or extra")
        normalized[role] = {}
        for case_id, req in planned.items():
            correction = (correction_requests or {}).get(role, {}).get(case_id)
            correction_receipt = (correction_receipts or {}).get(role, {}).get(case_id)
            if correction is not None or correction_receipt is not None:
                if correction is None or correction_receipt is None:
                    raise ValueError("correction request and receipt must be supplied together")
                # A correction is an alternate response for the same frozen
                # case/model/payload/schema.  Correction lineages may append a
                # narrowly scoped system-side formatting reminder, which
                # necessarily changes ``system``, ``prompt_sha256`` and
                # ``call_sha256``.  Every other request field remains bound to
                # the original plan, so a different model, payload, schema or
                # case cannot be smuggled in as a repair.
                mutable_correction_fields = {"system", "prompt_sha256", "call_sha256"}
                if ({key: value for key, value in correction.items()
                     if key not in mutable_correction_fields}
                        != {key: value for key, value in req.items()
                           if key not in mutable_correction_fields}):
                    raise ValueError("correction request does not match frozen v17 request identity")
                raw = _receipt(correction_receipt, correction)
            else:
                raw = _receipt(receipts[role][case_id], req)
            normalized[role][case_id] = _parse(raw, by_id[case_id], packet["contract"]["text"])
    a_model, b_model = (reference_plan["reviewers"][role] for role in ROLES)
    if a_model["model_id"] == b_model["model_id"] or a_model["model_family"] == b_model["model_family"]:
        raise ValueError("v17 reference reviewers must have distinct model IDs and families")
    rows = []
    disagreements = []
    for case in packet["cases"]:
        cid = case["case_id"]
        a, b = normalized["reviewer_a"][cid], normalized["reviewer_b"][cid]
        verdicts = {}
        for dim, positive, negative in (("assertion", "claim_free", "contains_claims"),
                                        ("contract", "compatible", "incompatible")):
            av, bv = a[dim]["verdict"], b[dim]["verdict"]
            verdicts[dim] = av if av == bv else "uncertain"
            if av != bv:
                disagreements.append({"case_id": cid, "dimension": dim,
                    "reviewer_a": av, "reviewer_b": bv, "resolution": "uncertain"})
        rows.append({"case_id": cid, "reviewers": {"reviewer_a": a, "reviewer_b": b},
            "verdicts": verdicts,
            "raw_response_sha256": {role: ((correction_receipts or {}).get(role, {}).get(cid)
                or receipts[role][cid])["raw_response_sha256"] for role in ROLES}})
    coverage = {}
    for dim, positive, negative in (("assertion", "claim_free", "contains_claims"),
                                    ("contract", "compatible", "incompatible")):
        coverage[dim] = {"approval_reference": sum(row["verdicts"][dim] == positive for row in rows),
                         "rejection_reference": sum(row["verdicts"][dim] == negative for row in rows),
                         "uncertain": sum(row["verdicts"][dim] == "uncertain" for row in rows)}
    invalid = sum(bool(normalized[role][case["case_id"]] is None)
                  for role in ROLES for case in packet["cases"])
    sufficient = (len(rows) >= THRESHOLDS["min_cases"] and invalid == 0 and
        all(coverage[dim][key] >= THRESHOLDS["min_approval_reference"]
            for dim in coverage for key in ("approval_reference", "rejection_reference")))
    body = {"schema": REFERENCE_SCHEMA, "status": "validity_and_class_coverage_passed" if sufficient else "reference_gate_failed",
        "packet_sha256": packet["packet_sha256"], "reference_plan_sha256": reference_plan["plan_sha256"],
        "execution_plan_sha256": execution_plan["execution_plan_sha256"],
        "annotation_origin": "ai", "natural_distribution": False,
        "reviewers": {role: {**reference_plan["reviewers"][role],
            "calls": len(receipts[role]), "receipt_hashes": [receipts[role][case["case_id"]]["raw_response_sha256"]
                for case in packet["cases"]]} for role in ROLES},
        "agreement_resolution": "exact_agreement_else_uncertain_no_adjudicator_in_v17",
        "disagreements": disagreements, "cases": rows,
        "valid_output_count": 2 * len(rows), "invalid_output_count": invalid,
        "reference_class_coverage": coverage, "thresholds": THRESHOLDS,
        "reference_class_coverage_sufficient": sufficient,
        "target_evaluation_authorized": sufficient,
        "limitations": ["synthetic challenge only", "AI references are not human ground truth"]}
    return {**body, "reference_sha256": sha256_obj(body)}


def validate_target_gate(packet: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the exact packet-bound AI reference passes frozen gates."""
    body = {key: value for key, value in reference.items() if key != "reference_sha256"}
    if (reference.get("schema") != REFERENCE_SCHEMA
            or reference.get("reference_sha256") != sha256_obj(body)
            or reference.get("packet_sha256") != packet.get("packet_sha256")
            or reference.get("status") != "validity_and_class_coverage_passed"
            or reference.get("reference_class_coverage_sufficient") is not True
            or reference.get("invalid_output_count") != 0
            or reference.get("target_evaluation_authorized") is not True
            or reference.get("valid_output_count") != 2 * packet.get("case_count", -1)):
        raise ValueError("target evaluation blocked: v17 AI reference validity/class coverage gate has not passed")
    return {"schema": GATE_SCHEMA, "status": "passed", "packet_sha256": packet["packet_sha256"],
        "reference_sha256": reference["reference_sha256"], "target_calls_authorized": True,
        "target_calls_before_gate": 0, "thresholds": THRESHOLDS}


__all__ = ["SCHEMA", "REFERENCE_SCHEMA", "GATE_SCHEMA", "THRESHOLDS",
    "build_execution_plan", "validate_execution_plan", "collect_reference", "validate_target_gate"]
