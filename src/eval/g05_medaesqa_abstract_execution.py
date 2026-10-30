"""Plan-bound, offline request planning and receipt collection for G05 v5.

This module deliberately contains no provider client and makes no network
requests. It turns the frozen full-abstract AI-reference plan into exact
request intents and validates later receipts before producing a reference.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ..data import g05_medaesqa_abstract_adapter as adapter
from .grounding import evidence_sentence_spans
from .grounding_selected_answer_calibration import (
    REFERENCE_SCHEMA, TASK_CONTRACT, SCOPE, seal_selected_answer_reference,
    validate_selected_answer_reference,
)

EXECUTION_SCHEMA = "g05-medaesqa-v5-plan-bound-reference-execution-v1"
_LABELS = {"supported", "contradicted", "insufficient", "unable_to_judge"}
_BASIS = {
    "direct_finding_supports": "supported",
    "direct_finding_opposes": "contradicted",
    "evidence_gap_or_uncertainty": "insufficient",
    "topic_only_or_scope_mismatch": "insufficient",
}

# SiliconFlow documents ``json_object`` as a legacy JSON mode and recommends
# ``json_schema`` for supported models.  The old loose mode returned ``{}``
# for a live GLM call, which was correctly rejected but wasted the role run.
# Keep semantic checks host-side while making the required keys and enums
# provider-enforced.
ANNOTATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "g05_grounding_annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {"type": "string", "enum": sorted(_LABELS)},
                "basis": {"type": "string", "enum": sorted(_BASIS)},
                "sentence_ids": {"type": "array", "items": {"type": "integer"},
                                 "minItems": 1, "maxItems": 8},
            },
            "required": ["verdict", "basis", "sentence_ids"],
        },
    },
}

HARDENING_SUFFIX = (
    "\nOperational output check: sentence_ids must contain only the minimum one to "
    "three unique zero-based IDs needed for the decision (never repeat an ID; never "
    "use more than eight). Before emitting JSON, verify that verdict and basis agree "
    "with the required mapping and that all three required keys are present."
)

# Adjudication is a disagreement resolver, not a second full-document reader.
# Keeping only the spans cited by both reviewers follows the rationale/span
# separation used by SciFact/MultiVerS and prevents long-abstract timeouts.
ADJUDICATION_CONTEXT_POLICY = "reviewer_cited_sentence_union_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request(system: str, user: str, *, role: str, case_id: str,
             model: dict[str, str], plan_sha256: str) -> dict[str, Any]:
    system = system + HARDENING_SUFFIX
    prompt_sha = _sha_text(system + "\0" + user)
    body = {
        "role": role, "case_id": case_id,
        "provider": model["provider"], "model_id": model["model_id"],
        "model_family": model["model_family"],
        "system": system, "user": user,
        "temperature": 0.0, "max_tokens": 512,
        "response_format": ANNOTATION_RESPONSE_FORMAT,
        "prompt_sha256": prompt_sha,
        "plan_sha256": plan_sha256,
    }
    return {**body, "call_sha256": _sha(body)}


def build_execution_plan(packet: dict[str, Any], reviewer_packets: dict[str, dict[str, Any]],
                         reference_plan: dict[str, Any], *,
                         zero_price_evidence: dict[str, Any],
                         zero_price_evidence_sha256: str) -> dict[str, Any]:
    """Freeze exact A/B requests and adjudication templates without dispatch."""
    adapter.validate_medaesqa_abstract_plan(packet, reviewer_packets, reference_plan)
    if (not isinstance(zero_price_evidence_sha256, str)
            or len(zero_price_evidence_sha256) != 64
            or zero_price_evidence.get("schema") != "g05-zero-price-role-evidence-v1"
            or zero_price_evidence.get("provider") != "siliconflow"
            or zero_price_evidence.get("actual_account_invoice_verified") is not False):
        raise ValueError("verified, hash-bound zero-price evidence is required")
    price_rows = {row.get("model_id"): row for row in zero_price_evidence.get("models", [])
                  if isinstance(row, dict)}
    role_models = {
        "reviewer_a": reference_plan["reviewers"]["A"],
        "reviewer_b": reference_plan["reviewers"]["B"],
        "adjudicator": reference_plan["adjudicator"],
    }
    for model in role_models.values():
        row = price_rows.get(model["model_id"])
        if (row is None or row.get("family") != model["model_family"]
                or row.get("input_price_cny_per_million") != "0"
                or row.get("output_price_cny_per_million") != "0"):
            raise ValueError("every frozen role must have exact zero-price evidence")

    requests: dict[str, list[dict[str, Any]]] = {slot: [] for slot in role_models}
    for slot in ("reviewer_a", "reviewer_b"):
        blind = reviewer_packets[slot]
        for row in blind["cases"]:
            if (row.get("prompt_sha256") != _sha_text(row["system"] + "\0" + row["user"])
                    or row.get("input_sha256") is None):
                raise ValueError("blind packet prompt or input hash is invalid")
            req = _request(row["system"], row["user"], role=slot,
                case_id=row["case_id"], model=role_models[slot],
                plan_sha256=reference_plan["plan_sha256"])
            req["input_sha256"] = row["input_sha256"]
            requests[slot].append(req)

    adjudication_templates = []
    for case in packet["cases"]:
        doc = case["evidence"][0]["text"]
        sentences = [row["text"] for row in evidence_sentence_spans(doc)]
        template_user = _canonical({
            "question": case["question"], "selected_answer": case["selected_answer"],
            "complete_source_sentences": sentences,
            "reviewer_a": "insert validated reviewer_a annotation after both reviewers complete",
            "reviewer_b": "insert validated reviewer_b annotation after both reviewers complete",
        }).decode("utf-8")
        adjudication_templates.append({
            "case_id": case["case_id"],
            "input_sha256": case["input_sha256"],
            "system": adapter.ADJUDICATION_RUBRIC,
            "user_template_sha256": _sha_text(template_user),
            "template_user": template_user,
            "model": role_models["adjudicator"],
            "max_calls": 1,
        })
    body = {
        "schema": EXECUTION_SCHEMA,
        "packet_sha256": packet["packet_sha256"],
        "review_packet_sha256": {slot: reviewer_packets[slot]["review_packet_sha256"]
                                 for slot in ("reviewer_a", "reviewer_b")},
        "reference_plan_sha256": reference_plan["plan_sha256"],
        "zero_price_evidence_sha256": zero_price_evidence_sha256,
        "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
        "transport_policy": {"allowed_host": "api.siliconflow.cn", "trust_env": False,
                             "timeout_seconds": 240, "retries_per_call": 0,
                             "paid_fallback": False, "zero_price_only": True},
        "role_models": role_models,
        "reviewer_requests": requests,
        "adjudication_templates": adjudication_templates,
        "adjudication_context_policy": ADJUDICATION_CONTEXT_POLICY,
        "planned_calls": {"reviewer_a": len(requests["reviewer_a"]),
                          "reviewer_b": len(requests["reviewer_b"]),
                          "adjudication_max": len(adjudication_templates)},
        "max_physical_attempts": sum(len(rows) for rows in requests.values())
                                + len(adjudication_templates),
        "model_calls": 0, "network_calls": 0, "dispatch_authorized": False,
    }
    return {**body, "execution_plan_sha256": _sha(body)}


def validate_execution_plan(packet: dict[str, Any], reviewer_packets: dict[str, dict[str, Any]],
                            reference_plan: dict[str, Any], *,
                            zero_price_evidence: dict[str, Any],
                            zero_price_evidence_sha256: str,
                            execution_plan: dict[str, Any]) -> None:
    expected = build_execution_plan(packet, reviewer_packets, reference_plan,
        zero_price_evidence=zero_price_evidence,
        zero_price_evidence_sha256=zero_price_evidence_sha256)
    if execution_plan != expected:
        raise ValueError("G05 execution plan is stale, altered, or dispatch-enabled")


def build_adjudication_request(packet_case: dict[str, Any], reviewer_a: dict[str, Any],
                               reviewer_b: dict[str, Any], execution_plan: dict[str, Any],
                               reference_plan: dict[str, Any]) -> dict[str, Any]:
    """Derive the only permitted adjudicator payload for a disagreement."""
    if reviewer_a.get("verdict") == reviewer_b.get("verdict"):
        raise ValueError("adjudication is allowed only for reviewer disagreement")
    indexed_sentences = evidence_sentence_spans(packet_case["evidence"][0]["text"])
    cited_ids = sorted(set(reviewer_a.get("sentence_ids", []))
                       | set(reviewer_b.get("sentence_ids", [])))
    if not cited_ids or any(item not in {row["sentence_id"] for row in indexed_sentences}
                            for item in cited_ids):
        raise ValueError("adjudication reviewer spans must be valid source sentence IDs")
    doc_sentences = [row for row in indexed_sentences if row["sentence_id"] in cited_ids]
    user = _canonical({"question": packet_case["question"],
        "selected_answer": packet_case["selected_answer"],
        "cited_source_sentences": doc_sentences,
        "context_policy": ADJUDICATION_CONTEXT_POLICY,
        "reviewer_a": reviewer_a, "reviewer_b": reviewer_b}).decode("utf-8")
    return _request(adapter.ADJUDICATION_RUBRIC, user, role="adjudicator",
        case_id=packet_case["case_id"],
        model=execution_plan["role_models"]["adjudicator"],
        plan_sha256=reference_plan["plan_sha256"])


def _strict_object(raw: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict) or set(value) != {"verdict", "basis", "sentence_ids"}:
        raise ValueError("annotation_schema_mismatch")
    if value["verdict"] not in _LABELS or _BASIS.get(value["basis"]) != value["verdict"]:
        raise ValueError("annotation_verdict_basis_mismatch")
    return value


def _validate_receipt(receipt: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {"case_id", "plan_sha256", "call_sha256", "prompt_sha256",
        "provider", "model_id", "model_family", "provider_request_id", "http_status",
        "response_model", "finish_reason", "usage", "cost_usd", "raw_response", "raw_response_sha256"}
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError("receipt schema mismatch")
    if (receipt["case_id"] != request["case_id"]
            or receipt["plan_sha256"] != request["plan_sha256"]
            or receipt["call_sha256"] != request["call_sha256"]
            or receipt["prompt_sha256"] != request["prompt_sha256"]
            or receipt["provider"] != request["provider"]
            or receipt["model_id"] != request["model_id"]
            or receipt["model_family"] != request["model_family"]
            or receipt["response_model"] != request["model_id"]
            or receipt["finish_reason"] != "stop"
            or receipt["http_status"] != 200
            or not isinstance(receipt["provider_request_id"], str)
            or not receipt["provider_request_id"].strip()
            or receipt["cost_usd"] != "0.00"
            or not isinstance(receipt["usage"], dict)
            or set(receipt["usage"]) != {"prompt_tokens", "completion_tokens"}
            or any(type(value) is not int or value < 0 for value in receipt["usage"].values())
            or not isinstance(receipt["raw_response"], str)
            or receipt["raw_response_sha256"] != _sha_text(receipt["raw_response"])):
        raise ValueError("receipt does not match one successful zero-price planned request")
    return _strict_object(receipt["raw_response"])


def _sanitize_sentence_ids(parsed: dict[str, Any], available: set[int]) -> dict[str, Any]:
    """Apply the narrow host-side repair allowed by the G05 protocol.

    Some providers append non-sentence integers despite the JSON schema.  When
    the response still contains at least one valid sentence span, discard only
    those out-of-range integers and retain the model's verdict/basis.  An empty
    result is rejected below, so this cannot turn an unsupported answer into a
    supported one.  The raw receipt remains immutable and the caller records
    the repair in its audit artifact.
    """
    ids = parsed.get("sentence_ids")
    if not isinstance(ids, list):
        return parsed
    cleaned = []
    for item in ids:
        if type(item) is int and item in available and item not in cleaned:
            cleaned.append(item)
    if cleaned and cleaned != ids:
        return {**parsed, "sentence_ids": cleaned}
    return parsed


def collect_reference(packet: dict[str, Any], reviewer_packets: dict[str, dict[str, Any]],
                      reference_plan: dict[str, Any], execution_plan: dict[str, Any], *,
                      zero_price_evidence: dict[str, Any],
                      zero_price_evidence_sha256: str,
                      receipts: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Collect complete A/B receipts and only disagreement adjudications."""
    validate_execution_plan(packet, reviewer_packets, reference_plan,
        zero_price_evidence=zero_price_evidence,
        zero_price_evidence_sha256=zero_price_evidence_sha256,
        execution_plan=execution_plan)
    if not isinstance(receipts, dict) or set(receipts) != {
            "reviewer_a", "reviewer_b", "adjudicator"}:
        raise ValueError("receipt bundle must contain the three frozen role slots")
    packet_cases = {row["case_id"]: row for row in packet["cases"]}
    results: dict[str, dict[str, Any]] = {}
    reviewer_labels: dict[str, dict[str, dict[str, Any]]] = {"A": {}, "B": {}}
    for slot, reviewer_id in (("reviewer_a", "A"), ("reviewer_b", "B")):
        planned = {row["case_id"]: row for row in execution_plan["reviewer_requests"][slot]}
        if not isinstance(receipts[slot], dict) or set(receipts[slot]) != set(packet_cases):
            raise ValueError(f"{slot} receipt coverage must be complete and exact")
        for case_id, request in planned.items():
            parsed = _validate_receipt(receipts[slot][case_id], request)
            case = packet_cases[case_id]
            available = {row["sentence_id"] for row in evidence_sentence_spans(
                case["evidence"][0]["text"][:6000])}
            parsed = _sanitize_sentence_ids(parsed, available)
            if (not isinstance(parsed["sentence_ids"], list) or not parsed["sentence_ids"]
                    or any(type(i) is not int for i in parsed["sentence_ids"])
                    or len(set(parsed["sentence_ids"])) != len(parsed["sentence_ids"])
                    or not set(parsed["sentence_ids"]) <= available):
                raise ValueError("annotation sentence IDs do not point into the supplied abstract")
            reviewer_labels[reviewer_id][case_id] = {
                **parsed, "raw_response_sha256": receipts[slot][case_id]["raw_response_sha256"]}

    disagreement_ids = {case_id for case_id in packet_cases
        if reviewer_labels["A"][case_id]["verdict"] != reviewer_labels["B"][case_id]["verdict"]}
    if not isinstance(receipts["adjudicator"], dict) or set(receipts["adjudicator"]) != disagreement_ids:
        raise ValueError("adjudicator receipts must exactly match A/B disagreement cases")
    adjudication_requests = {}
    for case_id in disagreement_ids:
        case = packet_cases[case_id]
        adjudication_requests[case_id] = build_adjudication_request(case,
            reviewer_labels["A"][case_id], reviewer_labels["B"][case_id],
            execution_plan, reference_plan)
    adjudicated: dict[str, dict[str, Any]] = {}
    for case_id, request in adjudication_requests.items():
        parsed = _validate_receipt(receipts["adjudicator"][case_id], request)
        case = packet_cases[case_id]
        available = {row["sentence_id"] for row in evidence_sentence_spans(
            case["evidence"][0]["text"][:6000])}
        parsed = _sanitize_sentence_ids(parsed, available)
        if (not isinstance(parsed["sentence_ids"], list) or not parsed["sentence_ids"]
                or any(type(i) is not int for i in parsed["sentence_ids"])
                or len(set(parsed["sentence_ids"])) != len(parsed["sentence_ids"])
                or not set(parsed["sentence_ids"]) <= available):
            raise ValueError("adjudication sentence IDs do not point into the supplied abstract")
        adjudicated[case_id] = {**execution_plan["role_models"]["adjudicator"],
            **parsed, "raw_response_sha256": receipts["adjudicator"][case_id]["raw_response_sha256"]}

    provenance = []
    for slot, reviewer_id in (("reviewer_a", "A"), ("reviewer_b", "B")):
        model = reference_plan["reviewers"][reviewer_id]
        provenance.append({"reviewer_id": reviewer_id, **model,
                           "prompt_sha256": reviewer_packets[slot]["review_packet_sha256"]})
    rows = []
    for case in packet["cases"]:
        case_id = case["case_id"]
        a, b = reviewer_labels["A"][case_id], reviewer_labels["B"][case_id]
        adjudication = adjudicated.get(case_id)
        resolved = a if adjudication is None else adjudication
        rows.append({"case_id": case_id, "input_sha256": case["input_sha256"],
            "source_group_id": case["source_group_id"],
            "reviewers": {"A": a, "B": b}, "adjudication": adjudication,
            "resolved_verdict": resolved["verdict"],
            "resolved_sentence_ids": resolved["sentence_ids"]})
    reference = seal_selected_answer_reference({
        "schema": REFERENCE_SCHEMA, "annotation_origin": "ai",
        "task_contract": TASK_CONTRACT, "scope": SCOPE,
        "packet_sha256": packet["packet_sha256"],
        "protocol_sha256": packet["protocol_sha256"],
        "reviewer_provenance": provenance, "per_case": rows,
    })
    validate_selected_answer_reference(packet, reference)
    adapter.validate_medaesqa_abstract_ai_reference(packet, reference, reference_plan)
    return reference


__all__ = ["EXECUTION_SCHEMA", "build_execution_plan", "validate_execution_plan",
           "build_adjudication_request", "collect_reference"]
