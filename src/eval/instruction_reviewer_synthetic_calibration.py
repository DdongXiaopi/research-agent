"""Diagnostic-only calibration of the instruction reviewer on synthetic G06 cohorts.

This protocol is intentionally separate from natural-distribution scorer
calibration and production candidate acceptance.  It cannot authorize prompt
promotion or claim natural-distribution validity.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from . import ai_jobs
from . import instruction_reviewer_cohort_reference as reference
from . import instruction_reviewer_sharded_freeze as freeze
from .instruction_reviewer_cohort_v4 import SCHEMA as COHORT_V4_SCHEMA
from .instruction_reviewer_cohort_v5 import SCHEMA as COHORT_V5_SCHEMA
from .instruction_reviewer_cohort_v6 import SCHEMA as COHORT_V6_SCHEMA
from .instruction_reviewer_cohort_v7 import SCHEMA as COHORT_V7_SCHEMA
from .instruction_reviewer_cohort_v8 import SCHEMA as COHORT_V8_SCHEMA
from .instruction_reviewer_cohort_v9 import SCHEMA as COHORT_V9_SCHEMA
from .instruction_reviewer_cohort_v10 import SCHEMA as COHORT_V10_SCHEMA
from . import instruction_reviewer_cohort_v11 as cohort_v11
from .instruction_reviewer_cohort_v11 import SCHEMA as COHORT_V11_SCHEMA
from .instruction_reviewer_pair_gate import (
    family_registry_sha256, load_g06_config, model_family,
)
from .instruction_reviewer_rubric_v2 import RUBRIC_V2
from .instruction_reviewer_rubric_v3 import RUBRIC_V3
from .instruction_reviewer_rubric_v4 import RUBRIC_V4
from .instruction_reviewer_rubric_v5 import RUBRIC_V5
from .instruction_reviewer_rubric_v6 import RUBRIC_V6
from .instruction_reviewer_rubric_v7 import RUBRIC_V7
from . import instruction_reviewer_rubric_v8 as rubric_v8
from .instruction_reviewer_rubric_v8 import RUBRIC_V8, protocol_metadata as rubric_v8_protocol_metadata
from ..evolution.instruction_review import (
    parse_prompt_grounding_review, prompt_grounding_messages,
    prompt_grounding_protocol, prompt_grounding_response_format,
    parse_prompt_grounding_review_v8, prompt_grounding_messages_v8,
    prompt_grounding_protocol_v8, prepare_prompt_grounding_v8,
    PROMPT_GROUNDING_MAX_TOKENS,
    PROMPT_GROUNDING_V8_MAX_TOKENS,
)
from ..evolution.task_contracts import CLINIFACT_CONTRACT
from ..runtime.provider import published_price_for_endpoint
from ..util import repo_root, sha256_file, sha256_obj


PLAN_SCHEMA = "g06-synthetic-instruction-reviewer-calibration-plan-v1"
OUTPUT_SCHEMA = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v1"
PLAN_SCHEMA_V3 = "g06-synthetic-instruction-reviewer-calibration-plan-v2"
OUTPUT_SCHEMA_V3 = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v2"
PLAN_SCHEMA_V4 = "g06-synthetic-instruction-reviewer-calibration-plan-v3"
OUTPUT_SCHEMA_V4 = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v3"
PLAN_SCHEMA_V5 = "g06-synthetic-instruction-reviewer-calibration-plan-v4"
OUTPUT_SCHEMA_V5 = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v4"
PLAN_SCHEMA_V6 = "g06-synthetic-instruction-reviewer-calibration-plan-v5"
OUTPUT_SCHEMA_V6 = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v5"
PLAN_SCHEMA_V7 = "g06-synthetic-instruction-reviewer-calibration-plan-v6"
OUTPUT_SCHEMA_V7 = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v6"
PLAN_SCHEMA_V8 = "g06-synthetic-instruction-reviewer-calibration-plan-v7"
OUTPUT_SCHEMA_V8 = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v7"
PLAN_SCHEMA_V9 = "g06-synthetic-instruction-reviewer-calibration-plan-v8"
OUTPUT_SCHEMA_V9 = "g06-synthetic-instruction-reviewer-diagnostic-calibration-v8"
MIN_CASES = 30
MIN_APPROVAL_REFERENCE = 3
MIN_REJECTION_REFERENCE = 3
MAX_TOKENS = PROMPT_GROUNDING_MAX_TOKENS
V11_MAX_TOKENS = PROMPT_GROUNDING_V8_MAX_TOKENS
MAX_COST_USD = 0.05
MAX_INPUT_CHARS = 16000

_ASSERTION = {"approve": "claim_free", "reject": "contains_claims"}
_CONTRACT = {"approve": "compatible", "reject": "incompatible"}


def _rubric_sha256(rubric_version: str) -> str:
    """Use each rubric protocol's canonical text-byte hashing convention."""
    if rubric_version == "v8":
        return rubric_v8_protocol_metadata()["rubric_sha256"]
    rubric = {"v7": RUBRIC_V7, "v6": RUBRIC_V6, "v5": RUBRIC_V5,
              "v4": RUBRIC_V4, "v3": RUBRIC_V3}.get(rubric_version, RUBRIC_V2)
    return sha256_obj(rubric)


def _protocol() -> dict:
    system, user = prompt_grounding_messages("<<AUTHORED_STRATEGY>>", CLINIFACT_CONTRACT)
    protocol = prompt_grounding_protocol(CLINIFACT_CONTRACT)
    return {"messages_protocol": protocol, "system_template": system,
        "user_template": user, "parser": "parse_prompt_grounding_review",
        "parser_candidate_evidence": "authored_strategy_only"}


def _protocol_v11(packet: dict) -> dict:
    """Freeze the v11 dynamic-schema transport and exact target parser identity."""
    system, user_template = prompt_grounding_messages_v8(
        "<<AUTHORED_STRATEGY>>", CLINIFACT_CONTRACT)
    return {
        "messages_protocol": packet["target_protocol"],
        "messages_protocol_sha256": sha256_obj(packet["target_protocol"]),
        "reference_protocol": packet["reference_protocol"],
        "reference_protocol_sha256": sha256_obj(packet["reference_protocol"]),
        "system_template": system,
        "user_template": user_template,
        "parser": "parse_prompt_grounding_review_v8",
        "parser_candidate_evidence": "authored_strategy_only",
        "transport": "json_object_per_case_dynamic_v8_schema_host_validated",
        "response_schema_binding": "case-specific-sha256-in-inputs-and-plan",
        "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
        "reference_response_format_template_sha256":
            packet["reference_protocol"]["response_format_template_sha256"],
        "dynamic_schema_manifest_sha256": sha256_obj([
            {"case_id": row["case_id"],
             "response_format_sha256": row["response_format_sha256"],
             "candidate_spans_sha256": row["candidate_spans_sha256"],
             "contract_spans_sha256": row["contract_spans_sha256"]}
            for row in packet["reference_packet_manifest"]]),
        "fail_closed_host_validation": True,
        "promotion_authorized": False,
    }


def _target_messages(candidate_strategy: str) -> tuple[str, str]:
    """Build the only payload delivered to the target scorer.

    Reference IDs, labels, reviewer evidence, and construction metadata are
    retained for host-side diagnostics and never enter this prompt.
    """
    return prompt_grounding_messages(candidate_strategy, CLINIFACT_CONTRACT)


def _parse_v11_target_output(raw: str, item: dict, packet: dict) -> tuple[dict, dict]:
    """Rebuild the exact per-case v8 schema before parsing a target response."""
    if packet.get("schema") != COHORT_V11_SCHEMA:
        raise ValueError("v11 target parser requires a v11 packet")
    current_protocol = prompt_grounding_protocol_v8(CLINIFACT_CONTRACT)
    if (packet.get("target_protocol") != current_protocol
            or packet.get("target_protocol", {}).get("version")
                != "instruction-review-v8-claim-extraction"):
        raise ValueError("v11 target protocol identity changed before parsing")
    request = prepare_prompt_grounding_v8(item["candidate_strategy"], CLINIFACT_CONTRACT)
    schema_sha256 = sha256_obj(request["response_format"])
    if (item.get("target_response_format_sha256") != schema_sha256
            or item.get("target_candidate_spans_sha256")
                != sha256_obj(request["candidate_spans"])
            or item.get("target_contract_spans_sha256")
                != sha256_obj(request["contract_spans"])):
        raise ValueError("v11 target per-case schema or span identity drifted")
    parsed = parse_prompt_grounding_review_v8(
        raw, item["candidate_strategy"], CLINIFACT_CONTRACT)
    provenance = {
        "target_protocol_version": current_protocol["version"],
        "target_protocol_sha256": sha256_obj(current_protocol),
        "response_format_sha256": schema_sha256,
        "candidate_spans_sha256": item["target_candidate_spans_sha256"],
        "contract_spans_sha256": item["target_contract_spans_sha256"],
        "parser": "parse_prompt_grounding_review_v8",
        "schema_valid": parsed.get("schema_valid") is True,
    }
    return parsed, provenance


def _assert_v4(packet: dict) -> None:
    reference.validate_reference_packet(packet)
    if (packet.get("schema") != COHORT_V4_SCHEMA
            or packet.get("natural_distribution") is not False
            or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                                           "text": CLINIFACT_CONTRACT.text,
                                           "sha256": CLINIFACT_CONTRACT.sha256}):
        raise ValueError("synthetic scorer accepts only the frozen G06 v4 Clinifact packet")
    if not isinstance(packet.get("cases"), list) or len(packet["cases"]) < MIN_CASES:
        raise ValueError("synthetic reference cohort is below the frozen minimum case count")


def _assert_packet(packet: dict) -> str:
    """Validate an eligible packet and return its rubric generation."""
    if packet.get("schema") == COHORT_V11_SCHEMA:
        from . import instruction_reviewer_cohort_v11 as cohort_v11
        cohort_v11.validate_packet(packet)
        current_target = prompt_grounding_protocol_v8(CLINIFACT_CONTRACT)
        current_reference = {**rubric_v8_protocol_metadata(),
                             "source_sha256": sha256_file(Path(rubric_v8.__file__))}
        if (packet.get("natural_distribution") is not False
                or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}
                or packet.get("case_count") != 40
                or len(packet.get("cases", [])) != 40
                or packet.get("target_protocol") != current_target
                or packet.get("reference_protocol") != current_reference):
            raise ValueError("synthetic scorer accepts only the exact v11 packet bound to current target/reference v8 protocols")
        return "v8"
    if packet.get("schema") == COHORT_V10_SCHEMA:
        reference.validate_reference_packet(packet)
        current_protocol = _protocol()["messages_protocol"]
        target_protocol = packet.get("target_protocol")
        if (packet.get("natural_distribution") is not False
                or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}
                or not isinstance(packet.get("cases"), list)
                or len(packet["cases"]) != 40
                or target_protocol != current_protocol
                or target_protocol.get("version") != "instruction-review-v7-span-atomic"):
            raise ValueError("synthetic scorer accepts only the exact G06 v10 packet bound to the current target protocol")
        return "v7"
    if packet.get("schema") == COHORT_V9_SCHEMA:
        reference.validate_reference_packet(packet)
        if (packet.get("natural_distribution") is not False
                or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}
                or not isinstance(packet.get("cases"), list)
                or len(packet["cases"]) != 40):
            raise ValueError("synthetic scorer accepts only the exact G06 v9 enum-keyed rubric Clinifact packet")
        return "v7"
    if packet.get("schema") == COHORT_V8_SCHEMA:
        reference.validate_reference_packet(packet)
        if (packet.get("natural_distribution") is not False
                or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}
                or not isinstance(packet.get("cases"), list)
                or len(packet["cases"]) != 40):
            raise ValueError("synthetic scorer accepts only the exact G06 v8 exhaustive-rubric Clinifact packet")
        return "v6"
    if packet.get("schema") == COHORT_V7_SCHEMA:
        reference.validate_reference_packet(packet)
        if (packet.get("natural_distribution") is not False
                or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}
                or not isinstance(packet.get("cases"), list)
                or len(packet["cases"]) != 40):
            raise ValueError("synthetic scorer accepts only the exact G06 v7 Clinifact packet")
        return "v5"
    if packet.get("schema") == COHORT_V6_SCHEMA:
        reference.validate_reference_packet(packet)
        if (packet.get("natural_distribution") is not False
                or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}
                or not isinstance(packet.get("cases"), list)
                or len(packet["cases"]) != 40):
            raise ValueError("synthetic scorer accepts only the exact G06 v6 Clinifact packet")
        return "v4"
    if packet.get("schema") == COHORT_V5_SCHEMA:
        reference.validate_reference_packet(packet)
        if (packet.get("natural_distribution") is not False
                or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}
                or not isinstance(packet.get("cases"), list)
                or len(packet["cases"]) < MIN_CASES):
            raise ValueError("synthetic scorer accepts only a frozen G06 v5 Clinifact packet")
        return "v3"
    _assert_v4(packet)
    return "v2"


def _validate_frozen(packet: dict, frozen: dict, schedule_a: dict, state_a,
                     aggregate_a: dict, schedule_b: dict, state_b,
                     aggregate_b: dict) -> dict:
    rubric_version = _assert_packet(packet)
    verified = freeze.verify_synthetic_diagnostic_reference(packet, frozen,
        schedule_a, state_a, aggregate_a, schedule_b, state_b, aggregate_b)
    bindings = freeze._protocol_bindings(packet)
    expected_frozen_schema = (getattr(freeze, "FROZEN_SCHEMA_V9", freeze.FROZEN_SCHEMA)
        if packet.get("schema") == COHORT_V11_SCHEMA else
        freeze.FROZEN_SCHEMA_V8 if packet.get("schema") == COHORT_V10_SCHEMA else
        {"v8": getattr(freeze, "FROZEN_SCHEMA_V9", freeze.FROZEN_SCHEMA),
                               "v7": freeze.FROZEN_SCHEMA_V7,
                               "v6": freeze.FROZEN_SCHEMA_V6,
                               "v5": freeze.FROZEN_SCHEMA_V5,
                               "v4": freeze.FROZEN_SCHEMA_V4,
                               "v3": freeze.FROZEN_SCHEMA_V3}.get(
                                   rubric_version, freeze.FROZEN_SCHEMA))
    expected_rubric_sha256 = _rubric_sha256(rubric_version)
    if (verified.get("schema") != expected_frozen_schema
            or verified.get("status") != "synthetic_diagnostic_frozen"
            or verified.get("synthetic_only") is not True
            or verified.get("natural_distribution") is not False
            or verified.get("promotion_authorized") is not False
            or verified.get("annotation_origin") != "ai"
            or verified.get("packet_sha256") != packet.get("packet_sha256")
            or any(verified.get(key) != value for key, value in bindings.items())
            or verified.get("rubric_version") != rubric_version
            or verified.get("rubric_sha256") != expected_rubric_sha256
            or verified.get("family_registry_sha256") != family_registry_sha256()):
        raise ValueError("frozen reference is old, natural-schema, or has changed provenance")
    report = verified.get("pair_gate_report")
    if (not isinstance(report, dict)
            or verified.get("pair_gate_report_sha256") != sha256_obj(report)
            or report.get("schema") != "g06-synthetic-reference-pair-verification-v1"
            or report.get("status") != "verified_zero_call_pair"
            or report.get("packet_sha256") != packet["packet_sha256"]
            or any(report.get(key) != value for key, value in bindings.items())
            or report.get("rubric_sha256") != expected_rubric_sha256
            or report.get("family_registry_sha256") != family_registry_sha256()
            or report.get("natural_distribution") is not False
            or report.get("promotion_authorized") is not False):
        raise ValueError("frozen reference lacks verifiable A/B pair-gate evidence")
    for slot in ("reviewer_a", "reviewer_b"):
        reviewer = verified.get(slot, {})
        gate_reviewer = report.get(slot, {})
        receipt_hashes = [row.get("receipt_file_sha256")
                          for row in reviewer.get("case_receipts", [])]
        if (gate_reviewer.get("model_id") != reviewer.get("model_id")
                or gate_reviewer.get("model_family") != reviewer.get("model_family")
                or gate_reviewer.get("schedule_sha256") != reviewer.get("schedule_sha256")
                or gate_reviewer.get("aggregate_sha256") != reviewer.get("aggregate_sha256")
                or gate_reviewer.get("case_receipt_hashes") != receipt_hashes):
            raise ValueError("frozen pair-gate report is not bound to the reviewer receipts")
    if (report.get("checks") != {"distinct_model_ids": True,
            "distinct_model_families": True, "identical_blind_payloads": True,
            "complete_ordered_case_coverage": True, "unique_per_case_receipts": True}):
        raise ValueError("frozen pair-gate report has incomplete verification checks")
    if verified.get("reference_class_coverage_sufficient") is not True:
        raise ValueError("frozen reference coverage is insufficient; refusing a zero-call plan")
    rows = verified.get("cases")
    if not isinstance(rows, list) or len(rows) != len(packet["cases"]):
        raise ValueError("frozen consensus does not cover every packet case")
    expected_ids = [case["case_id"] for case in packet["cases"]]
    if [row.get("case_id") for row in rows] != expected_ids:
        raise ValueError("frozen reference case order differs from the packet")
    coverage = {}
    verdict_sets = {"assertion": {"claim_free", "contains_claims", "uncertain"},
                    "contract": {"compatible", "incompatible", "uncertain"}}
    for dim, approved, rejected in (("assertion", "claim_free", "contains_claims"),
                                    ("contract", "compatible", "incompatible")):
        labels = [row.get("verdicts", {}).get(dim) for row in rows]
        if any(value not in verdict_sets[dim] for value in labels):
            raise ValueError(f"frozen reference has malformed {dim} labels")
        counts = {"approval_reference": labels.count(approved),
                  "rejection_reference": labels.count(rejected),
                  "uncertain_reference": labels.count("uncertain")}
        coverage[dim] = counts
        if (counts["approval_reference"] < MIN_APPROVAL_REFERENCE
                or counts["rejection_reference"] < MIN_REJECTION_REFERENCE):
            raise ValueError(f"frozen {dim} coverage is below approval/rejection thresholds")
    if verified.get("reference_class_coverage") != coverage:
        raise ValueError("frozen reference coverage counts do not match its consensus labels")
    for slot in ("reviewer_a", "reviewer_b"):
        reviewer = verified.get(slot, {})
        receipts = reviewer.get("case_receipts", [])
        evidence = reviewer.get("case_evidence", [])
        if (len(receipts) != len(expected_ids) or len(evidence) != len(expected_ids)
                or [row.get("case_id") for row in evidence] != expected_ids
                or any(ev.get("receipt_file_sha256") != receipt.get("receipt_file_sha256")
                       for ev, receipt in zip(evidence, receipts))):
            raise ValueError(f"{slot} raw verdict evidence is incomplete or not receipt-bound")
    return verified


def _zero_price_access(model_id: str, workspace: Path, cfg: dict) -> dict:
    model = cfg.get("models", {}).get("task", {})
    endpoint = model.get("base_url")
    rates = published_price_for_endpoint(model_id, endpoint)
    env_name = model.get("api_key_env")
    if (model.get("provider") != "http" or model.get("model_id") != model_id
            or endpoint != "https://api.siliconflow.cn/v1"
            or not isinstance(env_name, str) or not env_name.strip()
            or not os.environ.get(env_name)
            or not rates or any(value != 0 for value in rates.values())):
        raise ValueError("target config lacks current zero-price CN credentials or tariff")
    evidence = reference._free_access_evidence(model_id, workspace)
    if (evidence.get("model_id") != model_id
            or evidence.get("base_url") != endpoint
            or evidence.get("published_tariff_usd") != 0.0
            or evidence.get("paid_fallback") is not False
            or evidence.get("account_access_confirmed") is not True
            or evidence.get("execution_permitted") is not True
            or evidence.get("http_status") != 200
            or evidence.get("usage_complete") is not True):
        raise ValueError("target model lacks a verified zero-price account-access permission")
    return {**evidence, "api_key_env": env_name, "credential_present": True}


def _inputs(packet: dict, frozen: dict, protocol: dict) -> dict:
    left = {row["case_id"]: row for row in frozen["reviewer_a"]["case_evidence"]}
    right = {row["case_id"]: row for row in frozen["reviewer_b"]["case_evidence"]}
    consensus = {row["case_id"]: row for row in frozen["cases"]}
    reference_manifest = {row["case_id"]: row
                         for row in packet.get("reference_packet_manifest", [])}
    cases = []
    for case in packet["cases"]:
        a, b = left[case["case_id"]], right[case["case_id"]]
        case_input = {"case_id": case["case_id"],
            "candidate_strategy": case["strategy_text"],
            "reference": copy.deepcopy(consensus[case["case_id"]]["verdicts"]),
            "reviewer_evidence": {
                "reviewer_a": {"verdicts": copy.deepcopy(a["verdicts"]),
                    "normalization_issues": copy.deepcopy(a["normalization_issues"]),
                    "receipt_file_sha256": a["receipt_file_sha256"]},
                "reviewer_b": {"verdicts": copy.deepcopy(b["verdicts"]),
                    "normalization_issues": copy.deepcopy(b["normalization_issues"]),
                    "receipt_file_sha256": b["receipt_file_sha256"]}},
            "reviewer_disagreement": {dim: a["verdicts"][dim] != b["verdicts"][dim]
                for dim in ("assertion", "contract")}}
        if packet.get("schema") == COHORT_V11_SCHEMA:
            reference_packet = cohort_v11.reference_payload(case)["reference_spans"]
            target_request = prepare_prompt_grounding_v8(
                case["strategy_text"], CLINIFACT_CONTRACT)
            target_schema_sha256 = sha256_obj(target_request["response_format"])
            manifest_row = reference_manifest.get(case["case_id"])
            if (target_schema_sha256 != reference_packet["response_format_sha256"]
                    or not isinstance(manifest_row, dict)
                    or target_schema_sha256 != manifest_row.get("response_format_sha256")
                    or sha256_obj(target_request["candidate_spans"])
                        != manifest_row.get("candidate_spans_sha256")
                    or sha256_obj(target_request["contract_spans"])
                        != manifest_row.get("contract_spans_sha256")):
                raise ValueError("v11 per-case target schema differs from the frozen reference schema")
            case_input.update({
                "target_response_format_sha256": target_schema_sha256,
                "target_candidate_spans_sha256": sha256_obj(target_request["candidate_spans"]),
                "target_contract_spans_sha256": sha256_obj(target_request["contract_spans"]),
            })
        cases.append(case_input)
    rubric_version = frozen["rubric_version"]
    body = {"schema": ("g06-synthetic-instruction-reviewer-calibration-input-v8"
                        if packet.get("schema") == COHORT_V11_SCHEMA else
                        "g06-synthetic-instruction-reviewer-calibration-input-v7"
                        if packet.get("schema") == COHORT_V10_SCHEMA else
                        "g06-synthetic-instruction-reviewer-calibration-input-v6"
                        if rubric_version == "v7" else
                        "g06-synthetic-instruction-reviewer-calibration-input-v5"
                        if rubric_version == "v6" else
                        "g06-synthetic-instruction-reviewer-calibration-input-v4"
                        if rubric_version == "v5" else
                        "g06-synthetic-instruction-reviewer-calibration-input-v3"
                        if rubric_version == "v4" else
                        "g06-synthetic-instruction-reviewer-calibration-input-v2"
                        if rubric_version == "v3" else
                        "g06-synthetic-instruction-reviewer-calibration-input-v1"),
        "packet_sha256": packet["packet_sha256"],
        "frozen_reference_sha256": frozen["frozen_reference_sha256"],
        "pair_gate_report_sha256": frozen["pair_gate_report_sha256"],
        "rubric_version": frozen["rubric_version"],
        "rubric_sha256": frozen["rubric_sha256"],
        "protocol": protocol, "protocol_sha256": sha256_obj(protocol),
        "thresholds": {"min_cases": MIN_CASES,
            "min_approval_reference": MIN_APPROVAL_REFERENCE,
            "min_rejection_reference": MIN_REJECTION_REFERENCE},
        "contract": copy.deepcopy(packet["contract"]), "cases": cases}
    if rubric_version in {"v3", "v4", "v5", "v6", "v7", "v8"}:
        body.update({key: value for key, value in freeze._protocol_bindings(packet).items()
                     if key != "rubric_version" and key != "rubric_sha256"})
    if packet.get("schema") == COHORT_V10_SCHEMA:
        body["target_protocol_version"] = packet["target_protocol"]["version"]
        body["target_protocol_sha256"] = sha256_obj(packet["target_protocol"])
    if packet.get("schema") == COHORT_V11_SCHEMA:
        body["target_protocol_version"] = packet["target_protocol"]["version"]
        body["target_protocol_sha256"] = sha256_obj(packet["target_protocol"])
        body["reference_protocol_hash"] = packet["reference_protocol"]["protocol_hash"]
        body["target_transport"] = protocol["transport"]
        body["dynamic_schema_manifest_sha256"] = protocol["dynamic_schema_manifest_sha256"]
    return body


def _build_plan(packet, frozen, schedule_a, state_a, aggregate_a,
                schedule_b, state_b, aggregate_b, target_model_id: str,
                workspace: Path) -> tuple[dict, dict, dict, dict]:
    verified = _validate_frozen(packet, frozen, schedule_a, state_a, aggregate_a,
        schedule_b, state_b, aggregate_b)
    family = model_family(target_model_id)
    families = {verified["reviewer_a"]["model_family"],
                verified["reviewer_b"]["model_family"]}
    if family in families:
        raise ValueError("target scorer family must differ from both frozen reference families")
    cfg, config_provenance = load_g06_config(target_model_id, workspace)
    access = _zero_price_access(target_model_id, workspace, cfg)
    rubric_version = verified["rubric_version"]
    v11 = packet.get("schema") == COHORT_V11_SCHEMA
    protocol = _protocol_v11(packet) if v11 else _protocol()
    inputs = _inputs(packet, verified, protocol)
    job_protocol_metadata = ({
        "protocol_version": packet["target_protocol"]["version"],
        "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
        "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "response_mode": protocol["transport"],
        "parser": "parse_prompt_grounding_review_v8",
        "dynamic_schema_manifest_sha256": protocol["dynamic_schema_manifest_sha256"],
        "host_validation": "per_case_exact_v8_schema_and_source_quote_validation",
        "fail_closed": True,
    } if v11 else None)
    job_plan = ai_jobs.build_plan("instruction_judge", inputs, cfg,
        max_calls=len(packet["cases"]), tokens=V11_MAX_TOKENS if v11 else MAX_TOKENS,
        max_input_chars=MAX_INPUT_CHARS, max_cost_usd=MAX_COST_USD,
        response_format=({"type": "json_object"} if v11 else
                         prompt_grounding_response_format(CLINIFACT_CONTRACT)),
        protocol_metadata=job_protocol_metadata)
    source_artifacts = {slot: {
        "model_id": verified[slot]["model_id"],
        "model_family": verified[slot]["model_family"],
        "schedule_sha256": verified[slot]["schedule_sha256"],
        "aggregate_sha256": verified[slot]["aggregate_sha256"],
        "receipt_hashes": [row["receipt_file_sha256"]
                           for row in verified[slot]["case_receipts"]]}
        for slot in ("reviewer_a", "reviewer_b")}
    body = {"schema": (PLAN_SCHEMA_V9 if v11 else
        PLAN_SCHEMA_V8 if packet.get("schema") == COHORT_V10_SCHEMA else
        {"v7": PLAN_SCHEMA_V7, "v6": PLAN_SCHEMA_V6, "v5": PLAN_SCHEMA_V5, "v4": PLAN_SCHEMA_V4,
                        "v3": PLAN_SCHEMA_V3}.get(
            rubric_version, PLAN_SCHEMA)),
        "scope": "synthetic_diagnostic_only",
        "packet_sha256": packet["packet_sha256"],
        "frozen_reference_sha256": verified["frozen_reference_sha256"],
        "pair_gate_report_sha256": verified["pair_gate_report_sha256"],
        "family_registry_sha256": family_registry_sha256(),
        "source_artifacts": source_artifacts,
        "rubric_version": rubric_version,
        "rubric_sha256": _rubric_sha256(rubric_version),
        "protocol": protocol, "protocol_sha256": sha256_obj(protocol),
        "thresholds": inputs["thresholds"],
        "config_provenance": config_provenance,
        "config_sha256": sha256_obj(cfg), "target_model_id": target_model_id,
        "target_model_family": family, "zero_price_access": access,
        "paid_fallback": False,
        "max_calls": len(packet["cases"]),
        "max_physical_attempts": len(packet["cases"]),
        "max_cost_usd": MAX_COST_USD, "max_tokens": MAX_TOKENS,
        "planned_calls": 0, "ai_job_plan": job_plan,
        "inputs_sha256": sha256_obj(inputs),
        "natural_distribution": False, "promotion_authorized": False}
    if rubric_version in {"v3", "v4", "v5", "v6", "v7", "v8"}:
        body.update({key: value for key, value in freeze._protocol_bindings(packet).items()
                     if key not in {"rubric_version", "rubric_sha256"}})
        body["production_gate_compatible"] = False
        body["promotion_authorized"] = False
    if packet.get("schema") == COHORT_V10_SCHEMA:
        body["target_protocol_version"] = packet["target_protocol"]["version"]
        body["target_protocol_sha256"] = sha256_obj(packet["target_protocol"])
    if v11:
        body["max_tokens"] = V11_MAX_TOKENS
        body["target_protocol_version"] = packet["target_protocol"]["version"]
        body["target_protocol_sha256"] = sha256_obj(packet["target_protocol"])
        body["reference_protocol_hash"] = packet["reference_protocol"]["protocol_hash"]
        body["target_transport"] = protocol["transport"]
        body["dynamic_schema_manifest_sha256"] = protocol["dynamic_schema_manifest_sha256"]
        body["ai_job_protocol_metadata"] = job_protocol_metadata
    return {**body, "plan_sha256": sha256_obj(body)}, inputs, cfg, verified


def calibration_plan(packet: dict, frozen: dict,
        schedule_a: dict, state_a, aggregate_a: dict,
        schedule_b: dict, state_b, aggregate_b: dict,
        target_model_id: str, *, workspace: str | Path | None = None) -> dict:
    """Create a zero-call plan after source, coverage, family and account checks."""
    root = Path(workspace or repo_root()).resolve()
    plan, _inputs_value, _cfg, _verified = _build_plan(packet, frozen,
        schedule_a, state_a, aggregate_a, schedule_b, state_b, aggregate_b,
        target_model_id, root)
    return plan


def _label(dimension: str, verdict: str) -> str:
    table = _ASSERTION if dimension == "assertion" else _CONTRACT
    if verdict == table["approve"]:
        return "approve"
    if verdict == table["reject"]:
        return "reject"
    if verdict == "uncertain":
        return "uncertain"
    return "malformed"


def _normalization_issue_attribution(evidence: dict, dimension: str) -> tuple[int, int]:
    """Return dimension-tagged issues and unassigned scalar/unknown issues."""
    issues = evidence.get("normalization_issues", [])
    if not isinstance(issues, list):
        return 0, 1
    attributed = sum(isinstance(issue, dict)
                     and issue.get("dimension") == dimension for issue in issues)
    unassigned = sum(not isinstance(issue, dict) for issue in issues)
    return attributed, unassigned


def _validate_target_result_output(packet: dict, output: dict, *, inputs=None) -> None:
    """Bind v10 scorer output to its dedicated schema and current target protocol."""
    if packet.get("schema") == COHORT_V11_SCHEMA:
        protocol = _protocol_v11(packet)
        expected = {
            "schema": OUTPUT_SCHEMA_V9,
            "packet_sha256": packet["packet_sha256"],
            "target_protocol_version": packet["target_protocol"]["version"],
            "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
            "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
            "dynamic_schema_manifest_sha256": protocol["dynamic_schema_manifest_sha256"],
            "target_transport": protocol["transport"],
        }
        if any(output.get(key) != value for key, value in expected.items()):
            raise ValueError("v11 target scorer result is not bound to its v8 dynamic-schema protocol")
        if inputs is not None:
            expected_cases = inputs.get("cases", [])
            output_cases = output.get("cases")
            if (not isinstance(output_cases, list)
                    or [row.get("case_id") for row in output_cases]
                        != [row.get("case_id") for row in expected_cases]):
                raise ValueError("v11 target scorer result case order or coverage changed")
            for item, result in zip(expected_cases, output_cases):
                provenance = result.get("target_provenance", {})
                if (provenance.get("target_protocol_version")
                        != packet["target_protocol"]["version"]
                        or provenance.get("target_protocol_sha256")
                            != sha256_obj(packet["target_protocol"])
                        or provenance.get("response_format_sha256")
                            != item.get("target_response_format_sha256")
                        or provenance.get("candidate_spans_sha256")
                            != item.get("target_candidate_spans_sha256")
                        or provenance.get("contract_spans_sha256")
                            != item.get("target_contract_spans_sha256")
                        or provenance.get("parser") != "parse_prompt_grounding_review_v8"
                        or type(result.get("parser_schema_valid")) is not bool
                        or provenance.get("schema_valid")
                            is not result.get("parser_schema_valid")):
                    raise ValueError("v11 target result case is not bound to its parser/schema identity")
        return
    if packet.get("schema") != COHORT_V10_SCHEMA:
        return
    bindings = freeze._protocol_bindings(packet)
    if (output.get("schema") != OUTPUT_SCHEMA_V8
            or any(output.get(key) != value for key, value in bindings.items())):
        raise ValueError("v10 target scorer result is not bound to the frozen target protocol")


def _score(cases: list[dict], dimension: str) -> dict:
    table = _ASSERTION if dimension == "assertion" else _CONTRACT
    summary = {"reference_approval": 0, "reference_rejection": 0,
        "reference_uncertain": 0, "prediction_approval": 0,
        "prediction_rejection": 0, "prediction_uncertain": 0,
        "malformed_predictions": 0, "false_approval": 0,
        "false_rejection": 0, "decisive_matches": 0,
        "decisive_mismatches": 0, "reviewer_disagreement": 0,
        "reviewer_a_uncertain": 0, "reviewer_b_uncertain": 0,
        "reviewer_a_normalization_issues": 0,
        "reviewer_b_normalization_issues": 0}
    for row in cases:
        ref_label = _label(dimension, row["reference"][dimension])
        pred_label = _label(dimension, row["prediction"][dimension])
        a = row["reviewer_evidence"]["reviewer_a"]
        b = row["reviewer_evidence"]["reviewer_b"]
        summary["reviewer_a_uncertain"] += a["verdicts"][dimension] == "uncertain"
        summary["reviewer_b_uncertain"] += b["verdicts"][dimension] == "uncertain"
        issues_a, unassigned_a = _normalization_issue_attribution(a, dimension)
        issues_b, unassigned_b = _normalization_issue_attribution(b, dimension)
        summary["reviewer_a_normalization_issues"] += issues_a
        summary["reviewer_b_normalization_issues"] += issues_b
        # V8 normalizer issues are strings without a dimension field. Keep
        # them visible as unassigned parser issues instead of assigning them
        # to both rubric dimensions or calling `.get()` on strings.
        if unassigned_a or unassigned_b:
            summary["reviewer_a_unattributed_normalization_issues"] = (
                summary.get("reviewer_a_unattributed_normalization_issues", 0) + unassigned_a)
            summary["reviewer_b_unattributed_normalization_issues"] = (
                summary.get("reviewer_b_unattributed_normalization_issues", 0) + unassigned_b)
        summary["reviewer_disagreement"] += bool(row["reviewer_disagreement"][dimension])
        if ref_label == "approve":
            summary["reference_approval"] += 1
        elif ref_label == "reject":
            summary["reference_rejection"] += 1
        else:
            summary["reference_uncertain"] += 1
        if pred_label == "approve":
            summary["prediction_approval"] += 1
        elif pred_label == "reject":
            summary["prediction_rejection"] += 1
        elif pred_label == "uncertain":
            summary["prediction_uncertain"] += 1
        else:
            summary["malformed_predictions"] += 1
        if ref_label == "reject" and pred_label == "approve":
            summary["false_approval"] += 1
        elif ref_label == "approve" and pred_label == "reject":
            summary["false_rejection"] += 1
        if ref_label in {"approve", "reject"}:
            if ref_label == pred_label:
                summary["decisive_matches"] += 1
            else:
                summary["decisive_mismatches"] += 1
    n_approval = summary["reference_approval"]
    n_rejection = summary["reference_rejection"]
    n_decisive = n_approval + n_rejection
    summary["false_approval_rate"] = (
        summary["false_approval"] / n_rejection if n_rejection else None)
    summary["false_rejection_rate"] = (
        summary["false_rejection"] / n_approval if n_approval else None)
    summary["diagnostic_decisive_match_rate"] = (
        summary["decisive_matches"] / n_decisive if n_decisive else None)
    summary["decisive_reference_denominator"] = n_decisive
    summary["all_cases_denominator"] = len(cases)
    return summary


def calibration_execute(packet: dict, frozen: dict,
        schedule_a: dict, state_a, aggregate_a: dict,
        schedule_b: dict, state_b, aggregate_b: dict,
        target_model_id: str, plan: dict, *, workspace: str | Path | None = None,
        provider=None) -> dict:
    """Revalidate every binding, execute bounded calls, and return diagnostics only."""
    root = Path(workspace or repo_root()).resolve()
    expected_plan, inputs, cfg, verified = _build_plan(packet, frozen,
        schedule_a, state_a, aggregate_a, schedule_b, state_b, aggregate_b,
        target_model_id, root)
    if plan != expected_plan:
        raise ValueError("synthetic target calibration plan or source/config identity changed")
    v11 = packet.get("schema") == COHORT_V11_SCHEMA
    protocol = _protocol_v11(packet) if v11 else None

    def worker(call):
        results = []
        for item in inputs["cases"]:
            if v11:
                request = prepare_prompt_grounding_v8(
                    item["candidate_strategy"], CLINIFACT_CONTRACT)
                if (sha256_obj(request["response_format"])
                        != item.get("target_response_format_sha256")):
                    raise ValueError("v11 target response schema changed before provider call")
                system, user = request["system"], request["user"]
            else:
                system, user = _target_messages(item["candidate_strategy"])
            raw, provenance = call("instruction_judge",
                {"runtime_user_message": user}, system)
            if v11:
                parsed, v11_provenance = _parse_v11_target_output(raw, item, packet)
                provenance = {**provenance, **v11_provenance}
            else:
                parsed = parse_prompt_grounding_review(raw, item["candidate_strategy"],
                    CLINIFACT_CONTRACT, require_span_output=True)
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
        target_binding = (freeze._protocol_bindings(packet)
                          if packet.get("schema") == COHORT_V10_SCHEMA else {})
        v11_binding = ({
            "target_protocol_version": packet["target_protocol"]["version"],
            "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
            "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
            "dynamic_schema_manifest_sha256": protocol["dynamic_schema_manifest_sha256"],
            "target_transport": protocol["transport"],
        } if v11 else {})
        return {"schema": OUTPUT_SCHEMA_V9 if v11 else OUTPUT_SCHEMA_V8 if packet.get("schema") == COHORT_V10_SCHEMA else OUTPUT_SCHEMA,
            "packet_sha256": packet["packet_sha256"], **target_binding,
            **v11_binding,
            "frozen_reference_sha256": verified["frozen_reference_sha256"],
            "cases": results}

    job = ai_jobs.execute(plan["ai_job_plan"], inputs, cfg, worker, provider=provider)
    completed = ai_jobs.verify_completed(job["run_dir"], kind="instruction_judge")
    receipt = completed["receipt"]
    output = completed["output"]
    if (receipt.get("call_count") != len(packet["cases"])
            or receipt.get("usage_complete") is not True
            or receipt.get("measured_usd") != 0.0
            or len(output.get("cases", [])) != len(packet["cases"])):
        raise ValueError("target scorer result is incomplete, nonzero-cost, or usage-unverified")
    _validate_target_result_output(packet, output, inputs=inputs)
    if receipt.get("test_only") is not False:
        measurement_status = "test_only_rejected"
        diagnostic_eligible = False
    else:
        from ..runtime.provider import SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL
        calls = json.loads((completed["root"] / "calls.json").read_text(encoding="utf-8"))
        for index in range(len(packet["cases"])):
            raw_call = json.loads((completed["root"] / f"call-{index}.json").read_text(encoding="utf-8"))
            cost = raw_call.get("cost", {})
            if (raw_call.get("role") != "instruction_judge"
                    or cost.get("usd") != 0.0
                    or cost.get("usage_complete") is not True
                    or cost.get("pricing_region") != "CN"
                    or cost.get("pricing_is_account_invoice") is not False
                    or cost.get("pricing_evidence_report_sha256") !=
                        SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL.get(target_model_id)
                    or cost.get("unknown_retry_attempts") != 0):
                raise ValueError("target scorer call lacks exact zero-price usage evidence")
        accounting = receipt.get("request_accounting", {})
        if (len(calls) != len(packet["cases"])
                or any(row.get("index") != index or row.get("status") != "completed"
                       or row.get("role") != "instruction_judge"
                       for index, row in enumerate(calls))
                or accounting.get("logical_operations") != len(packet["cases"])
                or accounting.get("physical_http_attempts") != len(packet["cases"])
                or accounting.get("physical_status_counts") != {"200": len(packet["cases"])}
                or accounting.get("physical_attempts_exact") is not True):
            raise ValueError("target scorer lacks exactly one successful physical request per case")
        measurement_status = "measured_zero_price"
        diagnostic_eligible = True
    dimensions = {dim: _score(output["cases"], dim)
                  for dim in ("assertion", "contract")}
    rubric_version = verified["rubric_version"]
    v7 = rubric_version == "v7"
    v10 = packet.get("schema") == COHORT_V10_SCHEMA
    v11 = packet.get("schema") == COHORT_V11_SCHEMA
    v6 = rubric_version == "v6"
    v5 = rubric_version == "v5"
    v4 = rubric_version == "v4"
    v3 = rubric_version == "v3"
    result_body = {"schema": OUTPUT_SCHEMA_V9 if v11 else OUTPUT_SCHEMA_V8 if v10 else OUTPUT_SCHEMA_V7 if v7 else OUTPUT_SCHEMA_V6 if v6 else OUTPUT_SCHEMA_V5 if v5 else OUTPUT_SCHEMA_V4 if v4 else
                       OUTPUT_SCHEMA_V3 if v3 else OUTPUT_SCHEMA,
        "status": measurement_status,
        "diagnostic_scope": "synthetic_g06_v11_only" if v11 else "synthetic_g06_v10_only" if v10 else "synthetic_g06_v9_only" if v7 else "synthetic_g06_v8_only" if v6 else "synthetic_g06_v7_only" if v5 else
            "synthetic_g06_v6_only" if v4 else
            "synthetic_g06_v5_only" if v3 else "synthetic_g06_v4_only",
        "packet_sha256": packet["packet_sha256"],
        "frozen_reference_sha256": verified["frozen_reference_sha256"],
        "pair_gate_report_sha256": verified["pair_gate_report_sha256"],
        "rubric_version": rubric_version,
        "rubric_sha256": _rubric_sha256(rubric_version),
        "target_model_id": target_model_id,
        "target_model_family": model_family(target_model_id),
        "target_plan_sha256": plan["plan_sha256"],
        "ai_job_plan_sha256": plan["ai_job_plan"]["plan_sha256"],
        "ai_job_receipt": receipt,
        "test_only_rejected": receipt.get("test_only") is not False,
        "synthetic_diagnostic_eligible": diagnostic_eligible,
        "case_count": len(output["cases"]),
        "dimensions": dimensions,
        "cases": output["cases"],
        "synthetic_only": True, "natural_distribution": False,
        "natural_distribution_calibrated": False,
        "promotion_authorized": False,
        "production_gate_compatible": False,
        "limitations": ["AI reference labels are synthetic diagnostics, not human ground truth.",
            "This result cannot calibrate natural-distribution behavior or authorize production promotion."]}
    if v3 or v4 or v5 or v6 or v7 or v11:
        result_body.update({key: value for key, value in freeze._protocol_bindings(packet).items()
                            if key not in {"rubric_version", "rubric_sha256"}})
    if v11:
        protocol = _protocol_v11(packet)
        result_body.update({
            "target_protocol_version": packet["target_protocol"]["version"],
            "target_protocol_sha256": sha256_obj(packet["target_protocol"]),
            "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
            "dynamic_schema_manifest_sha256": protocol["dynamic_schema_manifest_sha256"],
            "target_transport": protocol["transport"],
            "production_gate_compatible": False,
            "promotion_authorized": False,
        })
    return {**result_body, "result_sha256": sha256_obj(result_body)}
