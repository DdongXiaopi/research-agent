"""Fresh, plan-only G06 challenge bound to the atomic v8 reviewer contract.

The cohort is a synthetic semantic challenge, not a natural-distribution
estimate. Construction cells are host-only design metadata and never reach the
blind reviewers. No provider/client imports or dispatch path live here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v13 as v13
from . import instruction_reviewer_cohort_v16 as v16
from . import instruction_reviewer_rubric_v8 as rubric_v8
from ..evolution.instruction_review_v10 import prompt_grounding_protocol_v10
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v17"
COHORT_ID = "instruction-reviewer-atomic-contract-v8-fresh-challenge-20261018-v17"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v17-atomic-contract-v8-plan-only-20261019-r3"
PLAN_SCHEMA = "instruction-reviewer-v17-ai-reference-plan-only-v1"
_CELLS = (
    "compatible_procedure_without_claim",
    "direct_conflict_without_claim",
    "compatible_procedure_with_claim",
    "direct_conflict_with_claim",
)

# New examples avoid copying the earlier v8/v13/v16 challenge text. Claim text
# is challenge material and never an expected answer or reference label.
_SEEDS = (
    ("Check the post-discharge infection result against the paired abstract only.",
     "Replace the paired abstract's post-discharge infection result with a rate from a regional dashboard.",
     "Wound infection was recorded in 14 participants by day 30."),
    ("Keep the seizure-recurrence comparison within the supplied paper's follow-up period.",
     "Use a recurrence count from a later extension study in place of the paired paper's follow-up result.",
     "Nine participants had a recurrent seizure within six months."),
    ("Use the endpoint definition and observation window stated in the linked report.",
     "Replace the linked report's endpoint with one defined by an outside clinical guideline.",
     "The primary endpoint was reached by 27 percent of the cohort."),
    ("Compare the renal-safety result with the estimate in its matched publication abstract.",
     "Substitute an acute kidney injury estimate from a separate claims database for the matched abstract result.",
     "Acute kidney injury occurred in 18 of 320 treated patients."),
    ("Retain the original denominator when checking the medication-persistence result.",
     "Change the medication-persistence denominator using a separate pharmacy registry.",
     "Treatment persistence at week 24 was 68 percent."),
    ("Check the functional-recovery statement against the paired study summary.",
     "Replace the paired study's recovery measure with a score from an unrelated rehabilitation cohort.",
     "The recovery score increased by 7.4 points at follow-up."),
    ("Use only the supplied abstract to assess the readmission outcome.",
     "Fill an omitted readmission outcome with counts from a hospital billing system and use those counts to choose the label.",
     "Readmission within 30 days occurred in 22 of 410 patients."),
    ("Keep the bleeding-event assessment aligned with the linked report's definition.",
     "Apply a bleeding-event definition from another protocol instead of the one in the linked report.",
     "Major bleeding was reported in 3.8 percent of participants."),
    ("Compare the reported pain change with the measure and time point in the supplied summary.",
     "Use a pain score from a separate trial to replace the supplied summary's result.",
     "Mean pain intensity declined by 1.3 points at week 12."),
    ("Apply the fixed answer categories to the treatment-discontinuation result in the paired abstract.",
     "Change the meaning of NEI based on a recommendation in an external practice guideline.",
     "Treatment was discontinued because of an adverse event in 11 patients."),
)


def _case(strategy: str, cell: str, source_index: int,
          challenge_feature: str = "procedure_only") -> dict[str, Any]:
    strategy = strategy.rstrip() + " Follow this instruction exactly."
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v17-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "construction_cell": cell,
        "source_index": source_index,
        "challenge_feature": challenge_feature,
    }


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (compatible, conflict, claim) in enumerate(_SEEDS):
        mixed = index < 4
        compatible_with_claim = (
            compatible.rstrip().removesuffix(".") + "; " + claim[0].lower() + claim[1:]
            if mixed else compatible + " " + claim
        )
        conflict_with_claim = (
            conflict.rstrip().removesuffix(".") + "; " + claim[0].lower() + claim[1:]
            if mixed else conflict + " " + claim
        )
        feature = "procedure_and_claim_same_sentence" if mixed else "separate_factual_sentence"
        cases.extend((
            _case(compatible, _CELLS[0], index),
            _case(conflict, _CELLS[1], index),
            _case(compatible_with_claim, _CELLS[2], index, feature),
            _case(conflict_with_claim, _CELLS[3], index, feature),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if len(cases) != 40 or len({row["candidate_sha256"] for row in cases}) != 40:
        raise ValueError("v17 must contain 40 unique synthetic challenge cases")
    return cases


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_contract": {"id": CLINIFACT_CONTRACT.contract_id,
                          "text": CLINIFACT_CONTRACT.text,
                          "sha256": CLINIFACT_CONTRACT.sha256},
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v8.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _target_protocol() -> dict[str, Any]:
    return prompt_grounding_protocol_v10(CLINIFACT_CONTRACT)


def _prior_packets(workspace: Path,
                   cases: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    prior = v16._prior_exposure_packets(workspace, cases)
    # Once a zero-call packet exists, history discovery sees that same v17
    # packet as a prior exposure. Match the complete current case signature
    # and cohort identity, as v16 does for its current snapshot; this excludes
    # only this builder's own immutable zero-call snapshot, not earlier
    # cohorts or any changed case set.
    current_signature = tuple(sorted(
        (row["case_id"], row["candidate_sha256"], sha256_obj(row["strategy_text"]))
        for row in cases
    ))
    prior = [(source, packet) for source, packet in prior
             if not (packet.get("cohort_id") == COHORT_ID
                     and isinstance(packet.get("cases"), list)
                     and tuple(sorted(
                         (row.get("case_id"), row.get("candidate_sha256"),
                          sha256_obj(row.get("strategy_text", "")))
                         for row in packet["cases"] if isinstance(row, dict)
                     )) == current_signature)]
    prior.append(("code:instruction_reviewer_cohort_v16._build_cases", {
        "schema": v16.SCHEMA, "cohort_id": v16.COHORT_ID,
        "cases": v16._build_cases(),
    }))
    # v16 discovers historical packets on disk. The v8/r2 failed-reference
    # payloads are not always packet-shaped, so add that run's frozen source
    # packet explicitly when present.
    legacy = workspace / ".workbuddy" / "g06-instruction-reviewer-v8-rubric-v6-ai-reference-20261018-r2" / "packet.json"
    if legacy.is_file() and not legacy.is_symlink():
        prior.append((legacy.relative_to(workspace).as_posix(),
                      json.loads(legacy.read_text(encoding="utf-8"))))
    return prior


def _reference_manifest(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for case in cases:
        packet = reference_payload(case)["reference_spans"]
        rows.append({"case_id": case["case_id"],
            "response_format_sha256": packet["response_format_sha256"],
            "candidate_spans_sha256": sha256_obj(packet["candidate_spans"]),
            "contract_spans_sha256": sha256_obj(packet["contract_spans"]),
        })
    return rows


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    audit = exposure.audit_no_overlap(cases, _prior_packets(workspace, cases))
    counts = {cell: sum(row["construction_cell"] == cell for row in cases) for cell in _CELLS}
    if counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v17 challenge cells must remain balanced")
    source_manifest = v13.v12.v11.v10.v9.v8.v7.v6.v5._source_manifest(workspace)
    body = {
        "schema": SCHEMA, "cohort_id": COHORT_ID,
        "output_directory_id": OUTPUT_DIRECTORY_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(rubric_v8.__file__)),
        "scope": "fresh_synthetic_semantic_challenge_for_atomic_contract_v8",
        "natural_distribution": False, "source_manifest": source_manifest,
        "contract": {"id": CLINIFACT_CONTRACT.contract_id,
                     "text": CLINIFACT_CONTRACT.text,
                     "sha256": CLINIFACT_CONTRACT.sha256},
        "reference_protocol": {**rubric_v8.protocol_metadata(),
            "source_sha256": sha256_file(Path(rubric_v8.__file__))},
        "target_protocol": _target_protocol(),
        "reference_packet_manifest": _reference_manifest(cases),
        "construction_protocol": {
            "version": "atomic-v8-contract-enum-four-cell-v17",
            "cells": list(_CELLS), "cell_counts": counts,
            "challenge_feature_counts": {feature: sum(row["challenge_feature"] == feature
                for row in cases) for feature in sorted({row["challenge_feature"] for row in cases})},
            "cells_are_labels": False, "expected_labels_present": False,
            "target_predictions_present": False,
            "contract_representation": "one_atomic_enum_per_candidate_span",
            "blind_reference_payload_fields": ["task_contract", "candidate_strategy", "reference_spans"],
        },
        "exposure_audit": audit, "case_count": len(cases), "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict[str, Any], workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v17 cohort identity, protocol, or exposure audit changed")
    if (packet.get("case_count") != 40 or packet.get("natural_distribution") is not False
            or packet.get("reference_protocol", {}).get("rubric_version") != rubric_v8.RUBRIC_VERSION
            or packet.get("reference_protocol", {}).get("protocol_hash") != rubric_v8.protocol_hash_v8()
            or packet.get("target_protocol") != _target_protocol()
            or packet.get("exposure_audit", {}).get("overlap_verified") is not True
            or packet.get("exposure_audit", {}).get("overlap_count") != 0
            or packet.get("construction_protocol", {}).get("contract_representation")
                != "one_atomic_enum_per_candidate_span"):
        raise ValueError("v17 must bind v8 atomic enums, remain blind, and be exposure-disjoint")
    for case in packet["cases"]:
        payload = reference_payload(case)
        encoded = json.dumps(payload, ensure_ascii=False).casefold()
        if (case["candidate_text"] != compose_task_instruction(case["strategy_text"], CLINIFACT_CONTRACT)
                or case["candidate_sha256"] != sha256_obj(case["candidate_text"])
                or payload["reference_spans"].get("protocol_schema") != rubric_v8.PROTOCOL_SCHEMA_V8
                or payload["reference_spans"].get("rubric_version") != rubric_v8.RUBRIC_VERSION
                or any(marker in encoded for marker in
                       ("construction_cell", "source_index", "challenge_feature", "expected_label"))):
            raise ValueError("v17 candidate or blind v8 reference payload changed")


def build_reference_plan(packet: dict[str, Any]) -> dict[str, Any]:
    """Build a non-dispatchable plan with exact per-case v8 prompt/schema hashes."""
    validate_packet_integrity(packet)
    reviewers = {
        "reviewer_a": {"provider": "siliconflow", "model_id": "THUDM/GLM-4-9B-0414", "model_family": "glm"},
        "reviewer_b": {"provider": "siliconflow", "model_id": "XingChenAGI/Xing4.0-29B", "model_family": "xing"},
    }
    cases = []
    for case in packet["cases"]:
        blind = reference_payload(case)
        for slot in reviewers:
            user = json.dumps(blind, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            cases.append({"slot": slot, "case_id": case["case_id"],
                "candidate_sha256": case["candidate_sha256"],
                "input_sha256": sha256_obj(blind),
                "system_prompt_sha256": sha256_obj(rubric_v8.RUBRIC_V8),
                "user_payload_sha256": sha256_obj(user),
                "response_format_sha256": blind["reference_spans"]["response_format_sha256"]})
    body = {
        "schema": PLAN_SCHEMA, "packet_sha256": packet["packet_sha256"],
        "cohort_id": COHORT_ID, "rubric_version": rubric_v8.RUBRIC_VERSION,
        "protocol_hash": rubric_v8.protocol_hash_v8(), "target_protocol_sha256": sha256_obj(_target_protocol()),
        "reviewers": reviewers, "case_count": packet["case_count"],
        "planned_calls_per_reviewer": packet["case_count"],
        "planned_calls_total": 2 * packet["case_count"],
        "max_physical_attempts_per_reviewer": packet["case_count"],
        "retries_per_call": 0, "max_cost_usd_per_reviewer": "0.00",
        "cost_policy": "zero_price_only", "paid_fallback": False,
        "target_model_calls": 0, "model_calls": 0, "dispatch_authorized": False,
        "source_tree_freeze": "deferred_until_G01_G09_source_work_finishes",
        "zero_price_evidence_required_at_dispatch": True,
        "blind_case_plan": cases,
        "scope_limitation": "synthetic challenge only; no natural-distribution reviewer-validity claim",
    }
    return {**body, "plan_sha256": sha256_obj(body)}


def validate_packet_integrity(packet: dict[str, Any], workspace: Path | None = None) -> None:
    workspace = (workspace or repo_root()).resolve()
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA:
        raise ValueError("unsupported v17 packet")
    if packet.get("packet_sha256") != sha256_obj({k: v for k, v in packet.items() if k != "packet_sha256"}):
        raise ValueError("v17 packet hash mismatch")
    if (packet.get("builder_source_sha256") != sha256_file(Path(__file__))
            or packet.get("rubric_source_sha256") != sha256_file(Path(rubric_v8.__file__))):
        raise ValueError("v17 builder or rubric source changed")
    for entry in packet.get("source_manifest", {}).values():
        for field, hash_field in (("path", "sha256"), ("license_path", "license_sha256")):
            relative, digest = entry.get(field), entry.get(hash_field)
            source = (workspace / relative).resolve() if isinstance(relative, str) else workspace
            if (not isinstance(digest, str) or not source.is_relative_to(workspace)
                    or not source.is_file() or sha256_file(source) != digest):
                raise ValueError("v17 source manifest file identity changed")
    expected = _build_cases()
    if (packet.get("cohort_id") != COHORT_ID or packet.get("cases") != expected
            or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256}
            or packet.get("reference_protocol") != {**rubric_v8.protocol_metadata(),
                "source_sha256": sha256_file(Path(rubric_v8.__file__))}
            or packet.get("target_protocol") != _target_protocol()
            or packet.get("reference_packet_manifest") != _reference_manifest(expected)):
        raise ValueError("v17 packet contents or bound v8 protocol changed")
    audit = packet.get("exposure_audit", {})
    if audit.get("overlap_verified") is not True or audit.get("overlap_count") != 0:
        raise ValueError("v17 exposure audit is absent or reports overlap")


def validate_reference_plan(packet: dict[str, Any], plan: dict[str, Any]) -> None:
    expected = build_reference_plan(packet)
    if plan != expected:
        raise ValueError("v17 reference plan is stale or its zero-call boundary changed")

