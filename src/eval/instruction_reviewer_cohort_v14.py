"""Fresh v14 synthetic challenge packet bound to atomic target protocol v10.

Construction cells are host-only sampling metadata. The blind reference payload
contains only the candidate, immutable contract, and source-bound rubric spans.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v13 as v13
from . import instruction_reviewer_rubric_v6 as rubric_v6
from ..evolution.instruction_review_v10 import (
    prepare_prompt_grounding_v10,
    prompt_grounding_protocol_v10,
)
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v14"
COHORT_ID = "instruction-reviewer-rubric-v6-target-v10-fresh-challenge-20261017-v14"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v14-target-v10-ai-reference-20261017"
_CELLS = (
    "compatible_procedure_without_claim",
    "direct_conflict_without_claim",
    "compatible_procedure_with_claim",
    "direct_conflict_with_claim",
)

# Ten newly authored scenarios, independent of v1-v13. Four claim cases combine a procedure and a
# truth-apt assertion in one sentence; six use a separate factual sentence.
# The assertions are challenge material, not frozen answer labels.
_SEEDS = (
    (
        "Compare the reported recurrence result with the paired abstract and retain its answer categories.",
        "Replace the paired abstract's recurrence estimate with a number from an external registry.",
        "Recurrence occurred in 12 percent of participants.",
    ),
    (
        "Use the endpoint definition stated in the paired abstract when evaluating the claim.",
        "Substitute an adherence estimate from a separate implementation survey for the paired result.",
        "Medication adherence reached 76 percent in the intervention arm.",
    ),
    (
        "Check the supplied clinical claim against results in its paired publication abstract only.",
        "Use an infection rate from another hospital network instead of the paired abstract's result.",
        "Postoperative infection occurred in 3.1 percent of participants.",
    ),
    (
        "Keep the answer tied to the publication abstract paired with the supplied claim.",
        "Import a ventilation-duration value from an external quality registry to replace the paired result.",
        "Median ventilation duration was 26 hours.",
    ),
    (
        "Apply the composite endpoint definition reported in the paired abstract.",
        "Replace the supplied endpoint with a composite outcome defined by another protocol.",
        "The composite endpoint occurred in 42 of 500 participants.",
    ),
    (
        "Compare the hemoglobin change with the result reported in the supplied paired abstract.",
        "Use a hemoglobin change from a different treatment study in place of the paired abstract.",
        "Mean hemoglobin increased by 0.8 grams per deciliter.",
    ),
    (
        "Use the paired abstract to assess the reported time to relapse.",
        "Substitute a relapse estimate from an extension publication for the result in the paired abstract.",
        "Median time to relapse was 11 months.",
    ),
    (
        "Check the functional score against the result in the paired publication abstract.",
        "Replace the paired abstract's functional score with a value from a rehabilitation registry.",
        "The mean functional score improved by 5.2 points.",
    ),
    (
        "Use rescue-medication use as reported in the paired abstract when checking the supplied claim.",
        "Fill missing rescue-medication information with counts from a separate pharmacy database.",
        "Thirty-one participants used rescue medication during follow-up.",
    ),
    (
        "Apply the fixed answer-label meanings to the reoperation result in the paired abstract.",
        "Change the meaning of NEI whenever an outside guideline recommends the procedure.",
        "Reoperation was required in 9 of 147 patients.",
    ),
)


def _case(strategy: str, cell: str, source_index: int,
          challenge_feature: str = "procedure_only") -> dict[str, Any]:
    strategy = strategy.rstrip() + " Apply the instruction to each record."
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v14-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "construction_cell": cell,
        "source_index": source_index,
        "challenge_feature": challenge_feature,
    }


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Build blinded source-bound reference inputs; omit all design metadata."""
    return {
        "task_contract": {"id": CLINIFACT_CONTRACT.contract_id,
                          "text": CLINIFACT_CONTRACT.text,
                          "sha256": CLINIFACT_CONTRACT.sha256},
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v6.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _target_protocol() -> dict[str, Any]:
    return prompt_grounding_protocol_v10(CLINIFACT_CONTRACT)


def _reference_packet_manifest(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    manifest = []
    for case in cases:
        reference = reference_payload(case)["reference_spans"]
        target = prepare_prompt_grounding_v10(case["candidate_text"], CLINIFACT_CONTRACT)
        manifest.append({
            "case_id": case["case_id"],
            "reference_format_sha256": reference["reference_format_sha256"],
            "candidate_spans_sha256": sha256_obj(reference["candidate_spans"]),
            "contract_spans_sha256": sha256_obj(reference["contract_spans"]),
            "target_dynamic_schema_sha256": sha256_obj(target["response_format"]),
            "target_candidate_spans_sha256": sha256_obj(target["candidate_spans"]),
            "target_contract_spans_sha256": sha256_obj(target["contract_spans"]),
        })
    return manifest


def _case_signature(cases: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((row.get("case_id", ""), sha256_obj(row.get("strategy_text", "")),
                         sha256_obj(row.get("candidate_text", ""))) for row in cases))


def _is_current_snapshot(packet: dict[str, Any], cases: list[dict[str, Any]]) -> bool:
    rows = packet.get("cases")
    return (packet.get("cohort_id") == COHORT_ID and isinstance(rows, list)
            and len(rows) == len(cases) and _case_signature(rows) == _case_signature(cases))


def _prior_exposure_packets(workspace: Path,
                            current_cases: list[dict[str, Any]] | None = None
                            ) -> list[tuple[str, dict[str, Any]]]:
    """Discover prior exposure through v13 and bind its in-code challenge set."""
    current_cases = current_cases or _build_cases()
    prior = v13._prior_exposure_packets(workspace, current_cases)
    prior = [(source, packet) for source, packet in prior
             if not _is_current_snapshot(packet, current_cases)]
    prior.append(("code:instruction_reviewer_cohort_v13._build_cases", {
        "schema": v13.SCHEMA,
        "cohort_id": v13.COHORT_ID,
        "cases": v13._build_cases(),
    }))
    return prior


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (compatible, conflict, claim) in enumerate(_SEEDS):
        is_mixed = index < 4
        compatible_with_claim = (
            compatible.rstrip().removesuffix(".") + "; " + claim[0].lower() + claim[1:]
            if is_mixed else compatible + " " + claim
        )
        conflict_with_claim = (
            conflict.rstrip().removesuffix(".") + "; " + claim[0].lower() + claim[1:]
            if is_mixed else conflict + " " + claim
        )
        feature = ("procedure_and_claim_same_sentence" if is_mixed
                   else "separate_factual_sentence")
        cases.extend((
            _case(compatible, _CELLS[0], index),
            _case(conflict, _CELLS[1], index),
            _case(compatible_with_claim, _CELLS[2], index, feature),
            _case(conflict_with_claim, _CELLS[3], index, feature),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if (len(cases) != 40 or len({row["case_id"] for row in cases}) != 40
            or len({row["candidate_sha256"] for row in cases}) != 40):
        raise ValueError("v14 must contain 40 unique challenge cases")
    return cases


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    audit = exposure.audit_no_overlap(cases, _prior_exposure_packets(workspace, cases))
    cell_counts = {cell: sum(case["construction_cell"] == cell for case in cases)
                   for cell in _CELLS}
    if cell_counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v14 design strata must remain balanced")
    source_manifest = v13.v12.v11.v10.v9.v8.v7.v6.v5._source_manifest(workspace)
    reference_protocol = {
        **rubric_v6.protocol_metadata(),
        "source_sha256": sha256_file(Path(rubric_v6.__file__)),
    }
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "output_directory_id": OUTPUT_DIRECTORY_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(rubric_v6.__file__)),
        "scope": "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False,
        "source_manifest": source_manifest,
        "contract": {"id": CLINIFACT_CONTRACT.contract_id,
                     "text": CLINIFACT_CONTRACT.text,
                     "sha256": CLINIFACT_CONTRACT.sha256},
        "reference_protocol": reference_protocol,
        "target_protocol": _target_protocol(),
        "reference_packet_manifest": _reference_packet_manifest(cases),
        "construction_protocol": {
            "version": "atomic-span-state-four-cell-v14",
            "cells": list(_CELLS),
            "cell_counts": cell_counts,
            "challenge_feature_counts": {
                feature: sum(case["challenge_feature"] == feature for case in cases)
                for feature in sorted({case["challenge_feature"] for case in cases})
            },
            "cells_are_labels": False,
            "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": [
                "task_contract", "candidate_strategy", "reference_spans"],
        },
        "exposure_audit": audit,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict[str, Any], workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v14 challenge identity, source audit, or overlap audit changed")
    if (packet.get("schema") != SCHEMA or packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("case_count") != 40 or packet.get("natural_distribution") is not False
            or packet.get("target_protocol") != _target_protocol()
            or packet["target_protocol"].get("version") != "instruction-review-v10-atomic-span-states"
            or packet["target_protocol"].get("contract_sha256") != CLINIFACT_CONTRACT.sha256
            or packet["reference_protocol"].get("rubric_version") != "v6"
            or packet["reference_protocol"].get("protocol_schema") != rubric_v6.PROTOCOL_SCHEMA_V6
            or packet["reference_protocol"].get("protocol_hash") != rubric_v6.protocol_hash()
            or packet.get("reference_packet_manifest") != _reference_packet_manifest(
                packet.get("cases", []))
            or packet["construction_protocol"].get("cell_counts") != {cell: 10 for cell in _CELLS}
            or packet["exposure_audit"].get("overlap_verified") is not True
            or packet["exposure_audit"].get("overlap_count") != 0):
        raise ValueError("v14 must remain blind, balanced, v10-bound, and disjoint")
    for case in packet["cases"]:
        payload = reference_payload(case)
        target = prepare_prompt_grounding_v10(case["candidate_text"], CLINIFACT_CONTRACT)
        serialized = json.dumps(payload, ensure_ascii=False).casefold()
        if (case.get("candidate_text") != compose_task_instruction(
                    case["strategy_text"], CLINIFACT_CONTRACT)
                or case.get("candidate_sha256") != sha256_obj(case["candidate_text"])
                or set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or any(marker in serialized for marker in
                       ("construction_cell", "source_index", "challenge_feature", "gold_label"))
                or payload["reference_spans"].get("rubric_version") != rubric_v6.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v6.PROTOCOL_SCHEMA_V6
                or target.get("target_protocol_version") != packet["target_protocol"].get("version")
                or target.get("candidate_span_count") != len(target.get("candidate_spans", []))
                or target.get("response_format") != payload["reference_spans"].get("response_format")):
            raise ValueError("v14 case or blinded reference payload changed")


def validate_packet_integrity(packet: dict[str, Any], workspace: Path | None = None) -> None:
    """Verify frozen packet bytes, source identities, and immutable contents."""
    workspace = (workspace or repo_root()).resolve()
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA:
        raise ValueError("unsupported v14 packet")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(body):
        raise ValueError("v14 packet hash mismatch")
    if (packet.get("builder_source_sha256") != sha256_file(Path(__file__))
            or packet.get("rubric_source_sha256") != sha256_file(Path(rubric_v6.__file__))):
        raise ValueError("v14 packet builder or rubric source changed")
    manifest = packet.get("source_manifest")
    expected_sources = {"paperqa_context_prompts", "gepa_json_task_prompt",
                        "gepa_terminal_task_prompt", "scholarqa_citation_correctness"}
    if not isinstance(manifest, dict) or set(manifest) != expected_sources:
        raise ValueError("v14 source manifest changed")
    for entry in manifest.values():
        for field, hash_field in (("path", "sha256"), ("license_path", "license_sha256")):
            relative, digest = entry.get(field), entry.get(hash_field)
            path = (workspace / relative).resolve() if isinstance(relative, str) else workspace
            if (not isinstance(digest, str) or not path.is_relative_to(workspace)
                    or not path.is_file() or sha256_file(path) != digest):
                raise ValueError("v14 source manifest file identity changed")
    if (packet.get("cases") != _build_cases()
            or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256}
            or packet.get("target_protocol") != _target_protocol()
            or packet.get("reference_protocol", {}).get("protocol_hash") != rubric_v6.protocol_hash()):
        raise ValueError("v14 frozen content or target/reference protocol changed")
    if packet.get("reference_packet_manifest") != _reference_packet_manifest(packet.get("cases", [])):
        raise ValueError("v14 reference/target dynamic schema manifest changed")
    audit = packet.get("exposure_audit", {})
    if (audit.get("overlap_verified") is not True or audit.get("overlap_count") != 0
            or audit.get("prior_packet_count", 0) < 13
            or not any(receipt.get("source") == "code:instruction_reviewer_cohort_v13._build_cases"
                       for receipt in audit.get("prior_packet_receipts", []))):
        raise ValueError("v14 exposure audit must cover prior v1-v13 inputs")
