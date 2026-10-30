"""Fresh challenge cohort for G06's span-atomic production reviewer protocol.

The 40 synthetic candidates are newly authored and checked against every
discoverable v1-v9 instruction-review exposure packet and code inventory.
Sampling cells are host-only metadata; they are never labels or model inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v9 as v9
from . import instruction_reviewer_rubric_v7 as rubric_v7
from ..evolution.instruction_review import prompt_grounding_protocol
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v10"
COHORT_ID = "instruction-reviewer-rubric-v7-span-atomic-fresh-challenge-20261017-v10"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v10-span-atomic-ai-reference-20261017"
_CELLS = (
    "compatible_procedure_without_proposition",
    "direct_conflict_without_proposition",
    "compatible_procedure_with_proposition",
    "direct_conflict_with_proposition",
)

# New domain scenarios intentionally differ from v1-v9 wording. Each aligned /
# conflicting pair changes only the evidence boundary. The proposition stratum
# is a sampling factor, not an answer key or a prediction of output labels.
_SEEDS = (
    (
        "Compare the claim's outcome direction with the paired abstract's prespecified primary endpoint.",
        "Replace the paired abstract's primary endpoint with a registry outcome from another trial when assigning the label.",
        "The primary endpoint was assessed at the prespecified final visit.",
    ),
    (
        "Use the study population and follow-up period stated in the paired abstract when reviewing the claim.",
        "Borrow a population subgroup estimate from another paper if it better matches the claim than the paired abstract.",
        "The abstract reports the number of participants included in the analysis.",
    ),
    (
        "Check the paired abstract's reported effect estimate and interval against the supplied claim.",
        "Substitute an estimate from a later conference presentation for the result in the paired abstract.",
        "The publication reports an effect estimate with an uncertainty interval.",
    ),
    (
        "For a null result, apply the task's fixed definitions to the paired abstract without adding interpretation from elsewhere.",
        "Treat a null result as proof of no effect whenever an external review says the treatment is ineffective.",
        "The abstract describes the result as not statistically significant.",
    ),
    (
        "Compare safety claims only with adverse-event findings reported in the paired abstract.",
        "Use a pharmacovigilance database to fill missing adverse-event results and assign the claim label.",
        "The report lists adverse events observed during follow-up.",
    ),
    (
        "Use the comparator and outcome definition given in the paired abstract to assess the claim.",
        "Replace the study comparator with usual care from an external guideline before choosing a label.",
        "The study compares the intervention with the control group described in the abstract.",
    ),
    (
        "When the abstract lacks a subgroup result, follow the task's NEI rule for that missing information.",
        "Import a subgroup result from a related cohort and use it to classify the claim despite its absence in the abstract.",
        "No subgroup-specific effect estimate appears in the abstract.",
    ),
    (
        "Apply the supplied label meanings to the reported estimate, regardless of whether the estimate favors the intervention.",
        "Redefine EVIDENCE to mean any statistically significant result, including a result in the opposite direction.",
        "The estimated effect direction is reported relative to the control group.",
    ),
    (
        "Assess the claim against the final result reported in the paired publication abstract.",
        "Use an interim press release instead of the final result in the paired abstract when assigning the label.",
        "The abstract presents results for the study's final analysis.",
    ),
    (
        "Check whether the paired abstract reports the outcome measure named in the claim before selecting a label.",
        "Use a surrogate outcome from a separate study to stand in for the outcome missing from the paired abstract.",
        "The abstract names the outcome measures used in the analysis.",
    ),
)


def _case(strategy: str, cell: str, source_index: int,
          challenge_feature: str = "single_span_type") -> dict[str, Any]:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v10-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "construction_cell": cell,
        "source_index": source_index,
        "challenge_feature": challenge_feature,
    }


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Return only blind candidate/contract/reference-span data."""
    return {
        "task_contract": {"id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256},
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v7.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _case_signature(cases: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((row.get("case_id", ""),
        sha256_obj(row.get("strategy_text", "")),
        sha256_obj(row.get("candidate_text", ""))) for row in cases))


def _is_current_snapshot(packet: dict[str, Any], cases: list[dict[str, Any]]) -> bool:
    rows = packet.get("cases")
    return (packet.get("cohort_id") == COHORT_ID and isinstance(rows, list)
            and len(rows) == len(cases) and _case_signature(rows) == _case_signature(cases))


def _prior_exposure_packets(workspace: Path,
        current_cases: list[dict[str, Any]] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Inherit full v1-v8 discovery, then bind and scan the complete v9 history."""
    current_cases = current_cases or _build_cases()
    prior = v9._prior_exposure_packets(workspace, current_cases)
    prior = [(source, packet) for source, packet in prior
             if not _is_current_snapshot(packet, current_cases)]
    # Include deterministic source inventory so v9 remains covered even if a
    # historical packet was archived or its directory no longer matches scans.
    prior.append(("code:instruction_reviewer_cohort_v9._build_cases", {
        "schema": v9.SCHEMA, "cohort_id": v9.COHORT_ID,
        "cases": v9._build_cases(),
    }))
    return prior


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (aligned, conflicting, proposition) in enumerate(_SEEDS):
        is_mixed_challenge = index < 2
        aligned_with_prop = (aligned.rstrip().removesuffix(".") + "; "
            + proposition[0].lower() + proposition[1:]) if is_mixed_challenge else aligned + " " + proposition
        conflicting_with_prop = (conflicting.rstrip().removesuffix(".") + "; "
            + proposition[0].lower() + proposition[1:]) if is_mixed_challenge else conflicting + " " + proposition
        cases.extend((
            _case(aligned, _CELLS[0], index),
            _case(conflicting, _CELLS[1], index),
            _case(aligned_with_prop, _CELLS[2], index,
                  "mixed_procedure_and_truth_apt_same_sentence" if is_mixed_challenge else "separate_empirical_sentence"),
            _case(conflicting_with_prop, _CELLS[3], index,
                  "mixed_procedure_and_truth_apt_same_sentence" if is_mixed_challenge else "separate_empirical_sentence"),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if (len(cases) != 40 or len({row["case_id"] for row in cases}) != 40
            or len({row["candidate_sha256"] for row in cases}) != 40):
        raise ValueError("v10 must contain 40 unique, newly authored challenge cases")
    return cases


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    audit = exposure.audit_no_overlap(cases, _prior_exposure_packets(workspace, cases))
    cell_counts = {cell: sum(row["construction_cell"] == cell for row in cases)
                   for cell in _CELLS}
    if cell_counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v10 design strata must remain balanced")
    source_manifest = v9.v8.v7.v6.v5._source_manifest(workspace)
    target_protocol = prompt_grounding_protocol(CLINIFACT_CONTRACT)
    body = {
        "schema": SCHEMA, "cohort_id": COHORT_ID,
        "output_directory_id": OUTPUT_DIRECTORY_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(rubric_v7.__file__)),
        "target_protocol_source_sha256": target_protocol["source_sha256"],
        "scope": "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False, "source_manifest": source_manifest,
        "contract": {"id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256},
        "reference_protocol": {
            "rubric_version": "v7", "protocol_schema": rubric_v7.PROTOCOL_SCHEMA_V7,
            "rubric_sha256": sha256_obj(rubric_v7.RUBRIC_V7),
            "response_format_template_sha256": sha256_obj(rubric_v7.RESPONSE_FORMAT_TEMPLATE_V7),
            "splitter_hash": rubric_v7.splitter_hash(),
            "protocol_hash": rubric_v7.protocol_hash(),
            "normalizer_version": rubric_v7.NORMALIZER_VERSION_V7,
            "normalizer_source_hash": rubric_v7.normalizer_source_hash(),
            "max_output_tokens": rubric_v7.REFERENCE_MAX_OUTPUT_TOKENS,
        },
        "target_protocol": target_protocol,
        "construction_protocol": {
            "version": "source-bound-four-cell-span-atomic-v10",
            "cells": list(_CELLS), "cell_counts": cell_counts,
            "challenge_feature_counts": {
                feature: sum(row["challenge_feature"] == feature for row in cases)
                for feature in sorted({row["challenge_feature"] for row in cases})
            },
            "cells_are_labels": False, "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": ["task_contract", "candidate_strategy", "reference_spans"],
        },
        "exposure_audit": audit, "case_count": len(cases), "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict[str, Any], workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v10 challenge identity, source audit, or overlap audit changed")
    if (packet.get("schema") != SCHEMA or packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("case_count") != 40 or packet.get("natural_distribution") is not False
            or packet["reference_protocol"].get("rubric_version") != "v7"
            or packet["reference_protocol"].get("protocol_hash") != rubric_v7.protocol_hash()
            or packet["target_protocol"].get("version") != "instruction-review-v7-span-atomic"
            or packet["target_protocol"].get("contract_sha256") != CLINIFACT_CONTRACT.sha256
            or packet["exposure_audit"].get("overlap_verified") is not True
            or packet["exposure_audit"].get("overlap_count") != 0):
        raise ValueError("v10 must remain blind, balanced, protocol-bound, and disjoint")
    for case in packet["cases"]:
        payload = reference_payload(case)
        serialized = json.dumps(payload, ensure_ascii=False)
        if (case.get("candidate_text") != compose_task_instruction(case["strategy_text"], CLINIFACT_CONTRACT)
                or case.get("candidate_sha256") != sha256_obj(case["candidate_text"])
                or set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or "construction_cell" in serialized or "source_index" in serialized
                or payload["reference_spans"].get("rubric_version") != rubric_v7.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v7.PROTOCOL_SCHEMA_V7):
            raise ValueError("v10 case or blinded reference payload changed")


def validate_packet_integrity(packet: dict[str, Any], workspace: Path | None = None) -> None:
    """Verify a saved packet's bytes, source identities and immutable content."""
    workspace = (workspace or repo_root()).resolve()
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA:
        raise ValueError("unsupported v10 packet")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(body):
        raise ValueError("v10 packet hash mismatch")
    if (packet.get("builder_source_sha256") != sha256_file(Path(__file__))
            or packet.get("rubric_source_sha256") != sha256_file(Path(rubric_v7.__file__))):
        raise ValueError("v10 packet builder or rubric source changed")
    manifest = packet.get("source_manifest")
    expected_sources = {"paperqa_context_prompts", "gepa_json_task_prompt",
                        "gepa_terminal_task_prompt", "scholarqa_citation_correctness"}
    if not isinstance(manifest, dict) or set(manifest) != expected_sources:
        raise ValueError("v10 source manifest changed")
    for entry in manifest.values():
        for field, hash_field in (("path", "sha256"), ("license_path", "license_sha256")):
            relative, digest = entry.get(field), entry.get(hash_field)
            path = (workspace / relative).resolve() if isinstance(relative, str) else workspace
            if (not isinstance(digest, str) or not path.is_relative_to(workspace)
                    or not path.is_file() or sha256_file(path) != digest):
                raise ValueError("v10 source manifest file identity changed")
    cases = _build_cases()
    if (packet.get("cohort_id") != COHORT_ID or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("case_count") != 40 or packet.get("cases") != cases
            or packet.get("target_protocol") != prompt_grounding_protocol(CLINIFACT_CONTRACT)
            or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256}):
        raise ValueError("v10 frozen packet content or target protocol changed")
    audit = packet.get("exposure_audit", {})
    if (audit.get("overlap_verified") is not True or audit.get("overlap_count") != 0
            or audit.get("prior_packet_count", 0) < 9):
        raise ValueError("v10 exposure audit must cover prior v1-v9 inputs")
