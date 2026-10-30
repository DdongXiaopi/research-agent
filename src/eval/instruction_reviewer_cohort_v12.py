"""Fresh v12 synthetic challenge packet bound to target prompt/protocol v9.

Construction cells are host-only sampling metadata. The blind reference payload
contains only the candidate, immutable contract, and source-bound rubric spans.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v11 as v11
from . import instruction_reviewer_rubric_v8 as rubric_v8
from ..evolution.instruction_review_v9 import prompt_grounding_protocol_v9
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v12"
COHORT_ID = "instruction-reviewer-rubric-v8-target-v9-fresh-challenge-20261017-v12"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v12-target-v9-ai-reference-20261017"
_CELLS = (
    "compatible_procedure_without_claim",
    "direct_conflict_without_claim",
    "compatible_procedure_with_claim",
    "direct_conflict_with_claim",
)

# Ten newly authored scenarios. Four claim cases combine a procedure and a
# truth-apt assertion in one sentence; six use a separate factual sentence.
# The assertions are challenge material, not frozen answer labels.
_SEEDS = (
    (
        "Match the assay timepoint to the timepoint named for the outcome in the paired abstract.",
        "Replace the paired abstract's assay timepoint with a timepoint from a separate laboratory report.",
        "The assay detected 14.2 units per milliliter at week 12.",
    ),
    (
        "Check that the confidence interval belongs to the estimate reported in the paired abstract.",
        "Substitute a confidence interval from a registry summary for the interval in the paired abstract.",
        "The confidence interval ranged from 0.61 to 0.93.",
    ),
    (
        "Use the paired abstract's account of treatment discontinuations when reviewing the claim.",
        "Fill a missing discontinuation count with a count from another publication.",
        "Six participants discontinued treatment because of an adverse event.",
    ),
    (
        "Compare the analyzed participant count with the population described in the paired abstract.",
        "Replace the analyzed participant count with a count from a companion cohort report.",
        "Among 310 randomized participants, 298 completed follow-up.",
    ),
    (
        "Assess symptom-resolution timing using the result reported in the paired abstract.",
        "Use a symptom-resolution time from a separate observational study instead of the paired abstract.",
        "Median time to symptom resolution was 8 days.",
    ),
    (
        "Keep the subgroup estimate tied to the population identified in the paired abstract.",
        "Borrow a subgroup estimate from a different trial and use it in place of the paired abstract's result.",
        "The subgroup estimate favored placebo by 1.4 points.",
    ),
    (
        "Use serious-event counts reported in the paired abstract when checking the safety claim.",
        "Add serious-event counts from an external safety database to replace missing paired-abstract information.",
        "Twenty-seven serious adverse events were reported.",
    ),
    (
        "Apply the contract's label meanings to the hospitalization result in the paired abstract.",
        "Redefine INCONCLUSIVE as proof of no difference even when the paired abstract reports a nonsignificant result.",
        "The trial found no statistically significant difference in hospitalization.",
    ),
    (
        "Compare biomarker changes only with the baseline comparison reported in the paired abstract.",
        "Import a biomarker change from a pharmacology report when the paired abstract does not provide it.",
        "The biomarker level decreased by 11 percent from baseline.",
    ),
    (
        "Check the primary-outcome event count against the paired abstract before selecting a label.",
        "Replace the paired abstract's primary-outcome event count with one from a later review article.",
        "The primary outcome occurred in 18 of 204 participants.",
    ),
)


def _case(strategy: str, cell: str, source_index: int,
          challenge_feature: str = "procedure_only") -> dict[str, Any]:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v12-" + sha256_obj(strategy)[:16],
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
        "reference_spans": rubric_v8.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _target_protocol() -> dict[str, Any]:
    return prompt_grounding_protocol_v9(CLINIFACT_CONTRACT)


def _reference_packet_manifest(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    manifest = []
    for case in cases:
        reference = reference_payload(case)["reference_spans"]
        manifest.append({
            "case_id": case["case_id"],
            "response_format_sha256": reference["response_format_sha256"],
            "candidate_spans_sha256": sha256_obj(reference["candidate_spans"]),
            "contract_spans_sha256": sha256_obj(reference["contract_spans"]),
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
    """Discover v1-v10 history through v11, then bind the full v11 code inventory."""
    current_cases = current_cases or _build_cases()
    prior = v11._prior_exposure_packets(workspace, current_cases)
    prior = [(source, packet) for source, packet in prior
             if not _is_current_snapshot(packet, current_cases)]
    prior.append(("code:instruction_reviewer_cohort_v11._build_cases", {
        "schema": v11.SCHEMA,
        "cohort_id": v11.COHORT_ID,
        "cases": v11._build_cases(),
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
        raise ValueError("v12 must contain 40 unique challenge cases")
    return cases


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    audit = exposure.audit_no_overlap(cases, _prior_exposure_packets(workspace, cases))
    cell_counts = {cell: sum(case["construction_cell"] == cell for case in cases)
                   for cell in _CELLS}
    if cell_counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v12 design strata must remain balanced")
    source_manifest = v11.v10.v9.v8.v7.v6.v5._source_manifest(workspace)
    reference_protocol = {
        **rubric_v8.protocol_metadata(),
        "source_sha256": sha256_file(Path(rubric_v8.__file__)),
    }
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "output_directory_id": OUTPUT_DIRECTORY_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(rubric_v8.__file__)),
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
            "version": "quote-extraction-nested-json-four-cell-v12",
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
        raise ValueError("v12 challenge identity, source audit, or overlap audit changed")
    if (packet.get("schema") != SCHEMA or packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("case_count") != 40 or packet.get("natural_distribution") is not False
            or packet.get("target_protocol") != _target_protocol()
            or packet["target_protocol"].get("version") != "instruction-review-v9-nested-json-claim-extraction"
            or packet["target_protocol"].get("contract_sha256") != CLINIFACT_CONTRACT.sha256
            or packet["reference_protocol"].get("rubric_version") != rubric_v8.RUBRIC_VERSION
            or packet["reference_protocol"].get("protocol_schema") != rubric_v8.PROTOCOL_SCHEMA_V8
            or packet["reference_protocol"].get("protocol_hash") != rubric_v8.protocol_hash_v8()
            or packet["construction_protocol"].get("cell_counts") != {cell: 10 for cell in _CELLS}
            or packet["exposure_audit"].get("overlap_verified") is not True
            or packet["exposure_audit"].get("overlap_count") != 0):
        raise ValueError("v12 must remain blind, balanced, v9-bound, and disjoint")
    for case in packet["cases"]:
        payload = reference_payload(case)
        serialized = json.dumps(payload, ensure_ascii=False).casefold()
        if (case.get("candidate_text") != compose_task_instruction(
                    case["strategy_text"], CLINIFACT_CONTRACT)
                or case.get("candidate_sha256") != sha256_obj(case["candidate_text"])
                or set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or any(marker in serialized for marker in
                       ("construction_cell", "source_index", "challenge_feature", "gold_label"))
                or payload["reference_spans"].get("rubric_version") != rubric_v8.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v8.PROTOCOL_SCHEMA_V8):
            raise ValueError("v12 case or blinded reference payload changed")


def validate_packet_integrity(packet: dict[str, Any], workspace: Path | None = None) -> None:
    """Verify frozen packet bytes, source identities, and immutable contents."""
    workspace = (workspace or repo_root()).resolve()
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA:
        raise ValueError("unsupported v12 packet")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(body):
        raise ValueError("v12 packet hash mismatch")
    if (packet.get("builder_source_sha256") != sha256_file(Path(__file__))
            or packet.get("rubric_source_sha256") != sha256_file(Path(rubric_v8.__file__))):
        raise ValueError("v12 packet builder or rubric source changed")
    manifest = packet.get("source_manifest")
    expected_sources = {"paperqa_context_prompts", "gepa_json_task_prompt",
                        "gepa_terminal_task_prompt", "scholarqa_citation_correctness"}
    if not isinstance(manifest, dict) or set(manifest) != expected_sources:
        raise ValueError("v12 source manifest changed")
    for entry in manifest.values():
        for field, hash_field in (("path", "sha256"), ("license_path", "license_sha256")):
            relative, digest = entry.get(field), entry.get(hash_field)
            path = (workspace / relative).resolve() if isinstance(relative, str) else workspace
            if (not isinstance(digest, str) or not path.is_relative_to(workspace)
                    or not path.is_file() or sha256_file(path) != digest):
                raise ValueError("v12 source manifest file identity changed")
    if (packet.get("cases") != _build_cases()
            or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256}
            or packet.get("target_protocol") != _target_protocol()
            or packet.get("reference_protocol", {}).get("protocol_hash") != rubric_v8.protocol_hash_v8()):
        raise ValueError("v12 frozen content or target/reference protocol changed")
    audit = packet.get("exposure_audit", {})
    if (audit.get("overlap_verified") is not True or audit.get("overlap_count") != 0
            or audit.get("prior_packet_count", 0) < 11
            or not any(receipt.get("source") == "code:instruction_reviewer_cohort_v11._build_cases"
                       for receipt in audit.get("prior_packet_receipts", []))):
        raise ValueError("v12 exposure audit must cover prior v1-v11 inputs")
