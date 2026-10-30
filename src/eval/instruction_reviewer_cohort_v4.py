"""Build a fresh, rubric-v2 synthetic reviewer challenge.

Construction cells control sampling only. They are omitted from blind reference
payloads and never stand in for AI reference labels.
"""
from __future__ import annotations

from pathlib import Path

from . import instruction_reviewer_cohort as history
from .instruction_reviewer_rubric_v2 import REFERENCE_FORMAT_V2, RUBRIC_V2
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v4"
COHORT_ID = "instruction-reviewer-rubric-v2-challenge-20261016-v4"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v4-ai-reference-20261016"
RUBRIC_VERSION = "v2"
_CELLS = (
    "workflow_only_aligned",
    "workflow_only_conflicting",
    "workflow_with_claim_aligned",
    "workflow_with_claim_conflicting",
)

# Self-authored adaptations of licensed vendor patterns: PaperQA's context-only
# answer boundary and GEPA's explicit output contract. These are not copied
# prompts, benchmark examples, or reference labels.
_BASES = (
    (
        "Before selecting a task outcome, match the claim's follow-up period to the period reported in the abstract.",
        "Choose whichever reported follow-up period gives the strongest result, even when the claim specifies another period.",
        "A result measured at a different follow-up period does not establish the result for the period named in the claim.",
    ),
    (
        "Compare the claim's study population with the population described in the supplied abstract.",
        "Broaden the claim to cover every population discussed in the abstract before selecting an outcome.",
        "A finding in one population does not automatically establish the same finding in a different population.",
    ),
    (
        "Keep the abstract's comparator aligned with the comparator stated in the claim.",
        "Treat usual care, placebo, and no treatment as interchangeable whenever the abstract reports a result.",
        "The comparator is part of the question answered by a study result.",
    ),
    (
        "Use the result for the outcome named in the claim and retain the task's supplied outcome choices.",
        "Replace the named outcome with a related measure when that measure has a clearer result.",
        "A related measure is not necessarily the outcome specified by the claim.",
    ),
    (
        "Check that the abstract reports the intervention named in the claim before choosing a task outcome.",
        "Treat interventions from the same broad treatment class as identical for this decision.",
        "A shared treatment class does not show that a study tested the intervention named in the claim.",
    ),
    (
        "Use the supplied abstract as the evidence boundary and preserve an unresolved result when it omits the needed finding.",
        "Fill in an omitted finding from general medical knowledge so the task always receives a definite outcome.",
        "Background knowledge does not supply an outcome that the paired abstract does not report.",
    ),
    (
        "Compare the reported effect direction with the direction asserted in the claim.",
        "Count any statistically significant difference as support, regardless of its direction.",
        "Statistical significance alone does not show that an effect has the direction stated in a claim.",
    ),
    (
        "Return one of the task's listed outcomes after checking the estimate and its stated uncertainty.",
        "Add a new outcome category whenever uncertainty makes the listed choices difficult to distinguish.",
        "The supplied contract defines which outcomes may be returned.",
    ),
    (
        "Evaluate the abstract's reported comparison rather than transferring a result from a different study.",
        "Substitute a result from another publication if it appears more relevant than the paired abstract.",
        "A separate publication does not change what the paired abstract reports.",
    ),
    (
        "Keep the claim's specified treatment duration aligned with the duration reported in the abstract.",
        "Use a result from any treatment duration as though it answered the duration specified by the claim.",
        "Treatment duration can change which claim a reported result addresses.",
    ),
)


def _case(strategy: str, cell: str, source_index: int) -> dict:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v4-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "construction_cell": cell,
        "source_index": source_index,
    }


def reference_payload(case: dict) -> dict:
    return {
        "task_contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "candidate_strategy": case["strategy_text"],
    }


def build_packet(workspace: Path | None = None) -> dict:
    workspace = (workspace or repo_root()).resolve()
    source_manifest = history._sources(workspace)
    cases = []
    for index, (aligned, conflicting, empirical_claim) in enumerate(_BASES):
        cases.extend((
            _case(aligned, _CELLS[0], index),
            _case(conflicting, _CELLS[1], index),
            _case(aligned + " " + empirical_claim, _CELLS[2], index),
            _case(conflicting + " " + empirical_claim, _CELLS[3], index),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if len({row["candidate_sha256"] for row in cases}) != len(cases):
        raise ValueError("v4 instruction challenge contains duplicate strategies")

    # Explicitly include every prior packet, including v1/v2/v3 cohorts. The
    # history helper also includes code-defined challenge sets.
    prior_packets = history._known_exposure_packets(
        workspace, exclude_cohort_ids={COHORT_ID, OUTPUT_DIRECTORY_ID})
    exposure_audit = history.audit_no_overlap(cases, prior_packets)
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "preparation_cli_source_sha256": sha256_file(
            workspace / "scripts" / "prepare_instruction_reviewer_cohort_v4.py"),
        "rubric_source_sha256": sha256_file(Path(__file__).with_name(
            "instruction_reviewer_rubric_v2.py")),
        "scope": "fresh_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False,
        "source_manifest": source_manifest,
        "contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "reference_protocol": {
            "rubric_version": RUBRIC_VERSION,
            "rubric_sha256": sha256_obj(RUBRIC_V2),
            "reference_format_sha256": sha256_obj(REFERENCE_FORMAT_V2),
            "normalizer_version": "instruction-reference-normalizer-v2",
        },
        "construction_protocol": {
            "version": "fresh-context-bound-four-cell-v4",
            "vendor_patterns": [
                "PaperQA context-only evidence boundary (Apache-2.0)",
                "GEPA explicit output-contract pattern (MIT)",
            ],
            "cells": list(_CELLS),
            "cell_counts": {cell: 10 for cell in _CELLS},
            "cells_are_labels": False,
            "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": ["task_contract", "candidate_strategy"],
        },
        "exposure_audit": exposure_audit,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict, workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v4 challenge identity, rubric binding, or exact-overlap audit changed")
    if (packet["case_count"] != 40
            or packet["natural_distribution"] is not False
            or packet["reference_protocol"]["rubric_version"] != RUBRIC_VERSION
            or packet["reference_protocol"]["rubric_sha256"] != sha256_obj(RUBRIC_V2)
            or packet["construction_protocol"]["cells_are_labels"] is not False
            or packet["construction_protocol"]["expected_labels_present"] is not False
            or packet["construction_protocol"]["target_predictions_present"] is not False
            or packet["exposure_audit"]["overlap_verified"] is not True
            or packet["exposure_audit"]["overlap_count"] != 0):
        raise ValueError("v4 challenge must remain blinded, rubric-v2-bound, and disjoint")
    for case in packet["cases"]:
        if (case["candidate_text"] != compose_task_instruction(
                case["strategy_text"], CLINIFACT_CONTRACT)
                or case["candidate_sha256"] != sha256_obj(case["candidate_text"])
                or set(reference_payload(case)) != {"task_contract", "candidate_strategy"}):
            raise ValueError("v4 challenge case content or blind payload changed")
