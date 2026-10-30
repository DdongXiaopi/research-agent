"""Build a fresh, zero-call rubric-v3 instruction-reviewer reference cohort."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as history
from . import instruction_reviewer_rubric_v3 as rubric_v3
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v5"
COHORT_ID = "instruction-reviewer-rubric-v3-challenge-20261016-v5"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v5-ai-reference-20261016"
RUBRIC_VERSION = "v3"
_CELLS = (
    "workflow_only_aligned", "workflow_only_conflicting",
    "workflow_with_claim_aligned", "workflow_with_claim_conflicting",
)

# Original CliniFact adaptations informed by evidence-bound QA, strict structured
# output, and citation-ID retrieval patterns. These strings are authored here;
# vendor examples, benchmark rows, labels, and model predictions are not reused.
_BASES = (
    (
        "Compare the claim's named outcome with the outcome reported in the paired abstract before choosing a listed label.",
        "Replace the claim's outcome with a nearby measure whenever that measure has a more decisive result.",
        "A result for a related outcome does not by itself establish the outcome stated in the claim.",
    ),
    (
        "Check the reported population and treatment period against the population and period named by the claim.",
        "Extend a result to other populations and follow-up periods whenever the intervention is similar.",
        "A study result applies only to the population and period it actually reports.",
    ),
    (
        "Keep the abstract's comparison group aligned with the comparator in the clinical claim.",
        "Treat placebo, usual care, and no treatment as equivalent comparators for this decision.",
        "The comparator affects which clinical claim a reported effect addresses.",
    ),
    (
        "Use the intervention actually evaluated in the paired abstract when deciding whether the claim is supported.",
        "Transfer a finding to another intervention in the same broad drug class.",
        "A shared treatment class does not show that the abstract tested the intervention in the claim.",
    ),
    (
        "When the abstract omits the finding needed for the claim, retain the task label for insufficient evidence.",
        "Complete missing study results from established medical knowledge so each claim gets a definite label.",
        "Information absent from the paired abstract cannot establish its unreported result.",
    ),
    (
        "Compare the direction of the reported effect with the direction asserted in the claim.",
        "Treat statistical significance as support even when the reported effect runs opposite to the claim.",
        "A significant result can contradict a claim when its direction is opposite.",
    ),
    (
        "Use the estimate and uncertainty reported for the claim's specified endpoint.",
        "Call an endpoint clinically meaningful whenever its point estimate favors the claim.",
        "A favorable estimate alone does not establish the task's required statistical result.",
    ),
    (
        "Base the decision on the paired abstract and preserve distinctions among the task's fixed labels.",
        "Combine insufficient information with an inconclusive result because neither confirms the claim.",
        "The task contract assigns distinct meanings to missing evidence and an inconclusive result.",
    ),
    (
        "Check whether the cited abstract reports the clinical endpoint asserted by the claim.",
        "Use a surrogate endpoint as proof of a different patient outcome without an explicit link in the abstract.",
        "Evidence for a surrogate does not automatically establish a separate clinical outcome.",
    ),
    (
        "Assess the trial result for the dose and schedule specified in the claim.",
        "Apply a result from any dose or schedule to the dose named in the claim.",
        "Dose and schedule can change the result that a study supports.",
    ),
)


def _case(strategy: str, cell: str, source_index: int) -> dict[str, Any]:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v5-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "construction_cell": cell,
        "source_index": source_index,
    }


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Return blinded material plus source-bound v3 span inventories."""
    return {
        "task_contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v3.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _source_manifest(workspace: Path) -> dict[str, Any]:
    # These pinned inventories are inherited from the existing licensed source
    # audit. Re-verify bytes and licenses rather than trusting the old manifest.
    inherited = history._sources(workspace)
    selected = ("paperqa_context_prompts", "gepa_json_task_prompt",
                "gepa_terminal_task_prompt")
    result = {key: inherited[key] for key in selected}
    citation_path = "vendor/ScholarQABench-main/ScholarQABench-main/scripts/citation_correctness_eval.py"
    license_path = "vendor/ScholarQABench-main/ScholarQABench-main/LICENSE"
    citation_hash = sha256_file(workspace / citation_path)
    license_hash = sha256_file(workspace / license_path)
    if (citation_hash != "ea12eeb9e140a0791adb5973278e73feee3cba818e878167b5f54212541cab76"
            or license_hash != "59127d0cd4e59e4d25a489b9a66828b7fb21407f778da299206af55741c45a9a"):
        raise ValueError("ScholarQABench citation source or license changed")
    result["scholarqa_citation_correctness"] = {
        "path": citation_path, "sha256": citation_hash, "license": "MIT",
        "license_path": license_path, "license_sha256": license_hash,
        "source_anchors": ["citation identifiers retrieve source passages",
                           "source-grounded attribution check"],
        "adaptation": "self-authored adaptation; only the ID-retrieval pattern is used",
    }
    return result


def _prior_exposure_packets(workspace: Path) -> list[tuple[str, dict[str, Any]]]:
    packets = history._known_exposure_packets(
        workspace, exclude_cohort_ids={COHORT_ID, OUTPUT_DIRECTORY_ID})
    # The failed live v4 run is under a directory whose name does not contain
    # "instruction", so the generic discovery filter misses its packet. Its
    # strategies were exposed to both reviewers and must remain in the audit.
    live_v4 = workspace / ".workbuddy" / "g06-v4-live-calibration-20261016" / "packet.json"
    if not live_v4.is_file():
        raise ValueError("failed live v4 packet is required for the v5 exposure audit")
    live_source = live_v4.relative_to(workspace / ".workbuddy").as_posix()
    if not any(source == live_source for source, _ in packets):
        try:
            packets.append((live_source, json.loads(live_v4.read_text(encoding="utf-8-sig"))))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot audit failed live v4 packet: {live_v4}") from exc
    return packets


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    source_manifest = _source_manifest(workspace)
    cases: list[dict[str, Any]] = []
    for index, (aligned, conflicting, assertion) in enumerate(_BASES):
        cases.extend((
            _case(aligned, _CELLS[0], index),
            _case(conflicting, _CELLS[1], index),
            _case(aligned + " " + assertion, _CELLS[2], index),
            _case(conflicting + " " + assertion, _CELLS[3], index),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if len({row["candidate_sha256"] for row in cases}) != len(cases):
        raise ValueError("v5 instruction challenge contains duplicate strategies")

    # Do not exclude failed cohorts: their text is exposed even when their
    # reference-coverage gate failed. This audits every on-disk v1-v4 packet.
    prior_packets = _prior_exposure_packets(workspace)
    exposure_audit = history.audit_no_overlap(cases, prior_packets)
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "preparation_cli_source_sha256": sha256_file(
            workspace / "scripts" / "prepare_instruction_reviewer_cohort_v5.py"),
        "rubric_source_sha256": sha256_file(Path(rubric_v3.__file__)),
        "scope": "fresh_synthetic_clinifact_instruction_reviewer_reference_cohort",
        "natural_distribution": False,
        "source_manifest": source_manifest,
        "contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "reference_protocol": {
            "rubric_version": RUBRIC_VERSION,
            "rubric_sha256": sha256_obj(rubric_v3.RUBRIC_V3),
            "reference_format_sha256": sha256_obj(rubric_v3.REFERENCE_FORMAT_V3),
            "splitter_hash": rubric_v3.splitter_hash(),
            "protocol_hash": rubric_v3.protocol_hash(),
            "normalizer_version": "instruction-reference-normalizer-v3",
        },
        "construction_protocol": {
            "version": "fresh-context-bound-four-cell-v5",
            "vendor_patterns": [
                "PaperQA context-only evidence boundary (Apache-2.0)",
                "GEPA explicit structured-output contract (MIT)",
                "ScholarQABench citation-ID evidence retrieval pattern (MIT)",
            ],
            "cells": list(_CELLS),
            "cell_counts": {cell: 10 for cell in _CELLS},
            "cells_are_labels": False,
            "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": [
                "task_contract", "candidate_strategy", "reference_spans"],
        },
        "exposure_audit": exposure_audit,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict[str, Any], workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v5 challenge identity, rubric binding, or exact-overlap audit changed")
    protocol = packet["reference_protocol"]
    if (packet["case_count"] != 40 or packet["natural_distribution"] is not False
            or protocol["rubric_version"] != RUBRIC_VERSION
            or protocol["rubric_sha256"] != sha256_obj(rubric_v3.RUBRIC_V3)
            or protocol["reference_format_sha256"] != sha256_obj(rubric_v3.REFERENCE_FORMAT_V3)
            or protocol["splitter_hash"] != rubric_v3.splitter_hash()
            or protocol["protocol_hash"] != rubric_v3.protocol_hash()
            or packet["construction_protocol"]["cells_are_labels"] is not False
            or packet["construction_protocol"]["expected_labels_present"] is not False
            or packet["construction_protocol"]["target_predictions_present"] is not False
            or packet["exposure_audit"]["overlap_verified"] is not True
            or packet["exposure_audit"]["overlap_count"] != 0):
        raise ValueError("v5 challenge must remain blinded, rubric-v3-bound, and disjoint")
    for case in packet["cases"]:
        if (case["candidate_text"] != compose_task_instruction(
                case["strategy_text"], CLINIFACT_CONTRACT)
                or case["candidate_sha256"] != sha256_obj(case["candidate_text"])
                or set(reference_payload(case)) != {
                    "task_contract", "candidate_strategy", "reference_spans"}):
            raise ValueError("v5 challenge case content or blind payload changed")
