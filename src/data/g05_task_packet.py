"""Materialize target-blind G05 task packets and gate resolved AI reference capacity.

Construction contract:
* Start from the immutable `selected_answer_support_v1` / `single_doc` sample plan.
* A separate task author supplies one natural-language question and the answer
  actually selected for every planned source. The builder never invents a label,
  expected class, rationale, rubric, or scorer result.
* Authored task inputs must not carry a label-intent field. The reference process
  later sees only the frozen question, answer, and one cited abstract; final class
  labels are accepted only from an AI-origin resolved-reference report.
* Calibration capacity is counted from resolved labels, never from author intent,
  task templates, answer-generation method, or unresolved/invalid references.
* Every accepted case is bound to one unique DOI group and exact source hash.

This module is intentionally independent of the production runner, source sampler,
and any model/API client. It only builds/verifies local JSON artifacts.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

from .g05_sampling import AUTHORING_CONTEXT_POLICY, authoring_context_exclusion

from ..eval.grounding import evidence_sentence_spans


LABELS = ("supported", "contradicted", "insufficient")
CONSTRUCTION_ROUTES = ("neutral", "source_claim_candidate",
    "counterclaim_candidate", "information_gap_candidate")
MIN_RESOLVED_CASES_PER_CLASS = 30
MIN_DISTINCT_DOI_GROUPS_PER_CLASS = 30
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

# Fail-closed identity allowlist for the providers/models used by the G05
# reference protocol. Family identity is derived here; packet authors cannot
# create independence by supplying arbitrary family strings.
_PROVIDER_ALIASES = {
    "siliconflow": "siliconflow",
    "api.siliconflow.cn": "siliconflow",
    "https://api.siliconflow.cn/v1": "siliconflow",
}
_MODEL_IDENTITIES = {
    "qwen/qwen3-8b": ("qwen3-8b", "qwen"),
    "qwen3-8b": ("qwen3-8b", "qwen"),
    "qwen/qwen3-32b": ("qwen3-32b", "qwen"),
    "qwen3-32b": ("qwen3-32b", "qwen"),
    "qwen/qwen2.5-7b-instruct": ("qwen2.5-7b-instruct", "qwen25"),
    "qwen2.5-7b-instruct": ("qwen2.5-7b-instruct", "qwen25"),
    "thudm/glm-4-9b-0414": ("glm-4-9b-0414", "glm"),
    "glm-4-9b-0414": ("glm-4-9b-0414", "glm"),
    "xingchenagi/xing4.0-29b": ("xing4.0-29b", "xing"),
    "xing4.0-29b": ("xing4.0-29b", "xing"),
    "deepseek/deepseek-chat": ("deepseek-chat", "deepseek"),
    "deepseek-chat": ("deepseek-chat", "deepseek"),
}


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_sha(value: Any, message: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(message)
    return value


def _canonical_identity(provider: Any, model_id: Any, model_family: Any) -> tuple[str, str, str]:
    if not all(isinstance(value, str) and value.strip()
               for value in (provider, model_id, model_family)):
        raise ValueError("AI reference model identity is incomplete")
    provider_key = provider.strip().casefold().rstrip("/")
    canonical_provider = _PROVIDER_ALIASES.get(provider_key)
    if canonical_provider is None:
        raise ValueError("AI reference provider is not in the trusted identity registry")
    model_key = model_id.strip().casefold()
    identity = _MODEL_IDENTITIES.get(model_key)
    if identity is None:
        raise ValueError("AI reference model is not in the trusted identity registry")
    canonical_model, derived_family = identity
    if model_family.strip().casefold() != derived_family:
        raise ValueError("AI reference model family does not match the trusted identity registry")
    return canonical_provider, canonical_model, derived_family


def _validated_reference_runs(reference_report: dict[str, Any]) -> dict[str, tuple[str, str, str]]:
    runs = reference_report.get("model_run_receipts")
    if not isinstance(runs, list):
        raise ValueError("AI reference model run receipts are missing")
    identities: dict[str, tuple[str, str, str]] = {}
    for receipt in runs:
        fields = {"run_id", "provider", "model_id", "model_family", "prompt_sha256",
            "response_sha256", "receipt_sha256"}
        if not isinstance(receipt, dict) or set(receipt) != fields:
            raise ValueError("AI reference model run receipt schema is invalid")
        run_id = receipt.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip() or run_id in identities:
            raise ValueError("AI reference model run receipt ID is duplicate or invalid")
        identity = _canonical_identity(receipt.get("provider"), receipt.get("model_id"),
            receipt.get("model_family"))
        _require_sha(receipt.get("prompt_sha256"), "AI reference prompt hash is invalid")
        _require_sha(receipt.get("response_sha256"), "AI reference response hash is invalid")
        receipt_sha = _require_sha(receipt.get("receipt_sha256"), "AI reference run receipt hash is invalid")
        unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if _canonical_sha(unsigned) != receipt_sha:
            raise ValueError("AI reference model run receipt hash mismatch")
        identities[run_id] = identity
    return identities


def _verify_sample_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(plan, dict) or plan.get("schema") != "g05-deterministic-sample-plan-v2":
        raise ValueError("unsupported G05 sample plan")
    plan_sha = _require_sha(plan.get("plan_sha256"), "sample plan hash is missing or invalid")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if _canonical_sha(unsigned) != plan_sha:
        raise ValueError("sample plan hash mismatch")
    if (plan.get("task_contract") != "selected_answer_support_v1"
            or plan.get("scope") != "single_doc"
            or plan.get("authoring_context_policy") != AUTHORING_CONTEXT_POLICY
            or plan.get("labels_generated") is not False
            or plan.get("model_calls") != 0
            or plan.get("target_scorer_calls") != 0
            or plan.get("calibration_eligible") is not False):
        raise ValueError("sample plan is not a target-blind selected-answer shell plan")
    shells = plan.get("task_shells")
    if (not isinstance(shells, list)
            or len(shells) != plan.get("primary_count", -1) + plan.get("reserve_count", -1)
            or len(shells) != len(plan.get("ordered_dois") or [])):
        raise ValueError("sample plan shell inventory is inconsistent")
    case_ids, groups = set(), set()
    for shell in shells:
        if not isinstance(shell, dict):
            raise ValueError("sample plan contains a non-object shell")
        case_id = shell.get("case_id")
        doi = str(shell.get("source_doi") or "").strip().casefold()
        group = str(shell.get("source_group_id") or "").strip().casefold()
        source_hash = shell.get("source_content_sha256")
        document = shell.get("cited_document")
        if (not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids
                or not doi or doi != group or group in groups
                or shell.get("task_contract") != "selected_answer_support_v1"
                or shell.get("scope") != "single_doc"
                or shell.get("question") is not None
                or shell.get("selected_answer") is not None
                or shell.get("reference_label") is not None
                or shell.get("rationale_spans") != []
                or shell.get("target_scorer_output") is not None
                or not isinstance(document, dict)
                or not isinstance(document.get("title"), str) or not document["title"].strip()
                or not isinstance(document.get("abstract"), str)
                or authoring_context_exclusion(document.get("abstract")) is not None
                or not isinstance(source_hash, str) or _SHA_RE.fullmatch(source_hash) is None):
            raise ValueError("sample plan shell has invalid task fields, authoring context, or source binding")
        case_ids.add(case_id)
        groups.add(group)
    return shells


def _clean_text(value: Any, *, label: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    normalized = value.strip()
    if (not normalized or len(normalized) > max_chars
            or any(ord(char) < 32 and char not in "\n\t\r" for char in normalized)
            or "\ufffd" in normalized):
        raise ValueError(f"{label} is empty, too long, or contains invalid text")
    return normalized


def build_g05_task_packet(sample_plan: dict[str, Any], authored_input: dict[str, Any]) -> dict[str, Any]:
    """Bind separately authored Q/A tasks to every source shell without labels.

    `authored_input` is a deliberately label-free file with exactly one entry per
    shell: case ID, source-group ID, question, selected answer. The function
    refuses partial packets and does not silently substitute reserve cases.
    """
    shells = _verify_sample_plan(sample_plan)
    sample_sha = sample_plan["plan_sha256"]
    if (not isinstance(authored_input, dict)
            or set(authored_input) != {"schema", "sample_plan_sha256", "cases"}
            or authored_input.get("schema") != "g05-task-authoring-input-v1"):
        raise ValueError("task authoring input has unsupported fields or schema")
    if authored_input.get("sample_plan_sha256") != sample_sha:
        raise ValueError("task authoring input is not bound to this sample plan")
    authored = authored_input.get("cases")
    if not isinstance(authored, list):
        raise ValueError("task authoring cases must be a list")
    shell_by_id = {row["case_id"]: row for row in shells}
    authored_by_id: dict[str, dict[str, Any]] = {}
    for row in authored:
        required = {"case_id", "source_group_id", "question", "selected_answer"}
        if not isinstance(row, dict) or not required <= set(row) or set(row) - required - {"construction_route"}:
            raise ValueError("authored task has unsupported fields, including possible label intent")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in shell_by_id or case_id in authored_by_id:
            raise ValueError("authored task case ID is duplicate or outside the sample plan")
        shell = shell_by_id[case_id]
        group = str(row.get("source_group_id") or "").strip().casefold()
        if group != shell["source_group_id"].casefold():
            raise ValueError("authored task source group does not match its sample shell")
        question = _clean_text(row.get("question"), label="question", max_chars=2000)
        answer = _clean_text(row.get("selected_answer"), label="selected_answer", max_chars=5000)
        if not question.endswith("?"):
            raise ValueError("question must be phrased as a question, not a claim-verification prompt")
        if question.casefold() == answer.casefold():
            raise ValueError("question and selected answer must be distinct fields")
        route = row.get("construction_route", "neutral")
        if route not in CONSTRUCTION_ROUTES:
            raise ValueError("construction route is invalid; it is audit metadata, never a reference label")
        authored_by_id[case_id] = {"question": question, "selected_answer": answer}
    if set(authored_by_id) != set(shell_by_id):
        raise ValueError("authored tasks must provide exact sample-plan coverage")

    cases = []
    for shell in shells:
        authored_case = authored_by_id[shell["case_id"]]
        cases.append({
            "case_id": shell["case_id"],
            "selection_role": shell["selection_role"],
            "task_contract": "selected_answer_support_v1",
            "scope": "single_doc",
            "source_group_id": shell["source_group_id"],
            "source_doi": shell["source_doi"],
            "source_content_sha256": shell["source_content_sha256"],
            "question": authored_case["question"],
            "selected_answer": authored_case["selected_answer"],
            "cited_document": {"title": shell["cited_document"]["title"],
                "abstract": shell["cited_document"]["abstract"]},
        })
    unsigned = {
        "schema": "g05-selected-answer-support-task-packet-v1",
        "task_contract": "selected_answer_support_v1",
        "scope": "single_doc",
        "source_sample_plan_sha256": sample_sha,
        "authoring_input_sha256": _canonical_sha(authored_input),
        "case_count": len(cases),
        "source_groups_are_unique": True,
        "reference_labels_generated": False,
        "reference_origin": "pending_independent_ai_reference",
        "target_scorer_outputs_generated": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
        "cases": cases,
    }
    return {**unsigned, "task_packet_sha256": _canonical_sha(unsigned)}


def validate_g05_resolved_label_capacity(
    task_packet: dict[str, Any], reference_report: dict[str, Any]
) -> dict[str, Any]:
    """Count only packet-bound, resolved AI labels toward the frozen 30/30 gate.

    This does not run annotation or inspect target-scorer results. It verifies the
    resolved-reference report contract, exact evidence spans, and per-class DOI
    group capacity before a later calibration plan may be considered.
    """
    if (not isinstance(task_packet, dict)
            or task_packet.get("schema") != "g05-selected-answer-support-task-packet-v1"):
        raise ValueError("unsupported G05 task packet")
    packet_sha = _require_sha(task_packet.get("task_packet_sha256"), "task packet hash is missing or invalid")
    packet_unsigned = {key: value for key, value in task_packet.items() if key != "task_packet_sha256"}
    if _canonical_sha(packet_unsigned) != packet_sha:
        raise ValueError("task packet hash mismatch")
    if (task_packet.get("task_contract") != "selected_answer_support_v1"
            or task_packet.get("scope") != "single_doc"
            or task_packet.get("reference_labels_generated") is not False
            or task_packet.get("target_scorer_outputs_generated") is not False
            or task_packet.get("model_calls") != 0
            or task_packet.get("target_scorer_calls") != 0):
        raise ValueError("task packet is not a frozen target-blind single-document packet")
    cases = task_packet.get("cases")
    if not isinstance(cases, list) or len(cases) != task_packet.get("case_count"):
        raise ValueError("task packet case inventory is inconsistent")
    packet_by_id = {}
    groups = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("task packet contains a non-object case")
        if set(case) != {"case_id", "selection_role", "task_contract", "scope",
                "source_group_id", "source_doi", "source_content_sha256", "question",
                "selected_answer", "cited_document"}:
            raise ValueError("task packet case contains a label, authoring route, or unsupported field")
        case_id = case.get("case_id")
        group = str(case.get("source_group_id") or "").strip().casefold()
        doi = str(case.get("source_doi") or "").strip().casefold()
        if (not isinstance(case_id, str) or not case_id or case_id in packet_by_id
                or not group or group != doi or group in groups
                or case.get("task_contract") != "selected_answer_support_v1"
                or case.get("scope") != "single_doc"):
            raise ValueError("task packet case identity or DOI group is invalid")
        packet_by_id[case_id] = case
        groups.add(group)

    if (not isinstance(reference_report, dict)
            or reference_report.get("schema") != "g05-ai-reference-labels-v1"
            or reference_report.get("annotation_origin") != "ai"):
        raise ValueError("AI reference report schema or origin is invalid")
    if reference_report.get("task_packet_sha256") != packet_sha:
        raise ValueError("AI reference report is bound to another task packet")
    report_sha = _require_sha(reference_report.get("reference_report_sha256"),
        "AI reference report hash is missing or invalid")
    report_unsigned = {key: value for key, value in reference_report.items()
        if key != "reference_report_sha256"}
    if _canonical_sha(report_unsigned) != report_sha:
        raise ValueError("AI reference report hash mismatch")
    model_runs = _validated_reference_runs(reference_report)
    annotations = reference_report.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("AI reference annotations must be a list")
    by_case = {}
    for row in annotations:
        expected = {"case_id", "source_group_id", "resolution_status", "resolved_label",
            "resolution_method", "reference_model_run_ids", "reference_labels",
            "adjudicator_run_id", "adjudicator_label", "rationale_spans"}
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("AI reference row has unsupported fields or schema")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in packet_by_id or case_id in by_case:
            raise ValueError("AI reference case is duplicate or unknown")
        case = packet_by_id[case_id]
        if str(row.get("source_group_id") or "").strip().casefold() != case["source_group_id"].casefold():
            raise ValueError("AI reference source group differs from the task packet")
        status, label, method = row.get("resolution_status"), row.get("resolved_label"), row.get("resolution_method")
        reference_run_ids = row.get("reference_model_run_ids")
        reviewer_labels = row.get("reference_labels")
        adjudicator_run_id = row.get("adjudicator_run_id")
        adjudicator_label = row.get("adjudicator_label")
        spans = row.get("rationale_spans")
        if status not in {"resolved", "unresolved"}:
            raise ValueError("AI reference resolution status is invalid")
        if (not isinstance(reference_run_ids, list) or len(reference_run_ids) != 2
                or any(not isinstance(run_id, str) or run_id not in model_runs for run_id in reference_run_ids)
                or len(set(reference_run_ids)) != 2):
            raise ValueError("AI reference model run identities are invalid")
        reference_identities = [model_runs[run_id] for run_id in reference_run_ids]
        reference_families = {identity[2] for identity in reference_identities}
        if len(reference_families) != 2:
            raise ValueError("AI reference models are not independent families")
        if (not isinstance(reviewer_labels, list) or len(reviewer_labels) != len(reference_run_ids)
                or any(value not in LABELS for value in reviewer_labels)):
            raise ValueError("AI reference reviewer labels are invalid")
        if not isinstance(spans, list) or any(not isinstance(span, str) for span in spans):
            raise ValueError("AI reference rationale spans are invalid")
        abstract = case["cited_document"]["abstract"]
        source_sentences = {span["text"] for span in evidence_sentence_spans(abstract)}
        for span in spans:
            if not span.strip() or span not in source_sentences:
                raise ValueError("AI reference rationale span is not an exact cited abstract sentence")
        if status == "resolved":
            if label not in LABELS or not spans:
                raise ValueError("resolved AI reference lacks a label, independent models, or exact span")
            if method == "two_model_consensus":
                if (any(vote != label for vote in reviewer_labels)
                        or adjudicator_run_id is not None or adjudicator_label is not None):
                    raise ValueError("AI reference consensus labels do not match the resolved label")
            elif method == "third_model_adjudication":
                if (len(set(reviewer_labels)) < 2 or not isinstance(adjudicator_run_id, str)
                        or adjudicator_run_id not in model_runs
                        or model_runs[adjudicator_run_id][2] in reference_families
                        or adjudicator_label != label):
                    raise ValueError("third-model adjudication identity is not independent or label-bound")
            else:
                raise ValueError("resolved AI reference method is invalid")
        else:
            if label is not None or method != "unresolved" or len(set(reviewer_labels)) < 2:
                raise ValueError("unresolved AI reference must reflect independent disagreement and no resolved label")
            if (not isinstance(adjudicator_run_id, str) or adjudicator_run_id not in model_runs
                    or model_runs[adjudicator_run_id][2] in reference_families):
                raise ValueError("third-model adjudication identity is not independent")
            if adjudicator_label is not None:
                raise ValueError("unresolved AI reference must not contain an adjudicator label")
        by_case[case_id] = row
    if set(by_case) != set(packet_by_id):
        raise ValueError("AI references contain duplicate or missing task cases")

    class_groups: dict[str, set[str]] = defaultdict(set)
    class_cases = Counter()
    unresolved = 0
    for case_id, row in by_case.items():
        if row["resolution_status"] != "resolved":
            unresolved += 1
            continue
        label = row["resolved_label"]
        class_cases[label] += 1
        class_groups[label].add(packet_by_id[case_id]["source_group_id"].casefold())
    class_counts = {label: class_cases[label] for label in LABELS}
    doi_counts = {label: len(class_groups[label]) for label in LABELS}
    short = [label for label in LABELS if
        class_counts[label] < MIN_RESOLVED_CASES_PER_CLASS
        or doi_counts[label] < MIN_DISTINCT_DOI_GROUPS_PER_CLASS]
    # A partial-success bundle may be sent through the reference-capacity
    # check to learn whether its labels have enough class coverage. That check
    # must not erase the bundle's upstream incomplete/ineligible status. Keep
    # label capacity and end-to-end calibration eligibility as separate facts.
    packet_complete = task_packet.get("complete_120", len(cases) == 120)
    packet_declares_eligible = task_packet.get("calibration_eligible", True)
    if type(packet_complete) is not bool or type(packet_declares_eligible) is not bool:
        raise ValueError("task packet completeness and eligibility flags must be booleans")
    packet_eligible = packet_complete and packet_declares_eligible and len(cases) == 120
    eligibility_blockers = []
    if not packet_complete or len(cases) != 120:
        eligibility_blockers.append("incomplete_120_case_authoring_packet")
    if not packet_declares_eligible:
        eligibility_blockers.append("upstream_packet_marked_calibration_ineligible")
    unsigned = {
        "schema": "g05-resolved-reference-capacity-v1",
        "task_packet_sha256": packet_sha,
        "reference_report_sha256": report_sha,
        "annotation_origin": "ai",
        "resolved_count": sum(class_counts.values()),
        "unresolved_count": unresolved,
        "resolved_cases_by_label": class_counts,
        "distinct_doi_groups_by_label": doi_counts,
        "minimum_resolved_cases_per_class": MIN_RESOLVED_CASES_PER_CLASS,
        "minimum_distinct_doi_groups_per_class": MIN_DISTINCT_DOI_GROUPS_PER_CLASS,
        "short_classes": short,
        "class_capacity_sufficient": not short,
        "packet_complete_120": packet_complete and len(cases) == 120,
        "eligibility_blockers": eligibility_blockers,
        "calibration_eligible": not short and packet_eligible,
        "target_scorer_calls": 0,
    }
    return {**unsigned, "capacity_report_sha256": _canonical_sha(unsigned)}
