"""Zero-call, label-blind G05 task-authoring plan and receipt validator.

Authoring routes are balanced audit metadata only. They are stripped from the
label-free task input consumed by the packet builder and never reach reference
annotation or target scoring. This module has no model client or network code.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .g05_task_packet import (
    CONSTRUCTION_ROUTES,
    _canonical_identity,
    _canonical_sha,
    _require_sha,
)
from .g05_sampling import AUTHORING_CONTEXT_POLICY, authoring_context_exclusion

PLAN_SCHEMA = "g05-task-authoring-plan-v1"
RESULT_SCHEMA = "g05-task-authoring-result-v1"
ROUTES_PER_120 = 30
MAX_COMPLETION_TOKENS = 512
MAX_QUESTION_CHARS = 500
MAX_ANSWER_CHARS = 1500
MAX_ABSTRACT_CHARS = 6000
_ROUTE_PROMPTS = {
    "neutral": "Form a neutral, direct question about a reported study fact.",
    "source_claim_candidate": "Form a question about a specific finding stated in the abstract.",
    "counterclaim_candidate": "Form a question about a comparison or outcome that requires careful reading of the abstract.",
    "information_gap_candidate": "Form a question about what the abstract does and does not report, without assuming an outcome.",
}


def _prompt(shell: dict[str, Any], route: str) -> tuple[str, str]:
    title = str(shell["cited_document"]["title"]).strip()
    abstract = str(shell["cited_document"]["abstract"]).strip()[:MAX_ABSTRACT_CHARS]
    system = (
        "You author one research question and one concise answer using only the supplied abstract. "
        "Do not assign a correctness, support, contradiction, or sufficiency label. "
        "Do not add any other fields. Return a JSON object with exactly question and selected_answer."
    )
    user = (
        f"Authoring style (audit metadata, not a truth label): {_ROUTE_PROMPTS[route]}\n\n"
        f"Title: {title}\nAbstract:\n{abstract}\n\n"
        "Return exactly: {\"question\": \"... ?\", \"selected_answer\": \"...\"}. "
        "The question must end with a question mark. Keep the answer concise and source-grounded."
    )
    return system, user


def _task_identity(raw: dict[str, Any]) -> tuple[str, str, str]:
    if not isinstance(raw, dict) or set(raw) != {"provider", "model_id", "model_family"}:
        raise ValueError("authoring model identity schema is invalid")
    return _canonical_identity(raw["provider"], raw["model_id"], raw["model_family"])


def _validate_sample_plan_for_authoring(sample_plan: dict[str, Any]) -> list[dict[str, Any]]:
    if (not isinstance(sample_plan, dict)
            or sample_plan.get("schema") != "g05-deterministic-sample-plan-v2"):
        raise ValueError("unsupported G05 sample plan")
    plan_sha = _require_sha(sample_plan.get("plan_sha256"), "sample plan hash is missing or invalid")
    if _canonical_sha({key: value for key, value in sample_plan.items() if key != "plan_sha256"}) != plan_sha:
        raise ValueError("sample plan hash mismatch")
    if (sample_plan.get("task_contract") != "selected_answer_support_v1"
            or sample_plan.get("scope") != "single_doc"
            or sample_plan.get("authoring_context_policy") != AUTHORING_CONTEXT_POLICY
            or sample_plan.get("labels_generated") is not False
            or sample_plan.get("model_calls") != 0
            or sample_plan.get("target_scorer_calls") != 0
            or sample_plan.get("calibration_eligible") is not False):
        raise ValueError("sample plan is not a target-blind empty task-shell plan")
    shells = sample_plan.get("task_shells")
    if (not isinstance(shells, list)
            or len(shells) != sample_plan.get("primary_count", -1) + sample_plan.get("reserve_count", -1)):
        raise ValueError("sample plan shell inventory is inconsistent")
    required = {"case_id", "selection_role", "task_contract", "scope", "source_group_id",
        "source_content_sha256", "source_doi", "question", "selected_answer", "cited_document",
        "reference_label", "rationale_spans", "target_scorer_output"}
    seen_ids, seen_groups = set(), set()
    for shell in shells:
        if not isinstance(shell, dict) or set(shell) != required:
            raise ValueError("sample plan shell schema is invalid")
        doi = str(shell.get("source_doi") or "").strip().casefold()
        group = str(shell.get("source_group_id") or "").strip().casefold()
        document = shell.get("cited_document")
        if (not isinstance(shell.get("case_id"), str) or not shell["case_id"].strip()
                or shell["case_id"] in seen_ids or not doi or doi != group or group in seen_groups
                or shell.get("task_contract") != "selected_answer_support_v1"
                or shell.get("scope") != "single_doc" or shell.get("question") is not None
                or shell.get("selected_answer") is not None or shell.get("reference_label") is not None
                or shell.get("rationale_spans") != [] or shell.get("target_scorer_output") is not None
                or not isinstance(document, dict)
                or not isinstance(document.get("title"), str) or not document["title"].strip()
                or not isinstance(document.get("abstract"), str)
                or authoring_context_exclusion(document.get("abstract")) is not None
                or not _is_sha(shell.get("source_content_sha256"))):
            raise ValueError("sample plan shell identity, authoring context, or empty-task contract is invalid")
        seen_ids.add(shell["case_id"])
        seen_groups.add(group)
    return shells


def _is_sha(value: Any) -> bool:
    try:
        _require_sha(value, "invalid hash")
        return True
    except ValueError:
        return False


def build_g05_task_authoring_plan(
    sample_plan: dict[str, Any], *, seed: str,
    author_model: dict[str, Any] | None = None,
    target_model: dict[str, Any] | None = None,
    reference_models: list[dict[str, Any]] | None = None,
    allow_dispatch: bool = True,
) -> dict[str, Any]:
    """Create a deterministic, zero-call plan over each frozen DOI shell.

    A draft without all role identities is review-only and cannot validate
    authoring calls. The target must be family-independent from both reference
    models, and the author must differ from the target. Authoring creates only
    question/answer text, not labels, so its family may overlap a reference model.
    Dispatch remains a separate explicit switch; all identities are allowlisted.
    """
    shells = _validate_sample_plan_for_authoring(sample_plan)
    if len(shells) != 120:
        raise ValueError("G05 authoring requires the frozen 120-shell sample plan")
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("deterministic authoring seed is required")
    role_values = (author_model, target_model, reference_models)
    supplied = [value is not None for value in role_values]
    if any(supplied) and not all(supplied):
        raise ValueError("author, target, and both reference model identities must be bound together")
    role_identities = None
    if all(supplied):
        if not isinstance(reference_models, list) or len(reference_models) != 2:
            raise ValueError("exactly two independent reference model identities are required")
        author = _task_identity(author_model)
        target = _task_identity(target_model)
        references = [_task_identity(item) for item in reference_models]
        if references[0][2] == references[1][2]:
            raise ValueError("reference models must use different trusted model families")
        if target in references or target[2] in {references[0][2], references[1][2]}:
            raise ValueError("target model must be independent from both reference models and families")
        if author[2] == target[2]:
            raise ValueError("author model family must be independent of target family")
        role_identities = {"author": author, "target": target, "references": references}
    if not isinstance(allow_dispatch, bool):
        raise ValueError("allow_dispatch must be a boolean")

    order = sorted(shells, key=lambda row: hashlib.sha256(
        f"{seed}\0{row['source_doi']}\0{row['source_content_sha256']}".encode("utf-8")
    ).hexdigest())
    route_for = {row["case_id"]: CONSTRUCTION_ROUTES[index % len(CONSTRUCTION_ROUTES)]
                 for index, row in enumerate(order)}
    candidates = []
    for index, shell in enumerate(shells):
        abstract = shell["cited_document"]["abstract"].strip()
        if len(abstract) < 400:
            preflight_status, preflight_reason = "excluded", "abstract_below_400_char_minimum"
        elif len(abstract) > MAX_ABSTRACT_CHARS:
            preflight_status, preflight_reason = "excluded", "abstract_exceeds_context_limit"
        elif "\ufffd" in abstract:
            preflight_status, preflight_reason = "excluded", "abstract_contains_replacement_character"
        else:
            preflight_status, preflight_reason = "eligible", None
        route = route_for[shell["case_id"]]
        system, user = _prompt(shell, route) if preflight_status == "eligible" else (None, None)
        prompt_sha = hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest() if system else None
        candidate = {
            "call_index": index,
            "case_id": shell["case_id"],
            "source_group_id": shell["source_group_id"],
            "source_doi": shell["source_doi"],
            "source_content_sha256": shell["source_content_sha256"],
            "selection_role": shell["selection_role"],
            "authoring_route": route,
            "preflight_status": preflight_status,
            "preflight_reason": preflight_reason,
            "system": system,
            "user": user,
            "prompt_sha256": prompt_sha,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        candidates.append(candidate)
    eligible_count = sum(row["preflight_status"] == "eligible" for row in candidates)
    excluded_count = len(candidates) - eligible_count
    route_counts = {route: sum(row["authoring_route"] == route for row in candidates)
                    for route in CONSTRUCTION_ROUTES}
    unsigned = {
        "schema": PLAN_SCHEMA,
        "status": ("blocked_source_context_preflight" if excluded_count else
            "draft_requires_root_review" if role_identities is None or not allow_dispatch
            else "frozen_zero_call_plan"),
        "sample_plan_sha256": sample_plan["plan_sha256"],
        "task_contract": "selected_answer_support_v1",
        "scope": "single_doc",
        "seed": seed,
        "route_allocation": "balanced authoring audit routes; not class quotas or labels",
        "route_counts": route_counts,
        "author_model_identity": role_identities["author"] if role_identities else None,
        "target_model_identity": role_identities["target"] if role_identities else None,
        "reference_model_identities": role_identities["references"] if role_identities else None,
        "model_family_independence_contract": (
            "target and both references pairwise family-disjoint; author differs from target; author emits no labels"),
        "reference_target_family_independence_required": True,
        "dispatch_authorized": bool(allow_dispatch and role_identities is not None and not excluded_count),
        "cases": candidates,
        "source_case_count": len(candidates),
        "preflight_eligible_count": eligible_count,
        "preflight_excluded_count": excluded_count,
        "request_count": eligible_count,
        "max_physical_attempts": eligible_count,
        "max_attempts_per_case": 1,
        "retries_per_case": 0,
        "max_completion_tokens_per_case": MAX_COMPLETION_TOKENS,
        "max_total_completion_tokens": eligible_count * MAX_COMPLETION_TOKENS,
        "model_calls_during_planning": 0,
        "target_scorer_calls_during_planning": 0,
        "reference_calls_during_planning": 0,
        "calls_allowed": bool(allow_dispatch and role_identities is not None and excluded_count == 0),
    }
    return {**unsigned, "plan_sha256": _canonical_sha(unsigned)}


def _valid_task_response(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"question", "selected_answer"}:
        return False
    question, answer = value["question"], value["selected_answer"]
    if (not isinstance(question, str) or not isinstance(answer, str)
            or not question.strip() or not answer.strip()
            or len(question.strip()) > MAX_QUESTION_CHARS
            or len(answer.strip()) > MAX_ANSWER_CHARS
            or not question.strip().endswith("?")
            or question.strip().casefold() == answer.strip().casefold()
            or any(key in value for key in ("label", "verdict", "expected_label", "rationale_spans"))):
        return False
    return True


def validate_g05_task_authoring_results(
    plan: dict[str, Any], call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate recorded author outputs and preserve every failure in the ledger."""
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported G05 task authoring plan")
    plan_sha = _require_sha(plan.get("plan_sha256"), "task authoring plan hash is invalid")
    if _canonical_sha({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan_sha:
        raise ValueError("task authoring plan hash mismatch")
    if not plan.get("calls_allowed") or plan.get("dispatch_authorized") is not True:
        raise ValueError("review-only authoring draft does not authorize model calls")
    if not isinstance(call_records, list):
        raise ValueError("task authoring call records must be a list")
    author_identity = plan["author_model_identity"]
    expected_identity = _canonical_identity(*author_identity)
    all_cases = {row["case_id"]: row for row in plan["cases"]}
    expected_cases = {row["case_id"]: row for row in plan["cases"]
                      if row["preflight_status"] == "eligible"}
    grouped: dict[str, list[dict[str, Any]]] = {}
    unexpected = 0
    attempts_total = 0
    bad_attempt_records = 0
    for call in call_records:
        if not isinstance(call, dict):
            unexpected += 1
            bad_attempt_records += 1
            continue
        attempts = call.get("physical_attempts")
        if type(attempts) is int and attempts >= 0:
            attempts_total += attempts
        else:
            bad_attempt_records += 1
        if call.get("case_id") not in expected_cases:
            unexpected += 1
            continue
        grouped.setdefault(call["case_id"], []).append(call)

    ledger, audit, authored = [], [], []
    failed = missing = duplicate = invalid = 0
    for source_case in plan["cases"]:
        if source_case["preflight_status"] == "eligible":
            continue
        excluded_entry = {"case_id": source_case["case_id"],
            "source_group_id": source_case["source_group_id"],
            "source_content_sha256": source_case["source_content_sha256"],
            "authoring_route": source_case["authoring_route"],
            "status": "preflight_excluded", "reason": source_case["preflight_reason"]}
        ledger.append(excluded_entry)
        audit.append({key: excluded_entry[key] for key in (
            "case_id", "source_group_id", "source_content_sha256", "authoring_route", "status", "reason")})
    for expected in plan["cases"]:
        if expected["preflight_status"] != "eligible":
            continue
        calls = grouped.get(expected["case_id"], [])
        entry = {"case_id": expected["case_id"], "source_group_id": expected["source_group_id"],
                 "source_content_sha256": expected["source_content_sha256"],
                 "authoring_route": expected["authoring_route"], "status": "invalid"}
        if not calls:
            entry["reason"] = "missing_call_receipt"
            missing += 1
        elif len(calls) != 1:
            entry["reason"] = "duplicate_call_receipt"
            duplicate += 1
        else:
            call = calls[0]
            identity = None
            try:
                identity = _canonical_identity(call.get("provider"), call.get("model_id"),
                                               call.get("model_family"))
            except ValueError:
                pass
            raw = call.get("raw_response")
            raw_sha = call.get("raw_response_sha256")
            expected_sha = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")).encode("utf-8")).hexdigest() if raw is not None else None
            reason = None
            if call.get("physical_attempts") != 1:
                reason = "attempt_budget_violation"
            elif call.get("status") != "succeeded":
                reason = "provider_call_failed"
                failed += 1
            elif identity != expected_identity:
                reason = "author_model_identity_mismatch"
            elif call.get("request_sha256") != expected["prompt_sha256"]:
                reason = "prompt_receipt_mismatch"
            elif not isinstance(call.get("completion_tokens"), int) or call["completion_tokens"] < 0 or call["completion_tokens"] > MAX_COMPLETION_TOKENS:
                reason = "completion_token_budget_violation"
            else:
                try:
                    _require_sha(raw_sha, "raw response hash is invalid")
                except ValueError:
                    reason = "raw_response_hash_mismatch"
                if reason is None and raw_sha != expected_sha:
                    reason = "raw_response_hash_mismatch"
            if reason is None and not _valid_task_response(raw):
                reason = "invalid_task_response_schema"
            if reason:
                entry["reason"] = reason
                if reason not in {"provider_call_failed", "attempt_budget_violation"}:
                    invalid += 1
            else:
                q = raw["question"].strip()
                answer = raw["selected_answer"].strip()
                entry.update(status="valid", raw_response_sha256=raw_sha)
                authored.append({"case_id": expected["case_id"],
                    "source_group_id": expected["source_group_id"],
                    "question": q, "selected_answer": answer})
        ledger.append(entry)
        audit.append({key: entry[key] for key in (
            "case_id", "source_group_id", "source_content_sha256", "authoring_route", "status", "reason"
        ) if key in entry})
    duplicate_or_missing_coverage = (missing > 0 or duplicate > 0 or unexpected > 0)
    budget_ok = (attempts_total <= plan["max_physical_attempts"] and bad_attempt_records == 0
                 and all(isinstance(call, dict) and call.get("physical_attempts") == 1
                         for call in call_records))
    complete = (len(authored) == len(all_cases) and not duplicate_or_missing_coverage
                and budget_ok and not failed and not invalid
                and plan["preflight_excluded_count"] == 0)
    authored_input = None
    if complete:
        authored_input = {
            "schema": "g05-task-authoring-input-v1",
            "sample_plan_sha256": plan["sample_plan_sha256"],
            "cases": authored,
        }
    unsigned = {
        "schema": RESULT_SCHEMA,
        "status": "complete" if complete else "incomplete_or_invalid",
        "plan_sha256": plan_sha,
        "source_case_count": len(all_cases),
        "request_count": len(expected_cases),
        "recorded_count": len(call_records),
        "valid_count": len(authored),
        "failed_count": failed,
        "missing_count": missing,
        "duplicate_count": duplicate,
        "unexpected_count": unexpected,
        "invalid_count": invalid,
        "physical_attempts": attempts_total,
        "physical_budget_passed": budget_ok,
        "authoring_audit": audit,
        "case_ledger": ledger,
        "authoring_input": authored_input,
        "model_calls": attempts_total,
        "target_scorer_calls": 0,
        "reference_calls": 0,
    }
    if authored_input is not None:
        unsigned["authoring_input_sha256"] = _canonical_sha(authored_input)
    return {**unsigned, "result_sha256": _canonical_sha(unsigned)}
