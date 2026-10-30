"""Mandatory pre-evaluation gates for task-input Prompt candidates."""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from ..contracts.types import Candidate, content_sha256
from ..governance.policy import static_gate
from .gepa_search import task_label_contract_conflict
from .fewshot import (DEFAULT_MAX_BLOCK_CHARS, DEFAULT_MAX_DEMO_CHARS,
                      render_demos_block_with_ids)
from .method_rule_gate import validate_statistical_method_rules
from .task_contracts import (task_contract_for_examples,
                             validate_task_contract_instruction)


_UNSUPPORTED_HIGH_STAKES_CLAIM = re.compile(
    r"\b(?:life[- ]threatening|misdiagnos\w*|catastroph\w*|irreversible)\b",
    re.IGNORECASE,
)


def validate_task_input_prompt(
    prompt: str,
    examples: Sequence[dict[str, Any]],
    cfg: dict[str, Any],
    state: Any,
    *,
    grounding_review: dict[str, Any] | None = None,
    candidate_id: str = "task-input-prompt-candidate",
) -> dict[str, Any]:
    """Require label compatibility, project static policy, and measured grounding.

    The grounding reviewer is called by the experiment runner and its exact
    result is passed here. Missing or unresolved grounding can never pass.
    """
    content = str(prompt or "").strip()
    if not content:
        return {"passed": False, "reason": "empty_prompt"}

    task_contract = task_contract_for_examples(examples)
    task_contract_gate = ({
        "required": True,
        "contract_id": task_contract.contract_id,
        "contract_sha256": task_contract.sha256,
        "passed": False,
        "reason": "task_contract_review_missing_or_failed",
    } if task_contract else {"required": False, "passed": True,
                             "reason": "no_registered_task_contract"})

    conflict = task_label_contract_conflict(content, examples)
    label_gate = {
        "passed": conflict is None,
        "reason": conflict or "explicit output labels are present in task options",
    }
    if conflict is not None:
        return {"passed": False, "reason": "task_label_contract_rejected",
                "task_label_contract_gate": label_gate,
                "static_gate": None, "grounding_review": None}

    contract_rule_gate = validate_task_contract_instruction(content, task_contract)
    if not contract_rule_gate.get("passed"):
        return {
            "passed": False,
            "reason": "task_contract_rule_rejected",
            "task_label_contract_gate": label_gate,
            "task_contract_rule_gate": contract_rule_gate,
            "static_gate": None,
            "grounding_review": None,
        }

    high_stakes_claim = _UNSUPPORTED_HIGH_STAKES_CLAIM.search(content)
    if high_stakes_claim:
        return {
            "passed": False,
            "reason": "unsupported_high_stakes_claim_rejected",
            "task_label_contract_gate": label_gate,
            "task_input_scope_gate": {
                "passed": False,
                "reason": "prompt_contains_unsubstantiated_high_stakes_consequence_language",
                "matched_term": high_stakes_claim.group(0),
            },
            "static_gate": None,
            "grounding_review": None,
        }

    candidate = Candidate(
        candidate_id=candidate_id,
        parent_release_id=str(getattr(state, "release_id", "")),
        artifact_type="system_prompt",
        artifact_id="system_prompt",
        diff="validate task-input Prompt before candidate evaluation",
        new_content=content,
        content_sha256=content_sha256(content),
    )
    static = static_gate(candidate, cfg, state)
    if not static.get("passed"):
        return {"passed": False, "reason": "static_gate_rejected",
                "task_label_contract_gate": label_gate,
                "static_gate": static, "grounding_review": None}

    method_rules = validate_statistical_method_rules(content)
    if not method_rules.get("passed"):
        return {"passed": False, "reason": "method_rule_gate_rejected",
                "task_label_contract_gate": label_gate,
                "static_gate": static, "method_rule_gate": method_rules,
                "grounding_review": None}

    grounding_review_identity_matches = bool(
        isinstance(grounding_review, dict)
        and grounding_review.get("candidate_sha256") == content_sha256(content)
    )
    contract_review = ((grounding_review or {}).get("task_contract_review")
                       if isinstance(grounding_review, dict) else None)
    contract_review_passed = bool(
        task_contract is None
        or (isinstance(contract_review, dict)
            and contract_review.get("passed") is True
            and contract_review.get("verdict") == "compatible"
            and contract_review.get("contract_id") == task_contract.contract_id
            and contract_review.get("contract_sha256") == task_contract.sha256
            and contract_review.get("candidate_sha256") == content_sha256(content)
            and contract_review.get("issues") == [])
    )
    if task_contract:
        task_contract_gate = {
            "required": True,
            "contract_id": task_contract.contract_id,
            "contract_sha256": task_contract.sha256,
            "passed": contract_review_passed,
            "reason": ("task_contract_review_passed" if contract_review_passed
                       else "task_contract_review_missing_or_failed"),
            "review": contract_review,
        }
    grounding_passed = bool(
        isinstance(grounding_review, dict)
        and grounding_review.get("passed") is True
        and grounding_review.get("verdict") == "claim_free"
        and grounding_review_identity_matches
        and grounding_review.get("artifact_type") == "system_prompt"
        and grounding_review.get("method") ==
        "separate measured verifier call; candidate artifact only, no benchmark evidence supplied"
        and str(grounding_review.get("model_id") or "").strip()
        and grounding_review.get("assertion_scan_complete") is True
        and contract_review_passed
    )
    if not grounding_passed:
        return {"passed": False,
                "reason": ("task_contract_review_missing_or_failed"
                           if task_contract and not contract_review_passed else
                           "grounding_review_candidate_mismatch"
                           if isinstance(grounding_review, dict)
                           and grounding_review.get("passed") is True
                           and not grounding_review_identity_matches else
                           "grounding_review_missing_or_failed"),
                "task_label_contract_gate": label_gate,
                "static_gate": static,
                "method_rule_gate": method_rules,
                "task_contract_rule_gate": contract_rule_gate,
                "task_contract_gate": task_contract_gate,
                "grounding_review_identity_matches": grounding_review_identity_matches,
                "grounding_review": grounding_review}

    return {"passed": True, "reason": "all_pre_evaluation_gates_passed",
            "task_label_contract_gate": label_gate,
            "static_gate": static,
            "method_rule_gate": method_rules,
            "task_contract_rule_gate": contract_rule_gate,
            "task_contract_gate": task_contract_gate,
            "grounding_review_identity_matches": grounding_review_identity_matches,
            "grounding_review": grounding_review}


def validate_task_demo_rendering(
        demos: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure every candidate demo fits and is therefore visible to the task model."""
    runtime = cfg.get("runtime") or {}
    try:
        block_limit = int(runtime.get("pi_task_demos_max_block_chars",
                                     DEFAULT_MAX_BLOCK_CHARS))
        per_demo_limit = int(runtime.get("pi_task_demo_max_example_chars",
                                         DEFAULT_MAX_DEMO_CHARS))
    except (TypeError, ValueError):
        return {"passed": False, "reason": "invalid_demo_render_budget"}
    ids = [str(item.get("task_id") or
               (item.get("task") or {}).get("task_id") or "") for item in demos]
    if not demos:
        return {"passed": True, "reason": "no_task_demos_to_render",
                "candidate_demo_ids": [], "rendered_demo_ids": [],
                "block_chars": 0, "block_limit_chars": block_limit,
                "per_demo_limit_chars": per_demo_limit}
    if (len(demos) > 3 or not all(ids)
            or len(ids) != len(set(ids))):
        return {"passed": False, "reason": "demo_ids_missing_duplicated_or_out_of_range",
                "candidate_demo_ids": ids, "rendered_demo_ids": []}
    if not 1 <= block_limit <= 4000 or per_demo_limit <= 0:
        return {"passed": False, "reason": "invalid_demo_render_budget",
                "block_limit_chars": block_limit,
                "per_demo_limit_chars": per_demo_limit}
    block, rendered_ids = render_demos_block_with_ids(
        demos, max_block_chars=block_limit, max_demo_chars=per_demo_limit)
    passed = bool(block and rendered_ids == ids)
    return {
        "passed": passed,
        "reason": "all_candidate_demos_rendered" if passed else "candidate_demo_truncated",
        "candidate_demo_ids": ids,
        "rendered_demo_ids": rendered_ids,
        "block_chars": len(block),
        "block_limit_chars": block_limit,
        "per_demo_limit_chars": per_demo_limit,
    }


def make_task_input_prompt_validator(
    examples: Sequence[dict[str, Any]],
    cfg: dict[str, Any],
    state: Any,
    grounding_reviewer: Callable[[str], dict[str, Any]],
    *,
    grounding_call_budget: int,
) -> Callable[[str], dict[str, Any]]:
    """Build the pre-scoring callback consumed by the MIPRO adapter.

    Label and static checks run before the measured verifier, so invalid
    proposals do not spend grounding calls. Every candidate record is retained
    on ``validator.records`` for the run report.
    """
    if grounding_call_budget < 0:
        raise ValueError("grounding call budget must be non-negative")
    records: list[dict[str, Any]] = []
    calls = 0

    def validate(prompt: str) -> dict[str, Any]:
        nonlocal calls
        # MIPRO's compiled artifact strips surrounding whitespace before the
        # final prompt-gate lookup and execution. Bind review records and the
        # retained candidate text to that same canonical instruction identity.
        candidate_text = str(prompt or "").strip()
        identity = content_sha256(candidate_text)
        preliminary = validate_task_input_prompt(
            candidate_text, examples, cfg, state)
        if preliminary.get("reason") not in {
                "grounding_review_missing_or_failed",
                "task_contract_review_missing_or_failed"}:
            record = {"content_sha256": identity, **preliminary}
            if (preliminary.get("static_gate") or {}).get("passed") is True:
                record["candidate_text"] = candidate_text
            records.append(record)
            return preliminary
        if calls >= grounding_call_budget:
            raise RuntimeError("MIPRO candidate-grounding call budget exhausted")
        calls += 1
        review = grounding_reviewer(candidate_text)
        result = validate_task_input_prompt(
            candidate_text, examples, cfg, state, grounding_review=review,
        )
        record = {"content_sha256": identity, **result}
        if (result.get("static_gate") or {}).get("passed") is True:
            record["candidate_text"] = candidate_text
        records.append(record)
        return result

    validate.records = records  # type: ignore[attr-defined]
    validate.grounding_calls = lambda: calls  # type: ignore[attr-defined]
    return validate
