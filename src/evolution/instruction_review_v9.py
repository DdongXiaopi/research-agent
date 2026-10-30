"""Nested-JSON target prompt protocol for instruction reviewer v9.

This target protocol is deliberately separate from v8: it tightens JSON-shape
instruction while retaining the v8 reference output contract and validator.
It does not perform provider calls or parse model output.
"""
from __future__ import annotations

from pathlib import Path

from ..util import sha256_file, sha256_obj
from ..eval.instruction_reviewer_output_contract_v8 import build_response_format_v8
from ..eval.instruction_reviewer_rubric_v3 import split_source
from ..eval.instruction_reviewer_rubric_v8 import protocol_metadata
from .instruction_review import (
    PROMPT_GROUNDING_MAX_SPANS,
    PROMPT_GROUNDING_V8_MAX_TOKENS,
    _model_strategy,
    prompt_grounding_messages_v8,
)


PROMPT_GROUNDING_V9_VERSION = "instruction-review-v9-nested-json-claim-extraction"
PROMPT_GROUNDING_V9_HARNESS_VERSION = "instruction-review-v9-nested-json-output"


def _v9_system(base_system: str, *, include_contract: bool) -> str:
    """Add explicit JSON object nesting and reject the observed dotted-key shape."""
    contract_example = (
        '  "contract": {"<candidate_span_id>": "<permitted contract value>"}\n'
        if include_contract else ""
    )
    assertion_close = "  },\n" if include_contract else "  }\n"
    return base_system + (
        " Output-shape rule: follow the nested JSON object structure in the supplied "
        "response schema. Structural example only (placeholders are not literal IDs "
        "or values):\n"
        "{\n"
        '  "assertion": {\n'
        '    "span_labels": {"<candidate_span_id>": "<permitted label>"},\n'
        '    "claims": [{"span_id": "<candidate_span_id>", "quote": "<exact quote>", '
        '"kind": "<permitted kind>"}]\n'
        f"{assertion_close}"
        f"{contract_example}"
        "}\n"
        "The keys span_labels and claims belong inside the assertion object. "
        + ("Contract, when requested by the supplied schema, is a separate top-level object. "
           if include_contract else "Do not add a contract object when it is absent from the supplied schema. ")
        + "Do not emit dotted literal keys such as "
        '`"assertion.span_labels"` or `"assertion.claims"`; those are invalid and do '
        "not substitute for nested objects. Return only JSON matching the exact supplied "
        "schema, with no markdown or explanatory text."
    )


def prepare_prompt_grounding_v9(prompt: str, task_contract=None) -> dict:
    """Prepare the v9 target request with nested-shape instructions and strict schema."""
    strategy = _model_strategy(prompt, task_contract)
    candidate_spans = split_source(strategy, "candidate")
    if not candidate_spans:
        raise ValueError("candidate_span_inventory_empty")
    if len(candidate_spans) > PROMPT_GROUNDING_MAX_SPANS:
        raise ValueError("candidate_span_limit_exceeded")
    contract_spans = (split_source(task_contract.text, "contract")
                      if task_contract is not None else [])
    response_format = build_response_format_v8(
        [row["id"] for row in candidate_spans],
        [row["id"] for row in contract_spans] if task_contract is not None else None,
    )
    base_system, user = prompt_grounding_messages_v8(
        prompt, task_contract,
        candidate_spans=candidate_spans,
        contract_spans=contract_spans,
    )
    return {
        "candidate_spans": candidate_spans,
        "contract_spans": contract_spans,
        "response_format": response_format,
        "system": _v9_system(base_system, include_contract=task_contract is not None),
        "user": user,
        "candidate_span_count": len(candidate_spans),
        "candidate_span_limit": PROMPT_GROUNDING_MAX_SPANS,
        "target_protocol_version": PROMPT_GROUNDING_V9_VERSION,
        "target_harness_version": PROMPT_GROUNDING_V9_HARNESS_VERSION,
    }


def prompt_grounding_protocol_v9(task_contract=None) -> dict:
    """Return immutable target/reference identity fields for the v9 prompt harness."""
    # This deterministic template binds protocol identity independently of any
    # cohort's candidate IDs. Per-request dynamic schema hashes belong in inputs.
    base_system, _ = prompt_grounding_messages_v8(
        "", task_contract, candidate_spans=[], contract_spans=[]
    )
    system = _v9_system(base_system, include_contract=task_contract is not None)
    schema_template = build_response_format_v8(
        ["V9_CANDIDATE_TEMPLATE"],
        ["V9_CONTRACT_TEMPLATE"] if task_contract is not None else None,
    )
    reference = protocol_metadata()
    protocol = {
        "version": PROMPT_GROUNDING_V9_VERSION,
        "target_harness_version": PROMPT_GROUNDING_V9_HARNESS_VERSION,
        "target_source_sha256": sha256_file(Path(__file__)),
        "messages_sha256": sha256_obj({"system": system}),
        "schema_builder_source_sha256": sha256_file(
            Path(__file__).parents[1] / "eval" / "instruction_reviewer_output_contract_v8.py"
        ),
        "response_format_template_sha256": sha256_obj(schema_template),
        "reference_protocol_hash": reference["protocol_hash"],
        "reference_rubric_version": reference["rubric_version"],
        "reference_rubric_sha256": reference["rubric_sha256"],
        "reference_normalizer_source_hash": reference["normalizer_source_hash"],
        "max_tokens": PROMPT_GROUNDING_V8_MAX_TOKENS,
        "max_spans": PROMPT_GROUNDING_MAX_SPANS,
        "contract_id": task_contract.contract_id if task_contract else None,
        "contract_sha256": task_contract.sha256 if task_contract else None,
    }
    protocol["protocol_hash"] = sha256_obj(protocol)
    return protocol
