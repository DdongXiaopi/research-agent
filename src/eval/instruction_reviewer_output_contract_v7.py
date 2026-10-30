"""Strict per-span output contract for isolated instruction reviewer v7.

Each candidate span is a required object key in both dimensions. One enum
string encodes the complete state, avoiding cross-field conditionals. This
module is intentionally not connected to any cohort, workflow, or API runner.
"""
from __future__ import annotations

import json
from typing import Any


CLAIM_KINDS = (
    "empirical_result", "biomedical_fact", "causal_claim",
    "methodological_generalization", "other_factual_proposition",
)
ISSUE_KINDS = (
    "direct_clause_conflict", "external_evidence_requirement",
    "changed_label_rule", "explicit_contract_conflict",
)
DELIMITER = "|"


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}


def _validate_ids(ids: list[str], name: str) -> list[str]:
    if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError(f"{name}_ids_must_be_nonempty_strings")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate_{name}_id")
    if any(DELIMITER in value for value in ids):
        raise ValueError(f"{name}_id_contains_reserved_delimiter")
    return list(ids)


def assertion_values() -> list[str]:
    return ["procedure", "uncertain", *[f"truth_apt{DELIMITER}{kind}"
            for kind in CLAIM_KINDS]]


def contract_values(contract_ids: list[str]) -> list[str]:
    clause_ids = _validate_ids(contract_ids, "contract")
    return ["no_conflict", "uncertain", *[
        f"direct_conflict{DELIMITER}{clause_id}{DELIMITER}{kind}"
        for clause_id in clause_ids for kind in ISSUE_KINDS
    ]]


def build_response_format_v7(candidate_ids: list[str],
                             contract_ids: list[str]) -> dict[str, Any]:
    """Create a packet-bound schema requiring every candidate ID twice."""
    candidate_ids = _validate_ids(candidate_ids, "candidate")
    contract_ids = _validate_ids(contract_ids, "contract")
    if not candidate_ids or not contract_ids:
        raise ValueError("candidate_and_contract_span_ids_required")
    assertion_enum = assertion_values()
    contract_enum = contract_values(contract_ids)
    total_enum_values = len(candidate_ids) * (len(assertion_enum) + len(contract_enum))
    if total_enum_values > 1000:
        raise ValueError("schema_enum_value_limit_exceeded")
    total_enum_string_length = len(candidate_ids) * (
        sum(map(len, assertion_enum)) + sum(map(len, contract_enum)))
    if total_enum_values > 250 and total_enum_string_length > 15000:
        raise ValueError("schema_single_enum_string_limit_exceeded")
    assertion = _object({candidate_id: {"type": "string", "enum": assertion_enum}
                         for candidate_id in candidate_ids}, candidate_ids)
    contract = _object({candidate_id: {"type": "string",
                                        "enum": contract_enum}
                         for candidate_id in candidate_ids}, candidate_ids)
    schema = _object({"assertion": assertion, "contract": contract},
                     ["assertion", "contract"])
    response_format = {"type": "json_schema", "json_schema": {
        "name": "instruction_reviewer_output_contract_v7",
        "strict": True,
        "schema": schema,
    }}
    if len(json.dumps(response_format, ensure_ascii=False)) > 120000:
        raise ValueError("schema_total_string_limit_exceeded")
    return response_format


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _span_map(spans: list[dict[str, str]], source: str) -> dict[str, str]:
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{source}_span_inventory_required")
    result: dict[str, str] = {}
    for span in spans:
        if (not isinstance(span, dict) or set(span) != {"id", "text"}
                or not isinstance(span["id"], str) or not span["id"]
                or DELIMITER in span["id"] or not isinstance(span["text"], str)
                or not span["text"]):
            raise ValueError(f"invalid_{source}_span_inventory")
        if span["id"] in result:
            raise ValueError(f"duplicate_{source}_span_id")
        result[span["id"]] = span["text"]
    return result


def _decode_assertion(value: Any) -> tuple[str, str | None]:
    if not isinstance(value, str):
        raise ValueError("assertion_value_must_be_string")
    if value in {"procedure", "uncertain"}:
        return value, None
    parts = value.split(DELIMITER)
    if len(parts) == 2 and parts[0] == "truth_apt" and parts[1] in CLAIM_KINDS:
        return "truth_apt", parts[1]
    raise ValueError("invalid_assertion_annotation")


def _decode_contract(value: Any, contract: dict[str, str]
                     ) -> tuple[str, str | None, str | None]:
    if not isinstance(value, str):
        raise ValueError("contract_value_must_be_string")
    if value in {"no_conflict", "uncertain"}:
        return value, None, None
    parts = value.split(DELIMITER)
    if (len(parts) == 3 and parts[0] == "direct_conflict"
            and parts[1] in contract and parts[2] in ISSUE_KINDS):
        return "direct_conflict", parts[1], parts[2]
    raise ValueError("invalid_contract_annotation")


def validate_output_v7(raw: str, candidate_spans: list[dict[str, str]],
                       contract_spans: list[dict[str, str]]) -> dict[str, Any]:
    """Reject duplicate keys and require exact, valid candidate-key coverage."""
    candidate = _span_map(candidate_spans, "candidate")
    contract = _span_map(contract_spans, "contract")
    if not isinstance(raw, str):
        raise ValueError("json_text_required")
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid_json") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"assertion", "contract"}:
        raise ValueError("top_level_schema_mismatch")
    expected_ids = set(candidate)
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for dimension in ("assertion", "contract"):
        mapping = parsed[dimension]
        if not isinstance(mapping, dict):
            raise ValueError(f"{dimension}_must_be_object")
        keys = set(mapping)
        if keys != expected_ids:
            if expected_ids - keys:
                raise ValueError(f"{dimension}_candidate_coverage_missing")
            raise ValueError(f"{dimension}_candidate_coverage_extra")
        decoded = {}
        for candidate_id in candidate:
            decoded[candidate_id] = (
                _decode_assertion(mapping[candidate_id]) if dimension == "assertion"
                else _decode_contract(mapping[candidate_id], contract)
            )
        result[dimension] = decoded
    return result


def normalize_output_v7(raw: str, candidate_spans: list[dict[str, str]],
                        contract_spans: list[dict[str, str]]) -> dict[str, Any]:
    """Derive decisions and reconstruct the v5 normalized evidence fields."""
    candidate = _span_map(candidate_spans, "candidate")
    contract = _span_map(contract_spans, "contract")
    decoded = validate_output_v7(raw, candidate_spans, contract_spans)
    claims = [{"candidate": {"id": candidate_id, "text": candidate[candidate_id]},
               "kind": kind}
              for candidate_id, (status, kind) in decoded["assertion"].items()
              if status == "truth_apt"]
    assertion_uncertain = [{"id": candidate_id, "text": candidate[candidate_id]}
                           for candidate_id, (status, _) in decoded["assertion"].items()
                           if status == "uncertain"]
    assertion_verdict = ("contains_claims" if claims else
                         "uncertain" if assertion_uncertain else "claim_free")
    all_issues = [{
        "candidate": {"id": candidate_id, "text": candidate[candidate_id]},
        "contract": {"id": contract_id, "text": contract[contract_id]},
        "kind": kind,
    } for candidate_id, (status, contract_id, kind)
      in decoded["contract"].items() if status == "direct_conflict"]
    contract_uncertain = [{"id": candidate_id, "text": candidate[candidate_id]}
                          for candidate_id, (status, _, _)
                          in decoded["contract"].items() if status == "uncertain"]
    contract_verdict = ("incompatible" if all_issues else
                        "uncertain" if contract_uncertain else "compatible")
    return {
        "assertion": {
            "verdict": assertion_verdict,
            "claims": claims,
            "uncertain": assertion_uncertain,
            "verdict_reason_code": {
                "claim_free": "no_truth_apt_proposition",
                "contains_claims": "truth_apt_proposition_found",
                "uncertain": "assertion_boundary_unclear",
            }[assertion_verdict],
        },
        "contract": {
            "verdict": contract_verdict,
            "issues": all_issues[:1],
            "uncertain": contract_uncertain,
            "verdict_reason_code": {
                "compatible": "no_explicit_direct_contract_conflict",
                "incompatible": "explicit_direct_contract_conflict_found",
                "uncertain": "direct_conflict_boundary_unclear",
            }[contract_verdict],
        },
        "all_direct_conflicts": all_issues,
        "source_coverage": decoded,
    }
