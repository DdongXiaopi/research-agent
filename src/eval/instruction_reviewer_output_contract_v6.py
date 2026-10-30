"""Standalone exhaustive-span output contract for instruction-reviewer v6.

The model must label every candidate span exactly once in each dimension.
Verdicts and source evidence are derived and reconstructed by the host. This
module is deliberately not wired into the frozen v5/v7 cohort or any API flow.
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
ASSERTION_STATES = ("procedure", "truth_apt", "uncertain")
CONTRACT_STATES = ("no_conflict", "direct_conflict", "uncertain")
_NO_KIND = "not_applicable"
_NO_CONTRACT_ID = "none"


def _obj(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "required": required, "properties": properties}


def _id_schema(ids: list[str] | None, *, sentinel: str | None = None) -> dict[str, Any]:
    if ids is None:
        return {"type": "string"}
    enum = list(dict.fromkeys(ids))
    if sentinel is not None:
        enum.append(sentinel)
    return {"type": "string", "enum": enum}


def build_response_format_v6(candidate_ids: list[str] | None = None,
                             contract_ids: list[str] | None = None,
                             *, expected_span_count: int | None = None) -> dict[str, Any]:
    """Build strict evidence schema; packet-bound IDs and exact array sizes optional.

    Runtime callers should pass IDs from the frozen span packet. The generic
    schema (no IDs) is only the protocol template used in version hashing.
    Array cardinality and uniqueness are additionally enforced by the host.
    """
    candidate_id = _id_schema(candidate_ids)
    contract_id = _id_schema(contract_ids, sentinel=_NO_CONTRACT_ID)
    assertion_annotation = _obj(["candidate_id", "status", "kind"], {
        "candidate_id": candidate_id,
        "status": {"type": "string", "enum": list(ASSERTION_STATES)},
        "kind": {"type": "string", "enum": [*CLAIM_KINDS, _NO_KIND]},
    })
    contract_annotation = _obj(
        ["candidate_id", "status", "contract_id", "issue_kind"], {
            "candidate_id": candidate_id,
            "status": {"type": "string", "enum": list(CONTRACT_STATES)},
            "contract_id": contract_id,
            "issue_kind": {"type": "string", "enum": [*ISSUE_KINDS, _NO_KIND]},
        })
    def annotation_array(item: dict[str, Any]) -> dict[str, Any]:
        value: dict[str, Any] = {"type": "array", "items": item,
                                 "uniqueItems": True}
        if expected_span_count is not None:
            value.update(minItems=expected_span_count, maxItems=expected_span_count)
        return value

    return {"type": "json_schema", "json_schema": {
        "name": "instruction_reference_output_contract_v6", "strict": True,
        "schema": _obj(["assertion", "contract"], {
            "assertion": _obj(["annotations"], {
                "annotations": annotation_array(assertion_annotation),
            }),
            "contract": _obj(["annotations"], {
                "annotations": annotation_array(contract_annotation),
            }),
        }),
    }}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _is_truncated_json(raw: str) -> bool:
    """Conservatively identify unfinished JSON containers/strings at EOF."""
    if not raw.strip():
        return False
    stack: list[str] = []
    in_string = escaped = False
    for char in raw.strip():
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or (char == "}" and stack[-1] != "{") or (char == "]" and stack[-1] != "["):
                return False
            stack.pop()
    return raw.strip().startswith(("{", "[")) and (in_string or bool(stack))


def _span_map(spans: list[dict[str, str]], source: str) -> dict[str, str]:
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{source}_span_inventory_required")
    result: dict[str, str] = {}
    for span in spans:
        if (not isinstance(span, dict) or set(span) != {"id", "text"}
                or not isinstance(span["id"], str) or not isinstance(span["text"], str)
                or not span["id"] or not span["text"]):
            raise ValueError(f"invalid_{source}_span_inventory")
        if span["id"] in result:
            raise ValueError(f"duplicate_{source}_span_id")
        result[span["id"]] = span["text"]
    return result


def _parse(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("json_text_required")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc) == "duplicate_json_key":
            raise
        raise ValueError("truncated_json" if _is_truncated_json(raw) else "invalid_json") from exc
    if not isinstance(value, dict) or set(value) != {"assertion", "contract"}:
        raise ValueError("top_level_schema_mismatch")
    return value


def _validate_assertion(rows: Any, candidate: dict[str, str]) -> list[dict[str, str]]:
    if not isinstance(rows, list) or len(rows) != len(candidate):
        raise ValueError("assertion_span_coverage_mismatch")
    seen: set[str] = set()
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"candidate_id", "status", "kind"}:
            raise ValueError("assertion_annotation_schema_mismatch")
        candidate_id, status, kind = row["candidate_id"], row["status"], row["kind"]
        if not isinstance(candidate_id, str) or candidate_id not in candidate:
            raise ValueError("unknown_candidate_id")
        if candidate_id in seen:
            raise ValueError("duplicate_assertion_candidate_id")
        seen.add(candidate_id)
        if status not in ASSERTION_STATES:
            raise ValueError("unknown_assertion_status")
        if status == "truth_apt":
            if kind not in CLAIM_KINDS:
                raise ValueError("truth_apt_requires_claim_kind")
        elif kind != _NO_KIND:
            raise ValueError("non_claim_requires_not_applicable_kind")
        normalized.append({"candidate_id": candidate_id, "status": status, "kind": kind})
    if seen != set(candidate):
        raise ValueError("assertion_span_coverage_mismatch")
    return normalized


def _validate_contract(rows: Any, candidate: dict[str, str],
                       contract: dict[str, str]) -> list[dict[str, str]]:
    if not isinstance(rows, list) or len(rows) != len(candidate):
        raise ValueError("contract_span_coverage_mismatch")
    seen: set[str] = set()
    normalized = []
    required = {"candidate_id", "status", "contract_id", "issue_kind"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("contract_annotation_schema_mismatch")
        candidate_id, status = row["candidate_id"], row["status"]
        contract_id, issue_kind = row["contract_id"], row["issue_kind"]
        if not isinstance(candidate_id, str) or candidate_id not in candidate:
            raise ValueError("unknown_candidate_id")
        if candidate_id in seen:
            raise ValueError("duplicate_contract_candidate_id")
        seen.add(candidate_id)
        if status not in CONTRACT_STATES:
            raise ValueError("unknown_contract_status")
        if status == "direct_conflict":
            if not isinstance(contract_id, str) or contract_id not in contract:
                raise ValueError("direct_conflict_requires_known_contract_id")
            if not isinstance(issue_kind, str) or issue_kind not in ISSUE_KINDS:
                raise ValueError("direct_conflict_requires_issue_kind")
        elif contract_id != _NO_CONTRACT_ID or issue_kind != _NO_KIND:
            raise ValueError("non_conflict_requires_empty_issue_fields")
        normalized.append({"candidate_id": candidate_id, "status": status,
                           "contract_id": contract_id, "issue_kind": issue_kind})
    if seen != set(candidate):
        raise ValueError("contract_span_coverage_mismatch")
    return normalized


def validate_output_v6(raw: str, candidate_spans: list[dict[str, str]],
                       contract_spans: list[dict[str, str]]) -> dict[str, Any]:
    """Validate exhaustive, unique, known-ID annotations in both dimensions."""
    candidate = _span_map(candidate_spans, "candidate")
    contract = _span_map(contract_spans, "contract")
    value = _parse(raw)
    assertion, review = value["assertion"], value["contract"]
    if not isinstance(assertion, dict) or set(assertion) != {"annotations"}:
        raise ValueError("assertion_schema_mismatch")
    if not isinstance(review, dict) or set(review) != {"annotations"}:
        raise ValueError("contract_schema_mismatch")
    return {
        "assertion_annotations": _validate_assertion(assertion["annotations"], candidate),
        "contract_annotations": _validate_contract(review["annotations"], candidate, contract),
    }


def normalize_output_v6(raw: str, candidate_spans: list[dict[str, str]],
                        contract_spans: list[dict[str, str]]) -> dict[str, Any]:
    """Derive verdicts and emit evidence fields compatible with v5 consumers."""
    candidate = _span_map(candidate_spans, "candidate")
    contract = _span_map(contract_spans, "contract")
    annotations = validate_output_v6(raw, candidate_spans, contract_spans)
    assertion_rows = annotations["assertion_annotations"]
    contract_rows = annotations["contract_annotations"]
    claims = [{"candidate": {"id": row["candidate_id"],
                              "text": candidate[row["candidate_id"]]},
               "kind": row["kind"]}
              for row in assertion_rows if row["status"] == "truth_apt"]
    assertion_uncertain = [{"id": row["candidate_id"],
                            "text": candidate[row["candidate_id"]]}
                           for row in assertion_rows if row["status"] == "uncertain"]
    assertion_verdict = ("contains_claims" if claims else
                         "uncertain" if assertion_uncertain else "claim_free")
    all_issues = [{"candidate": {"id": row["candidate_id"],
                                 "text": candidate[row["candidate_id"]]},
                   "contract": {"id": row["contract_id"],
                                "text": contract[row["contract_id"]]},
                   "kind": row["issue_kind"]}
                  for row in contract_rows if row["status"] == "direct_conflict"]
    # Existing v5 consumers expect a single issue pair. The exhaustive raw
    # annotations and every pair remain in source_coverage; expose the first
    # source-order pair as the v5-compatible representative.
    issues = all_issues[:1]
    contract_uncertain = [{"id": row["candidate_id"],
                           "text": candidate[row["candidate_id"]]}
                          for row in contract_rows if row["status"] == "uncertain"]
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
            "issues": issues,
            "uncertain": contract_uncertain,
            "verdict_reason_code": {
                "compatible": "no_explicit_direct_contract_conflict",
                "incompatible": "explicit_direct_contract_conflict_found",
                "uncertain": "direct_conflict_boundary_unclear",
            }[contract_verdict],
        },
        "source_coverage": annotations,
        "all_direct_conflicts": all_issues,
    }
