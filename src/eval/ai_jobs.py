"""Frozen, measured AI evaluation calls with private outputs and bounded spend."""
import copy
import json
import os
import re
from uuid import uuid4

from ..config import config_fingerprint
from ..runtime.provider import build_provider
from ..runtime.call_ledger import call_component, summarize_ledger
from ..util import artifacts_path, artifacts_root, repo_root, sha256_obj, write_json
from .run_manifest import source_tree_fingerprint


_RATE_KEYS = ("prompt", "completion", "cache_read", "cache_write")


def _configured_or_published_rates(model_cfg):
    """Resolve a frozen tariff without knowing a provider/model name.

    A caller may bind an explicit four-rate tariff in the model config.  The
    runtime table remains a convenience for already audited endpoints, while
    arbitrary OpenAI-compatible endpoints must bring their own provenance.
    """
    explicit = model_cfg.get("pricing_override")
    if explicit is None:
        explicit = model_cfg.get("pricing")
    if isinstance(explicit, dict) and all(key in explicit for key in _RATE_KEYS):
        rates = {key: explicit[key] for key in _RATE_KEYS}
        if all(type(value) in (int, float) and float(value) >= 0 for value in rates.values()):
            return rates
    from ..runtime.provider import published_price_for_endpoint
    return published_price_for_endpoint(model_cfg.get("model_id"), model_cfg.get("base_url", ""))


def decode_json_response(content):
    """Decode one structured JSON response with bounded endpoint-noise repair.

    The evaluator never searches arbitrary prose for a JSON object.  When a
    response starts with a JSON object/array but appends a second object or
    explanatory tail, the first complete value is accepted; this is the
    smallest repair needed for free endpoints that concatenate retries.  A
    prose prefix, truncated value, or malformed first value still fails.
    """
    if not isinstance(content, str):
        raise ValueError("AI evaluation response must be text")
    value = content.strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[len("```json\n"):-len("\n```")].strip()
    elif value.startswith("```\n") and value.endswith("\n```"):
        value = value[len("```\n"):-len("\n```")].strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError as first_error:
        # Some free endpoints emit non-standard Unicode/control whitespace
        # around otherwise valid JSON. Normalize whitespace only while
        # outside JSON strings; never add delimiters, truncate output, or
        # coerce a partial object.
        cleaned_chars, in_string, escaped = [], False, False
        for char in value.lstrip("\ufeff"):
            if in_string:
                cleaned_chars.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                    cleaned_chars.append(char)
                elif char.isspace() or ord(char) < 0x20:
                    cleaned_chars.append(" ")
                else:
                    cleaned_chars.append(char)
        cleaned = "".join(cleaned_chars)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # SiliconFlow/Qwen occasionally concatenates a second JSON value
            # after a valid reviewer object. Decode only the first complete
            # value; do not search inside prose or fabricate delimiters.
            try:
                stripped = cleaned.lstrip()
                first, _end = json.JSONDecoder().raw_decode(stripped)
            except json.JSONDecodeError:
                raise first_error
            if stripped[:1] not in "[{":
                raise first_error
            return first


def effective_config(cfg, tokens, *, retries_per_call=1):
    result = copy.deepcopy(cfg)
    models = result.setdefault("models", {})
    role = "judge" if (models.get("judge") or {}).get("provider") == "http" else "task"
    model = copy.deepcopy(models.get(role) or {})
    if model.get("provider") != "http" or not model.get("model_id"):
        raise ValueError("AI annotation requires an explicitly configured HTTP task/judge model")
    model.update(cache=False, max_retries=retries_per_call, max_tokens=tokens)
    model.pop("structured_output", None)
    model.setdefault("extra_body", {}).pop("response_format", None)
    models["task"] = model
    result.setdefault("runtime", {})["provider"] = "http"
    return result, role


def _validate_instruction_protocol_metadata(metadata, *, tokens=None):
    required = {"rubric_version", "rubric_sha256"}
    optional = {"reference_format_sha256", "protocol_hash", "splitter_hash"}
    v5_fields = {"protocol_schema", "normalizer_version", "max_output_tokens"}
    v6_fields = v5_fields | {"reference_format_template_sha256"}
    v7_fields = v5_fields | {"response_format_template_sha256", "normalizer_source_hash"}
    v8_fields = v5_fields | {"response_format_template_sha256", "normalizer_source_hash"}
    v7_optional = {"protocol_hash", "splitter_hash"}
    if (not isinstance(metadata, dict) or not required <= set(metadata)
            or not isinstance(metadata.get("rubric_version"), str)
            or not metadata["rubric_version"].strip()):
        return False
    if metadata["rubric_version"] == "v5":
        if set(metadata) != required | optional | v5_fields:
            return False
        if (metadata.get("protocol_schema") != "instruction-reviewer-reference-protocol-v5"
                or metadata.get("normalizer_version") != "instruction-reference-normalizer-v5"
                or type(metadata.get("max_output_tokens")) is not int
                or metadata["max_output_tokens"] != 1024
                or type(tokens) is not int or tokens != metadata["max_output_tokens"]):
            return False
    elif metadata["rubric_version"] == "v6":
        # v6 replaced v5's reference_format_sha256 with the per-template
        # binding reference_format_template_sha256. Keep the exact field set
        # aligned with rubric_v6.protocol_metadata(); requiring both names
        # makes every valid v6 reference plan fail before dispatch.
        v6_optional = {"protocol_hash", "splitter_hash"}
        # v6 binds the per-template digest.  The former v5-style
        # ``reference_format_sha256`` alias is deliberately rejected so a
        # resealed legacy plan cannot masquerade as a current v6 protocol.
        if set(metadata) - (required | v6_optional | v6_fields):
            return False
        if (metadata.get("protocol_schema") != "instruction-reviewer-reference-protocol-v6"
                or metadata.get("normalizer_version") != "instruction-reference-normalizer-v6"
                or type(metadata.get("max_output_tokens")) is not int
                or metadata["max_output_tokens"] != 1024
                or type(tokens) is not int or tokens != metadata["max_output_tokens"]):
            return False
    elif metadata["rubric_version"] == "v7":
        if set(metadata) != required | v7_optional | v7_fields:
            return False
        if (metadata.get("protocol_schema") != "instruction-reviewer-reference-protocol-v7"
                or metadata.get("normalizer_version") != "instruction-reference-normalizer-v7"
                or type(metadata.get("max_output_tokens")) is not int
                or metadata["max_output_tokens"] != 1024
                or type(tokens) is not int or tokens != metadata["max_output_tokens"]):
            return False
    elif metadata["rubric_version"] == "instruction-reviewer-reference-v8":
        if set(metadata) != required | v7_optional | v8_fields:
            return False
        if (metadata.get("protocol_schema") != "instruction-reviewer-reference-protocol-v8"
                or metadata.get("normalizer_version") != "instruction-reference-normalizer-v8"
                or type(metadata.get("max_output_tokens")) is not int
                or metadata["max_output_tokens"] != 2048
                or type(tokens) is not int or tokens != metadata["max_output_tokens"]):
            return False
    elif set(metadata) - required - optional:
        return False
    hashes = required - {"rubric_version"} | (set(metadata) & optional)
    if metadata["rubric_version"] in {"v6", "v7", "instruction-reviewer-reference-v8"}:
        hashes.add("reference_format_template_sha256" if metadata["rubric_version"] == "v6"
                   else "response_format_template_sha256")
    if metadata["rubric_version"] in {"v7", "instruction-reviewer-reference-v8"}:
        hashes.add("normalizer_source_hash")
    return all(isinstance(metadata.get(name), str)
               and re.fullmatch(r"[0-9a-f]{64}", metadata[name])
               for name in hashes)


def _validate_instruction_judge_v8_dynamic_metadata(metadata, *, tokens=None, inputs=None):
    """Allow dynamic per-case v8 schemas only with a frozen host-validation manifest."""
    required = {
        "protocol_version", "target_protocol_sha256", "reference_protocol_hash",
        "response_mode", "parser", "dynamic_schema_manifest_sha256",
        "host_validation", "fail_closed",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        return False
    hashes = ("target_protocol_sha256", "reference_protocol_hash",
              "dynamic_schema_manifest_sha256")
    return (
        isinstance(inputs, dict)
        and inputs.get("target_protocol_version") == metadata.get("protocol_version")
        and inputs.get("target_protocol_sha256") == metadata.get("target_protocol_sha256")
        and inputs.get("reference_protocol_hash") == metadata.get("reference_protocol_hash")
        and inputs.get("target_transport") == metadata.get("response_mode")
        and inputs.get("dynamic_schema_manifest_sha256")
            == metadata.get("dynamic_schema_manifest_sha256")
        and metadata.get("protocol_version") == "instruction-review-v8-claim-extraction"
        and metadata.get("response_mode")
            == "json_object_per_case_dynamic_v8_schema_host_validated"
        and metadata.get("parser") == "parse_prompt_grounding_review_v8"
        and metadata.get("host_validation")
            == "per_case_exact_v8_schema_and_source_quote_validation"
        and metadata.get("fail_closed") is True
        and type(tokens) is int and tokens == 2048
        and all(isinstance(metadata.get(name), str)
                and re.fullmatch(r"[0-9a-f]{64}", metadata[name])
                for name in hashes)
    )


def _strict_json_schema_format(value):
    """Return whether a provider format is a complete strict JSON Schema wrapper."""
    if not isinstance(value, dict) or set(value) != {"type", "json_schema"}:
        return False
    wrapper = value.get("json_schema")
    return (value.get("type") == "json_schema"
            and isinstance(wrapper, dict)
            and set(wrapper) == {"name", "strict", "schema"}
            and isinstance(wrapper.get("name"), str) and bool(wrapper["name"].strip())
            and wrapper.get("strict") is True
            and isinstance(wrapper.get("schema"), dict))


def _dynamic_v9_case_manifest(inputs):
    """Validate and return the canonical per-case strict-schema manifest."""
    if not isinstance(inputs, dict) or not isinstance(inputs.get("cases"), list):
        return None
    cases = inputs["cases"]
    if not cases:
        return None
    manifest = []
    seen = set()
    for row in cases:
        if not isinstance(row, dict):
            return None
        case_id = row.get("case_id")
        response_format = row.get("target_response_format")
        response_sha = row.get("target_response_format_sha256")
        candidate_sha = row.get("target_candidate_spans_sha256")
        contract_sha = row.get("target_contract_spans_sha256")
        if (not isinstance(case_id, str) or not case_id or case_id in seen
                or not _strict_json_schema_format(response_format)
                or not isinstance(response_sha, str)
                or response_sha != sha256_obj(response_format)
                or not isinstance(candidate_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha)
                or not isinstance(contract_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", contract_sha)):
            return None
        seen.add(case_id)
        manifest.append({"case_id": case_id,
            "response_format_sha256": response_sha,
            "candidate_spans_sha256": candidate_sha,
            "contract_spans_sha256": contract_sha})
    return manifest


def _validate_instruction_judge_v9_per_call_metadata(metadata, *, tokens=None,
                                                       inputs=None, max_calls=None):
    """Require each frozen v9 case schema to be strict, hashed, and host-bound."""
    required = {
        "protocol_version", "target_protocol_sha256", "reference_protocol_hash",
        "response_mode", "parser", "dynamic_schema_manifest_sha256",
        "host_validation", "fail_closed",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        return False
    manifest = _dynamic_v9_case_manifest(inputs)
    hashes = ("target_protocol_sha256", "reference_protocol_hash",
              "dynamic_schema_manifest_sha256")
    expected_manifest_sha = sha256_obj(manifest) if manifest is not None else None
    return (
        manifest is not None
        and isinstance(inputs, dict)
        and inputs.get("target_protocol_version") == metadata.get("protocol_version")
        and inputs.get("target_protocol_sha256") == metadata.get("target_protocol_sha256")
        and inputs.get("reference_protocol_hash") == metadata.get("reference_protocol_hash")
        and inputs.get("target_transport") == metadata.get("response_mode")
        and metadata.get("dynamic_schema_manifest_sha256") == expected_manifest_sha
        and metadata.get("protocol_version")
            == "instruction-review-v9-nested-json-claim-extraction"
        and metadata.get("response_mode")
            == "json_schema_per_case_dynamic_v9_host_validated"
        and metadata.get("parser") == "parse_prompt_grounding_review_v8"
        and metadata.get("host_validation")
            == "per_case_exact_v8_schema_and_source_quote_validation"
        and metadata.get("fail_closed") is True
        and type(tokens) is int and tokens == 2048
        and type(max_calls) is int and max_calls == len(manifest)
        and all(isinstance(metadata.get(name), str)
                and re.fullmatch(r"[0-9a-f]{64}", metadata[name])
                for name in hashes)
    )


def _dynamic_v10_case_manifest(inputs):
    """Validate exhaustive atomic-span target schemas for instruction-review v10."""
    if not isinstance(inputs, dict) or not isinstance(inputs.get("cases"), list):
        return None
    manifest = []
    seen = set()
    for row in inputs["cases"]:
        if not isinstance(row, dict):
            return None
        case_id = row.get("case_id")
        response_format = row.get("target_response_format")
        response_sha = row.get("target_response_format_sha256")
        candidate_sha = row.get("target_candidate_spans_sha256")
        contract_sha = row.get("target_contract_spans_sha256")
        if (not isinstance(case_id, str) or not case_id or case_id in seen
                or not _strict_json_schema_format(response_format)
                or response_sha != sha256_obj(response_format)
                or not isinstance(candidate_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha)
                or not isinstance(contract_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", contract_sha)):
            return None
        seen.add(case_id)
        manifest.append({"case_id": case_id,
            "response_format_sha256": response_sha,
            "candidate_spans_sha256": candidate_sha,
            "contract_spans_sha256": contract_sha})
    return manifest or None


def _validate_instruction_judge_v10_per_call_metadata(metadata, *, tokens=None,
        inputs=None, max_calls=None):
    """Require v10 atomic-state protocol, contract, and each dynamic schema to be frozen."""
    required = {
        "protocol_version", "target_protocol_sha256", "reference_protocol_hash",
        "contract_id", "contract_sha256", "response_mode", "parser",
        "dynamic_schema_manifest_sha256", "host_validation", "fail_closed",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        return False
    manifest = _dynamic_v10_case_manifest(inputs)
    hashes = ("target_protocol_sha256", "reference_protocol_hash",
              "contract_sha256", "dynamic_schema_manifest_sha256")
    if manifest is None or not isinstance(inputs, dict):
        return False
    try:
        from ..evolution.instruction_review_v10 import (
            PROMPT_GROUNDING_V10_MAX_TOKENS, PROMPT_GROUNDING_V10_VERSION,
        )
    except (ImportError, AttributeError):
        return False
    contract = inputs.get("contract")
    expected_manifest_sha = sha256_obj(manifest)
    return (
        inputs.get("target_protocol_version") == metadata.get("protocol_version")
        and inputs.get("target_protocol_sha256") == metadata.get("target_protocol_sha256")
        and inputs.get("reference_protocol_hash") == metadata.get("reference_protocol_hash")
        and inputs.get("target_transport") == metadata.get("response_mode")
        and inputs.get("dynamic_schema_manifest_sha256") == expected_manifest_sha
        and metadata.get("dynamic_schema_manifest_sha256") == expected_manifest_sha
        and isinstance(contract, dict)
        and contract.get("id") == metadata.get("contract_id")
        and contract.get("sha256") == metadata.get("contract_sha256")
        and metadata.get("protocol_version") == PROMPT_GROUNDING_V10_VERSION
        and metadata.get("response_mode")
            == "json_schema_per_case_dynamic_v10_atomic_span_states"
        and metadata.get("parser") == "parse_prompt_grounding_review_v10"
        and metadata.get("host_validation")
            == "per_case_exact_v6_schema_and_exhaustive_atomic_span_states"
        and metadata.get("fail_closed") is True
        and type(tokens) is int and tokens == PROMPT_GROUNDING_V10_MAX_TOKENS
        and type(max_calls) is int and max_calls == len(manifest)
        and all(isinstance(metadata.get(name), str)
                and re.fullmatch(r"[0-9a-f]{64}", metadata[name])
                for name in hashes)
    )


def _validate_instruction_judge_per_call_metadata(metadata, *, tokens=None,
        inputs=None, max_calls=None):
    version = metadata.get("protocol_version") if isinstance(metadata, dict) else None
    if version == "instruction-review-v10-atomic-span-states":
        return _validate_instruction_judge_v10_per_call_metadata(metadata,
            tokens=tokens, inputs=inputs, max_calls=max_calls)
    return _validate_instruction_judge_v9_per_call_metadata(metadata,
        tokens=tokens, inputs=inputs, max_calls=max_calls)


def build_plan(kind, inputs, cfg, *, max_calls, tokens=1200, max_input_chars=60000, max_cost_usd=0.25,
               response_format=None, protocol_metadata=None, zero_price_only=False):
    if kind not in {"rubric_author", "open_annotations", "grounding_reference", "grounding_judge", "grounding_adjudication",
                    "clinifact_reference",
                    "instruction_reference", "instruction_judge"}:
        raise ValueError("unsupported AI evaluation job")
    if not isinstance(inputs, (dict, list)) or not inputs:
        raise ValueError("AI evaluation requires nonempty frozen inputs")
    if type(zero_price_only) is not bool:
        raise ValueError("zero_price_only must be a boolean protocol choice")
    if (type(max_calls) is not int or not 1 <= max_calls <= 500 or type(tokens) is not int or not 220 <= tokens <= 4096
            or type(max_input_chars) is not int or not 1000 <= max_input_chars <= 500000
            or type(max_cost_usd) not in (int, float)
            or (not zero_price_only and not 0 < max_cost_usd <= 20)
            or (zero_price_only and (kind != "instruction_reference" or max_cost_usd != 0.0))):
        raise ValueError("invalid AI annotation request/token/input/spend budget")
    # Grounding calibration is a measured, per-item job. Allow bounded
    # transport recovery inside that same item; other AI jobs keep their
    # existing single-attempt contract.
    retries_per_call = 2 if kind in {"grounding_judge", "grounding_reference", "clinifact_reference",
                                     "rubric_author", "open_annotations"} else 1
    effective, role = effective_config(cfg, tokens, retries_per_call=retries_per_call)
    zero_price_rates = None
    if zero_price_only:
        model_cfg = effective["models"]["task"]
        zero_price_rates = _configured_or_published_rates(model_cfg)
        expected_rate_keys = {"prompt", "completion", "cache_read", "cache_write"}
        if (not isinstance(zero_price_rates, dict)
                or set(zero_price_rates) != expected_rate_keys
                or any(type(rate) not in (int, float) or rate != 0.0
                       for rate in zero_price_rates.values())):
            raise ValueError("zero-price-only job requires known exact-zero rates for every token class")
    body = {"schema": "ai-evaluation-plan-v1", "kind": kind, "inputs_sha256": sha256_obj(inputs),
        "config_sha256": config_fingerprint(effective), "source_tree_sha256": source_tree_fingerprint()["sha256"],
        "model_id": effective["models"]["task"]["model_id"], "model_role": role,
        "max_calls": max_calls, "max_tokens": tokens, "max_input_chars": max_input_chars,
        "max_retries_per_call": retries_per_call,
        "max_physical_attempts": max_calls * retries_per_call,
        "max_cost_usd": max_cost_usd, "model_calls": 0, "annotation_origin": "ai"}
    resolved_rates = _configured_or_published_rates(effective["models"]["task"])
    if isinstance(resolved_rates, dict):
        body["pricing_rates_sha256"] = sha256_obj(resolved_rates)
        body["pricing_is_exact_zero"] = all(
            type(resolved_rates[key]) in (int, float)
            and float(resolved_rates[key]) == 0.0 for key in _RATE_KEYS)
        pricing_metadata = effective["models"]["task"].get("pricing")
        if not isinstance(pricing_metadata, dict):
            from ..runtime.provider import published_tariff_metadata
            pricing_metadata = published_tariff_metadata(
                effective["models"]["task"].get("model_id"),
                effective["models"]["task"].get("base_url", ""))
        if isinstance(pricing_metadata, dict) and pricing_metadata:
            body["pricing_metadata_sha256"] = sha256_obj(pricing_metadata)
    if zero_price_only:
        body["zero_price_only"] = True
        body["zero_price_rates_sha256"] = sha256_obj(zero_price_rates)
        pricing_cfg = effective["models"]["task"].get("pricing")
        if isinstance(pricing_cfg, dict):
            body["zero_price_pricing_metadata_sha256"] = sha256_obj(pricing_cfg)
    if protocol_metadata is not None:
        valid_instruction_reference_metadata = (
            kind == "instruction_reference"
            and _validate_instruction_protocol_metadata(protocol_metadata, tokens=tokens))
        valid_v8_instruction_judge_metadata = (
            kind == "instruction_judge"
            and _validate_instruction_judge_v8_dynamic_metadata(
                protocol_metadata, tokens=tokens, inputs=inputs))
        valid_v9_instruction_judge_metadata = (
            kind == "instruction_judge"
            and _validate_instruction_judge_v9_per_call_metadata(
                protocol_metadata, tokens=tokens, inputs=inputs, max_calls=max_calls))
        valid_v10_instruction_judge_metadata = (
            kind == "instruction_judge"
            and _validate_instruction_judge_v10_per_call_metadata(
                protocol_metadata, tokens=tokens, inputs=inputs, max_calls=max_calls))
        if not (valid_instruction_reference_metadata
                or valid_v8_instruction_judge_metadata
                or valid_v9_instruction_judge_metadata
                or valid_v10_instruction_judge_metadata):
            raise ValueError("invalid instruction reference protocol metadata")
        body["protocol_metadata"] = dict(protocol_metadata)
    else:
        valid_v9_instruction_judge_metadata = False
        valid_v10_instruction_judge_metadata = False
    if response_format is not None:
        valid_instruction_schema = (kind in {"instruction_reference", "instruction_judge"}
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_schema")
        if valid_instruction_schema:
            schema_wrapper = response_format.get("json_schema")
            valid_instruction_schema = (
                isinstance(schema_wrapper, dict)
                and schema_wrapper.get("strict") is True
                and isinstance(schema_wrapper.get("schema"), dict))
        valid_instruction_judge_dynamic_json_mode = (
            kind == "instruction_judge"
            and response_format == {"type": "json_object"}
            and _validate_instruction_judge_v8_dynamic_metadata(
                protocol_metadata, tokens=tokens, inputs=inputs))
        valid_instruction_judge_per_call_schema_mode = (
            kind == "instruction_judge"
            and _strict_json_schema_format(response_format)
            and _validate_instruction_judge_per_call_metadata(
                protocol_metadata, tokens=tokens, inputs=inputs, max_calls=max_calls))
        valid_grounding_schema = (kind == "grounding_judge"
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and isinstance(response_format.get("json_schema"), dict)
            and response_format["json_schema"].get("strict") is True)
        valid_grounding_reference_schema = (kind == "grounding_reference"
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and isinstance(response_format.get("json_schema"), dict)
            and response_format["json_schema"].get("strict") is True)
        # Task-first open-research authoring/review jobs also use a frozen
        # strict schema.  Keeping this explicit prevents malformed JSON from
        # consuming a one-shot call while preserving the existing protocol
        # restrictions for every other job kind.
        valid_open_research_schema = (
            kind in {"rubric_author", "open_annotations"}
            and _strict_json_schema_format(response_format))
        valid_grounding_json_mode = (kind == "grounding_judge"
            and response_format == {"type": "json_object"})
        valid_grounding_reference_json_mode = (kind == "grounding_reference"
            and response_format == {"type": "json_object"})
        valid_grounding_adjudication_json_mode = (kind == "grounding_adjudication"
            and response_format == {"type": "json_object"})
        if (not (valid_instruction_schema or valid_instruction_judge_dynamic_json_mode
                 or valid_instruction_judge_per_call_schema_mode
                 or valid_grounding_schema
                 or valid_grounding_reference_schema
                 or valid_open_research_schema)
                and not valid_grounding_json_mode and not valid_grounding_reference_json_mode
                and not valid_grounding_adjudication_json_mode
                or len(json.dumps(response_format)) > 20000):
            raise ValueError("unsupported AI evaluation output schema")
        body["response_format"]=response_format
    if (valid_v9_instruction_judge_metadata or valid_v10_instruction_judge_metadata) \
            and response_format is not None:
        raise ValueError("per-case instruction schemas cannot be frozen as a global response format")
    return {**body, "plan_sha256": sha256_obj(body)}


def _frozen_plan_matches_current(expected, plan, *, allowed_legacy_source_tree_sha256=None):
    if expected == plan:
        return True
    if allowed_legacy_source_tree_sha256 is None:
        return False
    source_hash = plan.get("source_tree_sha256")
    if (not isinstance(source_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
            or allowed_legacy_source_tree_sha256 != source_hash
            or plan.get("plan_sha256") != sha256_obj({
                key: value for key, value in plan.items() if key != "plan_sha256"})):
        return False
    ignored = {"source_tree_sha256", "plan_sha256"}
    return ({key: value for key, value in expected.items() if key not in ignored}
            == {key: value for key, value in plan.items() if key not in ignored})


def _completion_diagnostics(output, max_tokens: int) -> dict:
    """Record provider stop status without changing usage or billing semantics.

    `length` is an explicit truncation signal. When a provider omits the stop
    reason, exhausting the requested completion budget is only a suspicion,
    not proof, and is recorded separately.
    """
    finish_reason = getattr(output, "finish_reason", None)
    if not isinstance(finish_reason, str):
        finish_reason = None
    completion_tokens = getattr(output, "completion_tokens", None)
    budget_exhausted = (type(completion_tokens) is int
                        and completion_tokens >= max_tokens)
    truncated = finish_reason == "length"
    return {
        "finish_reason": finish_reason,
        "output_truncated": truncated,
        "output_truncation_suspected": (
            not truncated and finish_reason is None and budget_exhausted),
    }


def execute(plan, inputs, cfg, worker, *, provider=None,
            allowed_legacy_source_tree_sha256=None):
    expected = build_plan(plan["kind"], inputs, cfg, max_calls=plan["max_calls"], tokens=plan["max_tokens"],
        max_input_chars=plan["max_input_chars"], max_cost_usd=plan["max_cost_usd"],
        response_format=plan.get("response_format"),
        protocol_metadata=plan.get("protocol_metadata"),
        zero_price_only=plan.get("zero_price_only", False))
    if not _frozen_plan_matches_current(expected, plan,
            allowed_legacy_source_tree_sha256=allowed_legacy_source_tree_sha256):
        raise ValueError("AI evaluation no longer matches its frozen plan")
    if provider is not None and (not os.environ.get("RSI_HOME") or artifacts_root().resolve() == (repo_root()/"artifacts").resolve()):
        raise ValueError("injected AI evaluation providers are test-only in an isolated RSI_HOME")
    injected = provider is not None
    root = artifacts_path("evaluation_only", "ai_annotations", "ai-" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    write_json(root / "inputs.json", inputs)
    write_json(root / "status.json", {"status": "started"})
    ledger = root / "call-ledger.jsonl"
    bindings = {"RSI_ACTIVE_EXECUTION_PLAN_SHA256": plan["plan_sha256"], "RSI_CALL_LEDGER_PATH": str(ledger),
                "RSI_CALL_LEDGER_MAX_LOGICAL": str(plan["max_calls"]),
                "RSI_CALL_LEDGER_MAX_PHYSICAL": str(plan["max_physical_attempts"])}
    before = {k: os.environ.get(k) for k in bindings}
    calls, spent, upper_spent = [], 0.0, 0.0
    try:
        os.environ.update(bindings)
        effective, _ = effective_config(cfg, plan["max_tokens"],
                                       retries_per_call=plan["max_retries_per_call"])
        provider = provider or build_provider(effective)
        if plan.get("response_format") and not injected:
            provider.extra_body["response_format"]=plan["response_format"]

        per_call_schema_mode = (
            (plan.get("protocol_metadata") or {}).get("response_mode")
            in {"json_schema_per_case_dynamic_v9_host_validated",
                "json_schema_per_case_dynamic_v10_atomic_span_states"})

        def call(role, payload, instruction, *, response_format=None):
            nonlocal spent, upper_spent
            if len(calls) >= plan["max_calls"]:
                raise ValueError("AI annotation logical request budget exhausted")
            response_format_sha256 = None
            if per_call_schema_mode:
                cases = inputs["cases"]
                if (role != "instruction_judge" or len(calls) >= len(cases)
                        or not _strict_json_schema_format(response_format)):
                    raise ValueError("versioned instruction review requires its frozen per-case strict schema")
                expected_format = cases[len(calls)]["target_response_format"]
                if response_format != expected_format:
                    raise ValueError("instruction review response schema differs from frozen case")
                response_format_sha256 = sha256_obj(response_format)
                if response_format_sha256 != cases[len(calls)]["target_response_format_sha256"]:
                    raise ValueError("instruction review response schema hash changed")
            elif response_format is not None:
                raise ValueError("per-call response formats require a frozen v9/v10 schema protocol")
            if role in {"grounding_judge", "instruction_judge"}:
                if set(payload) != {"runtime_user_message"} or not isinstance(payload["runtime_user_message"], str):
                    raise ValueError("grounding calibration requires the exact runtime user message")
                text = payload["runtime_user_message"]
            else:
                text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if len(text) + len(instruction) > plan["max_input_chars"]:
                raise ValueError("AI annotation input exceeds frozen character budget")
            # Reserve a conservative next-call ceiling from actual configured
            # price rates; unknown prices cannot quietly become zero spend.
            if not injected:
                model_cfg = effective["models"]["task"]
                rates = _configured_or_published_rates(model_cfg)
                if plan.get("zero_price_only"):
                    expected_rate_keys = {"prompt", "completion", "cache_read", "cache_write"}
                    if (plan.get("max_cost_usd") != 0.0
                            or not isinstance(rates, dict)
                            or set(rates) != expected_rate_keys
                            or any(type(rate) not in (int, float) or rate != 0.0
                                   for rate in rates.values())
                            or sha256_obj(rates) != plan.get("zero_price_rates_sha256")):
                        raise ValueError("zero-price-only job lost its exact-zero pre-dispatch tariff proof")
                call_format = response_format if response_format is not None else plan.get("response_format", {})
                format_bytes=len(json.dumps(call_format).encode("utf-8"))
                reserve = ((len((instruction + text).encode("utf-8")) + format_bytes + 128) * rates["prompt"]
                           + plan["max_tokens"] * rates["completion"]) / 1_000_000 if rates else None
                if (reserve is None or upper_spent
                        + reserve * plan["max_retries_per_call"] > plan["max_cost_usd"]):
                    raise ValueError("AI evaluation spend is unknown or exceeds frozen ceiling")
            index = len(calls)
            input_identity = ({"payload": payload,
                               "response_format_sha256": response_format_sha256}
                              if response_format_sha256 is not None else payload)
            item = {"index": index, "role": role,
                    "input_sha256": sha256_obj(input_identity), "status": "started"}
            if response_format_sha256 is not None:
                item["response_format_sha256"] = response_format_sha256
            calls.append(item)
            write_json(root / "calls.json", calls)
            with call_component("ai_evaluation"):
                call_kwargs = {"max_tokens": plan["max_tokens"]}
                if response_format is not None:
                    call_kwargs["response_format"] = response_format
                output = provider.chat_measured(instruction, text, **call_kwargs)
            cost = output.cost or {}
            usd = cost.get("usd")
            known_usd = usd if usd is not None else cost.get("known_success_usd")
            attempts = int(getattr(output, "http_attempts", 1))
            unknown_attempts = attempts - 1
            if (not injected and (attempts < 1
                    or (not plan.get("zero_price_only") and (
                        known_usd is None or attempts > plan["max_retries_per_call"]
                        or (usd is None and unknown_attempts == 0))))):
                raise ValueError("AI evaluation returned unbounded or unknown measured cost")
            unknown_reservation = (unknown_attempts * reserve) if not injected else 0.0
            cost = {**cost, "unknown_retry_attempts": unknown_attempts,
                    "unknown_retry_reservation_usd": unknown_reservation}
            completion_diagnostics = _completion_diagnostics(output, plan["max_tokens"])
            spent += float(known_usd or 0)
            upper_spent += float(known_usd or 0) + unknown_reservation
            if plan.get("zero_price_only") and (known_usd != 0.0 or unknown_attempts != 0
                    or unknown_reservation != 0.0 or cost.get("usage_complete") is not True):
                # Persist any nonzero or incomplete charge evidence before the
                # irreversible one-shot job is marked failed by the caller.
                raw_call = {"role": role, "payload": payload,
                    "system": instruction, "user": text, "raw_output": output.content,
                    "cost": cost, **completion_diagnostics}
                if response_format_sha256 is not None:
                    raw_call["response_format_sha256"] = response_format_sha256
                write_json(root / f"call-{index}.json", raw_call)
                raise ValueError("zero-price-only response measured a charge or incomplete usage")
            if upper_spent > plan["max_cost_usd"]:
                raise ValueError("AI evaluation measured spend exceeded its frozen ceiling")
            raw_call = {"role": role, "payload": payload,
                "system": instruction, "user": text, "raw_output": output.content,
                "cost": cost, **completion_diagnostics}
            if response_format_sha256 is not None:
                raw_call["response_format_sha256"] = response_format_sha256
            write_json(root / f"call-{index}.json", raw_call)
            try:
                parsed = (output.content if role in {"instruction_reference", "instruction_adjudicator", "instruction_judge"}
                          else decode_json_response(output.content))
            except (ValueError, TypeError):
                if role not in {"grounding_judge", "grounding_reference", "adjudicator",
                                "clinifact_label", "clinifact_adjudicator", "author",
                                "author_correction"}:
                    raise
                parsed = output.content  # measured invalid verdict, never a synthetic label
            item.update(status="completed", response_sha256=sha256_obj(parsed), cost=cost,
                        **completion_diagnostics)
            write_json(root / "calls.json", calls)
            return parsed, {"model_id": plan["model_id"], "review_run_id": root.name + f"-{index}",
                "rubric_sha256": sha256_obj(instruction), "blind": True,
                "other_review_outputs_consulted": role == "adjudicator", "test_only": injected}

        result = worker(call)
        if per_call_schema_mode and len(calls) != len(inputs["cases"]):
            raise ValueError("versioned instruction review did not evaluate every frozen case")
        write_json(root / "output.json", result)
        receipt = {"schema": "ai-evaluation-receipt-v1", "plan_sha256": plan["plan_sha256"],
            "inputs_sha256": sha256_obj(inputs), "output_sha256": sha256_obj(result),
            "call_count": len(calls), "measured_usd": spent,
            "cost_upper_bound_usd": upper_spent,
            # Zero-price retries can have unknown token usage even when both
            # monetary totals are zero. Monetary equality is not a usage test.
            "usage_complete": all(
                (row.get("cost") or {}).get("unknown_retry_attempts", 0) == 0
                and (row.get("cost") or {}).get("usage_complete", True) is True
                for row in calls), "test_only": injected,
            "request_accounting": summarize_ledger(ledger)}
        write_json(root / "receipt.json", receipt)
        write_json(root / "status.json", {"status": "completed"})
        return {"run_dir": str(root), "output": result, "receipt": receipt}
    except BaseException as exc:
        write_json(root / "status.json", {"status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error_type": type(exc).__name__, "call_count": len(calls), "measured_usd": spent,
            "cost_upper_bound_usd": upper_spent,
            "request_accounting": summarize_ledger(ledger)})
        raise
    finally:
        for k, value in before.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value


def verify_completed(run_dir, *, kind, allow_legacy_measurement_schema: bool = False):
    """Verify a private completed job before using its output as a reference.

    Content binding and measured accounting are checked independently; a
    test-injected provider can never acquire a real measurement identity.
    """
    from pathlib import Path
    from ..util import read_json, sha256_file
    root = Path(run_dir).resolve()
    root.relative_to(artifacts_path("evaluation_only", "ai_annotations").resolve())
    plan, inputs, output, receipt, status, calls = (
        read_json(root/name) for name in ("plan.json", "inputs.json", "output.json", "receipt.json", "status.json", "calls.json"))
    if (status.get("status") != "completed" or plan.get("kind") != kind
            or plan.get("plan_sha256") != sha256_obj({k:v for k,v in plan.items() if k!="plan_sha256"})
            or receipt.get("schema") != "ai-evaluation-receipt-v1"
            or receipt.get("plan_sha256") != plan["plan_sha256"]
            or receipt.get("inputs_sha256") != plan["inputs_sha256"]
            or sha256_obj(inputs) != plan["inputs_sha256"]
            or sha256_obj(output) != receipt.get("output_sha256")
            or receipt.get("call_count") != len(calls) or len(calls)>plan["max_calls"]
            or type(receipt.get("test_only")) is not bool):
        raise ValueError("AI completed-job identity or kind changed")
    per_call_schema_mode = (
        (plan.get("protocol_metadata") or {}).get("response_mode")
        in {"json_schema_per_case_dynamic_v9_host_validated",
            "json_schema_per_case_dynamic_v10_atomic_span_states"})
    if per_call_schema_mode:
        metadata = plan.get("protocol_metadata")
        if (kind != "instruction_judge"
                or not _validate_instruction_judge_per_call_metadata(
                    metadata, tokens=plan.get("max_tokens"), inputs=inputs,
                    max_calls=plan.get("max_calls"))
                or len(calls) != len(inputs["cases"])):
            raise ValueError("AI completed-job versioned schema protocol changed")
    spent = 0.0
    upper_spent = 0.0
    unknown_retry_attempts = 0
    all_usage_complete = True
    for index, call in enumerate(calls):
        raw = read_json(root/f"call-{index}.json")
        if per_call_schema_mode:
            case = inputs["cases"][index]
            expected_schema_sha = case["target_response_format_sha256"]
            if (raw.get("response_format_sha256") != expected_schema_sha
                    or call.get("response_format_sha256") != expected_schema_sha
                    or raw.get("role") != "instruction_judge"
                    or call.get("input_sha256") != sha256_obj({
                        "payload": raw.get("payload"),
                        "response_format_sha256": expected_schema_sha})):
                raise ValueError("AI completed-job per-call response schema binding changed")
        elif ("response_format_sha256" in raw or "response_format_sha256" in call):
            raise ValueError("AI completed-job has an unexpected per-call schema binding")
        response = raw["raw_output"]
        if raw["role"] not in {"instruction_reference", "instruction_adjudicator", "instruction_judge"}:
            try:
                response = decode_json_response(response)
            except (ValueError,TypeError):
                if raw["role"] not in {"grounding_judge", "grounding_reference", "clinifact_label",
                                       "clinifact_adjudicator"}: raise
        expected_input_sha = (sha256_obj({"payload": raw["payload"],
            "response_format_sha256": raw["response_format_sha256"]})
            if per_call_schema_mode else sha256_obj(raw["payload"]))
        if (call.get("index") != index or call.get("status") != "completed"
                or call.get("role") != raw["role"] or call.get("input_sha256") != expected_input_sha
                or call.get("response_sha256") != sha256_obj(response) or call.get("cost") != raw.get("cost")):
            raise ValueError("AI completed-job raw response identity changed")
        diagnostic_fields = ("finish_reason", "output_truncated",
                            "output_truncation_suspected")
        if any(field in raw or field in call for field in diagnostic_fields):
            cost_tokens = (raw.get("cost") or {}).get("completion_tokens")
            finish_reason = raw.get("finish_reason")
            expected_truncated = finish_reason == "length"
            expected_suspected = (not expected_truncated and finish_reason is None
                and type(cost_tokens) is int and cost_tokens >= plan["max_tokens"])
            if (any(field not in raw or field not in call for field in diagnostic_fields)
                    or any(raw.get(field) != call.get(field) for field in diagnostic_fields)
                    or raw.get("output_truncated") is not expected_truncated
                    or raw.get("output_truncation_suspected") is not expected_suspected
                    or (finish_reason is not None and not isinstance(finish_reason, str))):
                raise ValueError("AI completed-job finish-reason diagnostics changed")
        cost = raw.get("cost") or {}
        known_usd = cost.get("usd")
        if known_usd is None:
            known_usd = cost.get("known_success_usd")
        if known_usd is None and not receipt.get("test_only"):
            raise ValueError("AI completed-job known success cost is missing")
        unknown = cost.get("unknown_retry_attempts", 0)
        reservation = cost.get("unknown_retry_reservation_usd", 0.0)
        pricing_metadata = cost.get("pricing_metadata")
        pricing_metadata_matches = (
            isinstance(pricing_metadata, dict)
            and sha256_obj(pricing_metadata) == plan.get("pricing_metadata_sha256")
            and all(cost.get(key) == value for key, value in pricing_metadata.items()
                    if str(key).startswith("pricing_")))
        verified_zero_tariff = (
            plan.get("pricing_is_exact_zero") is True
            and known_usd == 0
            and cost.get("usd") in (0, None)
            and cost.get("pricing_rates_sha256") ==
                (plan.get("pricing_rates_sha256") or plan.get("zero_price_rates_sha256"))
            and cost.get("pricing_metadata_sha256") == plan.get("pricing_metadata_sha256")
            and pricing_metadata_matches)
        if (type(unknown) is not int or unknown < 0
                or unknown >= plan.get("max_retries_per_call", 1)
                or not isinstance(reservation, (int, float)) or reservation < 0
                or unknown == 0 and reservation != 0
                or unknown > 0 and reservation == 0 and not verified_zero_tariff):
            raise ValueError("AI completed-job retry reservation is malformed")
        unknown_retry_attempts += unknown
        all_usage_complete = all_usage_complete and cost.get("usage_complete", True) is True
        spent += float(known_usd or 0)
        upper_spent += float(known_usd or 0) + float(reservation)
    accounting = summarize_ledger(root/"call-ledger.jsonl")
    # Older immutable jobs predate the ``physical_timeouts`` field added to
    # ``summarize_ledger``.  They remain auditable when the receipt contains
    # the exact legacy subset and the only current extra is that derived
    # zero-timeout counter.  The compatibility switch is opt-in so current
    # callers retain strict schema equality.
    accounting_matches = accounting == receipt.get("request_accounting")
    if (allow_legacy_measurement_schema and not accounting_matches
            and isinstance(receipt.get("request_accounting"), dict)):
        current_without_timeout = dict(accounting)
        current_without_timeout.pop("physical_timeouts", None)
        accounting_matches = (current_without_timeout
                              == receipt.get("request_accounting"))
    if (abs(spent-receipt.get("measured_usd",-1))>1e-9
            or abs(upper_spent-receipt.get("cost_upper_bound_usd", receipt.get("measured_usd",-1)))>1e-9
            or receipt.get("usage_complete", True) is not (unknown_retry_attempts == 0 and all_usage_complete)
            or upper_spent > plan.get("max_cost_usd", 0)
            or not accounting_matches):
        raise ValueError("AI completed-job measurement changed")
    physical = accounting.get("physical_http_attempts")
    statuses = accounting.get("physical_status_counts") or {}
    if not receipt.get("test_only") and (accounting.get("logical_incomplete") or accounting.get("physical_incomplete")
            or accounting.get("logical_operations") != len(calls)
            or physical != len(calls) + unknown_retry_attempts
            or physical > plan.get("max_physical_attempts", plan["max_calls"])
            or accounting.get("physical_budget_consumed_or_reserved") > plan.get("max_physical_attempts", plan["max_calls"])
            or accounting.get("logical_status_counts") != {"success":len(calls)}
            or statuses.get("200") != len(calls)
            or any(code not in {"200", "None", "408", "409", "429", "500", "502", "503", "504"}
                   for code in statuses)):
        raise ValueError("AI completed-job physical/logical accounting is incomplete")
    return {"root":root,"plan":plan,"inputs":inputs,"output":output,"receipt":receipt,
            "receipt_file_sha256":sha256_file(root/"receipt.json")}
