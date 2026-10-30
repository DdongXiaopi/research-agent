"""Zero-call identity/config gate for G06's synthetic reviewer calibration.

This module validates frozen reviewer schedules and a target calibration plan.
It establishes explicit model-family separation only; it does not establish
natural-distribution validity, account access, or production-gate authority.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..util import sha256_file, sha256_obj


CONFIG_SCHEMA = "g06-instruction-reviewer-runtime-config-v1"
FAMILY_REGISTRY = {
    "Qwen/Qwen3-8B": "qwen",
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "THUDM/GLM-4-9B-0414": "glm",
    "XingChenAGI/Xing4.0-29B": "xing",
}
DEFAULT_CONFIG_PATH = Path("configs/g06-instruction-reviewer.yaml")


def family_registry_sha256() -> str:
    return sha256_obj(FAMILY_REGISTRY)


def model_family(model_id: str) -> str:
    try:
        return FAMILY_REGISTRY[model_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unregistered G06 model family: {model_id!r}") from exc


def load_g06_config(model_id: str, workspace: str | Path,
                    config_path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[dict, dict]:
    """Return the runtime config and provenance snapshot for one registered model."""
    model_family(model_id)
    root = Path(workspace).resolve()
    lexical_path = root / config_path
    if lexical_path.is_symlink():
        raise ValueError("G06 config cannot be a symlink")
    path = lexical_path.resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("G06 config must be a regular file inside the project workspace")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or raw.get("schema") != CONFIG_SCHEMA
            or set(raw) != {"schema", "endpoint", "api_key_env", "timeout_seconds",
                "max_retries", "temperature", "max_tokens", "enable_thinking", "models"}
            or raw.get("endpoint") != "https://api.siliconflow.cn/v1"
            or raw.get("api_key_env") != "SILICONFLOW_API_KEY"
            or raw.get("timeout_seconds") != 240
            or raw.get("max_retries") != 1
            or raw.get("temperature") != 0
            or raw.get("max_tokens") != 900
            or raw.get("enable_thinking") is not False
            or raw.get("models") != {key: {"family": value} for key, value in FAMILY_REGISTRY.items()}):
        raise ValueError("G06 runtime config is not the frozen 240s/one-retry family config")
    cfg = {"models": {"task": {"provider": "http", "model_id": model_id,
            "base_url": raw["endpoint"], "api_key_env": raw["api_key_env"],
            "timeout": raw["timeout_seconds"], "max_retries": raw["max_retries"],
            "temperature": raw["temperature"], "max_tokens": raw["max_tokens"],
            "enable_thinking": raw["enable_thinking"]}},
        "runtime": {"provider": "http"}}
    path_rel = path.relative_to(root).as_posix()
    provenance = {"config_path": path_rel, "config_file_sha256": sha256_file(path),
        "effective_config_sha256": sha256_obj(cfg), "timeout_seconds": 240,
        "max_retries": 1, "model_id": model_id, "model_family": model_family(model_id),
        "family_registry_sha256": family_registry_sha256()}
    return cfg, provenance


def validate_config_binding(cfg: dict, provenance: dict, model_id: str,
                            workspace: str | Path) -> None:
    """Fail closed if a schedule's config or config file has drifted."""
    if not isinstance(provenance, dict) or not isinstance(cfg, dict):
        raise ValueError("G06 schedule lacks a frozen config snapshot")
    expected_cfg, expected = load_g06_config(model_id, workspace,
        provenance.get("config_path", DEFAULT_CONFIG_PATH))
    if (cfg != expected_cfg or provenance != expected
            or sha256_obj(cfg) != provenance.get("effective_config_sha256")):
        raise ValueError("G06 frozen config binding changed")


def _validate_plan(plan: dict, expected_kind: str) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != "ai-evaluation-plan-v1":
        raise ValueError("target calibration plan is missing or malformed")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (plan.get("plan_sha256") != sha256_obj(body)
            or plan.get("kind") != expected_kind
            or not isinstance(plan.get("model_id"), str)
            or len(plan.get("inputs_sha256", "")) != 64
            or len(plan.get("config_sha256", "")) != 64
            or plan.get("model_calls") != 0):
        raise ValueError("target calibration plan identity or hash is invalid")


def validate_g06_pair(packet: dict, schedule_a: dict, schedule_b: dict,
                      target_plan: dict, *, workspace: str | Path) -> dict:
    """Validate A/B plus target family separation without executing model calls."""
    from . import instruction_reviewer_reference_shards as shards
    from .instruction_reviewer_cohort_reference import (
        COHORT_V5_SCHEMA, COHORT_V6_SCHEMA, COHORT_V7_SCHEMA,
        RUBRIC_V2_VERSION, RUBRIC_V3_VERSION, RUBRIC_V4_VERSION,
            RUBRIC_V5_VERSION, RUBRIC_V6_VERSION, RUBRIC_V7_VERSION,
    )
    from .instruction_reviewer_cohort_v8 import SCHEMA as COHORT_V8_SCHEMA
    from .instruction_reviewer_cohort_v9 import SCHEMA as COHORT_V9_SCHEMA
    from .instruction_reviewer_cohort_v10 import SCHEMA as COHORT_V10_SCHEMA
    from .instruction_reviewer_cohort_v11 import SCHEMA as COHORT_V11_SCHEMA
    from .instruction_reviewer_cohort_v12 import SCHEMA as COHORT_V12_SCHEMA
    from .instruction_reviewer_cohort_v13 import SCHEMA as COHORT_V13_SCHEMA
    from .instruction_reviewer_cohort_v14 import SCHEMA as COHORT_V14_SCHEMA
    from .instruction_reviewer_rubric_v2 import RUBRIC_V2
    from .instruction_reviewer_rubric_v3 import (
        RUBRIC_V3, protocol_hash as v3_protocol_hash,
        splitter_hash as v3_splitter_hash,
    )
    from .instruction_reviewer_rubric_v4 import (
        REFERENCE_FORMAT_V4, RUBRIC_V4,
        protocol_hash as v4_protocol_hash,
        splitter_hash as v4_splitter_hash,
    )
    from .instruction_reviewer_rubric_v5 import (
        NORMALIZER_VERSION_V5, PROTOCOL_SCHEMA_V5,
        REFERENCE_FORMAT_V5, REFERENCE_MAX_OUTPUT_TOKENS, RUBRIC_V5,
        protocol_hash as v5_protocol_hash,
        splitter_hash as v5_splitter_hash,
    )
    from .instruction_reviewer_rubric_v6 import (
        NORMALIZER_VERSION_V6, PROTOCOL_SCHEMA_V6, REFERENCE_FORMAT_V6,
        REFERENCE_MAX_OUTPUT_TOKENS as REFERENCE_MAX_OUTPUT_TOKENS_V6, RUBRIC_V6,
        protocol_hash as v6_protocol_hash, splitter_hash as v6_splitter_hash,
    )
    from .instruction_reviewer_rubric_v7 import (
        NORMALIZER_VERSION_V7, PROTOCOL_SCHEMA_V7, RESPONSE_FORMAT_TEMPLATE_V7,
        REFERENCE_MAX_OUTPUT_TOKENS as REFERENCE_MAX_OUTPUT_TOKENS_V7, RUBRIC_V7,
        normalizer_source_hash as v7_normalizer_source_hash,
        protocol_hash as v7_protocol_hash, splitter_hash as v7_splitter_hash,
    )
    from .instruction_reviewer_rubric_v8 import (
        NORMALIZER_VERSION_V8, PROTOCOL_SCHEMA_V8, RESPONSE_FORMAT_TEMPLATE_V8,
        REFERENCE_MAX_OUTPUT_TOKENS as REFERENCE_MAX_OUTPUT_TOKENS_V8, RUBRIC_V8,
        normalizer_source_hash_v8 as v8_normalizer_source_hash,
        protocol_hash_v8 as v8_protocol_hash, splitter_hash as v8_splitter_hash,
    )

    shards.validate_shard_schedule(packet, schedule_a, "reviewer_a")
    shards.validate_shard_schedule(packet, schedule_b, "reviewer_b")
    _validate_plan(target_plan, "instruction_judge")
    if packet.get("schema") == COHORT_V14_SCHEMA:
        from ..evolution.instruction_review_v10 import prompt_grounding_protocol_v10
        from ..evolution.task_contracts import CLINIFACT_CONTRACT
        from .instruction_reviewer_rubric_v6 import protocol_hash as reference_v6_protocol_hash
        expected_target_protocol = prompt_grounding_protocol_v10(CLINIFACT_CONTRACT)
        target_protocol = packet.get("target_protocol")
        if (not isinstance(target_protocol, dict)
                or target_protocol != expected_target_protocol
                or target_protocol.get("version") != "instruction-review-v10-atomic-span-states"
                or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}):
            raise ValueError("v14 target-v10 prompt protocol differs from its frozen identity")
        target_schema_manifest = [{
            "case_id": row.get("case_id"),
            "response_format_sha256": row.get("target_dynamic_schema_sha256"),
            "candidate_spans_sha256": row.get("target_candidate_spans_sha256"),
            "contract_spans_sha256": row.get("target_contract_spans_sha256"),
        } for row in packet.get("reference_packet_manifest", [])]
        if (len(target_schema_manifest) != len(packet.get("cases", []))
                or any(not isinstance(row["case_id"], str)
                    or any(not isinstance(row[field], str) or len(row[field]) != 64
                        for field in ("response_format_sha256", "candidate_spans_sha256",
                                      "contract_spans_sha256"))
                    for row in target_schema_manifest)):
            raise ValueError("v14 target schema manifest is incomplete")
        expected_metadata = {
            "protocol_version": expected_target_protocol["version"],
            "target_protocol_sha256": sha256_obj(expected_target_protocol),
            "reference_protocol_hash": reference_v6_protocol_hash(),
            "contract_id": CLINIFACT_CONTRACT.contract_id,
            "contract_sha256": CLINIFACT_CONTRACT.sha256,
            "response_mode": "json_schema_per_case_dynamic_v10_atomic_span_states",
            "parser": "parse_prompt_grounding_review_v10",
            "dynamic_schema_manifest_sha256": sha256_obj(target_schema_manifest),
            "host_validation": "per_case_exact_v6_schema_and_exhaustive_atomic_span_states",
            "fail_closed": True,
        }
        if target_plan.get("protocol_metadata") != expected_metadata:
            raise ValueError("v14 target plan protocol metadata does not bind the frozen v10 harness")
    if packet.get("schema") in {COHORT_V11_SCHEMA, COHORT_V12_SCHEMA, COHORT_V13_SCHEMA}:
        from ..evolution.instruction_review import prompt_grounding_protocol_v8
        from ..evolution.instruction_review_v9 import prompt_grounding_protocol_v9
        from ..evolution.task_contracts import CLINIFACT_CONTRACT
        target_protocol = packet.get("target_protocol")
        uses_target_v9 = packet.get("schema") in {COHORT_V12_SCHEMA, COHORT_V13_SCHEMA}
        expected_target_protocol = (prompt_grounding_protocol_v9(CLINIFACT_CONTRACT)
                                    if uses_target_v9 else
                                    prompt_grounding_protocol_v8(CLINIFACT_CONTRACT))
        expected_target_version = ("instruction-review-v9-nested-json-claim-extraction"
                                   if uses_target_v9 else
                                   "instruction-review-v8-claim-extraction")
        if (not isinstance(target_protocol, dict)
                or target_protocol != expected_target_protocol
                or target_protocol.get("version") != expected_target_version):
            version = "v13" if packet.get("schema") == COHORT_V13_SCHEMA else (
                "v12" if packet.get("schema") == COHORT_V12_SCHEMA else "v11")
            raise ValueError(f"{version} target prompt protocol differs from its frozen grounding protocol")
        if packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256}:
            version = "v13" if packet.get("schema") == COHORT_V13_SCHEMA else (
                "v12" if packet.get("schema") == COHORT_V12_SCHEMA else "v11")
            raise ValueError(f"{version} task contract differs from the frozen target protocol")
    schedules = (schedule_a, schedule_b)
    model_ids = [row.get("model_id") for row in schedules]
    families = [model_family(model_id) for model_id in model_ids]
    target_id = target_plan["model_id"]
    target_family = model_family(target_id)
    case_orders = [[row.get("case_id") for row in schedule["shards"]]
                   for schedule in schedules]
    expected_version = (
        RUBRIC_V6_VERSION if packet.get("schema") == COHORT_V14_SCHEMA else
        "v8" if packet.get("schema") in {COHORT_V11_SCHEMA, COHORT_V12_SCHEMA,
                                          COHORT_V13_SCHEMA} else
        RUBRIC_V7_VERSION if packet.get("schema") in {COHORT_V9_SCHEMA, COHORT_V10_SCHEMA} else
        RUBRIC_V6_VERSION if packet.get("schema") == COHORT_V8_SCHEMA else
        RUBRIC_V5_VERSION if packet.get("schema") == COHORT_V7_SCHEMA else
        RUBRIC_V4_VERSION if packet.get("schema") == COHORT_V6_SCHEMA else
        RUBRIC_V3_VERSION if packet.get("schema") == COHORT_V5_SCHEMA else RUBRIC_V2_VERSION)
    expected_rubric = (RUBRIC_V8 if expected_version == "v8" else
        RUBRIC_V7 if expected_version == RUBRIC_V7_VERSION else
        RUBRIC_V5 if expected_version == RUBRIC_V5_VERSION else
        RUBRIC_V6 if expected_version == RUBRIC_V6_VERSION else
        RUBRIC_V4 if expected_version == RUBRIC_V4_VERSION else
        RUBRIC_V3 if expected_version == RUBRIC_V3_VERSION else RUBRIC_V2)
    if expected_version == "v8":
        from . import instruction_reviewer_rubric_v8 as rubric_v8
        expected_rubric_sha256 = rubric_v8.protocol_metadata()["rubric_sha256"]
    else:
        expected_rubric_sha256 = sha256_obj(expected_rubric)
    expected_protocol_hash = (
        v8_protocol_hash() if expected_version == "v8" else
        v7_protocol_hash() if expected_version == RUBRIC_V7_VERSION else
        v6_protocol_hash() if expected_version == RUBRIC_V6_VERSION else
        v5_protocol_hash() if expected_version == RUBRIC_V5_VERSION else
        v4_protocol_hash() if expected_version == RUBRIC_V4_VERSION else
        v3_protocol_hash() if expected_version == RUBRIC_V3_VERSION else None)
    expected_splitter_hash = (
        v8_splitter_hash() if expected_version == "v8" else
        v7_splitter_hash() if expected_version == RUBRIC_V7_VERSION else
        v6_splitter_hash() if expected_version == RUBRIC_V6_VERSION else
        v5_splitter_hash() if expected_version == RUBRIC_V5_VERSION else
        v4_splitter_hash() if expected_version == RUBRIC_V4_VERSION else
        v3_splitter_hash() if expected_version == RUBRIC_V3_VERSION else None)
    expected_format_hash = (
        sha256_obj(RESPONSE_FORMAT_TEMPLATE_V8) if expected_version == "v8" else
        sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7) if expected_version == RUBRIC_V7_VERSION else
        sha256_obj(REFERENCE_FORMAT_V6) if expected_version == RUBRIC_V6_VERSION else
        sha256_obj(REFERENCE_FORMAT_V5) if expected_version == RUBRIC_V5_VERSION else
        sha256_obj(REFERENCE_FORMAT_V4) if expected_version == RUBRIC_V4_VERSION else None)
    identities = {(row.get("packet_sha256"), row.get("rubric_version"),
                   row.get("rubric_sha256"), row.get("protocol_hash"),
                   row.get("splitter_hash"),
                   row.get("reference_format_sha256") if expected_version in {
                       RUBRIC_V4_VERSION, RUBRIC_V5_VERSION} else None,
                   row.get("reference_format_template_sha256") if expected_version == RUBRIC_V6_VERSION else None,
                   row.get("response_format_template_sha256") if expected_version == RUBRIC_V7_VERSION else None,
                   row.get("protocol_schema") if expected_version == RUBRIC_V5_VERSION else None,
                   row.get("normalizer_version") if expected_version == RUBRIC_V5_VERSION else None,
                   row.get("max_output_tokens") if expected_version in {RUBRIC_V5_VERSION, RUBRIC_V6_VERSION} else None,
                   row.get("protocol_schema") if expected_version == RUBRIC_V6_VERSION else None,
                   row.get("normalizer_version") if expected_version == RUBRIC_V6_VERSION else None,
                   row.get("protocol_schema") if expected_version == RUBRIC_V7_VERSION else None,
                   row.get("normalizer_version") if expected_version == RUBRIC_V7_VERSION else None,
                   row.get("normalizer_source_hash") if expected_version == RUBRIC_V7_VERSION else None,
                   row.get("max_output_tokens") if expected_version == RUBRIC_V7_VERSION else None)
                  for row in schedules}
    if expected_version == "v8":
        # Exact binding tuple includes the independent rubric identity and the
        # actual v8 target protocol; schedule validation alone is not enough
        # for the pair-level proof object.
        v8_fields = ("reference_rubric_version", "response_format_template_sha256",
            "protocol_schema", "normalizer_version", "normalizer_source_hash",
            "max_output_tokens", "target_protocol_version", "target_protocol_sha256",
            "target_protocol_source_sha256")
        identities = {tuple([row.get("packet_sha256"), row.get("rubric_version"),
            row.get("rubric_sha256"), row.get("protocol_hash"), row.get("splitter_hash"),
            *(row.get(field) for field in v8_fields)]) for row in schedules}
    if packet.get("schema") == COHORT_V14_SCHEMA:
        v14_fields = ("reference_rubric_version", "reference_format_template_sha256",
            "protocol_schema", "normalizer_version", "target_protocol_version",
            "target_protocol_sha256", "target_protocol_source_sha256")
        identities = {tuple([row.get("packet_sha256"), row.get("rubric_version"),
            row.get("rubric_sha256"), row.get("protocol_hash"), row.get("splitter_hash"),
            *(row.get(field) for field in v14_fields)]) for row in schedules}
    if (len(set(model_ids)) != 2 or len(set(families)) != 2
            or target_family in set(families)
            or len(identities) != 1
            or schedule_a.get("packet_sha256") != packet.get("packet_sha256")
            or schedule_b.get("packet_sha256") != packet.get("packet_sha256")
            or schedule_a.get("rubric_version") != expected_version
            or schedule_b.get("rubric_version") != expected_version
            or schedule_a.get("rubric_sha256") != expected_rubric_sha256
            or schedule_b.get("rubric_sha256") != expected_rubric_sha256
            or schedule_a.get("protocol_hash") != expected_protocol_hash
            or schedule_b.get("protocol_hash") != expected_protocol_hash
            or schedule_a.get("splitter_hash") != expected_splitter_hash
            or schedule_b.get("splitter_hash") != expected_splitter_hash
            or (expected_version in {RUBRIC_V4_VERSION, RUBRIC_V5_VERSION} and (
                schedule_a.get("reference_format_sha256") != expected_format_hash
                or schedule_b.get("reference_format_sha256") != expected_format_hash))
            or (expected_version == RUBRIC_V6_VERSION and (
                schedule_a.get("reference_format_template_sha256") != expected_format_hash
                or schedule_b.get("reference_format_template_sha256") != expected_format_hash))
            or (expected_version == RUBRIC_V5_VERSION and (
                schedule_a.get("protocol_schema") != PROTOCOL_SCHEMA_V5
                or schedule_b.get("protocol_schema") != PROTOCOL_SCHEMA_V5
                or schedule_a.get("normalizer_version") != NORMALIZER_VERSION_V5
                or schedule_b.get("normalizer_version") != NORMALIZER_VERSION_V5
                or schedule_a.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS
                or schedule_b.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS))
            or (expected_version == RUBRIC_V6_VERSION and (
                schedule_a.get("protocol_schema") != PROTOCOL_SCHEMA_V6
                or schedule_b.get("protocol_schema") != PROTOCOL_SCHEMA_V6
                or schedule_a.get("normalizer_version") != NORMALIZER_VERSION_V6
                or schedule_b.get("normalizer_version") != NORMALIZER_VERSION_V6
                or schedule_a.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V6))
            or (packet.get("schema") == COHORT_V14_SCHEMA and (
                schedule_a.get("reference_rubric_version") != "instruction-reviewer-reference-v6"
                or schedule_b.get("reference_rubric_version") != "instruction-reviewer-reference-v6"
                or schedule_a.get("target_protocol_version") != expected_target_protocol["version"]
                or schedule_b.get("target_protocol_version") != expected_target_protocol["version"]
                or schedule_a.get("target_protocol_sha256") != sha256_obj(expected_target_protocol)
                or schedule_b.get("target_protocol_sha256") != sha256_obj(expected_target_protocol)
                or schedule_a.get("target_protocol_source_sha256") != expected_target_protocol.get(
                    "target_source_sha256", expected_target_protocol.get("source_sha256"))
                or schedule_b.get("target_protocol_source_sha256") != expected_target_protocol.get(
                    "target_source_sha256", expected_target_protocol.get("source_sha256"))))
            or (expected_version == RUBRIC_V7_VERSION and (
                schedule_a.get("response_format_template_sha256") != expected_format_hash
                or schedule_b.get("response_format_template_sha256") != expected_format_hash
                or schedule_a.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or schedule_b.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or schedule_a.get("normalizer_version") != NORMALIZER_VERSION_V7
                or schedule_b.get("normalizer_version") != NORMALIZER_VERSION_V7
                or schedule_a.get("normalizer_source_hash") != v7_normalizer_source_hash()
                or schedule_b.get("normalizer_source_hash") != v7_normalizer_source_hash()
                or schedule_a.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V7
                or schedule_b.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V7))
            or (expected_version == "v8" and (
                any(any(schedule.get(field) != expected for field, expected in {
                    "reference_rubric_version": "instruction-reviewer-reference-v8",
                    "response_format_template_sha256": expected_format_hash,
                    "protocol_schema": PROTOCOL_SCHEMA_V8,
                    "normalizer_version": NORMALIZER_VERSION_V8,
                    "normalizer_source_hash": v8_normalizer_source_hash(),
                    "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS_V8,
                }.items()) for schedule in schedules)
                or schedule_a.get("target_protocol_version") != expected_target_protocol["version"]
                or schedule_b.get("target_protocol_version") != expected_target_protocol["version"]
                or schedule_a.get("target_protocol_sha256") != sha256_obj(expected_target_protocol)
                or schedule_b.get("target_protocol_sha256") != sha256_obj(expected_target_protocol)
                or schedule_a.get("target_protocol_source_sha256") != expected_target_protocol.get(
                    "target_source_sha256", expected_target_protocol.get("source_sha256"))
                or schedule_b.get("target_protocol_source_sha256") != expected_target_protocol.get(
                    "target_source_sha256", expected_target_protocol.get("source_sha256"))
                or schedule_a.get("target_protocol_sha256") != schedule_b.get("target_protocol_sha256")
                or schedule_a.get("target_protocol_source_sha256") != schedule_b.get("target_protocol_source_sha256")))
            or case_orders[0] != case_orders[1]
            or case_orders[0] != [case["case_id"] for case in packet["cases"]]):
        raise ValueError("G06 A/B/target family, packet, rubric, or case-order gate failed")
    for schedule in schedules:
        config_provenance = schedule.get("config_provenance")
        cfg = schedule.get("runtime_config")
        if not isinstance(config_provenance, dict) or not isinstance(cfg, dict):
            raise ValueError("G06 schedule lacks its isolated config snapshot")
        validate_config_binding(cfg, config_provenance, schedule["model_id"], workspace)
    report = {"schema": "g06-instruction-reviewer-pair-gate-v1",
        "status": "verified_zero_call_pair", "packet_sha256": packet["packet_sha256"],
        "rubric_version": schedule_a["rubric_version"],
        "rubric_sha256": schedule_a["rubric_sha256"], "case_order": case_orders[0],
        "protocol_hash": schedule_a.get("protocol_hash"),
        "splitter_hash": schedule_a.get("splitter_hash"),
        "reviewer_a": {"model_id": model_ids[0], "family": families[0],
                       "schedule_sha256": schedule_a["schedule_sha256"]},
        "reviewer_b": {"model_id": model_ids[1], "family": families[1],
                       "schedule_sha256": schedule_b["schedule_sha256"]},
        "target": {"model_id": target_id, "family": target_family,
                    "plan_sha256": target_plan["plan_sha256"]},
        "family_registry_sha256": family_registry_sha256(),
        "natural_distribution": False, "promotion_authorized": False,
        "planned_calls": 0}
    if expected_version == "v8":
        from ..evolution.instruction_review import prompt_grounding_protocol_v8
        from ..evolution.task_contracts import CLINIFACT_CONTRACT
        target_protocol = packet.get("target_protocol")
        from ..evolution.instruction_review_v9 import prompt_grounding_protocol_v9
        uses_target_v9 = packet.get("schema") in {COHORT_V12_SCHEMA, COHORT_V13_SCHEMA}
        current_target_protocol = (prompt_grounding_protocol_v9(CLINIFACT_CONTRACT)
                                   if uses_target_v9 else
                                   prompt_grounding_protocol_v8(CLINIFACT_CONTRACT))
        if (not isinstance(target_protocol, dict)
                or target_protocol != current_target_protocol
                or target_protocol.get("version") != expected_target_protocol["version"]):
            version = "v13" if packet.get("schema") == COHORT_V13_SCHEMA else (
                "v12" if packet.get("schema") == COHORT_V12_SCHEMA else "v11")
            raise ValueError(f"{version} target prompt protocol differs from the frozen grounding protocol")
        report.update({"reference_rubric_version": "instruction-reviewer-reference-v8",
            "response_format_template_sha256": expected_format_hash,
            "protocol_schema": PROTOCOL_SCHEMA_V8,
            "normalizer_version": NORMALIZER_VERSION_V8,
            "normalizer_source_hash": v8_normalizer_source_hash(),
            "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS_V8,
            "target_protocol_version": target_protocol["version"],
            "target_protocol_sha256": sha256_obj(target_protocol),
            "target_protocol_source_sha256": target_protocol.get(
                "target_source_sha256", target_protocol.get("source_sha256"))})
    if packet.get("schema") == COHORT_V14_SCHEMA:
        target_protocol = packet["target_protocol"]
        report.update({"reference_rubric_version": "instruction-reviewer-reference-v6",
            "reference_format_template_sha256": expected_format_hash,
            "protocol_schema": PROTOCOL_SCHEMA_V6,
            "normalizer_version": NORMALIZER_VERSION_V6,
            "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS_V6,
            "target_protocol_version": target_protocol["version"],
            "target_protocol_sha256": sha256_obj(target_protocol),
            "target_protocol_source_sha256": target_protocol.get(
                "target_source_sha256", target_protocol.get("source_sha256")),
            "target_transport": target_plan["protocol_metadata"]["response_mode"],
            "dynamic_schema_manifest_sha256": target_plan["protocol_metadata"][
                "dynamic_schema_manifest_sha256"],
            "target_protocol_metadata_sha256": sha256_obj(target_plan["protocol_metadata"])})
    if packet.get("schema") == COHORT_V10_SCHEMA:
        from ..evolution.instruction_review import prompt_grounding_protocol
        from ..evolution.task_contracts import CLINIFACT_CONTRACT
        target_protocol = packet.get("target_protocol")
        current_target_protocol = prompt_grounding_protocol(CLINIFACT_CONTRACT)
        if (not isinstance(target_protocol, dict)
                or target_protocol != current_target_protocol
                or target_protocol.get("version") != "instruction-review-v7-span-atomic"):
            raise ValueError("v10 target prompt protocol differs from the current grounding protocol")
        report["target_protocol_version"] = target_protocol["version"]
        report["target_protocol_sha256"] = sha256_obj(target_protocol)
        report["target_protocol_source_sha256"] = target_protocol.get("source_sha256")
    if expected_version in {RUBRIC_V4_VERSION, RUBRIC_V5_VERSION}:
        report["reference_format_sha256"] = expected_format_hash
    if expected_version == RUBRIC_V6_VERSION:
        report["reference_format_template_sha256"] = expected_format_hash
        report.update({"protocol_schema": PROTOCOL_SCHEMA_V6,
            "normalizer_version": NORMALIZER_VERSION_V6,
            "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS_V6})
    if expected_version == RUBRIC_V7_VERSION:
        report["response_format_template_sha256"] = expected_format_hash
        report.update({"protocol_schema": PROTOCOL_SCHEMA_V7,
            "normalizer_version": NORMALIZER_VERSION_V7,
            "normalizer_source_hash": v7_normalizer_source_hash(),
            "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS_V7})
    if expected_version == RUBRIC_V5_VERSION:
        report.update({"protocol_schema": PROTOCOL_SCHEMA_V5,
            "normalizer_version": NORMALIZER_VERSION_V5,
            "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS})
    return report
