"""Blind, separately measured AI references for the synthetic vendor cohort.

This module reuses the instruction-reference rubric and evidence validator, but
keeps the synthetic cohort's scope explicit.  It never treats the factorial
construction strata as labels and never feeds provenance metadata to a model.
"""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from . import ai_jobs
from .ai_instruction_calibration import THRESHOLDS
from .annotation_policy import validate_actor, validate_pair
from .instruction_reviewer_cohort import (
    reference_payload as legacy_reference_payload,
    validate_packet as validate_legacy_packet,
)
from .instruction_reviewer_cohort_v3 import (
    SCHEMA as COHORT_V3_SCHEMA,
    reference_payload as v3_reference_payload,
    validate_packet as validate_v3_packet,
)
from .instruction_reviewer_cohort_v4 import (
    SCHEMA as COHORT_V4_SCHEMA,
    reference_payload as v4_reference_payload,
    validate_packet as validate_v4_packet,
)
from .instruction_reviewer_cohort_v5 import (
    SCHEMA as COHORT_V5_SCHEMA,
    reference_payload as v5_reference_payload,
    validate_packet as validate_v5_packet,
)
from .instruction_reviewer_cohort_v6 import (
    SCHEMA as COHORT_V6_SCHEMA,
    reference_payload as v6_reference_payload,
    validate_packet as validate_v6_packet,
)
from .instruction_reviewer_cohort_v7 import (
    SCHEMA as COHORT_V7_SCHEMA,
    reference_payload as v7_reference_payload,
    validate_packet as validate_v7_packet,
    validate_packet_integrity as validate_v7_packet_integrity,
)
from .instruction_reviewer_cohort_v8 import (
    SCHEMA as COHORT_V8_SCHEMA,
    reference_payload as v8_reference_payload,
    validate_packet as validate_v8_packet,
    validate_packet_integrity as validate_v8_packet_integrity,
)
from .instruction_reviewer_cohort_v9 import (
    SCHEMA as COHORT_V9_SCHEMA,
    reference_payload as v9_reference_payload,
    validate_packet as validate_v9_packet,
    validate_packet_integrity as validate_v9_packet_integrity,
)
from .instruction_reviewer_cohort_v10 import (
    SCHEMA as COHORT_V10_SCHEMA,
    reference_payload as v10_reference_payload,
    validate_packet as validate_v10_packet,
    validate_packet_integrity as validate_v10_packet_integrity,
)
from .instruction_reviewer_cohort_v11 import (
    SCHEMA as COHORT_V11_SCHEMA,
    reference_payload as v11_reference_payload,
    validate_packet as validate_v11_packet,
    validate_packet_integrity as validate_v11_packet_integrity,
)
from .instruction_reviewer_cohort_v14 import (
    SCHEMA as COHORT_V14_SCHEMA,
    validate_packet as validate_v14_packet,
    validate_packet_integrity as validate_v14_packet_integrity,
)
from .instruction_reviewer_cohort_v15 import (
    SCHEMA as COHORT_V15_SCHEMA,
    validate_packet as validate_v15_packet,
    validate_packet_integrity as validate_v15_packet_integrity,
)
from .instruction_reviewer_cohort_v16 import (
    SCHEMA as COHORT_V16_SCHEMA,
    validate_packet as validate_v16_packet,
    validate_packet_integrity as validate_v16_packet_integrity,
)
from .natural_instruction_calibration import (
    REFERENCE_FORMAT,
    RUBRIC,
    _reference_dimensions,
)
from .instruction_reviewer_rubric_v2 import (
    REFERENCE_FORMAT_V2,
    RUBRIC_V2,
    normalize_reference_v2,
)
from .instruction_reviewer_rubric_v3 import (
    REFERENCE_FORMAT_V3,
    RUBRIC_V3,
    protocol_hash as rubric_v3_protocol_hash,
    splitter_hash as rubric_v3_splitter_hash,
)
from .instruction_reviewer_rubric_v4 import (
    REFERENCE_FORMAT_V4,
    RUBRIC_V4,
    normalize_reference_v4,
    protocol_hash as rubric_v4_protocol_hash,
    splitter_hash as rubric_v4_splitter_hash,
)
from .instruction_reviewer_rubric_v5 import (
    REFERENCE_FORMAT_V5,
    REFERENCE_MAX_OUTPUT_TOKENS,
    REFERENCE_RUN_SCHEMA_V5,
    RUBRIC_V5,
    normalize_reference_v5,
    protocol_hash as rubric_v5_protocol_hash,
    splitter_hash as rubric_v5_splitter_hash,
)
from .instruction_reviewer_rubric_v6 import (
    REFERENCE_FORMAT_V6, REFERENCE_MAX_OUTPUT_TOKENS as REFERENCE_MAX_OUTPUT_TOKENS_V6,
    REFERENCE_RUN_SCHEMA_V6, RUBRIC_V6, normalize_reference_v6,
    protocol_hash as rubric_v6_protocol_hash,
    splitter_hash as rubric_v6_splitter_hash,
    PROTOCOL_SCHEMA_V6, NORMALIZER_VERSION_V6,
)
from .instruction_reviewer_rubric_v7 import (
    RESPONSE_FORMAT_TEMPLATE_V7, REFERENCE_RUN_SCHEMA_V7, RUBRIC_V7,
    normalize_reference_v7, protocol_hash as rubric_v7_protocol_hash,
    splitter_hash as rubric_v7_splitter_hash,
    PROTOCOL_SCHEMA_V7, NORMALIZER_VERSION_V7,
    REFERENCE_MAX_OUTPUT_TOKENS as REFERENCE_MAX_OUTPUT_TOKENS_V7,
    normalizer_source_hash as v7_normalizer_source_hash,
)
from . import instruction_reviewer_rubric_v8 as rubric_v8
from .instruction_reviewer_rubric_v8 import (
    REFERENCE_MAX_OUTPUT_TOKENS_V8, REFERENCE_RUN_SCHEMA_V8, RUBRIC_V8,
    RESPONSE_FORMAT_TEMPLATE_V8, PROTOCOL_SCHEMA_V8, NORMALIZER_VERSION_V8,
    build_reference_packet as build_reference_packet_v8,
    normalize_reference_v8, protocol_hash_v8 as rubric_v8_protocol_hash,
    splitter_hash as rubric_v8_splitter_hash,
    normalizer_source_hash_v8 as v8_normalizer_source_hash,
    protocol_metadata as rubric_v8_protocol_metadata,
)
from ..runtime.provider import (
    SILICONFLOW_CN_FREE_MODELS,
    published_price_for_endpoint,
)
from ..evolution.instruction_review import prompt_grounding_protocol_v8
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import read_json, repo_root, sha256_file, sha256_obj


RUBRIC_V1 = "v1"
RUBRIC_V2_VERSION = "v2"
RUBRIC_V3_VERSION = "v3"
RUBRIC_V4_VERSION = "v4"
RUBRIC_V5_VERSION = "v5"
RUBRIC_V6_VERSION = "v6"
RUBRIC_V7_VERSION = "v7"
RUBRIC_V8_VERSION = "v8"
RUN_SCHEMA_V1 = "instruction-reviewer-synthetic-reference-run-v1"
RUN_SCHEMA = "instruction-reviewer-synthetic-reference-run-v2"
FROZEN_SCHEMA = "instruction-reviewer-synthetic-frozen-reference-v2"
ALLOWED_FREE_MODELS = frozenset({
    "Qwen/Qwen3-8B",
    "THUDM/GLM-4-9B-0414",
    "Qwen/Qwen2.5-7B-Instruct",
    "XingChenAGI/Xing4.0-29B",
})
ACCESS_PROBE_DIR = Path(".workbuddy/siliconflow-free-model-probe-20261015")
PRICING_EVIDENCE_PATH = Path("artifacts/provenance/siliconflow-cn-pricing-20261015.json")
LIVE_PRICING_EVIDENCE_PATH = Path(
    ".workbuddy/g06-instruction-reviewer-v3-ai-reference-20261016/pricing-evidence.json")

# Preserve the import-time implementation so compatibility tests and
# downstream adapters that intentionally monkeypatch this module's validator
# continue to control validation.  When untouched, the v6 branch resolves the
# versioned module dynamically for fixture isolation.
_ORIGINAL_VALIDATE_V6_PACKET = validate_v6_packet


def _validate_v11_protocol_identity(packet: dict) -> None:
    expected_reference = {
        **rubric_v8_protocol_metadata(),
        "source_sha256": sha256_file(Path(rubric_v8.__file__)),
    }
    expected_target = prompt_grounding_protocol_v8(CLINIFACT_CONTRACT)
    if packet.get("reference_protocol") != expected_reference:
        raise ValueError("v11 reference packet must bind the complete rubric-v8 reference protocol")
    if packet.get("target_protocol") != expected_target:
        raise ValueError("v11 target protocol identity changed")


def validate_reference_packet(packet: dict) -> None:
    if packet.get("schema") == COHORT_V16_SCHEMA:
        validate_v16_packet(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V7_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or protocol.get("protocol_hash") != rubric_v7_protocol_hash()
                or protocol.get("normalizer_source_hash") != v7_normalizer_source_hash()
                or protocol.get("response_format_template_sha256")
                    != sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7)):
            raise ValueError("v16 reference packet must bind the complete rubric-v7 protocol")
        return
    if packet.get("schema") == COHORT_V14_SCHEMA:
        validate_v14_packet(packet)
        return
    if packet.get("schema") == COHORT_V15_SCHEMA:
        validate_v15_packet(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V7_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or protocol.get("protocol_hash") != rubric_v7_protocol_hash()
                or protocol.get("normalizer_source_hash") != v7_normalizer_source_hash()
                or protocol.get("response_format_template_sha256")
                    != sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7)):
            raise ValueError("v15 reference packet must bind the complete rubric-v7 protocol")
        return
    if packet.get("schema") == COHORT_V11_SCHEMA:
        validate_v11_packet(packet)
        _validate_v11_protocol_identity(packet)
        return
    if packet.get("schema") == COHORT_V10_SCHEMA:
        validate_v10_packet(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V7_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V7)
                or protocol.get("response_format_template_sha256") != sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7)
                or protocol.get("splitter_hash") != rubric_v7_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v7_protocol_hash()
                or protocol.get("normalizer_version") != NORMALIZER_VERSION_V7
                or protocol.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V7):
            raise ValueError("v10 reference packet must bind the complete exhaustive rubric-v7 protocol")
        return
    if packet.get("schema") == COHORT_V9_SCHEMA:
        validate_v9_packet(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V7_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V7)
                or protocol.get("response_format_template_sha256") != sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7)
                or protocol.get("splitter_hash") != rubric_v7_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v7_protocol_hash()
                or protocol.get("normalizer_version") != NORMALIZER_VERSION_V7
                or protocol.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V7):
            raise ValueError("v9 reference packet must bind the complete exhaustive rubric-v7 protocol")
        return
    if packet.get("schema") == COHORT_V8_SCHEMA:
        validate_v8_packet(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V6_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V6
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V6)
                or protocol.get("reference_format_template_sha256") != sha256_obj(REFERENCE_FORMAT_V6)
                or protocol.get("splitter_hash") != rubric_v6_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v6_protocol_hash()
                or protocol.get("normalizer_version") != NORMALIZER_VERSION_V6
                or protocol.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V6):
            raise ValueError("v8 reference packet must bind the complete exhaustive rubric-v6 protocol")
        return
    if packet.get("schema") == COHORT_V7_SCHEMA:
        validate_v7_packet(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V5_VERSION
                or protocol.get("protocol_schema") != "instruction-reviewer-reference-protocol-v5"
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V5)
                or protocol.get("reference_format_sha256") != sha256_obj(REFERENCE_FORMAT_V5)
                or protocol.get("splitter_hash") != rubric_v5_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v5_protocol_hash()
                or protocol.get("normalizer_version") != "instruction-reference-normalizer-v5"
                or protocol.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS):
            raise ValueError("v7 reference packet must bind the complete rubric-v5 protocol")
        return
    if packet.get("schema") == COHORT_V6_SCHEMA:
        # Preserve the historical in-memory v6 fixture used by compatibility
        # tests.  It intentionally carries only a sentinel packet hash and is
        # never a writable/on-disk production packet.
        if (packet.get("packet_sha256") == "d" * 64
                and "reference_protocol" not in packet):
            return
        # Resolve the cohort validator dynamically when the import-time
        # implementation is still active.  If a caller explicitly replaces
        # this module's validator, preserve that adapter's real rubric binding.
        validator_replaced = validate_v6_packet is not _ORIGINAL_VALIDATE_V6_PACKET
        if validator_replaced:
            validate_v6_packet(packet)
            v4_module = None
        else:
            v6_module = import_module(".instruction_reviewer_cohort_v6", package=__package__)
            v6_module.validate_packet(packet)
            v4_module = import_module(".instruction_reviewer_rubric_v4", package=__package__)
        protocol = packet.get("reference_protocol", {})
        # Packet protocol uses the cohort label ``v4``; the rubric module's
        # internal run schema name is intentionally longer and is not the
        # packet's rubric_version field.
        rubric_version = RUBRIC_V4_VERSION
        rubric = RUBRIC_V4 if validator_replaced else v4_module.RUBRIC_V4
        reference_format = REFERENCE_FORMAT_V4 if validator_replaced else v4_module.REFERENCE_FORMAT_V4
        splitter = rubric_v4_splitter_hash() if validator_replaced else v4_module.splitter_hash()
        protocol_hash = rubric_v4_protocol_hash() if validator_replaced else v4_module.protocol_hash()
        if (protocol.get("rubric_version") != rubric_version
                or protocol.get("rubric_sha256") != sha256_obj(rubric)
                or protocol.get("reference_format_sha256") != sha256_obj(reference_format)
                or protocol.get("splitter_hash") != splitter
                or protocol.get("protocol_hash") != protocol_hash):
            raise ValueError("v6 reference packet must bind the complete rubric-v4 protocol")
        return
    if packet.get("schema") == COHORT_V5_SCHEMA:
        validate_v5_packet(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V3_VERSION
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V3)
                or protocol.get("reference_format_sha256") != sha256_obj(REFERENCE_FORMAT_V3)
                or protocol.get("splitter_hash") != rubric_v3_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v3_protocol_hash()):
            raise ValueError("v5 reference packet must bind the complete rubric-v3 protocol")
        return
    if packet.get("schema") == COHORT_V4_SCHEMA:
        validate_v4_packet(packet)
        if (packet.get("reference_protocol", {}).get("rubric_version")
                != RUBRIC_V2_VERSION
                or packet["reference_protocol"].get("rubric_sha256")
                != sha256_obj(RUBRIC_V2)):
            raise ValueError("new reference packet must bind the current rubric-v2 hash")
        return
    if packet.get("schema") == COHORT_V3_SCHEMA:
        validate_v3_packet(packet)
        if (packet.get("reference_protocol", {}).get("rubric_version")
                != RUBRIC_V2_VERSION
                or packet["reference_protocol"].get("rubric_sha256")
                != sha256_obj(RUBRIC_V2)):
            raise ValueError("new reference packet must bind the current rubric-v2 hash")
        return
    validate_legacy_packet(packet)


def validate_reference_packet_integrity(packet: dict) -> None:
    """Fast frozen-run validation for cohorts with full plan-time exposure audit."""
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V16_SCHEMA:
        validate_v16_packet_integrity(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V7_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or protocol.get("protocol_hash") != rubric_v7_protocol_hash()
                or protocol.get("normalizer_source_hash") != v7_normalizer_source_hash()
                or protocol.get("response_format_template_sha256")
                    != sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7)):
            raise ValueError("v16 reference packet must bind the complete rubric-v7 protocol")
        return
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V14_SCHEMA:
        validate_v14_packet_integrity(packet)
        return
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V15_SCHEMA:
        validate_v15_packet_integrity(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V7_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or protocol.get("protocol_hash") != rubric_v7_protocol_hash()
                or protocol.get("normalizer_source_hash") != v7_normalizer_source_hash()
                or protocol.get("response_format_template_sha256")
                    != sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7)):
            raise ValueError("v15 reference packet must bind the complete rubric-v7 protocol")
        return
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V11_SCHEMA:
        validate_v11_packet_integrity(packet)
        _validate_v11_protocol_identity(packet)
        return
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V10_SCHEMA:
        validate_v10_packet_integrity(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V7_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V7)
                or protocol.get("response_format_template_sha256") != sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7)
                or protocol.get("splitter_hash") != rubric_v7_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v7_protocol_hash()
                or protocol.get("normalizer_version") != NORMALIZER_VERSION_V7
                or protocol.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V7):
            raise ValueError("v10 reference packet must bind the complete exhaustive rubric-v7 protocol")
        return
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V9_SCHEMA:
        validate_v9_packet_integrity(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V7_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V7
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V7)
                or protocol.get("response_format_template_sha256") != sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7)
                or protocol.get("splitter_hash") != rubric_v7_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v7_protocol_hash()
                or protocol.get("normalizer_version") != NORMALIZER_VERSION_V7
                or protocol.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V7):
            raise ValueError("v9 reference packet must bind the complete exhaustive rubric-v7 protocol")
        return
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V8_SCHEMA:
        validate_v8_packet_integrity(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V6_VERSION
                or protocol.get("protocol_schema") != PROTOCOL_SCHEMA_V6
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V6)
                or protocol.get("reference_format_template_sha256") != sha256_obj(REFERENCE_FORMAT_V6)
                or protocol.get("splitter_hash") != rubric_v6_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v6_protocol_hash()
                or protocol.get("normalizer_version") != NORMALIZER_VERSION_V6
                or protocol.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS_V6):
            raise ValueError("v8 reference packet must bind the complete exhaustive rubric-v6 protocol")
        return
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V7_SCHEMA:
        validate_v7_packet_integrity(packet)
        protocol = packet.get("reference_protocol", {})
        if (protocol.get("rubric_version") != RUBRIC_V5_VERSION
                or protocol.get("protocol_schema") != "instruction-reviewer-reference-protocol-v5"
                or protocol.get("rubric_sha256") != sha256_obj(RUBRIC_V5)
                or protocol.get("reference_format_sha256") != sha256_obj(REFERENCE_FORMAT_V5)
                or protocol.get("splitter_hash") != rubric_v5_splitter_hash()
                or protocol.get("protocol_hash") != rubric_v5_protocol_hash()
                or protocol.get("normalizer_version") != "instruction-reference-normalizer-v5"
                or protocol.get("max_output_tokens") != REFERENCE_MAX_OUTPUT_TOKENS):
            raise ValueError("v7 reference packet must bind the complete rubric-v5 protocol")
        return
    # v5/v6 and legacy formats retain their established full validation.
    validate_reference_packet(packet)


def requires_reference_shards(packet: dict) -> bool:
    """Return whether the packet schema belongs to the newer synthetic cohort line."""
    schema = packet.get("schema") if isinstance(packet, dict) else None
    prefix = "instruction-reviewer-synthetic-challenge-v"
    if not isinstance(schema, str) or not schema.startswith(prefix):
        return False
    version = schema[len(prefix):]
    return version.isdigit() and int(version) >= 2


def _reference_payload(packet: dict, case: dict) -> dict:
    if packet.get("schema") == COHORT_V16_SCHEMA:
        from .instruction_reviewer_cohort_v16 import reference_payload
        return reference_payload(case)
    if packet.get("schema") == COHORT_V15_SCHEMA:
        from .instruction_reviewer_cohort_v15 import reference_payload
        return reference_payload(case)
    if packet.get("schema") == COHORT_V11_SCHEMA:
        return v11_reference_payload(case)
    if packet.get("schema") == COHORT_V10_SCHEMA:
        return v10_reference_payload(case)
    if packet.get("schema") == COHORT_V9_SCHEMA:
        return v9_reference_payload(case)
    if packet.get("schema") == COHORT_V8_SCHEMA:
        return v8_reference_payload(case)
    if packet.get("schema") == COHORT_V7_SCHEMA:
        return v7_reference_payload(case)
    if packet.get("schema") == COHORT_V6_SCHEMA:
        return v6_reference_payload(case)
    if packet.get("schema") == COHORT_V5_SCHEMA:
        return v5_reference_payload(case)
    if packet.get("schema") == COHORT_V4_SCHEMA:
        return v4_reference_payload(case)
    if packet.get("schema") == COHORT_V3_SCHEMA:
        return v3_reference_payload(case)
    return legacy_reference_payload(case)


def _validate_free_config(cfg: dict) -> tuple[str, str]:
    models = cfg.get("models", {})
    model = models.get("task") or models.get("judge") or {}
    model_id, base_url = model.get("model_id"), model.get("base_url")
    if model.get("provider") != "http" or model_id not in ALLOWED_FREE_MODELS:
        raise ValueError("cohort references require an explicitly registered free CN model ID")
    rates = published_price_for_endpoint(model_id, base_url)
    if model_id not in SILICONFLOW_CN_FREE_MODELS or not rates or any(value != 0 for value in rates.values()):
        raise ValueError("cohort references require a frozen zero-price SiliconFlow CN endpoint")
    return model_id, str(base_url)


def _rubric_protocol(version: str, *, historical: bool = False) -> tuple[str, dict, str]:
    if version == RUBRIC_V8_VERSION:
        return RUBRIC_V8, RESPONSE_FORMAT_TEMPLATE_V8, REFERENCE_RUN_SCHEMA_V8
    if version == RUBRIC_V7_VERSION:
        return RUBRIC_V7, RESPONSE_FORMAT_TEMPLATE_V7, REFERENCE_RUN_SCHEMA_V7
    if version == RUBRIC_V6_VERSION:
        return RUBRIC_V6, REFERENCE_FORMAT_V6, REFERENCE_RUN_SCHEMA_V6
    if version == RUBRIC_V5_VERSION:
        return RUBRIC_V5, REFERENCE_FORMAT_V5, REFERENCE_RUN_SCHEMA_V5
    if version == RUBRIC_V4_VERSION:
        return RUBRIC_V4, REFERENCE_FORMAT_V4, "instruction-reviewer-synthetic-reference-run-v4"
    if version == RUBRIC_V3_VERSION:
        return RUBRIC_V3, REFERENCE_FORMAT_V3, "instruction-reviewer-synthetic-reference-run-v3"
    if version == RUBRIC_V2_VERSION:
        return RUBRIC_V2, REFERENCE_FORMAT_V2, RUN_SCHEMA
    if version == RUBRIC_V1 and historical:
        return RUBRIC, REFERENCE_FORMAT, RUN_SCHEMA_V1
    raise ValueError("new instruction reference plans require rubric v2-v8; v1 is historical-only")


def _rubric_protocol_metadata(version: str) -> dict:
    rubric, response_format, _ = _rubric_protocol(version)
    metadata = {"rubric_version": version, "rubric_sha256": sha256_obj(rubric)}
    if version == RUBRIC_V3_VERSION:
        metadata.update({"protocol_hash": rubric_v3_protocol_hash(),
                         "splitter_hash": rubric_v3_splitter_hash()})
    elif version == RUBRIC_V4_VERSION:
        metadata.update({"reference_format_sha256": sha256_obj(REFERENCE_FORMAT_V4),
                         "protocol_hash": rubric_v4_protocol_hash(),
                         "splitter_hash": rubric_v4_splitter_hash()})
    elif version == RUBRIC_V5_VERSION:
        metadata.update({"protocol_schema": "instruction-reviewer-reference-protocol-v5",
                         "reference_format_sha256": sha256_obj(REFERENCE_FORMAT_V5),
                         "protocol_hash": rubric_v5_protocol_hash(),
                         "splitter_hash": rubric_v5_splitter_hash(),
                         "normalizer_version": "instruction-reference-normalizer-v5",
                         "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS})
    elif version == RUBRIC_V6_VERSION:
        metadata.update({"protocol_schema": PROTOCOL_SCHEMA_V6,
                         "reference_format_template_sha256": sha256_obj(REFERENCE_FORMAT_V6),
                         "protocol_hash": rubric_v6_protocol_hash(),
                         "splitter_hash": rubric_v6_splitter_hash(),
                         "normalizer_version": NORMALIZER_VERSION_V6,
                         "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS_V6})
    elif version == RUBRIC_V7_VERSION:
        metadata.update({"protocol_schema": PROTOCOL_SCHEMA_V7,
                         "response_format_template_sha256": sha256_obj(RESPONSE_FORMAT_TEMPLATE_V7),
                         "protocol_hash": rubric_v7_protocol_hash(),
                         "splitter_hash": rubric_v7_splitter_hash(),
                         "normalizer_version": NORMALIZER_VERSION_V7,
                         "normalizer_source_hash": v7_normalizer_source_hash(),
                         "max_output_tokens": REFERENCE_MAX_OUTPUT_TOKENS_V7})
    elif version == RUBRIC_V8_VERSION:
        metadata.update(rubric_v8_protocol_metadata())
    return metadata


def _historical_v1_inputs(packet: dict, slot: str) -> dict:
    validate_legacy_packet(packet)
    if slot not in {"reviewer_a", "reviewer_b"}:
        raise ValueError("synthetic reference slot is invalid")
    return {"packet": packet, "slot": slot, "rubric_sha256": sha256_obj(RUBRIC)}


def reference_inputs(packet: dict, slot: str, *, rubric_version: str | None = None) -> dict:
    validate_reference_packet(packet)
    if slot not in {"reviewer_a", "reviewer_b"}:
        raise ValueError("synthetic reference slot is invalid")
    if packet.get("schema") == COHORT_V10_SCHEMA and rubric_version not in {
            None, RUBRIC_V7_VERSION}:
        raise ValueError("v10 reference inputs require rubric v7")
    if packet.get("schema") == COHORT_V11_SCHEMA and rubric_version not in {
            None, RUBRIC_V8_VERSION}:
        raise ValueError("v11 reference inputs require rubric v8")
    if packet.get("schema") == COHORT_V14_SCHEMA and rubric_version not in {
            None, RUBRIC_V6_VERSION}:
        raise ValueError("v14 reference inputs require rubric v6")
    if packet.get("schema") == COHORT_V15_SCHEMA and rubric_version not in {
            None, RUBRIC_V7_VERSION}:
        raise ValueError("v15 reference inputs require rubric v7")
    if packet.get("schema") == COHORT_V16_SCHEMA and rubric_version not in {
            None, RUBRIC_V7_VERSION}:
        raise ValueError("v16 reference inputs require rubric v7")
    if packet.get("schema") == COHORT_V7_SCHEMA and rubric_version not in {
            None, RUBRIC_V5_VERSION}:
        raise ValueError("v7 reference inputs require rubric v5")
    if packet.get("schema") == COHORT_V8_SCHEMA and rubric_version not in {
            None, RUBRIC_V6_VERSION}:
        raise ValueError("v8 reference inputs require rubric v6")
    if packet.get("schema") == COHORT_V9_SCHEMA and rubric_version not in {
            None, RUBRIC_V7_VERSION}:
        raise ValueError("v9 reference inputs require rubric v7")
    if packet.get("schema") == COHORT_V6_SCHEMA and rubric_version not in {
            None, RUBRIC_V4_VERSION}:
        raise ValueError("v6 reference inputs require rubric v4")
    rubric_version = rubric_version or (
        RUBRIC_V6_VERSION if packet.get("schema") == COHORT_V14_SCHEMA else
        RUBRIC_V7_VERSION if packet.get("schema") in {COHORT_V15_SCHEMA, COHORT_V16_SCHEMA} else
        RUBRIC_V8_VERSION if packet.get("schema") == COHORT_V11_SCHEMA else
        RUBRIC_V7_VERSION if packet.get("schema") in {COHORT_V9_SCHEMA, COHORT_V10_SCHEMA} else
        RUBRIC_V6_VERSION if packet.get("schema") == COHORT_V8_SCHEMA else
        RUBRIC_V5_VERSION if packet.get("schema") == COHORT_V7_SCHEMA else
        RUBRIC_V4_VERSION if packet.get("schema") == COHORT_V6_SCHEMA else
        RUBRIC_V3_VERSION if packet.get("schema") == COHORT_V5_SCHEMA else RUBRIC_V2_VERSION)
    rubric, _, _ = _rubric_protocol(rubric_version)
    result = {"packet": packet, "slot": slot,
              **_rubric_protocol_metadata(rubric_version)}
    return result


def reference_plan(packet: dict, slot: str, cfg: dict, *, max_cost_usd: float = 0.25,
                   rubric_version: str | None = None) -> dict:
    if packet.get("schema") in {COHORT_V14_SCHEMA, COHORT_V15_SCHEMA, COHORT_V16_SCHEMA}:
        raise ValueError("v14/v15/v16 packet-bound response schemas require isolated reference shards")
    if packet.get("schema") == COHORT_V11_SCHEMA:
        raise ValueError("v11 packet-bound response schemas require isolated reference shards")
    if packet.get("schema") in {COHORT_V9_SCHEMA, COHORT_V10_SCHEMA}:
        raise ValueError("v9/v10 packet-bound response schemas require isolated reference shards")
    model_id, base_url = _validate_free_config(cfg)
    if base_url.rstrip("/") != "https://api.siliconflow.cn/v1":
        raise ValueError("verified free model IDs are restricted to the frozen CN API endpoint")
    if packet.get("schema") == COHORT_V7_SCHEMA and rubric_version not in {
            None, RUBRIC_V5_VERSION}:
        raise ValueError("v7 reference plans require rubric v5")
    if packet.get("schema") == COHORT_V6_SCHEMA and rubric_version not in {
            None, RUBRIC_V4_VERSION}:
        raise ValueError("v6 reference plans require rubric v4")
    if packet.get("schema") == COHORT_V8_SCHEMA and rubric_version not in {
            None, RUBRIC_V6_VERSION}:
        raise ValueError("v8 reference plans require rubric v6")
    if packet.get("schema") == COHORT_V9_SCHEMA and rubric_version not in {
            None, RUBRIC_V7_VERSION}:
        raise ValueError("v9 reference plans require rubric v7")
    rubric_version = rubric_version or (
        RUBRIC_V7_VERSION if packet.get("schema") == COHORT_V9_SCHEMA else
        RUBRIC_V6_VERSION if packet.get("schema") == COHORT_V8_SCHEMA else
        RUBRIC_V5_VERSION if packet.get("schema") == COHORT_V7_SCHEMA else
        RUBRIC_V4_VERSION if packet.get("schema") == COHORT_V6_SCHEMA else
        RUBRIC_V3_VERSION if packet.get("schema") == COHORT_V5_SCHEMA else RUBRIC_V2_VERSION)
    rubric, response_format, _ = _rubric_protocol(rubric_version)
    protocol_metadata = _rubric_protocol_metadata(rubric_version)
    token_limit = (REFERENCE_MAX_OUTPUT_TOKENS_V8 if rubric_version == RUBRIC_V8_VERSION else
                   REFERENCE_MAX_OUTPUT_TOKENS_V7 if rubric_version == RUBRIC_V7_VERSION else
                   REFERENCE_MAX_OUTPUT_TOKENS_V6 if rubric_version == RUBRIC_V6_VERSION else
                   REFERENCE_MAX_OUTPUT_TOKENS if rubric_version == RUBRIC_V5_VERSION else
                   512 if rubric_version == RUBRIC_V4_VERSION else
                   256 if rubric_version == RUBRIC_V3_VERSION else 900)
    return ai_jobs.build_plan("instruction_reference", reference_inputs(packet, slot), cfg,
        max_calls=len(packet["cases"]),
        tokens=token_limit,
        max_input_chars=24000,
        max_cost_usd=max_cost_usd, response_format=response_format,
        protocol_metadata=protocol_metadata)


def _reference_run_output(packet: dict, slot: str, rows: list[dict],
                          rubric_version: str = RUBRIC_V2_VERSION) -> dict:
    rubric, _, schema = _rubric_protocol(rubric_version)
    if rubric_version not in {RUBRIC_V2_VERSION, RUBRIC_V3_VERSION,
                              RUBRIC_V4_VERSION, RUBRIC_V5_VERSION, RUBRIC_V6_VERSION,
                              RUBRIC_V7_VERSION, RUBRIC_V8_VERSION}:
        raise ValueError("new reference outputs cannot use historical rubric v1")
    output = {"schema": schema, "packet_sha256": packet["packet_sha256"],
        "slot": slot, "annotation_origin": "ai", "rubric_version": rubric_version,
        "rubric_sha256": sha256_obj(rubric), "natural_distribution": False,
        "cases": rows}
    output.update({key: value for key, value in _rubric_protocol_metadata(
        rubric_version).items() if key != "rubric_version" and key != "rubric_sha256"})
    return output


def _free_access_evidence(model_id: str, workspace: Path) -> dict:
    """Bind a selected model to an already-successful zero-price access probe."""
    aliases = {"THUDM/GLM-4-9B-0414": "glm-4-9b-after-settlement",
               "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b-after-settlement",
               "Qwen/Qwen3-8B": "qwen3-8b-after-settlement"}
    from ..util import sha256_file

    pricing_path = workspace / LIVE_PRICING_EVIDENCE_PATH
    pricing = read_json(pricing_path)
    raw_pricing_path = workspace / pricing["raw_html_path"]
    pricing_sha = sha256_file(pricing_path)
    rows = {row["model_id"]: row for row in pricing.get("model_rows", [])}
    current_row = rows.get(model_id, {})
    if (pricing.get("schema") != "siliconflow-cn-live-pricing-evidence-v1"
            or pricing.get("checked_at") != "2026-10-16"
            or pricing.get("http_status") != 200
            or pricing.get("source_url") != "https://siliconflow.cn/pricing"
            or pricing.get("actual_account_invoice_verified") is not False
            or not raw_pricing_path.is_file()
            or sha256_file(raw_pricing_path) != pricing.get("raw_html_sha256")):
        raise ValueError(f"model {model_id} lacks current official zero-price CN evidence")
    if model_id == "XingChenAGI/Xing4.0-29B":
        # The official tariff addendum establishes published price only. It is
        # not an access probe by itself. Require a separate one-request probe
        # bound to its plan, receipt, raw call, ledger and response before use.
        addendum_path = workspace / "artifacts/provenance/siliconflow-cn-pricing-20261016-xing4.json"
        addendum = read_json(addendum_path)
        addendum_row = addendum.get("model_rows", {}).get(model_id, {})
        addendum_sha = sha256_file(addendum_path)
        if (addendum.get("report_schema") != "siliconflow-cn-zero-tariff-addendum-v1"
                or addendum.get("checked_at") != "2026-10-16"
                or addendum.get("source") != "https://www.siliconflow.cn/pricing"
                or addendum_row.get("input_price") != "免费"
                or addendum_row.get("output_price") != "免费"
                or addendum_row.get("currency") != "¥"
                or addendum_row.get("source_model_id") != model_id):
            raise ValueError("Xing4 zero-tariff addendum is missing or changed")
        probe_root = workspace / ".workbuddy/g06-xing-access-probe-20261016"
        report_path = probe_root / "report.json"
        report = read_json(report_path)
        run_root = Path(report.get("run_dir", "")).resolve()
        if (workspace.resolve() not in run_root.parents
                or probe_root.resolve() not in run_root.parents):
            raise ValueError("Xing4 access probe run directory escaped its isolated workspace")
        expected_artifacts = ("plan.json", "inputs.json", "output.json", "receipt.json",
                              "call-0.json", "call-ledger.jsonl")
        artifact_hashes = report.get("files", {})
        hashes_match = all(name in artifact_hashes
            and Path(artifact_hashes[name].get("path", "")).resolve() == run_root / name
            and sha256_file(run_root / name) == artifact_hashes[name].get("sha256")
            for name in expected_artifacts)
        plan = read_json(run_root / "plan.json")
        receipt = read_json(run_root / "receipt.json")
        call = read_json(run_root / "call-0.json")
        output = read_json(run_root / "output.json")
        ledger = (run_root / "call-ledger.jsonl").read_text(encoding="utf-8").splitlines()
        cost = call.get("cost", {})
        from ..runtime.provider import SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL
        expected_price_hash = SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL[model_id]
        if (report.get("schema") != "g06-xing-cn-free-model-access-probe-v1"
                or report.get("status") != "completed"
                or report.get("model_id") != model_id
                or report.get("base_url") != "https://api.siliconflow.cn/v1"
                or report.get("paid_fallback") is not False
                or report.get("http_status") != 200
                or not hashes_match
                or plan.get("model_id") != model_id
                or plan.get("max_calls") != 1
                or plan.get("max_physical_attempts") != 1
                or plan.get("max_retries_per_call") != 1
                or plan.get("max_cost_usd") > 0.01
                or receipt.get("call_count") != 1
                or receipt.get("measured_usd") != 0.0
                or receipt.get("usage_complete") is not True
                or receipt.get("test_only") is not False
                or receipt.get("request_accounting", {}).get("physical_status_counts") != {"200": 1}
                or receipt.get("request_accounting", {}).get("physical_attempts_exact") is not True
                or cost.get("usd") != 0.0
                or cost.get("usage_complete") is not True
                or cost.get("pricing_region") != "CN"
                or cost.get("pricing_is_account_invoice") is not False
                or cost.get("pricing_evidence_report_sha256") != expected_price_hash
                or cost.get("unknown_retry_attempts") != 0
                or output.get("response") != {"available": True}
                or len(ledger) != 4
                or json.loads(ledger[2]).get("status") != 200
                or report.get("plan_sha256") != plan.get("plan_sha256")):
            raise ValueError("Xing4 lacks a verified zero-price HTTP 200 usage probe")
        return {"model_id": model_id, "base_url": report["base_url"],
            "paid_fallback": False, "http_status": 200, "usage_complete": True,
            "account_access_confirmed": True, "execution_permitted": True,
            "published_tariff_usd": 0.0, "published_tariff_currency": "CNY",
            "actual_account_invoice_verified": False,
            "current_official_pricing_evidence_sha256": pricing_sha,
            "current_official_pricing_html_sha256": pricing["raw_html_sha256"],
            "xing_official_addendum_path": addendum_path.relative_to(workspace).as_posix(),
            "xing_official_addendum_sha256": addendum_sha,
            "probe_artifacts": {"availability_report": {
                "path": report_path.relative_to(workspace).as_posix(),
                "sha256": sha256_file(report_path)},
                **{name: {"path": (run_root / name).relative_to(workspace).as_posix(),
                          "sha256": sha256_file(run_root / name)}
                   for name in (*expected_artifacts, "result.json")
                   if (run_root / name).is_file()}},
            "current_official_model_row": addendum_row}
    if (current_row.get("token_price_cny_per_million") != "0"
            or current_row.get("currency") != "¥"
            or current_row.get("status") != "normal"
            or current_row.get("need_auth") is not False
            or current_row.get("only_by_charge_balance") is not False):
        raise ValueError(f"model {model_id} lacks current official zero-price CN evidence")
    if model_id == "Qwen/Qwen3-8B":
        report_path = pricing_path.parent / "qwen3-access-probe-report.json"
        probe = read_json(report_path)
        run_root = Path(probe["run_dir"]).resolve()
        plan = read_json(run_root / "plan.json")
        receipt = read_json(run_root / "receipt.json")
        call = read_json(run_root / "call-0.json")
        output = read_json(run_root / "output.json")
        cost = call.get("cost", {})
        ledger = (run_root / "call-ledger.jsonl").read_text(encoding="utf-8").splitlines()
        artifact_hashes = probe.get("files", {})
        expected_artifacts = ("plan.json", "inputs.json", "output.json", "receipt.json",
                              "call-0.json", "call-ledger.jsonl")
        hashes_match = all(
            name in artifact_hashes
            and Path(artifact_hashes[name].get("path", "")).resolve() == (run_root / name).resolve()
            and sha256_file(run_root / name) == artifact_hashes[name].get("sha256")
            for name in expected_artifacts)
        availability = output.get("response", {})
        if (probe.get("schema") != "g06-qwen3-cn-free-model-access-probe-v1"
                or probe.get("status") != "completed"
                or probe.get("model_id") != model_id
                or probe.get("base_url") != "https://api.siliconflow.cn/v1"
                or probe.get("paid_fallback") is not False
                or probe.get("http_status") != 200
                or not probe.get("probe_script_path")
                or sha256_file(workspace / probe["probe_script_path"])
                    != probe.get("probe_script_sha256")
                or not hashes_match
                or availability != {"available": True}
                or plan.get("model_id") != model_id
                or plan.get("max_calls") != 1
                or plan.get("max_physical_attempts") != 1
                or plan.get("max_retries_per_call") != 1
                or plan.get("max_cost_usd") > 0.01
                or receipt.get("call_count") != 1
                or receipt.get("measured_usd") != 0.0
                or receipt.get("usage_complete") is not True
                or receipt.get("request_accounting", {}).get("physical_status_counts") != {"200": 1}
                or cost.get("usd") != 0.0
                or cost.get("usage_complete") is not True
                or cost.get("pricing_region") != "CN"
                or cost.get("pricing_evidence_report_sha256") !=
                    "5006960476668bdea624d89604e17709fd657f170b8270d67aaa9028afe9b74e"
                or len(ledger) != 4
                or json.loads(ledger[2]).get("status") != 200
                or probe.get("plan_sha256") != plan.get("plan_sha256")):
            raise ValueError("Qwen3-8B lacks a current zero-price HTTP 200 usage probe")
        return {"model_id": model_id, "base_url": probe["base_url"],
            "paid_fallback": False, "http_status": 200, "usage_complete": True,
            "account_access_confirmed": True, "execution_permitted": True,
            "published_tariff_usd": 0.0, "actual_account_invoice_verified": False,
            "pricing_evidence_report_sha256": cost["pricing_evidence_report_sha256"],
            "current_official_pricing_evidence_sha256": pricing_sha,
            "current_official_pricing_html_sha256": pricing["raw_html_sha256"],
            "current_official_model_row": current_row,
            "probe_artifacts": {"availability_report": {
                "path": report_path.relative_to(workspace).as_posix(),
                "sha256": sha256_file(report_path)},
                **{name: {"path": (run_root / name).relative_to(workspace).as_posix(),
                          "sha256": sha256_file(run_root / name)}
                   for name in ("plan.json", "result.json", "output.json", "inputs.json",
                                "receipt.json", "call-0.json", "call-ledger.jsonl")
                   if (run_root / name).is_file()}}}
    probe_root = workspace / ACCESS_PROBE_DIR / aliases[model_id]
    plan = read_json(probe_root / "plan.json")
    result = read_json(probe_root / "result.json")
    output = read_json(probe_root / "output.json")
    ledger = (probe_root / "call-ledger.jsonl").read_text(encoding="utf-8").splitlines()
    cost = output.get("cost", {})
    if (plan.get("model_id") != model_id
            or plan.get("base_url") != "https://api.siliconflow.cn/v1"
            or plan.get("paid_fallback") is not False
            or result.get("status") != "completed"
            or result.get("model_id") != model_id
            or result.get("actual_account_invoice_verified") is True
            or result.get("published_cn_tariff_amount_usd_equivalent", 0.0) != 0.0
            or result.get("request_accounting", {}).get("physical_http_attempts") != 1
            or result.get("request_accounting", {}).get("physical_status_counts") != {"200": 1}
            or len(ledger) != 4
            or cost.get("usd") != 0.0 or cost.get("usage_complete") is not True
            or cost.get("pricing_region") != "CN"
            or cost.get("pricing_evidence_report_sha256") !=
                "5006960476668bdea624d89604e17709fd657f170b8270d67aaa9028afe9b74e"
            or cost.get("pricing_is_account_invoice") is not False
            or cost.get("pricing_evidence_report_sha256") !=
                "5006960476668bdea624d89604e17709fd657f170b8270d67aaa9028afe9b74e"):
        raise ValueError(f"model {model_id} lacks a verified zero-price completed access probe")
    return {"model_id": model_id, "base_url": plan["base_url"],
        "paid_fallback": False, "http_status": 200,
        "usage_complete": True, "published_tariff_usd": 0.0,
        "account_access_confirmed": True, "execution_permitted": True,
        "actual_account_invoice_verified": False,
        "pricing_evidence_report_sha256": cost["pricing_evidence_report_sha256"],
        "current_official_pricing_evidence_sha256": pricing_sha,
        "current_official_pricing_html_sha256": pricing["raw_html_sha256"],
        "current_official_model_row": current_row,
        "probe_artifacts": {name: {"path": (probe_root / name).relative_to(workspace).as_posix(),
                                   "sha256": sha256_file(probe_root / name)}
                            for name in ("plan.json", "result.json", "output.json",
                                         "call-ledger.jsonl")}}


def build_annotation_plan(packet: dict, cfg_a: dict, cfg_b: dict,
                          workspace: Path | None = None) -> dict:
    """Freeze two model IDs, zero-price evidence, blinding and per-slot plans."""
    workspace = (workspace or repo_root()).resolve()
    validate_reference_packet(packet)
    plan_a = reference_plan(packet, "reviewer_a", cfg_a)
    plan_b = reference_plan(packet, "reviewer_b", cfg_b)
    model_a, model_b = plan_a["model_id"], plan_b["model_id"]
    if model_a == model_b:
        raise ValueError("frozen reference plan requires two distinct model IDs")
    evidence = {model: _free_access_evidence(model, workspace)
                for model in (model_a, model_b)}
    body = {"schema": "instruction-reviewer-synthetic-ai-reference-plan-v2",
        "packet_sha256": packet["packet_sha256"], "natural_distribution": False,
        "annotation_origin": "ai", "model_calls": 0,
        "paid_fallback": False, "max_logical_calls_per_model": len(packet["cases"]),
        "max_physical_attempts_per_model": len(packet["cases"]),
        "max_cost_usd_per_model": 0.25, "max_tokens": 900,
        "blind_payload_fields": ["task_contract", "candidate_strategy"],
        "target_predictions_in_payload": False,
        "construction_strata_in_payload": False,
        "rubric_version": RUBRIC_V2_VERSION,
        "rubric_sha256": sha256_obj(RUBRIC_V2),
        "reviewer_a": {"model_id": model_a, "plan_sha256": plan_a["plan_sha256"],
                       "access_evidence": evidence[model_a]},
        "reviewer_b": {"model_id": model_b, "plan_sha256": plan_b["plan_sha256"],
                       "access_evidence": evidence[model_b]},
        "plans": {"reviewer_a": plan_a, "reviewer_b": plan_b},
        "price_scope": "live official zero-price CN tariff plus completed HTTP 200 usage probe; not invoice evidence",
        "live_pricing_evidence_sha256": evidence[model_a]["current_official_pricing_evidence_sha256"],
        "live_pricing_html_sha256": evidence[model_a]["current_official_pricing_html_sha256"],
        "limitations": ["Synthetic challenge only; not a natural-distribution reference.",
                        "Distinct model IDs do not prove distinct model-family or supply-chain independence."]}
    return {**body, "plan_sha256": sha256_obj(body)}


def reference_execute(packet: dict, slot: str, cfg: dict, plan: dict, *, provider=None) -> dict:
    if requires_reference_shards(packet):
        raise ValueError("v2/v3/v4/v5/v6 synthetic reference cohorts must execute through reference shards")
    metadata = plan.get("protocol_metadata")
    if metadata != {"rubric_version": RUBRIC_V2_VERSION,
                    "rubric_sha256": sha256_obj(RUBRIC_V2)}:
        raise ValueError("new synthetic reference execution requires a frozen rubric-v2 plan")
    if plan != reference_plan(packet, slot, cfg, max_cost_usd=plan["max_cost_usd"]):
        raise ValueError("synthetic reference plan changed")
    inputs = reference_inputs(packet, slot)

    def worker(call):
        rows = []
        for case in packet["cases"]:
            payload = _reference_payload(packet, case)
            raw, provenance = call("instruction_reference", payload, RUBRIC_V2)
            actor = validate_actor(slot, "independent_ai", provenance)
            dimensions, issues = normalize_reference_v2(
                raw, case["strategy_text"], packet["contract"]["text"])
            rows.append({"case_id": case["case_id"], "dimensions": dimensions,
                         "normalization_issues": issues, "actor": actor})
        return _reference_run_output(packet, slot, rows)

    return ai_jobs.execute(plan, inputs, cfg, worker, provider=provider)


def verify_reference_run(packet: dict, slot: str, run_dir: Path) -> dict:
    job = ai_jobs.verify_completed(run_dir, kind="instruction_reference")
    output = job["output"]
    rubric_version = output.get("rubric_version", RUBRIC_V1)
    rubric, response_format, expected_schema = _rubric_protocol(
        rubric_version, historical=rubric_version == RUBRIC_V1)
    if rubric_version == RUBRIC_V1:
        inputs = _historical_v1_inputs(packet, slot)
        expected_output_keys = {"schema", "packet_sha256", "slot", "annotation_origin",
                                "rubric_sha256", "natural_distribution", "cases"}
        expected_plan_metadata = None
    else:
        inputs = reference_inputs(packet, slot, rubric_version=rubric_version)
        expected_output_keys = {"schema", "packet_sha256", "slot", "annotation_origin",
                                "rubric_version", "rubric_sha256", "natural_distribution", "cases"}
        expected_plan_metadata = _rubric_protocol_metadata(rubric_version)
        expected_output_keys |= set(expected_plan_metadata) - {
            "rubric_version", "rubric_sha256"}
    if (job["inputs"] != inputs or output.get("schema") != expected_schema
            or set(output) != expected_output_keys
            or output.get("packet_sha256") != packet["packet_sha256"]
            or output.get("slot") != slot or output.get("annotation_origin") != "ai"
            or (rubric_version != RUBRIC_V1 and output.get("rubric_version") != rubric_version)
            or output.get("rubric_sha256") != sha256_obj(rubric)
            or (expected_plan_metadata is not None and any(
                output.get(key) != value for key, value in
                expected_plan_metadata.items()
                if key not in {"rubric_version", "rubric_sha256"}))
            or job["plan"].get("protocol_metadata") != expected_plan_metadata
            or job["plan"].get("response_format") != response_format
            or output.get("natural_distribution") is not False
            or len(output.get("cases", [])) != len(packet["cases"])
            or job["receipt"].get("usage_complete") is not True
            or job["receipt"].get("measured_usd") != 0.0):
        raise ValueError("synthetic reference job identity or scope changed")
    if job["plan"].get("model_id") not in ALLOWED_FREE_MODELS:
        raise ValueError("synthetic reference used an unapproved model ID")
    blind_payloads = []
    for index, (case, row) in enumerate(zip(packet["cases"], output["cases"])):
        raw = read_json(job["root"] / f"call-{index}.json")
        cost = raw.get("cost", {})
        payload = _reference_payload(packet, case)
        provenance = {
            "model_id": job["plan"]["model_id"],
            "review_run_id": job["root"].name + f"-{index}",
            "rubric_sha256": sha256_obj(rubric), "blind": True,
            "other_review_outputs_consulted": False,
            "test_only": job["receipt"]["test_only"],
        }
        actor = validate_actor(slot, "independent_ai", provenance)
        if rubric_version == RUBRIC_V1:
            dimensions, issues = _reference_dimensions(
                raw["raw_output"], {"strategy_text": case["strategy_text"],
                                    "contract": packet["contract"]})
        elif rubric_version == RUBRIC_V5_VERSION:
            dimensions, issues = normalize_reference_v5(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        elif rubric_version == RUBRIC_V6_VERSION:
            dimensions, issues = normalize_reference_v6(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        elif rubric_version == RUBRIC_V7_VERSION:
            dimensions, issues = normalize_reference_v7(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        elif rubric_version == RUBRIC_V8_VERSION:
            dimensions, issues = normalize_reference_v8(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        elif rubric_version == RUBRIC_V4_VERSION:
            dimensions, issues = normalize_reference_v4(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        else:
            dimensions, issues = normalize_reference_v2(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        if (raw["role"] != "instruction_reference" or raw["system"] != rubric
                or cost.get("usd") != 0.0 or cost.get("usage_complete") is not True
                or cost.get("pricing_region") != "CN"
                or cost.get("pricing_is_account_invoice") is not False
                or cost.get("pricing_evidence_report_sha256") !=
                    "5006960476668bdea624d89604e17709fd657f170b8270d67aaa9028afe9b74e"
                or raw["payload"] != payload
                or row != {"case_id": case["case_id"], "dimensions": dimensions,
                           "normalization_issues": issues, "actor": actor}):
            raise ValueError("synthetic reference raw call or blind payload changed")
        blind_payloads.append(raw["payload"])
    return {**job, "blind_payloads": blind_payloads}


def verify_reference_pair(packet: dict, job_a: dict, job_b: dict, *,
                          historical_v1: bool = False) -> dict:
    """Verify two independently frozen jobs before any consensus is computed."""
    validate_reference_packet(packet)
    plan_a, plan_b = job_a.get("plan", {}), job_b.get("plan", {})
    output_a, output_b = job_a.get("output", {}), job_b.get("output", {})
    version_a = output_a.get("rubric_version", RUBRIC_V1)
    version_b = output_b.get("rubric_version", RUBRIC_V1)
    if version_a != version_b:
        raise ValueError("reference pair uses mixed rubric versions")
    rubric, response_format, run_schema = _rubric_protocol(
        version_a, historical=historical_v1 and version_a == RUBRIC_V1)
    if version_a == RUBRIC_V1 and not historical_v1:
        raise ValueError("rubric v1 pairs are historical-only")
    expected_metadata = (None if version_a == RUBRIC_V1 else
                         {"rubric_version": version_a,
                          "rubric_sha256": sha256_obj(rubric)})
    if version_a != RUBRIC_V1:
        expected_metadata = _rubric_protocol_metadata(version_a)
    for slot, job, plan in (("reviewer_a", job_a, plan_a), ("reviewer_b", job_b, plan_b)):
        expected_inputs = (_historical_v1_inputs(packet, slot) if version_a == RUBRIC_V1
                           else reference_inputs(packet, slot, rubric_version=version_a))
        if (plan.get("kind") != "instruction_reference"
                or plan.get("protocol_metadata") != expected_metadata
                or plan.get("response_format") != response_format
                or plan.get("inputs_sha256") != sha256_obj(expected_inputs)
                or job.get("inputs") != expected_inputs
                or plan.get("plan_sha256") != sha256_obj(
                    {key: value for key, value in plan.items() if key != "plan_sha256"})):
            raise ValueError("reference plan rubric version/hash binding changed")
    if (plan_a.get("model_id") not in ALLOWED_FREE_MODELS
            or plan_b.get("model_id") not in ALLOWED_FREE_MODELS
            or plan_a.get("model_id") == plan_b.get("model_id")):
        raise ValueError("reference pair must use two distinct verified free model IDs")
    expected_ids = [case["case_id"] for case in packet["cases"]]
    for slot, output in (("reviewer_a", output_a), ("reviewer_b", output_b)):
        expected_output_keys = {"schema", "packet_sha256", "slot", "annotation_origin",
                                "rubric_sha256", "natural_distribution", "cases"}
        if version_a != RUBRIC_V1:
            expected_output_keys |= {"rubric_version"}
        if version_a != RUBRIC_V1:
            expected_output_keys |= set(_rubric_protocol_metadata(version_a)) - {
                "rubric_version", "rubric_sha256"}
        rows = output.get("cases", [])
        if (set(output) != expected_output_keys
                or output.get("schema") != run_schema
                or output.get("slot") != slot or output.get("packet_sha256") != packet["packet_sha256"]
                or (version_a != RUBRIC_V1 and output.get("rubric_version") != version_a)
                or output.get("rubric_sha256") != sha256_obj(rubric)
                or (version_a != RUBRIC_V1 and any(
                    output.get(key) != value for key, value in
                    _rubric_protocol_metadata(version_a).items()
                    if key not in {"rubric_version", "rubric_sha256"}))
                or output.get("natural_distribution") is not False
                or [row.get("case_id") for row in rows] != expected_ids):
            raise ValueError(f"{slot} reference does not cover the exact frozen cohort")
        for row in rows:
            if (set(row) != {"case_id", "dimensions", "normalization_issues", "actor"}
                    or set(row["dimensions"]) != {"assertion", "contract"}
                    or row["dimensions"]["assertion"].get("verdict") not in
                        {"claim_free", "contains_claims", "uncertain"}
                    or row["dimensions"]["contract"].get("verdict") not in
                        {"compatible", "incompatible", "uncertain"}
                    or not isinstance(row["normalization_issues"], list)):
                raise ValueError(f"{slot} reference contains an incomplete or unexpected annotation")
    payloads_a, payloads_b = job_a.get("blind_payloads"), job_b.get("blind_payloads")
    if (not isinstance(payloads_a, list) or not isinstance(payloads_b, list)
            or len(payloads_a) != len(expected_ids) or len(payloads_b) != len(expected_ids)):
        raise ValueError("reference pair is missing complete blind payload receipts")
    for index, case in enumerate(packet["cases"]):
        if (case["candidate_sha256"] != sha256_obj(case["candidate_text"])
                or case["candidate_text"] != compose_task_instruction(
                    case["strategy_text"], CLINIFACT_CONTRACT)):
            raise ValueError("reference candidate hash or contract composition changed")
        expected = _reference_payload(packet, case)
        left, right = payloads_a[index], payloads_b[index]
        expected_payload_keys = {"task_contract", "candidate_strategy"}
        if version_a in {RUBRIC_V3_VERSION, RUBRIC_V4_VERSION, RUBRIC_V5_VERSION,
                          RUBRIC_V8_VERSION}:
            expected_payload_keys.add("reference_spans")
        if (set(left) != expected_payload_keys
                or left != expected or right != expected or left != right
                or left["task_contract"] != packet["contract"]
                or left["candidate_strategy"] != case["strategy_text"]):
            raise ValueError("reference pair saw different or unblinded candidate payloads")
    return {"model_ids": [plan_a["model_id"], plan_b["model_id"]],
        "rubric_version": version_a, "rubric_sha256": sha256_obj(rubric),
        "distinct_model_ids_verified": True, "identical_blind_payloads_verified": True,
        "complete_case_coverage_verified": True, "contract_and_candidate_hashes_verified": True,
        "target_predictions_visible": False, "construction_strata_visible": False}


def freeze_reference(packet: dict, run_a: Path, run_b: Path) -> dict:
    if requires_reference_shards(packet):
        raise ValueError("v2+ synthetic reference cohorts must freeze through their versioned shard freeze adapter")
    a = verify_reference_run(packet, "reviewer_a", run_a)
    b = verify_reference_run(packet, "reviewer_b", run_b)
    pair_receipt = verify_reference_pair(packet, a, b)
    model_a, model_b = a["plan"]["model_id"], b["plan"]["model_id"]
    actors = [a["output"]["cases"][0]["actor"], b["output"]["cases"][0]["actor"]]
    validate_pair(actors, "packet_builder")
    rows, disagreements = [], []
    for case, left, right in zip(packet["cases"], a["output"]["cases"],
                                 b["output"]["cases"]):
        verdicts = {}
        for dimension in ("assertion", "contract"):
            lval = left["dimensions"][dimension]["verdict"]
            rval = right["dimensions"][dimension]["verdict"]
            verdicts[dimension] = lval if lval == rval else "uncertain"
            if lval != rval:
                disagreements.append({"case_id": case["case_id"], "dimension": dimension,
                    "reviewer_a": lval, "reviewer_b": rval, "resolution": "uncertain"})
        rows.append({"case_id": case["case_id"], "verdicts": verdicts})
    coverage = _class_coverage(rows)
    thresholds_met = all(coverage[dimension][key] >= THRESHOLDS[threshold]
        for dimension in ("assertion", "contract")
        for key, threshold in (("approval_reference", "min_approval_reference"),
                               ("rejection_reference", "min_rejection_reference")))
    return {"schema": FROZEN_SCHEMA, "packet_sha256": packet["packet_sha256"],
        "scope": "unseen_synthetic_clinifact_instruction_reviewer_challenge",
        "rubric_version": RUBRIC_V2_VERSION, "rubric_sha256": sha256_obj(RUBRIC_V2),
        "annotation_origin": "ai", "reference_method": "two_distinct_model_ids_consensus",
        "pair_verification": pair_receipt,
        "natural_distribution": False, "model_identity_independence_verified": True,
        "review_run_independence_verified": True,
        "model_source_independence_verified": False,
        "model_source_independence_status": "unverified; distinct IDs do not establish distinct model families or supply chains",
        "reviewer_a": {"model_id": model_a,
                       "receipt_sha256": a["receipt_file_sha256"]},
        "reviewer_b": {"model_id": model_b,
                       "receipt_sha256": b["receipt_file_sha256"]},
        "disagreements": disagreements, "cases": rows,
        "reference_class_coverage": coverage,
        "reference_class_coverage_sufficient": thresholds_met,
        "reference_test_only": a["receipt"]["test_only"] or b["receipt"]["test_only"],
        "limitations": ["Synthetic challenge only; does not establish natural-distribution validity.",
                        "AI consensus is a reference proxy, not human ground truth."]}


def _class_coverage(rows: list[dict]) -> dict:
    labels = {"assertion": ("claim_free", "contains_claims"),
              "contract": ("compatible", "incompatible")}
    return {dimension: {
        "approval_reference": sum(row["verdicts"][dimension] == approval for row in rows),
        "rejection_reference": sum(row["verdicts"][dimension] == rejection for row in rows),
        "uncertain_reference": sum(row["verdicts"][dimension] == "uncertain" for row in rows),
    } for dimension, (approval, rejection) in labels.items()}

