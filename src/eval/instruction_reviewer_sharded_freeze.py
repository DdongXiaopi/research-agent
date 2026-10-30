"""Freeze a synthetic diagnostic reference from independently verified shards.

This adapter deliberately emits its own schema.  A shard cohort is not an
``ai_jobs`` cohort receipt and it is not a natural-distribution reference.
Every invocation re-aggregates both reviewers from the frozen packet,
schedule, per-case state, and the underlying completed ``ai_jobs`` receipts.
"""
from __future__ import annotations

import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
from contextlib import contextmanager

from . import instruction_reviewer_cohort_reference as reference
from . import instruction_reviewer_reference_shards as shards
from .annotation_policy import validate_actor, validate_pair
from .instruction_reviewer_cohort_v4 import SCHEMA as COHORT_V4_SCHEMA
from .instruction_reviewer_cohort_v5 import SCHEMA as COHORT_V5_SCHEMA
from .instruction_reviewer_cohort_v6 import SCHEMA as COHORT_V6_SCHEMA
from .instruction_reviewer_cohort_v7 import SCHEMA as COHORT_V7_SCHEMA
from .instruction_reviewer_cohort_v8 import SCHEMA as COHORT_V8_SCHEMA
from .instruction_reviewer_cohort_v9 import SCHEMA as COHORT_V9_SCHEMA
from .instruction_reviewer_cohort_v10 import SCHEMA as COHORT_V10_SCHEMA
from .instruction_reviewer_cohort_v11 import SCHEMA as COHORT_V11_SCHEMA
from .instruction_reviewer_cohort_v12 import SCHEMA as COHORT_V12_SCHEMA
from .instruction_reviewer_cohort_v13 import SCHEMA as COHORT_V13_SCHEMA
from .instruction_reviewer_cohort_v14 import SCHEMA as COHORT_V14_SCHEMA
from .instruction_reviewer_pair_gate import family_registry_sha256, model_family
from .instruction_reviewer_rubric_v2 import RUBRIC_V2, normalize_reference_v2
from .instruction_reviewer_rubric_v3 import (
    RUBRIC_V3, normalize_reference_v3,
    protocol_hash as rubric_v3_protocol_hash,
    splitter_hash as rubric_v3_splitter_hash,
    REFERENCE_FORMAT_V3,
)
from .instruction_reviewer_rubric_v4 import (
    RUBRIC_V4, normalize_reference_v4,
    protocol_hash as rubric_v4_protocol_hash,
    splitter_hash as rubric_v4_splitter_hash,
    REFERENCE_FORMAT_V4,
)
from .instruction_reviewer_rubric_v5 import (
    RUBRIC_V5, normalize_reference_v5,
    protocol_hash as rubric_v5_protocol_hash,
    splitter_hash as rubric_v5_splitter_hash,
    REFERENCE_FORMAT_V5,
    PROTOCOL_SCHEMA_V5 as RUBRIC_V5_PROTOCOL_SCHEMA,
    NORMALIZER_VERSION_V5 as RUBRIC_V5_NORMALIZER_VERSION,
    REFERENCE_MAX_OUTPUT_TOKENS as RUBRIC_V5_MAX_OUTPUT_TOKENS,
)
from .instruction_reviewer_rubric_v6 import (
    RUBRIC_V6, normalize_reference_v6,
    protocol_hash as rubric_v6_protocol_hash, splitter_hash as rubric_v6_splitter_hash,
    REFERENCE_FORMAT_V6 as RUBRIC_V6_FORMAT,
    PROTOCOL_SCHEMA_V6 as RUBRIC_V6_PROTOCOL_SCHEMA,
    NORMALIZER_VERSION_V6 as RUBRIC_V6_NORMALIZER_VERSION,
    REFERENCE_MAX_OUTPUT_TOKENS as RUBRIC_V6_MAX_OUTPUT_TOKENS,
)
from .instruction_reviewer_rubric_v7 import (
    RUBRIC_V7, normalize_reference_v7,
    protocol_hash as rubric_v7_protocol_hash, splitter_hash as rubric_v7_splitter_hash,
    RESPONSE_FORMAT_TEMPLATE_V7 as RUBRIC_V7_FORMAT,
    PROTOCOL_SCHEMA_V7 as RUBRIC_V7_PROTOCOL_SCHEMA,
    NORMALIZER_VERSION_V7 as RUBRIC_V7_NORMALIZER_VERSION,
    REFERENCE_MAX_OUTPUT_TOKENS as RUBRIC_V7_MAX_OUTPUT_TOKENS,
    normalizer_source_hash as rubric_v7_normalizer_source_hash,
)
from .instruction_reviewer_rubric_v8 import (
    RUBRIC_V8, normalize_reference_v8,
    RESPONSE_FORMAT_TEMPLATE_V8 as RUBRIC_V8_FORMAT,
    PROTOCOL_SCHEMA_V8 as RUBRIC_V8_PROTOCOL_SCHEMA,
    NORMALIZER_VERSION_V8 as RUBRIC_V8_NORMALIZER_VERSION,
    REFERENCE_MAX_OUTPUT_TOKENS as RUBRIC_V8_MAX_OUTPUT_TOKENS,
    protocol_hash_v8 as rubric_v8_protocol_hash,
    splitter_hash as rubric_v8_splitter_hash,
    normalizer_source_hash_v8 as rubric_v8_normalizer_source_hash,
)
from ..runtime.provider import SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL
from ..util import read_json, sha256_obj


FROZEN_SCHEMA = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v2"
FROZEN_SCHEMA_V3 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v3"
FROZEN_SCHEMA_V4 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v4"
FROZEN_SCHEMA_V5 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v5"
FROZEN_SCHEMA_V6 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v6"
FROZEN_SCHEMA_V7 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v7"
FROZEN_SCHEMA_V8 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v8"
FROZEN_SCHEMA_V9 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v9"
FROZEN_SCHEMA_V10 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v10"
FROZEN_SCHEMA_V11 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v11"
FROZEN_SCHEMA_V12 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v12"
FROZEN_SCHEMA_V6 = "instruction-reviewer-synthetic-diagnostic-frozen-reference-v6"


def _validate_packet_for_freeze(packet: dict) -> None:
    """Validate a packet, allowing only the exact preflight v7 build identity.

    That immutable packet was fully source/exposure-audited before plans were
    frozen. Later runner hardening changed the builder file hash, so freeze
    rechecks its packet/case/protocol/source integrity without rescanning the
    historical exposure corpus. Every other packet gets the full validator.
    """
    if packet.get("schema") == COHORT_V14_SCHEMA:
        import_module(".instruction_reviewer_cohort_v14", package=__package__).validate_packet(
            packet, reference.repo_root())
        return
    if packet.get("schema") in {COHORT_V11_SCHEMA, COHORT_V12_SCHEMA, COHORT_V13_SCHEMA}:
        version = ("v13" if packet.get("schema") == COHORT_V13_SCHEMA else
                   "v12" if packet.get("schema") == COHORT_V12_SCHEMA else "v11")
        cohort = import_module(f".instruction_reviewer_cohort_{version}", package=__package__)
        cohort.validate_packet(packet, reference.repo_root())
        return
    if packet.get("schema") == COHORT_V7_SCHEMA:
        from . import instruction_reviewer_cohort_v7 as cohort_v7
        if (packet.get("packet_sha256") == cohort_v7.LEGACY_PREFLIGHT_PACKET_SHA256
                and packet.get("builder_source_sha256")
                    == cohort_v7.LEGACY_PREFLIGHT_BUILDER_SHA256):
            cohort_v7.validate_packet_integrity(packet)
            return
    reference.validate_reference_packet(packet)


def _protocol_bindings(packet: dict) -> dict:
    if packet.get("schema") in {COHORT_V13_SCHEMA, COHORT_V14_SCHEMA}:
        return shards._protocol_bindings(packet)
    if packet.get("schema") in {COHORT_V11_SCHEMA, COHORT_V12_SCHEMA}:
        # Reuse the shard boundary's version-specific packet, rubric and target
        # identity checks; freeze persists the same frozen binding tuple.
        return shards._protocol_bindings(packet)
    if packet.get("schema") in {COHORT_V9_SCHEMA, COHORT_V10_SCHEMA}:
        bindings = {"rubric_version": "v7", "rubric_sha256": sha256_obj(RUBRIC_V7),
            "response_format_template_sha256": sha256_obj(RUBRIC_V7_FORMAT),
            "protocol_hash": rubric_v7_protocol_hash(),
            "splitter_hash": rubric_v7_splitter_hash(),
            "protocol_schema": RUBRIC_V7_PROTOCOL_SCHEMA,
            "normalizer_version": RUBRIC_V7_NORMALIZER_VERSION,
            "normalizer_source_hash": rubric_v7_normalizer_source_hash(),
            "max_output_tokens": RUBRIC_V7_MAX_OUTPUT_TOKENS}
        if packet.get("schema") == COHORT_V10_SCHEMA:
            from ..evolution.instruction_review import prompt_grounding_protocol
            from ..evolution.task_contracts import CLINIFACT_CONTRACT
            target_protocol = packet.get("target_protocol")
            current_protocol = prompt_grounding_protocol(CLINIFACT_CONTRACT)
            if (not isinstance(target_protocol, dict) or target_protocol != current_protocol
                    or target_protocol.get("version") != "instruction-review-v7-span-atomic"):
                raise ValueError("v10 target protocol differs from the current grounding protocol")
            bindings.update({"target_protocol_version": target_protocol["version"],
                "target_protocol_sha256": sha256_obj(target_protocol),
                "target_protocol_source_sha256": target_protocol.get("source_sha256")})
        return bindings
    if packet.get("schema") == COHORT_V8_SCHEMA:
        return {"rubric_version": "v6", "rubric_sha256": sha256_obj(RUBRIC_V6),
            "reference_format_template_sha256": sha256_obj(RUBRIC_V6_FORMAT),
            "protocol_hash": rubric_v6_protocol_hash(),
            "splitter_hash": rubric_v6_splitter_hash(),
            "protocol_schema": RUBRIC_V6_PROTOCOL_SCHEMA,
            "normalizer_version": RUBRIC_V6_NORMALIZER_VERSION,
            "max_output_tokens": RUBRIC_V6_MAX_OUTPUT_TOKENS}
    if packet.get("schema") == COHORT_V7_SCHEMA:
        return {"rubric_version": "v5", "rubric_sha256": sha256_obj(RUBRIC_V5),
            "reference_format_sha256": sha256_obj(REFERENCE_FORMAT_V5),
            "protocol_hash": rubric_v5_protocol_hash(),
            "splitter_hash": rubric_v5_splitter_hash(),
            "protocol_schema": RUBRIC_V5_PROTOCOL_SCHEMA,
            "normalizer_version": RUBRIC_V5_NORMALIZER_VERSION,
            "max_output_tokens": RUBRIC_V5_MAX_OUTPUT_TOKENS}
    if packet.get("schema") == COHORT_V6_SCHEMA:
        return {"rubric_version": "v4", "rubric_sha256": sha256_obj(RUBRIC_V4),
            "reference_format_sha256": sha256_obj(REFERENCE_FORMAT_V4),
            "protocol_hash": rubric_v4_protocol_hash(),
            "splitter_hash": rubric_v4_splitter_hash()}
    if packet.get("schema") == COHORT_V5_SCHEMA:
        return {"rubric_version": "v3", "rubric_sha256": sha256_obj(RUBRIC_V3),
            "reference_format_sha256": sha256_obj(REFERENCE_FORMAT_V3),
            "protocol_hash": rubric_v3_protocol_hash(),
            "splitter_hash": rubric_v3_splitter_hash()}
    return {"rubric_version": reference.RUBRIC_V2_VERSION,
            "rubric_sha256": sha256_obj(RUBRIC_V2)}


def _raw_call_for_shard(schedule: dict, shard: dict, packet: dict) -> tuple[dict, dict]:
    """Recheck the raw response-to-label path after the shard aggregator verifies receipts."""
    index = shard.get("index")
    if not isinstance(index, int) or index < 0 or index >= len(schedule["shards"]):
        raise ValueError("shard manifest index is outside the frozen schedule")
    planned_shard = schedule["shards"][index]
    if shard.get("case_id") != planned_shard.get("case_id"):
        raise ValueError("shard manifest case identity differs from the frozen schedule")
    job = shards.ai_jobs.verify_completed(shard["run_dir"], kind="instruction_reference")
    if (job.get("plan") != planned_shard["plan"]
            or job.get("inputs") != planned_shard["inputs"]
            or job.get("receipt_file_sha256") != shard["receipt_file_sha256"]):
        raise ValueError("shard job does not match the aggregate plan or receipt binding")
    root = Path(job["root"])
    raw = read_json(root / "call-0.json")
    call = read_json(root / "calls.json")
    receipt = job["receipt"]
    cost = raw.get("cost", {})
    model_id = schedule["model_id"]
    bindings = _protocol_bindings(packet)
    rubric = (RUBRIC_V8 if bindings["rubric_version"] == "v8" else
              RUBRIC_V7 if bindings["rubric_version"] == "v7" else
              RUBRIC_V6 if bindings["rubric_version"] == "v6" else
              RUBRIC_V5 if bindings["rubric_version"] == "v5" else
              RUBRIC_V4 if bindings["rubric_version"] == "v4" else
              RUBRIC_V3 if bindings["rubric_version"] == "v3" else RUBRIC_V2)
    if (len(call) != 1 or raw.get("role") != "instruction_reference"
            or raw.get("system") != rubric
            or raw.get("payload") != planned_shard["inputs"]["blind_payload"]
            or raw.get("user") != json.dumps(raw["payload"], ensure_ascii=False, sort_keys=True)
            or cost.get("usd") != 0.0 or cost.get("usage_complete") is not True
            or cost.get("pricing_region") != "CN"
            or cost.get("pricing_is_account_invoice") is not False
            or cost.get("pricing_evidence_report_sha256")
                != SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL.get(model_id)
            or receipt.get("test_only") is not False):
        raise ValueError("shard raw call is not the frozen blind, zero-price reference request")
    # A completed HTTP response may still be malformed model output. Keep its
    # exact bytes receipt-bound and let the versioned normalizer map it to
    # dimension-level uncertain; rejecting the whole pair here would discard
    # valid evidence from the other cases and misclassify a scored failure as
    # missing transport evidence.
    raw_output = raw.get("raw_output")
    if not isinstance(raw_output, str):
        raise ValueError("shard raw response is missing textual content")
    if (call[0].get("index") != 0 or call[0].get("status") != "completed"
            or call[0].get("role") != raw["role"]
            or call[0].get("input_sha256") != sha256_obj(raw["payload"])
            or call[0].get("response_sha256") != sha256_obj(raw_output)
            or call[0].get("cost") != cost):
        raise ValueError("shard raw response differs from its measured call ledger")
    return job, raw


def _aggregate_artifacts_root(supplied_aggregate: dict,
                              *, allow_legacy_layout: bool = False) -> Path:
    """Resolve the isolated RSI_HOME bound by a reviewer aggregate's receipts."""
    manifest = supplied_aggregate.get("shards") if isinstance(supplied_aggregate, dict) else None
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("reviewer aggregate has no completed-shard artifact manifest")
    roots = set()
    for row in manifest:
        if not isinstance(row, dict) or not isinstance(row.get("run_dir"), str):
            raise ValueError("reviewer aggregate contains an invalid artifact path")
        run_dir = Path(row["run_dir"]).resolve()
        if (run_dir.parent.name == "ai_annotations"
                and run_dir.parent.parent.name == "evaluation_only"):
            roots.add(run_dir.parent.parent.parent)
        elif allow_legacy_layout:
            # v3-v6 fixtures predate the evaluation_only/ai_annotations
            # layout.  Keep their compatibility path bounded to one common
            # run root; v7+ production aggregates remain strict by default.
            roots.add(run_dir.parent)
        else:
            raise ValueError("reviewer run is outside the expected isolated AI artifact layout")
    if len(roots) != 1:
        raise ValueError("reviewer aggregate mixes runs from multiple RSI_HOME roots")
    return roots.pop()


@contextmanager
def _reviewer_rsi_home(supplied_aggregate: dict, *, allow_legacy_layout: bool = False):
    """Verify each reviewer under its own isolated RSI_HOME, then restore env."""
    root = _aggregate_artifacts_root(supplied_aggregate,
                                     allow_legacy_layout=allow_legacy_layout)
    previous = os.environ.get("RSI_HOME")
    os.environ["RSI_HOME"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RSI_HOME", None)
        else:
            os.environ["RSI_HOME"] = previous


def _verify_reviewer(packet: dict, schedule: dict, state_dir: str | Path,
                     supplied_aggregate: dict, slot: str) -> dict:
    with _reviewer_rsi_home(
            supplied_aggregate,
            allow_legacy_layout=packet.get("schema") in {
                COHORT_V4_SCHEMA, COHORT_V5_SCHEMA, COHORT_V6_SCHEMA}):
        return _verify_reviewer_under_isolation(
            packet, schedule, state_dir, supplied_aggregate, slot)


def _verified_shard_aggregate(packet: dict, schedule: dict, state_dir: str | Path,
                              supplied_aggregate: dict, slot: str) -> dict:
    """Rebuild receipts, allowing only the exact all-absent v10 legacy bindings."""
    if packet.get("schema") == COHORT_V10_SCHEMA:
        legacy_v10 = shards.validate_shard_schedule(packet, schedule, slot,
            allow_legacy_v10=True)
    else:
        legacy_v10 = shards.validate_shard_schedule(packet, schedule, slot)
    expected_schedule_bindings = shards._schedule_protocol_bindings(packet, legacy_v10)
    if any(schedule.get(key) != value for key, value in expected_schedule_bindings.items()):
        raise ValueError(f"{slot} schedule does not bind the exact rubric/format/protocol/splitter")
    aggregate = shards.aggregate_verified_shards(packet, schedule, state_dir)
    aggregate_matches = (supplied_aggregate == aggregate or
        (legacy_v10 and shards._legacy_aggregate_matches(aggregate, supplied_aggregate)))
    if not aggregate_matches:
        raise ValueError(f"{slot} aggregate is missing, changed, or not bound to verified shard receipts")
    return aggregate


def _verify_reviewer_under_isolation(packet: dict, schedule: dict,
                     state_dir: str | Path, supplied_aggregate: dict,
                     slot: str) -> dict:
    aggregate = _verified_shard_aggregate(packet, schedule, state_dir,
        supplied_aggregate, slot)
    if (aggregate.get("schema") != shards.AGGREGATE_SCHEMA
            or aggregate.get("status") != "verified_complete"
            or aggregate.get("aggregate_sha256") != sha256_obj(
                {key: value for key, value in aggregate.items() if key != "aggregate_sha256"})
            or aggregate.get("packet_sha256") != packet["packet_sha256"]
            or aggregate.get("schedule_sha256") != schedule["schedule_sha256"]
            or aggregate.get("slot") != slot
            or aggregate.get("model_id") != schedule["model_id"]
            or any(aggregate.get(key) != value for key, value in _protocol_bindings(packet).items())
            or aggregate.get("annotation_origin") != "ai"
            or aggregate.get("natural_distribution") is not False
            or aggregate.get("case_count") != len(packet["cases"])
            or aggregate.get("ai_jobs_receipt_fabricated") is not False):
        raise ValueError(f"{slot} aggregate scope or integrity fields changed")

    if len(aggregate.get("cases", [])) != len(packet["cases"]):
        raise ValueError(f"{slot} aggregate case coverage is incomplete")
    if len(aggregate.get("shards", [])) != len(packet["cases"]):
        raise ValueError(f"{slot} aggregate receipt manifest is incomplete")

    expected_ids = [case["case_id"] for case in packet["cases"]]
    actual_ids = [row.get("case_id") for row in aggregate["cases"]]
    manifest_ids = [row.get("case_id") for row in aggregate["shards"]]
    if actual_ids != expected_ids or manifest_ids != expected_ids:
        raise ValueError(f"{slot} aggregate case order or coverage differs from the packet")

    receipts, plans, run_ids = set(), set(), set()
    case_receipts = []
    for index, (case, case_result, manifest, shard) in enumerate(zip(
            packet["cases"], aggregate["cases"], aggregate["shards"], schedule["shards"])):
        receipt_sha = manifest.get("receipt_file_sha256")
        plan_sha = manifest.get("plan_sha256")
        run_dir = manifest.get("run_dir")
        if (manifest.get("index") != index or shard.get("index") != index
                or manifest.get("plan_sha256") != shard["plan"].get("plan_sha256")
                or not isinstance(receipt_sha, str) or len(receipt_sha) != 64
                or not isinstance(plan_sha, str) or len(plan_sha) != 64
                or not isinstance(run_dir, str) or not run_dir):
            raise ValueError(f"{slot} case {index} receipt/plan identity is incomplete or misbound")
        if receipt_sha in receipts:
            raise ValueError(f"{slot} aggregate contains a duplicate receipt hash")
        if plan_sha in plans:
            raise ValueError(f"{slot} aggregate contains a duplicate shard plan hash")
        receipts.add(receipt_sha)
        plans.add(plan_sha)
        job, raw = _raw_call_for_shard(schedule, manifest, packet)
        run_id = job["root"].name + "-0"
        if run_id in run_ids:
            raise ValueError(f"{slot} aggregate contains a duplicate review run identity")
        run_ids.add(run_id)
        bindings = _protocol_bindings(packet)
        if bindings["rubric_version"] == "v8":
            dimensions, issues = normalize_reference_v8(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
            if dimensions is None:
                reason = ";".join(issues) or "invalid_reference_output"
                dimensions = {
                    "assertion": {"verdict": "uncertain", "claims": [], "uncertain": [],
                        "verdict_reason_code": "invalid_reference_evidence_fail_closed",
                        "normalization_failure": reason},
                    "contract": {"verdict": "uncertain", "issues": [], "uncertain": [],
                        "verdict_reason_code": "invalid_reference_evidence_fail_closed",
                        "normalization_failure": reason},
                }
        elif bindings["rubric_version"] == "v7":
            dimensions, issues = normalize_reference_v7(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        elif bindings["rubric_version"] == "v6":
            dimensions, issues = normalize_reference_v6(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        elif bindings["rubric_version"] == "v5":
            dimensions, issues = normalize_reference_v5(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        elif bindings["rubric_version"] == "v4":
            dimensions, issues = normalize_reference_v4(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        elif bindings["rubric_version"] == "v3":
            dimensions, issues = normalize_reference_v3(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        else:
            dimensions, issues = normalize_reference_v2(
                raw["raw_output"], case["strategy_text"], packet["contract"]["text"])
        actor = validate_actor(slot, "independent_ai", {
            "model_id": schedule["model_id"], "review_run_id": run_id,
            # ai_jobs records the exact system instruction hash in actor
            # provenance. Protocol bindings may use a canonical raw-text
            # rubric hash, so derive this actor field from the receipt-bound
            # call itself rather than from the schedule metadata.
            "rubric_sha256": sha256_obj(raw["system"]), "blind": True,
            "other_review_outputs_consulted": False,
            "test_only": job["receipt"]["test_only"]})
        expected_result = {"case_id": case["case_id"], "dimensions": dimensions,
                           "normalization_issues": issues, "actor": actor}
        # Aggregates are persisted and reloaded as JSON. The v3 span splitter
        # returns tuple-valued coverage entries in memory, which JSON encodes
        # as arrays. Compare the same JSON representation used by the receipt
        # path so a verified round-trip does not fail on tuple/list shape alone.
        expected_result = json.loads(json.dumps(expected_result, ensure_ascii=False))
        if case_result != expected_result:
            raise ValueError(f"{slot} case result differs from its receipt-bound raw response")
        case_receipt = {"case_id": case["case_id"], "index": index,
            "plan_sha256": plan_sha, "receipt_file_sha256": receipt_sha,
            "record_sha256": manifest.get("record_sha256"),
            "review_run_id": run_id}
        if bindings["rubric_version"] in {"v5", "v6", "v7", "v8"}:
            case_receipt.update({
                "blind_payload_sha256": sha256_obj(raw["payload"]),
                "raw_response_sha256": sha256_obj(raw["raw_output"]),
                "normalizer_version": bindings["normalizer_version"],
            })
        case_receipts.append(case_receipt)
    return {"slot": slot, "model_id": schedule["model_id"],
        "model_family": model_family(schedule["model_id"]),
        "schedule_sha256": schedule["schedule_sha256"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "blind_payload_sha256": sha256_obj([
            row["inputs"]["blind_payload"] for row in schedule["shards"]]),
        "case_receipts": case_receipts,
        "cases": aggregate["cases"]}


def freeze_synthetic_diagnostic_reference(packet: dict,
        schedule_a: dict, state_a: str | Path, aggregate_a: dict,
        schedule_b: dict, state_b: str | Path, aggregate_b: dict) -> dict:
    """Reverify two shard aggregates and freeze diagnostic consensus labels."""
    _validate_packet_for_freeze(packet)
    if packet.get("schema") not in {COHORT_V4_SCHEMA, COHORT_V5_SCHEMA, COHORT_V6_SCHEMA,
                                     COHORT_V7_SCHEMA, COHORT_V8_SCHEMA, COHORT_V9_SCHEMA,
                                     COHORT_V10_SCHEMA, COHORT_V11_SCHEMA,
                                     COHORT_V12_SCHEMA, COHORT_V13_SCHEMA,
                                     COHORT_V14_SCHEMA} or packet.get("natural_distribution") is not False:
        raise ValueError("shard freeze adapter accepts only synthetic G06 v4-v14 packets")
    reviewer_a = _verify_reviewer(packet, schedule_a, state_a, aggregate_a, "reviewer_a")
    reviewer_b = _verify_reviewer(packet, schedule_b, state_b, aggregate_b, "reviewer_b")
    if reviewer_a["model_id"] == reviewer_b["model_id"]:
        raise ValueError("reference reviewers must use distinct model IDs")
    if reviewer_a["model_family"] == reviewer_b["model_family"]:
        raise ValueError("reference reviewers must use distinct registered model families")
    if reviewer_a["blind_payload_sha256"] != reviewer_b["blind_payload_sha256"]:
        raise ValueError("reference reviewers were not given identical blind payloads")
    if [row["case_id"] for row in reviewer_a["case_receipts"]] != [
            row["case_id"] for row in reviewer_b["case_receipts"]]:
        raise ValueError("reviewer receipt manifests do not cover identical cases in order")
    run_ids = [row["review_run_id"] for reviewer in (reviewer_a, reviewer_b)
               for row in reviewer["case_receipts"]]
    receipt_hashes = [row["receipt_file_sha256"] for reviewer in (reviewer_a, reviewer_b)
                      for row in reviewer["case_receipts"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("reviewer shards do not have independent run identities")
    if len(receipt_hashes) != len(set(receipt_hashes)):
        raise ValueError("reviewers share or duplicate a per-case receipt")

    first_actors = [reviewer["cases"][0]["actor"] for reviewer in (reviewer_a, reviewer_b)]
    validate_pair(first_actors, "packet_builder")
    disagreements, rows = [], []
    for case, left, right in zip(packet["cases"], reviewer_a["cases"], reviewer_b["cases"]):
        verdicts = {}
        for dimension in ("assertion", "contract"):
            left_verdict = left["dimensions"][dimension]["verdict"]
            right_verdict = right["dimensions"][dimension]["verdict"]
            allowed = ({"claim_free", "contains_claims", "uncertain"}
                       if dimension == "assertion"
                       else {"compatible", "incompatible", "uncertain"})
            if left_verdict not in allowed or right_verdict not in allowed:
                raise ValueError(f"reference contains an unknown {dimension} verdict")
            verdicts[dimension] = left_verdict if left_verdict == right_verdict else "uncertain"
            if left_verdict != right_verdict:
                disagreements.append({"case_id": case["case_id"], "dimension": dimension,
                    "reviewer_a": left_verdict, "reviewer_b": right_verdict,
                    "resolution": "uncertain"})
        rows.append({"case_id": case["case_id"], "verdicts": verdicts})

    coverage = reference._class_coverage(rows)
    sufficient = all(coverage[dimension][key] >= reference.THRESHOLDS[threshold]
        for dimension in ("assertion", "contract")
        for key, threshold in (("approval_reference", "min_approval_reference"),
                               ("rejection_reference", "min_rejection_reference")))
    pair_gate_report = {
        "schema": "g06-synthetic-reference-pair-verification-v1",
        "status": "verified_zero_call_pair",
        "packet_sha256": packet["packet_sha256"],
        **_protocol_bindings(packet),
        "family_registry_sha256": family_registry_sha256(),
        "reviewer_a": {"model_id": reviewer_a["model_id"],
            "model_family": reviewer_a["model_family"],
            "schedule_sha256": reviewer_a["schedule_sha256"],
            "aggregate_sha256": reviewer_a["aggregate_sha256"],
            "case_receipt_hashes": [row["receipt_file_sha256"]
                for row in reviewer_a["case_receipts"]]},
        "reviewer_b": {"model_id": reviewer_b["model_id"],
            "model_family": reviewer_b["model_family"],
            "schedule_sha256": reviewer_b["schedule_sha256"],
            "aggregate_sha256": reviewer_b["aggregate_sha256"],
            "case_receipt_hashes": [row["receipt_file_sha256"]
                for row in reviewer_b["case_receipts"]]},
        "checks": {"distinct_model_ids": True, "distinct_model_families": True,
            "identical_blind_payloads": True, "complete_ordered_case_coverage": True,
            "unique_per_case_receipts": True},
        "natural_distribution": False, "promotion_authorized": False,
    }
    protocol_bindings = _protocol_bindings(packet)
    frozen_schema = (FROZEN_SCHEMA_V12 if packet.get("schema") == COHORT_V14_SCHEMA else
                     FROZEN_SCHEMA_V11 if packet.get("schema") == COHORT_V13_SCHEMA else
                     FROZEN_SCHEMA_V10 if packet.get("schema") == COHORT_V12_SCHEMA else
                     FROZEN_SCHEMA_V9 if packet.get("schema") == COHORT_V11_SCHEMA else
                     FROZEN_SCHEMA_V8 if packet.get("schema") == COHORT_V10_SCHEMA else
                     {"v2": FROZEN_SCHEMA, "v3": FROZEN_SCHEMA_V3,
                      "v4": FROZEN_SCHEMA_V4, "v5": FROZEN_SCHEMA_V5,
                      "v6": FROZEN_SCHEMA_V6, "v7": FROZEN_SCHEMA_V7}
                     [protocol_bindings["rubric_version"]])
    body = {"schema": frozen_schema, "status": "synthetic_diagnostic_frozen",
        "packet_sha256": packet["packet_sha256"],
        "scope": "fresh_synthetic_clinifact_instruction_reviewer_challenge",
        **protocol_bindings, "annotation_origin": "ai",
        "synthetic_only": True, "natural_distribution": False,
        "promotion_authorized": False,
        "reference_method": "two_distinct_model_families_consensus",
        "family_registry_sha256": family_registry_sha256(),
        "pair_gate_report": pair_gate_report,
        "pair_gate_report_sha256": sha256_obj(pair_gate_report),
        "pair_verification": {"distinct_model_ids_verified": True,
            "distinct_model_families_verified": True,
            "identical_blind_payloads_verified": True,
            "complete_case_coverage_verified": True,
            "unique_per_case_receipts_verified": True,
            "model_source_independence_verified": False},
        "reviewer_a": {key: reviewer_a[key] for key in (
            "model_id", "model_family", "schedule_sha256", "aggregate_sha256",
            "blind_payload_sha256", "case_receipts")},
        "reviewer_b": {key: reviewer_b[key] for key in (
            "model_id", "model_family", "schedule_sha256", "aggregate_sha256",
            "blind_payload_sha256", "case_receipts")},
        "disagreements": disagreements, "cases": rows,
        "reference_class_coverage": coverage,
        "reference_class_coverage_sufficient": sufficient,
        "ai_jobs_receipt_fabricated": False,
        "limitations": ["Synthetic diagnostic references do not establish natural-distribution validity.",
            "AI consensus is a reference proxy, not human ground truth.",
            "Distinct model families do not establish independent model supply chains."]}
    for slot, reviewer in (("reviewer_a", reviewer_a), ("reviewer_b", reviewer_b)):
        receipt_by_case = {row["case_id"]: row["receipt_file_sha256"]
                           for row in reviewer["case_receipts"]}
        body[slot]["case_evidence"] = [{
            "case_id": row["case_id"],
            "receipt_file_sha256": receipt_by_case[row["case_id"]],
            "verdicts": {dimension: row["dimensions"][dimension]["verdict"]
                         for dimension in ("assertion", "contract")},
            "normalization_issues": row["normalization_issues"],
            **({key: reviewer["case_receipts"][row_index][key] for key in (
                    "blind_payload_sha256", "raw_response_sha256", "normalizer_version")}
               if protocol_bindings["rubric_version"] in {"v5", "v6", "v7", "v8"} else {}),
        } for row_index, row in enumerate(reviewer["cases"])]
        body[slot]["normalization_issue_counts"] = {
            dimension: sum(
                (isinstance(issue, dict) and issue.get("dimension") == dimension)
                or (protocol_bindings["rubric_version"] == "v8"
                    and isinstance(issue, str))
                for row in body[slot]["case_evidence"]
                for issue in row["normalization_issues"]
            ) for dimension in ("assertion", "contract")}
    return {**body, "frozen_reference_sha256": sha256_obj(body)}


def verify_synthetic_diagnostic_reference(packet: dict, frozen: dict,
        schedule_a: dict, state_a: str | Path, aggregate_a: dict,
        schedule_b: dict, state_b: str | Path, aggregate_b: dict) -> dict:
    """Rebuild and compare a frozen diagnostic reference against its shard evidence."""
    expected = freeze_synthetic_diagnostic_reference(packet, schedule_a, state_a,
        aggregate_a, schedule_b, state_b, aggregate_b)
    if frozen != expected:
        raise ValueError("synthetic diagnostic frozen reference identity or scope changed")
    return expected


def write_synthetic_diagnostic_reference_once(path: str | Path, frozen: dict) -> str:
    """Write a validated synthetic freeze with exclusive create semantics."""
    body = {key: value for key, value in frozen.items()
            if key != "frozen_reference_sha256"}
    if (frozen.get("schema") not in {FROZEN_SCHEMA, FROZEN_SCHEMA_V3, FROZEN_SCHEMA_V4,
                                     FROZEN_SCHEMA_V5, FROZEN_SCHEMA_V6, FROZEN_SCHEMA_V7,
                                     FROZEN_SCHEMA_V8, FROZEN_SCHEMA_V9, FROZEN_SCHEMA_V10,
                                     FROZEN_SCHEMA_V11, FROZEN_SCHEMA_V12}
            or frozen.get("synthetic_only") is not True
            or frozen.get("natural_distribution") is not False
            or frozen.get("promotion_authorized") is not False
            or frozen.get("frozen_reference_sha256") != sha256_obj(body)):
        raise ValueError("only a correctly hashed synthetic-only diagnostic freeze can be written")
    supplied = Path(path)
    if ".." in supplied.parts:
        raise ValueError("frozen reference output cannot use parent traversal")
    lexical = Path(os.path.abspath(supplied))
    resolved = lexical.resolve(strict=False)
    parts = lexical.parts
    marker = next((index for index in range(len(parts) - 1, -1, -1)
                   if parts[index].casefold() == ".workbuddy"), None)
    if marker is None or marker == len(parts) - 1:
        raise ValueError("frozen reference output must be below .workbuddy")
    probe = Path(*parts[:marker + 1])
    for component in parts[marker + 1:]:
        probe = probe / component
        if probe.is_symlink():
            raise ValueError("frozen reference output cannot traverse a symlink")
    if resolved != lexical:
        raise ValueError("frozen reference output path resolution changed")
    target = resolved
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(frozen, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()
