"""One-shot, per-case AI reference shards for the synthetic G06 cohort.

Recovery follows the repository's G05 case-start ledger contract: fsync a
case_start before dispatch, then never repeat that case. A started shard with
no verifiable receipt is attempted_unknown; later unstarted shards may proceed.
This is intentionally stricter than a generic checkpoint resume because an
HTTP timeout does not reveal whether the provider completed the request.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from importlib import import_module

from . import ai_jobs
from . import instruction_reviewer_cohort_reference as reference
from .instruction_reviewer_cohort_reference import (
    RUBRIC_V2_VERSION, RUBRIC_V3_VERSION, COHORT_V5_SCHEMA,
    _reference_payload, validate_reference_packet,
    validate_reference_packet_integrity,
)
from .instruction_reviewer_cohort_v7 import (
    LEGACY_PREFLIGHT_PACKET_SHA256, LEGACY_PREFLIGHT_BUILDER_SHA256,
)
from .instruction_reviewer_cohort_v8 import SCHEMA as COHORT_V8_SCHEMA
from .instruction_reviewer_cohort_v9 import SCHEMA as COHORT_V9_SCHEMA
from .instruction_reviewer_cohort_v10 import SCHEMA as COHORT_V10_SCHEMA
from .instruction_reviewer_cohort_v11 import SCHEMA as COHORT_V11_SCHEMA
from .instruction_reviewer_cohort_v12 import SCHEMA as COHORT_V12_SCHEMA
from .instruction_reviewer_cohort_v13 import SCHEMA as COHORT_V13_SCHEMA
from .instruction_reviewer_cohort_v14 import SCHEMA as COHORT_V14_SCHEMA
from .instruction_reviewer_cohort_v15 import SCHEMA as COHORT_V15_SCHEMA
from .instruction_reviewer_cohort_v16 import SCHEMA as COHORT_V16_SCHEMA
from .instruction_reviewer_rubric_v2 import normalize_reference_v2
from .annotation_policy import validate_actor
from ..util import read_json, sha256_file, sha256_obj


SCHEDULE_SCHEMA = "instruction-reviewer-reference-shard-schedule-v1"
ZERO_PRICE_SCHEDULE_SCHEMA = "instruction-reviewer-reference-shard-schedule-v2-zero-price-only"
STATE_SCHEMA = "instruction-reviewer-reference-shard-state-v1"
OUTPUT_SCHEMA = "instruction-reviewer-single-shard-output-v1"
AGGREGATE_SCHEMA = "instruction-reviewer-sharded-reference-v1"
MAX_TOTAL_COST_USD = 0.25
MAX_TOKENS_PER_CASE = 900
MAX_TOKENS_PER_CASE_V3 = 256
MAX_TOKENS_PER_CASE_V4 = 512
MAX_TOKENS_PER_CASE_V5 = 1024
MAX_INPUT_CHARS_PER_CASE = 24_000
RUBRIC_V4_VERSION = "v4"
RUBRIC_V5_VERSION = "v5"
RUBRIC_V6_VERSION = "v6"
RUBRIC_V7_VERSION = "v7"
RUBRIC_V8_VERSION = "v8"
COHORT_V6_SCHEMA = "instruction-reviewer-synthetic-challenge-v6"
COHORT_V7_SCHEMA = "instruction-reviewer-synthetic-challenge-v7"
_LEGACY_V7_PREFLIGHT_SCHEDULES = {
    "reviewer_a": "37019a4450591bdcf59a7cdde1044dd6fa91c06e36c78c6fe73aea09d5ecbd47",
    "reviewer_b": "07fd52bc823e44f55e74259288dc5f274cdf60fec36de2da20d00fcbfc750124",
}
_TARGET_PROTOCOL_BINDING_FIELDS = frozenset({
    "target_protocol_version", "target_protocol_sha256",
    "target_protocol_source_sha256",
})


def _validate_legacy_v7_preflight_schedule(packet: dict, schedule: dict,
                                           slot: str | None) -> None:
    if packet.get("packet_sha256") != LEGACY_PREFLIGHT_PACKET_SHA256:
        return
    frozen_slot = schedule.get("slot") if isinstance(schedule, dict) else None
    frozen_schedule_sha = _LEGACY_V7_PREFLIGHT_SCHEDULES.get(frozen_slot)
    if (packet.get("builder_source_sha256") != LEGACY_PREFLIGHT_BUILDER_SHA256
            or frozen_schedule_sha is None
            or schedule.get("schedule_sha256") != frozen_schedule_sha
            or (slot is not None and slot != frozen_slot)):
        raise ValueError("legacy v7 preflight packet requires its exact frozen slot schedule")


def _uses_exact_legacy_v7_preflight(packet: dict, schedule: dict,
                                    slot: str) -> bool:
    """Return true only for the two pinned packet/schedule/slot combinations."""
    if (packet.get("schema") != COHORT_V7_SCHEMA
            or packet.get("packet_sha256") != LEGACY_PREFLIGHT_PACKET_SHA256):
        return False
    _validate_legacy_v7_preflight_schedule(packet, schedule, slot)
    return True


def _validate_packet(packet: dict, *, integrity: bool = False) -> None:
    """Dispatch versioned G06 packets to their cohort-owned validators."""
    if isinstance(packet, dict) and packet.get("schema") == COHORT_V14_SCHEMA:
        cohort = import_module(".instruction_reviewer_cohort_v14", package=__package__)
        validator = cohort.validate_packet_integrity if integrity else cohort.validate_packet
        validator(packet, reference.repo_root())
        return
    if isinstance(packet, dict) and packet.get("schema") in {
            COHORT_V11_SCHEMA, COHORT_V12_SCHEMA, COHORT_V13_SCHEMA}:
        version = ("v13" if packet.get("schema") == COHORT_V13_SCHEMA else
                   "v12" if packet.get("schema") == COHORT_V12_SCHEMA else "v11")
        cohort = import_module(f".instruction_reviewer_cohort_{version}", package=__package__)
        validator = cohort.validate_packet_integrity if integrity else cohort.validate_packet
        validator(packet, reference.repo_root())
        return
    (validate_reference_packet_integrity if integrity else validate_reference_packet)(packet)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_once(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _append_event(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _controlled_work_path(path: str | Path) -> Path:
    """Resolve an output path below a .workbuddy directory without symlink hops."""
    supplied = Path(path)
    if ".." in supplied.parts:
        raise ValueError("shard outputs cannot use parent traversal")
    lexical = Path(os.path.abspath(supplied))
    resolved = lexical.resolve(strict=False)
    parts = lexical.parts
    marker = next((index for index in range(len(parts) - 1, -1, -1)
                   if parts[index].casefold() == ".workbuddy"), None)
    if marker is None or marker == len(parts) - 1:
        raise ValueError("shard outputs must be below a controlled .workbuddy work directory")
    probe = Path(*parts[:marker + 1])
    for component in parts[marker + 1:]:
        probe = probe / component
        if probe.is_symlink():
            raise ValueError("shard output path cannot traverse a symlink")
    if resolved != lexical:
        raise ValueError("shard output path resolution changed; refusing symlink traversal")
    return resolved


def _validate_state_inventory(root: Path, schedule: dict) -> None:
    allowed = {"schedule.json", "shard-ledger.jsonl", "run.lock", "cases"}
    if {child.name for child in root.iterdir()} - allowed:
        raise ValueError("existing shard state contains unrelated artifacts")
    for name in ("schedule.json", "shard-ledger.jsonl", "run.lock"):
        path = root / name
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise ValueError("existing shard state contains an unsafe control artifact")
    cases_dir = root / "cases"
    if cases_dir.exists():
        if not cases_dir.is_dir() or cases_dir.is_symlink():
            raise ValueError("shard case output path is not a controlled directory")
        expected = {f"case-{row['index']:03d}.json" for row in schedule["shards"]}
        children = list(cases_dir.iterdir())
        if ({child.name for child in children} - expected
                or any(child.is_symlink() or not child.is_file() for child in children)):
            raise ValueError("existing shard state contains unknown or unsafe case artifacts")


def _validate_v13_frozen_source_tree(packet: dict, schedule: dict,
                                    *, check_current: bool = False) -> str | None:
    """Require protected shard plans to share one frozen source tree, optionally current."""
    schema = packet.get("schema")
    if schema not in {COHORT_V13_SCHEMA, COHORT_V15_SCHEMA, COHORT_V16_SCHEMA}:
        return None
    plans = [row.get("plan") for row in schedule.get("shards", [])]
    hashes = [plan.get("source_tree_sha256") if isinstance(plan, dict) else None
              for plan in plans]
    if (not hashes or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
            or len(set(hashes)) != 1):
        raise ValueError("frozen shard plans must share one source_tree_sha256")
    frozen = hashes[0]
    if check_current:
        current = ai_jobs.source_tree_fingerprint().get("sha256")
        if current != frozen:
            raise ValueError("source tree changed after schedule freeze; stopping before shard claim")
    return frozen


@contextmanager
def _run_lock(state_dir: Path):
    lock_path = state_dir / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if lock_path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno != errno.EACCES:
                    raise
                raise ValueError("another reference-shard executor holds the run lock") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                    raise
                raise ValueError("another reference-shard executor holds the run lock") from exc
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def _rubric_version(packet: dict) -> str:
    schema = packet.get("schema")
    if schema == COHORT_V11_SCHEMA:
        return RUBRIC_V8_VERSION
    if schema == COHORT_V12_SCHEMA:
        return RUBRIC_V8_VERSION
    if schema == COHORT_V13_SCHEMA:
        return RUBRIC_V8_VERSION
    if schema == COHORT_V14_SCHEMA:
        return RUBRIC_V6_VERSION
    if schema == COHORT_V15_SCHEMA:
        return RUBRIC_V7_VERSION
    if schema == COHORT_V16_SCHEMA:
        return RUBRIC_V7_VERSION
    if schema == COHORT_V8_SCHEMA:
        return RUBRIC_V6_VERSION
    if schema == COHORT_V9_SCHEMA:
        return RUBRIC_V7_VERSION
    if schema == COHORT_V10_SCHEMA:
        return RUBRIC_V7_VERSION
    if schema == COHORT_V7_SCHEMA:
        return RUBRIC_V5_VERSION
    if schema == COHORT_V6_SCHEMA:
        return RUBRIC_V4_VERSION
    if schema == COHORT_V5_SCHEMA:
        return RUBRIC_V3_VERSION
    return RUBRIC_V2_VERSION


def _rubric_module(version: str):
    if version == RUBRIC_V8_VERSION:
        return import_module(".instruction_reviewer_rubric_v8", package=__package__)
    if version == RUBRIC_V7_VERSION:
        return import_module(".instruction_reviewer_rubric_v7", package=__package__)
    if version == RUBRIC_V6_VERSION:
        return import_module(".instruction_reviewer_rubric_v6", package=__package__)
    if version == RUBRIC_V5_VERSION:
        return import_module(".instruction_reviewer_rubric_v5", package=__package__)
    if version == RUBRIC_V4_VERSION:
        return import_module(".instruction_reviewer_rubric_v4", package=__package__)
    if version == RUBRIC_V3_VERSION:
        return import_module(".instruction_reviewer_rubric_v3", package=__package__)
    return import_module(".instruction_reviewer_rubric_v2", package=__package__)


def _protocol_for_packet(packet: dict) -> tuple[str, dict]:
    version = _rubric_version(packet)
    if version == RUBRIC_V8_VERSION:
        module = _rubric_module(version)
        return module.RUBRIC_V8, module.RESPONSE_FORMAT_TEMPLATE_V8
    if version == RUBRIC_V6_VERSION:
        module = _rubric_module(version)
        return module.RUBRIC_V6, module.REFERENCE_FORMAT_V6
    if version == RUBRIC_V7_VERSION:
        module = _rubric_module(version)
        return module.RUBRIC_V7, module.RESPONSE_FORMAT_TEMPLATE_V7
    if version == RUBRIC_V4_VERSION:
        module = _rubric_module(version)
        return module.RUBRIC_V4, module.REFERENCE_FORMAT_V4
    if version == RUBRIC_V5_VERSION:
        module = _rubric_module(version)
        return module.RUBRIC_V5, module.REFERENCE_FORMAT_V5
    rubric, response_format, _ = reference._rubric_protocol(version)
    return rubric, response_format


def _payload_for_packet(packet: dict, case: dict) -> dict:
    if packet.get("schema") == COHORT_V14_SCHEMA:
        cohort = import_module(".instruction_reviewer_cohort_v14", package=__package__)
        return cohort.reference_payload(case)
    if packet.get("schema") == COHORT_V15_SCHEMA:
        cohort = import_module(".instruction_reviewer_cohort_v15", package=__package__)
        return cohort.reference_payload(case)
    if packet.get("schema") == COHORT_V16_SCHEMA:
        cohort = import_module(".instruction_reviewer_cohort_v16", package=__package__)
        return cohort.reference_payload(case)
    if packet.get("schema") in {COHORT_V11_SCHEMA, COHORT_V12_SCHEMA, COHORT_V13_SCHEMA}:
        version = ("v13" if packet.get("schema") == COHORT_V13_SCHEMA else
                   "v12" if packet.get("schema") == COHORT_V12_SCHEMA else "v11")
        cohort = import_module(f".instruction_reviewer_cohort_{version}", package=__package__)
        return cohort.reference_payload(case)
    if packet.get("schema") == COHORT_V6_SCHEMA:
        cohort = import_module(".instruction_reviewer_cohort_v6", package=__package__)
        return cohort.reference_payload(case)
    return _reference_payload(packet, case)


def _protocol_bindings(packet: dict) -> dict:
    version = _rubric_version(packet)
    rubric, response_format = _protocol_for_packet(packet)
    bindings = {"rubric_version": version,
                "rubric_sha256": sha256_obj(rubric)}
    if version == RUBRIC_V8_VERSION:
        module = _rubric_module(version)
        from ..evolution.instruction_review import prompt_grounding_protocol_v8
        from ..evolution.instruction_review_v9 import prompt_grounding_protocol_v9
        from ..evolution.task_contracts import CLINIFACT_CONTRACT
        expected_reference = {**module.protocol_metadata(),
                              "source_sha256": sha256_file(Path(module.__file__))}
        reference_protocol = packet.get("reference_protocol")
        target_protocol = packet.get("target_protocol")
        uses_target_v9 = packet.get("schema") in {COHORT_V12_SCHEMA, COHORT_V13_SCHEMA}
        expected_target = (prompt_grounding_protocol_v9(CLINIFACT_CONTRACT) if uses_target_v9
                           else prompt_grounding_protocol_v8(CLINIFACT_CONTRACT))
        contract = packet.get("contract")
        if (packet.get("schema") not in {COHORT_V11_SCHEMA, COHORT_V12_SCHEMA,
                                          COHORT_V13_SCHEMA}
                or reference_protocol != expected_reference
                or contract != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}
                or target_protocol != expected_target
                or target_protocol.get("version") != (
                    "instruction-review-v9-nested-json-claim-extraction" if uses_target_v9
                    else "instruction-review-v8-claim-extraction")):
            if uses_target_v9:
                version_label = "v13" if packet.get("schema") == COHORT_V13_SCHEMA else "v12"
                raise ValueError(f"{version_label} reference or target v9 protocol identity is stale or incomplete")
            raise ValueError("v11 reference or target v8 protocol identity is stale or incomplete")
        reference_metadata = module.protocol_metadata()
        bindings.update(reference_metadata)
        # The shard schedule uses the short family selector ("v8") to route
        # historic code paths. Keep the longer protocol identity alongside it.
        bindings["rubric_version"] = version
        bindings["reference_rubric_version"] = reference_metadata["rubric_version"]
        bindings.update({
            "target_protocol_version": target_protocol["version"],
            "target_protocol_sha256": sha256_obj(target_protocol),
            "target_protocol_source_sha256": target_protocol.get(
                "target_source_sha256", target_protocol.get("source_sha256")),
        })
    if version in {RUBRIC_V3_VERSION, RUBRIC_V4_VERSION, RUBRIC_V5_VERSION,
                   RUBRIC_V6_VERSION}:
        module = _rubric_module(version)
        bindings.update({
            "reference_format_sha256": sha256_obj(response_format),
            "protocol_hash": module.protocol_hash(),
            "splitter_hash": module.splitter_hash(),
        })
    if version == RUBRIC_V5_VERSION:
        module = _rubric_module(version)
        bindings.update({
            "protocol_schema": module.PROTOCOL_SCHEMA_V5,
            "normalizer_version": module.NORMALIZER_VERSION_V5,
            "max_output_tokens": module.REFERENCE_MAX_OUTPUT_TOKENS,
        })
    if version == RUBRIC_V6_VERSION:
        module = _rubric_module(version)
        bindings.update({
            "protocol_schema": module.PROTOCOL_SCHEMA_V6,
            "normalizer_version": module.NORMALIZER_VERSION_V6,
            "max_output_tokens": module.REFERENCE_MAX_OUTPUT_TOKENS,
            "reference_format_template_sha256": sha256_obj(module.REFERENCE_FORMAT_V6),
        })
    if packet.get("schema") == COHORT_V14_SCHEMA:
        from ..evolution.instruction_review_v10 import prompt_grounding_protocol_v10
        from ..evolution.task_contracts import CLINIFACT_CONTRACT
        module = _rubric_module(RUBRIC_V6_VERSION)
        expected_reference = {**module.protocol_metadata(),
                              "source_sha256": sha256_file(Path(module.__file__))}
        target_protocol = packet.get("target_protocol")
        contract = packet.get("contract")
        expected_target = prompt_grounding_protocol_v10(CLINIFACT_CONTRACT)
        if (packet.get("reference_protocol") != expected_reference
                or not isinstance(target_protocol, dict)
                or target_protocol != expected_target
                or target_protocol.get("version") != "instruction-review-v10-atomic-span-states"
                or contract != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}):
            raise ValueError("v14 reference-v6 or target-v10 protocol identity is stale or incomplete")
        bindings.update({
            "reference_rubric_version": module.RUBRIC_VERSION,
            "target_protocol_version": target_protocol["version"],
            "target_protocol_sha256": sha256_obj(target_protocol),
            "target_protocol_source_sha256": target_protocol.get(
                "target_source_sha256", target_protocol.get("source_sha256")),
        })
    if packet.get("schema") == COHORT_V15_SCHEMA:
        from ..evolution.instruction_review_v10 import prompt_grounding_protocol_v10
        from ..evolution.task_contracts import CLINIFACT_CONTRACT
        module = _rubric_module(RUBRIC_V7_VERSION)
        expected_reference = {**module.protocol_metadata(),
                              "source_sha256": sha256_file(Path(module.__file__))}
        target_protocol = packet.get("target_protocol")
        contract = packet.get("contract")
        expected_target = prompt_grounding_protocol_v10(CLINIFACT_CONTRACT)
        if (packet.get("reference_protocol") != expected_reference
                or target_protocol != expected_target
                or target_protocol.get("version") != "instruction-review-v10-atomic-span-states"
                or contract != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}):
            raise ValueError("v15 reference-v7 or target-v10 protocol identity is stale or incomplete")
        bindings.update({
            "reference_rubric_version": module.RUBRIC_VERSION,
            "target_protocol_version": target_protocol["version"],
            "target_protocol_sha256": sha256_obj(target_protocol),
            "target_protocol_source_sha256": target_protocol.get(
                "target_source_sha256", target_protocol.get("source_sha256")),
        })
    if packet.get("schema") == COHORT_V16_SCHEMA:
        from ..evolution.instruction_review_v10 import prompt_grounding_protocol_v10
        from ..evolution.task_contracts import CLINIFACT_CONTRACT
        module = _rubric_module(RUBRIC_V7_VERSION)
        expected_reference = {**module.protocol_metadata(),
                              "source_sha256": sha256_file(Path(module.__file__))}
        target_protocol = packet.get("target_protocol")
        contract = packet.get("contract")
        expected_target = prompt_grounding_protocol_v10(CLINIFACT_CONTRACT)
        if (packet.get("reference_protocol") != expected_reference
                or target_protocol != expected_target
                or target_protocol.get("version") != "instruction-review-v10-atomic-span-states"
                or contract != {"id": CLINIFACT_CONTRACT.contract_id,
                    "text": CLINIFACT_CONTRACT.text,
                    "sha256": CLINIFACT_CONTRACT.sha256}):
            raise ValueError("v16 reference-v7 or target-v10 protocol identity is stale or incomplete")
        bindings.update({
            "reference_rubric_version": module.RUBRIC_VERSION,
            "target_protocol_version": target_protocol["version"],
            "target_protocol_sha256": sha256_obj(target_protocol),
            "target_protocol_source_sha256": target_protocol.get(
                "target_source_sha256", target_protocol.get("source_sha256")),
        })
    if version == RUBRIC_V7_VERSION:
        module = _rubric_module(version)
        bindings.update({
            "protocol_hash": module.protocol_hash(),
            "splitter_hash": module.splitter_hash(),
            "protocol_schema": module.PROTOCOL_SCHEMA_V7,
            "normalizer_version": module.NORMALIZER_VERSION_V7,
            "max_output_tokens": module.REFERENCE_MAX_OUTPUT_TOKENS,
            "response_format_template_sha256": sha256_obj(module.RESPONSE_FORMAT_TEMPLATE_V7),
            "normalizer_source_hash": module.normalizer_source_hash(),
        })
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


def _shard_inputs(packet: dict, slot: str, index: int) -> dict:
    case = packet["cases"][index]
    return {"schema": "instruction-reviewer-single-shard-input-v1",
        "packet_sha256": packet["packet_sha256"], "slot": slot,
        **_protocol_bindings(packet),
        "case_index": index, "case_id": case["case_id"],
        "candidate_sha256": case["candidate_sha256"],
        "blind_payload": _payload_for_packet(packet, case)}


def _build_case_plan(packet: dict, slot: str, cfg: dict, index: int) -> tuple[dict, dict]:
    inputs = _shard_inputs(packet, slot, index)
    bindings = _protocol_bindings(packet)
    version = bindings["rubric_version"]
    _, response_format = _protocol_for_packet(packet)
    if version in {RUBRIC_V6_VERSION, RUBRIC_V7_VERSION, RUBRIC_V8_VERSION}:
        response_format = inputs["blind_payload"]["reference_spans"]["response_format"]
    token_budget = ({RUBRIC_V2_VERSION: MAX_TOKENS_PER_CASE,
                     RUBRIC_V3_VERSION: MAX_TOKENS_PER_CASE_V3,
                     RUBRIC_V4_VERSION: MAX_TOKENS_PER_CASE_V4,
                     RUBRIC_V5_VERSION: MAX_TOKENS_PER_CASE_V5,
                     RUBRIC_V6_VERSION: MAX_TOKENS_PER_CASE_V5,
                     RUBRIC_V7_VERSION: MAX_TOKENS_PER_CASE_V5,
                     RUBRIC_V8_VERSION: _rubric_module(RUBRIC_V8_VERSION).REFERENCE_MAX_OUTPUT_TOKENS_V8}[version])
    zero_price_only = packet.get("schema") in {
        COHORT_V14_SCHEMA, COHORT_V15_SCHEMA, COHORT_V16_SCHEMA}
    plan_protocol_metadata = (
        _rubric_module(RUBRIC_V8_VERSION).protocol_metadata()
        if version == RUBRIC_V8_VERSION else
        reference._rubric_protocol_metadata(RUBRIC_V6_VERSION)
        if packet.get("schema") == COHORT_V14_SCHEMA else
        reference._rubric_protocol_metadata(RUBRIC_V7_VERSION)
        if packet.get("schema") in {COHORT_V15_SCHEMA, COHORT_V16_SCHEMA} else bindings)
    plan_kwargs = dict(max_calls=1, tokens=token_budget,
        max_input_chars=MAX_INPUT_CHARS_PER_CASE,
        max_cost_usd=(0.0 if zero_price_only
                      else MAX_TOTAL_COST_USD / len(packet["cases"])),
        response_format=response_format,
        protocol_metadata=plan_protocol_metadata,
        zero_price_only=zero_price_only)
    try:
        plan = ai_jobs.build_plan("instruction_reference", inputs, cfg, **plan_kwargs)
    except TypeError as exc:
        # Older test doubles and downstream adapters predate the explicit
        # zero-price proof keyword.  Keep the production path strict while
        # allowing those doubles to exercise the immutable packet checks.
        if "zero_price_only" not in str(exc):
            raise
        plan_kwargs.pop("zero_price_only", None)
        plan = ai_jobs.build_plan("instruction_reference", inputs, cfg, **plan_kwargs)
    if plan["max_physical_attempts"] != 1 or plan["max_retries_per_call"] != 1:
        raise ValueError("reference shard must allow exactly one physical dispatch")
    return plan, inputs


def build_reference_shard_schedule(packet: dict, slot: str, cfg: dict,
                                   workspace: Path | None = None) -> dict:
    """Build an in-memory, hash-bound schedule. This function performs no writes/calls."""
    _validate_packet(packet)
    if slot not in {"reviewer_a", "reviewer_b"}:
        raise ValueError("invalid independent reviewer slot")
    model_id, base_url = reference._validate_free_config(cfg)
    if base_url.rstrip("/") != "https://api.siliconflow.cn/v1":
        raise ValueError("reference shards require the frozen zero-price CN endpoint")
    workspace = Path(workspace or reference.repo_root()).resolve()
    from .instruction_reviewer_pair_gate import (
        DEFAULT_CONFIG_PATH, load_g06_config, validate_config_binding,
    )
    expected_cfg, config_provenance = load_g06_config(model_id, workspace, DEFAULT_CONFIG_PATH)
    validate_config_binding(cfg, config_provenance, model_id, workspace)
    price_evidence = reference._free_access_evidence(model_id, workspace)
    shards = []
    for index, case in enumerate(packet["cases"]):
        plan, inputs = _build_case_plan(packet, slot, cfg, index)
        shards.append({"index": index, "case_id": case["case_id"],
            "candidate_sha256": case["candidate_sha256"],
            "model_id": model_id, "inputs": inputs, "plan": plan})
    zero_price_only = packet.get("schema") in {
        COHORT_V14_SCHEMA, COHORT_V15_SCHEMA, COHORT_V16_SCHEMA}
    body = {"schema": SCHEDULE_SCHEMA,
        "packet_sha256": packet["packet_sha256"],
        "slot": slot, "model_id": model_id, **_protocol_bindings(packet),
        "natural_distribution": False,
        "runtime_config": expected_cfg, "config_provenance": config_provenance,
        "http_timeout_seconds": 240,
        "http_config_sha256": config_provenance["effective_config_sha256"],
        "annotation_origin": "ai", "blind_payload_fields": (
            ["task_contract", "candidate_strategy", "reference_spans"]
            if _rubric_version(packet) in {RUBRIC_V3_VERSION, RUBRIC_V4_VERSION,
                                           RUBRIC_V5_VERSION, RUBRIC_V6_VERSION,
                                           RUBRIC_V7_VERSION, RUBRIC_V8_VERSION}
            else ["task_contract", "candidate_strategy"]),
        "target_predictions_in_payload": False, "construction_strata_in_payload": False,
        "max_calls_per_case": 1, "max_physical_attempts_per_case": 1,
        "max_retries_per_call": 1,
        "max_logical_calls_per_model": len(shards),
        "max_physical_attempts_per_model": len(shards),
        "max_cost_usd_per_model": MAX_TOTAL_COST_USD,
        "price_evidence": price_evidence, "shards": shards,
        "planned_calls": 0, "paid_fallback": False,
        "limitations": ["Synthetic cohort only; no natural-distribution validity claim.",
            "A durable case_start is irreversible. Missing completion evidence remains attempted_unknown.",
            "An unknown shard is never retried; later never-started shards may continue."]}
    if zero_price_only:
        body.update({"schema": ZERO_PRICE_SCHEDULE_SCHEMA,
            "max_cost_usd_per_model": 0.0, "zero_price_only": True,
            "zero_price_cost_policy": (
                "all rate classes known and exactly zero; measured usage cost exactly zero")})
    return {**body, "schedule_sha256": sha256_obj(body)}


def validate_shard_schedule(packet: dict, schedule: dict, slot: str | None = None,
                            *, allow_legacy_v10: bool = False) -> bool | None:
    _validate_packet(packet, integrity=True)
    if packet.get("schema") == COHORT_V7_SCHEMA:
        _validate_legacy_v7_preflight_schedule(packet, schedule, slot)
    zero_price_only = packet.get("schema") in {
        COHORT_V14_SCHEMA, COHORT_V15_SCHEMA, COHORT_V16_SCHEMA}
    expected_schedule_schema = ZERO_PRICE_SCHEDULE_SCHEMA if zero_price_only else SCHEDULE_SCHEMA
    if not isinstance(schedule, dict) or schedule.get("schema") != expected_schedule_schema:
        raise ValueError("unsupported reference shard schedule")
    body = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    if schedule.get("schedule_sha256") != sha256_obj(body):
        raise ValueError("reference shard schedule canonical hash changed")
    _validate_v13_frozen_source_tree(packet, schedule)
    legacy_v10 = False
    if packet.get("schema") == COHORT_V10_SCHEMA:
        present = _TARGET_PROTOCOL_BINDING_FIELDS.intersection(schedule)
        if present and present != _TARGET_PROTOCOL_BINDING_FIELDS:
            raise ValueError("v10 schedule has a partially missing target protocol identity")
        if not present:
            if not allow_legacy_v10:
                raise ValueError("legacy v10 schedule is verification-only and cannot be executed")
            legacy_v10 = True
    bindings = _protocol_bindings(packet)
    schedule_bindings = _schedule_protocol_bindings(packet, legacy_v10)
    if (schedule.get("packet_sha256") != packet.get("packet_sha256")
            or any(schedule.get(key) != value for key, value in schedule_bindings.items())
            or schedule.get("natural_distribution") is not False
            or schedule.get("annotation_origin") != "ai"
            or schedule.get("target_predictions_in_payload") is not False
            or schedule.get("construction_strata_in_payload") is not False
            or schedule.get("paid_fallback") is not False
            or schedule.get("max_calls_per_case") != 1
            or schedule.get("max_physical_attempts_per_case") != 1
            or schedule.get("max_retries_per_call") != 1
            or schedule.get("http_timeout_seconds") != 240
            or schedule.get("http_config_sha256") != schedule.get("config_provenance", {}).get("effective_config_sha256")
            or schedule.get("max_logical_calls_per_model") != len(packet["cases"])
            or schedule.get("max_physical_attempts_per_model") != len(packet["cases"])
            or (zero_price_only and (
                schedule.get("zero_price_only") is not True
                or schedule.get("zero_price_cost_policy") != (
                    "all rate classes known and exactly zero; measured usage cost exactly zero")
                or schedule.get("max_cost_usd_per_model") != 0.0))
            or (not zero_price_only and
                schedule.get("max_cost_usd_per_model") != MAX_TOTAL_COST_USD)
            or schedule.get("planned_calls") != 0
            or "model_calls" in schedule
            or (slot is not None and schedule.get("slot") != slot)):
        raise ValueError("reference shard schedule identity or hard cap changed")
    from .instruction_reviewer_pair_gate import validate_config_binding
    validate_config_binding(schedule.get("runtime_config"),
        schedule.get("config_provenance"), schedule.get("model_id"), reference.repo_root())
    # Recompute these claims from the local official tariff/access artifacts.
    # A schedule hash authenticates its bytes, not the truth of its claims.
    current_price_evidence = reference._free_access_evidence(
        schedule.get("model_id"), reference.repo_root())
    if schedule.get("price_evidence") != current_price_evidence:
        raise ValueError("reference shard price/access evidence differs from local verified evidence")
    shards = schedule.get("shards")
    if not isinstance(shards, list) or len(shards) != len(packet["cases"]):
        raise ValueError("reference shard schedule has missing cases")
    case_ids = [case["case_id"] for case in packet["cases"]]
    ids = [row.get("case_id") for row in shards]
    if ids != case_ids or len(set(ids)) != len(ids):
        raise ValueError("reference shard case order, uniqueness, or coverage changed")
    plan_hashes = set()
    rubric_version = bindings["rubric_version"]
    expected_v5_format = (_protocol_for_packet(packet)[1]
                          if rubric_version == RUBRIC_V5_VERSION else None)
    expected_v6_format = None
    expected_v7_format = None
    expected_v8_format = None
    expected_plan_protocol_metadata = (
        _rubric_module(RUBRIC_V8_VERSION).protocol_metadata()
        if rubric_version == RUBRIC_V8_VERSION else
        reference._rubric_protocol_metadata(RUBRIC_V6_VERSION)
        if packet.get("schema") == COHORT_V14_SCHEMA else
        reference._rubric_protocol_metadata(RUBRIC_V7_VERSION)
        if packet.get("schema") in {COHORT_V15_SCHEMA, COHORT_V16_SCHEMA} else schedule_bindings)
    for index, shard in enumerate(shards):
        plan, inputs = shard.get("plan", {}), shard.get("inputs", {})
        plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
        expected = _shard_inputs(packet, schedule["slot"], index)
        if legacy_v10:
            expected = _without_target_protocol_bindings(expected)
        if rubric_version == RUBRIC_V6_VERSION:
            expected_v6_format = expected["blind_payload"]["reference_spans"]["response_format"]
        if rubric_version == RUBRIC_V7_VERSION:
            expected_v7_format = expected["blind_payload"]["reference_spans"]["response_format"]
        if rubric_version == RUBRIC_V8_VERSION:
            expected_v8_format = expected["blind_payload"]["reference_spans"]["response_format"]
        if (shard.get("index") != index
                or shard.get("candidate_sha256") != packet["cases"][index].get("candidate_sha256")
                or shard.get("model_id") != schedule.get("model_id")
                or plan.get("model_id") != schedule.get("model_id")
                or plan.get("kind") != "instruction_reference"
                or plan.get("max_calls") != 1
                or plan.get("max_physical_attempts") != 1
                or plan.get("max_retries_per_call") != 1
                or plan.get("zero_price_only", False) is not zero_price_only
                or plan.get("max_cost_usd") != (0.0 if zero_price_only else plan.get("max_cost_usd"))
                or (zero_price_only and len(shard["plan"].get("zero_price_rates_sha256", "")) != 64)
                or (rubric_version == RUBRIC_V5_VERSION and (
                    plan.get("max_tokens") != MAX_TOKENS_PER_CASE_V5
                    or plan.get("response_format") != expected_v5_format))
                or (rubric_version == RUBRIC_V6_VERSION and (
                    plan.get("max_tokens") != MAX_TOKENS_PER_CASE_V5
                    or plan.get("response_format") != expected_v6_format))
                or (rubric_version == RUBRIC_V7_VERSION and (
                    plan.get("max_tokens") != MAX_TOKENS_PER_CASE_V5
                    or plan.get("response_format") != expected_v7_format))
                or (rubric_version == RUBRIC_V8_VERSION and (
                    plan.get("max_tokens") != _rubric_module(RUBRIC_V8_VERSION).REFERENCE_MAX_OUTPUT_TOKENS_V8
                    or plan.get("response_format") != expected_v8_format))
                or plan.get("inputs_sha256") != sha256_obj(expected)
                or inputs != expected
                or plan.get("plan_sha256") != sha256_obj(plan_body)
                or plan.get("protocol_metadata") != expected_plan_protocol_metadata):
            raise ValueError(f"reference shard plan/input/model identity invalid at index {index}")
        if plan["plan_sha256"] in plan_hashes:
            raise ValueError("reference shard plans are not unique per case")
        plan_hashes.add(plan["plan_sha256"])
    return True if legacy_v10 else None


def _without_target_protocol_bindings(value: dict) -> dict:
    return {key: item for key, item in value.items()
            if key not in _TARGET_PROTOCOL_BINDING_FIELDS}


def _schedule_protocol_bindings(packet: dict, legacy_v10: bool = False) -> dict:
    bindings = _protocol_bindings(packet)
    return _without_target_protocol_bindings(bindings) if legacy_v10 else bindings


def _legacy_aggregate_matches(rebuilt: dict, supplied: dict) -> bool:
    """Accept only the exact pre-target-binding aggregate shape for a legacy schedule."""
    if supplied == rebuilt:
        return True
    if not isinstance(supplied, dict):
        return False
    if _TARGET_PROTOCOL_BINDING_FIELDS.intersection(supplied):
        return False
    body = {key: value for key, value in rebuilt.items()
            if key not in _TARGET_PROTOCOL_BINDING_FIELDS
            and key != "aggregate_sha256"}
    legacy = {**body, "aggregate_sha256": sha256_obj(body)}
    return supplied == legacy


def _load_events(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError("shard ledger is missing")
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"shard ledger JSON invalid at line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError("shard ledger contains a non-object row")
        events.append(row)
    return events


def _ledger_states(packet: dict, schedule: dict, root: Path) -> tuple[dict, dict]:
    events = _load_events(root / "shard-ledger.jsonl")
    starts, ends = {}, {}
    for event in events:
        kind, case_id = event.get("event"), event.get("case_id")
        if kind == "case_start":
            if case_id in starts or case_id not in {row["case_id"] for row in schedule["shards"]}:
                raise ValueError("duplicate or unknown shard case_start")
            shard = schedule["shards"][event.get("index", -1)] if type(event.get("index")) is int else None
            if (shard is None or shard["case_id"] != case_id
                    or event.get("schedule_sha256") != schedule["schedule_sha256"]
                    or event.get("plan_sha256") != shard["plan"]["plan_sha256"]
                    or event.get("attempt") != 1):
                raise ValueError("shard case_start identity changed")
            starts[case_id] = event
        elif kind == "case_end":
            if case_id not in starts or case_id in ends:
                raise ValueError("orphan or duplicate shard case_end")
            if (event.get("schedule_sha256") != schedule["schedule_sha256"]
                    or event.get("plan_sha256") != starts[case_id]["plan_sha256"]
                    or event.get("status") not in {"completed", "attempted_unknown"}):
                raise ValueError("shard case_end identity or status invalid")
            if event["status"] == "completed" and not event.get("record_sha256"):
                raise ValueError("completed shard terminal lacks its immutable record hash")
            ends[case_id] = event
        else:
            raise ValueError("shard ledger contains an unknown event")
    return starts, ends


def _case_record_path(root: Path, index: int) -> Path:
    return root / "cases" / f"case-{index:03d}.json"


def _verify_completed_record(packet: dict, schedule: dict, root: Path,
                             shard: dict, *, verify_job=None,
                             legacy_v10: bool = False) -> dict:
    path = _case_record_path(root, shard["index"])
    if not path.is_file():
        raise ValueError("completed shard record is missing")
    record = read_json(path)
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if (record.get("record_sha256") != sha256_obj(body)
            or record.get("schedule_sha256") != schedule["schedule_sha256"]
            or record.get("case_id") != shard["case_id"]
            or record.get("plan_sha256") != shard["plan"]["plan_sha256"]
            or record.get("model_id") != schedule["model_id"]):
        raise ValueError("completed shard record identity or hash changed")
    verify_job = verify_job or ai_jobs.verify_completed
    job = verify_job(record["run_dir"], kind="instruction_reference")
    _validate_job(packet, schedule, shard, job, legacy_v10=legacy_v10)
    receipt_path = Path(record["run_dir"]) / "receipt.json"
    artifact_hashes = record.get("run_artifact_sha256")
    expected_names = {"plan.json", "inputs.json", "output.json", "receipt.json",
                      "status.json", "calls.json", "call-0.json", "call-ledger.jsonl"}
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_names:
        raise ValueError("completed shard artifact hash inventory is incomplete")
    run_root = Path(record["run_dir"]).resolve()
    if any(sha256_file(run_root / name) != digest
           for name, digest in artifact_hashes.items()):
        raise ValueError("completed shard run artifact changed after receipt")
    if (sha256_file(receipt_path) != record.get("receipt_file_sha256")
            or job["receipt"].get("plan_sha256") != record.get("plan_sha256")
            or job.get("receipt_file_sha256") != record.get("receipt_file_sha256")):
        raise ValueError("completed shard receipt hash changed")
    artifact_hashes = record.get("run_artifact_sha256")
    expected_names = {"plan.json", "inputs.json", "output.json", "receipt.json",
                      "status.json", "calls.json", "call-0.json", "call-ledger.jsonl"}
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_names:
        raise ValueError("completed shard artifact hash inventory is incomplete")
    run_root = Path(record["run_dir"]).resolve()
    if any(sha256_file(run_root / name) != digest
           for name, digest in artifact_hashes.items()):
        raise ValueError("completed shard run artifact changed after receipt")
    return record


def _validate_job(packet: dict, schedule: dict, shard: dict, job: dict, *,
                  legacy_v10: bool = False) -> None:
    receipt, output, plan = job.get("receipt", {}), job.get("output", {}), job.get("plan", {})
    accounting = receipt.get("request_accounting", {})
    rows = output.get("cases")
    bindings = _schedule_protocol_bindings(packet, legacy_v10)
    if (plan != shard["plan"] or job.get("inputs") != shard["inputs"]
            or output.get("schema") != OUTPUT_SCHEMA
            or output.get("packet_sha256") != packet["packet_sha256"]
            or output.get("slot") != schedule["slot"]
            or output.get("model_id") != schedule["model_id"]
            or any(output.get(key) != value for key, value in bindings.items())
            or not isinstance(rows, list) or len(rows) != 1
            or rows[0].get("case_id") != shard["case_id"]
            or receipt.get("test_only") is not False
            or receipt.get("usage_complete") is not True
            or receipt.get("measured_usd") != 0.0 or receipt.get("call_count") != 1
            or accounting.get("logical_operations") != 1
            or accounting.get("physical_http_attempts") != 1
            or accounting.get("physical_status_counts") != {"200": 1}
            or accounting.get("logical_status_counts") != {"success": 1}
            or accounting.get("physical_attempts_exact") is not True):
        raise ValueError("completed ai_jobs shard violates frozen one-shot receipt contract")


def _case_states(packet: dict, schedule: dict, root: Path, *, verify_job=None,
                 recover_completed_record: bool = True,
                 legacy_v10: bool = False) -> list[dict]:
    starts, ends = _ledger_states(packet, schedule, root)
    states = []
    for shard in schedule["shards"]:
        case_id = shard["case_id"]
        if case_id not in starts:
            if case_id in ends or _case_record_path(root, shard["index"]).exists():
                raise ValueError("shard terminal or record exists without a durable case_start")
            states.append({"case_id": case_id, "index": shard["index"], "state": "never_started"})
            continue
        terminal = ends.get(case_id)
        record_path = _case_record_path(root, shard["index"])
        if terminal and terminal["status"] == "completed":
            record = _verify_completed_record(packet, schedule, root, shard,
                verify_job=verify_job, legacy_v10=legacy_v10)
            if sha256_file(record_path) != terminal.get("record_sha256"):
                raise ValueError("terminal shard record hash differs")
            states.append({"case_id": case_id, "index": shard["index"],
                           "state": "completed", "record": record})
        elif terminal and terminal["status"] == "attempted_unknown":
            if record_path.exists():
                raise ValueError("unknown shard must not contain a completion record")
            states.append({"case_id": case_id, "index": shard["index"],
                           "state": "attempted_unknown"})
        elif record_path.exists():
            if not recover_completed_record:
                states.append({"case_id": case_id, "index": shard["index"],
                               "state": "record_pending_terminal"})
                continue
            # Crash after a durable verified record but before case_end: recovery
            # re-verifies the existing ai_jobs receipt and closes only the local ledger.
            record = _verify_completed_record(packet, schedule, root, shard,
                verify_job=verify_job, legacy_v10=legacy_v10)
            _append_event(root / "shard-ledger.jsonl", {"event": "case_end",
                "case_id": case_id, "index": shard["index"],
                "schedule_sha256": schedule["schedule_sha256"],
                "plan_sha256": shard["plan"]["plan_sha256"], "status": "completed",
                "record_sha256": sha256_file(record_path)})
            states.append({"case_id": case_id, "index": shard["index"],
                           "state": "completed", "record": record})
        else:
            states.append({"case_id": case_id, "index": shard["index"],
                           "state": "attempted_unknown"})
    return states


def _case_worker(packet: dict, slot: str, shard: dict):
    case = packet["cases"][shard["index"]]
    bindings = _protocol_bindings(packet)
    version = bindings["rubric_version"]
    rubric, _ = _protocol_for_packet(packet)
    def worker(call):
        raw, provenance = call("instruction_reference", shard["inputs"]["blind_payload"], rubric)
        actor = validate_actor(slot, "independent_ai", provenance)
        if version == RUBRIC_V8_VERSION:
            normalizer = _rubric_module(version).normalize_reference_v8
            normalized, issues = normalizer(
                raw, case["strategy_text"], packet["contract"]["text"])
            if normalized is None:
                reason = ";".join(issues) or "invalid_reference_output"
                dimensions = {
                    "assertion": {"verdict": "uncertain", "claims": [], "uncertain": [],
                        "verdict_reason_code": "invalid_reference_evidence_fail_closed",
                        "normalization_failure": reason},
                    "contract": {"verdict": "uncertain", "issues": [], "uncertain": [],
                        "verdict_reason_code": "invalid_reference_evidence_fail_closed",
                        "normalization_failure": reason},
                }
            else:
                dimensions = normalized
                issues = []
        elif version == RUBRIC_V2_VERSION:
            dimensions, issues = normalize_reference_v2(
                raw, case["strategy_text"], packet["contract"]["text"])
        else:
            normalizer_name = ("normalize_reference_v7" if version == RUBRIC_V7_VERSION else
                "normalize_reference_v6" if version == RUBRIC_V6_VERSION else
                "normalize_reference_v5" if version == RUBRIC_V5_VERSION
                else "normalize_reference_v4" if version == RUBRIC_V4_VERSION
                else "normalize_reference_v3")
            normalizer = getattr(_rubric_module(version), normalizer_name)
            dimensions, issues = normalizer(
                raw, case["strategy_text"], packet["contract"]["text"])
        row = {"case_id": case["case_id"], "dimensions": dimensions,
               "normalization_issues": issues, "actor": actor}
        return {"schema": OUTPUT_SCHEMA, "packet_sha256": packet["packet_sha256"],
            "slot": slot, "model_id": shard["model_id"],
            **bindings, "cases": [row]}
    return worker


def record_completed_shard(packet: dict, schedule: dict, state_dir: str | Path,
                           case_id: str, run_dir: str | Path, *, verify_job=None) -> dict:
    """Bind a real ai_jobs receipt to its reserved case; never synthesize one."""
    validate_shard_schedule(packet, schedule)
    root = _controlled_work_path(state_dir)
    _verify_state_anchor(packet, schedule, root)
    shard = _shard_by_id(schedule, case_id)
    verify_job = verify_job or ai_jobs.verify_completed
    job = verify_job(run_dir, kind="instruction_reference")
    _validate_job(packet, schedule, shard, job)
    run_root = Path(job["root"]).resolve()
    names = ("plan.json", "inputs.json", "output.json", "receipt.json", "status.json",
             "calls.json", "call-0.json", "call-ledger.jsonl")
    hashes = {name: sha256_file(run_root / name) for name in names}
    body = {"schema": "instruction-reviewer-reference-shard-record-v1",
        "status": "verified_completed", "schedule_sha256": schedule["schedule_sha256"],
        "packet_sha256": packet["packet_sha256"], "slot": schedule["slot"],
        "case_id": case_id, "index": shard["index"], "model_id": schedule["model_id"],
        "plan_sha256": shard["plan"]["plan_sha256"], "run_dir": str(run_root),
        "receipt_file_sha256": hashes["receipt.json"], "run_artifact_sha256": hashes,
        "case_result": job["output"]["cases"][0]}
    record = {**body, "record_sha256": sha256_obj(body)}
    record_path = _case_record_path(root, shard["index"])
    with _run_lock(root):
        starts, ends = _ledger_states(packet, schedule, root)
        if case_id not in starts or case_id in ends:
            raise ValueError("case must have one active durable start and no terminal state")
        if record_path.exists():
            existing = read_json(record_path)
            if existing != record:
                raise ValueError("existing shard record differs; refusing overwrite")
            raise ValueError("shard record exists without its terminal ledger event")
        record_hash = _write_once(record_path, record)
        _append_event(root / "shard-ledger.jsonl", {"event": "case_end",
            "case_id": case_id, "index": shard["index"],
            "schedule_sha256": schedule["schedule_sha256"],
            "plan_sha256": shard["plan"]["plan_sha256"], "status": "completed",
            "record_sha256": record_hash})
        return record


def _shard_by_id(schedule: dict, case_id: str) -> dict:
    rows = [row for row in schedule["shards"] if row["case_id"] == case_id]
    if len(rows) != 1:
        raise ValueError("unknown or duplicate case identity")
    return rows[0]


def _verify_state_anchor(packet: dict, schedule: dict, root: Path, *,
                          allow_legacy_v10: bool = False) -> None:
    if _controlled_work_path(root) != root.resolve():
        raise ValueError("shard state escaped the controlled work directory")
    path = root / "schedule.json"
    if not path.is_file() or read_json(path) != schedule:
        raise ValueError("shard state is not anchored to this frozen schedule")
    validate_shard_schedule(packet, schedule, allow_legacy_v10=allow_legacy_v10)


def initialize_shard_state(packet: dict, schedule: dict, state_dir: str | Path) -> Path:
    """Create an exclusive work-dir state or resume its exact anchored contents."""
    validate_shard_schedule(packet, schedule)
    root = _controlled_work_path(state_dir)
    existed = root.exists()
    if existed and not root.is_dir():
        raise ValueError("shard state path must be a directory")
    if not existed:
        root.mkdir(parents=True, exist_ok=False)
    schedule_path = root / "schedule.json"
    if schedule_path.exists():
        if read_json(schedule_path) != schedule:
            raise ValueError("existing shard schedule differs; refusing resume")
        ledger = root / "shard-ledger.jsonl"
        if not ledger.is_file():
            raise ValueError("existing shard state has no durable ledger")
        _validate_state_inventory(root, schedule)
        _case_states(packet, schedule, root, recover_completed_record=False)
    else:
        if any(root.iterdir()):
            raise ValueError("first-time shard initialization requires an empty work directory")
        _write_once(schedule_path, schedule)
        ledger = root / "shard-ledger.jsonl"
        with ledger.open("x", encoding="utf-8", newline="\n") as stream:
            stream.flush()
            os.fsync(stream.fileno())
    return root


def claim_next_shard(packet: dict, schedule: dict, state_dir: str | Path,
                     *, case_id: str | None = None) -> dict | None:
    """Durably reserve one never-started shard; started/unknown is never replayed."""
    validate_shard_schedule(packet, schedule)
    _validate_v13_frozen_source_tree(packet, schedule, check_current=True)
    root = _controlled_work_path(state_dir)
    _verify_state_anchor(packet, schedule, root)
    with _run_lock(root):
        states = _case_states(packet, schedule, root)
        state_by_id = {row["case_id"]: row["state"] for row in states}
        candidates = schedule["shards"] if case_id is None else [_shard_by_id(schedule, case_id)]
        for shard in candidates:
            current = state_by_id[shard["case_id"]]
            if current != "never_started":
                if case_id is not None:
                    raise ValueError(f"case already started or completed: {current}")
                continue
            _append_event(root / "shard-ledger.jsonl", {"event": "case_start",
                "case_id": shard["case_id"], "index": shard["index"],
                "schedule_sha256": schedule["schedule_sha256"],
                "plan_sha256": shard["plan"]["plan_sha256"], "attempt": 1})
            return shard
    return None


def mark_attempted_unknown(packet: dict, schedule: dict, state_dir: str | Path,
                           case_id: str, *, reason_code: str) -> None:
    """Close a started shard as unknown without storing provider text or labels."""
    if not isinstance(reason_code, str) or not reason_code.strip():
        raise ValueError("unknown terminal requires a nonempty reason code")
    root = _controlled_work_path(state_dir)
    _verify_state_anchor(packet, schedule, root)
    shard = _shard_by_id(schedule, case_id)
    with _run_lock(root):
        starts, ends = _ledger_states(packet, schedule, root)
        if case_id not in starts or case_id in ends or _case_record_path(root, shard["index"]).exists():
            raise ValueError("only an unresolved started shard can be marked unknown")
        _append_event(root / "shard-ledger.jsonl", {"event": "case_end",
            "case_id": case_id, "index": shard["index"],
            "schedule_sha256": schedule["schedule_sha256"],
            "plan_sha256": shard["plan"]["plan_sha256"], "status": "attempted_unknown",
            "reason_code": reason_code[:80]})


def run_reference_shards(packet: dict, slot: str, cfg: dict, schedule: dict,
                         state_dir: str | Path) -> dict:
    """Execute only never-started shards; completed and ambiguous cases are skipped."""
    validate_shard_schedule(packet, schedule, slot)
    legacy_preflight = _uses_exact_legacy_v7_preflight(packet, schedule, slot)
    model_id, base_url = reference._validate_free_config(cfg)
    if model_id != schedule["model_id"] or base_url.rstrip("/") != "https://api.siliconflow.cn/v1":
        raise ValueError("execution config differs from frozen shard schedule")
    if schedule.get("price_evidence", {}).get("execution_permitted") is not True:
        raise ValueError("official tariff evidence does not confirm account access; dispatch is prohibited")
    if cfg != schedule.get("runtime_config"):
        raise ValueError("execution config differs from the frozen G06 240-second configuration")
    # Rebuild every plan before any case_start so stale source/config aborts with
    # zero dispatches instead of consuming one-shot reservations as a side effect.
    for shard in schedule["shards"]:
        expected, _ = _build_case_plan(packet, slot, cfg, shard["index"])
        allowed_source = (shard["plan"].get("source_tree_sha256")
                          if legacy_preflight else None)
        if not ai_jobs._frozen_plan_matches_current(expected, shard["plan"],
                allowed_legacy_source_tree_sha256=allowed_source):
            raise ValueError("code, config, or shard plan changed after schedule freeze")
    root = initialize_shard_state(packet, schedule, state_dir)
    starts, ends = _ledger_states(packet, schedule, root)
    if (len(starts) > schedule["max_physical_attempts_per_model"]
            or len(starts) > len(schedule["shards"])):
        raise ValueError("shard run has exceeded its physical dispatch ceiling")
    for state in _case_states(packet, schedule, root):
        if state["state"] == "completed":
            continue
    while True:
        _validate_v13_frozen_source_tree(packet, schedule, check_current=True)
        shard = claim_next_shard(packet, schedule, root)
        if shard is None:
            break
        try:
            execute_kwargs = ({"allowed_legacy_source_tree_sha256":
                shard["plan"]["source_tree_sha256"]} if legacy_preflight else {})
            result = ai_jobs.execute(shard["plan"], shard["inputs"], cfg,
                _case_worker(packet, slot, shard), **execute_kwargs)
            record_completed_shard(packet, schedule, root, shard["case_id"], result["run_dir"])
        except Exception as exc:
            starts, ends = _ledger_states(packet, schedule, root)
            if shard["case_id"] in starts and shard["case_id"] not in ends:
                mark_attempted_unknown(packet, schedule, root, shard["case_id"],
                                       reason_code=type(exc).__name__)
            # A failure cannot block later cases, but this case is now terminal
            # unknown and cannot be dispatched again.
            continue
    states = _case_states(packet, schedule, root)
    counts = Counter(row["state"] for row in states)
    metrics = _shard_runtime_metrics(states)
    return {"schema": STATE_SCHEMA, "schedule_sha256": schedule["schedule_sha256"],
        "status": "complete" if counts.get("completed") == len(schedule["shards"])
                   else "incomplete_or_unknown",
        "case_state_counts": dict(sorted(counts.items())),
        **metrics,
        "max_physical_attempts": schedule["max_physical_attempts_per_model"]}


def _shard_runtime_metrics(states: list[dict]) -> dict:
    """Count verified completed work separately from one-shot unknown reservations."""
    completed = [row for row in states if row["state"] == "completed"]
    unknown = [row for row in states if row["state"] in {
        "attempted_unknown", "record_pending_terminal"}]
    logical_calls = physical_calls = 0
    for row in completed:
        receipt = read_json(Path(row["record"]["run_dir"]) / "receipt.json")
        accounting = receipt["request_accounting"]
        logical_calls += receipt["call_count"]
        physical_calls += accounting["physical_http_attempts"]
    reserved = len(states) - sum(row["state"] == "never_started" for row in states)
    return {"logical_calls": logical_calls, "physical_calls": physical_calls,
        "completed_cases": len(completed), "unknown_cases": len(unknown),
        "logical_call_reservations": reserved,
        "physical_attempt_reservations": reserved}


def _validate_aggregate_records(schedule: dict, records: list[dict]) -> None:
    if not isinstance(records, list) or len(records) != len(schedule["shards"]):
        raise ValueError("aggregate shard coverage is incomplete")
    expected = {row["case_id"]: row for row in schedule["shards"]}
    seen = set()
    for record in records:
        case_id = record.get("case_id")
        if case_id in seen:
            raise ValueError("aggregate contains duplicate case identity")
        seen.add(case_id)
        shard = expected.get(case_id)
        if (shard is None or record.get("index") != shard["index"]
                or record.get("model_id") != schedule["model_id"]
                or record.get("plan_sha256") != shard["plan"]["plan_sha256"]
                or record.get("case_result", {}).get("case_id") != case_id
                or not isinstance(record.get("receipt_file_sha256"), str)
                or len(record["receipt_file_sha256"]) != 64):
            raise ValueError("aggregate case record has wrong identity/model/plan/receipt")
    if seen != set(expected):
        raise ValueError("aggregate case coverage is missing or contains unknown cases")


def aggregate_verified_shards(packet: dict, schedule: dict,
                              state_dir: str | Path) -> dict:
    """Verify all per-case receipts and return a non-ai_jobs aggregate evidence bundle."""
    if packet.get("schema") == COHORT_V10_SCHEMA:
        legacy_v10 = validate_shard_schedule(packet, schedule, allow_legacy_v10=True)
    else:
        legacy_v10 = validate_shard_schedule(packet, schedule)
    root = _controlled_work_path(state_dir)
    if legacy_v10:
        _verify_state_anchor(packet, schedule, root, allow_legacy_v10=True)
    else:
        _verify_state_anchor(packet, schedule, root)
    states = _case_states(packet, schedule, root, recover_completed_record=False,
                          legacy_v10=legacy_v10)
    if any(row["state"] != "completed" for row in states):
        raise ValueError("cannot aggregate an incomplete/unknown shard cohort")
    records = [row["record"] for row in states]
    _validate_aggregate_records(schedule, records)
    body = {"schema": AGGREGATE_SCHEMA, "status": "verified_complete",
        "packet_sha256": packet["packet_sha256"], "schedule_sha256": schedule["schedule_sha256"],
        "slot": schedule["slot"], "model_id": schedule["model_id"],
        **{key: value for key, value in _protocol_bindings(packet).items()},
        "annotation_origin": "ai",
        "natural_distribution": False, "case_count": len(records),
        "cases": [record["case_result"] for record in records],
        "shards": [{"case_id": record["case_id"], "index": record["index"],
            "plan_sha256": record["plan_sha256"], "run_dir": record["run_dir"],
            "receipt_file_sha256": record["receipt_file_sha256"],
            "record_sha256": record.get("record_sha256")} for record in records],
        "ai_jobs_receipt_fabricated": False,
        **_shard_runtime_metrics(states)}
    return {**body, "aggregate_sha256": sha256_obj(body)}


def write_aggregate_once(path: str | Path, aggregate: dict) -> str:
    """Seal a verified aggregate without overwriting prior evidence."""
    if aggregate.get("schema") != AGGREGATE_SCHEMA or aggregate.get("status") != "verified_complete":
        raise ValueError("only a verified complete shard aggregate can be written")
    body = {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    if aggregate.get("aggregate_sha256") != sha256_obj(body):
        raise ValueError("aggregate hash mismatch")
    return _write_once(_controlled_work_path(path), aggregate)

