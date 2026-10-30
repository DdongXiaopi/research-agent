"""Build a blinded, provenance-bound packet from external instruction sources.

This adapter is deliberately separate from the historical candidate census.  A
Natural Instructions task definition or a MUFFIN rematching pair is a useful
source of naturally written instruction text, but it is not a persisted MIPRO
candidate and its source labels are not CliniFact reviewer gold.  The packet
therefore keeps source metadata out of the blind payload, carries no expected
labels, and records that the resulting cohort is an external natural proxy.

The adapter accepts small normalized extracts rather than downloading a whole
upstream repository.  Supported record shapes are the public Natural
Instructions task schema (``Definition``), MUFFIN rematching records
(``instruction``/``input``), and LLMBar records (``input`` plus two outputs).
LLMBar is supported for audit and meta-evaluation only: its output preference
label must never be reinterpreted as an assertion or task-contract label.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..evolution.task_contracts import CLINIFACT_CONTRACT, TaskContract, compose_task_instruction
from ..util import sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-external-source-packet-v1"
PACKET_HASH_FIELD = "packet_sha256"
BLIND_FIELDS = ("task_contract", "candidate_strategy")

SOURCE_POLICIES: dict[str, dict[str, Any]] = {
    "natural_instructions": {
        "natural_distribution": True,
        "candidate_distribution_alignment": "external_human_authored_proxy",
        "production_gate_eligible": False,
        "reason": "human-authored task definitions are not persisted evolved candidates",
    },
    "muffin_rematch": {
        "natural_distribution": True,
        "candidate_distribution_alignment": "external_rematching_proxy",
        "production_gate_eligible": False,
        "reason": "rematching validity labels are not CliniFact assertion/contract gold",
    },
    "llmbar": {
        "natural_distribution": False,
        "candidate_distribution_alignment": "output_preference_only",
        "production_gate_eligible": False,
        "reason": "LLMBar labels compare outputs rather than review candidate strategies",
    },
}


class ExternalSourceError(ValueError):
    """Raised when an external record cannot be safely adapted."""


def _first_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    return None


def _record_id(record: Mapping[str, Any], index: int, source_kind: str) -> str:
    for key in ("id", "task_id", "case_id", "uid"):
        value = record.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return f"{source_kind}-{index:06d}"


def _instruction(record: Mapping[str, Any], source_kind: str) -> str:
    if source_kind == "natural_instructions":
        value = _first_text(record.get("Definition"))
    elif source_kind == "llmbar":
        # LLMBar calls this field input.  It is the task instruction, while
        # output_1/output_2/label are deliberately ignored by this adapter.
        value = _first_text(record.get("input"))
    else:
        value = _first_text(record.get("instruction"))
    if not value:
        raise ExternalSourceError(
            f"{source_kind} record has no non-empty instruction/Definition")
    return value


def _context_digest(record: Mapping[str, Any], source_kind: str) -> str:
    """Hash context without copying it into the blind reference payload."""
    if source_kind == "natural_instructions":
        context = {
            key: record.get(key)
            for key in ("Positive Examples", "Negative Examples", "Instances")
            if key in record
        }
    elif source_kind == "muffin_rematch":
        context = {"input": record.get("input"), "output": record.get("output")}
    else:
        context = {"output_1": record.get("output_1"),
                   "output_2": record.get("output_2")}
    return sha256_obj(context)


def normalize_record(record: Mapping[str, Any], source_kind: str, index: int,
                     contract: TaskContract = CLINIFACT_CONTRACT) -> dict[str, Any]:
    """Normalize one supported upstream record without importing its labels."""
    if source_kind not in SOURCE_POLICIES:
        raise ExternalSourceError(f"unsupported external source kind: {source_kind}")
    if not isinstance(record, Mapping):
        raise ExternalSourceError("external record must be an object")
    strategy = _instruction(record, source_kind)
    candidate = compose_task_instruction(strategy, contract)
    source_id = _record_id(record, index, source_kind)
    provenance = {
        "source_kind": source_kind,
        "source_record_id": source_id,
        "source_record_sha256": sha256_obj(dict(record)),
        "context_sha256": _context_digest(record, source_kind),
    }
    # Provenance is auditable but never sent to reviewers.  Do not persist
    # source labels (Natural Instructions outputs, MUFFIN judge, or LLMBar
    # label) in the case; all reference labels are generated independently.
    return {
        "case_id": f"external-{source_kind}-{sha256_obj(candidate)[:16]}",
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "provenance": provenance,
    }


def _candidate_hashes(value: Any) -> set[str]:
    """Collect exact candidate/strategy hashes from prior packet documents."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key in ("candidate_text", "strategy_text", "candidate_strategy"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                found.add(sha256_obj(text.strip()))
        for child in value.values():
            found.update(_candidate_hashes(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_candidate_hashes(child))
    return found


def build_exposure_audit(cases: Sequence[Mapping[str, Any]],
                         prior_packets: Iterable[tuple[str, Mapping[str, Any]]] = ()) -> dict[str, Any]:
    prior_hashes: set[str] = set()
    receipts = []
    for source, packet in prior_packets:
        hashes = _candidate_hashes(packet)
        prior_hashes.update(hashes)
        receipts.append({"source": str(source),
                         "packet_sha256": packet.get(PACKET_HASH_FIELD),
                         "candidate_text_count": len(hashes),
                         "document_sha256": sha256_obj(packet)})
    overlaps = []
    for case in cases:
        if case["candidate_sha256"] in prior_hashes:
            overlaps.append(case["case_id"])
    if overlaps:
        raise ExternalSourceError(
            f"external packet overlaps exposed candidate text: {overlaps[:3]}")
    return {
        "prior_packet_count": len(receipts),
        "prior_packet_receipts": receipts,
        "overlap_count": 0,
        "overlap_verified": True,
    }


def build_packet(records: Iterable[Mapping[str, Any]], *, source_kind: str,
                 source_manifest: Mapping[str, Any],
                 contract: TaskContract = CLINIFACT_CONTRACT,
                 prior_packets: Iterable[tuple[str, Mapping[str, Any]]] = (),
                 max_cases: int | None = None) -> dict[str, Any]:
    """Build a deterministic external-source packet for independent AI review."""
    if source_kind not in SOURCE_POLICIES:
        raise ExternalSourceError(f"unsupported external source kind: {source_kind}")
    if not isinstance(source_manifest, Mapping) or not source_manifest:
        raise ExternalSourceError("source_manifest must contain immutable provenance")
    cases = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if max_cases is not None and len(cases) >= max_cases:
            break
        case = normalize_record(record, source_kind, index, contract)
        if case["candidate_sha256"] in seen:
            continue
        seen.add(case["candidate_sha256"])
        cases.append(case)
    cases.sort(key=lambda row: row["case_id"])
    if not cases:
        raise ExternalSourceError("external source produced no usable instruction records")
    policy = SOURCE_POLICIES[source_kind]
    body = {
        "schema": SCHEMA,
        "source_kind": source_kind,
        "scope": "external_instruction_source_ai_reference_proxy",
        "natural_distribution": policy["natural_distribution"],
        "candidate_distribution_alignment": policy["candidate_distribution_alignment"],
        "production_gate_eligible": policy["production_gate_eligible"],
        "eligibility_reason": policy["reason"],
        "source_manifest": dict(source_manifest),
        "contract": {"id": contract.contract_id, "text": contract.text,
                     "sha256": contract.sha256},
        "reference_protocol": {
            "blind_payload_fields": list(BLIND_FIELDS),
            "expected_labels_present": False,
            "target_predictions_present": False,
            "source_labels_ignored": True,
            "two_distinct_reference_models_and_adjudication": True,
            "disagreements": "uncertain; never force external source labels",
        },
        "exposure_audit": build_exposure_audit(cases, prior_packets),
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, PACKET_HASH_FIELD: sha256_obj(body)}


def reference_payload(packet: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the blinded payload used by the reference annotator."""
    if packet.get("schema") != SCHEMA:
        raise ExternalSourceError("external packet schema changed")
    if case not in packet.get("cases", []):
        raise ExternalSourceError("case is not a member of packet")
    contract = packet["contract"]
    return {"task_contract": contract, "candidate_strategy": case["strategy_text"]}


def validate_packet(packet: Mapping[str, Any]) -> None:
    """Validate packet identity and ensure source labels cannot enter review."""
    if not isinstance(packet, Mapping) or packet.get("schema") != SCHEMA:
        raise ExternalSourceError("external packet schema changed")
    source_kind = packet.get("source_kind")
    if source_kind not in SOURCE_POLICIES:
        raise ExternalSourceError("external source kind changed")
    body = {key: value for key, value in packet.items() if key != PACKET_HASH_FIELD}
    if packet.get(PACKET_HASH_FIELD) != sha256_obj(body):
        raise ExternalSourceError("external packet hash changed")
    if packet.get("reference_protocol", {}).get("blind_payload_fields") != list(BLIND_FIELDS):
        raise ExternalSourceError("external blind payload protocol changed")
    if packet["reference_protocol"].get("expected_labels_present") is not False:
        raise ExternalSourceError("external source labels cannot be treated as expected labels")
    if packet["reference_protocol"].get("target_predictions_present") is not False:
        raise ExternalSourceError("target predictions must be absent")
    cases = packet.get("cases")
    if not isinstance(cases, list) or packet.get("case_count") != len(cases):
        raise ExternalSourceError("external case count changed")
    ids, hashes = [], set()
    contract = CLINIFACT_CONTRACT
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != {
                "case_id", "strategy_text", "candidate_text", "candidate_sha256", "provenance"}:
            raise ExternalSourceError("external case schema changed")
        if (case["candidate_text"] != compose_task_instruction(case["strategy_text"], contract)
                or case["candidate_sha256"] != sha256_obj(case["candidate_text"])
                or case["candidate_sha256"] in hashes):
            raise ExternalSourceError("external case identity changed")
        if case["case_id"] != "external-" + source_kind + "-" + case["candidate_sha256"][:16]:
            raise ExternalSourceError("external case id changed")
        provenance = case["provenance"]
        if not isinstance(provenance, Mapping) or set(provenance) != {
                "source_kind", "source_record_id", "source_record_sha256", "context_sha256"}:
            raise ExternalSourceError("external provenance changed")
        if provenance["source_kind"] != source_kind:
            raise ExternalSourceError("external provenance source mismatch")
        ids.append(case["case_id"])
        hashes.add(case["candidate_sha256"])
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ExternalSourceError("external cases are not sorted and unique")


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a compact JSON array, object with ``records``, or JSONL file."""
    if not path.is_file():
        raise ExternalSourceError(f"external record file not found: {path}")
    try:
        if path.suffix.casefold() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()]
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalSourceError(f"cannot read external record file: {path}") from exc
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("records"), list):
        return value["records"]
    if isinstance(value, Mapping):
        return [dict(value)]
    raise ExternalSourceError("external record file must contain an object or list")


def file_source_manifest(path: Path, *, source_url: str, repository: str,
                         revision: str | None = None) -> dict[str, Any]:
    """Create a source manifest for a downloaded extract without trusting it blindly."""
    if not path.is_file():
        raise ExternalSourceError(f"source file not found: {path}")
    value: dict[str, Any] = {
        "path": path.as_posix(), "sha256": sha256_file(path),
        "source_url": str(source_url), "repository": str(repository),
        "extraction": "only instruction text and context digests; source labels excluded",
    }
    if revision:
        value["revision"] = str(revision)
    return value


__all__ = ["SCHEMA", "SOURCE_POLICIES", "ExternalSourceError", "build_packet",
           "build_exposure_audit", "file_source_manifest", "normalize_record",
           "read_records", "reference_payload", "validate_packet"]
