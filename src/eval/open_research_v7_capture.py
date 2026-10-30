"""Approved, hash-bound executor for a frozen G09 v7 query-arm plan.

This module discovers source candidates only. It never creates a cohort or
labels, and retains the existing source identity/PICO gates. License metadata
is recorded as provenance; it is not an eligibility gate for this personal project.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from ..util import sha256_obj
from . import open_research_sources as sources
from . import open_research_v6_continuation as v6
from . import open_research_v7_query_plan as v7

V7_CAPTURE_SCHEMA = "g09-topic-source-capture-v7"
V7_APPROVAL_SCHEMA = "g09-v7-capture-approval-v1"
V7_MAX_NEW_REQUESTS = 252
V7_REQUIRED_PRIMARY_GROUPS = 48


def _payload_sha(value: dict, field: str) -> str:
    return sha256_obj({key: item for key, item in value.items() if key != field})


def _capture_sha(value: dict) -> str:
    return sha256_obj({key: item for key, item in value.items()
        if key not in {"capture_sha256", "audit"}})


class _WriteAheadLedger:
    """One-shot durable marker: existing ledger means this plan cannot be replayed."""

    def __init__(self, path: str | Path, *, plan_sha256: str,
            approval_sha256: str, captured_at: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = {"schema": "g09-v7-capture-write-ahead-ledger-v1",
            "plan_sha256": plan_sha256, "approval_sha256": approval_sha256,
            "captured_at": captured_at, "status": "started",
            "request_intents": [], "final_capture_sha256": None}
        # Exclusive creation makes same-plan replay fail closed, including after a crash.
        with self.path.open("x", encoding="utf-8", newline="\n") as handle:
            self.state["ledger_sha256"] = sha256_obj(self.state)
            json.dump(self.state, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_parent()

    def _fsync_parent(self):
        try:
            fd = os.open(str(self.path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp",
            dir=str(self.path.parent))
        try:
            self.state.pop("ledger_sha256", None)
            self.state["ledger_sha256"] = sha256_obj(self.state)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(self.state, handle, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._fsync_parent()
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def before_request(self, request: dict):
        self.state["request_intents"].append({
            "ordinal": len(self.state["request_intents"]) + 1,
            "arm_id": request["arm_id"], "cursor_mark": request["cursor_mark"],
            "query_sha256": request["query_sha256"], "state": "in_flight"})
        self.state["status"] = "request_in_flight"
        self._write()

    def after_request(self, request: dict, *, state: str,
            response_sha256: str | None = None, response_bytes: int | None = None,
            identities: list[dict] | None = None, records_returned: int | None = None,
            candidate_projection: list[dict] | None = None, hit_count: int | None = None,
            next_cursor_mark: str | None = None):
        item = self.state["request_intents"][-1]
        item["state"] = state
        if response_sha256:
            receipt = {"response_sha256": response_sha256, "response_bytes": response_bytes,
                "hit_count": hit_count, "next_cursor_mark": next_cursor_mark,
                "records_returned": records_returned,
                "identities": identities or [],
                "candidate_projection": candidate_projection or []}
            receipt["identity_receipt_sha256"] = sha256_obj(receipt["identities"])
            receipt["candidate_projection_sha256"] = sha256_obj(
                receipt["candidate_projection"])
            item["response_receipt"] = receipt
        if request.get("error_type"):
            item["error_type"] = request["error_type"]
        self.state["status"] = "request_terminal"
        self._write()

    def finish(self, *, capture_sha256: str, status: str,
            artifact_path: str, artifact_sha256: str):
        self.state["final_capture_sha256"] = capture_sha256
        self.state["final_artifact_path"] = artifact_path
        self.state["final_artifact_file_sha256"] = artifact_sha256
        self.state["status"] = status
        self._write()


def _atomic_write_json(path: str | Path, value: dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"capture artifact already exists: {target}")
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
        dir=str(target.parent))
    try:
        encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes atomically and fails if the
        # destination appeared concurrently; it never overwrites another run.
        os.link(temporary, target)
        os.unlink(temporary)
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return hashlib.sha256(encoded).hexdigest()
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def audit_v7_write_ahead_ledger(path: str | Path) -> dict:
    """Verify durable write-ahead receipts without reading source abstracts."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = value.get("ledger_sha256")
    actual = sha256_obj({key: item for key, item in value.items() if key != "ledger_sha256"})
    receipt_valid, consumed_cursors, identities = True, [], []
    for item in value.get("request_intents", []):
        consumed_cursors.append({"arm_id": item.get("arm_id"),
            "cursor_mark": item.get("cursor_mark"), "query_sha256": item.get("query_sha256"),
            "state": item.get("state")})
        receipt = item.get("response_receipt")
        if receipt is not None:
            if (receipt.get("identity_receipt_sha256") != sha256_obj(receipt.get("identities", [])) or
                    receipt.get("candidate_projection_sha256") !=
                    sha256_obj(receipt.get("candidate_projection", []))):
                receipt_valid = False
            for candidate in receipt.get("candidate_projection", []):
                if (not _canonical_doi(candidate.get("doi")) or
                        not re.fullmatch(r"pmc\d+", _canonical_pmcid(candidate.get("pmcid"))) or
                        not isinstance(candidate.get("abstractText"), str) or
                        len(candidate["abstractText"].strip()) < 400):
                    receipt_valid = False
            identities.extend(receipt.get("identities", []))
    body = {"schema": "g09-v7-write-ahead-ledger-audit-v1",
        "ledger_hash_valid": expected == actual, "response_receipts_valid": receipt_valid,
        "status": value.get("status"), "plan_sha256": value.get("plan_sha256"),
        "request_attempts_started": len(value.get("request_intents", [])),
        "ambiguous_in_flight_count": sum(1 for row in value.get("request_intents", [])
            if row.get("state") == "in_flight"),
        "consumed_cursors": consumed_cursors,
        "observed_identity_count": len(identities),
        "observed_identities_sha256": sha256_obj(identities),
        "final_capture_sha256": value.get("final_capture_sha256"),
        "final_artifact_file_sha256": value.get("final_artifact_file_sha256")}
    return {**body, "audit_sha256": sha256_obj(body)}


def _canonical_doi(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _canonical_pmcid(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_approval(approval: dict, expected_sha256: str, plan_sha256: str) -> None:
    if not isinstance(approval, dict) or approval.get("schema") != V7_APPROVAL_SCHEMA:
        raise ValueError("an explicit root approval artifact is required before capture")
    actual = _payload_sha(approval, "approval_sha256")
    if (not _valid_sha(expected_sha256) or actual != expected_sha256 or
            approval.get("approval_sha256") != actual or approval.get("status") != "approved" or
            approval.get("approved_plan_sha256") != plan_sha256 or
            not _valid_sha(approval.get("review_package_sha256")) or
            approval.get("approved_by") != "root"):
        raise ValueError("root approval does not bind the exact v7 plan and review package")
    try:
        timestamp = datetime.fromisoformat(approval["approved_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("root approval must include a timezone-aware timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("root approval timestamp must include a timezone")


def _response_json(url: str, *, opener) -> tuple[dict, bytes]:
    status, raw = sources.bounded_source_response(
        url, opener=opener, user_agent="G09-research-assistant-evaluation-v7/1.0")
    if status != 200:
        raise RuntimeError(f"Europe PMC request failed with HTTP {status}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Europe PMC returned a non-object page")
    return value, raw


def _request_url(query: str, cursor: str, page_size: int) -> str:
    return sources.API_ROOT + "/search?" + urlencode({"query": query,
        "format": "json", "resulttype": "core", "pageSize": page_size,
        "cursorMark": cursor})


def _all_existing_identities(parent_capture: dict, v6_capture: dict) -> tuple[set[str], set[str]]:
    seen_dois, seen_pmcids = v7._collect_seen_identities(parent_capture, v6_capture)
    return {_canonical_doi(item) for item in seen_dois}, {
        _canonical_pmcid(item) for item in seen_pmcids}


def _complete_primary_groups(outputs: dict) -> int:
    return sum(1 for row in outputs.values() if row["slot_kind"] != "conflict" and
        row["parent_record_count"] + row["v6_record_count"] + len(row["records_added"]) >= 2)


def _prepare_topics(parent_capture: dict, v6_capture: dict, topic_specs: list[dict],
        plan: dict) -> tuple[dict, dict, dict]:
    parent_topics = {row["topic_id"]: row for row in parent_capture["topics"]}
    v6_topics = v6_capture["topics"]
    specs = {row["topic_id"]: row for row in topic_specs}
    outputs = {}
    for topic_id, spec in specs.items():
        parent, continued = parent_topics[topic_id], v6_topics[topic_id]
        parent_count, v6_rows = len(parent.get("records", [])), continued.get("records_added", [])
        if not isinstance(v6_rows, list) or parent_count + len(v6_rows) > 2:
            raise ValueError(f"frozen source group is malformed or exceeds cap: {topic_id}")
        identities = []
        for row in list(parent.get("records", [])) + v6_rows:
            if isinstance(row, dict):
                identities.append({"doi": _canonical_doi(row.get("doi")) or None,
                    "pmcid": _canonical_pmcid(row.get("pmcid")) or None})
        outputs[topic_id] = {"topic_id": topic_id, "slot_kind": spec["slot_kind"],
            "parent_record_count": parent_count, "v6_record_count": len(v6_rows),
            "parent_record_identities": identities, "records_added": [],
            "core_pages": [], "candidate_audit": [], "exclusions": [],
            "failure_reasons": [], "status": "not_in_frozen_v7_plan"}
    return parent_topics, v6_topics, outputs


def _failure(output: dict, state: dict, reason: str) -> None:
    state["failure_reason"] = reason
    state["done"] = True
    output["failure_reasons"].append({"arm_id": state["arm"]["arm_id"], "reason": reason})


def _fetch_arm_page(state: dict, output: dict, spec: dict, group: dict,
        parent_dois: set[str], parent_pmcids: set[str],
        seen_dois: set[str], seen_pmcids: set[str],
        request_log: list, *, phase: str, opener, ledger: _WriteAheadLedger) -> None:
    arm = state["arm"]
    cursor, query = state["cursor"], arm["query"]
    if cursor in state["seen_cursors"]:
        _failure(output, state, "cursor_replay_prevented")
        return
    url = _request_url(query, cursor, arm["page_size"])
    request = {"arm_id": arm["arm_id"], "topic_id": arm["topic_id"],
        "arm_type": arm["arm_type"], "phase": phase,
        "page_index": state["page_index"] + 1, "cursor_mark": cursor,
        "query_sha256": arm["query_sha256"], "method": "GET", "url": url}
    ledger.before_request(request)
    try:
        response, raw = _response_json(url, opener=opener)
    except Exception as exc:
        request.update({"http_status": None, "error_type": type(exc).__name__,
            "error": str(exc)[:500]})
        request_log.append(request)
        ledger.after_request(request, state="transport_or_parse_failure")
        _failure(output, state, "core_search_request_failed")
        return
    response_sha = hashlib.sha256(raw).hexdigest()
    request.update({"http_status": 200, "response_bytes": len(raw),
        "response_sha256": response_sha})
    request_log.append(request)
    rows = (response.get("resultList") or {}).get("result")
    identity_receipt, candidate_projection = [], []
    if isinstance(rows, list):
        for result_index, row in enumerate(rows):
            if isinstance(row, dict):
                doi = _canonical_doi(row.get("doi"))
                pmcid = _canonical_pmcid(row.get("pmcid"))
                identity_receipt.append({"doi": doi or None, "pmcid": pmcid or None})
                abstract = row.get("abstractText")
                parent_identity = doi in parent_dois or pmcid in parent_pmcids
                if (not parent_identity and doi and re.fullmatch(r"pmc\d+", pmcid) and
                        isinstance(abstract, str) and len(abstract.strip()) >= 400):
                    candidate_projection.append({"result_index": result_index,
                        "doi": doi, "pmcid": pmcid,
                        "license": row.get("license") if isinstance(row.get("license"), str) else None,
                        "title": row.get("title") if isinstance(row.get("title"), str) else "",
                        "abstractText": abstract})
    ledger.after_request(request, state="response_received",
        response_sha256=response_sha, response_bytes=len(raw),
        identities=identity_receipt, records_returned=len(rows) if isinstance(rows, list) else None,
        candidate_projection=candidate_projection,
        hit_count=response.get("hitCount") if type(response.get("hitCount")) is int else None,
        next_cursor_mark=response.get("nextCursorMark")
            if isinstance(response.get("nextCursorMark"), str) else None)
    if not isinstance(rows, list) or len(rows) > arm["page_size"]:
        request["error_type"] = "invalid_core_search_response"
        _failure(output, state, "invalid_core_search_response")
        return
    next_cursor, hit_count = response.get("nextCursorMark"), response.get("hitCount")
    if type(hit_count) is not int or hit_count < 0:
        request["error_type"] = "invalid_hit_count"
        _failure(output, state, "invalid_hit_count")
        return
    if state["hit_count"] is not None and state["hit_count"] != hit_count:
        request["error_type"] = "hit_count_changed_during_cursor_continuation"
        _failure(output, state, "hit_count_changed_during_cursor_continuation")
        return

    page_index, offset = state["page_index"] + 1, state["record_offset"]
    receipt = {"arm_id": arm["arm_id"], "arm_type": arm["arm_type"],
        "page_index": page_index, "cursor_mark": cursor,
        "next_cursor_mark": next_cursor if isinstance(next_cursor, str) else None,
        "response_sha256": response_sha, "records_returned": len(rows),
        "hit_count": hit_count, "record_offset": offset}
    output["core_pages"].append(receipt)
    state["page_index"], state["hit_count"] = page_index, hit_count
    for result_index, record in enumerate(rows):
        rank = offset + result_index
        if not isinstance(record, dict):
            output["exclusions"].append({"arm_id": arm["arm_id"],
                "reason": "invalid_core_record", "core_rank": rank,
                "response_sha256": response_sha})
            continue
        doi, pmcid = v6._canonical_doi(record.get("doi")), v6._canonical_pmcid(record.get("pmcid"))
        if not doi or not re.fullmatch(r"pmc\d+", pmcid):
            output["exclusions"].append({"arm_id": arm["arm_id"], "doi": doi or None,
                "pmcid": pmcid or None, "reason": "missing_source_identity",
                "core_rank": rank, "response_sha256": response_sha})
            continue
        if any(doi.startswith(prefix) for prefix in sources.HISTORICAL_DOI_PREFIX_EXCLUSIONS):
            request["identity_error"] = "forbidden_source_family"
            _failure(output, state, "forbidden_source_family")
            break
        parent_seen = doi in parent_dois or pmcid in parent_pmcids
        already_seen = doi in seen_dois or pmcid in seen_pmcids
        # Consume both non-empty identifiers even when only one collides, so a
        # DOI/PMCID pair cannot be reintroduced under the previously unseen half.
        seen_dois.add(doi)
        seen_pmcids.add(pmcid)
        if already_seen:
            reason = "parent_seen_source_identity" if parent_seen else "duplicate_source_across_v7_arms"
            output["exclusions"].append({"arm_id": arm["arm_id"], "doi": doi,
                "pmcid": pmcid, "reason": reason, "core_rank": rank,
                "response_sha256": response_sha})
            continue
        # New identities are consumed immediately, even if completeness/PICO fails,
        # so another arm cannot retry the same source.
        audit = {"arm_id": arm["arm_id"], "doi": doi, "pmcid": pmcid,
            "core_rank": rank, "page_index": page_index, "result_index": result_index,
            "core_response_sha256": response_sha}
        output["candidate_audit"].append(audit)
        abstract = record.get("abstractText")
        if not isinstance(abstract, str) or len(abstract.strip()) < 400:
            output["exclusions"].append({"arm_id": arm["arm_id"], "doi": doi,
                "pmcid": pmcid, "reason": "abstract_missing_or_too_short", "core_rank": rank,
                "response_sha256": response_sha})
            continue
        relevance = sources.assess_topic_pico_relevance(record, spec)
        audit["abstract_pico_relevance"] = relevance
        if not relevance["eligible"]:
            output["exclusions"].append({"arm_id": arm["arm_id"], "doi": doi,
                "pmcid": pmcid, "reason": "pico_relevance_missing_dimensions",
                "pico_relevance": relevance, "abstract_text": abstract,
                "license": record.get("license"), "abstract_length": len(abstract.strip()),
                "core_rank": rank, "response_sha256": response_sha})
            continue
        existing_count = group["parent_record_count"] + group["v6_record_count"]
        if existing_count + len(group["records_added"]) >= 2:
            output["exclusions"].append({"arm_id": arm["arm_id"], "doi": doi,
                "pmcid": pmcid, "reason": "group_source_cap_reached",
                "core_rank": rank, "response_sha256": response_sha})
            continue
        group["records_added"].append(record)
        if existing_count + len(group["records_added"]) >= 2:
            for sibling in group["_arm_states"]:
                sibling["done"] = True
            break

    state["record_offset"] = offset + len(rows)
    if state["failure_reason"]:
        return
    if group["parent_record_count"] + group["v6_record_count"] + len(group["records_added"]) >= 2:
        state["done"] = True
    elif not rows or state["record_offset"] >= hit_count:
        state["exhausted"], state["done"] = True, True
    elif (not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor or
            next_cursor in state["seen_cursors"]):
        request["error_type"] = "cursor_missing_or_repeated"
        _failure(output, state, "cursor_missing_or_repeated")
    else:
        state["seen_cursors"].add(cursor)
        state["cursor"] = next_cursor


def audit_v7_capture(result: dict, *, plan: dict, parent_capture: dict,
        v6_capture: dict, parent_plan_manifest: dict) -> dict:
    """Recompute artifact, budget, cursor, and source-acceptance audit flags."""
    artifact_valid = result.get("capture_sha256") == _capture_sha(result)
    plan_valid = result.get("plan_sha256") == plan.get("plan_sha256")
    parent_dois, parent_pmcids = _all_existing_identities(parent_capture, v6_capture)
    new_dois, new_pmcids, accepted_valid = set(), set(), True
    specs = {row["topic_id"]: row for row in parent_plan_manifest["topic_specs"]}
    for topic in result.get("topics", {}).values():
        cap = max(0, 2 - topic.get("parent_record_count", 0) - topic.get("v6_record_count", 0))
        if len(topic.get("records_added", [])) > cap:
            accepted_valid = False
        for record in topic.get("records_added", []):
            doi, pmcid = _canonical_doi(record.get("doi")), _canonical_pmcid(record.get("pmcid"))
            if (not doi or not re.fullmatch(r"pmc\d+", pmcid) or
                    not isinstance(record.get("abstractText"), str) or
                    len(record["abstractText"].strip()) < 400 or doi in parent_dois or
                    pmcid in parent_pmcids or doi in new_dois or pmcid in new_pmcids or
                    not sources.assess_topic_pico_relevance(record, specs[topic["topic_id"]])["eligible"] or
                    any(doi.startswith(prefix) for prefix in sources.HISTORICAL_DOI_PREFIX_EXCLUSIONS)):
                accepted_valid = False
            new_dois.add(doi)
            new_pmcids.add(pmcid)
    receipts = {}
    for topic in result.get("topics", {}).values():
        for receipt in topic.get("core_pages", []):
            receipts.setdefault(receipt.get("arm_id"), {})[receipt.get("page_index")] = receipt
    arms = {row["arm_id"]: row for row in plan.get("arms", [])}
    last_cursor, cursor_valid, phases = {}, True, []
    for request in result.get("request_log", []):
        arm = arms.get(request.get("arm_id"))
        if arm is None:
            cursor_valid = False
            continue
        arm_id = arm["arm_id"]
        phases.append(request.get("phase"))
        if request.get("cursor_mark") != last_cursor.get(arm_id, arm["start_cursor"]):
            cursor_valid = False
        receipt = receipts.get(arm_id, {}).get(request.get("page_index"))
        if request.get("http_status") is None:
            # Transport failures have no page receipt and are terminal per arm.
            continue
        if (receipt is None or receipt.get("cursor_mark") != request.get("cursor_mark") or
                receipt.get("response_sha256") != request.get("response_sha256")):
            cursor_valid = False
        next_cursor = receipt.get("next_cursor_mark") if receipt else None
        if next_cursor is not None:
            if next_cursor == request.get("cursor_mark"):
                cursor_valid = False
            last_cursor[arm_id] = next_cursor
    phase_valid = (all(p in {"planned", "reserve"} for p in phases) and
        not ("planned" in phases[phases.index("reserve"):] if "reserve" in phases else False) and
        phases.count("planned") == result.get("planned_requests_used") and
        phases.count("reserve") == result.get("reserve_requests_used"))
    reserve = result.get("reserve_gate", {})
    reserve_valid = (isinstance(reserve, dict) and (
        result.get("reserve_requests_used", 0) == 0 or
        (reserve.get("authorized") is True and reserve.get("planned_phase_complete") is True and
         reserve.get("primary_groups_before_reserve", 60) < V7_REQUIRED_PRIMARY_GROUPS and
         bool(reserve.get("searchable_arm_ids_before_reserve")))))
    attempts = result.get("request_attempts", -1)
    budget_valid = (result.get("hard_new_request_max") == V7_MAX_NEW_REQUESTS and
        attempts == len(result.get("request_log", [])) and 0 <= attempts <=
        result.get("request_budget", -1) <= V7_MAX_NEW_REQUESTS and
        result.get("planned_requests_used", -1) <= result.get("planned_request_budget", -1) <= 215 and
        result.get("reserve_requests_used", -1) <= result.get("reserve_request_budget", -1) <= 37 and
        result.get("cumulative_request_attempts") == 252 + attempts and
        result.get("cumulative_request_attempts") <= result.get("cumulative_request_hard_max", -1) <= 504 and
        phase_valid and reserve_valid)
    impl_valid = result.get("capture_implementation_sha256") == hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    body = {"schema": "g09-v7-capture-audit-v1",
        "capture_sha256": result.get("capture_sha256"), "plan_sha256": result.get("plan_sha256"),
        "artifact_hash_valid": artifact_valid, "plan_binding_valid": plan_valid,
        "implementation_hash_valid": impl_valid, "budget_valid": budget_valid,
        "reserve_gate_valid": reserve_valid, "cursor_continuity_valid": cursor_valid,
        "request_phase_accounting_valid": phase_valid,
        "accepted_records_compliant": accepted_valid, "request_attempts": attempts,
        "request_failures": result.get("request_failures"),
        "cumulative_request_attempts": result.get("cumulative_request_attempts"),
        "primary_groups_complete": result.get("primary_groups_complete"),
        "diagnostic_only": result.get("diagnostic_only") is True,
        "scoreable_cohort_created": result.get("scoreable_cohort_created") is True}
    return {**body, "audit_sha256": sha256_obj(body)}


def capture_v7_query_arm_plan(*, plan: dict, expected_plan_sha256: str,
        parent_capture: dict, parent_capture_file_sha256: str,
        parent_plan_manifest: dict, v6_capture: dict, v6_capture_file_sha256: str,
        v6_root_audit: dict, v6_root_audit_file_sha256: str,
        approval: dict, expected_approval_sha256: str, captured_at: str,
        ledger_path: str | Path, capture_output_path: str | Path | None = None,
        opener=urlopen, request_budget: int | None = None,
        sleep_fn=time.sleep, monotonic_fn=time.monotonic) -> dict:
    """Run a frozen v7 plan only when an exact root approval artifact is supplied."""
    try:
        timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("captured_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    frozen = v7.validate_v7_query_arm_plan(plan=plan,
        expected_plan_sha256=expected_plan_sha256, parent_capture=parent_capture,
        parent_capture_file_sha256=parent_capture_file_sha256,
        parent_plan_manifest=parent_plan_manifest, v6_capture=v6_capture,
        v6_capture_file_sha256=v6_capture_file_sha256, v6_root_audit=v6_root_audit,
        v6_root_audit_file_sha256=v6_root_audit_file_sha256)
    _validate_approval(approval, expected_approval_sha256, frozen["plan_sha256"])
    if request_budget is None:
        request_budget = frozen["max_new_requests"]
    if type(request_budget) is not int or not 0 <= request_budget <= frozen["max_new_requests"]:
        raise ValueError("request_budget cannot exceed the frozen v7 plan")
    if frozen["max_new_requests"] != V7_MAX_NEW_REQUESTS or \
            frozen["cumulative_request_hard_max"] > 2 * v6.HARD_REQUEST_MAX:
        raise ValueError("v7 plan exceeds its hard request ceiling")
    ledger_path = Path(ledger_path)
    capture_output_path = (Path(capture_output_path) if capture_output_path is not None else
        ledger_path.with_name(ledger_path.stem + ".capture.json"))
    if capture_output_path.resolve() == ledger_path.resolve():
        raise ValueError("capture output and write-ahead ledger must be separate files")
    if capture_output_path.exists():
        raise FileExistsError(f"capture artifact already exists: {capture_output_path}")
    ledger = _WriteAheadLedger(ledger_path, plan_sha256=frozen["plan_sha256"],
        approval_sha256=approval["approval_sha256"], captured_at=captured_at)

    parent_topics, v6_topics, outputs = _prepare_topics(parent_capture, v6_capture,
        parent_plan_manifest["topic_specs"], frozen)
    arms = sorted(frozen["arms"], key=lambda row: (row["topic_id"], row["arm_id"]))
    arm_topics = {row["topic_id"] for row in arms}
    for topic_id, output in outputs.items():
        if output["slot_kind"] == "conflict":
            output["status"] = "conflict_diagnostic_only"
        elif output["parent_record_count"] + output["v6_record_count"] >= 2:
            output["status"] = "complete_before_v7"
        elif topic_id in arm_topics:
            output["status"] = "incomplete_searchable"
    parent_dois, parent_pmcids = _all_existing_identities(parent_capture, v6_capture)
    seen_dois, seen_pmcids = set(parent_dois), set(parent_pmcids)
    if (sorted(seen_dois) != frozen["excluded_parent_dois"] or
            sorted(seen_pmcids) != frozen["excluded_parent_pmcids"]):
        raise ValueError("v7 identity exclusion set differs from parent-derived plan")
    specs = {row["topic_id"]: row for row in parent_plan_manifest["topic_specs"]}
    states = {}
    for arm in arms:
        if not isinstance(arm.get("start_cursor"), str) or not arm["start_cursor"]:
            raise ValueError(f"frozen v7 arm has no initial cursor: {arm['arm_id']}")
        state = {"arm": arm, "cursor": arm["start_cursor"], "seen_cursors": set(),
            "hit_count": None, "record_offset": 0, "page_index": 0,
            "planned_used": 0, "reserve_used": 0, "done": False,
            "exhausted": False, "failure_reason": None}
        states[arm["arm_id"]] = state
        outputs[arm["topic_id"]].setdefault("_arm_states", []).append(state)

    request_log, planned_used, reserve_used = [], 0, 0
    last_request_started_at = None
    minimum_interval = frozen["minimum_request_interval_seconds"]
    base_limit = min(frozen["base_request_budget"], request_budget)
    reserve_limit = min(frozen["reserve_request_budget"], request_budget - base_limit)

    def searchable(state: dict, *, reserve: bool) -> bool:
        arm = state["arm"]
        group = outputs[arm["topic_id"]]
        count = group["parent_record_count"] + group["v6_record_count"] + len(group["records_added"])
        used = state["planned_used"] + state["reserve_used"] if reserve else state["planned_used"]
        cap = arm["max_pages_with_reserve"] if reserve else arm["max_pages"]
        return not state["done"] and state["failure_reason"] is None and count < 2 and used < cap

    def spend(limit: int, phase: str) -> int:
        nonlocal planned_used, reserve_used, last_request_started_at
        used = 0
        while used < limit and _complete_primary_groups(outputs) < V7_REQUIRED_PRIMARY_GROUPS:
            progressed = False
            for state in states.values():
                if used >= limit or _complete_primary_groups(outputs) >= V7_REQUIRED_PRIMARY_GROUPS:
                    break
                if not searchable(state, reserve=(phase == "reserve")):
                    continue
                arm, topic_id = state["arm"], state["arm"]["topic_id"]
                if phase == "planned":
                    state["planned_used"] += 1
                    planned_used += 1
                else:
                    state["reserve_used"] += 1
                    reserve_used += 1
                if last_request_started_at is not None:
                    elapsed = monotonic_fn() - last_request_started_at
                    remaining = minimum_interval - elapsed
                    if remaining > 0:
                        sleep_fn(remaining)
                last_request_started_at = monotonic_fn()
                _fetch_arm_page(state, outputs[topic_id], specs[topic_id],
                    outputs[topic_id], parent_dois, parent_pmcids,
                    seen_dois, seen_pmcids, request_log,
                    phase=phase, opener=opener, ledger=ledger)
                used += 1
                progressed = True
            if not progressed:
                break
        return used

    spend(base_limit, "planned")
    planned_complete = not any(searchable(state, reserve=False) for state in states.values())
    reserve_arms = [state["arm"]["arm_id"] for state in states.values()
        if searchable(state, reserve=True)]
    reserve_gate = {"planned_phase_complete": planned_complete,
        "primary_groups_before_reserve": _complete_primary_groups(outputs),
        "searchable_arm_ids_before_reserve": reserve_arms,
        "authorized": planned_complete and _complete_primary_groups(outputs) <
            V7_REQUIRED_PRIMARY_GROUPS and bool(reserve_arms)}
    if reserve_limit and reserve_gate["authorized"]:
        spend(reserve_limit, "reserve")

    for topic in outputs.values():
        topic.pop("_arm_states", None)
        total = topic["parent_record_count"] + topic["v6_record_count"] + len(topic["records_added"])
        if total >= 2:
            topic["status"] = "complete"
        elif topic["failure_reasons"]:
            topic["status"] = "incomplete_error"
        elif any(state["exhausted"] for state in states.values()
                if state["arm"]["topic_id"] == topic["topic_id"]):
            topic["status"] = "incomplete_exhausted"
        elif topic["status"] == "incomplete_searchable":
            topic["status"] = "incomplete_searchable"

    body = {"schema": V7_CAPTURE_SCHEMA, "captured_at": captured_at,
        "plan_sha256": frozen["plan_sha256"], "approval_sha256": approval["approval_sha256"],
        "approval_review_package_sha256": approval["review_package_sha256"],
        "parent_capture_sha256": frozen["parent_capture_sha256"],
        "parent_capture_file_sha256": frozen["parent_capture_file_sha256"],
        "parent_plan_sha256": frozen["parent_plan_sha256"],
        "v6_capture_sha256": frozen["v6_capture_sha256"],
        "v6_capture_file_sha256": frozen["v6_capture_file_sha256"],
        "v6_root_audit_sha256": frozen["v6_root_audit_sha256"],
        "v6_root_audit_file_sha256": frozen["v6_root_audit_file_sha256"],
        "source_gate_sha256": frozen["source_gate_sha256"],
        "v6_implementation_sha256": frozen["v6_implementation_sha256"],
        "v7_plan_builder_sha256": frozen["plan_builder_sha256"],
        "capture_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "topic_specs_sha256": frozen["topic_specs_sha256"],
        "identity_exclusion_sha256": frozen["identity_exclusion_sha256"],
        "hard_new_request_max": V7_MAX_NEW_REQUESTS,
        "planned_request_budget": base_limit, "planned_requests_used": planned_used,
        "reserve_request_budget": reserve_limit, "reserve_requests_used": reserve_used,
        "reserve_gate": reserve_gate, "request_budget": request_budget,
        "request_attempts": len(request_log), "request_failures": sum(1 for row in request_log
            if row.get("http_status") != 200 or row.get("error_type") or row.get("identity_error")),
        "cumulative_request_attempts": frozen["parent_cumulative_request_attempts"] + len(request_log),
        "cumulative_request_hard_max": frozen["cumulative_request_hard_max"],
        "request_log": request_log, "topics": outputs,
        "accepted_additions": sum(len(row["records_added"]) for row in outputs.values()),
        "primary_groups_complete": _complete_primary_groups(outputs),
        "required_primary_groups": V7_REQUIRED_PRIMARY_GROUPS,
        "diagnostic_only": True, "scoreable_cohort_created": False,
        "identity_exclusion_policy": "All v5/v6 DOI and PMCID identities are excluded before candidate audit, acceptance, retention, or ID-specific fetch; search-page matches are identity-only exclusions.",
        "pico_policy": "four_dimension_literal_pico: title/abstract must contain frozen literal phrases for all four PICO dimensions; no semantic inference or synonym expansion.",
        "license_policy": "License metadata is retained verbatim as source provenance and does not filter eligibility for this personal project; identity, abstract completeness, strict four-dimension PICO, deduplication, source-family, and per-group-cap gates remain mandatory.",
        "source_family_policy": "Forbidden historical DOI prefixes remain fail-closed.",
        "arms": [{key: row[key] for key in ("arm_id", "topic_id", "arm_type",
            "query_sha256", "start_cursor", "max_pages", "max_pages_with_reserve")}
            for row in arms]}
    body["capture_sha256"] = _capture_sha(body)
    body["audit"] = audit_v7_capture(body, plan=frozen,
        parent_capture=parent_capture, v6_capture=v6_capture,
        parent_plan_manifest=parent_plan_manifest)
    final_status = "complete" if body["audit"]["artifact_hash_valid"] and \
        body["audit"]["budget_valid"] and body["audit"]["cursor_continuity_valid"] and \
        body["audit"]["accepted_records_compliant"] else "terminal_audit_failure"
    artifact_file_sha = _atomic_write_json(capture_output_path, body)
    ledger.finish(capture_sha256=body["capture_sha256"], status=final_status,
        artifact_path=str(capture_output_path.resolve()), artifact_sha256=artifact_file_sha)
    return body
