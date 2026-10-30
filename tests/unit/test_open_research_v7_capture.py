import hashlib
import json
from copy import deepcopy
from urllib.parse import parse_qs, urlparse

import pytest

import src.eval.open_research_sources as sources
import src.eval.open_research_v6_continuation as v6
import src.eval.open_research_v7_capture as v7_capture
import src.eval.open_research_v7_query_plan as v7
from src.eval.open_research_sources import build_frozen_topic_plan_manifest
from src.util import sha256_obj


CAPTURED_AT = "2026-10-16T18:00:00Z"


def _topic(topic_id, kind):
    return {"topic_id": topic_id, "topic": f"topic {topic_id}", "slot_kind": kind,
        "condition_terms": [f"condition {topic_id}"],
        "population_terms": [f"adult cohort {topic_id}"],
        "intervention_terms": [f"intervention {topic_id}"],
        "outcome_terms": [f"outcome {topic_id}"]}


def _fixtures():
    specs = ([_topic(f"regular-{i:03}", "regular") for i in range(1, 40)] +
        [_topic(f"conflict-{i:03}", "conflict") for i in range(1, 10)] +
        [_topic(f"replacement-{i:03}", "replacement") for i in range(1, 13)])
    source_plan = build_frozen_topic_plan_manifest(specs, capture_date="2026-10-16")
    incomplete = {"regular-002", "regular-003", "regular-010", "regular-011",
        "regular-013", "regular-014", "regular-015", "regular-017", "regular-028",
        "regular-029", "regular-038", "replacement-001", "replacement-002",
        "replacement-006", "replacement-009", "replacement-010"}
    open_cursor = {"regular-002", "regular-003", "regular-010", "regular-011",
        "regular-013", "regular-015", "regular-017", "regular-029", "regular-038",
        "replacement-001", "replacement-006"}
    v5_topics, v6_topics = [], {}
    for index, spec in enumerate(source_plan["topic_specs"]):
        tid = spec["topic_id"]
        complete = tid not in incomplete
        records = []
        if complete:
            records = [{"doi": f"10.5555/{tid}.parent{n}", "pmcid": f"PMC{1000+index*2+n}",
                "license": "cc by", "title": "source", "abstractText": "x" * 500}
                for n in range(2)]
        pages = []
        if tid in open_cursor:
            pages = [{"page_index": 1, "cursor_mark": "*",
                "next_cursor_mark": f"v5-{tid}-next", "response_sha256": "a" * 64,
                "records_returned": 10, "hit_count": 100, "record_offset": 0}]
        seen = {"doi": f"10.5555/{tid}.seen", "pmcid": f"PMC{5000+index}"}
        v5_topics.append({"topic_id": tid, "topic": spec["topic"],
            "slot_kind": spec["slot_kind"], "query": sources.build_topic_query(spec,
                capture_date=source_plan["capture_date"]), "records": records,
            "core_pages": pages, "candidate_audit": [seen], "exclusions": [],
            "status": "complete" if complete else "incomplete"})
        v6_pages = ([{"page_index": 2, "cursor_mark": f"v5-{tid}-next",
            "next_cursor_mark": f"v6-{tid}-next", "response_sha256": "b" * 64,
            "records_returned": 10, "hit_count": 100, "record_offset": 10}]
            if tid in open_cursor else [])
        v6_topics[tid] = {"topic_id": tid, "slot_kind": spec["slot_kind"],
            "parent_record_count": len(records), "records_added": [], "core_pages": v6_pages,
            "candidate_audit": [{"doi": f"10.5555/{tid}.v6seen", "pmcid": f"PMC{6000+index}"}],
            "exclusions": [], "query": sources.build_topic_query(spec,
                capture_date=source_plan["capture_date"]),
            "search_exhausted": tid not in open_cursor, "status": "incomplete"}
    parent = {"schema": sources.TOPIC_SOURCE_CAPTURE_SCHEMA,
        "capture_date": source_plan["capture_date"], "captured_at": CAPTURED_AT,
        "frozen_topic_plan_sha256": source_plan["plan_sha256"],
        "capture_implementation_sha256": v6.EXPECTED_V5_IMPLEMENTATION_SHA256,
        "request_attempts": 160, "request_failures": 0,
        "request_log": [{"phase": "historical"} for _ in range(160)],
        "topics": v5_topics, "diagnostic_only": True, "scoreable_cohort_created": False}
    parent["capture_sha256"] = sha256_obj(parent)
    continuation = {"schema": v6.V6_CONTINUATION_SCHEMA,
        "parent_capture_sha256": parent["capture_sha256"],
        "parent_capture_file_sha256": "d" * 64, "parent_plan_sha256": source_plan["plan_sha256"],
        "parent_implementation_sha256": v6.EXPECTED_V5_IMPLEMENTATION_SHA256,
        "continuation_implementation_sha256": "b" * 64,
        "continuation_plan_sha256": "c" * 64, "cumulative_request_attempts": 252,
        "request_attempts": 92, "request_failures": 0, "primary_groups_complete": 35,
        "diagnostic_only": True, "scoreable_cohort_created": False, "topics": v6_topics}
    continuation["capture_sha256"] = sha256_obj(continuation)
    audit = {"schema": v7.V7_AUDIT_SCHEMA,
        "v6_capture_sha256": continuation["capture_sha256"],
        "continuation_plan_sha256": continuation["continuation_plan_sha256"],
        "parent_capture_sha256": parent["capture_sha256"],
        "parent_plan_sha256": source_plan["plan_sha256"],
        "cumulative_request_attempts": 252, "primary_groups_complete": 35,
        "parent_request_attempts": 160, "v6_request_attempts": 92,
        "diagnostic_only": True, "scoreable_cohort_created": False,
        "artifact_hash_valid": True, "budget_valid": True, "cursor_continuity_valid": True,
        "implementation_hash_valid": True, "accepted_records_compliant": True,
        "rejected_abstract_retention_compliant": True, "reserve_gate_valid": True,
        "request_phase_accounting_valid": True}
    audit["audit_sha256"] = sha256_obj(audit)
    plan = v7.build_v7_query_arm_plan(parent_capture=parent,
        parent_capture_file_sha256="d" * 64, parent_plan_manifest=source_plan,
        v6_capture=continuation, v6_capture_file_sha256="e" * 64,
        v6_root_audit=audit, v6_root_audit_file_sha256="f" * 64,
        max_new_requests=252)
    return source_plan, parent, continuation, audit, plan


class Response:
    status = 200

    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.body[:size] if size >= 0 else self.body


def _page(rows=(), *, hit_count=None, next_cursor="next"):
    if hit_count is None:
        hit_count = len(rows)
    return {"hitCount": hit_count, "nextCursorMark": next_cursor,
        "resultList": {"result": list(rows)}}


def _valid_source(spec, doi="10.7777/new-source", pmcid="PMC7777"):
    phrase = " ".join((spec["condition_terms"][0], spec["population_terms"][0],
        spec["intervention_terms"][0], spec["outcome_terms"][0]))
    return {"doi": doi, "pmcid": pmcid, "license": "cc by",
        "title": phrase, "abstractText": (phrase + ". " + "methods results. " * 70)}


def _run_inputs(plan, parent, continuation, audit, *, budget, opener, ledger_path):
    source_plan, _, _, _, _ = _fixtures()
    approval = {"schema": "g09-v7-capture-approval-v1", "status": "approved",
        "approved_plan_sha256": plan["plan_sha256"],
        "review_package_sha256": "a" * 64, "approved_at": CAPTURED_AT,
        "approved_by": "root"}
    approval["approval_sha256"] = sha256_obj(approval)
    return v7_capture.capture_v7_query_arm_plan(plan=plan,
        expected_plan_sha256=plan["plan_sha256"], parent_capture=parent,
        parent_capture_file_sha256="d" * 64, parent_plan_manifest=source_plan,
        v6_capture=continuation, v6_capture_file_sha256="e" * 64,
        v6_root_audit=audit, v6_root_audit_file_sha256="f" * 64,
        approval=approval, expected_approval_sha256=approval["approval_sha256"],
        captured_at=CAPTURED_AT, request_budget=budget, ledger_path=ledger_path,
        opener=opener, sleep_fn=lambda _seconds: None, monotonic_fn=lambda: 0.0)


def test_runner_binds_frozen_plan_discards_parent_identity_before_acceptance_and_stays_diagnostic(tmp_path):
    source_plan, parent, continuation, audit, plan = _fixtures()
    calls = []
    first_arm = plan["arms"][0]
    parent_seen = {"doi": "10.5555/regular-002.seen", "pmcid": "PMC5001",
        "license": "cc by", "title": "must never be retained", "abstractText": "secret" * 200}

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(_page([parent_seen], hit_count=1))

    result = _run_inputs(plan, parent, continuation, audit, budget=1, opener=opener,
        ledger_path=tmp_path / "run-ledger.json")
    topic = result["topics"][first_arm["topic_id"]]
    assert len(calls) == 1
    assert topic["records_added"] == []
    assert topic["exclusions"][0]["reason"] == "parent_seen_source_identity"
    assert "abstract_text" not in topic["exclusions"][0]
    ledger_receipt = json.loads((tmp_path / "run-ledger.json").read_text(
        encoding="utf-8"))["request_intents"][0]["response_receipt"]
    assert ledger_receipt["identities"] == [{"doi": "10.5555/regular-002.seen",
        "pmcid": "pmc5001"}]
    assert ledger_receipt["candidate_projection"] == []
    assert result["request_attempts"] == 1
    assert result["diagnostic_only"] is True
    assert result["scoreable_cohort_created"] is False
    assert result["capture_sha256"] == v7_capture._capture_sha(result)
    assert result["parent_capture_sha256"] == parent["capture_sha256"]
    assert result["v6_capture_sha256"] == continuation["capture_sha256"]
    assert result["reserve_requests_used"] == 0


def test_runner_keeps_strict_pico_gate_and_deduplicates_across_query_arms(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    valid = _valid_source(next(row for row in _fixtures()[0]["topic_specs"]
        if row["topic_id"] == plan["arms"][0]["topic_id"]))
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        # The first two arms belong to the same topic; the same source must not be accepted twice.
        return Response(_page([valid], hit_count=1))

    result = _run_inputs(plan, parent, continuation, audit, budget=2, opener=opener,
        ledger_path=tmp_path / "run-ledger.json")
    topic_id = plan["arms"][0]["topic_id"]
    topic = result["topics"][topic_id]
    assert len(calls) == 2
    assert len(topic["records_added"]) == 1
    assert any(row["reason"] == "duplicate_source_across_v7_arms"
        for row in topic["exclusions"])
    assert result["accepted_additions"] == 1
    assert "four_dimension_literal_pico" in result["pico_policy"]
    ledger = json.loads((tmp_path / "run-ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["request_intents"][0]["response_receipt"]["candidate_projection"]) == 1


def test_runner_stops_on_cursor_replay_and_does_not_repeat_request(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    first = plan["arms"][0]
    calls = []

    def opener(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        cursor = query["cursorMark"][0]
        calls.append(cursor)
        if len(calls) == 1:
            return Response(_page([{}], hit_count=20, next_cursor=cursor))
        return Response(_page([]))

    result = _run_inputs(plan, parent, continuation, audit, budget=43, opener=opener,
        ledger_path=tmp_path / "run-ledger.json")
    assert len(calls) == 43
    first_topic = result["topics"][first["topic_id"]]
    # The repeated next cursor is rejected before any second-page request is scheduled.
    assert result["request_attempts"] == 43
    assert any(row["reason"] == "cursor_missing_or_repeated"
        for row in first_topic["failure_reasons"])
    assert result["audit"]["cursor_continuity_valid"] is False


def test_runner_failure_injection_is_recorded_and_does_not_replay_or_exceed_budget(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        raise TimeoutError("synthetic timeout")

    result = _run_inputs(plan, parent, continuation, audit, budget=1, opener=opener,
        ledger_path=tmp_path / "run-ledger.json")
    assert len(calls) == 1
    assert result["request_failures"] == 1
    assert result["request_log"][0]["error_type"] == "TimeoutError"
    assert result["topics"][plan["arms"][0]["topic_id"]]["failure_reasons"][0]["reason"] == \
        "core_search_request_failed"
    assert "cursor_missing_or_repeated" not in [
        row["reason"] for row in result["topics"][plan["arms"][0]["topic_id"]]["failure_reasons"]]
    ledger = json.loads((tmp_path / "run-ledger.json").read_text(encoding="utf-8"))
    assert ledger["request_intents"][0]["state"] == "transport_or_parse_failure"
    assert ledger["status"] == "complete"


def test_malformed_response_is_terminal_transport_failure_not_cursor_replay(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()

    class BadJsonResponse(Response):
        def read(self, size=-1):
            return b"not-json"

    calls = []
    result = _run_inputs(plan, parent, continuation, audit, budget=1,
        opener=lambda request, timeout: (calls.append(request.full_url) or BadJsonResponse(b"")),
        ledger_path=tmp_path / "run-ledger.json")
    topic = result["topics"][plan["arms"][0]["topic_id"]]
    assert len(calls) == 1
    assert topic["status"] == "incomplete_error"
    assert topic["failure_reasons"][0]["reason"] == "core_search_request_failed"
    assert all(row["reason"] != "cursor_missing_or_repeated"
        for row in topic["failure_reasons"])
    assert result["request_failures"] == 1


def test_incomplete_non_cc_by_source_is_not_projected_or_accepted(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    unlicensed = {"doi": "10.7000/unlicensed", "pmcid": "PMC7000",
        "license": "cc-by-nc", "title": "restricted source", "abstractText": "private " * 20}
    result = _run_inputs(plan, parent, continuation, audit, budget=1,
        opener=lambda request, timeout: Response(_page([unlicensed], hit_count=1)),
        ledger_path=tmp_path / "run-ledger.json")
    receipt = json.loads((tmp_path / "run-ledger.json").read_text(
        encoding="utf-8"))["request_intents"][0]["response_receipt"]
    assert result["accepted_additions"] == 0
    assert receipt["identities"] == [{"doi": "10.7000/unlicensed", "pmcid": "pmc7000"}]
    assert receipt["candidate_projection"] == []
    assert "abstractText" not in json.dumps(receipt)


@pytest.mark.parametrize("license_value", ["cc-by-nc", None])
def test_unknown_or_non_cc_by_license_is_provenance_not_an_acceptance_gate(tmp_path,
        license_value):
    _, parent, continuation, audit, plan = _fixtures()
    spec = next(row for row in _fixtures()[0]["topic_specs"]
        if row["topic_id"] == plan["arms"][0]["topic_id"])
    source = _valid_source(spec, doi="10.7001/non-cc-by", pmcid="PMC7001")
    source["license"] = license_value
    result = _run_inputs(plan, parent, continuation, audit, budget=1,
        opener=lambda request, timeout: Response(_page([source], hit_count=1)),
        ledger_path=tmp_path / "non-cc-by-ledger.json")
    topic = result["topics"][plan["arms"][0]["topic_id"]]
    assert result["accepted_additions"] == 1
    assert topic["records_added"][0]["license"] == license_value
    assert result["audit"]["accepted_records_compliant"] is True
    receipt = json.loads((tmp_path / "non-cc-by-ledger.json").read_text(
        encoding="utf-8"))["request_intents"][0]["response_receipt"]
    assert receipt["candidate_projection"][0]["license"] == license_value
    assert "personal project" in result["license_policy"]


def test_reserve_is_used_only_after_base_arms_finish_and_remains_within_each_cap(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(_page([{}], hit_count=100_000, next_cursor=f"c{len(calls)}"))

    result = _run_inputs(plan, parent, continuation, audit, budget=252, opener=opener,
        ledger_path=tmp_path / "run-ledger.json")
    assert result["planned_requests_used"] == plan["base_request_budget"]
    assert result["reserve_requests_used"] == plan["reserve_request_budget"]
    assert result["reserve_gate"]["planned_phase_complete"] is True
    assert result["reserve_gate"]["authorized"] is True
    assert result["request_attempts"] == 252
    assert result["audit"]["budget_valid"] is True
    per_arm = {}
    for request in result["request_log"]:
        per_arm[request["arm_id"]] = per_arm.get(request["arm_id"], 0) + 1
    assert max(per_arm.values()) <= 8
    phases = [row["phase"] for row in result["request_log"]]
    assert phases.index("reserve") >= phases.index("planned")


def test_response_identity_receipt_survives_artifact_finalization_crash_and_blocks_replay(
        tmp_path, monkeypatch):
    _, parent, continuation, audit, plan = _fixtures()
    ledger_path = tmp_path / "run-ledger.json"
    calls = []
    seen = {"doi": "10.5555/regular-002.seen", "pmcid": "PMC5001",
        "license": "cc by", "title": "not retained", "abstractText": "x" * 500}
    monkeypatch.setattr(v7_capture, "_atomic_write_json",
        lambda path, body: (_ for _ in ()).throw(OSError("simulated finalization crash")))
    with pytest.raises(OSError, match="simulated finalization crash"):
        _run_inputs(plan, parent, continuation, audit, budget=1,
            opener=lambda request, timeout: (calls.append(request.full_url) or
                Response(_page([seen], hit_count=1))),
            ledger_path=ledger_path)
    ledger_audit = v7_capture.audit_v7_write_ahead_ledger(ledger_path)
    assert ledger_audit["ledger_hash_valid"] is True
    assert ledger_audit["response_receipts_valid"] is True
    assert ledger_audit["observed_identity_count"] == 1
    first_arm = next(row for row in plan["arms"]
        if row["arm_id"] == ledger_audit["consumed_cursors"][0]["arm_id"])
    assert ledger_audit["consumed_cursors"][0]["cursor_mark"] == \
        first_arm["start_cursor"]
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        _run_inputs(plan, parent, continuation, audit, budget=1,
            opener=lambda request, timeout: calls.append(request.full_url),
            ledger_path=ledger_path)
    assert len(calls) == 1


def test_response_persistence_crash_marks_cursor_unknown_and_forbids_replay(
        tmp_path, monkeypatch):
    _, parent, continuation, audit, plan = _fixtures()
    ledger_path = tmp_path / "run-ledger.json"
    calls = []
    monkeypatch.setattr(v7_capture._WriteAheadLedger, "after_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("simulated receipt crash")))
    with pytest.raises(OSError, match="simulated receipt crash"):
        _run_inputs(plan, parent, continuation, audit, budget=1,
            opener=lambda request, timeout: (calls.append(request.full_url) or
                Response(_page([{"doi": "10.9999/unknown", "pmcid": "PMC9999"}],
                    hit_count=1))),
            ledger_path=ledger_path)
    ledger_audit = v7_capture.audit_v7_write_ahead_ledger(ledger_path)
    assert ledger_audit["ambiguous_in_flight_count"] == 1
    assert ledger_audit["observed_identity_count"] == 0
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        _run_inputs(plan, parent, continuation, audit, budget=1,
            opener=lambda request, timeout: calls.append(request.full_url),
            ledger_path=ledger_path)
    assert len(calls) == 1


def test_parent_pmcid_collision_consumes_novel_doi_for_later_arm(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    novel_doi = "10.8888/half-collision"
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            record = {"doi": novel_doi, "pmcid": "PMC5001", "license": "cc by",
                "title": "unrelated", "abstractText": "x" * 500}
        else:
            spec = next(row for row in _fixtures()[0]["topic_specs"]
                if row["topic_id"] == plan["arms"][1]["topic_id"])
            record = _valid_source(spec, doi=novel_doi, pmcid="PMC8888")
        return Response(_page([record], hit_count=1))

    result = _run_inputs(plan, parent, continuation, audit, budget=2, opener=opener,
        ledger_path=tmp_path / "run-ledger.json")
    assert result["accepted_additions"] == 0
    assert any(row.get("doi") == novel_doi and
        row["reason"] == "duplicate_source_across_v7_arms"
        for topic in result["topics"].values() for row in topic["exclusions"])


def test_runner_rejects_budget_over_plan_max_before_transport(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    calls = []
    with pytest.raises(ValueError, match="cannot exceed the frozen v7 plan"):
        _run_inputs(plan, parent, continuation, audit, budget=253,
            opener=lambda request, timeout: calls.append(request.full_url),
            ledger_path=tmp_path / "run-ledger.json")
    assert calls == []


def test_runner_paces_request_starts_at_frozen_minimum_interval(tmp_path):
    source_plan, parent, continuation, audit, plan = _fixtures()
    times = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        times[0] += seconds

    def opener(request, timeout):
        times[0] += 0.05
        return Response(_page([], hit_count=100, next_cursor=f"cursor-{len(sleeps)}"))

    approval = {"schema": "g09-v7-capture-approval-v1", "status": "approved",
        "approved_plan_sha256": plan["plan_sha256"], "review_package_sha256": "a" * 64,
        "approved_at": CAPTURED_AT, "approved_by": "root"}
    approval["approval_sha256"] = sha256_obj(approval)
    v7_capture.capture_v7_query_arm_plan(plan=plan,
        expected_plan_sha256=plan["plan_sha256"], parent_capture=parent,
        parent_capture_file_sha256="d" * 64, parent_plan_manifest=source_plan,
        v6_capture=continuation, v6_capture_file_sha256="e" * 64,
        v6_root_audit=audit, v6_root_audit_file_sha256="f" * 64,
        approval=approval, expected_approval_sha256=approval["approval_sha256"],
        captured_at=CAPTURED_AT, ledger_path=tmp_path / "paced-ledger.json",
        request_budget=2, opener=opener, sleep_fn=sleep,
        monotonic_fn=lambda: times[0])
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.95)


def test_existing_write_ahead_ledger_forbids_same_plan_replay(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    calls = []
    ledger = tmp_path / "run-ledger.json"
    ledger.write_text(json.dumps({"plan_sha256": plan["plan_sha256"],
        "status": "request_in_flight"}), encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run_inputs(plan, parent, continuation, audit, budget=1,
            opener=lambda request, timeout: calls.append(request.full_url),
            ledger_path=ledger)
    assert calls == []


def test_unapproved_or_mismatched_plan_never_creates_ledger_or_calls_transport(tmp_path):
    _, parent, continuation, audit, plan = _fixtures()
    source_plan = _fixtures()[0]
    approval = {"schema": "g09-v7-capture-approval-v1", "status": "approved",
        "approved_plan_sha256": "0" * 64, "review_package_sha256": "a" * 64,
        "approved_at": CAPTURED_AT, "approved_by": "root"}
    approval["approval_sha256"] = sha256_obj(approval)
    calls = []
    with pytest.raises(ValueError, match="root approval does not bind"):
        v7_capture.capture_v7_query_arm_plan(plan=plan,
            expected_plan_sha256=plan["plan_sha256"], parent_capture=parent,
            parent_capture_file_sha256="d" * 64, parent_plan_manifest=source_plan,
            v6_capture=continuation, v6_capture_file_sha256="e" * 64,
            v6_root_audit=audit, v6_root_audit_file_sha256="f" * 64,
            approval=approval, expected_approval_sha256=approval["approval_sha256"],
            captured_at=CAPTURED_AT, ledger_path=tmp_path / "run-ledger.json",
            opener=lambda request, timeout: calls.append(request.full_url), request_budget=1)
    assert calls == []
    assert not (tmp_path / "run-ledger.json").exists()
