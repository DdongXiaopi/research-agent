import hashlib
import json
from copy import deepcopy
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import src.eval.open_research_sources as source_module
import src.eval.open_research_v6_continuation as v6
from src.eval.open_research_sources import build_frozen_topic_plan_manifest
from src.util import sha256_obj


CAPTURED_AT = "2026-10-16T18:00:00Z"
FILE_SHA = "c" * 64


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


def topic_spec(topic_id, slot_kind):
    return {"topic_id": topic_id, "topic": f"research topic {topic_id}",
        "slot_kind": slot_kind, "condition_terms": [f"condition {topic_id}"],
        "population_terms": [f"adult patients {topic_id}"],
        "intervention_terms": [f"treatment {topic_id}"],
        "outcome_terms": [f"clinical outcome {topic_id}"]}


def make_plan():
    specs = ([topic_spec(f"regular-{i:03}", "regular") for i in range(1, 40)] +
        [topic_spec(f"conflict-{i:03}", "conflict") for i in range(1, 10)] +
        [topic_spec(f"replacement-{i:03}", "replacement") for i in range(1, 13)])
    return build_frozen_topic_plan_manifest(specs, capture_date="2026-10-16")


def good_record(doi, pmcid, *, spec=None, outcome=True, license_value="cc by"):
    spec = spec or topic_spec("generic", "regular")
    condition, population = spec["condition_terms"][0], spec["population_terms"][0]
    intervention, outcome_phrase = spec["intervention_terms"][0], spec["outcome_terms"][0]
    text = f"{condition} in {population} received {intervention}. "
    if outcome:
        text += f"{outcome_phrase} was measured and reported. "
    text += "Methods and results are described for the study population. " * 30
    return {"doi": doi, "pmcid": pmcid, "license": license_value,
        "title": f"{condition} trial of {intervention} in {population}",
        "abstractText": text}


def page(cursor, rows, *, hit_count=1000, next_cursor=None):
    return {"hitCount": hit_count,
        "nextCursorMark": next_cursor or f"next-{cursor}",
        "resultList": {"result": rows}}


V5_INCOMPLETE_IDS = {
    "regular-002", "regular-003", "regular-010", "regular-011", "regular-013",
    "regular-014", "regular-015", "regular-017", "regular-026", "regular-028",
    "regular-029", "regular-038", "regular-039",
    "conflict-001", "conflict-002", "conflict-003", "conflict-004", "conflict-005",
    "conflict-006", "conflict-007", "conflict-008", "conflict-009",
    "replacement-001", "replacement-002", "replacement-006", "replacement-007",
    "replacement-009", "replacement-010",
}
EXHAUSTED_IDS = {"regular-014", "regular-028", "replacement-002", "replacement-009"}
V5_PARENT_COUNTS = {
    "regular-002": 0, "regular-003": 0, "regular-010": 0, "regular-011": 1,
    "regular-013": 0, "regular-014": 1, "regular-015": 1, "regular-017": 0,
    "regular-026": 1, "regular-028": 0, "regular-029": 0, "regular-038": 0,
    "regular-039": 1, "conflict-001": 0, "conflict-002": 0, "conflict-003": 0,
    "conflict-004": 0, "conflict-005": 0, "conflict-006": 0, "conflict-007": 0,
    "conflict-008": 0, "conflict-009": 0, "replacement-001": 0,
    "replacement-002": 1, "replacement-006": 0, "replacement-007": 1,
    "replacement-009": 0, "replacement-010": 0,
}


def make_v5_parent(plan, *, complete_ids=None, active_ids=(), exhausted_ids=EXHAUSTED_IDS,
        parent_record_counts=None):
    active_ids = set(active_ids)
    exhausted_ids = set(exhausted_ids)
    parent_record_counts = parent_record_counts or {}
    if complete_ids is None:
        complete_ids = {row["topic_id"] for row in plan["topic_specs"]} - V5_INCOMPLETE_IDS
    complete_ids = set(complete_ids)
    specs = {row["topic_id"]: row for row in plan["topic_specs"]}
    topics = []
    for index, spec in enumerate(plan["topic_specs"]):
        topic_id = spec["topic_id"]
        complete = topic_id in complete_ids
        records = []
        if complete:
            records = [good_record(f"10.5555/{topic_id}.{n}",
                f"PMC{100000 + index * 2 + n}", spec=spec) for n in range(2)]
        elif topic_id in active_ids or topic_id in exhausted_ids:
            count = parent_record_counts.get(topic_id, V5_PARENT_COUNTS.get(topic_id, 0))
            for n in range(count):
                records.append(good_record(f"10.5555/{topic_id}.parent{n}",
                    f"PMC{200000 + index * 2 + n}", spec=spec))
        if not complete:
            is_exhausted = topic_id in exhausted_ids
        else:
            is_exhausted = False
        if not complete and (topic_id in active_ids or topic_id in exhausted_ids):
            exhausted = is_exhausted
            returned = 0 if exhausted and parent_record_counts.get(topic_id,
                V5_PARENT_COUNTS.get(topic_id, 0)) == 0 else (1 if exhausted else 10)
            hit_count = returned if exhausted else 1000
            pages = [{"page_index": 1, "cursor_mark": "*",
                "next_cursor_mark": f"cursor-{topic_id}",
                "response_sha256": hashlib.sha256(topic_id.encode()).hexdigest(),
                "records_returned": returned, "hit_count": hit_count, "record_offset": 0}]
        elif not complete:
            pages = [{"page_index": 1, "cursor_mark": "*",
                "next_cursor_mark": f"cursor-{topic_id}",
                "response_sha256": hashlib.sha256(topic_id.encode()).hexdigest(),
                "records_returned": 10, "hit_count": 1000, "record_offset": 0}]
        else:
            pages = []
        candidate_audit = []
        for row in records:
            candidate_audit.append({"doi": row["doi"], "pmcid": row["pmcid"],
                "core_rank": len(candidate_audit), "core_response_sha256": "a" * 64})
        topics.append({"topic_id": topic_id, "topic": spec["topic"],
            "slot_kind": spec["slot_kind"], "query": source_module.build_topic_query(
                spec, capture_date="2026-10-16"), "records": records,
            "core_pages": pages, "core_attempts": len(pages),
            "candidate_audit": candidate_audit, "exclusions": [],
            "status": "complete" if complete else "incomplete"})
    capture = {"schema": source_module.TOPIC_SOURCE_CAPTURE_SCHEMA,
        "capture_date": "2026-10-16", "captured_at": CAPTURED_AT,
        "frozen_topic_plan_sha256": plan["plan_sha256"],
        "capture_implementation_sha256": v6.EXPECTED_V5_IMPLEMENTATION_SHA256,
        "core_request_budget_max": 252, "request_attempts": 160,
        "request_failures": 0, "request_log": [{"phase": "historical"} for _ in range(160)],
        "topic_slots_requested": 60, "topics": topics,
        "diagnostic_only": True, "scoreable_cohort_created": False}
    capture["capture_sha256"] = sha256_obj(capture)
    return capture


def freeze(monkeypatch, parent, plan):
    monkeypatch.setattr(v6, "EXPECTED_V5_CAPTURE_SHA256", parent["capture_sha256"])
    monkeypatch.setattr(v6, "EXPECTED_V5_CAPTURE_FILE_SHA256", FILE_SHA)
    monkeypatch.setattr(v6, "EXPECTED_V5_PLAN_SHA256", plan["plan_sha256"])
    return v6.build_v6_continuation_plan(parent_capture=parent,
        parent_capture_file_sha256=FILE_SHA, parent_plan_manifest=plan)


def test_v6_plan_content_binds_immutable_v5_parent_and_fixed_request_split(monkeypatch):
    plan = make_plan()
    parent = make_v5_parent(plan, active_ids={"regular-033"})
    frozen = freeze(monkeypatch, parent, plan)

    assert frozen["parent_capture_sha256"] == parent["capture_sha256"]
    assert frozen["parent_capture_file_sha256"] == FILE_SHA
    assert frozen["parent_plan_sha256"] == plan["plan_sha256"]
    assert frozen["continuation_implementation_sha256"] == hashlib.sha256(
        Path(v6.__file__).read_bytes()).hexdigest()
    assert frozen["parent_request_attempts"] == 160
    assert frozen["planned_additional_requests"] == 89
    assert frozen["reserve_requests"] == 3
    assert frozen["hard_request_max"] == 252
    assert frozen["plan_sha256"] == sha256_obj(
        {key: value for key, value in frozen.items() if key != "plan_sha256"})


def test_v6_rejects_changed_parent_before_any_request_and_does_not_mutate_it(monkeypatch):
    plan = make_plan()
    parent = make_v5_parent(plan, active_ids={"regular-033"})
    frozen = freeze(monkeypatch, parent, plan)
    changed = deepcopy(parent)
    changed["topics"][0]["records"][0]["abstractText"] += "tamper"
    changed["capture_sha256"] = sha256_obj(
        {key: value for key, value in changed.items() if key != "capture_sha256"})
    calls = []

    with pytest.raises(ValueError, match="parent capture identity"):
        v6.continue_v5_capture(parent_capture=changed, parent_capture_file_sha256=FILE_SHA,
            parent_plan_manifest=plan, continuation_plan=frozen,
            verified_plan_sha256=frozen["plan_sha256"], captured_at=CAPTURED_AT,
            opener=lambda request, timeout: calls.append(request.full_url))
    assert calls == []
    assert parent["capture_sha256"] == frozen["parent_capture_sha256"]


def test_v6_rejects_rehashed_plan_that_changes_parent_derived_slot_before_network(monkeypatch):
    plan = make_plan()
    parent = make_v5_parent(plan)
    frozen = freeze(monkeypatch, parent, plan)
    changed = deepcopy(frozen)
    changed["slots"][0]["query"] += " OR DOI:10.1234/unsafe"
    changed["plan_sha256"] = sha256_obj(
        {key: value for key, value in changed.items() if key != "plan_sha256"})
    calls = []

    with pytest.raises(ValueError, match="parent-derived continuation slots"):
        v6.continue_v5_capture(parent_capture=parent, parent_capture_file_sha256=FILE_SHA,
            parent_plan_manifest=plan, continuation_plan=changed,
            verified_plan_sha256=changed["plan_sha256"], captured_at=CAPTURED_AT,
            opener=lambda request, timeout: calls.append(request.full_url))
    assert calls == []


def test_active_slots_resume_after_last_parent_cursor_in_topic_id_round_robin(monkeypatch):
    plan = make_plan()
    active = {"regular-033", "regular-034"}
    complete = {row["topic_id"] for row in plan["topic_specs"]} - active - EXHAUSTED_IDS
    parent = make_v5_parent(plan, complete_ids=complete, active_ids=active,
        parent_record_counts={"regular-033": 1, "regular-034": 1,
            "regular-028": 1, "replacement-009": 1})
    frozen = freeze(monkeypatch, parent, plan)
    calls = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        calls.append((params["query"][0], params["cursorMark"][0]))
        topic_id = next(row["topic_id"] for row in frozen["slots"]
            if row["query"] == params["query"][0] and row["initial_cursor"] == params["cursorMark"][0])
        call_index = sum(1 for query, _ in calls if query == params["query"][0])
        row = good_record(f"10.7777/{topic_id}.{call_index}",
            f"PMC{700000 + len(calls)}",
            spec=next(spec for spec in plan["topic_specs"] if spec["topic_id"] == topic_id))
        prior_hit_count = next(slot["parent_terminal_page"]["hit_count"]
            for slot in frozen["slots"] if slot["topic_id"] == topic_id)
        return Response(page(params["cursorMark"][0], [row], hit_count=prior_hit_count,
            next_cursor=f"next-{topic_id}-{call_index}"))

    result = v6.continue_v5_capture(parent_capture=parent,
        parent_capture_file_sha256=FILE_SHA, parent_plan_manifest=plan,
        continuation_plan=frozen, verified_plan_sha256=frozen["plan_sha256"],
        captured_at=CAPTURED_AT, opener=opener)

    active_requests = [row for row in result["request_log"]
        if row["topic_id"] in active]
    assert [row["topic_id"] for row in active_requests] == ["regular-033", "regular-034"]
    assert [row["cursor_mark"] for row in active_requests] == [
        "cursor-regular-033", "cursor-regular-034"]
    assert result["topics"]["regular-033"]["status"] == "complete"
    assert result["topics"]["regular-034"]["status"] == "complete"
    assert result["diagnostic_only"] is True
    assert result["scoreable_cohort_created"] is False


def test_exhausted_primary_slots_use_strict_four_dimension_query_and_keep_gate(monkeypatch):
    plan = make_plan()
    exhausted = EXHAUSTED_IDS
    complete = {row["topic_id"] for row in plan["topic_specs"]} - exhausted
    parent = make_v5_parent(plan, complete_ids=complete, exhausted_ids=exhausted)
    for topic_id in exhausted:
        row = next(topic for topic in parent["topics"] if topic["topic_id"] == topic_id)
        if not row["records"]:
            spec = next(spec for spec in plan["topic_specs"] if spec["topic_id"] == topic_id)
            row["records"].append(good_record(f"10.5555/{topic_id}.parent",
                f"PMC{850000 + len(row['records'])}", spec=spec))
            row["candidate_audit"].append({"doi": row["records"][0]["doi"],
                "pmcid": row["records"][0]["pmcid"], "core_response_sha256": "b" * 64})
    parent["capture_sha256"] = sha256_obj(
        {key: value for key, value in parent.items() if key != "capture_sha256"})
    frozen = freeze(monkeypatch, parent, plan)
    calls = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        query = params["query"][0]
        calls.append(query)
        assert query.count('TITLE:"') >= 4 and query.count('ABSTRACT:"') >= 4
        topic_id = next(row["topic_id"] for row in frozen["slots"]
            if row["query"] == query)
        spec = next(spec for spec in plan["topic_specs"] if spec["topic_id"] == topic_id)
        row = good_record(f"10.8888/{topic_id}", f"PMC{800000 + len(calls)}", spec=spec)
        return Response(page(params["cursorMark"][0], [row], hit_count=1,
            next_cursor=f"alt-next-{topic_id}"))

    result = v6.continue_v5_capture(parent_capture=parent,
        parent_capture_file_sha256=FILE_SHA, parent_plan_manifest=plan,
        continuation_plan=frozen, verified_plan_sha256=frozen["plan_sha256"],
        captured_at=CAPTURED_AT, opener=opener)

    for topic_id in exhausted:
        topic = result["topics"][topic_id]
        assert topic["query_mode"] == "strict_four_dimension_alternative"
        assert topic["status"] == "complete"
        request = next(row for row in result["request_log"] if row["topic_id"] == topic_id)
        assert request["cursor_mark"] == "*"
    assert len(calls) == 4
    assert result["accepted_additions"] == 4


def test_exclusion_retains_abstract_only_after_identity_license_and_length_pass(monkeypatch):
    plan = make_plan()
    active = {"replacement-011", "replacement-012"}
    complete = {row["topic_id"] for row in plan["topic_specs"]} - active - EXHAUSTED_IDS
    parent = make_v5_parent(plan, complete_ids=complete, active_ids=active,
        parent_record_counts={"replacement-011": 1, "replacement-012": 1})
    frozen = freeze(monkeypatch, parent, plan)
    calls = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        slot = next(row for row in frozen["slots"] if row["query"] == params["query"][0])
        topic_id = slot["topic_id"]
        spec = next(spec for spec in plan["topic_specs"] if spec["topic_id"] == topic_id)
        calls.append(request.full_url)
        if topic_id == "replacement-011":
            row = good_record("10.9999/pico-miss", "PMC920001", spec=spec, outcome=False)
        elif topic_id == "replacement-012":
            row = good_record("10.9999/unlicensed", "PMC920002", spec=spec,
                license_value="cc by-nc")
        else:
            row = good_record(f"10.9999/{topic_id}-{len(calls)}", f"PMC{900000 + len(calls)}", spec=spec)
        hit_count = slot["parent_terminal_page"]["hit_count"] if slot["query_mode"] == "resume_v5_cursor" else 1
        return Response(page(params["cursorMark"][0], [row], hit_count=hit_count,
            next_cursor=f"next-{topic_id}-{len(calls)}"))

    result = v6.continue_v5_capture(parent_capture=parent,
        parent_capture_file_sha256=FILE_SHA, parent_plan_manifest=plan,
        continuation_plan=frozen, verified_plan_sha256=frozen["plan_sha256"],
        captured_at=CAPTURED_AT, opener=opener, request_budget=6)
    exclusions = [row for topic in result["topics"].values() for row in topic["exclusions"]]
    pico = next((row for row in exclusions if row["reason"] == "pico_relevance_missing_dimensions"), exclusions)
    assert isinstance(pico, dict) and pico.get("reason") == "pico_relevance_missing_dimensions", (exclusions, result["topics"]["replacement-011"])
    unlicensed = next(row for row in exclusions if row["reason"] == "core_license_not_cc_by")
    assert len(pico["abstract_text"]) >= 400
    assert set(pico["pico_relevance"]["missing_dimensions"]) == {"outcome"}
    assert "abstract_text" not in unlicensed


def test_budget_uses_89_planned_then_at_most_three_reserve_and_audit_is_self_verifying(monkeypatch):
    plan = make_plan()
    parent = make_v5_parent(plan, active_ids=V5_INCOMPLETE_IDS - EXHAUSTED_IDS,
        exhausted_ids=EXHAUSTED_IDS)
    frozen = freeze(monkeypatch, parent, plan)
    calls = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        calls.append((params["query"][0], params["cursorMark"][0]))
        index = len(calls)
        slot = next(row for row in frozen["slots"] if row["query"] == params["query"][0])
        spec = next(spec for spec in plan["topic_specs"] if spec["topic_id"] == slot["topic_id"])
        row = good_record(f"10.3333/missing-{index}", f"PMC{910000 + index}",
            spec=spec, outcome=False)
        return Response(page(params["cursorMark"][0], [row], hit_count=1000,
            next_cursor=f"v6-cursor-{index}"))

    result = v6.continue_v5_capture(parent_capture=parent,
        parent_capture_file_sha256=FILE_SHA, parent_plan_manifest=plan,
        continuation_plan=frozen, verified_plan_sha256=frozen["plan_sha256"],
        captured_at=CAPTURED_AT, opener=opener)
    audit = v6.audit_v6_continuation(result, parent_capture=parent,
        parent_plan_manifest=plan)

    assert len(calls) == 92, (result["planned_requests_used"], result["reserve_requests_used"],
        [(key, value["status"], value["failure_reason"], value["search_exhausted"])
            for key, value in result["topics"].items() if value["query_mode"] != "parent_unchanged"])
    assert result["planned_requests_used"] == 89
    assert result["reserve_requests_used"] == 3
    assert [row["phase"] for row in result["request_log"][:89]] == ["planned"] * 89
    assert [row["phase"] for row in result["request_log"][89:]] == ["reserve"] * 3
    assert result["cumulative_request_attempts"] == 252
    assert audit["budget_valid"] is True
    assert audit["cursor_continuity_valid"] is True
    assert audit["parent_capture_sha256"] == parent["capture_sha256"]
    assert audit["diagnostic_only"] is True
    assert audit["accepted_records_compliant"] is True
    assert audit["rejected_abstract_retention_compliant"] is True
    assert audit["request_phase_accounting_valid"] is True
    assert result["diagnostic_only"] is True
    assert result["scoreable_cohort_created"] is False


def test_root_audit_detects_first_cursor_replay_even_when_child_hash_is_recomputed(monkeypatch):
    plan = make_plan()
    active = {"replacement-011"}
    complete = {row["topic_id"] for row in plan["topic_specs"]} - active - EXHAUSTED_IDS
    parent = make_v5_parent(plan, complete_ids=complete, active_ids=active,
        parent_record_counts={"replacement-011": 1})
    frozen = freeze(monkeypatch, parent, plan)

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        spec = next(spec for spec in plan["topic_specs"] if spec["topic_id"] == "replacement-011")
        record = good_record("10.1212/new", "PMC921212", spec=spec)
        return Response(page(params["cursorMark"][0], [record], hit_count=1000,
            next_cursor="after-v6"))

    result = v6.continue_v5_capture(parent_capture=parent,
        parent_capture_file_sha256=FILE_SHA, parent_plan_manifest=plan,
        continuation_plan=frozen, verified_plan_sha256=frozen["plan_sha256"],
        captured_at=CAPTURED_AT, opener=opener, request_budget=5)
    continuation_request = next(row for row in result["request_log"]
        if row["topic_id"] == "replacement-011")
    continuation_request["cursor_mark"] = "*"
    result["capture_sha256"] = sha256_obj(
        {key: value for key, value in result.items() if key != "capture_sha256"})

    audit = v6.audit_v6_continuation(result, parent_capture=parent,
        parent_plan_manifest=plan)
    assert audit["artifact_hash_valid"] is True
    assert audit["cursor_continuity_valid"] is False


def test_full_abstract_is_not_retained_when_identity_is_missing_or_abstract_is_short(monkeypatch):
    plan = make_plan()
    active = {"replacement-011", "replacement-012"}
    complete = {row["topic_id"] for row in plan["topic_specs"]} - active - EXHAUSTED_IDS
    parent = make_v5_parent(plan, complete_ids=complete, active_ids=active,
        parent_record_counts={"replacement-011": 1, "replacement-012": 1})
    frozen = freeze(monkeypatch, parent, plan)

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        slot = next(row for row in frozen["slots"] if row["query"] == params["query"][0])
        spec = next(spec for spec in plan["topic_specs"] if spec["topic_id"] == slot["topic_id"])
        record = good_record(f"10.1313/{slot['topic_id']}", "PMC931313", spec=spec)
        if slot["topic_id"] == "replacement-011":
            record["doi"] = None
        else:
            record["abstractText"] = "too short"
        return Response(page(params["cursorMark"][0], [record], hit_count=1000,
            next_cursor=f"cursor-{slot['topic_id']}"))

    result = v6.continue_v5_capture(parent_capture=parent,
        parent_capture_file_sha256=FILE_SHA, parent_plan_manifest=plan,
        continuation_plan=frozen, verified_plan_sha256=frozen["plan_sha256"],
        captured_at=CAPTURED_AT, opener=opener, request_budget=6)
    excluded = [row for topic in result["topics"].values() for row in topic["exclusions"]]
    assert all("abstract_text" not in row for row in excluded)


def test_all_parent_seen_identities_are_skipped_and_accepted_additions_are_globally_unique(monkeypatch):
    plan = make_plan()
    active = {"replacement-011", "replacement-012"}
    complete = {row["topic_id"] for row in plan["topic_specs"]} - active - EXHAUSTED_IDS
    parent = make_v5_parent(plan, complete_ids=complete, active_ids=active,
        parent_record_counts={"replacement-011": 1, "replacement-012": 1})
    parent["topics"][0]["candidate_audit"].append({
        "doi": "10.4444/seen-but-rejected", "pmcid": "PMC123456",
        "core_response_sha256": "d" * 64})
    parent["capture_sha256"] = sha256_obj(
        {key: value for key, value in parent.items() if key != "capture_sha256"})
    frozen = freeze(monkeypatch, parent, plan)
    calls = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        calls.append(params["cursorMark"][0])
        index = len(calls)
        slot = next(row for row in frozen["slots"]
            if row["query"] == params["query"][0] and row["query_mode"] == "resume_v5_cursor")
        topic_id = slot["topic_id"]
        spec = next(spec for spec in plan["topic_specs"] if spec["topic_id"] == topic_id)
        if topic_id == "replacement-011":
            row = good_record("10.4444/seen-but-rejected", "PMC123456", spec=spec)
        else:
            row = good_record(f"10.4444/new-source-{topic_id}", f"PMC123457", spec=spec)
        return Response(page(params["cursorMark"][0], [row], hit_count=1000,
            next_cursor=f"next-{topic_id}-{index}"))

    result = v6.continue_v5_capture(parent_capture=parent,
        parent_capture_file_sha256=FILE_SHA, parent_plan_manifest=plan,
        continuation_plan=frozen, verified_plan_sha256=frozen["plan_sha256"],
        captured_at=CAPTURED_AT, opener=opener, request_budget=6)

    assert result["topics"]["replacement-011"]["records_added"] == []
    assert result["topics"]["replacement-012"]["records_added"][0]["doi"] == "10.4444/new-source-replacement-012"
    assert any(row["reason"] == "parent_seen_source_identity" for row in
        result["topics"]["replacement-011"]["exclusions"])
    assert len({row["doi"] for topic in result["topics"].values()
        for row in topic["records_added"]}) == result["accepted_additions"]
