import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from src.eval.open_research_sources import (
    assess_topic_pico_relevance,
    build_topic_query,
    build_frozen_topic_plan_manifest,
    capture_frozen_topic_stratified_source_pool,
    capture_europe_pmc_source_pool,
    capture_topic_stratified_source_pool,
    select_lite_candidates,
    select_topic_candidates,
    source_query,
    validate_frozen_topic_plan,
)
import src.eval.open_research_sources as source_module
from src.util import sha256_obj


def lite_row(index, *, doi=None, journal=None):
    return {"doi": doi or f"10.5555/g09.{index}", "pmcid": f"PMC{800000 + index}",
        "title": f"A diverse research study with experimental comparison {index}",
        "journalTitle": journal or f"Journal {index}", "pubYear": "2025"}


def core_record(row):
    idx = int(row["pmcid"].removeprefix("PMC")) - 800000
    return {"doi": row["doi"], "pmcid": row["pmcid"], "license": "cc by",
        "title": row["title"], "abstractText": (
            "condition alpha adult patients treatment alpha clinical outcome. "
            f"BACKGROUND Study {idx} investigated a defined population. "
            "METHODS The investigators compared the prespecified intervention with a control. "
            "RESULTS The primary outcome changed in the intervention group; the abstract reports a confidence interval. "
            "CONCLUSIONS The findings are limited to the studied population and design. " * 3),
        "pubYear": "2025", "journalInfo": {"journal": {"title": row["journalTitle"]}}}


def topic_spec(topic_id, *, slot_kind="regular"):
    return {"topic_id": topic_id, "topic": f"topic {topic_id}", "slot_kind": slot_kind,
        "condition_terms": ["condition alpha", "condition beta"],
        "population_terms": ["adult patients"],
        "intervention_terms": ["treatment alpha"],
        "outcome_terms": ["clinical outcome"]}


class Response:
    status = 200

    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.body[:size] if size >= 0 else self.body


@pytest.mark.parametrize("module_name", ["src.eval.open_research_sources",
                                      "src.eval.open_research_v6_continuation",
                                      "src.eval.open_research_v7_capture"])
def test_capture_response_rejects_unapproved_url_and_oversized_body(module_name, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    request_json = getattr(module, "_request_json", None) or module._response_json
    monkeypatch.setattr(source_module, "MAX_SOURCE_RESPONSE_BYTES", 16)
    calls = []
    opener = lambda request, timeout: (calls.append(request.full_url) or Response({"payload": "x" * 32}))
    with pytest.raises(ValueError, match="approved Europe PMC"):
        request_json("http://127.0.0.1/private", opener=opener)
    assert calls == []
    with pytest.raises(ValueError, match="byte limit"):
        request_json(source_module.API_ROOT + "/search?query=test", opener=opener)
    assert len(calls) == 1


def test_capture_default_transport_does_not_follow_redirects(monkeypatch):
    from types import SimpleNamespace
    handlers = []
    def fake_builder(handler):
        handlers.append(handler)
        return SimpleNamespace(open=lambda *args, **kwargs: Response({"ok": True}))
    monkeypatch.setattr(source_module, "build_opener", fake_builder)
    value, _ = source_module._request_json(source_module.API_ROOT + "/search?query=test")
    assert value == {"ok": True}
    assert isinstance(handlers[0], source_module._NoRedirect)
    assert handlers[0].redirect_request(None, None, 302, "redirect", {}, "http://127.0.0.1") is None


def test_source_query_pins_cc_by_date_scope_and_excludes_historical_source_family():
    query = source_query("2026-10-16")
    assert "LICENSE:cc-by" in query
    assert "FIRST_PDATE:[2020-01-01 TO 2026-10-16]" in query
    assert "NOT DOI:10.1371*" in query
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        source_query("20261016")


def test_topic_query_quotes_and_escapes_terms_without_allowing_query_injection():
    spec = topic_spec("condition-alpha")
    spec["condition_terms"][0] = 'condition "alpha" OR DOI:10.1371'
    query = build_topic_query(spec, capture_date="2026-10-16")
    assert '"condition \\"alpha\\" OR DOI:10.1371"' in query
    assert query.count("NOT DOI:10.1371*") == 1
    assert "LICENSE:cc-by" in query and "HAS_ABSTRACT:Y" in query
    assert 'TITLE:"condition \\"alpha\\" OR DOI:10.1371"' in query
    assert 'ABSTRACT:"condition \\"alpha\\" OR DOI:10.1371"' in query
    assert 'AND "condition \\"alpha\\" OR DOI:10.1371"' not in query
    bad = topic_spec("unsafe-term")
    bad["condition_terms"][0] = "unsafe\nterm"
    with pytest.raises(ValueError, match="query term"):
        build_topic_query(bad, capture_date="2026-10-16")
    bad_omitted_dimension = topic_spec("unsafe-omitted-dimension")
    bad_omitted_dimension["outcome_terms"][0] = "unsafe\noutcome"
    with pytest.raises(ValueError, match="query term"):
        build_topic_query(bad_omitted_dimension, capture_date="2026-10-16")


def test_topic_search_query_uses_condition_intervention_and_local_gate_keeps_all_four_pico():
    query = build_topic_query(topic_spec("topic-scoped"), capture_date="2026-10-16")
    for term in ("condition alpha", "condition beta", "treatment alpha"):
        assert f'TITLE:"{term}"' in query
        assert f'ABSTRACT:"{term}"' in query
        assert f'AND "{term}"' not in query
    assert 'TITLE:"adult patients"' not in query
    assert 'TITLE:"clinical outcome"' not in query
    assert assess_topic_pico_relevance({
        "title": "Condition alpha treatment alpha trial",
        "abstractText": "Adult patients received treatment alpha and the clinical outcome was measured."},
        topic_spec("topic-scoped"))["eligible"] is True


def test_topic_candidate_selection_is_stable_bounded_and_journal_diverse():
    rows = [lite_row(index) for index in range(48)]
    selected = select_topic_candidates(rows, topic_id="topic-a", max_candidates=8)
    assert selected == select_topic_candidates(rows, topic_id="topic-a", max_candidates=8)
    assert len(selected) == 8
    assert len({row["journal"] for row in selected}) == 8
    assert [row["metadata_rank"] for row in selected] == list(range(8))
    assert len(select_topic_candidates(rows, topic_id="topic-a", max_candidates=4)) == 4


def test_topic_candidate_selection_preserves_europe_pmc_rank_and_binds_metadata_receipt():
    rows = [lite_row(index) for index in range(6)]
    rows[0]["journalTitle"] = rows[1]["journalTitle"] = "same journal"
    selected = select_topic_candidates(rows, topic_id="topic-a", max_candidates=4,
        metadata_response_sha256="a" * 64)
    assert [row["doi"] for row in selected] == [rows[i]["doi"] for i in (0, 2, 3, 4)]
    assert [row["metadata_rank"] for row in selected] == [0, 2, 3, 4]
    assert all(row["metadata_response_sha256"] == "a" * 64 for row in selected)
    spec = topic_spec("topic-a")
    rows[1]["title"] = "condition alpha adult patients treatment alpha clinical outcome"
    reranked = select_topic_candidates(rows, topic_id="topic-a", max_candidates=1,
        topic_spec=spec)
    assert reranked[0]["doi"] == rows[1]["doi"]
    assert reranked[0]["metadata_rank"] == 1
    assert reranked[0]["title_pico_dimensions_matched"] == 4


def test_topic_candidate_rerank_sees_strong_title_candidate_beyond_old_100_row_window():
    rows = [lite_row(index) for index in range(150)]
    strong = rows[120]
    strong["title"] = "condition alpha adult patients treatment alpha clinical outcome trial"
    selected = select_topic_candidates(rows, topic_id="wide-window", max_candidates=1,
        topic_spec=topic_spec("wide-window"))
    assert selected[0]["doi"] == strong["doi"]
    assert selected[0]["metadata_rank"] == 120
    assert selected[0]["title_pico_dimensions_matched"] == 4


@pytest.mark.parametrize("spec,record", [
    ({"condition_terms": ["heart failure"],
      "population_terms": ["reduced ejection fraction"],
      "intervention_terms": ["SGLT2 inhibitor"],
      "outcome_terms": ["heart failure hospitalization"]},
     {"title": "SGLT2 inhibitor in heart failure with preserved ejection fraction",
      "abstractText": "Adults with preserved ejection fraction received an SGLT2 inhibitor; heart failure hospitalization was measured."}),
    ({"condition_terms": ["influenza"], "population_terms": ["older adults"],
      "intervention_terms": ["influenza vaccine"], "outcome_terms": ["hospitalization"]},
     {"title": "RSV vaccine effectiveness among older adults",
      "abstractText": "Respiratory syncytial virus vaccination reduced hospitalization among older adults."}),
    ({"condition_terms": ["chronic kidney disease"],
      "population_terms": ["adults with chronic kidney disease"],
      "intervention_terms": ["intravenous iron"], "outcome_terms": ["hemoglobin"]},
     {"title": "Intravenous iron for anemia during pregnancy",
      "abstractText": "Pregnant adults received intravenous iron and hemoglobin was measured."}),
])
def test_pico_relevance_filter_rejects_hf_preserved_influenza_rsv_and_pregnancy_ckd_mismatch(
        spec, record):
    assessment = assess_topic_pico_relevance(record, spec)
    assert assessment["eligible"] is False
    assert assessment["missing_dimensions"]


def test_pico_relevance_requires_evidence_for_every_dimension_and_reports_matches():
    spec = {"condition_terms": ["condition alpha", "condition beta"],
        "population_terms": ["adult patients"], "intervention_terms": ["treatment alpha"],
        "outcome_terms": ["clinical outcome"]}
    candidate = {"title": "Condition alpha study", "abstractText": (
        "Adult patients received treatment alpha and the clinical outcome was measured.")}
    assessment = assess_topic_pico_relevance(candidate, spec)
    assert assessment == {"eligible": True, "matched_terms": {
        "condition": ["condition alpha"], "population": ["adult patients"],
        "intervention": ["treatment alpha"], "outcome": ["clinical outcome"]},
        "missing_dimensions": []}
    del candidate["abstractText"]
    assert assess_topic_pico_relevance(candidate, spec)["eligible"] is False


def test_topic_capture_skips_ineligible_records_refills_within_slot_and_logs_caps():
    metadata = [lite_row(index) for index in range(64)]
    selected = select_topic_candidates(metadata, topic_id="group-a", max_candidates=4)
    by_pmcid = {item["pmcid"]: core_record(item) for item in metadata}
    by_pmcid[selected[0]["pmcid"]]["license"] = "cc by-nc"
    by_pmcid[selected[1]["pmcid"]]["abstractText"] = "Short abstract."
    requested = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        if params.get("resulttype") == ["lite"]:
            assert params.get("pageSize") == ["100"]
            return Response({"hitCount": len(metadata), "resultList": {"result": metadata}})
        pmcid = params["query"][0].removeprefix("PMCID:")
        requested.append(pmcid)
        return Response({"hitCount": 1, "resultList": {"result": [by_pmcid[pmcid]]}})

    result = capture_topic_stratified_source_pool(
        topic_specs=[topic_spec("group-a")], capture_date="2026-10-16",
        captured_at="2026-10-16T12:00:00Z", opener=opener)
    topic = result["topics"][0]
    assert topic["status"] == "complete"
    assert len(topic["records"]) == 2
    assert topic["core_attempts"] == len(requested) == 4
    assert [item["reason"] for item in topic["exclusions"]] == [
        "core_license_not_cc_by", "abstract_missing_or_too_short"]
    assert result["request_attempts"] == 5  # one metadata plus four bounded core requests
    assert result["request_failures"] == 0
    assert result["diagnostic_only"] is True  # a single topic never meets the frozen cohort minimum


def test_topic_capture_audits_metadata_hits_rows_and_selected_core_candidates():
    metadata = [lite_row(index) for index in range(12)]
    by_pmcid = {item["pmcid"]: core_record(item) for item in metadata}

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        if params.get("resulttype") == ["lite"]:
            return Response({"hitCount": 87, "resultList": {"result": metadata[:6]}})
        pmcid = params["query"][0].removeprefix("PMCID:")
        return Response({"hitCount": 1, "resultList": {"result": [by_pmcid[pmcid]]}})

    result = capture_topic_stratified_source_pool(
        topic_specs=[topic_spec("metadata-audit")], capture_date="2026-10-16",
        captured_at="2026-10-16T12:00:00Z", opener=opener)
    topic = result["topics"][0]
    assert topic["metadata_hit_count"] == 87
    assert topic["metadata_rows_returned"] == 6
    assert topic["core_candidate_slots_selected"] == 4
    assert topic["core_attempts"] == 2


def test_topic_capture_records_zero_metadata_hits_separately_from_core_exclusions():
    result = capture_topic_stratified_source_pool(
        topic_specs=[topic_spec("zero-hits", slot_kind="conflict")],
        capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z",
        opener=lambda request, timeout: Response({"hitCount": 0,
            "resultList": {"result": []}}))
    topic = result["topics"][0]
    assert topic["metadata_hit_count"] == 0
    assert topic["metadata_rows_returned"] == 0
    assert topic["core_candidate_slots_selected"] == 0
    assert topic["core_attempts"] == 0
    assert topic["exclusions"] == []


def test_topic_capture_rejects_pico_mismatch_then_refills_in_europe_pmc_rank_order_within_cap():
    metadata = [lite_row(index) for index in range(64)]
    selected = select_topic_candidates(metadata, topic_id="group-a", max_candidates=4)
    by_pmcid = {item["pmcid"]: core_record(item) for item in metadata}
    for candidate in selected[:2]:
        by_pmcid[candidate["pmcid"]]["abstractText"] = (
            "condition alpha " + "unrelated material " * 100)
    requested = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        if params.get("resulttype") == ["lite"]:
            return Response({"hitCount": len(metadata), "resultList": {"result": metadata}})
        pmcid = params["query"][0].removeprefix("PMCID:")
        requested.append(pmcid)
        return Response({"hitCount": 1, "resultList": {"result": [by_pmcid[pmcid]]}})

    result = capture_topic_stratified_source_pool(
        topic_specs=[topic_spec("group-a")], capture_date="2026-10-16",
        captured_at="2026-10-16T12:00:00Z", opener=opener)
    topic = result["topics"][0]
    assert topic["status"] == "complete"
    assert len(topic["records"]) == 2
    assert topic["core_attempts"] == len(requested) == 4
    assert [row["metadata_rank"] for row in topic["candidate_audit"]] == [0, 1, 2, 3]
    assert all(row["metadata_response_sha256"] == topic["metadata_response_sha256"]
        for row in topic["candidate_audit"])
    assert [row["reason"] for row in topic["exclusions"]] == [
        "pico_relevance_missing_dimensions", "pico_relevance_missing_dimensions"]
    assert all(row["abstract_pico_relevance"]["eligible"]
        for row in topic["candidate_audit"][2:])
    assert result["core_request_budget_max"] == 252
    assert result["candidate_selection_policy"]["version"] == "legacy-lite-metadata-candidate-helper"


@pytest.mark.parametrize("identity_field", ["doi", "pmcid"])
def test_topic_capture_stops_and_records_identity_mismatch_before_fetching_next_topic(identity_field):
    metadata = [lite_row(index) for index in range(64)]
    first = select_topic_candidates(metadata, topic_id="group-a", max_candidates=4)[0]
    by_pmcid = {item["pmcid"]: core_record(item) for item in metadata}
    by_pmcid[first["pmcid"]][identity_field] = (
        "PMC999999" if identity_field == "pmcid" else "10.5555/identity-mismatch")
    requested = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        if params.get("resulttype") == ["lite"]:
            return Response({"hitCount": len(metadata), "resultList": {"result": metadata}})
        requested.append(params["query"][0])
        pmcid = params["query"][0].removeprefix("PMCID:")
        return Response({"hitCount": 1, "resultList": {"result": [by_pmcid[pmcid]]}})

    result = capture_topic_stratified_source_pool(
        topic_specs=[topic_spec("group-a"), topic_spec("group-b")],
        capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert result["status"] == "failed_identity_mismatch"
    assert result["request_attempts"] == 2
    assert result["request_failures"] == 1
    assert len(result["request_log"]) == 2
    assert len(requested) == 1  # fail closed; group-b was not queried


def test_topic_capture_rejects_more_than_sixty_slots_before_any_request():
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})
    with pytest.raises(ValueError, match="60 topic metadata requests"):
        capture_topic_stratified_source_pool(
            topic_specs=[topic_spec(f"slot-{index}") for index in range(61)],
            capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []


def test_topic_capture_rejects_core_budget_over_252_before_any_request():
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})
    specs = [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(32)]
    with pytest.raises(ValueError, match="252-request maximum"):
        capture_topic_stratified_source_pool(topic_specs=specs,
            capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []


def test_frozen_production_plan_requires_exact_39_9_12_allocation_before_network():
    plan = ([topic_spec(f"regular-{index}") for index in range(39)] +
        [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(9)] +
        [topic_spec(f"replacement-{index}", slot_kind="replacement") for index in range(12)])
    manifest = build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16")
    assert manifest["schema"] == "g09-frozen-topic-plan-v5"
    assert manifest["candidate_selection_policy"]["version"] == "europe-pmc-core-search-pico-pages-v5"
    assert manifest["candidate_selection_policy"]["query_scope"] == (
        "condition and intervention are field-scoped to TITLE or ABSTRACT for recall; population and outcome are checked on returned core abstracts")
    assert manifest["candidate_selection_policy"]["core_page_size"] == 10
    assert manifest["candidate_selection_policy"]["core_page_caps_by_slot_kind"] == {
        "regular": 5, "conflict": 2, "replacement": 3}
    assert manifest["candidate_selection_policy"]["core_request_reserve"] == 3
    validated = validate_frozen_topic_plan(manifest, expected_plan_sha256=manifest["plan_sha256"])
    assert validated["slot_counts"] == {"regular": 39, "conflict": 9, "replacement": 12}
    assert validated["topic_searches"] == 60
    assert validated["core_request_budget"] == 249
    assert validated["core_request_reserve"] == 3
    assert validated["topic_specs"] == plan
    assert build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16") == manifest

    changed_content = [dict(spec) for spec in plan]
    changed_content[0]["condition_terms"] = ["different condition"]
    changed_manifest = build_frozen_topic_plan_manifest(changed_content, capture_date="2026-10-16")
    assert changed_manifest["plan_sha256"] != manifest["plan_sha256"]
    with pytest.raises(ValueError, match="verified plan hash"):
        validate_frozen_topic_plan(changed_manifest,
            expected_plan_sha256=manifest["plan_sha256"])

    policy_changed = dict(manifest)
    policy_changed["candidate_selection_policy"] = {
        **manifest["candidate_selection_policy"], "core_eligibility": "trust query match only"}
    with pytest.raises(ValueError, match="candidate-selection policy"):
        validate_frozen_topic_plan(policy_changed,
            expected_plan_sha256=manifest["plan_sha256"])

    duplicate_id = [dict(spec) for spec in plan]
    duplicate_id[1]["topic_id"] = duplicate_id[0]["topic_id"]
    with pytest.raises(ValueError, match="unique"):
        build_frozen_topic_plan_manifest(duplicate_id, capture_date="2026-10-16")

    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})
    invalid_plan = [topic_spec(f"regular-{index}") for index in range(60)]
    with pytest.raises(ValueError, match="39 regular, 9 conflict and 12 replacement"):
        invalid_manifest = build_frozen_topic_plan_manifest(invalid_plan, capture_date="2026-10-16")
        capture_frozen_topic_stratified_source_pool(plan_manifest=invalid_manifest,
            verified_plan_sha256=invalid_manifest["plan_sha256"],
            captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []


def test_production_capture_rejects_manifest_hash_mismatch_before_network():
    plan = ([topic_spec(f"regular-{index}") for index in range(39)] +
        [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(9)] +
        [topic_spec(f"replacement-{index}", slot_kind="replacement") for index in range(12)])
    manifest = build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16")
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})
    with pytest.raises(ValueError, match="verified plan hash"):
        capture_frozen_topic_stratified_source_pool(plan_manifest=manifest,
            verified_plan_sha256="0" * 64, captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []


def test_production_capture_rejects_frozen_v1_plan_before_network():
    plan = ([topic_spec(f"regular-{index}") for index in range(39)] +
        [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(9)] +
        [topic_spec(f"replacement-{index}", slot_kind="replacement") for index in range(12)])
    manifest = build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16")
    legacy_v1 = {**manifest, "schema": "g09-frozen-topic-plan-v1"}
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})
    with pytest.raises(ValueError, match="unsupported schema"):
        capture_frozen_topic_stratified_source_pool(plan_manifest=legacy_v1,
            verified_plan_sha256=manifest["plan_sha256"],
            captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []


def test_production_capture_rejects_frozen_v2_plan_before_network():
    plan = ([topic_spec(f"regular-{index}") for index in range(39)] +
        [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(9)] +
        [topic_spec(f"replacement-{index}", slot_kind="replacement") for index in range(12)])
    manifest = build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16")
    legacy_v2 = {**manifest, "schema": "g09-frozen-topic-plan-v2"}
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})
    with pytest.raises(ValueError, match="unsupported schema"):
        capture_frozen_topic_stratified_source_pool(plan_manifest=legacy_v2,
            verified_plan_sha256=manifest["plan_sha256"],
            captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []


def test_production_capture_rejects_frozen_v3_plan_before_network():
    plan = ([topic_spec(f"regular-{index}") for index in range(39)] +
        [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(9)] +
        [topic_spec(f"replacement-{index}", slot_kind="replacement") for index in range(12)])
    manifest = build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16")
    legacy_v3 = {**manifest, "schema": "g09-frozen-topic-plan-v3"}
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})
    with pytest.raises(ValueError, match="unsupported schema"):
        capture_frozen_topic_stratified_source_pool(plan_manifest=legacy_v3,
            verified_plan_sha256=manifest["plan_sha256"],
            captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []


def test_production_capture_rejects_frozen_v4_plan_before_network():
    plan = ([topic_spec(f"regular-{index}") for index in range(39)] +
        [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(9)] +
        [topic_spec(f"replacement-{index}", slot_kind="replacement") for index in range(12)])
    manifest = build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16")
    legacy_v4 = {**manifest, "schema": "g09-frozen-topic-plan-v4"}
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})
    with pytest.raises(ValueError, match="unsupported schema"):
        capture_frozen_topic_stratified_source_pool(plan_manifest=legacy_v4,
            verified_plan_sha256=manifest["plan_sha256"],
            captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []


def test_core_search_pages_apply_abstract_gate_before_selection_and_preserve_cursor_audit():
    first_row = core_record(lite_row(0))
    first_row["abstractText"] = "condition alpha treatment alpha " + "unrelated evidence " * 100
    second_page = [core_record(lite_row(20)), core_record(lite_row(21))]
    calls = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        calls.append(params)
        assert params["resulttype"] == ["core"]
        assert params["pageSize"] == ["10"]
        if params["cursorMark"] == ["*"]:
            return Response({"hitCount": 20, "nextCursorMark": "cursor-1",
                "resultList": {"result": [first_row]}})
        assert params["cursorMark"] == ["cursor-1"]
        return Response({"hitCount": 20, "nextCursorMark": "cursor-2",
            "resultList": {"result": second_page}})

    result = source_module.capture_topic_core_search_pool(topic_specs=[topic_spec("core-page")],
        capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z", opener=opener)
    topic = result["topics"][0]
    assert topic["status"] == "complete"
    assert [record["pmcid"] for record in topic["records"]] == ["PMC800020", "PMC800021"]
    assert topic["core_attempts"] == 2
    assert [entry["page_index"] for entry in topic["core_pages"]] == [1, 2]
    assert [entry["cursor_mark"] for entry in topic["core_pages"]] == ["*", "cursor-1"]
    assert [entry["record_offset"] for entry in topic["core_pages"]] == [0, 1]
    assert [row["core_rank"] for row in topic["candidate_audit"]] == [0, 1, 2]
    assert [row["core_rank"] for row in topic["candidate_audit"]
        if row["abstract_pico_relevance"]["eligible"]] == [1, 2]
    assert all(len(entry["response_sha256"]) == 64 for entry in topic["core_pages"])
    assert len(topic["exclusions"]) == 1
    assert all(entry["reason"] == "pico_relevance_missing_dimensions"
        for entry in topic["exclusions"])
    assert result["request_attempts"] == len(calls) == 2
    assert result["request_failures"] == 0
    assert result["core_request_budget_max"] == 252
    assert result["core_request_budget_reserved"] == 3


def test_core_search_rejects_repeated_cursor_without_retry_or_page_overrun():
    calls = []

    def opener(request, timeout):
        calls.append(parse_qs(urlparse(request.full_url).query))
        rows = [core_record(lite_row(index)) for index in range(10)]
        for row in rows:
            row["abstractText"] = "condition alpha treatment alpha " + "unrelated evidence " * 100
        return Response({"hitCount": 100, "nextCursorMark": "*",
            "resultList": {"result": rows}})

    result = source_module.capture_topic_core_search_pool(
        topic_specs=[topic_spec("repeated-cursor", slot_kind="conflict")],
        capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z", opener=opener)
    topic = result["topics"][0]
    assert len(calls) == topic["core_attempts"] == 1
    assert topic["failure_reason"] == "core_search_cursor_missing_or_repeated"
    assert topic["status"] == "incomplete"
    assert result["request_attempts"] == 1  # no blind retry


def test_core_search_fails_closed_if_server_exceeds_frozen_page_size():
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        rows = [core_record(lite_row(index)) for index in range(11)]
        return Response({"hitCount": 11, "nextCursorMark": "next",
            "resultList": {"result": rows}})

    result = source_module.capture_topic_core_search_pool(
        topic_specs=[topic_spec("oversize", slot_kind="conflict")],
        capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z", opener=opener)
    topic = result["topics"][0]
    assert len(calls) == topic["core_attempts"] == 1
    assert topic["failure_reason"] == "core_page_exceeds_frozen_page_size"
    assert topic["records"] == []
    assert result["request_failures"] == 1


def test_frozen_capture_binds_stable_source_implementation_sha256_without_network():
    plan = ([topic_spec(f"regular-{index}") for index in range(39)] +
        [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(9)] +
        [topic_spec(f"replacement-{index}", slot_kind="replacement") for index in range(12)])
    manifest = build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16")
    module_path = Path(source_module.__file__)
    expected_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    requested = []

    def mock_opener(request, timeout):
        requested.append(request.full_url)
        return Response({"hitCount": 0, "resultList": {"result": []}})

    first = capture_frozen_topic_stratified_source_pool(
        plan_manifest=manifest, verified_plan_sha256=manifest["plan_sha256"],
        captured_at="2026-10-16T12:00:00Z", opener=mock_opener)
    second = capture_frozen_topic_stratified_source_pool(
        plan_manifest=manifest, verified_plan_sha256=manifest["plan_sha256"],
        captured_at="2026-10-16T12:00:00Z", opener=mock_opener)
    assert first["capture_implementation_sha256"] == expected_sha256
    core_urls = [url for url in requested if "resulttype=core" in url.casefold()]
    assert len(core_urls) == 120
    assert all(parse_qs(urlparse(url).query).get("pageSize") == ["10"]
        for url in core_urls)
    assert all(parse_qs(urlparse(url).query).get("cursorMark") == ["*"]
        for url in core_urls)
    assert second["capture_implementation_sha256"] == expected_sha256
    assert first["capture_sha256"] == second["capture_sha256"]
    assert first["capture_sha256"] == sha256_obj(
        {key: value for key, value in first.items() if key != "capture_sha256"})
    assert len(requested) == 120  # mock-only first core pages; no live requests


def test_frozen_core_page_capture_never_exceeds_request_plan_or_retries_failures():
    plan = ([topic_spec(f"regular-{index}") for index in range(39)] +
        [topic_spec(f"conflict-{index}", slot_kind="conflict") for index in range(9)] +
        [topic_spec(f"replacement-{index}", slot_kind="replacement") for index in range(12)])
    manifest = build_frozen_topic_plan_manifest(plan, capture_date="2026-10-16")
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        raise OSError("one-shot page failure")
    result = capture_frozen_topic_stratified_source_pool(plan_manifest=manifest,
        verified_plan_sha256=manifest["plan_sha256"],
        captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert len(requests) == 60  # one initial page per topic; each failure stops without retry
    assert result["request_failures"] == 60
    assert result["core_request_budget_planned"] == 249
    assert result["core_request_budget_reserved"] == 3
    assert result["request_attempts"] <= 252
    assert all(topic["core_attempts"] == 1 for topic in result["topics"])


def test_topic_capture_filters_cross_group_duplicates_before_candidate_cap_and_refills():
    metadata = [lite_row(index) for index in range(64)]
    by_pmcid = {item["pmcid"]: core_record(item) for item in metadata}
    requested = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        if params.get("resulttype") == ["lite"]:
            return Response({"hitCount": len(metadata), "resultList": {"result": metadata}})
        pmcid = params["query"][0].removeprefix("PMCID:")
        requested.append(pmcid)
        return Response({"hitCount": 1, "resultList": {"result": [by_pmcid[pmcid]]}})

    result = capture_topic_stratified_source_pool(topic_specs=[topic_spec("group-a"),
            topic_spec("group-b")], capture_date="2026-10-16",
        captured_at="2026-10-16T12:00:00Z", opener=opener)
    first, second = result["topics"]
    assert first["status"] == second["status"] == "complete"
    first_ids = {row["pmcid"] for row in first["records"]}
    second_ids = {row["pmcid"] for row in second["records"]}
    assert len(first_ids) == len(second_ids) == 2
    assert first_ids.isdisjoint(second_ids)
    assert len(requested) == 4
    assert [row["metadata_rank"] for row in second["candidate_audit"]] == [2, 3]
    assert sum(item["reason"] == "duplicate_source_across_topic_groups"
        and item["stage"] == "preselection" for item in second["exclusions"]) >= 2
    assert result["core_request_budget_planned"] == 8
    assert result["core_request_budget_max"] == 252


def test_topic_capture_preflights_topic_label_before_network_and_records_core_cap():
    requests = []
    def opener(request, timeout):
        requests.append(request.full_url)
        return Response({})

    invalid = topic_spec("group-a")
    invalid["topic"] = "   "
    with pytest.raises(ValueError, match="topic must be"):
        capture_topic_stratified_source_pool(topic_specs=[invalid],
            capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z", opener=opener)
    assert requests == []

    conflict = topic_spec("group-conflict", slot_kind="conflict")
    result = capture_topic_stratified_source_pool(topic_specs=[conflict],
        capture_date="2026-10-16", captured_at="2026-10-16T12:00:00Z",
        opener=lambda request, timeout: Response({"resultList": {"result": []}}))
    assert result["topics"][0]["core_request_cap"] == 8


def test_lite_selection_is_deterministic_diverse_and_excludes_full_plos_family():
    rows = [lite_row(i) for i in range(36)] + [
        lite_row(100, doi="10.1371/journal.pone.1234567"),
        lite_row(101, doi="10.1371/journal.pone.0302385"),
    ]
    selected = select_lite_candidates(rows, target_records=24, max_per_journal=4)
    assert selected == select_lite_candidates(rows, target_records=24, max_per_journal=4)
    assert len(selected) == 24
    assert len({row["journal"] for row in selected}) >= 12
    assert all(not row["doi"].startswith("10.1371/") for row in selected)


def test_capture_fetches_only_selected_non_plos_core_records_and_binds_raw_hashes():
    metadata = [lite_row(i) for i in range(64)] + [
        lite_row(100, doi="10.1371/journal.pone.0302385"),
        lite_row(101, doi="10.1371/journal.pone.7777777"),
    ]
    by_pmcid = {row["pmcid"]: core_record(row) for row in metadata if not row["doi"].startswith("10.1371/")}
    requested_core = []

    def opener(request, timeout):
        parsed = urlparse(request.full_url)
        params = parse_qs(parsed.query)
        if params.get("resulttype") == ["lite"]:
            assert "NOT DOI:10.1371*" in params["query"][0]
            return Response({"hitCount": len(metadata), "resultList": {"result": metadata}})
        assert params.get("resulttype") == ["core"]
        pmcid = params["query"][0].removeprefix("PMCID:")
        requested_core.append(pmcid)
        return Response({"hitCount": 1, "resultList": {"result": [by_pmcid[pmcid]]}})

    capture = capture_europe_pmc_source_pool(capture_date="2026-10-16",
        captured_at="2026-10-16T12:00:00Z", target_records=24, opener=opener)
    assert len(capture["records"]) == 24
    assert len(requested_core) == 24
    assert all(not record["doi"].startswith("10.1371/") for record in capture["records"])
    assert all(len(item["response_sha256"]) == 64 for item in capture["core_record_sha256"])
    assert len(capture["search_response_sha256"]) == 64
    assert capture["capture_sha256"]
    assert capture["historically_exposed_doi_identities"] == [
        "10.1371/journal.pone.0302385", "10.1371/journal.pone.0297184",
        "10.1371/journal.pone.0000332"]


def test_capture_skips_ineligible_core_records_and_refills_target_with_counters():
    metadata = [lite_row(i) for i in range(64)]
    selected = select_lite_candidates(metadata, target_records=48)
    by_pmcid = {item["pmcid"]: core_record(item) for item in metadata}
    by_pmcid[selected[0]["pmcid"]]["license"] = "cc by-nc"
    by_pmcid[selected[1]["pmcid"]]["abstractText"] = "Too short."
    requested_core = []

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        if params.get("resulttype") == ["lite"]:
            return Response({"hitCount": len(metadata), "resultList": {"result": metadata}})
        pmcid = params["query"][0].removeprefix("PMCID:")
        requested_core.append(pmcid)
        return Response({"hitCount": 1, "resultList": {"result": [by_pmcid[pmcid]]}})

    capture = capture_europe_pmc_source_pool(capture_date="2026-10-16",
        captured_at="2026-10-16T12:00:00Z", target_records=24, opener=opener)
    assert len(capture["records"]) == 24
    assert capture["core_attempts"] == len(requested_core) == 26
    assert [item["reason"] for item in capture["core_record_exclusions"]] == [
        "core_license_not_cc_by", "abstract_missing_or_too_short"]
    assert len(capture["core_record_exclusions"]) == 2
    assert all(record["license"] == "cc by" for record in capture["records"])


@pytest.mark.parametrize("identity_field", ["doi", "pmcid"])
def test_capture_fails_closed_on_core_identity_mismatch(identity_field):
    metadata = [lite_row(i) for i in range(64)]
    selected = select_lite_candidates(metadata, target_records=48)
    by_pmcid = {item["pmcid"]: core_record(item) for item in metadata}
    first = selected[0]
    by_pmcid[first["pmcid"]][identity_field] = (
        "PMC999999" if identity_field == "pmcid" else "10.5555/not-the-candidate")

    def opener(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        if params.get("resulttype") == ["lite"]:
            return Response({"hitCount": len(metadata), "resultList": {"result": metadata}})
        pmcid = params["query"][0].removeprefix("PMCID:")
        return Response({"hitCount": 1, "resultList": {"result": [by_pmcid[pmcid]]}})

    message = "core DOI differs" if identity_field == "doi" else "core PMCID differs"
    with pytest.raises(ValueError, match=message):
        capture_europe_pmc_source_pool(capture_date="2026-10-16",
            captured_at="2026-10-16T12:00:00Z", target_records=24, opener=opener)
