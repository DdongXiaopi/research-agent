"""Europe PMC records participate in the existing source/evidence contract."""
import json

import pytest

from src.retrieval import literature


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def iter_content(self, chunk_size):
        yield self.payload

    def close(self):
        pass


class Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.payload)


def record(doi="10.1234/example"):
    return {"id": "12345", "source": "MED", "doi": doi, "title": "Example study",
            "pubYear": "2024", "abstractText": "<p>Two controls were used.</p>",
            "authorList": {"author": [{"fullName": "Jane Doe"}]},
            "journalInfo": {"journal": {"title": "Example Journal"}}}


def test_discovery_maps_and_preserves_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(literature, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    session = Session({"resultList": {"result": [record(), record(None)]}})
    plan = literature.build_plan("discover", query="example study", providers=["europe_pmc"], retries=0)
    snapshot = literature.execute(plan, session=session)
    assert snapshot["physical_requests"] == 1
    assert snapshot["status"] == "completed"
    assert session.calls[0][1]["params"]["resultType"] == "core"
    assert snapshot["records"][0]["doi"] == "10.1234/example"
    assert snapshot["records"][0]["authors"] == ["Jane Doe"]
    assert snapshot["records"][1]["url"] == "https://europepmc.org/article/MED/12345"
    evidence = literature.evidence_from_snapshot(snapshot)
    assert evidence[0]["text"] == "Two controls were used."
    assert evidence[0]["locator"] == "abstract (europe_pmc)"
    snapshot["records"][0]["abstract"] = "Changed."
    with pytest.raises(ValueError, match="identity changed"):
        literature.evidence_from_snapshot(snapshot)


def test_doi_lookup_rejects_different_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(literature, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    session = Session({"resultList": {"result": [record("10.1234/other")]}})
    plan = literature.build_plan("lookup", doi="10.1234/example", providers=["europe_pmc"], retries=0)
    snapshot = literature.execute(plan, session=session)
    assert session.calls[0][1]["params"]["query"] == 'DOI:"10.1234/example"'
    assert snapshot["records"] == []
    assert snapshot["events"][0]["status"] == "malformed_or_oversized_response"


def test_pubmed_subset_has_explicit_filter_and_checks_returned_source(tmp_path, monkeypatch):
    monkeypatch.setattr(literature, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    plan = literature.build_plan("discover", query="cancer OR diabetes", providers=["pubmed"], retries=0)
    session = Session({"resultList": {"result": [record()]}})
    snapshot = literature.execute(plan, session=session)
    assert session.calls[0][1]["params"]["query"] == "(cancer OR diabetes) AND SRC:MED"
    assert snapshot["records"][0]["pmid"] == "12345"
    assert literature.evidence_from_snapshot(snapshot)[0]["text"] == "Two controls were used."
    other = {**record(), "source": "PPR"}
    snapshot = literature.execute(plan, session=Session({"resultList": {"result": [other]}}))
    assert snapshot["records"] == []
    assert snapshot["events"][0]["status"] == "malformed_or_oversized_response"


def test_pubmed_and_europe_pmc_without_doi_deduplicate_by_pmid(tmp_path, monkeypatch):
    monkeypatch.setattr(literature, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    body = {"resultList": {"result": [record(None)]}}
    sessions = iter([Response(body), Response(body)])
    session = type("TwoRequests", (), {"get": lambda self, *args, **kwargs: next(sessions)})()
    plan = literature.build_plan("discover", query="example", providers=["pubmed", "europe_pmc"], retries=0)
    snapshot = literature.execute(plan, session=session)
    assert len(snapshot["records"]) == 2
    assert len(literature.evidence_from_snapshot(snapshot)) == 1
