"""Bounded public-literature discovery and metadata hydration.

Adapts the provider/record/error separation in locked paper-qa clients without
importing its RAG/lmi runtime. Metadata and OA links are never full-text evidence.
This gateway is separate from frozen task-input and MCQ evaluation protocols.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import requests

from ..eval.run_manifest import source_tree_fingerprint
from ..util import artifacts_path, now_iso, read_json, sha256_obj, write_json

PROVIDERS = ("crossref", "openalex", "semantic_scholar", "europe_pmc", "pubmed", "unpaywall")
BASES = {"crossref": "https://api.crossref.org/works",
         "openalex": "https://api.openalex.org/works",
         "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper",
         "europe_pmc": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
         "pubmed": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
         "unpaywall": "https://api.unpaywall.org/v2"}
_DOI = re.compile(r"^10\.\d{4,9}/[^\s<>\x00-\x1f]+$", re.I)


def normalize_doi(value):
    if not isinstance(value, str):
        raise ValueError("DOI must be text")
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)
    if len(value) > 300 or not _DOI.fullmatch(value):
        raise ValueError("invalid DOI")
    return value.casefold()


def _plain(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", str(value or "")))).strip()


def _list_title(value):
    return _plain(value[0] if isinstance(value, list) and value else value)


def _abstract_index(value):
    if not isinstance(value, dict):
        return ""
    positions = {}
    for word, indices in value.items():
        if not isinstance(word, str) or not isinstance(indices, list):
            raise ValueError("malformed abstract index")
        for position in indices:
            if type(position) is not int or not 0 <= position < 50000 or position in positions:
                raise ValueError("malformed abstract position")
            positions[position] = word
    if positions and set(positions) != set(range(max(positions) + 1)):
        raise ValueError("incomplete abstract index")
    return " ".join(positions[i] for i in sorted(positions))


def _record(provider, raw, retrieved_at):
    if not isinstance(raw, dict):
        raise ValueError("metadata record must be an object")
    if provider == "crossref":
        parts = (raw.get("published") or raw.get("issued") or {}).get("date-parts") or [[]]
        doi, title, year = raw.get("DOI"), _list_title(raw.get("title")), parts[0][0] if parts[0] else None
        authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in raw.get("author", [])]
        abstract, venue = _plain(raw.get("abstract")), _list_title(raw.get("container-title"))
        oa_url, oa_license = "", ""
    elif provider == "openalex":
        doi, title, year = raw.get("doi"), _plain(raw.get("title")), raw.get("publication_year")
        authors = [(a.get("author") or {}).get("display_name", "") for a in raw.get("authorships", [])]
        abstract = _abstract_index(raw.get("abstract_inverted_index"))
        location = raw.get("best_oa_location") or {}
        venue = ((raw.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        oa_url, oa_license = location.get("pdf_url") or location.get("landing_page_url") or "", location.get("license") or ""
    elif provider == "semantic_scholar":
        doi, title, year = (raw.get("externalIds") or {}).get("DOI"), _plain(raw.get("title")), raw.get("year")
        authors = [a.get("name", "") for a in raw.get("authors", [])]
        abstract, venue = _plain(raw.get("abstract")), raw.get("venue") or ""
        oa_url, oa_license = (raw.get("openAccessPdf") or {}).get("url") or "", ""
    elif provider in {"europe_pmc", "pubmed"}:
        doi, title = raw.get("doi"), _plain(raw.get("title"))
        year_text = raw.get("pubYear")
        year = int(year_text) if str(year_text or "").isdigit() else None
        author_list = (raw.get("authorList") or {}).get("author", [])
        authors = [a.get("fullName", "") for a in author_list]
        abstract = _plain(raw.get("abstractText"))
        venue = (raw.get("journalInfo") or {}).get("journal", {}).get("title", "")
        oa_url, oa_license = "", ""
    else:
        doi, title, year = raw.get("doi"), _plain(raw.get("title")), raw.get("year")
        authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in raw.get("z_authors") or []]
        abstract, venue = "", raw.get("journal_name") or ""
        location = raw.get("best_oa_location") or {}
        oa_url, oa_license = location.get("url_for_pdf") or location.get("url") or "", location.get("license") or ""
    doi = normalize_doi(doi) if doi else ""
    if not title:
        raise ValueError("metadata lacks title")
    identity = doi or _plain(raw.get("id") or raw.get("paperId") or title)
    if provider in {"europe_pmc", "pubmed"} and not doi:
        identity = _plain(raw.get("source")) + ":" + identity
    url = "https://doi.org/" + doi if doi else raw.get("url") or raw.get("id") or ""
    if provider in {"europe_pmc", "pubmed"} and not doi:
        source, identifier = str(raw.get("source") or ""), str(raw.get("id") or "")
        url = ("https://europepmc.org/article/" + quote(source, safe="") + "/" + quote(identifier, safe="")) if source and identifier else ""
    record = {"source_id": "lit-" + sha256_obj([provider, identity])[:20], "provider": provider,
        "doi": doi, "title": title, "year": year, "authors": [_plain(a) for a in authors if a],
        "venue": _plain(venue), "abstract": abstract, "url": url,
        "oa_url": oa_url, "oa_license": oa_license, "retrieved_at": retrieved_at,
        "raw_metadata_sha256": sha256_obj(raw), "content_kind": "abstract" if abstract else "metadata",
        "full_text_downloaded": False, "license_scope": "metadata/OA-link; no full-text rights inferred"}
    if provider in {"europe_pmc", "pubmed"}:
        record["pmid"] = str(raw.get("pmid") or (raw.get("id") if raw.get("source") == "MED" else "") or "")
        record["pmcid"] = str(raw.get("pmcid") or "")
    return record


def build_plan(operation, *, query="", doi="", providers=None, limit=5, max_requests=8,
               retries=1, timeout=15, max_response_bytes=2_000_000):
    if operation not in {"discover", "lookup"}:
        raise ValueError("unknown literature operation")
    # PubMed is a subset of Europe PMC; offer it explicitly without duplicating
    # the same abstracts in the default search and spending another request.
    defaults = ("crossref", "openalex", "semantic_scholar", "europe_pmc")
    providers = list(providers or (defaults if operation == "discover" else (*defaults, "unpaywall")))
    if (not providers or len(providers) != len(set(providers)) or set(providers) - set(PROVIDERS)
            or (operation == "discover" and "unpaywall" in providers)):
        raise ValueError("unsupported/duplicate provider; Unpaywall is DOI lookup only")
    if operation == "discover":
        if not isinstance(query, str) or not query.strip() or len(query) > 2000 or doi:
            raise ValueError("discovery requires bounded query, without DOI")
        query = query.strip()
    else:
        doi = normalize_doi(doi)
        if query:
            raise ValueError("lookup accepts DOI only")
    for key, value, low, high in (("limit", limit, 1, 20), ("max_requests", max_requests, 1, 20),
            ("retries", retries, 0, 2), ("timeout", timeout, 1, 30),
            ("max_response_bytes", max_response_bytes, 1000, 5_000_000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"invalid literature {key} budget")
    body = {"schema": "literature-plan-v1", "operation": operation, "query": query, "doi": doi,
        "providers": providers, "limit": limit, "max_requests": max_requests, "retries": retries,
        "timeout": timeout, "max_response_bytes": max_response_bytes,
        "unpaywall_configured": bool(os.environ.get("UNPAYWALL_EMAIL")),
        "source_tree_sha256": source_tree_fingerprint()["sha256"],
        "model_calls": 0, "final_holdout_read": False, "full_text_downloaded": False}
    return {**body, "plan_sha256": sha256_obj(body)}


def _endpoint(provider, plan):
    doi = quote(plan["doi"], safe="")
    search = plan["operation"] == "discover"
    if provider == "crossref":
        return BASES[provider] + ("" if search else "/" + doi), {"query.bibliographic": plan["query"], "rows": plan["limit"]} if search else {}
    if provider == "openalex":
        return BASES[provider] + ("" if search else "/" + quote("https://doi.org/" + plan["doi"], safe="")), {"search": plan["query"], "per_page": plan["limit"]} if search else {}
    if provider == "semantic_scholar":
        fields = "title,year,authors,abstract,venue,externalIds,openAccessPdf,url"
        return BASES[provider] + ("/search" if search else "/DOI:" + doi), {"fields": fields, **({"query": plan["query"], "limit": plan["limit"]} if search else {})}
    if provider in {"europe_pmc", "pubmed"}:
        escaped_doi = plan["doi"].replace("\\", "\\\\").replace('"', '\\"')
        query = plan["query"] if search else 'DOI:"' + escaped_doi + '"'
        if provider == "pubmed":
            query = "(" + query + ") AND SRC:MED"
        return BASES[provider], {"query": query,
                                 "format": "json", "resultType": "core", "pageSize": plan["limit"]}
    return BASES[provider] + "/" + doi, {"email": os.environ["UNPAYWALL_EMAIL"]}


def execute(plan, *, session=None):
    expected = build_plan(plan["operation"], **{k: plan[k] for k in (
        "query", "doi", "providers", "limit", "max_requests", "retries", "timeout", "max_response_bytes")})
    if expected != plan:
        raise ValueError("literature execution inputs differ from frozen plan")
    root = artifacts_path("research_tools", "literature-" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", {"status": "started"})
    own_session = session is None
    session = session or requests.Session()
    events, records, physical = [], [], 0
    try:
        for provider in plan["providers"]:
            if provider == "unpaywall" and not plan["unpaywall_configured"]:
                events.append({"provider": provider, "status": "missing_email", "physical_requests": 0})
                continue
            url, params = _endpoint(provider, plan)
            headers = {"User-Agent": "rsi-personal-research/1.0"}
            if provider == "openalex" and os.environ.get("OPENALEX_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["OPENALEX_API_KEY"]
            if provider == "semantic_scholar" and os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
                headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
            for attempt in range(plan["retries"] + 1):
                if physical >= plan["max_requests"]:
                    events.append({"provider": provider, "status": "request_budget_exhausted", "physical_requests": 0})
                    break
                physical += 1
                event = {"provider": provider, "attempt": attempt + 1, "status": "dispatching"}
                events.append(event)
                write_json(root / "requests.json", {"physical_requests": physical, "events": events})
                response = None
                started = time.monotonic()
                try:
                    response = session.get(url, params=params, headers=headers, timeout=plan["timeout"],
                                           stream=True, allow_redirects=False)
                    event["http_status"] = response.status_code
                    if response.status_code != 200:
                        event["status"] = "http_error"
                        if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < plan["retries"]:
                            time.sleep(min(2 ** attempt, 2))
                            continue
                        break
                    payload = bytearray()
                    for block in response.iter_content(chunk_size=65536):
                        if len(payload) + len(block) > plan["max_response_bytes"] or time.monotonic() - started > plan["timeout"]:
                            raise ValueError("response_size_or_deadline")
                        payload.extend(block)
                    raw = json.loads(payload)
                    if provider in {"europe_pmc", "pubmed"}:
                        rows = (raw.get("resultList") or {}).get("result", [])
                    elif plan["operation"] == "lookup":
                        rows = [(raw.get("message") if provider == "crossref" else raw)]
                    else:
                        rows = ((raw.get("message") or {}).get("items", []) if provider == "crossref"
                                else raw.get("results", []) if provider == "openalex" else raw.get("data", []))
                    if not isinstance(rows, list):
                        raise ValueError("invalid_result_collection")
                    if provider == "pubmed" and any(row.get("source") != "MED" for row in rows):
                        raise ValueError("provider_returned_non_pubmed_record")
                    normalized = [_record(provider, row, now_iso()) for row in rows[:plan["limit"]]]
                    if plan["operation"] == "lookup" and any(row["doi"] != plan["doi"] for row in normalized):
                        raise ValueError("provider_returned_different_doi")
                    write_json(root / f"{provider}-metadata.json", raw)
                    records.extend(normalized)
                    event.update({"status": "ok", "record_count": len(normalized)})
                    break
                except requests.RequestException:
                    event["status"] = "transport_error"
                    if attempt < plan["retries"]:
                        time.sleep(min(2 ** attempt, 2))
                        continue
                except (ValueError, TypeError, KeyError, AttributeError, IndexError):
                    event["status"] = "malformed_or_oversized_response"
                finally:
                    if response is not None:
                        response.close()
                    event["elapsed_ms"] = int((time.monotonic() - started) * 1000)
                    write_json(root / "requests.json", {"physical_requests": physical, "events": events})
                break
        unique = {row["source_id"]: row for row in records}
        result = {"schema": "literature-snapshot-v1", "plan": plan, "records": list(unique.values()),
            "events": events, "physical_requests": physical, "model_calls": 0,
            "status": "completed" if records else "no_sources", "run_dir": str(root),
            "retrieval_complete": all(e.get("status") == "ok" for e in events),
            "full_text_downloaded": False, "final_holdout_read": False}
        result["snapshot_sha256"] = sha256_obj(result)
        write_json(root / "snapshot.json", result)
        write_json(root / "status.json", {"status": result["status"], "snapshot_sha256": result["snapshot_sha256"]})
        return result
    except BaseException as exc:
        write_json(root / "status.json", {"status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                   "error_type": type(exc).__name__, "physical_requests": physical})
        raise
    finally:
        if own_session:
            session.close()


def validate_snapshot(snapshot):
    if (snapshot.get("schema") != "literature-snapshot-v1"
            or snapshot.get("snapshot_sha256") != sha256_obj({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})):
        raise ValueError("literature snapshot content identity changed")
    return snapshot


def evidence_from_snapshot(snapshot):
    if snapshot.get("schema") == "fulltext-snapshot-v1":
        from .documents import evidence
        return evidence(snapshot)
    if snapshot.get("schema") == "document-evidence-snapshot-v1":
        from .documents import selected_evidence
        return selected_evidence(snapshot)
    validate_snapshot(snapshot)
    sources, seen = [], set()
    for row in snapshot["records"]:
        if not row["abstract"]:
            continue
        identities = {"doi:" + row["doi"]} if row["doi"] else set()
        for key in ("pmid", "pmcid"):
            if row.get(key):
                identities.add(key + ":" + str(row[key]).strip().casefold())
        if not identities:
            identities.add("source:" + row["source_id"])
        duplicate = bool(identities & seen)
        seen.update(identities)
        if duplicate:
            continue
        sources.append({"source_id": row["source_id"], "text": row["abstract"], "url": row["url"],
                        "locator": "abstract (" + row["provider"] + ")", "retrieved_at": row["retrieved_at"]})
    return sources


def verify_reference(reference, snapshot):
    validate_snapshot(snapshot)
    if not isinstance(reference, dict) or set(reference) - {"doi", "title", "year", "authors"}:
        raise ValueError("unsupported reference fields")
    doi = normalize_doi(reference.get("doi"))
    if not any(k in reference for k in ("title", "year", "authors")):
        raise ValueError("reference needs at least one metadata field to verify")
    comparisons, conflicts = [], []
    def norm(value):
        if isinstance(value, str):
            return " ".join(re.findall(r"\w+", html.unescape(value).casefold()))
        return value
    for field in ("title", "year", "authors"):
        if field not in reference:
            continue
        expected = reference[field]
        if ((field == "year" and (type(expected) is not int or not 1000 <= expected <= 3000))
                or (field == "title" and (not isinstance(expected, str) or not expected.strip()))
                or (field == "authors" and (not isinstance(expected, list) or not expected
                    or any(not isinstance(a, str) or not a.strip() for a in expected)))):
            raise ValueError("invalid reference field " + field)
        rows = []
        for row in snapshot["records"]:
            if row["doi"] != doi:
                continue
            actual = row.get(field)
            expected_value = [norm(a) for a in expected] if field == "authors" else norm(expected)
            actual_value = [norm(a) for a in actual] if field == "authors" and actual else norm(actual)
            rows.append({"provider": row["provider"], "actual": actual,
                         "status": "missing" if actual in (None, "", []) else "matched" if actual_value == expected_value else "mismatch"})
        nonempty = [([norm(a) for a in r["actual"]] if field == "authors" else norm(r["actual"]))
                    for r in rows if r["status"] != "missing"]
        if len({sha256_obj(value) for value in nonempty}) > 1:
            conflicts.append(field)
        comparisons.append({"field": field, "sources": rows})
    matched = {r["provider"] for c in comparisons for r in c["sources"] if r["status"] == "matched"}
    complete = all(c["sources"] and all(r["status"] == "matched" for r in c["sources"]) for c in comparisons)
    return {"schema": "reference-verification-v1", "doi": doi, "comparisons": comparisons,
        "conflicting_fields": conflicts, "snapshot_sha256": snapshot["snapshot_sha256"],
        "status": "metadata_consistent" if complete and len(matched) >= 2 and not conflicts else "needs_review",
        "semantic_support_verified": False, "full_text_verified": False}
