"""Approval-gated, hash-bound CT.gov expansion capture (no automatic labels).

The module is deliberately transport-injectable: unit tests exercise the full
write-ahead and failure protocol without network access. A capture is one-shot;
an uncertain physical request is terminal and must never be replayed.
"""
from __future__ import annotations

import datetime as dt
import ctypes
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import clinifact_ctgov_expansion_plan as frozen
from . import clinifact_ctgov_pubmed_adapter_v1 as adapter

SCHEMA = "clinifact-ctgov-expansion-capture-v1"
APPROVAL_SCHEMA = "clinifact-ctgov-expansion-root-approval-v1"
PUBMED_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USER_AGENT = "new-project-ctgov-expansion/1.0 (bounded source metadata capture)"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return frozen.canonical(value)


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        create_file.restype = ctypes.c_void_p
        flush = kernel32.FlushFileBuffers
        flush.argtypes = [ctypes.c_void_p]
        flush.restype = ctypes.c_int
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        handle = create_file(str(path), 0x80000000 | 0x40000000, 0x1 | 0x2 | 0x4,
                             None, 3, 0x02000000, None)
        if handle in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_last_error(), "CreateFileW directory flush handle failed", str(path))
        try:
            if not flush(handle):
                raise OSError(ctypes.get_last_error(), "FlushFileBuffers directory failed", str(path))
        finally:
            close(handle)
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(path.parent)


def _replace_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise RuntimeError("stale write-ahead temp file makes capture terminal")
    _write_new(tmp, _canonical(value))
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_bound_inputs(plan: Mapping[str, Any], plan_path: Path,
                        bindings: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Validate plan and every external artifact hash before any physical request."""
    frozen.validate_plan(plan)
    try:
        plan_path.resolve().relative_to((root / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("frozen plan must stay inside project .workbuddy") from exc
    expected = plan["input_bindings"]
    scalar = ("oas", "oas_receipt", "parent_page", "parent_receipt", "exposure_inventory")
    for key in scalar:
        path = Path(bindings[key]).resolve()
        try:
            path.relative_to((root / ".workbuddy").resolve())
        except ValueError as exc:
            raise ValueError(f"{key} input must stay inside .workbuddy") from exc
    checks = {
        "oas": (expected["oas_sha256"], Path(bindings["oas"])),
        "oas_receipt": (expected["oas_receipt_sha256"], Path(bindings["oas_receipt"])),
        "parent_page": (expected["parent_page_sha256"], Path(bindings["parent_page"])),
        "parent_receipt": (expected["parent_receipt_sha256"], Path(bindings["parent_receipt"])),
        "exposure_inventory": (expected["exposure_inventory_sha256"], Path(bindings["exposure_inventory"])),
        "expansion_module": (expected["expansion_module_sha256"], Path(frozen.__file__)),
        "source_adapter": (expected["source_adapter_sha256"], Path(adapter.__file__)),
    }
    seen = bindings.get("seen_census_pages")
    if not isinstance(seen, Sequence) or len(seen) != len(expected["seen_census_pages"]):
        raise ValueError("all frozen prior census page bindings are required")
    for index, (record, expected_record) in enumerate(zip(seen, expected["seen_census_pages"])):
        page = Path(record["page"]).resolve()
        receipt = Path(record["receipt"]).resolve()
        for bound in (page, receipt):
            try:
                bound.relative_to((root / ".workbuddy").resolve())
            except ValueError as exc:
                raise ValueError("prior census inputs must stay inside .workbuddy") from exc
        if frozen.sha_file(page) != expected_record["page_sha256"]:
            raise ValueError(f"seen census page {index + 1} hash mismatch")
        if frozen.sha_file(receipt) != expected_record["receipt_sha256"]:
            raise ValueError(f"seen census receipt {index + 1} hash mismatch")
    verified = {}
    for name, (wanted, path) in checks.items():
        if name == "seen_census_pages":
            continue
        actual = frozen.sha_file(path.resolve())
        if actual != wanted:
            raise ValueError(f"{name} hash mismatch")
        verified[name] = actual
    verified["seen_census_pages"] = [dict(x) for x in expected["seen_census_pages"]]
    if frozen.sha_file(plan_path.resolve()) != _sha(_canonical(plan)):
        raise ValueError("plan file is not the canonical frozen plan")
    verified["plan_file_sha256"] = frozen.sha_file(plan_path.resolve())
    return verified


def pending_root_approval(plan: Mapping[str, Any], *, output_dir: str) -> dict[str, Any]:
    body = {
        "schema": APPROVAL_SCHEMA,
        "status": "pending_root_approval",
        "plan_sha256": plan["plan_sha256"],
        "output_dir": output_dir,
        "dispatch_authorized": False,
        "approved_by": None,
        "approved_at_utc": None,
        "approval_statement": None,
        "scope": "CT.gov expansion discovery, ResultsSection, and bounded PubMed metadata capture only; no labels, references, confirmation, candidate, or release",
    }
    return {**body, "approval_sha256": _sha(_canonical(body))}


def validate_approval(approval: Mapping[str, Any], plan: Mapping[str, Any], output_dir: Path) -> None:
    body = {k: v for k, v in approval.items() if k != "approval_sha256"}
    if (approval.get("schema") != APPROVAL_SCHEMA or approval.get("status") != "approved"
            or approval.get("dispatch_authorized") is not True
            or approval.get("plan_sha256") != plan.get("plan_sha256")
            or Path(str(approval.get("output_dir", ""))).resolve() != output_dir.resolve()
            or not approval.get("approved_by") or not approval.get("approval_statement")
            or approval.get("approval_sha256") != _sha(_canonical(body))):
        raise ValueError("a separate valid root approval bound to the exact plan/output is required")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def urllib_transport(url: str, *, timeout: int, cap: int, accept: str) -> tuple[int, dict[str, str], bytes]:
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read(cap + 1)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()) if error.headers else {}, error.read(cap + 1)


def _reserve(output: Path, index: int, request: Mapping[str, Any]) -> Path:
    record = {"schema": "clinifact-ctgov-expansion-attempt-v1", "request_index": index,
              "state": "reserved_before_dispatch", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "retries": 0, "redirects": 0, **dict(request)}
    path = output / "ledger" / f"attempt-{index:04d}.json"
    _write_new(path, _canonical(record))
    return path


def _finish_attempt(path: Path, record: dict[str, Any]) -> None:
    record["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _replace_json(path, record)


def _mark_unknown_best_effort(output: Path, path: Path, record: dict[str, Any], reason: str) -> None:
    marker = path.with_suffix(".outcome-unknown.json")
    payload = {"schema": "clinifact-ctgov-expansion-outcome-unknown-v1",
               "request_index": record.get("request_index"), "plan_sha256": record.get("plan_sha256"),
               "state": "outcome_unknown", "terminal": True, "reason": reason,
               "observed_status": record.get("status"), "response_sha256": record.get("response_sha256")}
    try:
        _write_new(marker, _canonical(payload))
    except Exception:
        # The main reservation remains terminal by default (reserved, never replayed).
        pass


def _persist_raw(output: Path, index: int, body: bytes, *, error: bool = False) -> Path:
    name = f"request-{index:04d}-{'error' if error else 'response'}.bin"
    path = output / "raw" / name
    _write_new(path, body)
    return path


def _parse_pubmed(raw: bytes) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(raw)
    result: dict[str, dict[str, Any]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            value = "".join(node.itertext()).strip()
            if value:
                label = node.attrib.get("Label") or node.attrib.get("NlmCategory")
                abstract_parts.append(f"{label}: {value}" if label else value)
        if pmid:
            result[pmid] = {"pmid": pmid, "returned": True, "title": title,
                            "abstract": " ".join(abstract_parts), "links": []}
    return result


def _build_unlabeled_bundle(studies: Mapping[str, Mapping[str, Any]], pubmed: Mapping[str, Mapping[str, Any]],
                            plan: Mapping[str, Any], capture_hash: str, pubmed_hash: str) -> dict[str, Any]:
    components = []
    for nct, study in sorted(studies.items()):
        refs = (((study.get("protocolSection") or {}).get("referencesModule") or {}).get("references") or [])
        for ref in refs:
            pmid = str(ref.get("pmid") or "")
            if ref.get("type") != "RESULT" or pmid not in pubmed:
                continue
            title = str(pubmed[pmid].get("title") or "").strip()
            citation = str(ref.get("citation") or "")
            title_match = bool(title and adapter.normalize_text(title) in adapter.normalize_text(citation))
            pubmed[pmid]["links"].append({"nct_id": nct, "result_reference_citation": citation,
                                             "result_reference_title_identity_match": title_match})
            components.append({"provisional_component_id": f"{nct}:{pmid}", "nct_ids": [nct], "pmids": [pmid]})
    excluded = plan["historical_identity_exclusions"]
    pool_body = {"schema": "clinifact-ctgov-expansion-provisional-components-v1", "components": components}
    pool_sha = _sha(_canonical(pool_body))
    history = adapter.make_history_exclusion_manifest(
        source_report_sha256=plan["plan_sha256"], candidate_pool_sha256=pool_sha,
        excluded_nct_ids=excluded["nct_ids"], excluded_pmids=excluded["pmids"], excluded_claims=[])
    return adapter.build_source_bundle(studies, pubmed, components,
        source_hashes={"ctgov_capture_sha256": capture_hash, "pubmed_capture_sha256": pubmed_hash,
                       "candidate_pool_sha256": pool_sha}, history_exclusion=history)


def execute_capture(*, plan: Mapping[str, Any], plan_path: Path, bindings: Mapping[str, Any],
                    output_dir: Path, approval: Mapping[str, Any], root: Path,
                    transport: Callable[..., tuple[int, Mapping[str, str], bytes]] = urllib_transport,
                    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)) -> dict[str, Any]:
    """Run the complete bounded capture after pre-request hashes and approval pass.

    The function is never called by the default CLI mode. Tests inject a fake
    transport. It cannot resume an existing directory or replay an attempt.
    """
    plan = frozen.validate_plan(plan)
    verified = verify_bound_inputs(plan, plan_path, bindings, root=root.resolve())
    validate_approval(approval, plan, output_dir)
    try:
        output_dir.resolve().relative_to((root.resolve() / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("capture output must be an exclusive directory under .workbuddy") from exc
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "raw").mkdir()
    (output_dir / "ledger").mkdir()
    for path, payload in ((output_dir / "plan.json", _canonical(plan)),
                          (output_dir / "verified-bindings.json", _canonical(verified)),
                          (output_dir / "run.json", _canonical({"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
                           "state": "ready", "dispatch_authorized": True, "labels_generated": False,
                           "confirmation_created": False}))):
        _write_new(path, payload)

    caps = plan["response_limits"]
    cap = int(caps["max_bytes_per_request"])
    timeout = int(caps["timeout_seconds"])
    counters = {"requests": 0, "bytes": {"discovery": 0, "results": 0, "pubmed": 0},
                "studies": 0, "result_records": 0, "result_edges": 0, "unique_pmids": set()}
    nct_seen: set[str] = set()
    studies: dict[str, Mapping[str, Any]] = {}
    discovery_candidates: list[dict[str, Any]] = []
    token = plan["first_request_params"].get("pageToken")
    params = dict(plan["first_request_params"])
    discovery_stop_reason = "server_exhausted"

    def request(url: str, req_params: Mapping[str, str], phase: str, accept: str) -> bytes:
        if counters["requests"] >= (plan["page_policy"]["max_pages"] +
                plan["downstream_caps"]["max_result_requests"] + plan["downstream_caps"]["max_pubmed_requests"]):
            raise ValueError("total physical request cap exceeded")
        index = counters["requests"] + 1
        full_url = url + "?" + urllib.parse.urlencode(req_params, safe=",")
        rec = {"phase": phase, "method": "GET", "url": full_url, "params": dict(req_params),
               "timeout_seconds": timeout, "response_cap_bytes": cap,
               "plan_sha256": plan["plan_sha256"]}
        attempt = _reserve(output_dir, index, rec)
        counters["requests"] = index
        reservation = _read_json(attempt)
        try:
            status, headers, body = transport(full_url, timeout=timeout, cap=cap, accept=accept)
        except Exception as exc:
            reservation.update({"state": "outcome_unknown", "error": f"{type(exc).__name__}: {exc}",
                                "terminal": True, "plan_sha256": plan["plan_sha256"]})
            try: _finish_attempt(attempt, reservation)
            except Exception: _mark_unknown_best_effort(output_dir, attempt, reservation, "transport result unknown and ledger update failed")
            raise RuntimeError("physical request outcome unknown; capture is terminal") from exc
        truncated = len(body) > cap
        raw = body[:cap] if truncated else body
        try:
            raw_path = _persist_raw(output_dir, index, raw, error=status != 200)
        except Exception as exc:
            reservation.update({"state": "outcome_unknown", "status": status,
                                "response_sha256": _sha(raw), "terminal": True,
                                "plan_sha256": plan["plan_sha256"],
                                "persistence_error": f"{type(exc).__name__}: {exc}"})
            try: _finish_attempt(attempt, reservation)
            except Exception: _mark_unknown_best_effort(output_dir, attempt, reservation, "raw response persistence failed")
            raise RuntimeError("response received but raw persistence failed; capture is terminal") from exc
        reservation.update({"state": "captured" if status == 200 and not truncated else "failed",
                            "status": status, "headers": dict(headers), "stored_bytes": len(raw),
                            "response_sha256": _sha(raw), "raw_file": str(raw_path.relative_to(output_dir)),
                            "truncated": truncated, "terminal": status != 200 or truncated})
        try:
            _finish_attempt(attempt, reservation)
        except Exception as exc:
            _mark_unknown_best_effort(output_dir, attempt, reservation,
                                      f"receipt update failed after dispatch: {type(exc).__name__}: {exc}")
            raise RuntimeError("response persisted but receipt update failed; capture is terminal") from exc
        counters["bytes"][phase] += len(raw)
        if status != 200 or truncated:
            raise RuntimeError(f"{phase} request failed (status={status}, truncated={truncated}); stopped")
        limit_key = {"discovery": "max_discovery_bytes_total", "results": "max_result_bytes_total",
                     "pubmed": "max_pubmed_bytes_total"}[phase]
        if counters["bytes"][phase] > int(caps[limit_key]):
            raise RuntimeError(f"{phase} total response byte cap exceeded")
        return raw

    try:
        for page_no in range(1, int(plan["page_policy"]["max_pages"]) + 1):
            params["pageToken"] = str(token)
            raw = request(plan["endpoint"], params, "discovery", "application/json")
            page = json.loads(raw)
            rows = page.get("studies")
            if not isinstance(rows, list) or len(rows) > int(plan["page_policy"]["page_size"]):
                raise RuntimeError("discovery page shape or page-size cap mismatch")
            page_sha = _sha(raw)
            selected = frozen.select_fresh_candidates(rows, page_number=page_no,
                response_sha256=page_sha, excluded_nct_ids=set(plan["historical_identity_exclusions"]["nct_ids"]),
                excluded_pmids=set(plan["historical_identity_exclusions"]["pmids"]),
                already_selected_nct_ids=nct_seen,
                already_selected_pmids=set(counters["unique_pmids"]),
                already_selected_result_edges=counters["result_edges"])
            for row in rows:
                nct = frozen._study_nct(row)
                if nct in nct_seen: raise RuntimeError("duplicate NCT across discovery pages")
                nct_seen.add(nct)
            counters["studies"] += len(rows)
            if counters["studies"] > plan["page_policy"]["max_studies"]:
                raise RuntimeError("study cap exceeded")
            discovery_candidates.extend(selected["candidates"])
            counters["result_edges"] += sum(len(x["result_pmids"]) for x in selected["candidates"])
            counters["unique_pmids"].update(selected["selected_result_pmids"])
            if counters["result_edges"] > plan["downstream_caps"]["max_result_reference_edges"]:
                raise RuntimeError("RESULT reference edge cap exceeded")
            if len(counters["unique_pmids"]) > plan["downstream_caps"]["max_unique_pmids"]:
                raise RuntimeError("unique PMID cap exceeded")
            token = page.get("nextPageToken")
            if token is None or token == "":
                discovery_stop_reason = "server_exhausted"
                break
            if not isinstance(token, str): raise RuntimeError("server nextPageToken is not a string")
        else:
            # Reaching the frozen page cap is the planned discovery boundary.
            # Process only the already-captured candidates downstream; never
            # dispatch an additional discovery request from the leftover token.
            discovery_stop_reason = "planned_page_cap_with_unconsumed_server_cursor"

        ncts = sorted({x["nct_id"] for x in discovery_candidates})
        refs_count = sum(len(x["result_pmids"]) for x in discovery_candidates)
        if refs_count > plan["downstream_caps"]["max_result_reference_edges"]:
            raise RuntimeError("selected RESULT edge cap exceeded")
        for start in range(0, len(ncts), int(plan["downstream_caps"]["max_result_ids_per_request"])):
            batch = ncts[start:start + int(plan["downstream_caps"]["max_result_ids_per_request"])]
            if counters["result_records"] + len(batch) > plan["downstream_caps"]["max_result_records"]:
                raise RuntimeError("ResultsSection record cap exceeded")
            query = {"filter.ids": ",".join(batch), "fields": frozen.DETAIL_FIELDS,
                     "format": "json", "pageSize": str(len(batch)), "countTotal": "true"}
            raw = request(plan["endpoint"], query, "results", "application/json")
            parsed = json.loads(raw)
            returned = parsed.get("studies")
            if not isinstance(returned, list) or len(returned) != len(batch):
                raise RuntimeError("ResultsSection response count differs from bound NCT IDs")
            for study in returned:
                nct = frozen._study_nct(study)
                if nct not in batch or nct in studies: raise RuntimeError("ResultsSection identity mismatch/duplicate")
                studies[nct] = study
            counters["result_records"] += len(returned)

        pmids = sorted(counters["unique_pmids"], key=lambda x: (len(x), x))
        crosswalk: dict[str, dict[str, Any]] = {}
        for start in range(0, len(pmids), int(plan["downstream_caps"]["max_pmids_per_request"])):
            batch = pmids[start:start + int(plan["downstream_caps"]["max_pmids_per_request"])]
            query = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            raw = request(PUBMED_ENDPOINT, query, "pubmed", "application/xml")
            crosswalk.update(_parse_pubmed(raw))
        capture_hash = _sha(_canonical({"records": {k: studies[k] for k in sorted(studies)}}))
        pubmed_hash = _sha(_canonical(crosswalk))
        bundle = _build_unlabeled_bundle(studies, crosswalk, plan, capture_hash, pubmed_hash)
        _write_new(output_dir / "unlabeled-source-bundle.json", _canonical(bundle))
        eligible_count = int(bundle["capacity_gate"]["independent_component_count"])
        summary = {"schema": SCHEMA, "state": "complete_capture_only", "plan_sha256": plan["plan_sha256"],
                   "requests": counters["requests"], "studies": counters["studies"],
                   "result_records": counters["result_records"], "result_edges": counters["result_edges"],
                   "unique_pmids": len(counters["unique_pmids"]), "independent_components": eligible_count,
                   "capacity_eligible": eligible_count >= plan["downstream_caps"]["capacity_pass_threshold"],
                   "discovery_stop_reason": discovery_stop_reason,
                   "labels_generated": False, "candidate_created": False, "confirmation_created": False,
                   "release_authorized": False, "bytes": counters["bytes"]}
        _write_new(output_dir / "capture-summary.json", _canonical(summary))
        return summary
    except Exception as exc:
        summary = {"schema": SCHEMA, "state": "terminal_failure_or_unknown", "plan_sha256": plan["plan_sha256"],
                   "requests": counters["requests"], "studies": counters["studies"],
                   "result_records": counters["result_records"], "result_edges": counters["result_edges"],
                   "unique_pmids": len(counters["unique_pmids"]), "bytes": counters["bytes"],
                   "error": f"{type(exc).__name__}: {exc}", "no_resume": True}
        try: _write_new(output_dir / "capture-summary.json", _canonical(summary))
        except Exception: pass
        raise

