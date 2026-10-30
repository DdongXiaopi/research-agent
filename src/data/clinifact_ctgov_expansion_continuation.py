"""Freeze a no-replay continuation plan from a completed capped v3 discovery."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from . import clinifact_ctgov_expansion_plan as frozen
from .clinifact_ctgov_expansion_capture import _canonical, _sha, _write_new, verify_bound_inputs, pending_root_approval

SCHEMA = "clinifact-ctgov-expansion-continuation-plan-v1"


def select_continuation_page(studies: list[Mapping[str, Any]], *, page_number: int,
                             response_sha256: str, excluded_nct_ids: set[str],
                             excluded_pmids: set[str], already_selected_nct_ids: set[str],
                             already_selected_pmids: set[str], already_selected_result_edges: int,
                             max_candidates: int) -> dict[str, Any]:
    """Apply the frozen 100-row selector in chunks to a documented 1,000-row page.

    The v3 selector module is hash-bound and remains unchanged. This wrapper
    preserves its semantics while translating offsets back to the 1,000-row
    source page and respecting combined downstream ceilings.
    """
    if not 1 <= page_number <= 18 or len(studies) > 1000:
        raise ValueError("continuation page is outside the 18 x 1,000 bound")
    if not 1 <= max_candidates <= 1500:
        raise ValueError("remaining ResultsSection candidate capacity is invalid")
    page_ncts = [frozen._study_nct(row) for row in studies]
    if len(page_ncts) != len(set(page_ncts)):
        raise ValueError("duplicate NCT identity within 1,000-record continuation page")
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    selected_ncts = set(already_selected_nct_ids)
    selected_pmids = set(already_selected_pmids)
    edge_count = already_selected_result_edges
    cap_stop: dict[str, Any] | None = None
    # Process one row at a time. The frozen selector raises when a whole
    # 100-row chunk crosses its unique-PMID/edge ceiling, which discards a
    # perfectly usable deterministic prefix of an already captured response.
    # Single-row calls preserve source order and stop at the first row that
    # would exceed a cap; they never skip forward to cherry-pick later rows.
    for row_index, study in enumerate(studies):
        remaining_slots = max_candidates - len(candidates)
        if remaining_slots <= 0:
            break
        try:
            sub = frozen.select_fresh_candidates([study], page_number=1,
                response_sha256=response_sha256, excluded_nct_ids=excluded_nct_ids,
                excluded_pmids=excluded_pmids, already_selected_nct_ids=selected_ncts,
                already_selected_pmids=selected_pmids,
                already_selected_result_edges=edge_count, max_candidates=remaining_slots)
        except ValueError as exc:
            reason = str(exc)
            if reason not in {"fresh unique PMID cap exceeded", "fresh RESULT reference edge cap exceeded"}:
                raise
            cap_stop = {"row_index": row_index, "nct_id": frozen._study_nct(study),
                        "reason": reason, "stop_before_candidate": True}
            break
        for row in sub["candidates"] + sub["exclusions"]:
            row["page_number"] = page_number
            row["row_index"] = row_index
        candidates.extend(sub["candidates"])
        exclusions.extend(sub["exclusions"])
        selected_ncts.update(sub["selected_nct_ids"])
        selected_pmids.update(sub["selected_result_pmids"])
        edge_count += sum(len(row["result_pmids"]) for row in sub["candidates"])
    return {"page_number": page_number, "source_response_sha256": response_sha256,
            "candidates": candidates, "exclusions": exclusions,
            "selected_nct_ids": [row["nct_id"] for row in candidates],
            "selected_result_pmids": sorted(selected_pmids - already_selected_pmids,
                                             key=lambda x: (len(x), x)),
            "cap_stop": cap_stop,
            "candidate_search_authorized": False, "labels_generated": False}


def audit_discovery_capture(*, plan_path: Path, capture_dir: Path, bindings_path: Path,
                            root: Path) -> dict[str, Any]:
    """Verify every persisted discovery response/receipt, then recompute the selector offline."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frozen.validate_plan(plan)
    captured_plan = json.loads((capture_dir / "plan.json").read_text(encoding="utf-8"))
    if captured_plan != plan:
        raise ValueError("capture plan differs from authoritative v3 plan")
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    verified = verify_bound_inputs(plan, plan_path, bindings, root=root)
    saved_verification = json.loads((capture_dir / "verified-bindings.json").read_text(encoding="utf-8"))
    if saved_verification != verified:
        raise ValueError("capture verified-binding receipt differs from recomputed hashes")
    all_studies: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    seen_ncts: set[str] = set()
    seen_pmids: set[str] = set()
    captured_pmids: set[str] = set()
    edges = 0
    page_receipts = []
    next_token = plan["first_request_params"]["pageToken"]
    for index in range(1, int(plan["page_policy"]["max_pages"]) + 1):
        ledger_path = capture_dir / "ledger" / f"attempt-{index:04d}.json"
        rec = json.loads(ledger_path.read_text(encoding="utf-8"))
        raw_path = capture_dir / rec.get("raw_file", "")
        raw = raw_path.read_bytes()
        if (rec.get("phase") != "discovery" or rec.get("state") != "captured"
                or rec.get("status") != 200 or rec.get("truncated") is not False
                or rec.get("retries") != 0 or rec.get("redirects") != 0
                or rec.get("plan_sha256") != plan["plan_sha256"]
                or _sha(raw) != rec.get("response_sha256")):
            raise ValueError(f"discovery attempt {index} is incomplete or hash-invalid")
        params = rec.get("params") or {}
        if (params.get("pageToken") != next_token
                or params.get("filter.advanced") != plan["query_selector"]["filter.advanced"]
                or params.get("fields") != plan["query_selector"]["fields"]
                or params.get("pageSize") != str(plan["page_policy"]["page_size"])):
            raise ValueError(f"discovery attempt {index} did not follow the frozen cursor/query")
        response = json.loads(raw)
        rows = response.get("studies")
        if not isinstance(rows, list) or len(rows) > plan["page_policy"]["page_size"]:
            raise ValueError(f"discovery page {index} violates its page cap")
        selected = frozen.select_fresh_candidates(rows, page_number=index,
            response_sha256=rec["response_sha256"],
            excluded_nct_ids=set(plan["historical_identity_exclusions"]["nct_ids"]),
            excluded_pmids=set(plan["historical_identity_exclusions"]["pmids"]),
            already_selected_nct_ids=seen_ncts, already_selected_pmids=seen_pmids,
            already_selected_result_edges=edges)
        for row in rows:
            seen_ncts.add(frozen._study_nct(row))
            captured_pmids.update(frozen._result_pmids(row))
        all_studies.extend(rows)
        all_candidates.extend(selected["candidates"])
        edges += sum(len(x["result_pmids"]) for x in selected["candidates"])
        seen_pmids.update(selected["selected_result_pmids"])
        next_token = response.get("nextPageToken")
        page_receipts.append({"attempt_index": index, "ledger_sha256": frozen.sha_file(ledger_path),
                              "raw_sha256": _sha(raw), "raw_path": str(raw_path.resolve())})
    if not isinstance(next_token, str) or not next_token:
        raise ValueError("captured page-cap response has no continuation cursor")
    if len(all_studies) != plan["page_policy"]["max_studies"]:
        raise ValueError("persisted capture did not complete the frozen discovery page range")
    if len(seen_ncts) != len(all_studies):
        raise ValueError("duplicate NCT identity in persisted discovery range")
    return {"plan": plan, "bindings": bindings, "verified_bindings": verified,
            "page_receipts": page_receipts, "studies": all_studies,
            "candidate_nct_ids": sorted({x["nct_id"] for x in all_candidates}),
            "candidate_edges": edges, "candidate_pmids": sorted(seen_pmids, key=lambda x: (len(x), x)),
            "captured_pmids": sorted(captured_pmids, key=lambda x: (len(x), x)),
            "next_page_token": next_token, "study_count": len(all_studies)}


def build_continuation_plan(audit: Mapping[str, Any], *, output_dir: Path,
                            additional_discovery_pages: int = 18) -> dict[str, Any]:
    """Bind a fresh cursor continuation, excluding all first-range NCTs/PMIDs."""
    old = audit["plan"]
    if type(additional_discovery_pages) is not int or not 1 <= additional_discovery_pages <= 18:
        raise ValueError("continuation discovery cap must be between 1 and 18 pages")
    fresh_ncts = set(audit["candidate_nct_ids"])
    fresh_pmids = set(audit["candidate_pmids"])
    excluded = old["historical_identity_exclusions"]
    all_ncts = sorted(set(excluded["nct_ids"]) | set(audit["studies"] and
        [frozen._study_nct(row) for row in audit["studies"]]))
    captured_pmids = set(audit["captured_pmids"])
    all_pmids = sorted(set(excluded["pmids"]) | captured_pmids, key=lambda x: (len(x), x))
    per_study = audit["candidate_edges"] / max(1, audit["study_count"])
    unique_per_study = len(fresh_pmids) / max(1, audit["study_count"])
    candidate_per_study = len(fresh_ncts) / max(1, audit["study_count"])
    continuation_page_size = 1000
    projected_studies = additional_discovery_pages * continuation_page_size
    projected_new_candidates = int(round(projected_studies * candidate_per_study))
    projected_new_edges = int(round(projected_studies * per_study))
    projected_new_pmids = int(round(projected_studies * unique_per_study))
    strict_per_record = 27 / 205  # previously completed strict source-pool conversion
    projected_components = (len(fresh_ncts) + projected_new_candidates) * strict_per_record
    endpoint = old["endpoint"]
    output_dir = output_dir.resolve()
    unsigned = {
        "schema": SCHEMA,
        "parent_plan_sha256": old["plan_sha256"],
        "implementation_bindings": {"continuation_module_sha256": frozen.sha_file(Path(__file__)),
            "capture_runner_sha256": frozen.sha_file(Path(__file__).with_name("clinifact_ctgov_expansion_capture.py")),
            "strict_adapter_sha256": frozen.sha_file(Path(__file__).with_name("clinifact_ctgov_pubmed_adapter_v1.py"))},
        "source_capture": {"study_count": audit["study_count"],
            "capture_page_receipts": audit["page_receipts"],
            "capture_candidate_nct_ids": sorted(fresh_ncts),
            "capture_candidate_edge_count": audit["candidate_edges"],
            "capture_candidate_pmids": sorted(fresh_pmids, key=lambda x: (len(x), x)),
            "capture_excluded_nct_sha256": _sha(_canonical(sorted(all_ncts))),
            "capture_excluded_pmid_sha256": _sha(_canonical(sorted(all_pmids, key=lambda x: (len(x), x)))),
            "captured_result_pmid_count": len(captured_pmids)},
        "continuation_query": {"endpoint": endpoint, "method": "GET",
            "params": {**old["first_request_params"], "pageSize": str(continuation_page_size),
                       "pageToken": audit["next_page_token"]},
            "follow_only_returned_nextPageToken": True, "no_restart_or_page_replay": True,
            "oas_max_page_size_documented": old["oas"]["max_page_size_documented"],
            "max_pages": additional_discovery_pages, "page_size": continuation_page_size,
            "max_additional_studies": projected_studies,
            "stop_at_page_cap_and_continue_selected_downstream": True},
        "identity_exclusions": {"nct_ids": all_ncts, "pmids": all_pmids,
            "additional_capture_nct_ids_are_added_before_selection": True,
            "additional_capture_pmid_ids_are_added_before_selection": True},
        "remaining_downstream_budget": {"max_results_records": 1500 - len(fresh_ncts),
            "max_results_batches": 15, "max_nct_ids_per_results_batch": 100,
            "max_new_unique_pmids_after_saved_capture": 2000 - len(fresh_pmids),
            "max_unique_pmids_including_saved_capture": 2000,
            "max_pubmed_batches": 20, "max_pmids_per_pubmed_batch": 100,
            "max_new_result_edges_after_saved_capture": 3000 - audit["candidate_edges"],
            "max_result_edges_including_saved_capture": 3000,
            "max_requests": additional_discovery_pages + 15 + 20},
        "response_policy": {"max_bytes_each": frozen.MAX_RESPONSE_BYTES_PER_REQUEST,
            "timeout_seconds": frozen.TIMEOUT_SECONDS, "redirects": 0, "retries": 0,
            "persist_reservation_before_each_request": True, "persist_raw_before_parse": True,
            "unknown_after_post_dispatch_persistence_failure_is_terminal": True},
        "capacity_gate": {"target_independent_components": 144,
            "observed_candidate_nct_rate": candidate_per_study,
            "observed_result_edge_rate": per_study,
            "observed_unique_pmid_rate": unique_per_study,
            "historical_strict_components_per_source_record": strict_per_record,
            "projected_total_candidate_ncts": len(fresh_ncts) + projected_new_candidates,
            "max_total_candidate_ncts": 1500,
            "projected_total_result_edges": audit["candidate_edges"] + projected_new_edges,
            "projected_total_unique_pmids": len(fresh_pmids) + projected_new_pmids,
            "projected_independent_components": round(projected_components, 1),
            "projection_is_not_a_guarantee": True,
            "below_144_disposition": "stop_at_cap_and_report_insufficient_no_gate_relaxation"},
        "output_dir": str(output_dir),
        "dispatch_authorized": False, "network_calls": 0, "model_calls": 0,
        "labels_generated": False, "references_created": False,
        "confirmation_created": False, "candidate_search_authorized": False,
        "release_authorized": False,
    }
    return {**unsigned, "plan_sha256": _sha(_canonical(unsigned))}


def persist_continuation_bundle(plan: Mapping[str, Any], *, plan_dir: Path,
                                approval_path: Path) -> dict[str, Any]:
    """Create write-once plan, zero-call preflight, WAL and separate pending approval."""
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("schema") != SCHEMA or plan.get("plan_sha256") != _sha(_canonical(unsigned)):
        raise ValueError("continuation plan hash mismatch")
    plan_dir.mkdir(parents=True, exist_ok=False)
    plan_path = plan_dir / "plan.json"
    _write_new(plan_path, _canonical(plan))
    wal = {"schema": "clinifact-ctgov-expansion-continuation-wal-v1",
           "plan_sha256": plan["plan_sha256"], "state": "not_dispatched",
           "dispatch_authorized": False, "network_calls": 0, "model_calls": 0,
           "existing_discovery_replay_allowed": False}
    wal["write_ahead_sha256"] = _sha(_canonical(wal))
    _write_new(plan_dir / "write-ahead.json", _canonical(wal))
    preflight = {"schema": "clinifact-ctgov-expansion-continuation-preflight-v1",
        "status": "passed_zero_call_continuation_plan_integrity", "plan_sha256": plan["plan_sha256"],
        "parent_capture_hashes_verified": True, "dispatch_authorized": False,
        "network_calls": 0, "model_calls": 0}
    preflight["preflight_sha256"] = _sha(_canonical(preflight))
    _write_new(plan_dir / "preflight.json", _canonical(preflight))
    approval = pending_root_approval({"plan_sha256": plan["plan_sha256"]},
        output_dir=plan["output_dir"])
    approval["schema"] = "clinifact-ctgov-expansion-continuation-root-approval-v1"
    approval["scope"] = "Continue only from the captured page-15 cursor, excluding all captured identities; bounded ResultsSection/PubMed metadata only"
    body = {k: v for k, v in approval.items() if k != "approval_sha256"}
    approval["approval_sha256"] = _sha(_canonical(body))
    _write_new(approval_path, _canonical(approval))
    return {"plan_sha256": plan["plan_sha256"], "plan_path": str(plan_path),
            "approval_path": str(approval_path), "dispatch_authorized": False,
            "network_calls": 0, "model_calls": 0}
