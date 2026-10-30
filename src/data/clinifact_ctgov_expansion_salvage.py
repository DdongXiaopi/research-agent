"""Offline-only salvage plan for a durable CT.gov cap-crossing capture.

This module verifies persisted v3/v4 evidence, recomputes the deterministic
source-order candidate prefix, and freezes a downstream-only plan. It never
opens a network connection or creates labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import clinifact_ctgov_expansion_plan as frozen
from . import clinifact_ctgov_expansion_continuation as continuation
from . import clinifact_ctgov_expansion_continuation_execute as old_runner
from . import clinifact_ctgov_expansion_capture as capture

SCHEMA = "clinifact-ctgov-expansion-cap-aware-salvage-plan-v1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_v4_plan(plan: dict[str, Any], path: Path) -> None:
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan.get("schema") != continuation.SCHEMA
            or plan.get("plan_sha256") != capture._sha(capture._canonical(unsigned))
            or capture._sha(path.read_bytes()) != capture._sha(capture._canonical(plan))):
        raise ValueError("immutable v4 continuation plan hash mismatch")


def build_offline_salvage(*, v3_plan_path: Path, v3_approval_path: Path,
                          v3_capture_dir: Path, bindings_path: Path,
                          v4_plan_path: Path, v4_capture_dir: Path,
                          v4_execution_manifest_path: Path,
                          v4_execution_approval_path: Path,
                          v4_root_approval_path: Path, output_dir: Path,
                          preflight_cli_path: Path,
                          root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify all 16 durable v4 responses and freeze a source-order prefix."""
    root = root.resolve()
    v3_plan = _read(v3_plan_path)
    frozen.validate_plan(v3_plan)
    if v3_plan.get("plan_sha256") != capture._sha(capture._canonical(
            {k: v for k, v in v3_plan.items() if k != "plan_sha256"})):
        raise ValueError("v3 plan hash mismatch")
    v3_approval = _read(v3_approval_path)
    capture.validate_approval(v3_approval, v3_plan, v3_capture_dir)
    audit = continuation.audit_discovery_capture(plan_path=v3_plan_path,
        capture_dir=v3_capture_dir, bindings_path=bindings_path, root=root)

    v4_plan = _read(v4_plan_path)
    _validate_v4_plan(v4_plan, v4_plan_path)
    if v4_plan.get("parent_plan_sha256") != v3_plan["plan_sha256"]:
        raise ValueError("v4 plan is not bound to the supplied v3 plan")
    if v4_plan.get("output_dir") != str(v4_capture_dir.resolve()):
        raise ValueError("v4 plan output path mismatch")
    if v4_plan["source_capture"]["capture_page_receipts"] != audit["page_receipts"]:
        raise ValueError("v4 source capture receipts differ from verified v3 capture")
    if v4_plan["continuation_query"]["params"].get("pageToken") != audit["next_page_token"]:
        raise ValueError("v4 start cursor differs from verified v3 cursor")
    execution_manifest = _read(v4_execution_manifest_path)
    if (execution_manifest.get("plan_sha256") != v4_plan["plan_sha256"]
            or execution_manifest.get("output_dir") != str(v4_capture_dir.resolve())):
        raise ValueError("v4 execution manifest is not bound to the saved capture")
    v4_execution_approval = _read(v4_execution_approval_path)
    v4_root_approval = _read(v4_root_approval_path)
    for approval_doc in (v4_execution_approval, v4_root_approval):
        if approval_doc.get("plan_sha256") != v4_plan["plan_sha256"]:
            raise ValueError("v4 approval artifact is not bound to its immutable plan")
    run = _read(v4_capture_dir / "run.json")
    if run.get("plan_sha256") != v4_plan["plan_sha256"] or run.get("dispatch_authorized") is not True:
        raise ValueError("v4 capture does not record an authorized dispatch")

    # Verify each response and ledger and the one-way cursor chain. The first
    # 15 saved selector receipts are recomputed and compared; page 16 is
    # selected by the cap-aware selector and stops before the first overflow.
    parent_ncts = set(v4_plan["source_capture"]["capture_candidate_nct_ids"])
    parent_pmids = set(v4_plan["source_capture"]["capture_candidate_pmids"])
    candidates: dict[str, dict[str, Any]] = {}
    for row in audit["studies"]:
        nct = frozen._study_nct(row)
        if nct in parent_ncts:
            refs = [p for p in frozen._result_pmids(row) if p in parent_pmids]
            if refs:
                candidates[nct] = {"nct_id": nct, "result_pmids": refs,
                    "row": row, "page_number": 0, "row_index": 0}
    if set(candidates) != parent_ncts:
        raise ValueError("cannot reconstruct v3 candidates from verified raw pages")
    budget = v4_plan["remaining_downstream_budget"]
    all_seen_ncts: set[str] = set()
    all_seen_pmids: set[str] = set()
    selected_pmids = set(parent_pmids)
    edges = int(v4_plan["source_capture"]["capture_candidate_edge_count"])
    next_token = str(v4_plan["continuation_query"]["params"]["pageToken"])
    raw_receipts: list[dict[str, Any]] = []
    page_selection_hashes: list[dict[str, Any]] = []
    page16_receipt: dict[str, Any] | None = None
    for index in range(1, 17):
        ledger_path = v4_capture_dir / "ledger" / f"attempt-{index:04d}.json"
        raw_path = v4_capture_dir / "raw" / f"request-{index:04d}-response.bin"
        rec = _read(ledger_path)
        raw = raw_path.read_bytes()
        raw_sha = capture._sha(raw)
        if (rec.get("request_index") != index or rec.get("phase") != "discovery"
                or rec.get("state") != "captured" or rec.get("status") != 200
                or rec.get("truncated") is not False or rec.get("retries") != 0
                or rec.get("redirects") != 0 or rec.get("plan_sha256") != v4_plan["plan_sha256"]
                or rec.get("response_sha256") != raw_sha or len(raw) > frozen.MAX_RESPONSE_BYTES_PER_REQUEST
                or rec.get("params", {}).get("pageToken") != next_token):
            raise ValueError(f"v4 response/ledger {index} is invalid or out of cursor order")
        page = json.loads(raw)
        rows = page.get("studies")
        if not isinstance(rows, list) or len(rows) > 1000:
            raise ValueError(f"v4 discovery response {index} violates page bound")
        ncts = [frozen._study_nct(row) for row in rows]
        if len(ncts) != len(set(ncts)) or set(ncts) & all_seen_ncts:
            raise ValueError(f"duplicate NCT in v4 continuation pages at page {index}")
        if index <= 15:
            saved = _read(v4_capture_dir / "selection" / f"page-{index:02d}.json")
            saved_body = {k: v for k, v in saved.items() if k != "selection_sha256"}
            if (saved.get("response_sha256") != raw_sha
                    or saved.get("selection_sha256") != capture._sha(capture._canonical(saved_body))):
                raise ValueError(f"v4 saved selector receipt {index} hash mismatch")
            recomputed = continuation.select_continuation_page(rows, page_number=index,
                response_sha256=raw_sha,
                excluded_nct_ids=set(v4_plan["identity_exclusions"]["nct_ids"]) | all_seen_ncts,
                excluded_pmids=set(v4_plan["identity_exclusions"]["pmids"]) | all_seen_pmids,
                already_selected_nct_ids=set(candidates), already_selected_pmids=selected_pmids,
                already_selected_result_edges=edges,
                max_candidates=max(1, int(budget["max_results_records"]) - (len(candidates)-len(parent_ncts))))
            if [x["nct_id"] for x in recomputed["candidates"]] != [x["nct_id"] for x in saved["candidates"]]:
                raise ValueError(f"v4 saved selector receipt {index} differs from deterministic recomputation")
            for candidate in recomputed["candidates"]:
                candidates[candidate["nct_id"]] = candidate
                edges += len(candidate["result_pmids"])
                selected_pmids.update(candidate["result_pmids"])
            page_selection_hashes.append({"page": index, "selection_sha256": saved["selection_sha256"]})
        else:
            selected = continuation.select_continuation_page(rows, page_number=index,
                response_sha256=raw_sha,
                excluded_nct_ids=set(v4_plan["identity_exclusions"]["nct_ids"]) | all_seen_ncts,
                excluded_pmids=set(v4_plan["identity_exclusions"]["pmids"]) | all_seen_pmids,
                already_selected_nct_ids=set(candidates), already_selected_pmids=selected_pmids,
                already_selected_result_edges=edges,
                max_candidates=max(1, int(budget["max_results_records"]) - (len(candidates)-len(parent_ncts))))
            if not selected.get("cap_stop"):
                raise ValueError("saved page 16 did not reproduce the expected frozen-cap stop")
            for candidate in selected["candidates"]:
                candidates[candidate["nct_id"]] = candidate
                edges += len(candidate["result_pmids"])
                selected_pmids.update(candidate["result_pmids"])
            page16_receipt = {"schema": "clinifact-ctgov-continuation-page-selection-v1",
                "plan_sha256": v4_plan["plan_sha256"], "page_number": 16,
                "response_sha256": raw_sha, "rows": len(rows),
                "candidates": selected["candidates"], "exclusions": selected["exclusions"],
                "cap_stop": selected["cap_stop"], "labels_generated": False}
            page16_receipt["selection_sha256"] = capture._sha(capture._canonical(page16_receipt))
            page_selection_hashes.append({"page": 16,
                "selection_sha256": page16_receipt["selection_sha256"]})
        all_seen_ncts.update(ncts)
        all_seen_pmids.update(p for row in rows for p in frozen._result_pmids(row))
        raw_receipts.append({"request_index": index, "raw_sha256": raw_sha,
            "ledger_sha256": frozen.sha_file(ledger_path), "bytes": len(raw),
            "page_token": rec["params"]["pageToken"],
            "returned_next_page_token": page.get("nextPageToken")})
        next_token = page.get("nextPageToken")
        if not isinstance(next_token, str) or not next_token:
            raise ValueError(f"v4 page {index} lacks the next cursor")
    if page16_receipt is None:
        raise ValueError("page16 safe-prefix receipt was not constructed")
    if (len(selected_pmids) > int(budget["max_unique_pmids_including_saved_capture"])
            or edges > int(budget["max_result_edges_including_saved_capture"])
            or len(candidates)-len(parent_ncts) > int(budget["max_results_records"])):
        raise ValueError("safe prefix exceeds a frozen combined downstream cap")
    result_batches = (len(candidates)+99)//100
    pubmed_batches = (len(selected_pmids)+99)//100
    if result_batches > int(budget["max_results_batches"]) or pubmed_batches > int(budget["max_pubmed_batches"]):
        raise ValueError("safe prefix exceeds frozen ResultsSection/PubMed batch count")
    output_dir = output_dir.resolve()
    output_dir.relative_to((root / ".workbuddy").resolve())
    unsigned = {"schema": SCHEMA, "parent_plan_sha256": v3_plan["plan_sha256"],
        "continuation_plan_sha256": v4_plan["plan_sha256"],
        "v3_approval_sha256": v3_approval["approval_sha256"],
        "v4_execution_manifest_sha256": capture._sha(capture._canonical(execution_manifest)),
        "v4_execution_approval_sha256": capture._sha(capture._canonical(v4_execution_approval)),
        "v4_root_approval_sha256": capture._sha(capture._canonical(v4_root_approval)),
        "v4_approval_artifacts_status": {"execution": v4_execution_approval.get("status"),
            "root": v4_root_approval.get("status")},
        "v3_capture_receipts": audit["page_receipts"], "v4_response_receipts": raw_receipts,
        "v4_saved_selector_receipt_hashes": page_selection_hashes[:15],
        "page16_safe_prefix_receipt": page16_receipt,
        "code_bindings": {"selector_module_sha256": frozen.sha_file(Path(continuation.__file__)),
            "executor_module_sha256": frozen.sha_file(Path(old_runner.__file__)),
            "preflight_cli_sha256": frozen.sha_file(preflight_cli_path)},
        "selection": {"v3_candidate_ncts": len(parent_ncts),
            "v3_result_edges": int(v4_plan["source_capture"]["capture_candidate_edge_count"]),
            "v3_unique_pmids": len(parent_pmids),
            "v4_new_candidate_ncts": len(candidates)-len(parent_ncts),
            "combined_candidate_ncts": len(candidates), "combined_result_edges": edges,
            "combined_unique_pmids": len(selected_pmids),
            "remaining_result_records": int(budget["max_results_records"])-(len(candidates)-len(parent_ncts)),
            "remaining_result_edges": int(budget["max_result_edges_including_saved_capture"])-edges,
            "remaining_unique_pmids": int(budget["max_unique_pmids_including_saved_capture"])-len(selected_pmids),
            "discovery_stop": page16_receipt["cap_stop"],
            "next_page_token_after_captured_page16": next_token},
        "downstream_only": {"discovery_requests": 0,
            "results_records": len(candidates), "results_batches_max": result_batches,
            "results_batch_size": 100, "pubmed_pmids": len(selected_pmids),
            "pubmed_batches_max": pubmed_batches, "pubmed_batch_size": 100,
            "max_future_physical_requests": result_batches+pubmed_batches,
            "requires_fresh_execution_approval": True, "no_discovery_replay": True},
        "output_dir": str(output_dir), "network_calls": 0, "model_calls": 0,
        "labels_generated": False, "references_created": False,
        "candidate_search_authorized": False, "release_authorized": False}
    plan = {**unsigned, "plan_sha256": capture._sha(capture._canonical(unsigned))}
    source_projection = {"v3_plan_sha256": v3_plan["plan_sha256"],
        "v3_approval_sha256": v3_approval["approval_sha256"],
        "v4_plan_sha256": v4_plan["plan_sha256"],
        "v4_execution_manifest_sha256": plan["v4_execution_manifest_sha256"],
        "v4_execution_approval_sha256": plan["v4_execution_approval_sha256"],
        "v4_root_approval_sha256": plan["v4_root_approval_sha256"],
        "v3_receipts": audit["page_receipts"], "v4_receipts": raw_receipts,
        "selection_receipts": page_selection_hashes,
        "selector_sha256": plan["code_bindings"]["selector_module_sha256"],
        "executor_sha256": plan["code_bindings"]["executor_module_sha256"]}
    preflight = {"schema": "clinifact-ctgov-cap-aware-salvage-preflight-v1",
        "status": "passed_zero_call_salvage_preflight", "plan_sha256": plan["plan_sha256"],
        "verified_inputs_sha256": capture._sha(capture._canonical(source_projection)),
        "v3_pages_verified": len(audit["page_receipts"]), "v4_pages_verified": len(raw_receipts),
        "v4_page16_safe_prefix_rows": page16_receipt["cap_stop"]["row_index"],
        "downstream_request_limit": result_batches+pubmed_batches,
        "network_calls": 0, "model_calls": 0, "dispatch_authorized": False,
        "labels_generated": False}
    preflight["preflight_sha256"] = capture._sha(capture._canonical(preflight))
    return plan, {"preflight": preflight, "page16_receipt": page16_receipt,
        "verified_inputs": source_projection, "candidate_records": list(candidates.values()),
        "selected_pmids": sorted(selected_pmids, key=lambda x: (len(x), x))}


def persist_offline_bundle(plan: dict[str, Any], details: dict[str, Any], *, plan_dir: Path) -> None:
    plan_dir.mkdir(parents=True, exist_ok=False)
    capture._write_new(plan_dir / "plan.json", capture._canonical(plan))
    capture._write_new(plan_dir / "preflight.json", capture._canonical(details["preflight"]))
    capture._write_new(plan_dir / "page-16-safe-prefix-selection.json",
        capture._canonical(details["page16_receipt"]))
    # This compact manifest is a pending downstream-only approval contract.
    # A later executor/CLI must bind its own exact source hashes before approval.
    pending = {"schema": "clinifact-ctgov-cap-aware-salvage-approval-v1",
        "status": "pending_root_approval", "plan_sha256": plan["plan_sha256"],
        "verified_inputs_sha256": details["preflight"]["verified_inputs_sha256"],
        "dispatch_authorized": False, "approved_by": None, "approved_at_utc": None,
        "approval_statement": None,
        "scope": "Downstream ResultsSection and PubMed metadata requests from saved deterministic candidate prefix only; zero discovery replay; no labels/release"}
    pending["approval_sha256"] = capture._sha(capture._canonical(pending))
    capture._write_new(plan_dir / "downstream-approval.pending.json", capture._canonical(pending))

