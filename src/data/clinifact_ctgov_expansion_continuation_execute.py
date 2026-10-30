"""One-shot, approval-gated executor for the frozen CT.gov v4 continuation.

This module is intentionally separate from the frozen v3 capture and v4 planner.
It has no resume/replay path. Every physical request reserves a durable ledger
entry, persists the bounded raw response, then parses it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping

from . import clinifact_ctgov_expansion_plan as frozen
from . import clinifact_ctgov_expansion_continuation as continuation
from . import clinifact_ctgov_expansion_capture as capture

EXEC_SCHEMA = "clinifact-ctgov-expansion-continuation-execution-v1"
APPROVAL_SCHEMA = "clinifact-ctgov-expansion-continuation-execution-approval-v1"
PUBMED_ENDPOINT = capture.PUBMED_ENDPOINT


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return frozen.canonical(value)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_continuation_plan(plan: Mapping[str, Any], path: Path) -> None:
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("schema") != continuation.SCHEMA or plan.get("plan_sha256") != _sha(_canonical(unsigned)):
        raise ValueError("continuation plan schema/hash mismatch")
    if _sha(path.read_bytes()) != _sha(_canonical(plan)):
        raise ValueError("continuation plan file is not canonical or changed")
    q = plan["continuation_query"]
    p = plan["remaining_downstream_budget"]
    params = q.get("params") or {}
    if (q["endpoint"] != frozen.ENDPOINT or q.get("method") != "GET"
            or params.get("filter.advanced") != frozen.EXPECTED_FILTER
            or params.get("fields") != frozen.EXPECTED_DISCOVERY_FIELDS
            or params.get("format") != "json" or params.get("countTotal") != "true"
            or q["max_pages"] != 18 or q["page_size"] != 1000
            or q["params"].get("pageSize") != "1000"
            or q["no_restart_or_page_replay"] is not True
            or q["follow_only_returned_nextPageToken"] is not True
            or p["max_results_batches"] > 15 or p["max_pubmed_batches"] > 20
            or p["max_results_records"] < 0 or p["max_unique_pmids_including_saved_capture"] > 2000
            or p["max_result_edges_including_saved_capture"] > 3000
            or plan["response_policy"]["redirects"] != 0
            or plan["response_policy"]["retries"] != 0):
        raise ValueError("continuation plan exceeds frozen request or combined downstream limits")


def _validate_v3_approval(approval: Mapping[str, Any], parent_plan: Mapping[str, Any],
                          parent_output: Path) -> None:
    capture.validate_approval(approval, parent_plan, parent_output)


def verify_execution_inputs(*, parent_plan_path: Path, parent_approval_path: Path,
                            parent_capture_dir: Path, bindings_path: Path,
                            continuation_plan_path: Path, output_dir: Path,
                            root: Path) -> dict[str, Any]:
    """Verify all v3 and v4 immutable inputs before any possible dispatch."""
    parent_plan = _read(parent_plan_path)
    frozen.validate_plan(parent_plan)
    parent_unsigned = {k: v for k, v in parent_plan.items() if k != "plan_sha256"}
    if parent_plan["plan_sha256"] != _sha(_canonical(parent_unsigned)):
        raise ValueError("v3 parent plan hash mismatch")
    if parent_plan["plan_sha256"] != _read(continuation_plan_path)["parent_plan_sha256"]:
        raise ValueError("v4 is not bound to this v3 plan")
    parent_approval = _read(parent_approval_path)
    _validate_v3_approval(parent_approval, parent_plan, parent_capture_dir)

    plan = _read(continuation_plan_path)
    validate_continuation_plan(plan, continuation_plan_path)
    if plan["output_dir"] != str(output_dir.resolve()):
        raise ValueError("v4 output directory differs from the frozen plan")
    expected_impl = plan["implementation_bindings"]
    actual_impl = {
        "continuation_module_sha256": frozen.sha_file(Path(continuation.__file__)),
        "capture_runner_sha256": frozen.sha_file(Path(capture.__file__)),
        "strict_adapter_sha256": frozen.sha_file(Path(__file__).with_name("clinifact_ctgov_pubmed_adapter_v1.py")),
    }
    if expected_impl != actual_impl:
        raise ValueError("v4 implementation binding changed")

    audit = continuation.audit_discovery_capture(plan_path=parent_plan_path,
        capture_dir=parent_capture_dir, bindings_path=bindings_path, root=root.resolve())
    receipts = plan["source_capture"]["capture_page_receipts"]
    if receipts != audit["page_receipts"]:
        raise ValueError("v4 source receipts differ from verified v3 capture")
    if plan["source_capture"]["study_count"] != audit["study_count"]:
        raise ValueError("v4 source study count differs from v3 capture")
    if plan["continuation_query"]["params"].get("pageToken") != audit["next_page_token"]:
        raise ValueError("v4 start cursor is not the verified page-15 nextPageToken")
    if plan["source_capture"]["capture_candidate_nct_ids"] != audit["candidate_nct_ids"]:
        raise ValueError("v4 saved candidate NCT set differs from recomputed selector")
    if plan["source_capture"]["capture_candidate_pmids"] != audit["candidate_pmids"]:
        raise ValueError("v4 saved candidate PMID set differs from recomputed selector")

    # Include all explicitly bound historical/exposure inputs, and the complete
    # captured v3 discovery range, rather than only selected candidates.
    expected_excluded_nct = set(parent_plan["historical_identity_exclusions"]["nct_ids"])
    expected_excluded_nct.update(frozen._study_nct(row) for row in audit["studies"])
    expected_excluded_pmid = set(parent_plan["historical_identity_exclusions"]["pmids"])
    expected_excluded_pmid.update(audit["captured_pmids"])
    if plan["identity_exclusions"]["nct_ids"] != sorted(expected_excluded_nct):
        raise ValueError("v4 NCT exclusions do not cover all historical and v3 captured identities")
    canonical_pmids = sorted(expected_excluded_pmid, key=lambda x: (len(x), x))
    if plan["identity_exclusions"]["pmids"] != canonical_pmids:
        raise ValueError("v4 PMID exclusions do not cover all historical and v3 captured identities")
    try:
        output_dir.resolve().relative_to((root.resolve() / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("continuation output must be under project .workbuddy") from exc
    return {"parent_plan_sha256": parent_plan["plan_sha256"],
            "parent_approval_sha256": parent_approval["approval_sha256"],
            "parent_capture_page_count": len(audit["page_receipts"]),
            "parent_capture_raw_hashes": [x["raw_sha256"] for x in audit["page_receipts"]],
            "parent_capture_ledger_hashes": [x["ledger_sha256"] for x in audit["page_receipts"]],
            "continuation_plan_sha256": plan["plan_sha256"],
            "continuation_module_sha256": actual_impl["continuation_module_sha256"],
            "capture_runner_sha256": actual_impl["capture_runner_sha256"],
            "strict_adapter_sha256": actual_impl["strict_adapter_sha256"],
            "historical_nct_exclusions": len(parent_plan["historical_identity_exclusions"]["nct_ids"]),
            "historical_pmid_exclusions": len(parent_plan["historical_identity_exclusions"]["pmids"]),
            "complete_nct_exclusions": len(expected_excluded_nct),
            "complete_pmid_exclusions": len(expected_excluded_pmid),
            "next_page_token": audit["next_page_token"], "audit": audit,
            "parent_plan": parent_plan, "continuation_plan": plan}


def execution_approval_body(*, plan_sha256: str, inputs_sha256: str,
                            executor_sha256: str, cli_sha256: str,
                            output_dir: Path) -> dict[str, Any]:
    return {"schema": APPROVAL_SCHEMA, "status": "pending_root_approval",
            "plan_sha256": plan_sha256, "inputs_sha256": inputs_sha256,
            "executor_sha256": executor_sha256, "cli_sha256": cli_sha256,
            "output_dir": str(output_dir.resolve()), "dispatch_authorized": False,
            "approved_by": None, "approved_at_utc": None, "approval_statement": None,
            "scope": "18 pageSize=1000 continuation requests from saved v3 page-15 cursor, then bounded ResultsSection/PubMed metadata only; no labels or release"}


def execution_inputs_sha256(verified: Mapping[str, Any]) -> str:
    projection = {k: v for k, v in verified.items()
                  if k not in ("audit", "parent_plan", "continuation_plan")}
    return _sha(_canonical(projection))


def validate_execution_approval(approval: Mapping[str, Any], *, manifest: Mapping[str, Any],
                                approval_path: Path) -> None:
    body = {k: v for k, v in approval.items() if k != "approval_sha256"}
    if (approval.get("schema") != APPROVAL_SCHEMA or approval.get("status") != "approved"
            or approval.get("dispatch_authorized") is not True
            or approval.get("approval_sha256") != _sha(_canonical(body))
            or any(approval.get(k) != manifest.get(k) for k in (
                "plan_sha256", "inputs_sha256", "executor_sha256", "cli_sha256", "output_dir"))
            or not approval.get("approved_by") or not approval.get("approval_statement")
            or approval.get("output_dir") != str(Path(approval["output_dir"]).resolve())):
        raise ValueError("separate root approval must bind exact continuation plan, inputs, code, and output")


def _record_ids(study: Mapping[str, Any]) -> tuple[str, list[str]]:
    nct = frozen._study_nct(study)
    return nct, frozen._result_pmids(study)


def execute_continuation(*, verified: Mapping[str, Any], approval: Mapping[str, Any],
                         approval_path: Path, manifest: Mapping[str, Any], output_dir: Path,
                         root: Path, transport: Callable[..., tuple[int, Mapping[str, str], bytes]] = capture.urllib_transport) -> dict[str, Any]:
    """Execute one bounded continuation. Any uncertain attempt is terminal."""
    plan = verified["continuation_plan"]
    cli_path = Path(__file__).resolve().parents[2] / "scripts" / "run_clinifact_ctgov_expansion_continuation.py"
    if (manifest.get("plan_sha256") != plan.get("plan_sha256")
            or manifest.get("inputs_sha256") != execution_inputs_sha256(verified)
            or manifest.get("executor_sha256") != _sha(Path(__file__).read_bytes())
            or not cli_path.is_file() or manifest.get("cli_sha256") != _sha(cli_path.read_bytes())
            or manifest.get("output_dir") != str(output_dir.resolve())):
        raise ValueError("execution manifest no longer matches verified inputs/code/output")
    validate_execution_approval(approval, manifest=manifest, approval_path=approval_path)
    if output_dir.exists():
        raise FileExistsError("continuation output already exists; replay/resume forbidden")
    try:
        output_dir.resolve().relative_to((root.resolve() / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("continuation output must be under project .workbuddy") from exc
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "raw").mkdir(); (output_dir / "ledger").mkdir(); (output_dir / "selection").mkdir()
    for path, payload in ((output_dir / "plan.json", _canonical(plan)),
                          (output_dir / "execution-manifest.json", _canonical(manifest)),
                          (output_dir / "run.json", _canonical({"schema": EXEC_SCHEMA,
                           "plan_sha256": plan["plan_sha256"], "state": "ready",
                           "dispatch_authorized": True, "labels_generated": False}))):
        capture._write_new(path, payload)

    policy = plan["response_policy"]
    cap, timeout = int(policy["max_bytes_each"]), int(policy["timeout_seconds"])
    budgets = plan["remaining_downstream_budget"]
    counters: dict[str, Any] = {"requests": 0, "discovery": 0, "results": 0, "pubmed": 0, "studies": 0,
        "result_records": 0, "result_edges": int(plan["source_capture"]["capture_candidate_edge_count"]),
        "unique_pmids": set(plan["source_capture"]["capture_candidate_pmids"]), "bytes": {"discovery": 0, "results": 0, "pubmed": 0}}
    historical_ncts = set(verified["parent_plan"]["historical_identity_exclusions"]["nct_ids"])
    historical_pmids = set(verified["parent_plan"]["historical_identity_exclusions"]["pmids"])
    excluded_ncts = set(plan["identity_exclusions"]["nct_ids"])
    excluded_pmids = set(plan["identity_exclusions"]["pmids"])
    parent_ncts = set(plan["source_capture"]["capture_candidate_nct_ids"])
    parent_pmids = set(plan["source_capture"]["capture_candidate_pmids"])
    candidates: dict[str, dict[str, Any]] = {}
    # Recompute original v3 candidate references from its verified saved pages.
    for raw_study in verified["audit"]["studies"]:
        nct, refs = _record_ids(raw_study)
        if nct in parent_ncts:
            fresh = [x for x in refs if x in parent_pmids]
            if fresh:
                candidates[nct] = {"nct_id": nct, "result_pmids": fresh,
                    "row": raw_study, "page_number": 0, "row_index": 0}
    if set(candidates) != parent_ncts:
        raise ValueError("cannot reconstruct complete saved v3 candidate records")
    all_discovered_ncts: set[str] = set()
    all_discovered_pmids: set[str] = set()
    next_token = str(plan["continuation_query"]["params"]["pageToken"])
    params_base = dict(plan["continuation_query"]["params"])
    stop_reason = "server_exhausted"

    def request(url: str, params: Mapping[str, str], phase: str, accept: str) -> bytes:
        max_requests = int(plan["remaining_downstream_budget"]["max_requests"])
        if counters["requests"] >= max_requests:
            raise RuntimeError("combined physical request cap exceeded")
        phase_cap = {"discovery": 18, "results": int(budgets["max_results_batches"]),
                     "pubmed": int(budgets["max_pubmed_batches"])}[phase]
        if counters[phase] >= phase_cap:
            raise RuntimeError(f"{phase} request cap exceeded")
        idx = counters["requests"] + 1
        url_full = url + "?" + urllib.parse.urlencode(params, safe=",")
        req = {"phase": phase, "method": "GET", "url": url_full, "params": dict(params),
               "timeout_seconds": timeout, "response_cap_bytes": cap,
               "plan_sha256": plan["plan_sha256"], "execution_manifest_sha256": _sha(_canonical(manifest))}
        ledger = capture._reserve(output_dir, idx, req)
        rec = capture._read_json(ledger)
        counters["requests"] += 1; counters[phase] += 1
        try:
            status, headers, body = transport(url_full, timeout=timeout, cap=cap, accept=accept)
        except Exception as exc:
            rec.update({"state": "outcome_unknown", "error": f"{type(exc).__name__}: {exc}", "terminal": True})
            try: capture._finish_attempt(ledger, rec)
            except Exception: capture._mark_unknown_best_effort(output_dir, ledger, rec, "transport outcome unknown")
            raise RuntimeError("physical request outcome unknown; continuation is terminal") from exc
        truncated = len(body) > cap
        raw = body[:cap] if truncated else body
        try:
            raw_path = capture._persist_raw(output_dir, idx, raw, error=status != 200)
        except Exception as exc:
            rec.update({"state": "outcome_unknown", "status": status, "response_sha256": _sha(raw),
                        "terminal": True, "persistence_error": f"{type(exc).__name__}: {exc}"})
            try: capture._finish_attempt(ledger, rec)
            except Exception: capture._mark_unknown_best_effort(output_dir, ledger, rec, "raw persistence failed")
            raise RuntimeError("response persistence failed after dispatch; no retry") from exc
        rec.update({"state": "captured" if status == 200 and not truncated else "failed",
                    "status": status, "headers": dict(headers), "stored_bytes": len(raw),
                    "response_sha256": _sha(raw), "raw_file": str(raw_path.relative_to(output_dir)),
                    "truncated": truncated, "terminal": status != 200 or truncated})
        try: capture._finish_attempt(ledger, rec)
        except Exception as exc:
            capture._mark_unknown_best_effort(output_dir, ledger, rec, f"receipt failure: {exc}")
            raise RuntimeError("receipt persistence failed after dispatch; no retry") from exc
        counters["bytes"][phase] += len(raw)
        if status != 200 or truncated:
            raise RuntimeError(f"{phase} response failed status/size gate")
        if counters["bytes"][phase] > int(budgets.get({"discovery":"max_discovery_bytes_total",
                "results":"max_result_bytes_total","pubmed":"max_pubmed_bytes_total"}[phase], 2**63-1)):
            raise RuntimeError(f"{phase} aggregate response byte cap exceeded")
        return raw

    try:
        for page_no in range(1, 19):
            params = dict(params_base); params["pageToken"] = next_token
            raw = request(plan["continuation_query"]["endpoint"], params, "discovery", "application/json")
            page = json.loads(raw); rows = page.get("studies")
            if not isinstance(rows, list) or len(rows) > 1000:
                raise RuntimeError("continuation page exceeds 1,000-row bound")
            counters["studies"] += len(rows)
            if counters["studies"] > 18000: raise RuntimeError("continuation study cap exceeded")
            page_ncts = [frozen._study_nct(row) for row in rows]
            if len(page_ncts) != len(set(page_ncts)) or set(page_ncts) & all_discovered_ncts:
                raise RuntimeError("duplicate NCT across continuation pages")
            # Newly observed identities join exclusions before selection, even
            # when they are ineligible, so later pages cannot reuse them.
            page_pmids = {pmid for row in rows for pmid in frozen._result_pmids(row)}
            new_candidate_count = len(set(candidates) - parent_ncts)
            candidate_slots = int(budgets["max_results_records"]) - new_candidate_count
            if candidate_slots > 0:
                selected = continuation.select_continuation_page(rows, page_number=page_no,
                    response_sha256=_sha(raw), excluded_nct_ids=excluded_ncts | all_discovered_ncts,
                    excluded_pmids=excluded_pmids | all_discovered_pmids,
                    already_selected_nct_ids=set(candidates), already_selected_pmids=set(counters["unique_pmids"]),
                    already_selected_result_edges=counters["result_edges"], max_candidates=candidate_slots)
            else:
                selected = {"candidates": []}
            selection_receipt = {"schema": "clinifact-ctgov-continuation-page-selection-v1",
                "plan_sha256": plan["plan_sha256"], "page_number": page_no,
                "response_sha256": _sha(raw), "rows": len(rows),
                "candidate_cap_exhausted": candidate_slots <= 0,
                "candidates": selected.get("candidates", []),
                "exclusions": selected.get("exclusions", []),
                "cap_stop": selected.get("cap_stop"),
                "labels_generated": False}
            selection_receipt["selection_sha256"] = _sha(_canonical(selection_receipt))
            capture._write_new(output_dir / "selection" / f"page-{page_no:02d}.json",
                               _canonical(selection_receipt))
            for candidate in selected["candidates"]:
                nct = candidate["nct_id"]
                if nct in candidates: raise RuntimeError("candidate NCT repeated")
                candidates[nct] = candidate
                all_discovered_pmids.update(candidate["result_pmids"])
            all_discovered_ncts.update(page_ncts)
            all_discovered_pmids.update(page_pmids)
            counters["result_edges"] += sum(len(c["result_pmids"]) for c in selected["candidates"])
            counters["unique_pmids"].update(pmid for c in selected["candidates"] for pmid in c["result_pmids"])
            if counters["result_edges"] > int(budgets["max_result_edges_including_saved_capture"]):
                raise RuntimeError("combined RESULT edge cap exceeded")
            if len(counters["unique_pmids"]) > int(budgets["max_unique_pmids_including_saved_capture"]):
                raise RuntimeError("combined unique PMID cap exceeded")
            if selected.get("cap_stop"):
                # The current response is durable. Preserve the deterministic
                # source-order prefix and stop discovery before the first
                # candidate that would cross a frozen aggregate ceiling.
                stop_reason = selected["cap_stop"]["reason"]
                break
            next_token = page.get("nextPageToken")
            if not next_token:
                stop_reason = "server_exhausted"; break
            if not isinstance(next_token, str): raise RuntimeError("server cursor is not a string")
        else:
            # A planned cap is a normal stop; downstream proceeds without an
            # extra discovery request or cursor replay.
            stop_reason = "planned_page_cap_with_unconsumed_server_cursor"

        candidate_ncts = sorted(candidates)
        if len(candidate_ncts) > 1500 or len(set(candidate_ncts) - parent_ncts) > int(budgets["max_results_records"]):
            raise RuntimeError("combined ResultsSection record cap exceeded")
        result_studies: dict[str, Mapping[str, Any]] = {}
        max_batch = int(budgets["max_nct_ids_per_results_batch"])
        for start in range(0, len(candidate_ncts), max_batch):
            batch = candidate_ncts[start:start + max_batch]
            if counters["result_records"] + len(batch) > 1500:
                raise RuntimeError("combined ResultsSection physical-record cap exceeded")
            query = {"filter.ids": ",".join(batch), "fields": frozen.DETAIL_FIELDS,
                     "format": "json", "pageSize": str(len(batch)), "countTotal": "true"}
            raw = request(plan["continuation_query"]["endpoint"], query, "results", "application/json")
            resp = json.loads(raw); rows = resp.get("studies")
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise RuntimeError("ResultsSection count differs from requested candidate IDs")
            for row in rows:
                nct = frozen._study_nct(row)
                if nct not in batch or nct in result_studies: raise RuntimeError("ResultsSection NCT mismatch")
                result_studies[nct] = row
            counters["result_records"] += len(rows)

        pmids = sorted(counters["unique_pmids"], key=lambda x: (len(x), x))
        if len(pmids) > int(budgets["max_unique_pmids_including_saved_capture"]):
            raise RuntimeError("combined PubMed PMID cap exceeded")
        pubmed: dict[str, dict[str, Any]] = {}
        max_pmids = int(budgets["max_pmids_per_pubmed_batch"])
        for start in range(0, len(pmids), max_pmids):
            batch = pmids[start:start + max_pmids]
            raw = request(PUBMED_ENDPOINT, {"db":"pubmed", "id":",".join(batch), "retmode":"xml"},
                          "pubmed", "application/xml")
            pubmed.update(capture._parse_pubmed(raw))
        capture_hash = _sha(_canonical({"records": {nct: result_studies[nct] for nct in sorted(result_studies)}}))
        pubmed_hash = _sha(_canonical(pubmed))
        bundle = capture._build_unlabeled_bundle(result_studies, pubmed, verified["parent_plan"], capture_hash, pubmed_hash)
        capture._write_new(output_dir / "unlabeled-source-bundle.json", _canonical(bundle))
        components = int(bundle["capacity_gate"]["independent_component_count"])
        summary = {"schema": EXEC_SCHEMA, "state": "complete_capture_only",
            "plan_sha256": plan["plan_sha256"], "requests": counters["requests"],
            "discovery_requests": counters["discovery"], "studies": counters["studies"],
            "candidate_ncts_including_saved_v3": len(candidate_ncts),
            "result_records": counters["result_records"], "result_edges_including_saved_v3": counters["result_edges"],
            "unique_pmids_including_saved_v3": len(counters["unique_pmids"]),
            "independent_components": components, "capacity_eligible": components >= 144,
            "discovery_stop_reason": stop_reason, "labels_generated": False,
            "references_created": False, "confirmation_created": False, "release_authorized": False,
            "bytes": counters["bytes"]}
        capture._write_new(output_dir / "capture-summary.json", _canonical(summary))
        return summary
    except Exception as exc:
        summary = {"schema": EXEC_SCHEMA, "state": "terminal_failure_or_unknown",
            "plan_sha256": plan["plan_sha256"], "requests": counters["requests"],
            "discovery_requests": counters["discovery"], "studies": counters["studies"],
            "bytes": counters["bytes"], "error": f"{type(exc).__name__}: {exc}", "no_resume": True}
        try: capture._write_new(output_dir / "capture-summary.json", _canonical(summary))
        except Exception: pass
        raise
