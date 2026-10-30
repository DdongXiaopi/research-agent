"""One-shot, downstream-only executor for the frozen cap-aware CT.gov prefix.

The plan and its page-16 safe-prefix selection are immutable. This module
reconstructs and verifies the prefix from the v3/v4 raw captures before any
request, then fetches only ResultsSection and PubMed metadata. It never runs
ClinicalTrials.gov discovery and never creates labels.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping

from . import clinifact_ctgov_expansion_capture as capture
from . import clinifact_ctgov_expansion_salvage as salvage

EXEC_SCHEMA = "clinifact-ctgov-expansion-cap-aware-salvage-execution-v1"
MANIFEST_SCHEMA = "clinifact-ctgov-cap-aware-salvage-execution-manifest-v1"
APPROVAL_SCHEMA = "clinifact-ctgov-cap-aware-salvage-execution-approval-v1"
MAX_RESULTS_REQUESTS = 12
MAX_PUBMED_REQUESTS = 20
MAX_PHYSICAL_REQUESTS = 32
MAX_BATCH_SIZE = 100


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cli_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "run_clinifact_ctgov_cap_aware_salvage.py"


def _canonical_sha(value: Any) -> str:
    return capture._sha(capture._canonical(value))


def execution_manifest_body(*, plan: Mapping[str, Any], verified: Mapping[str, Any],
                            output_dir: Path, module_path: Path, cli_path: Path) -> dict[str, Any]:
    return {"schema": MANIFEST_SCHEMA, "plan_sha256": plan["plan_sha256"],
        "verified_inputs_sha256": verified["verified_inputs_sha256"],
        "safe_prefix_sha256": verified["safe_prefix_sha256"],
        "candidate_records_sha256": verified["candidate_records_sha256"],
        "selected_pmids_sha256": verified["selected_pmids_sha256"],
        "executor_sha256": capture._sha(module_path.read_bytes()),
        "cli_sha256": capture._sha(cli_path.read_bytes()),
        "output_dir": str(output_dir.resolve()),
        "discovery_requests": 0, "results_requests_max": MAX_RESULTS_REQUESTS,
        "pubmed_requests_max": MAX_PUBMED_REQUESTS,
        "physical_requests_max": MAX_PHYSICAL_REQUESTS,
        "retries": 0, "redirects": 0, "labels_generated": False}


def validate_approval(approval: Mapping[str, Any], *, manifest: Mapping[str, Any],
                      approval_path: Path) -> None:
    body = {k: v for k, v in approval.items() if k != "approval_sha256"}
    if (approval.get("schema") != APPROVAL_SCHEMA or approval.get("status") != "approved"
            or approval.get("dispatch_authorized") is not True
            or approval.get("approval_sha256") != _canonical_sha(body)
            or approval.get("execution_manifest_sha256") != _canonical_sha(manifest)
            or any(approval.get(k) != manifest.get(k) for k in (
                "plan_sha256", "verified_inputs_sha256", "safe_prefix_sha256",
                "candidate_records_sha256", "selected_pmids_sha256", "executor_sha256",
                "cli_sha256", "output_dir", "discovery_requests", "results_requests_max",
                "pubmed_requests_max", "physical_requests_max", "retries", "redirects"))
            or not approval.get("approved_by") or not approval.get("approval_statement")):
        raise ValueError("execution approval does not bind the frozen plan, inputs, code and output")


def validate_manifest(*, manifest: Mapping[str, Any], plan: Mapping[str, Any],
                      verified: Mapping[str, Any], output_dir: Path,
                      module_path: Path, cli: Path) -> None:
    expected = execution_manifest_body(plan=plan, verified=verified,
        output_dir=output_dir, module_path=module_path, cli_path=cli)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"downstream execution manifest mismatch: {key}")
    if plan.get("downstream_only", {}).get("discovery_requests") != 0:
        raise ValueError("frozen plan does not prohibit discovery requests")
    if plan.get("downstream_only", {}).get("no_discovery_replay") is not True:
        raise ValueError("frozen plan does not prohibit discovery replay")
    if int(plan.get("downstream_only", {}).get("results_batches_max", 999)) > MAX_RESULTS_REQUESTS:
        raise ValueError("ResultsSection plan exceeds the 12-request downstream cap")
    if int(plan.get("downstream_only", {}).get("pubmed_batches_max", 999)) > MAX_PUBMED_REQUESTS:
        raise ValueError("PubMed plan exceeds the 20-request downstream cap")
    if int(plan.get("downstream_only", {}).get("max_future_physical_requests", 999)) > MAX_PHYSICAL_REQUESTS:
        raise ValueError("downstream plan exceeds the 32-request total cap")


def _request(*, output_dir: Path, plan: Mapping[str, Any], manifest: Mapping[str, Any],
             index: int, phase: str, endpoint: str, params: Mapping[str, str], accept: str,
             cap: int, timeout: int,
             transport: Callable[..., tuple[int, Mapping[str, str], bytes]]) -> bytes:
    if phase not in {"results", "pubmed"}:
        raise ValueError("v5 downstream executor permits only ResultsSection and PubMed")
    if index > MAX_PHYSICAL_REQUESTS:
        raise RuntimeError("physical request ceiling exceeded")
    query = urllib.parse.urlencode(params, safe=",")
    full_url = endpoint + "?" + query
    record = {"phase": phase, "method": "GET", "url": full_url,
        "params": dict(params), "timeout_seconds": timeout,
        "response_cap_bytes": cap, "plan_sha256": plan["plan_sha256"],
        "execution_manifest_sha256": _canonical_sha(manifest)}
    wal_path = capture._reserve(output_dir, index, record)
    wal = capture._read_json(wal_path)
    try:
        status, headers, body = transport(full_url, timeout=timeout, cap=cap, accept=accept)
    except Exception as exc:
        wal.update({"state": "outcome_unknown", "terminal": True,
                    "error": f"{type(exc).__name__}: {exc}"})
        try:
            capture._finish_attempt(wal_path, wal)
        except Exception:
            capture._mark_unknown_best_effort(output_dir, wal_path, wal,
                "transport outcome unknown and ledger persistence failed")
        raise RuntimeError("downstream request outcome is ambiguous; capture is terminal") from exc

    truncated = len(body) > cap
    raw = body[:cap] if truncated else body
    try:
        raw_path = capture._persist_raw(output_dir, index, raw, error=status != 200)
    except Exception as exc:
        wal.update({"state": "outcome_unknown", "terminal": True, "status": status,
                    "response_sha256": capture._sha(raw),
                    "persistence_error": f"{type(exc).__name__}: {exc}"})
        try:
            capture._finish_attempt(wal_path, wal)
        except Exception:
            capture._mark_unknown_best_effort(output_dir, wal_path, wal,
                "raw persistence failed after dispatch")
        raise RuntimeError("raw response persistence failed; capture is terminal") from exc

    wal.update({"state": "captured" if status == 200 and not truncated else "failed",
        "status": status, "headers": dict(headers), "stored_bytes": len(raw),
        "response_sha256": capture._sha(raw),
        "raw_file": str(raw_path.relative_to(output_dir)), "truncated": truncated,
        "terminal": status != 200 or truncated})
    try:
        capture._finish_attempt(wal_path, wal)
    except Exception as exc:
        capture._mark_unknown_best_effort(output_dir, wal_path, wal,
            f"receipt persistence failed: {type(exc).__name__}: {exc}")
        raise RuntimeError("response captured but WAL receipt did not persist; terminal") from exc
    if status != 200 or truncated:
        raise RuntimeError(f"{phase} response failed the status/size gate")
    return raw


def execute_downstream(*, verified: Mapping[str, Any], manifest: Mapping[str, Any],
                       approval: Mapping[str, Any], approval_path: Path,
                       output_dir: Path, root: Path,
                       transport: Callable[..., tuple[int, Mapping[str, str], bytes]] = capture.urllib_transport,
                       now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)) -> dict[str, Any]:
    plan: Mapping[str, Any] = verified["plan"]
    module_path = Path(__file__).resolve()
    cli = cli_path()
    validate_manifest(manifest=manifest, plan=plan, verified=verified,
        output_dir=output_dir, module_path=module_path, cli=cli)
    validate_approval(approval, manifest=manifest, approval_path=approval_path)
    try:
        output_dir.resolve().relative_to((root.resolve() / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("downstream output must be under the exclusive .workbuddy tree") from exc
    if output_dir.exists():
        raise FileExistsError("downstream capture already exists; resume/replay is forbidden")

    # Preserve the source-order selection in the verified receipt, but send
    # canonical request batches.  Selection order is provenance; API request
    # order is only a deterministic transport detail.
    candidates = sorted(verified["candidate_records"], key=lambda row: str(row["nct_id"]))
    pmids = list(verified["selected_pmids"])
    expected_candidates = int(plan["downstream_only"]["results_records"])
    expected_pmids = int(plan["downstream_only"]["pubmed_pmids"])
    if len(candidates) != expected_candidates or len(pmids) != expected_pmids:
        raise ValueError("reconstructed page-16 safe-prefix counts differ from the frozen plan")
    ncts = [str(row["nct_id"]) for row in candidates]
    if len(ncts) != len(set(ncts)) or ncts != sorted(ncts):
        raise ValueError("selected NCT records are duplicated or not in canonical order")
    if len(pmids) != len(set(pmids)) or pmids != sorted(pmids, key=lambda x: (len(x), x)):
        raise ValueError("selected PMID records are duplicated or not in canonical order")
    result_batches = [ncts[i:i + MAX_BATCH_SIZE] for i in range(0, len(ncts), MAX_BATCH_SIZE)]
    pubmed_batches = [pmids[i:i + MAX_BATCH_SIZE] for i in range(0, len(pmids), MAX_BATCH_SIZE)]
    if len(result_batches) > MAX_RESULTS_REQUESTS or len(pubmed_batches) > MAX_PUBMED_REQUESTS:
        raise ValueError("safe-prefix batch count exceeds its frozen physical request limits")
    if len(result_batches) + len(pubmed_batches) > MAX_PHYSICAL_REQUESTS:
        raise ValueError("combined downstream request count exceeds 32")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "raw").mkdir(); (output_dir / "ledger").mkdir()
    for name, value in (("plan.json", plan), ("execution-manifest.json", manifest),
        ("execution-approval.json", approval),
        ("run.json", {"schema": EXEC_SCHEMA, "plan_sha256": plan["plan_sha256"],
            "state": "ready", "dispatch_authorized": True, "discovery_requests": 0,
            "labels_generated": False, "replay_permitted": False})):
        capture._write_new(output_dir / name, capture._canonical(value))

    cap = int(plan.get("response_policy", {}).get("max_bytes_each", 16 * 1024 * 1024))
    timeout = int(plan.get("response_policy", {}).get("timeout_seconds", 45))
    counters = {"requests": 0, "results": 0, "pubmed": 0,
                "result_bytes": 0, "pubmed_bytes": 0}
    result_studies: dict[str, Mapping[str, Any]] = {}
    crosswalk: dict[str, dict[str, Any]] = {}
    try:
        for batch in result_batches:
            params = {"filter.ids": ",".join(batch),
                "fields": salvage.frozen.DETAIL_FIELDS, "format": "json",
                "pageSize": str(len(batch)), "countTotal": "true"}
            idx = counters["requests"] + 1
            counters["requests"] += 1; counters["results"] += 1
            raw = _request(output_dir=output_dir, plan=plan, manifest=manifest,
                index=idx, phase="results", endpoint=plan.get("endpoint", "https://clinicaltrials.gov/api/v2/studies"),
                params=params, accept="application/json", cap=cap, timeout=timeout, transport=transport)
            counters["result_bytes"] += len(raw)
            payload = json.loads(raw)
            rows = payload.get("studies")
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise RuntimeError("ResultsSection returned a different count than the frozen NCT batch")
            for row in rows:
                nct = salvage.frozen._study_nct(row)
                if nct not in batch or nct in result_studies:
                    raise RuntimeError("ResultsSection NCT identity mismatch or duplicate")
                result_studies[nct] = row
        if set(result_studies) != set(ncts):
            raise RuntimeError("ResultsSection did not return the complete frozen candidate set")

        for batch in pubmed_batches:
            idx = counters["requests"] + 1
            counters["requests"] += 1; counters["pubmed"] += 1
            raw = _request(output_dir=output_dir, plan=plan, manifest=manifest,
                index=idx, phase="pubmed", endpoint=capture.PUBMED_ENDPOINT,
                params={"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
                accept="application/xml", cap=cap, timeout=timeout, transport=transport)
            counters["pubmed_bytes"] += len(raw)
            parsed = capture._parse_pubmed(raw)
            if set(parsed) - set(batch):
                raise RuntimeError("PubMed returned an identity outside the frozen PMID batch")
            if set(parsed) & set(crosswalk):
                raise RuntimeError("PubMed returned a duplicate PMID across physical batches")
            crosswalk.update(parsed)

        capture_hash = capture._sha(capture._canonical({"records": {
            nct: result_studies[nct] for nct in sorted(result_studies)}}))
        pubmed_hash = capture._sha(capture._canonical(crosswalk))
        parent_plan = verified["parent_plan"]
        bundle = capture._build_unlabeled_bundle(result_studies, crosswalk,
            parent_plan, capture_hash, pubmed_hash)
        capture._write_new(output_dir / "unlabeled-source-bundle.json", capture._canonical(bundle))
        summary = {"schema": EXEC_SCHEMA, "state": "complete_capture_only",
            "plan_sha256": plan["plan_sha256"], "requests": counters["requests"],
            "discovery_requests": 0, "results_requests": counters["results"],
            "pubmed_requests": counters["pubmed"], "candidate_ncts": len(ncts),
            "result_records": len(result_studies), "selected_pmids": len(pmids),
            "pubmed_records_returned": len(crosswalk),
            "independent_components": int(bundle["capacity_gate"]["independent_component_count"]),
            "capacity_eligible": int(bundle["capacity_gate"]["independent_component_count"]) >= 144,
            "bytes": {"results": counters["result_bytes"], "pubmed": counters["pubmed_bytes"]},
            "labels_generated": False, "references_created": False,
            "confirmation_created": False, "release_authorized": False,
            "replay_permitted": False, "completed_at_utc": now().isoformat()}
        capture._write_new(output_dir / "capture-summary.json", capture._canonical(summary))
        return summary
    except Exception as exc:
        summary = {"schema": EXEC_SCHEMA, "state": "terminal_failure_or_unknown",
            "plan_sha256": plan["plan_sha256"], "requests": counters["requests"],
            "discovery_requests": 0, "results_requests": counters["results"],
            "pubmed_requests": counters["pubmed"], "error": f"{type(exc).__name__}: {exc}",
            "replay_permitted": False, "no_resume": True}
        try:
            capture._write_new(output_dir / "capture-summary.json", capture._canonical(summary))
        except Exception:
            pass
        raise


def verify_v5_inputs(*, plan_dir: Path, v3_plan_path: Path, v3_approval_path: Path,
                     v3_capture_dir: Path, bindings_path: Path, v4_plan_path: Path,
                     v4_capture_dir: Path, v4_execution_manifest_path: Path,
                     v4_execution_approval_path: Path, v4_root_approval_path: Path,
                     root: Path) -> dict[str, Any]:
    plan_path = plan_dir / "plan.json"
    plan = _read(plan_path)
    preflight = _read(plan_dir / "preflight.json")
    if preflight.get("status") != "passed_zero_call_salvage_preflight":
        raise ValueError("the frozen salvage zero-call preflight did not pass")
    rebuilt, details = salvage.build_offline_salvage(v3_plan_path=v3_plan_path,
        v3_approval_path=v3_approval_path, v3_capture_dir=v3_capture_dir,
        bindings_path=bindings_path, v4_plan_path=v4_plan_path,
        v4_capture_dir=v4_capture_dir,
        v4_execution_manifest_path=v4_execution_manifest_path,
        v4_execution_approval_path=v4_execution_approval_path,
        v4_root_approval_path=v4_root_approval_path,
        output_dir=Path(plan["output_dir"]),
        preflight_cli_path=Path(salvage.__file__).resolve().parents[2] /
            "scripts" / "prepare_clinifact_ctgov_cap_aware_salvage.py", root=root)
    if rebuilt != plan or details["preflight"] != preflight:
        raise ValueError("frozen v5 plan/preflight differ from independent source reconstruction")
    saved_selection = _read(plan_dir / "page-16-safe-prefix-selection.json")
    if saved_selection != details["page16_receipt"]:
        raise ValueError("saved page-16 safe-prefix selection differs from source reconstruction")
    candidate_digest = _canonical_sha(details["candidate_records"])
    pmid_digest = _canonical_sha(details["selected_pmids"])
    verified_inputs_sha = _canonical_sha({"plan_sha256": plan["plan_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "verified_source_projection": details["verified_inputs"],
        "page16_selection_sha256": saved_selection["selection_sha256"],
        "candidate_records_sha256": candidate_digest, "selected_pmids_sha256": pmid_digest})
    return {"plan": plan, "parent_plan": _read(v3_plan_path),
        "candidate_records": details["candidate_records"],
        "selected_pmids": details["selected_pmids"],
        "verified_inputs_sha256": verified_inputs_sha,
        "safe_prefix_sha256": saved_selection["selection_sha256"],
        "candidate_records_sha256": candidate_digest,
        "selected_pmids_sha256": pmid_digest}


def write_pending_manifest(*, plan: Mapping[str, Any], verified: Mapping[str, Any],
                           plan_dir: Path) -> dict[str, Any]:
    manifest = execution_manifest_body(plan=plan, verified=verified,
        output_dir=Path(plan["output_dir"]), module_path=Path(__file__).resolve(),
        cli_path=cli_path())
    pending_approval = {**manifest, "schema": APPROVAL_SCHEMA,
        "status": "pending_root_approval", "dispatch_authorized": False,
        "approved_by": None, "approved_at_utc": None, "approval_statement": None,
        "scope": "ResultsSection then PubMed from the frozen page-16 safe prefix only; 0 discovery, 0 retries/redirects, 0 labels"}
    pending_approval["execution_manifest_sha256"] = _canonical_sha(manifest)
    pending_approval["approval_sha256"] = _canonical_sha(pending_approval)
    capture._write_new(plan_dir / "downstream-execution-manifest.pending.json",
        capture._canonical(manifest))
    capture._write_new(plan_dir / "downstream-execution-approval.pending.json",
        capture._canonical(pending_approval))
    return manifest
