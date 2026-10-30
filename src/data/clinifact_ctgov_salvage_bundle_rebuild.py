"""Offline rebuild of a corrected v2 CliniFact bundle from v5 receipts.

This tool is intentionally separate from the already executed v5 runner. It
audits every captured downstream request and derives a new versioned bundle
without making requests or replacing the preserved v1 artifact.
"""
from __future__ import annotations

import json
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from . import clinifact_ctgov_expansion_capture as capture
from . import clinifact_ctgov_expansion_continuation as continuation
from . import clinifact_ctgov_expansion_plan as frozen
from . import clinifact_ctgov_expansion_salvage as salvage
from . import clinifact_ctgov_expansion_salvage_execute as runner
from . import clinifact_ctgov_pubmed_adapter_v1 as adapter

REBUILD_SCHEMA = "clinifact-ctgov-salvage-bundle-rebuild-v1"
V2_BUNDLE_NAME = "unlabeled-source-bundle.v2.json"
RECEIPT_NAME = "bundle-rebuild-v2-receipt.json"
RECEIPT_HASH_NAME = "bundle-rebuild-v2-receipt.sha256"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    return capture._sha(path.read_bytes())


def _study_nct(study: Mapping[str, Any]) -> str:
    return salvage.frozen._study_nct(study)


def _verify_v3_bindings_at_capture_time(plan: Mapping[str, Any], plan_path: Path,
                                        bindings: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Verify immutable source inputs while preserving the adapter hash captured in v3.

    The v3 selection was computed by the registry selector, not the downstream
    source adapter. The adapter was subsequently corrected for the exact CT.gov
    detail projection. For this offline rebuild only, all input and selector
    hashes are rechecked while the plan's historical adapter SHA is retained
    as provenance and the current adapter SHA is recorded in the new receipt.
    """
    frozen.validate_plan(plan)
    workbuddy = (root.resolve() / ".workbuddy").resolve()
    try:
        plan_path.resolve().relative_to(workbuddy)
    except ValueError as exc:
        raise ValueError("frozen v3 plan must stay inside project .workbuddy") from exc
    expected = plan["input_bindings"]
    for key in ("oas", "oas_receipt", "parent_page", "parent_receipt", "exposure_inventory"):
        path = Path(bindings[key]).resolve()
        try:
            path.relative_to(workbuddy)
        except ValueError as exc:
            raise ValueError(f"{key} input must stay inside .workbuddy") from exc
    checks = {
        "oas": (expected["oas_sha256"], Path(bindings["oas"])),
        "oas_receipt": (expected["oas_receipt_sha256"], Path(bindings["oas_receipt"])),
        "parent_page": (expected["parent_page_sha256"], Path(bindings["parent_page"])),
        "parent_receipt": (expected["parent_receipt_sha256"], Path(bindings["parent_receipt"])),
        "exposure_inventory": (expected["exposure_inventory_sha256"], Path(bindings["exposure_inventory"])),
        "expansion_module": (expected["expansion_module_sha256"], Path(frozen.__file__)),
    }
    seen = bindings.get("seen_census_pages")
    expected_seen = expected["seen_census_pages"]
    if not isinstance(seen, list) or len(seen) != len(expected_seen):
        raise ValueError("all frozen prior census page bindings are required")
    for index, (record, expected_record) in enumerate(zip(seen, expected_seen)):
        page, receipt = Path(record["page"]).resolve(), Path(record["receipt"]).resolve()
        for bound in (page, receipt):
            try:
                bound.relative_to(workbuddy)
            except ValueError as exc:
                raise ValueError("prior census inputs must stay inside .workbuddy") from exc
        if frozen.sha_file(page) != expected_record["page_sha256"]:
            raise ValueError(f"prior census page {index + 1} hash mismatch")
        if frozen.sha_file(receipt) != expected_record["receipt_sha256"]:
            raise ValueError(f"prior census receipt {index + 1} hash mismatch")
    verified: dict[str, Any] = {}
    for name, (wanted, path) in checks.items():
        actual = frozen.sha_file(path.resolve())
        if actual != wanted:
            raise ValueError(f"{name} hash mismatch")
        verified[name] = actual
    old_adapter_sha = str(expected.get("source_adapter_sha256") or "")
    if len(old_adapter_sha) != 64 or any(ch not in "0123456789abcdef" for ch in old_adapter_sha):
        raise ValueError("v3 plan does not contain a valid historical source-adapter hash")
    verified["source_adapter"] = old_adapter_sha
    verified["seen_census_pages"] = [dict(row) for row in expected_seen]
    plan_file_sha = frozen.sha_file(plan_path.resolve())
    if plan_file_sha != capture._sha(capture._canonical(plan)):
        raise ValueError("v3 plan file is not the canonical frozen plan")
    verified["plan_file_sha256"] = plan_file_sha
    return verified


def verify_v5_inputs_for_rebuild(**kwargs: Any) -> dict[str, Any]:
    """Reverify v3/v4/v5 source evidence with the post-capture adapter upgrade recorded."""
    original = continuation.verify_bound_inputs
    continuation.verify_bound_inputs = _verify_v3_bindings_at_capture_time
    try:
        return runner.verify_v5_inputs(**kwargs)
    finally:
        continuation.verify_bound_inputs = original


def _expected_requests(plan: Mapping[str, Any], verified: Mapping[str, Any]) -> list[dict[str, Any]]:
    ncts = sorted({str(row["nct_id"]) for row in verified["candidate_records"]})
    pmids = list(verified["selected_pmids"])
    if (len(ncts) != len(set(ncts)) or len(pmids) != len(set(pmids))
            or pmids != sorted(pmids, key=lambda x: (len(x), x))):
        raise ValueError("frozen selected NCT/PMID identities are duplicate or noncanonical")
    downstream = plan["downstream_only"]
    if (len(ncts) != int(downstream["results_records"])
            or len(pmids) != int(downstream["pubmed_pmids"])):
        raise ValueError("source reconstruction differs from planned candidate identity counts")
    endpoint = plan.get("endpoint", "https://clinicaltrials.gov/api/v2/studies")
    expected: list[dict[str, Any]] = []
    for offset in range(0, len(ncts), 100):
        batch = ncts[offset:offset + 100]
        params = {"filter.ids": ",".join(batch), "fields": salvage.frozen.DETAIL_FIELDS,
            "format": "json", "pageSize": str(len(batch)), "countTotal": "true"}
        expected.append({"phase": "results", "endpoint": endpoint, "params": params,
            "accept": "application/json", "batch": batch})
    for offset in range(0, len(pmids), 100):
        batch = pmids[offset:offset + 100]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        expected.append({"phase": "pubmed", "endpoint": capture.PUBMED_ENDPOINT,
            "params": params, "accept": "application/xml", "batch": batch})
    if (sum(x["phase"] == "results" for x in expected) > 12
            or sum(x["phase"] == "pubmed" for x in expected) > 20
            or len(expected) > 32):
        raise ValueError("frozen candidate identities exceed the physical request plan")
    return expected


def _validate_request_record(*, record: Mapping[str, Any], expected: Mapping[str, Any],
                             index: int, plan_sha256: str, manifest_sha256: str,
                             timeout: int, cap: int) -> None:
    if (record.get("request_index") != index or record.get("phase") != expected["phase"]
            or record.get("method") != "GET" or record.get("state") != "captured"
            or record.get("status") != 200 or record.get("truncated") is not False
            or record.get("retries") != 0 or record.get("redirects") != 0
            or record.get("terminal") is not False or record.get("plan_sha256") != plan_sha256
            or record.get("execution_manifest_sha256") != manifest_sha256
            or record.get("timeout_seconds") != timeout
            or record.get("response_cap_bytes") != cap
            or record.get("params") != expected["params"]):
        raise ValueError(f"captured downstream WAL entry {index} differs from frozen request/response contract")
    url = urllib.parse.urlsplit(str(record.get("url") or ""))
    if url.scheme != "https" or url.netloc != urllib.parse.urlsplit(expected["endpoint"]).netloc:
        raise ValueError(f"captured downstream WAL entry {index} has an unexpected endpoint")
    actual_params = urllib.parse.parse_qs(url.query, keep_blank_values=True)
    normalized_expected = {k: [str(v)] for k, v in expected["params"].items()}
    if actual_params != normalized_expected:
        raise ValueError(f"captured downstream WAL URL {index} does not match its frozen query parameters")


def rebuild_from_captured_receipts(*, plan_dir: Path, capture_dir: Path,
                                   verified: Mapping[str, Any], root: Path,
                                   module_path: Path | None = None,
                                   cli_path: Path | None = None) -> dict[str, Any]:
    """Validate the complete 32-request capture and write a v2 bundle once."""
    plan: Mapping[str, Any] = verified["plan"]
    module_path = module_path or Path(__file__).resolve()
    cli_path = cli_path or Path(__file__).resolve().parents[2] / "scripts" / "rebuild_clinifact_ctgov_salvage_bundle.py"
    capture_dir = capture_dir.resolve()
    plan_dir = plan_dir.resolve()
    try:
        capture_dir.relative_to((root.resolve() / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("captured input must stay under project .workbuddy") from exc

    on_disk_plan = _read(capture_dir / "plan.json")
    if on_disk_plan != plan or capture._sha((capture_dir / "plan.json").read_bytes()) != _sha_file(plan_dir / "plan.json"):
        raise ValueError("captured plan differs from the independently verified v5 plan")
    manifest = _read(capture_dir / "execution-manifest.json")
    approval = _read(capture_dir / "execution-approval.json")
    runner.validate_manifest(manifest=manifest, plan=plan, verified=verified,
        output_dir=capture_dir, module_path=Path(runner.__file__).resolve(), cli=runner.cli_path())
    runner.validate_approval(approval, manifest=manifest,
        approval_path=capture_dir / "execution-approval.json")
    manifest_sha = capture._sha(capture._canonical(manifest))
    if approval.get("execution_manifest_sha256") != manifest_sha:
        raise ValueError("execution approval does not hash-bind the captured manifest")
    run = _read(capture_dir / "run.json")
    summary = _read(capture_dir / "capture-summary.json")
    downstream = plan["downstream_only"]
    expected_results = int(downstream["results_batches_max"])
    expected_pubmed = int(downstream["pubmed_batches_max"])
    expected_total = expected_results + expected_pubmed
    if (run.get("plan_sha256") != plan["plan_sha256"] or run.get("dispatch_authorized") is not True
            or run.get("discovery_requests") != 0 or run.get("labels_generated") is not False
            or summary.get("state") != "complete_capture_only"
            or summary.get("plan_sha256") != plan["plan_sha256"]
            or summary.get("requests") != expected_total or summary.get("discovery_requests") != 0
            or summary.get("results_requests") != expected_results
            or summary.get("pubmed_requests") != expected_pubmed):
        raise ValueError("capture run/summary is incomplete or outside the approved downstream-only scope")

    candidates = list(verified["candidate_records"])
    selected_pmids = list(verified["selected_pmids"])
    expected = _expected_requests(plan, verified)
    if len(expected) != expected_total:
        raise ValueError("captured request schedule differs from the frozen downstream batches")
    timeout = int(plan.get("response_policy", {}).get("timeout_seconds", 45))
    cap = int(plan.get("response_policy", {}).get("max_bytes_each", 16 * 1024 * 1024))
    ledger_dir, raw_dir = capture_dir / "ledger", capture_dir / "raw"
    ledgers = sorted(ledger_dir.glob("attempt-*.json"))
    raws = sorted(raw_dir.glob("request-*-response.bin"))
    if len(ledgers) != expected_total or len(raws) != expected_total:
        raise ValueError("complete downstream derivation requires every planned WAL/raw response pair")

    result_studies: dict[str, Mapping[str, Any]] = {}
    pubmed: dict[str, dict[str, Any]] = {}
    receipt_rows = []
    for index, expected_request in enumerate(expected, start=1):
        ledger_path = ledger_dir / f"attempt-{index:04d}.json"
        raw_path = raw_dir / f"request-{index:04d}-response.bin"
        if not ledger_path.is_file() or not raw_path.is_file():
            raise ValueError(f"downstream receipt pair {index} is missing")
        ledger = _read(ledger_path)
        raw = raw_path.read_bytes()
        raw_sha = capture._sha(raw)
        _validate_request_record(record=ledger, expected=expected_request,
            index=index, plan_sha256=plan["plan_sha256"], manifest_sha256=manifest_sha,
            timeout=timeout, cap=cap)
        raw_file = str(ledger.get("raw_file") or "").replace("\\", "/")
        if (len(raw) > cap or ledger.get("response_sha256") != raw_sha
                or ledger.get("stored_bytes") != len(raw)
                or raw_file != f"raw/{raw_path.name}"):
            raise ValueError(f"downstream raw response {index} does not match its durable ledger receipt")
        if expected_request["phase"] == "results":
            try:
                payload = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"ResultsSection raw response {index} is not valid JSON") from exc
            rows = payload.get("studies")
            if not isinstance(rows, list) or len(rows) != len(expected_request["batch"]):
                raise ValueError(f"ResultsSection response {index} count differs from the exact frozen batch")
            for row in rows:
                nct = _study_nct(row)
                if nct not in expected_request["batch"] or nct in result_studies:
                    raise ValueError(f"ResultsSection response {index} has unexpected/duplicate NCT identity")
                result_studies[nct] = row
        else:
            try:
                parsed = capture._parse_pubmed(raw)
            except Exception as exc:
                raise ValueError(f"PubMed raw response {index} is not valid XML") from exc
            if set(parsed) - set(expected_request["batch"]):
                raise ValueError(f"PubMed response {index} contains an unplanned PMID")
            if set(parsed) & set(pubmed):
                raise ValueError(f"PubMed response {index} repeats a PMID from another physical batch")
            pubmed.update(parsed)
        receipt_rows.append({"request_index": index, "phase": expected_request["phase"],
            "ledger_sha256": _sha_file(ledger_path), "raw_sha256": raw_sha,
            "bytes": len(raw), "planned_identities": expected_request["batch"]})

    planned_ncts = {str(row["nct_id"]) for row in candidates}
    planned_pmids = set(selected_pmids)
    if set(result_studies) != planned_ncts:
        raise ValueError("complete ResultsSection responses differ from the exact planned NCT identity set")
    if set(identity for req in expected if req["phase"] == "pubmed" for identity in req["batch"]) != planned_pmids:
        raise ValueError("PubMed request batches differ from the exact planned PMID identity set")

    old_bundle_path = capture_dir / "unlabeled-source-bundle.json"
    if not old_bundle_path.is_file():
        raise ValueError("preserved v1 source bundle is missing")
    old_bundle_sha = _sha_file(old_bundle_path)
    bundle_path = capture_dir / V2_BUNDLE_NAME
    receipt_path = capture_dir / RECEIPT_NAME
    receipt_hash_path = capture_dir / RECEIPT_HASH_NAME
    if bundle_path.exists() or receipt_path.exists() or receipt_hash_path.exists():
        raise FileExistsError("versioned v2 bundle/receipt already exists; overwrite is forbidden")

    result_capture_sha = capture._sha(capture._canonical({
        "records": {nct: result_studies[nct] for nct in sorted(result_studies)}}))
    pubmed_capture_sha = capture._sha(capture._canonical(pubmed))
    rebuilt = capture._build_unlabeled_bundle(result_studies, pubmed,
        verified["parent_plan"], result_capture_sha, pubmed_capture_sha)
    bundle_bytes = capture._canonical(rebuilt)
    exclusions = Counter(row.get("reason", "unknown") for row in rebuilt.get("exclusions", []))
    adapter_path = Path(salvage.capture.adapter.__file__).resolve()
    receipt = {"schema": REBUILD_SCHEMA, "status": "complete_offline_rebuild",
        "v1_bundle_sha256": old_bundle_sha, "v2_bundle_sha256": capture._sha(bundle_bytes),
        "v2_bundle_file": V2_BUNDLE_NAME, "v1_bundle_preserved": True,
        "plan_sha256": plan["plan_sha256"], "execution_manifest_sha256": manifest_sha,
        "execution_approval_sha256": approval["approval_sha256"],
        "verified_inputs_sha256": verified["verified_inputs_sha256"],
        "safe_prefix_sha256": verified["safe_prefix_sha256"],
        "candidate_nct_ids_sha256": capture._sha(capture._canonical(sorted(planned_ncts))),
        "planned_pmids_sha256": capture._sha(capture._canonical(sorted(planned_pmids, key=lambda x: (len(x), x)))),
        "result_capture_sha256": result_capture_sha, "pubmed_capture_sha256": pubmed_capture_sha,
        "adapter_sha256": _sha_file(adapter_path), "rebuild_module_sha256": _sha_file(module_path),
        "source_adapter_at_v3_capture_sha256": verified["parent_plan"]["input_bindings"]["source_adapter_sha256"],
        "source_adapter_current_rebuild_sha256": _sha_file(adapter_path),
        "rebuild_cli_sha256": _sha_file(cli_path),
        "calls_verified": {"total": len(expected), "discovery": 0,
            "results": expected_results, "pubmed": expected_pubmed,
            "network_calls_during_rebuild": 0, "model_calls": 0, "labels_generated": False},
        "identity_counts": {"planned_ncts": len(planned_ncts), "results_returned_ncts": len(result_studies),
            "planned_pmids": len(planned_pmids), "pubmed_returned_pmids": len(pubmed)},
        "bundle_counts": {"cases": len(rebuilt.get("cases", [])),
            "independent_components": rebuilt.get("capacity_gate", {}).get("independent_component_count", 0),
            "exclusions": dict(sorted(exclusions.items()))},
        "request_receipts": receipt_rows, "no_labels_or_release": True}
    receipt["receipt_sha256"] = capture._sha(capture._canonical(receipt))
    receipt_bytes = capture._canonical(receipt)
    # Both names are checked above, and _write_new is exclusive + fsynced.
    capture._write_new(bundle_path, bundle_bytes)
    capture._write_new(receipt_path, receipt_bytes)
    capture._write_new(receipt_hash_path, (capture._sha(receipt_bytes) + "\n").encode("ascii"))
    if _sha_file(old_bundle_path) != old_bundle_sha:
        raise RuntimeError("preserved v1 bundle changed during offline rebuild")
    return receipt

