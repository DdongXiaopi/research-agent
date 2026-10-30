"""Fail-closed calibration attestations for required semantic grounding gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from ..util import read_json, sha256_file

_METRIC_TOLERANCES = {
    "citation_support_rate_pp": "citation_support_rate",
    "answer_evidence_consistency_rate_pp": "answer_evidence_consistency_rate",
}
_GROUNDING_METRICS = set(_METRIC_TOLERANCES.values())
_PRODUCTION_TASK_CONTRACT = "selected_answer_support_v1"


def required_grounding_metrics(cfg: Dict[str, Any]) -> set[str]:
    accept = ((cfg.get("gates") or {}).get("accept") or {})
    tolerance = accept.get("tolerance") or {}
    required = set(tolerance.get("required") or [])
    metrics = set()
    for tolerance_key, metric in _METRIC_TOLERANCES.items():
        if tolerance_key in tolerance and (
                tolerance_key in required or metric in required
                or tolerance.get("strict") is True):
            metrics.add(metric)
    minimum = accept.get("minimum") or {}
    metrics.update(_GROUNDING_METRICS.intersection(minimum))
    return metrics


def _scope_for_report(report: Dict[str, Any]) -> str:
    inputs_path = Path(report["job_run_dir"]) / "inputs.json"
    packet = read_json(inputs_path)["packet"]
    evidence_counts = [len(case.get("evidence") or [])
                       for case in packet.get("cases") or []]
    if not evidence_counts:
        raise ValueError("grounding calibration has no evidence cases")
    if all(count == 1 for count in evidence_counts):
        return "single_doc"
    if all(1 <= count <= 3 for count in evidence_counts) and max(evidence_counts) == 3:
        return "top3"
    raise ValueError("grounding calibration evidence scope is neither single_doc nor top3")


def require_grounding_calibration(cfg: Dict[str, Any], datasets: Iterable[str]) -> Dict[str, Any]:
    """Validate exact, passed calibration receipts before required-gate scoring.

    A calibration can only attest the exact dataset named in its report and the
    actual evidence multiplicity in its private packet. Runtime model, prompt,
    source, scoring protocol, thresholds and completed-job receipts are checked
    by ``verify_report``.
    """
    metrics = required_grounding_metrics(cfg)
    if not metrics:
        return {"required": False, "metrics": [], "attestations": []}
    semantic = ((cfg.get("evaluation") or {}).get("semantic_citation") or {})
    if semantic.get("enabled") is not True or semantic.get("mode", "answer") != "proposition":
        raise ValueError("required semantic grounding gate needs enabled proposition judge calibration")
    expected_scope = "top3" if semantic.get("check_top3") is True else "single_doc"
    configured = semantic.get("calibration_attestations") or []
    from ..eval.grounding_calibration import verify_report
    from ..eval.grounding_calibration import runtime_scorer_sha256

    def expected_scope(metric: str) -> str:
        if metric == "answer_evidence_consistency_rate":
            if semantic.get("check_top3") is not True:
                raise ValueError("answer_evidence_consistency_rate requires check_top3=true")
            return "top3"
        top_k = (cfg.get("retrieval") or {}).get("top_k", 3)
        if top_k == 1:
            return "single_doc"
        if top_k == 3:
            return "top3"
        raise ValueError("citation_support_rate calibration only supports retrieval top_k 1 or 3")

    verified = []
    target_datasets = sorted(set(str(ds) for ds in datasets))
    if not target_datasets:
        raise ValueError("required semantic grounding gate has no target datasets to attest")
    for dataset in target_datasets:
        for metric in sorted(metrics):
            entry = next((row for row in configured
                          if isinstance(row, dict) and row.get("dataset") == dataset
                          and metric in (row.get("metrics") or [])), None)
            if entry is None:
                raise ValueError(f"required semantic grounding gate lacks calibration for dataset {dataset}/{metric}")
            report_path = Path(str(entry.get("report_path", ""))).expanduser()
            expected_sha = str(entry.get("report_sha256", ""))
            if not report_path.is_file() or sha256_file(report_path) != expected_sha:
                raise ValueError(f"grounding calibration report is absent or stale for dataset {dataset}")
            # The new offline format reproduces a selected-answer measurement;
            # it does not bind a completed provider job or a production runtime
            # dispatcher. Never route it through the older claim-veracity
            # verifier or treat its calibration_pass flag as an attestation.
            from ..eval.grounding_selected_answer_calibration import REPORT_SCHEMA
            if read_json(report_path).get("schema") == REPORT_SCHEMA:
                from ..eval.grounding_selected_answer_verification import production_attestation_rejection
                raise ValueError(production_attestation_rejection())
            scope = expected_scope(metric)
            if entry.get("scope") != scope:
                raise ValueError(f"grounding calibration scope mismatch for dataset {dataset}: expected {scope}")
            if entry.get("task_contract") != _PRODUCTION_TASK_CONTRACT:
                raise ValueError(f"grounding calibration task contract mismatch for dataset {dataset}/{metric}")
            if entry.get("runtime_scorer_sha256") != runtime_scorer_sha256():
                raise ValueError(f"grounding calibration production scorer identity mismatch for dataset {dataset}")
            report = verify_report(str(report_path), cfg, dataset)
            if (entry.get("packet_sha256") != report.get("packet_sha256")
                    or entry.get("runtime_scorer_sha256") != report.get("runtime_scorer_sha256")):
                raise ValueError(f"grounding calibration cohort/scorer identity mismatch for dataset {dataset}")
            if report.get("task_contract") != _PRODUCTION_TASK_CONTRACT:
                raise ValueError(f"grounding calibration task target mismatch for dataset {dataset}/{metric}")
            actual_scope = _scope_for_report(report)
            if (actual_scope != scope or actual_scope != entry["scope"]
                    or report.get("evidence_scope") != actual_scope):
                raise ValueError(f"grounding calibration evidence scope mismatch for dataset {dataset}")
            verified.append({"dataset": dataset, "metric": metric, "scope": actual_scope,
                             "task_contract": report["task_contract"],
                             "report_sha256": expected_sha,
                             "model_protocol_sha256": report["model_protocol_sha256"],
                             "prompt_version": report["prompt_version"],
                             "grounding_source_sha256": report["grounding_source_sha256"],
                             "runtime_scorer_sha256": report["runtime_scorer_sha256"],
                             "packet_sha256": report["packet_sha256"],
                             "report_path": str(report_path.resolve())})
    return {"required": True, "metrics": sorted(metrics), "attestations": verified}
