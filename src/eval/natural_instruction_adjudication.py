"""Third-model adjudication for the historical instruction-review census.

The original natural census and its two reviewer runs are immutable.  This
module creates a new lineage that asks a third AI only about case dimensions
where the two original reviewers disagree.  It never rewrites the v1 frozen
reference and keeps the resulting reference explicitly AI-derived (not human
gold).  The resulting v2 reference may unlock the *target calibration plan*;
it does not by itself authorize candidate publication.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import ai_jobs
from . import natural_instruction_calibration as natural
from .instruction_reviewer_pair_gate import model_family
from ..util import repo_root, sha256_obj


PLAN_SCHEMA = "natural-instruction-adjudication-plan-v2"
RUN_SCHEMA = "natural-instruction-adjudication-run-v2"
FROZEN_SCHEMA = "natural-instruction-frozen-reference-v2"
MAX_TOKENS = 900

ADJUDICATION_RUBRIC = natural.RUBRIC + (
    " Two independent AI references are supplied only as fallible evidence. "
    "Resolve the listed disputed dimensions using the candidate strategy and "
    "the frozen contract. Do not import external facts. Return the exact same "
    "assertion and contract objects. Preserve uncertain when the supplied text "
    "does not support a decisive verdict."
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair(packet: dict[str, Any], run_a: Path, run_b: Path,
          *, allow_legacy_measurement_schema: bool = True) -> tuple[dict, dict, dict, dict]:
    """Verify both immutable raw runs and return their jobs plus per-case rows."""
    a = natural.verify_reference_run(
        packet, "reviewer_a", run_a,
        allow_legacy_measurement_schema=allow_legacy_measurement_schema)
    b = natural.verify_reference_run(
        packet, "reviewer_b", run_b,
        allow_legacy_measurement_schema=allow_legacy_measurement_schema)
    if a["root"].resolve() == b["root"].resolve():
        raise ValueError("natural adjudication requires two distinct reviewer run directories")
    if a["plan"]["model_id"] == b["plan"]["model_id"]:
        raise ValueError("natural adjudication requires distinct reviewer model IDs")
    if a["output"].get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("reviewer A packet binding changed")
    if b["output"].get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("reviewer B packet binding changed")
    by_a = {row["case_id"]: row for row in a["output"]["cases"]}
    by_b = {row["case_id"]: row for row in b["output"]["cases"]}
    expected = [case["case_id"] for case in packet["cases"]]
    if list(by_a) != expected or list(by_b) != expected:
        raise ValueError("reviewer case order or coverage changed")
    return a, b, by_a, by_b


def _dispute_cases(packet: dict[str, Any], by_a: dict, by_b: dict) -> list[dict[str, Any]]:
    disputes: list[dict[str, Any]] = []
    for case in packet["cases"]:
        cid = case["case_id"]
        dimensions = [dimension for dimension in ("assertion", "contract")
                      if by_a[cid]["dimensions"][dimension]["verdict"]
                      != by_b[cid]["dimensions"][dimension]["verdict"]]
        if dimensions:
            disputes.append({
                "case_id": cid,
                "disputed_dimensions": dimensions,
                "reviewer_a": by_a[cid]["dimensions"],
                "reviewer_b": by_b[cid]["dimensions"],
            })
    return disputes


def _inputs(packet: dict[str, Any], a: dict, b: dict,
            disputes: list[dict[str, Any]], target_model_id: str) -> dict[str, Any]:
    return {
        "schema": "natural-instruction-adjudication-inputs-v2",
        "packet_sha256": packet["packet_sha256"],
        "reviewer_a": {"model_id": a["plan"]["model_id"],
                        "plan_sha256": a["plan"]["plan_sha256"],
                        "receipt_sha256": a["receipt_file_sha256"],
                        "output_sha256": sha256_obj(a["output"])},
        "reviewer_b": {"model_id": b["plan"]["model_id"],
                        "plan_sha256": b["plan"]["plan_sha256"],
                        "receipt_sha256": b["receipt_file_sha256"],
                        "output_sha256": sha256_obj(b["output"])},
        "adjudicator": {"model_id": target_model_id,
                         "model_family": model_family(target_model_id)},
        "rubric_sha256": sha256_obj(ADJUDICATION_RUBRIC),
        "disputes": disputes,
    }


def build_adjudication_plan(packet: dict[str, Any], run_a: Path, run_b: Path,
                            cfg: dict[str, Any], *, target_model_id: str | None = None,
                            max_cost_usd: float = 0.0) -> dict[str, Any]:
    """Build a zero-call plan for exactly the disputed natural cases."""
    natural.validate_packet(packet)
    a, b, by_a, by_b = _pair(packet, run_a, run_b)
    target_model_id = target_model_id or cfg.get("models", {}).get("task", {}).get("model_id")
    if not isinstance(target_model_id, str) or not target_model_id.strip():
        raise ValueError("adjudicator model ID is required")
    target_family = model_family(target_model_id)
    reviewer_ids = {a["plan"]["model_id"], b["plan"]["model_id"]}
    reviewer_families = {model_family(model_id) for model_id in reviewer_ids}
    if target_model_id in reviewer_ids or target_family in reviewer_families:
        raise ValueError("adjudicator must use a third model ID and family")
    disputes = _dispute_cases(packet, by_a, by_b)
    if not disputes:
        raise ValueError("natural reference has no disagreements to adjudicate")
    inputs = _inputs(packet, a, b, disputes, target_model_id)
    plan = ai_jobs.build_plan(
        "instruction_reference", inputs, cfg, max_calls=len(disputes),
        tokens=MAX_TOKENS, max_input_chars=24000, max_cost_usd=max_cost_usd,
        response_format=natural.REFERENCE_FORMAT, zero_price_only=max_cost_usd == 0.0)
    body = {
        "schema": PLAN_SCHEMA,
        "packet_sha256": packet["packet_sha256"],
        "reference_inputs_sha256": sha256_obj(inputs),
        "reviewer_a": inputs["reviewer_a"],
        "reviewer_b": inputs["reviewer_b"],
        "adjudicator": inputs["adjudicator"],
        "disputed_case_count": len(disputes),
        "disputed_dimension_count": sum(len(row["disputed_dimensions"]) for row in disputes),
        "planned_calls": len(disputes),
        "max_physical_attempts": plan["max_physical_attempts"],
        "zero_price_only": plan.get("zero_price_only") is True,
        "paid_fallback": False,
        "source_runs_immutable": True,
        "scope": "natural_historical_census_disagreements_only",
        "ai_job_plan": plan,
    }
    return {**body, "plan_sha256": sha256_obj(body)}


def _request(case: dict[str, Any], dispute: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task_contract": case["contract"],
        "candidate_strategy": case["strategy_text"],
        "disputed_dimensions": dispute["disputed_dimensions"],
        "reviewer_a": dispute["reviewer_a"],
        "reviewer_b": dispute["reviewer_b"],
    }
    return payload


def adjudication_execute(packet: dict[str, Any], run_a: Path, run_b: Path,
                         cfg: dict[str, Any], plan: dict[str, Any], *,
                         provider=None) -> dict[str, Any]:
    """Execute one bounded third-model call per disputed case."""
    a, b, by_a, by_b = _pair(packet, run_a, run_b)
    target_model_id = plan.get("adjudicator", {}).get("model_id")
    expected = build_adjudication_plan(packet, run_a, run_b, cfg,
        target_model_id=target_model_id,
        max_cost_usd=plan["ai_job_plan"]["max_cost_usd"])
    if plan != expected:
        raise ValueError("natural adjudication plan or immutable reviewer binding changed")
    disputes = _dispute_cases(packet, by_a, by_b)
    case_by_id = {case["case_id"]: case for case in packet["cases"]}
    dispute_by_id = {row["case_id"]: row for row in disputes}
    inputs = _inputs(packet, a, b, disputes, target_model_id)

    def worker(call):
        rows = []
        for dispute in disputes:
            case = case_by_id[dispute["case_id"]]
            payload = _request(case, dispute)
            raw, provenance = call("instruction_adjudicator", payload, ADJUDICATION_RUBRIC)
            dimensions, issues = natural._reference_dimensions(raw, case)
            actor = {**provenance, "blind": False,
                     "other_review_outputs_consulted": True,
                     "origin": "independent_ai", "identity": "adjudicator"}
            rows.append({"case_id": case["case_id"],
                         "disputed_dimensions": dispute["disputed_dimensions"],
                         "dimensions": dimensions,
                         "normalization_issues": issues,
                         "actor": actor})
        return {"schema": RUN_SCHEMA, "packet_sha256": packet["packet_sha256"],
                "annotation_origin": "ai", "natural_distribution": True,
                "adjudication_rubric_sha256": sha256_obj(ADJUDICATION_RUBRIC),
                "inputs_sha256": sha256_obj(inputs), "cases": rows}

    return ai_jobs.execute(plan["ai_job_plan"], inputs, cfg, worker, provider=provider)


def verify_adjudication_run(packet: dict[str, Any], run_a: Path, run_b: Path,
                            plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Verify every adjudication call and its exact disputed-case coverage."""
    a, b, by_a, by_b = _pair(packet, run_a, run_b)
    job = ai_jobs.verify_completed(run_dir, kind="instruction_reference",
                                   allow_legacy_measurement_schema=False)
    output = job["output"]
    disputes = _dispute_cases(packet, by_a, by_b)
    inputs = _inputs(packet, a, b, disputes, plan["adjudicator"]["model_id"])
    if (plan.get("schema") != PLAN_SCHEMA
            or plan.get("plan_sha256") != sha256_obj({k: v for k, v in plan.items()
                                                       if k != "plan_sha256"})
            or job["plan"] != plan.get("ai_job_plan")
            or job["plan"].get("model_id") != plan["adjudicator"]["model_id"]
            or type(job["receipt"].get("test_only")) is not bool
            or output.get("schema") != RUN_SCHEMA
            or output.get("packet_sha256") != packet["packet_sha256"]
            or output.get("inputs_sha256") != sha256_obj(inputs)
            or output.get("adjudication_rubric_sha256") != sha256_obj(ADJUDICATION_RUBRIC)
            or [row.get("case_id") for row in output.get("cases", [])]
                != [row["case_id"] for row in disputes]):
        raise ValueError("natural adjudication run identity or coverage changed")
    for index, dispute in enumerate(disputes):
        case = next(case for case in packet["cases"] if case["case_id"] == dispute["case_id"])
        raw = _read_json(Path(job["root"]) / f"call-{index}.json")
        payload = _request(case, dispute)
        dims, issues = natural._reference_dimensions(raw["raw_output"], case)
        # Reconstruct the actor from the frozen job and call index; the raw
        # call has no separate provenance field.
        expected_actor = {"model_id": job["plan"]["model_id"],
                          "review_run_id": Path(job["root"]).name + f"-{index}",
                          "rubric_sha256": sha256_obj(ADJUDICATION_RUBRIC),
                          "blind": False, "other_review_outputs_consulted": True,
                          "test_only": job["receipt"]["test_only"],
                          "origin": "independent_ai", "identity": "adjudicator"}
        row = output["cases"][index]
        if (raw.get("role") != "instruction_adjudicator"
                or raw.get("system") != ADJUDICATION_RUBRIC
                or raw.get("payload") != payload
                or row.get("dimensions") != dims
                or row.get("normalization_issues") != issues
                or row.get("disputed_dimensions") != dispute["disputed_dimensions"]
                or row.get("actor") != expected_actor):
            raise ValueError("natural adjudication raw receipt or actor identity changed")
    return job


def freeze_reference_v2(packet: dict[str, Any], run_a: Path, run_b: Path,
                        adjudication_run: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Resolve v1 disagreements into a new immutable AI-only natural reference."""
    a, b, by_a, by_b = _pair(packet, run_a, run_b)
    job = verify_adjudication_run(packet, run_a, run_b, plan, adjudication_run)
    disputes = _dispute_cases(packet, by_a, by_b)
    adjudications = {row["case_id"]: row for row in job["output"]["cases"]}
    if set(adjudications) != {row["case_id"] for row in disputes}:
        raise ValueError("adjudication must cover every and only disputed case")
    rows = []
    disagreement_rows = []
    for case in packet["cases"]:
        cid = case["case_id"]
        dispute = next((row for row in disputes if row["case_id"] == cid), None)
        adjudication = adjudications.get(cid)
        verdicts = {}
        for dimension in ("assertion", "contract"):
            left = by_a[cid]["dimensions"][dimension]["verdict"]
            right = by_b[cid]["dimensions"][dimension]["verdict"]
            if left == right:
                verdict = left
            elif dimension not in (dispute or {}).get("disputed_dimensions", []):
                raise ValueError("dispute index does not cover reviewer disagreement")
            else:
                verdict = adjudication["dimensions"][dimension]["verdict"]
                disagreement_rows.append({"case_id": cid, "dimension": dimension,
                    "reviewer_a": left, "reviewer_b": right,
                    "adjudicator": verdict, "resolution": "third_ai_adjudication"})
            verdicts[dimension] = verdict
        rows.append({"case_id": cid, "verdicts": verdicts,
                     "adjudication": adjudication})
    coverage = natural.reference_class_coverage(rows)
    sufficient = all(
        coverage[dimension][key] >= natural.THRESHOLDS[threshold]
        for dimension in ("assertion", "contract")
        for key, threshold in (("approval_reference", "min_approval_reference"),
                               ("rejection_reference", "min_rejection_reference")))
    body = {
        "schema": FROZEN_SCHEMA, "packet_sha256": packet["packet_sha256"],
        "annotation_origin": "ai",
        "reference_method": "two_distinct_model_ids_plus_third_ai_adjudication",
        "natural_distribution": True,
        "model_identity_independence_verified": True,
        "review_run_independence_verified": True,
        "model_source_independence_verified": False,
        "model_source_independence_status": "unverified; model IDs and runs are distinct, supplier independence is not established",
        "reviewer_a": {"model_id": a["plan"]["model_id"],
                       "receipt_sha256": a["receipt_file_sha256"]},
        "reviewer_b": {"model_id": b["plan"]["model_id"],
                       "receipt_sha256": b["receipt_file_sha256"]},
        "adjudicator": {"model_id": plan["adjudicator"]["model_id"],
                         "model_family": plan["adjudicator"]["model_family"],
                         "receipt_sha256": job["receipt_file_sha256"],
                         "run_output_sha256": sha256_obj(job["output"])},
        "disagreements": disagreement_rows,
        "cases": rows,
        "reference_class_coverage": coverage,
        "reference_class_coverage_sufficient": sufficient,
        "reference_test_only": bool(a["receipt"]["test_only"] or b["receipt"]["test_only"]
                                     or job["receipt"]["test_only"]),
        "limitations": [
            "AI adjudication is a proxy reference and not human gold.",
            "Natural-distribution coverage permits target calibration only; it does not by itself authorize candidate publication.",
            "Model supplier/source independence remains unverified.",
        ],
    }
    return {**body, "reference_sha256": sha256_obj(body)}


__all__ = ["PLAN_SCHEMA", "RUN_SCHEMA", "FROZEN_SCHEMA",
           "build_adjudication_plan", "adjudication_execute",
           "verify_adjudication_run", "freeze_reference_v2"]
