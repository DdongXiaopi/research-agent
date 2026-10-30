"""Bridge independently assessed feedback candidates to existing review/release."""
from __future__ import annotations

import os
import math
from pathlib import Path

from ..config import config_fingerprint
from ..contracts.types import Candidate
from ..eval import open_benchmark as benchmark
from ..eval.run_manifest import source_tree_fingerprint
from ..runtime import research_assistant as research
from ..runtime.research_tools import audit_open_research_trace
from ..eval.open_research import source_identity_keys
from ..util import artifacts_path, artifacts_root, repo_root, read_json, sha256_file, sha256_obj
from . import release, review
from .policy import static_gate


def _feedback_sources(candidate):
    from ..evolution.session_feedback import verify_research_run
    expected = set(candidate.extra.get("feedback_ids") or [])
    sources, found = set(), set()
    for path in artifacts_path("research_feedback").glob("*/*/*.json"):
        feedback = read_json(path)
        if feedback.get("feedback_id") not in expected:
            continue
        if (feedback.get("use_for_evolution") is not True
                or feedback.get("category") == "preference"
                or feedback.get("feedback_sha256") != sha256_obj({
                    k: v for k, v in feedback.items() if k != "feedback_sha256"})):
            raise ValueError("feedback consent/content changed after proposal")
        root = Path(feedback["run_dir"])
        request, _ = verify_research_run(root)
        if feedback["run_receipt_sha256"] != sha256_file(root / "receipt.json"):
            raise ValueError("feedback source receipt changed")
        found.add(feedback["feedback_id"])
        for source in request["evidence"]:
            sources.update(source_identity_keys(source))
    if found != expected:
        raise ValueError("feedback candidate source records are missing")
    return sources


def _assessment(evaluation_id, split, candidate, cfg, feedback_sources):
    root = benchmark._evaluation(evaluation_id)
    plan = read_json(root / "plan.json")
    if (plan.get("split") != split or plan.get("unpublished_candidate") != candidate.to_dict()
            or plan.get("config_sha256") != sha256_obj(cfg)
            or plan.get("protocol_sha256") != config_fingerprint(cfg)
            or plan.get("source_tree_sha256") != source_tree_fingerprint()["sha256"]
            or set(plan.get("arms", {})) != {"baseline", "candidate"}
            or plan["arms"]["baseline"]["release_id"] != release.current_release_id()):
        raise ValueError("open promotion requires current-source paired evaluations of the exact candidate")
    annotations = read_json(root / "assessment.json")["annotations"]
    report = benchmark.assess(evaluation_id, annotations)  # Recompute from frozen answers and original annotations.
    if report["status"] != "assessed" or report["unresolved_metrics"]:
        raise ValueError("open promotion requires complete independent annotations")
    parent = release.load_state()
    states = {"baseline": parent, "candidate": benchmark._candidate_state(parent, cfg, candidate.to_dict())}
    efficiency = {arm: {"cost": [], "latency": [], "measured": []} for arm in states}
    for case in read_json(root / "cases.json"):
        run = Path(case["run_dir"])
        request = research.request_from_dict(read_json(run / "request.json"))
        for source in request.evidence:
            keys = source_identity_keys(source)
            if keys & feedback_sources:
                raise ValueError("feedback source leaked into independent candidate evaluation")
        expected = research.build_plan(request, cfg, states[case["arm"]])
        if read_json(run / "plan.json") != expected:
            raise ValueError("open evaluation run differs from frozen candidate/configuration")
        audit_open_research_trace(run)
        raw = read_json(run / "raw-model-output.json")
        from ..runtime.trace import read_trace
        calls = [event for event in read_trace(run / "workflow-trace.jsonl") if event.get("event_type") == "model.call"]
        cost = raw.get("cost") or {}
        response = read_json(run / "response.json")
        accounting = response["execution_summary"]["request_accounting"]
        efficiency[case["arm"]]["cost"].append(cost.get("usd"))
        efficiency[case["arm"]]["latency"].append(calls[0].get("latency_ms") if len(calls) == 1 else None)
        efficiency[case["arm"]]["measured"].append(bool(cost.get("cost_estimated") is False
            and cost.get("usage_complete") is True and accounting["physical_http_attempts"] >= 1
            and not accounting["logical_incomplete"] and not accounting["physical_incomplete"]))
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    return report, plan, files, efficiency


def _evidence(candidate, cfg, dev_id, gate_id, engineering_only):
    if dev_id == gate_id:
        raise ValueError("development and selection evaluations must be separate")
    if engineering_only and (not os.environ.get("RSI_HOME") or artifacts_root() in {
            repo_root(), repo_root() / "artifacts"}):
        raise ValueError("engineering promotion requires an isolated RSI_HOME")
    feedback_sources = _feedback_sources(candidate)
    dev, dp, df, de = _assessment(dev_id, "dev_optimize", candidate, cfg, feedback_sources)
    gate, gp, gf, ge = _assessment(gate_id, "selection_gate", candidate, cfg, feedback_sources)
    if dp["dataset_sha256"] != gp["dataset_sha256"]:
        raise ValueError("development and selection must belong to the same frozen benchmark")
    manifest = benchmark.verify_dataset(artifacts_path("open_benchmarks", dp["dataset_sha256"]))
    synthetic = "synthetic" in str(manifest["provenance"]).lower()
    if synthetic and not engineering_only:
        raise ValueError("synthetic evidence can only authorize isolated engineering publication")
    if not engineering_only and any(report["annotation_origin"] == "engineering_fixture" for report in (dev, gate)):
        raise ValueError("engineering fixture annotations cannot authorize production publication")
    checks = []
    for name, report in (("development", dev), ("selection", gate)):
        scores = report["arm_scores"]
        before, after = scores["baseline"].get("rubric"), scores["candidate"].get("rubric")
        checks.append({"check": name + "_rubric_non_regression", "passed":
                       before is not None and after is not None and after >= before})
        citations = report["citation_metrics"]
        b, c = citations["baseline"], citations["candidate"]
        bp, cp = b["citation_precision_micro"], c["citation_precision_micro"]
        baseline_ok = bp is not None or (engineering_only and b["citation_count"] == 0)
        checks.append({"check": name + "_citation_non_regression", "passed": bool(
            baseline_ok and cp is not None and cp >= (bp or 0.0)
            and c["citation_response_coverage"] >= b["citation_response_coverage"])})
    if not engineering_only:
        tolerances = ((cfg.get("gates") or {}).get("accept") or {}).get("tolerance") or {}
        for name, quantities in (("development", de), ("selection", ge)):
            measured = cfg["runtime"]["provider"] == "http" and all(
                all(value["measured"]) for value in quantities.values())
            checks.append({"check": name + "_measured_cost_and_latency", "passed": bool(measured)})
            for metric, field, default in (("cost", "cost_pct", 15), ("latency", "p95_latency_pct", 20)):
                arrays = {arm: values[metric] for arm, values in quantities.items()}
                finite = all(values and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values)
                             for values in arrays.values())
                if finite:
                    aggregate = {arm: (sum(values) / len(values) if metric == "cost" else
                        sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]) for arm, values in arrays.items()}
                    passed = aggregate["candidate"] <= aggregate["baseline"] * (1 + float(tolerances.get(field, default)) / 100)
                else:
                    passed = False
                checks.append({"check": name + "_" + metric + "_tolerance", "passed": bool(measured and passed)})
    interval = ((gate.get("paired_group_analysis") or {}).get("metrics") or {}).get("rubric") or {}
    checks.append({"check": "independent_selection_improvement", "passed": bool(
        interval.get("groups", 0) >= 3 and interval.get("delta", 0) > 0
        and interval.get("ci95_low", 0) > 0)})
    gate_result = static_gate(candidate, cfg, release.load_state())
    checks.append({"check": "static_gate", "passed": gate_result["passed"] is True})
    evidence = {"schema": "open-promotion-evidence-v1", "development_id": dev_id,
                "selection_id": gate_id, "engineering_only": engineering_only,
                "dataset_sha256": dp["dataset_sha256"], "development_files": df, "selection_files": gf,
                "measured_efficiency": {"development": de, "selection": ge},
                "checks": checks, "generalization_claim_authorized": False}
    return evidence, dev, gate, gate_result


def create_request(candidate_value, cfg, *, development_id, selection_id, engineering_only=False):
    candidate = Candidate(**candidate_value)
    benchmark._candidate_state(release.load_state(), cfg, candidate_value)
    evidence, dev, gate, static = _evidence(candidate, cfg, development_id, selection_id, engineering_only)
    request = review.build_request(candidate, dev["arm_scores"], gate["arm_scores"],
                                   candidate.parent_release_id, static)
    request.update(evidence_type="open_research_evaluation_v1", evaluation_summary=evidence,
                   candidate=candidate_value, configuration=cfg,
                   verdict="proposed_accept" if all(x["passed"] for x in evidence["checks"]) else "proposed_reject")
    if artifacts_path("reviews", candidate.candidate_id + ".review.json").exists():
        raise ValueError("candidate already reviewed; preserve the prior review identity")
    from ..util import write_json
    target = artifacts_path("candidates", candidate.candidate_id + ".json")
    if target.exists() and review.candidate_artifact_sha256(Candidate(**read_json(target))) != review.candidate_artifact_sha256(candidate):
        raise ValueError("candidate identity already stores different content")
    write_json(target, candidate_value)
    location = review.write_request(request)
    return {"candidate_id": candidate.candidate_id, "review_request": str(location),
            "verdict": request["verdict"], "checks": evidence["checks"], "publication_authorized": False}


def verify_request(request):
    try:
        candidate = Candidate(**request["candidate"])
        cfg = request["configuration"]
        saved = request["evaluation_summary"]
        expected, dev, gate, static = _evidence(candidate, cfg, saved["development_id"],
                                                saved["selection_id"], saved["engineering_only"])
        return bool(expected == saved and all(x["passed"] for x in expected["checks"])
                    and request["optimize_metrics"] == dev["arm_scores"]
                    and request["gate_metrics"] == gate["arm_scores"]
                    and request["static_gate"] == {key: static[key] for key in ("passed", "results")}
                    and request["candidate_artifact_sha256"] == review.candidate_artifact_sha256(candidate)
                    and request["parent_release_id"] == release.current_release_id())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False
