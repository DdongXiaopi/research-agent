"""Measured fixed-task publication acceptance for isolated synthetic fixtures only."""
from __future__ import annotations
import os
from pathlib import Path

from ..contracts.types import Candidate
from ..data.split import load_manifest
from ..eval import controlled_experiment as experiment
from ..eval.harness import prepare_tasks
from ..runtime.call_ledger import summarize_ledger
from ..runtime.trace import audit_task_execution, read_trace
from ..util import artifacts_path, artifacts_root, repo_root, read_json, sha256_obj, sha256_file, write_json
from . import release, review
from .policy import static_gate


def _measurements(report_path, cfg, candidate):
    if not os.environ.get("RSI_HOME") or artifacts_root() in {repo_root(), repo_root() / "artifacts"}:
        raise ValueError("engineering acceptance is restricted to an isolated RSI_HOME")
    path = Path(report_path).resolve()
    path.relative_to(artifacts_path("experiments", "multiseed").resolve())
    report, plan = read_json(path), read_json(path.parent / "plan.json")
    if (report.get("status") != "completed" or read_json(path.parent / "status.json").get("status") != "completed"
            or plan["spec"].get("candidate") != candidate.to_dict()
            or set(plan["splits"]) != {"dev_optimize", "selection_gate"}
            or plan["spec"]["base"] != release.current_release_id()):
        raise ValueError("engineering promotion requires complete paired optimize/gate measurements")
    rebuilt = experiment.build_plan(cfg, kind="multiseed", spec=plan["spec"], datasets=plan["datasets"],
        splits=plan["splits"], seeds=plan["seeds"], limit=plan["limit"],
        logical_request_cap=plan["logical_request_cap"], physical_request_cap=plan["physical_request_cap"])
    if rebuilt != plan or report.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("engineering measurement source/config/cohort changed")
    for dataset in plan["datasets"]:
        manifest = load_manifest(dataset)
        if manifest.get("schema") != "public-task-input-dataset-v1" or "synthetic" not in manifest.get("source_file", "").lower():
            raise ValueError("this acceptance path accepts explicit synthetic public fixtures only")
    accounting = summarize_ledger(path.parent / "call-ledger.jsonl")
    if (accounting != report["request_accounting"] or accounting["logical_incomplete"]
            or accounting["physical_incomplete"] or accounting["physical_timeouts"]
            or (plan["provider"] != "fake" and accounting["physical_http_attempts"] < plan["planned_task_evaluations"])):
        raise ValueError("engineering request accounting is incomplete")
    comparisons = report["result"]["comparisons"]
    expected_scopes = {(x["dataset"], x["split"]): x for x in plan["scopes"]}
    if len(comparisons) != len(expected_scopes):
        raise ValueError("engineering comparisons are incomplete")
    seen, physical_runs, checks, metrics, files = set(), set(), [], {}, {}
    for comparison in comparisons:
        key = (comparison["dataset"], comparison["split"])
        if key in seen or key not in expected_scopes:
            raise ValueError("duplicate or unplanned engineering scope")
        seen.add(key)
        scope = expected_scopes[key]
        if {r["seed"] for r in comparison["per_seed"]} != set(plan["seeds"]) or len(comparison["per_seed"]) != len(plan["seeds"]):
            raise ValueError("engineering measurements omitted a planned seed")
        tasks, _ = prepare_tasks(cfg, scope["dataset"], scope["split"], plan["limit"])
        for pair in comparison["per_seed"]:
            values = {}
            for arm, id_key in (("baseline", "base_run_id"), ("candidate", "cand_run_id")):
                run_id = pair[id_key]
                if run_id in physical_runs or Path(run_id).name != run_id:
                    raise ValueError("engineering arms need distinct physical run IDs")
                physical_runs.add(run_id)
                root = artifacts_path("runs", run_id)
                receipt = read_json(root / "receipt.json")
                required_files = {"manifest.json", "summary.json", "leakage.json"}
                if key[1] == "dev_optimize":
                    required_files.add("trace.jsonl")
                if (receipt.get("status") != "completed" or receipt.get("receipt_sha256") != sha256_obj({
                        k: v for k, v in receipt.items() if k != "receipt_sha256"})
                        or not required_files <= set(receipt["files"])):
                    raise ValueError("engineering harness receipt is incomplete")
                for name, digest in receipt["files"].items():
                    if Path(name).name != name or sha256_file(root / name) != digest:
                        raise ValueError("engineering harness evidence changed")
                manifest, summary = read_json(root / "manifest.json"), read_json(root / "summary.json")
                identity = plan["conditions"][arm]
                if (manifest["configuration"]["config_sha256"] != plan["configuration_sha256"]
                        or manifest["code"]["source_tree_sha256"] != plan["source_tree_sha256"]
                        or manifest["data"]["evaluated_task_content_sha256"] != scope["task_content_sha256"]
                        or manifest["artifacts"]["system_prompt_sha256"] != identity["prompt"]
                        or manifest["artifacts"]["skill_registry_fingerprint"] != identity["registry"]
                        or manifest["artifacts"]["task_demos_sha256"] != identity["demos"]
                        or (summary["dataset"], summary["split"], summary["seed"]) != (*key, pair["seed"])
                        or summary["metrics"]["n_tasks"] != scope["n"]
                        or read_json(root / "leakage.json").get("ok") is not True):
                    raise ValueError("engineering harness run identity mismatch")
                if key[1] == "dev_optimize":
                    events = read_trace(root / "trace.jsonl")
                    for task in tasks:
                        audit_task_execution(events, task["task_id"])
                values[arm] = summary["metrics"]
                files[run_id] = {**receipt["files"], "receipt.json": sha256_file(root / "receipt.json")}
            b, c = values["baseline"], values["candidate"]
            passed = (c["accuracy"] > b["accuracy"] and c["coverage"] >= b["coverage"]
                      and b.get("cost_usd") == 0 and c.get("cost_usd") == 0)
            checks.append({"check": f"{key[0]}:{key[1]}:{pair['seed']}:measured_improvement", "passed": passed})
            metrics[f"{key[0]}:{key[1]}:{pair['seed']}"] = values
    static = static_gate(candidate, cfg, release.load_state())
    checks.append({"check": "static_gate", "passed": static["passed"] is True})
    evidence = {"schema": "engineering-fixed-acceptance-v1", "report_path": str(path),
        "report_sha256": sha256_file(path), "plan_sha256": plan["plan_sha256"],
        "run_files": files, "checks": checks, "engineering_only": True,
        "scientific_gain_claim": False}
    return evidence, metrics, static


def create_request(report_path, cfg, candidate_value):
    candidate = Candidate(**candidate_value)
    evidence, metrics, static = _measurements(report_path, cfg, candidate)
    before = {key: row["baseline"] for key, row in metrics.items()}
    after = {key: row["candidate"] for key, row in metrics.items()}
    req = review.build_request(candidate, before, after, candidate.parent_release_id, static)
    req.update(evidence_type="engineering_fixed_task_v1", evaluation_summary=evidence,
               candidate=candidate_value, configuration=cfg,
               verdict="proposed_accept" if all(x["passed"] for x in evidence["checks"]) else "proposed_reject")
    write_json(artifacts_path("candidates", candidate.candidate_id + ".json"), candidate_value)
    location = review.write_request(req)
    return {"candidate_id": candidate.candidate_id, "verdict": req["verdict"], "checks": evidence["checks"],
            "review_request": str(location), "engineering_only": True}


def verify_request(req):
    try:
        candidate = Candidate(**req["candidate"])
        expected, metrics, static = _measurements(req["evaluation_summary"]["report_path"], req["configuration"], candidate)
        return bool(expected == req["evaluation_summary"] and all(x["passed"] for x in expected["checks"])
            and req["optimize_metrics"] == {key: row["baseline"] for key, row in metrics.items()}
            and req["gate_metrics"] == {key: row["candidate"] for key, row in metrics.items()}
            and req["candidate_artifact_sha256"] == review.candidate_artifact_sha256(candidate)
            and req["static_gate"] == {key: static[key] for key in ("passed", "results")})
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return False
