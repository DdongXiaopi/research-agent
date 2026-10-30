"""Evaluate, review and release manual Skill maintenance outside the optimizer.

Adapts same-cohort gates and content lineage from pk-pi-hermes-evolve. Real
harness receipts, all seeds, negative datasets and exact review are required;
no caller-provided passed flag can authorize activation.
"""
from __future__ import annotations
import copy
import json
import os
from difflib import unified_diff
from pathlib import Path
import shutil
from uuid import uuid4
import yaml

from . import release
from .budget import check_tolerances
from .policy import static_gate
from ..config import config_fingerprint
from ..eval import harness
from ..eval.assessment_policy import classwise_non_regression, finite_rate
from ..eval.run_manifest import source_tree_fingerprint
from ..evolution.skill_curation import verified_trial
from ..runtime.runner import ArtifactState
from ..runtime.call_ledger import summarize_ledger
from ..runtime.skills import load_body, load_registry
from ..contracts.types import Candidate, content_sha256
from ..util import artifacts_path, artifacts_root, repo_root, read_json, sha256_file, sha256_obj, write_json, now_iso


def build_plan(proposal, cfg, *, negatives, seeds=(13, 37, 71), limit=None):
    parent = release.load_state()
    trial = verified_trial(parent.registry, proposal)
    before = json.dumps({"registry": parent.registry.to_dict(), "bodies": {
        s.name: load_body(s) for s in parent.registry.skills}}, ensure_ascii=False, indent=2, sort_keys=True)
    after = json.dumps({"registry": proposal["trial_registry"], "bodies": proposal["trial_bodies"]},
                       ensure_ascii=False, indent=2, sort_keys=True)
    maintenance_diff = "\n".join(unified_diff(before.splitlines(), after.splitlines(),
                                             fromfile="parent/skill-maintenance.json", tofile="trial/skill-maintenance.json"))
    static_checks = []
    for action in proposal["actions"]:
        if action["action"] != "merge":
            continue
        body = action["merged_body"]
        candidate = Candidate(candidate_id="migration-body-" + sha256_obj(action),
            parent_release_id=parent.release_id, artifact_type="skill_body", artifact_id=action["survivor"],
            new_content=body, content_sha256=content_sha256(body), diff=maintenance_diff,
            rationale="static screening inside explicit manual Skill merge; not an optimizer candidate")
        gate_cfg = copy.deepcopy(cfg)
        gate_cfg.setdefault("evolution", {})["allowed_artifacts"] = ["skill_body"]
        gate = static_gate(candidate, gate_cfg, parent)
        if gate["passed"] is not True:
            raise ValueError("migration body failed the shared static gate: " + ",".join(
                row["check"] for row in gate["results"] if row["passed"] is not True))
        static_checks.append(gate)
    main = list((cfg.get("evaluation") or {}).get("primary_final_datasets")
                or (cfg.get("evaluation") or {}).get("datasets") or [])
    if (not main or not negatives or set(main) & set(negatives) or
            len(set(main)) != len(main) or len(set(negatives)) != len(negatives) or
            len(seeds) < 3 or len(set(seeds)) != len(seeds) or
            any(isinstance(s, bool) or not isinstance(s, int) for s in seeds) or
            (limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1))):
        raise ValueError("migration requires configured main datasets, distinct negative domains and >=3 unique seeds")
    scopes = []
    for dataset, split, role in ([(d, s, "main") for d in main for s in ("dev_optimize", "selection_gate")]
                                 + [(d, "dev_optimize", "negative") for d in negatives]):
        tasks, _ = harness.prepare_tasks(cfg, dataset, split, limit)
        scopes.append({"dataset": dataset, "split": split, "role": role,
                       "n": len(tasks), "task_content_sha256": sha256_obj(tasks)})
    provider = str((cfg.get("runtime") or {}).get("provider", "fake"))
    if provider not in {"fake", "http", "pi"}:
        raise ValueError("unsupported migration provider")
    calls = int((cfg.get("runtime") or {}).get("max_model_calls", 12))
    if not 1 <= calls <= 40:
        raise ValueError("migration requires bounded per-task model calls")
    # Includes task/router/judge calls. Pi bridges reserve bounded retries too.
    attempts = max([1] + [int(model.get("max_retries", 1)) for model in (cfg.get("models") or {}).values()
                          if isinstance(model, dict)])
    attempts = max(attempts, 15 if provider == "pi" else 1)
    if attempts > 30:
        raise ValueError("migration retry bound exceeds 30 attempts")
    logical = sum(s["n"] for s in scopes) * 2 * len(seeds) * calls if provider != "fake" else 0
    value = {"schema": "manual-skill-migration-plan-v1", "proposal_sha256": proposal["proposal_sha256"],
        "parent_release_id": parent.release_id, "parent_registry_sha256": parent.registry.fingerprint(),
        "trial_registry_sha256": trial.fingerprint(), "prompt_sha256": sha256_obj(parent.system_prompt),
        "maintenance_diff": maintenance_diff,
        "static_body_checks": static_checks,
        "task_demos_sha256": sha256_obj(parent.task_demos), "configuration_sha256": sha256_obj(cfg),
        "protocol_sha256": config_fingerprint(cfg), "source_tree_sha256": source_tree_fingerprint()["sha256"],
        "scopes": scopes, "negatives": list(negatives), "seeds": list(seeds), "limit": limit,
        "provider": provider, "logical_request_cap": logical, "physical_request_cap": logical * attempts,
        "final_holdout_read": False, "preflight_model_calls": 0}
    return {**value, "plan_sha256": sha256_obj(value)}


def _scope_dir(value):
    root = Path(value).resolve()
    root.relative_to(artifacts_path("skill_migrations").resolve())
    return root


def _run_summary(path, plan, state, scope, seed):
    root = Path(path).resolve()
    root.relative_to(artifacts_path("runs").resolve())
    receipt = read_json(root / "receipt.json")
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != sha256_obj(unsigned) or receipt.get("status") != "completed":
        raise ValueError("migration requires a completed harness receipt")
    if not {"manifest.json", "summary.json", "leakage.json"} <= set(receipt.get("files") or {}):
        raise ValueError("migration receipt omits mandatory evidence")
    for name, digest in receipt["files"].items():
        if name not in {"manifest.json", "summary.json", "leakage.json", "per_task.jsonl", "failures.jsonl", "trace.jsonl"}:
            raise ValueError("unexpected receipt path")
        if sha256_file(root / name) != digest:
            raise ValueError("migration execution evidence changed")
    manifest, summary = read_json(root / "manifest.json"), read_json(root / "summary.json")
    if (manifest["code"]["source_tree_sha256"] != plan["source_tree_sha256"] or
            manifest["configuration"]["config_sha256"] != plan["configuration_sha256"] or
            manifest["data"]["evaluated_task_content_sha256"] != scope["task_content_sha256"] or
            manifest["artifacts"]["skill_registry_fingerprint"] != state.registry.fingerprint() or
            manifest["artifacts"]["system_prompt_sha256"] != sha256_obj(state.system_prompt) or
            manifest["artifacts"]["task_demos_sha256"] != sha256_obj(state.task_demos) or
            (summary["dataset"], summary["split"], summary["seed"]) != (scope["dataset"], scope["split"], seed) or
            read_json(root / "leakage.json").get("ok") is not True):
        raise ValueError("migration run differs from its frozen scope/artifacts")
    return summary


def _verify_result(root, cfg):
    root = _scope_dir(root)
    proposal, plan, report = (read_json(root / name) for name in ("proposal.json", "plan.json", "report.json"))
    expected = build_plan(proposal, cfg, negatives=plan["negatives"], seeds=plan["seeds"], limit=plan["limit"])
    if expected != plan or report.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("migration plan/config/source/parent changed")
    parent = release.load_state()
    trial = ArtifactState(parent.release_id, parent.system_prompt, verified_trial(parent.registry, proposal),
                          task_demos=copy.deepcopy(parent.task_demos))
    pairs = report.get("pairs") or []
    if len(pairs) != len(plan["scopes"]) * len(plan["seeds"]):
        raise ValueError("incomplete migration comparison")
    checks, seen, runs = [], set(), set()
    for row in pairs:
        index, seed = row["scope_index"], row["seed"]
        if (index, seed) in seen or index not in range(len(plan["scopes"])) or seed not in plan["seeds"]:
            raise ValueError("duplicate or unplanned migration comparison")
        seen.add((index, seed))
        scope = plan["scopes"][index]
        if row["baseline_run"] == row["candidate_run"] or any(r in runs for r in (row["baseline_run"], row["candidate_run"])):
            raise ValueError("migration arms must have distinct physical executions")
        runs.update((row["baseline_run"], row["candidate_run"]))
        b = _run_summary(row["baseline_run"], plan, parent, scope, seed)
        c = _run_summary(row["candidate_run"], plan, trial, scope, seed)
        passed = b["metrics"].get("n_tasks") == c["metrics"].get("n_tasks") == scope["n"]
        for metric in ("accuracy", "coverage", "answered_precision"):
            bv, cv = b["metrics"].get(metric), c["metrics"].get(metric)
            passed = passed and finite_rate(bv) and finite_rate(cv) and cv + 1e-9 >= bv
        classes = classwise_non_regression(b["class_metrics"], c["class_metrics"])
        gate_cfg = copy.deepcopy(cfg)
        if plan["provider"] != "fake":
            tolerance = gate_cfg.setdefault("gates", {}).setdefault("accept", {}).setdefault("tolerance", {})
            tolerance["required"] = sorted(set(tolerance.get("required") or []) | {"cost", "p95_latency"})
        tolerances = check_tolerances(b["metrics"], c["metrics"], gate_cfg)
        checks.append({"dataset": scope["dataset"], "role": scope["role"], "split": scope["split"],
            "seed": seed, "passed": bool(passed and classes["passed"] and tolerances["passed"]),
            "classwise": classes, "tolerances": tolerances})
    ledger = summarize_ledger(root / "call-ledger.jsonl")
    if (ledger.get("logical_incomplete", 0) or ledger.get("physical_incomplete", 0)
            or ledger.get("logical_status_counts", {}).get("failed", 0)):
        raise ValueError("migration has unknown in-flight requests")
    return plan, proposal, parent, trial, checks


def evaluate(proposal, cfg, plan, *, provider_factory=None):
    if build_plan(proposal, cfg, negatives=plan["negatives"], seeds=plan["seeds"], limit=plan["limit"]) != plan:
        raise ValueError("migration execution plan differs")
    root = artifacts_path("skill_migrations", "migration-" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "proposal.json", proposal)
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", {"status": "started"})
    parent = release.load_state()
    trial = ArtifactState(parent.release_id, parent.system_prompt, verified_trial(parent.registry, proposal),
                          task_demos=copy.deepcopy(parent.task_demos))
    bindings = {"RSI_CALL_LEDGER_PATH": str(root / "call-ledger.jsonl"),
                "RSI_ACTIVE_EXECUTION_PLAN_SHA256": plan["plan_sha256"],
                "RSI_CALL_LEDGER_MAX_LOGICAL": str(plan["logical_request_cap"]),
                "RSI_CALL_LEDGER_MAX_PHYSICAL": str(plan["physical_request_cap"])}
    before = {k: os.environ.get(k) for k in bindings}
    report = {"schema": "manual-skill-migration-evaluation-v1", "plan_sha256": plan["plan_sha256"], "pairs": []}
    try:
        os.environ.update(bindings)
        for index, scope in enumerate(plan["scopes"]):
            for seed in plan["seeds"]:
                runs = []
                for state in (parent, trial):
                    result = harness.evaluate(cfg, state, scope["dataset"], scope["split"], seed=seed,
                        limit=plan["limit"], run_tag="manual-skill-migration",
                        provider=provider_factory(state, seed) if provider_factory else None)
                    runs.append(str(artifacts_path("runs", result.run_id)))
                report["pairs"].append({"scope_index": index, "seed": seed,
                                        "baseline_run": runs[0], "candidate_run": runs[1]})
                write_json(root / "report.json", report)
        _, _, _, _, checks = _verify_result(root, cfg)
        report.update(checks=checks, passed=all(c["passed"] for c in checks))
        write_json(root / "report.json", report)
        write_json(root / "status.json", {"status": "completed"})
        return {"run_dir": str(root), "passed": report["passed"], "plan_sha256": plan["plan_sha256"]}
    except BaseException as exc:
        write_json(root / "status.json", {"status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                                          "error_type": type(exc).__name__})
        raise
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def review(root, cfg, *, reviewer, reason, decision, expected_report_sha256):
    root = _scope_dir(root)
    if (not str(reviewer).strip() or not str(reason).strip() or decision not in {"approve", "reject"} or
            sha256_file(root / "report.json") != expected_report_sha256 or
            read_json(root / "status.json").get("status") != "completed"):
        raise ValueError("review requires a completed exact report and explicit human decision/reason")
    plan, proposal, parent, _, checks = _verify_result(root, cfg)
    if decision == "approve" and not all(c["passed"] for c in checks):
        raise ValueError("migration regression gates failed")
    value = {"schema": "manual-skill-migration-review-v1", "reviewer": reviewer, "reason": reason,
        "decision": decision, "report_sha256": expected_report_sha256,
        "plan_sha256": plan["plan_sha256"], "proposal_sha256": proposal["proposal_sha256"],
        "parent_release_id": parent.release_id, "created_at": now_iso()}
    write_json(root / "review.json", value)
    return value


def publish(root, cfg):
    root = _scope_dir(root)
    if read_json(root / "status.json").get("status") != "completed":
        raise ValueError("migration execution is not complete")
    decision = read_json(root / "review.json")
    plan, proposal, parent, trial, checks = _verify_result(root, cfg)
    if (decision.get("decision") != "approve" or not str(decision.get("reviewer") or "").strip()
            or not str(decision.get("reason") or "").strip() or not all(c["passed"] for c in checks)
            or decision.get("report_sha256") != sha256_file(root / "report.json")
            or decision.get("plan_sha256") != plan["plan_sha256"]
            or decision.get("proposal_sha256") != proposal["proposal_sha256"]
            or decision.get("parent_release_id") != parent.release_id):
        raise ValueError("migration lacks exact human approval")
    if plan["provider"] == "fake" and artifacts_root().resolve() == (repo_root() / "artifacts").resolve():
        raise ValueError("fake migration publication is restricted to an isolated engineering workspace")
    lock = artifacts_path("skill_migrations", "publication.lock")
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(plan["plan_sha256"])
    rid = "rel-migration-" + proposal["proposal_sha256"][:20] + "-" + uuid4().hex[:8]
    directory = release.RELEASES_ROOT / rid
    try:
        directory.mkdir(exist_ok=False)
        shutil.copy2(release.RELEASES_ROOT / parent.release_id / "system_prompt.txt", directory / "system_prompt.txt")
        snapshot = release._materialize_bodies(trial.registry, rid)
        snapshot.release_id = rid
        with (directory / "skills/registry.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snapshot.to_dict(), handle, allow_unicode=True, sort_keys=True)
        demos = release.RELEASES_ROOT / parent.release_id / "task_demos.json"
        if demos.exists():
            shutil.copy2(demos, directory / demos.name)
        written = load_registry(str(directory / "skills/registry.yaml"))
        write_json(directory / "release.json", {"release_id": rid, "parent": parent.release_id,
            "created_at": now_iso(), "candidate": None, "manual_migration": proposal,
            "review": decision, "evaluation_report_sha256": sha256_file(root / "report.json"),
            "rollback_point": parent.release_id, "artifacts": {
                "system_prompt_sha256": sha256_obj(parent.system_prompt),
                "skills_registry_sha256": sha256_obj(written.to_dict()),
                "skills_body_sha256": {s.name: content_sha256(load_body(s)) for s in written.skills},
                **({"task_demos_sha256": sha256_obj(demos.read_text(encoding="utf-8"))} if demos.exists() else {})}})
        if release.current_release_id() != parent.release_id:
            raise ValueError("ACTIVE changed during migration snapshot creation")
        release.load_state(rid)
        aliases = release.load_aliases()
        aliases["champion"] = rid
        aliases["challenger"] = None
        aliases["archived"] = sorted(set(aliases.get("archived", []) + [parent.release_id]))
        aliases.setdefault("history", []).append({"action": "manual_skill_migration", "from": parent.release_id,
                                                  "to": rid, "at": now_iso()})
        release._activate_release(rid, aliases)
        write_json(root / "published.json", {"release_id": rid, "rollback_point": parent.release_id})
        return rid
    finally:
        lock.unlink(missing_ok=True)
