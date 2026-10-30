"""Plan-bound visible-split experiments shared by multiseed and ablation."""
from __future__ import annotations

import copy
import os
from contextlib import contextmanager

from ..config import config_fingerprint
from ..data.freeze import require_verified
from ..governance import release
from ..runtime.call_ledger import summarize_ledger
from ..runtime.runner import ArtifactState
from ..runtime.skills import load_body, with_body, with_description
from ..util import artifacts_path, artifacts_root, repo_root, sha256_obj, write_json
from . import ablation, multiseed
from .final_execution_plan import _attempts_per_operation, _release_fingerprint
from .harness import prepare_tasks
from .run_manifest import source_tree_fingerprint


@contextmanager
def request_scope(plan, ledger):
    values = {"RSI_ACTIVE_EXECUTION_PLAN_SHA256": plan["plan_sha256"],
              "RSI_CALL_LEDGER_PATH": str(ledger),
              "RSI_CALL_LEDGER_MAX_LOGICAL": str(plan["logical_request_cap"]),
              "RSI_CALL_LEDGER_MAX_PHYSICAL": str(plan["physical_request_cap"])}
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _identity(state):
    return {"release_id": state.release_id, "prompt": sha256_obj(state.system_prompt),
            "registry": state.registry.fingerprint(), "demos": sha256_obj(state.task_demos),
            "bodies": {s.name: sha256_obj(load_body(s)) for s in state.registry.skills}}


def _conditions(kind, spec, cfg):
    base = release.load_state(spec["base"])
    if kind == "multiseed":
        out = {"baseline": base}
        if spec.get("candidate"):
            from ..contracts.types import Candidate
            from ..evolution.apply import state_with
            from ..governance.policy import static_gate
            candidate = Candidate(**spec["candidate"])
            if (candidate.parent_release_id != base.release_id or spec.get("compare")
                    or not static_gate(candidate, cfg, base)["passed"]):
                raise ValueError("unpublished comparison candidate failed parent/static contract")
            out["candidate"] = state_with(base, candidate)
        if spec.get("compare"):
            out["candidate"] = release.load_state(spec["compare"])
        return out, {}
    if spec.get("conditions"):
        out = {name: release.load_state(rid) for name, rid in spec["conditions"].items()}
        if "A0" not in out or len(out) < 2 or _identity(out["A0"]) != _identity(base):
            raise ValueError("explicit ablation needs A0=base and at least one comparison")
        if not set(out) <= set(ablation.CONDITIONS):
            raise ValueError("explicit ablation conditions must use A0 through A4")
        return out, {name: True for name in out}
    prompt = release.load_state(spec["prompt"]).system_prompt if spec.get("prompt") else None
    desc = body = combined = None
    if spec.get("description"):
        target = release.load_state(spec["description"]).registry
        if {s.name for s in target.skills} != {s.name for s in base.registry.skills}:
            raise ValueError("description ablation cannot add/remove Skills")
        desc = copy.deepcopy(base.registry)
        for skill in target.skills:
            desc = with_description(desc, skill.name, skill.description, list(skill.triggers))
    if spec.get("body"):
        target = release.load_state(spec["body"]).registry
        if not {s.name for s in base.registry.skills} <= {s.name for s in target.skills}:
            raise ValueError("body ablation cannot remove existing Skills")
        body = copy.deepcopy(base.registry)
        for skill in target.skills:
            if body.get(skill.name):
                body = with_body(body, skill.name, load_body(skill))
            else:
                body.skills.append(copy.deepcopy(skill))
        combined = copy.deepcopy(body)
        if desc:
            for skill in desc.skills:
                combined = with_description(combined, skill.name, skill.description, list(skill.triggers))
    if not any((prompt, desc, body)):
        raise ValueError("ablation requires explicit condition releases or at least one factor release")
    flags = {"P": prompt is not None, "D": desc is not None, "B": body is not None}
    states = ablation.build_factorial_conditions(base, prompt, desc, body, combined or desc)
    available = {code: all(not enabled or flags[factor] for factor, enabled in ablation.decode(code).items())
                 for code in states}
    return states, available


def build_plan(cfg, *, kind, spec, datasets, splits, seeds, limit,
               logical_request_cap=None, physical_request_cap=None):
    if kind not in {"multiseed", "ablate"}:
        raise ValueError("unsupported controlled experiment")
    if (not datasets or len(set(datasets)) != len(datasets) or not splits
            or len(set(splits)) != len(splits) or set(splits) - {"dev_optimize", "selection_gate"}):
        raise ValueError("experiments require unique datasets and visible splits only")
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)):
        raise ValueError("controlled experiments require at least three distinct integer seeds")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("controlled experiments require an explicit positive task limit")
    frozen = require_verified(datasets)
    states, available = _conditions(kind, spec, cfg)
    scopes = []
    for dataset in datasets:
        for split in splits:
            tasks, _ = prepare_tasks(cfg, dataset, split, limit)
            scopes.append({"dataset": dataset, "split": split, "n": len(tasks),
                           "task_content_sha256": sha256_obj(tasks)})
    provider = cfg["runtime"]["provider"]
    if provider != "fake" and cfg["runtime"].get("fake_noise", 0) != 0:
        raise ValueError("live experiments cannot use synthetic fake_noise")
    evaluations = sum(s["n"] for s in scopes) * len(states) * len(seeds)
    calls_per_task = int(cfg["runtime"]["max_model_calls"])
    minimum = evaluations if provider != "fake" else 0
    logical = logical_request_cap if logical_request_cap is not None else (0 if provider == "fake" else -1)
    physical = physical_request_cap if physical_request_cap is not None else (0 if provider == "fake" else -1)
    if (isinstance(logical, bool) or isinstance(physical, bool)
            or not isinstance(logical, int) or not isinstance(physical, int)
            or logical < minimum or physical < minimum or physical < logical):
        raise ValueError("live experiments require explicit positive logical/physical ceilings covering planned tasks")
    releases = sorted({state.release_id for state in states.values()})
    plan = {"schema": "controlled-visible-experiment-v1", "kind": kind, "spec": spec,
            "configuration_sha256": sha256_obj(cfg), "protocol_sha256": config_fingerprint(cfg),
            "source_tree_sha256": source_tree_fingerprint()["sha256"],
            "freeze_sha256": frozen["freeze_sha256"], "datasets": list(datasets), "splits": list(splits),
            "seeds": list(seeds), "limit": limit, "scopes": scopes, "provider": provider,
            "conditions": {name: _identity(state) for name, state in states.items()},
            "available": available, "release_files": {rid: _release_fingerprint(rid) for rid in releases},
            "planned_task_evaluations": evaluations, "max_model_calls_per_task": calls_per_task,
            "attempt_policy": _attempts_per_operation(cfg),
            "logical_request_cap": logical, "physical_request_cap": physical,
            "final_holdout_read": False, "preflight_model_calls": 0, "publication_authorized": False}
    return {**plan, "plan_sha256": sha256_obj(plan)}


def execute(cfg, plan, expected_plan_sha256):
    if (not os.environ.get("RSI_HOME") or artifacts_root() == repo_root() / "artifacts"
            or artifacts_root() == repo_root()):
        raise ValueError("controlled experiment execution requires an isolated RSI_HOME")
    rebuilt = build_plan(cfg, kind=plan["kind"], spec=plan["spec"], datasets=plan["datasets"],
                         splits=plan["splits"], seeds=plan["seeds"], limit=plan["limit"],
                         logical_request_cap=plan["logical_request_cap"],
                         physical_request_cap=plan["physical_request_cap"])
    if rebuilt != plan or expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("controlled experiment differs from caller-reviewed plan")
    root = artifacts_path("experiments", plan["kind"], plan["plan_sha256"])
    root.mkdir(parents=True, exist_ok=False)  # One attempt, including failed/unknown dispatch.
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", {"status": "started"})
    ledger = root / "call-ledger.jsonl"
    ledger.touch(exist_ok=False)
    try:
        states, available = _conditions(plan["kind"], plan["spec"], cfg)
        with request_scope(plan, ledger):
            if plan["kind"] == "ablate":
                result = ablation.run_ablation(cfg, states, plan["datasets"], splits=plan["splits"],
                    seeds=plan["seeds"], limit=plan["limit"], available=available,
                    execution_plan_sha256=plan["plan_sha256"])
            else:
                result = {"comparisons": []}
                for scope in plan["scopes"]:
                    common = dict(dataset=scope["dataset"], split=scope["split"],
                                  seeds=plan["seeds"], limit=plan["limit"])
                    row = (multiseed.compare(cfg, states["baseline"], states["candidate"], **common)
                           if "candidate" in states else multiseed.eval_seeds(cfg, states["baseline"], **common))
                    result["comparisons"].append(row)
        accounting = summarize_ledger(ledger)
        if accounting["logical_incomplete"] or accounting["physical_incomplete"]:
            raise ValueError("experiment request ledger is incomplete")
        report = {"status": "completed", "plan_sha256": plan["plan_sha256"], "result": result,
                  "request_accounting": accounting, "run_dir": str(root), "publication_authorized": False}
        write_json(root / "report.json", report)
        write_json(root / "status.json", {"status": "completed"})
        return report
    except BaseException as exc:
        write_json(root / "status.json", {"status": "failed", "error_type": type(exc).__name__,
                                          "request_accounting": summarize_ledger(ledger)})
        raise
