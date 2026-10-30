"""反事实重跑评估器（M3）。

为什么需要它：
    M2 选 Skill 用的是"召回覆盖率"这种**代理指标**——它只说明"改了之后会被召回"，
    不能说明"召回之后真的答对了"。M3 开放的工件（Skill 正文、结晶新 Skill）
    直接影响答题策略，代理指标已经不够用，必须真的跑一遍才知道效果。

它做什么：
    给定父状态与试验状态，在同一批 dev_optimize 题目上各跑一次配置的
    task_input 或检索链路，验证答案合同后比对对错，输出 n_fixed / n_broke / net。

它不做什么（别在报告里说反）：
    - provider 由配置决定；离线 fake 结果不能证明真实模型收益
    - 在失败题上选出的 net>0 是候选生成诊断，不是独立确认或逐题根因证明
    - 它**不能**替代 dev_optimize 的正式评测：候选仍要过静态门 + 双分区评测 + 人工 review
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..config import config_fingerprint
from ..data.split import load_tasks
from ..eval.task_rows import validate_task_answer_row
from ..runtime.runner import ArtifactState, TaskRunner
from ..runtime.skills import load_body
from ..util import artifacts_path, sha256_obj
from ..runtime.trace import Trace, new_run_id

REPEATED_SEED_REPORT_SCHEMA = "counterfactual-repeated-seed-v3-balanced-arms"
COUNTERFACTUAL_ARM_ORDER_POLICY = "seed-keyed-task-block-balanced-position-v1"


def _counterfactual_arm_schedule(task_ids: Sequence[str], arm_names: Sequence[str],
                                 seed: int) -> tuple[list[str], dict[str, list[str]]]:
    """Freeze a deterministic randomized block schedule balanced across tasks.

    Each task is an adjacent block containing every arm once. Cyclic rotations
    distribute each arm across positions as evenly as the cohort size allows.
    The seed randomizes both block order and the starting arm order; it does
    not claim to control a remote model's sampling RNG.
    """
    normalized_tasks = [str(value) for value in task_ids]
    normalized_arms = [str(value) for value in arm_names]
    if (not normalized_tasks or any(not value for value in normalized_tasks)
            or len(set(normalized_tasks)) != len(normalized_tasks)):
        raise ValueError("counterfactual schedule requires unique non-empty task IDs")
    if (len(normalized_arms) < 2 or any(not value for value in normalized_arms)
            or len(set(normalized_arms)) != len(normalized_arms)):
        raise ValueError("counterfactual schedule requires unique non-empty arms")

    def key(namespace: str, value: str) -> str:
        return sha256_obj({"seed": int(seed), "namespace": namespace, "value": value})

    task_order = sorted(normalized_tasks, key=lambda value: key("task", value))
    arm_order = sorted(normalized_arms, key=lambda value: key("arm", value))
    schedule: dict[str, list[str]] = {}
    for position, task_id in enumerate(task_order):
        shift = position % len(arm_order)
        schedule[task_id] = arm_order[shift:] + arm_order[:shift]
    return task_order, schedule


class CounterfactualBaselineCache(dict):
    """In-memory baseline answers bound to one frozen intervention protocol."""

    def __init__(self, protocol_sha256: str, rows=None):
        super().__init__(rows or {})
        self.protocol_sha256 = protocol_sha256


def _state_snapshot(state: ArtifactState) -> Dict[str, Any]:
    """Hash runtime-effective artifact content, independent of release IDs."""
    skills = []
    names = set()
    for skill in state.registry.skills:
        name = str(skill.name)
        if name in names:
            raise ValueError("counterfactual artifact registry has duplicate Skill names")
        names.add(name)
        # Storage location is not runtime behavior. The resolved body is
        # included below, so two equivalent release paths must not masquerade
        # as different Skill interventions.
        serialized = skill.to_dict()
        serialized.pop("path", None)
        skills.append({**serialized, "body": load_body(skill)})
    return {
        "system_prompt": str(state.system_prompt),
        "skills": sorted(skills, key=lambda row: str(row["name"])),
        "task_demos": state.task_demos,
    }


def describe_intervention(base_state: ArtifactState,
                          trial_state: ArtifactState) -> Dict[str, Any]:
    """Bind counterfactual scores to the exact artifact changes being tested.

    This is necessary intervention provenance, not causal proof. The output
    deliberately keeps causal attribution false even for a single component.
    """
    base = _state_snapshot(base_state)
    trial = _state_snapshot(trial_state)
    components: list[str] = []
    if base["system_prompt"] != trial["system_prompt"]:
        components.append("system_prompt")
    if sha256_obj(base["task_demos"]) != sha256_obj(trial["task_demos"]):
        components.append("task_demos")

    base_skills = {row["name"]: row for row in base["skills"]}
    trial_skills = {row["name"]: row for row in trial["skills"]}
    for name in sorted(base_skills.keys() | trial_skills.keys()):
        before, after = base_skills.get(name), trial_skills.get(name)
        if before is None or after is None:
            components.append(f"skill_presence:{name}")
            continue
        if (before.get("description") != after.get("description")
                or before.get("triggers") != after.get("triggers")):
            components.append(f"skill_description:{name}")
        if before.get("body") != after.get("body"):
            components.append(f"skill_body:{name}")
        metadata = ("group", "frozen", "license")
        if any(before.get(key) != after.get(key) for key in metadata):
            components.append(f"skill_metadata:{name}")

    base_sha = sha256_obj(base)
    trial_sha = sha256_obj(trial)
    intervention_sha = sha256_obj({
        "base_state_sha256": base_sha,
        "trial_state_sha256": trial_sha,
        "changed_components": components,
    })
    return {
        "state_changed": bool(components),
        "changed_components": components,
        "single_component_intervention": len(components) == 1,
        "base_state_sha256": base_sha,
        "trial_state_sha256": trial_sha,
        "intervention_sha256": intervention_sha,
        "causal_attribution_supported": False,
    }


def task_index(datasets: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """按 task_id 索引已重新校验的 dev_optimize 题目。"""
    idx: Dict[str, Dict[str, Any]] = {}
    for ds in sorted({d for d in datasets if d}):
        rows = load_tasks(ds, splits=["dev_optimize"])
        if any(t.get("split") != "dev_optimize" for t in rows):
            raise ValueError("counterfactual tasks must come from dev_optimize")
        per_dataset = {t["task_id"]: t for t in rows}
        if len(per_dataset) != len(rows):
            raise ValueError("counterfactual task IDs must be unique within each dataset")
        if idx.keys() & per_dataset.keys():
            raise ValueError("counterfactual task IDs must be unique across datasets")
        idx.update(per_dataset)
    return idx


def tasks_of(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把失败样本记录解析回题目对象。"""
    if any(r.get("split") not in (None, "dev_optimize") for r in rows):
        raise ValueError("counterfactual failure rows must come from dev_optimize")
    idx = task_index([str(r.get("dataset", "")) for r in rows])
    out: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        t = idx.get(str(r.get("task_id")))
        if t:
            if t.get("dataset") != r.get("dataset"):
                raise ValueError("counterfactual failure task dataset differs from its scope")
            identity = (str(t.get("dataset")), str(t["task_id"]))
            if identity not in seen:
                out.append(t)
                seen.add(identity)
        elif r.get("split") == "dev_optimize":
            raise ValueError("counterfactual failure task is unavailable in dev_optimize")
    return out


def _validate_outcome(outcome: Any, task: Dict[str, Any], arm: str) -> Dict[str, Any]:
    """Reject incomplete execution and cross-task cached rows before scoring."""
    context = f"counterfactual {arm}:{task['task_id']}"
    row = validate_task_answer_row(
        outcome.to_dict(), task, context=context, require_reported_gold_index=True)
    if (row.get("dataset") != task.get("dataset")
            or row.get("split") != "dev_optimize"):
        raise ValueError(f"{context} outcome differs from the frozen task scope")
    return row


def _preflight_counterfactual_batch(base_state: ArtifactState,
                                    trials: Sequence[Dict[str, Any]],
                                    tasks: Sequence[Dict[str, Any]]) -> None:
    """Validate every intervention and task before a batch spends runtime budget."""
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    if not task_ids or not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("counterfactual task IDs must be non-empty and unique")
    for task in tasks:
        if task.get("split") != "dev_optimize":
            raise ValueError("counterfactual tasks must come from dev_optimize")
        validate_task_answer_row(
            {"task_id": task["task_id"], "status": "abstain", "answer_index": -1,
             "correct": False}, task, context="counterfactual task preflight")

    # State snapshots can fail (for example, duplicate Skill identities or an
    # unreadable effective body). Check all trial states now so a bad later
    # intervention cannot be discovered after earlier trials have spent calls.
    for trial in trials:
        if not isinstance(trial, dict) or not isinstance(
                trial.get("state"), ArtifactState):
            raise ValueError("counterfactual trials require ArtifactState values")
        describe_intervention(base_state, trial["state"])


def _trace_provenance(outcome: Any) -> Dict[str, List[str]]:
    """Return task-outcome trace identities without copying event payloads."""
    run_ids: list[str] = []
    event_ids: list[str] = []
    for event in list(getattr(outcome, "events", []) or []):
        if not isinstance(event, dict):
            continue
        run_id = event.get("run_id")
        event_id = event.get("event_id")
        if isinstance(run_id, str) and run_id and run_id not in run_ids:
            run_ids.append(run_id)
        if isinstance(event_id, str) and event_id and event_id not in event_ids:
            event_ids.append(event_id)
    artifact_path = getattr(outcome, "trace_artifact_path", "")
    paths = [artifact_path] if isinstance(artifact_path, str) and artifact_path else []
    return {"trace_run_ids": run_ids, "trace_event_ids": event_ids,
            "trace_artifact_paths": paths}


def run_one(cfg: Dict[str, Any], state: ArtifactState, task: Dict[str, Any],
            seed: int = 0) -> Any:
    """Run one optimize-scope arm and retain a resolvable, run-scoped trace."""
    if task.get("split") != "dev_optimize":
        raise ValueError("counterfactual traces are limited to dev_optimize")
    run_id = new_run_id("counterfactual")
    relative_path = f"counterfactual_traces/{run_id}/trace.jsonl"
    trace = Trace(run_id, task_id=str(task.get("task_id") or ""),
                  out_path=artifacts_path(*relative_path.split("/")))
    outcome = TaskRunner(cfg, state, trace=trace, seed=int(seed)).run(task)
    # Keep the normal TaskOutcome wire format unchanged; this reference is
    # consumed only by counterfactual report construction.
    outcome.trace_artifact_path = relative_path
    return outcome


def rerun(cfg: Dict[str, Any], base_state: ArtifactState, trial_state: ArtifactState,
          tasks: Sequence[Dict[str, Any]], seed: int = 0,
          base_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """在 `tasks` 上各跑一遍 base 与 trial，返回逐题与汇总的反事实结果。

    `base_results` 可传入本函数返回的、绑定同一协议的缓存（task_id -> outcome），
    避免重跑基线；命中缓存时报告会明确标记它不是同一时间块内的新配对。
    """
    task_ids = [str(t.get("task_id") or "") for t in tasks]
    if not task_ids or not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("counterfactual task IDs must be non-empty and unique")
    # Check the whole input/cache scope before starting any new executions.
    for t in tasks:
        if t.get("split") != "dev_optimize":
            raise ValueError("counterfactual tasks must come from dev_optimize")
        validate_task_answer_row(
            {"task_id": t["task_id"], "status": "abstain", "answer_index": -1,
             "correct": False}, t, context="counterfactual task preflight")
    task_order, schedule = _counterfactual_arm_schedule(
        task_ids, ["baseline", "trial"], int(seed))
    schedule_sha256 = sha256_obj(schedule)
    base_snapshot = _state_snapshot(base_state)
    trial_snapshot = _state_snapshot(trial_state)
    protocol_sha = sha256_obj({
        "schema": "counterfactual-paired-rerun-v2-balanced-arms",
        "config_sha256": config_fingerprint(cfg),
        "base_state_sha256": sha256_obj(base_snapshot),
        "trial_state_sha256": sha256_obj(trial_snapshot),
        "seed": int(seed),
        "tasks": sorted(tasks, key=lambda t: str(t["task_id"])),
        "arm_order_policy": COUNTERFACTUAL_ARM_ORDER_POLICY,
        "arm_order_schedule_sha256": schedule_sha256,
    })
    if base_results is not None and (
            not isinstance(base_results, CounterfactualBaselineCache)
            or base_results.protocol_sha256 != protocol_sha):
        raise ValueError("counterfactual baseline cache lacks a matching frozen protocol")
    intervention = describe_intervention(base_state, trial_state)
    cache = CounterfactualBaselineCache(protocol_sha, base_results)
    for t in tasks:
        if str(t["task_id"]) in cache:
            _validate_outcome(cache[str(t["task_id"])], t, "baseline")
    fixed = 0
    broke = 0
    base_correct = 0
    trial_correct = 0
    task_by_id = {str(t["task_id"]): t for t in tasks}
    outcome_pairs: Dict[str, Dict[str, Any]] = {}
    for tid in task_order:
        t = task_by_id[tid]
        pair = outcome_pairs[tid] = {}
        for arm in schedule[tid]:
            if arm == "baseline" and tid in cache:
                pair[arm] = cache[tid]
                continue
            arm_state = base_state if arm == "baseline" else trial_state
            result = run_one(cfg, arm_state, t, seed)
            _validate_outcome(result, t, arm)
            pair[arm] = result
            if arm == "baseline":
                cache[tid] = result

    details: List[Dict[str, Any]] = []
    for t in tasks:
        tid = str(t["task_id"])
        b = outcome_pairs[tid]["baseline"]
        c = outcome_pairs[tid]["trial"]
        base_correct += int(bool(b.correct))
        trial_correct += int(bool(c.correct))
        if c.correct and not b.correct:
            fixed += 1
        elif b.correct and not c.correct:
            broke += 1
        base_trace = _trace_provenance(b)
        trial_trace = _trace_provenance(c)
        details.append({
            "task_id": tid,
            "dataset": t.get("dataset", ""),
            "base_correct": bool(b.correct),
            "trial_correct": bool(c.correct),
            "base_answer": int(b.answer_index),
            "trial_answer": int(c.answer_index),
            "base_trace_run_ids": base_trace["trace_run_ids"],
            "base_trace_event_ids": base_trace["trace_event_ids"],
            "base_trace_artifact_paths": base_trace["trace_artifact_paths"],
            "trial_trace_run_ids": trial_trace["trace_run_ids"],
            "trial_trace_event_ids": trial_trace["trace_event_ids"],
            "trial_trace_artifact_paths": trial_trace["trace_artifact_paths"],
            "planned_arm_order": list(schedule[tid]),
            "baseline_reused_from_prior_run": tid in (base_results or {}),
            "gold_missing_in_corpus": not bool(b.gold_in_corpus),
        })
    return {
        "seed": int(seed),
        "n": len(tasks),
        "base_correct": base_correct,
        "trial_correct": trial_correct,
        "n_fixed": fixed,
        "n_broke": broke,
        "net": fixed - broke,
        "base_results": cache,
        "details": details,
        "evidence_scope": "dev_optimize_candidate_intervention",
        "attribution_is_causal": False,
        "independent_confirmation": False,
        "intervention": intervention,
        "protocol_sha256": protocol_sha,
        "arm_order_policy": COUNTERFACTUAL_ARM_ORDER_POLICY,
        "arm_order_schedule_sha256": schedule_sha256,
        "arm_order_schedule": schedule,
        "task_execution_order": task_order,
    }


def rerun_many(cfg: Dict[str, Any], base_state: ArtifactState,
               trials: Sequence[Dict[str, Any]], tasks: Sequence[Dict[str, Any]],
               seed: int = 0) -> List[Dict[str, Any]]:
    """Run baseline and all trial arms in randomized, task-local blocks.

    `trials` 每项形如 {"name": ..., "state": ArtifactState, ...}。Each task
    uses one shared baseline observation and one observation per candidate, but
    arm execution is interleaved within that task block and position-balanced
    across tasks. Request count is unchanged; reports still make no causal claim.
    """
    if not trials:
        raise ValueError("counterfactual trial batch must be non-empty")
    _preflight_counterfactual_batch(base_state, trials, tasks)
    task_ids = [str(task["task_id"]) for task in tasks]
    trial_arms = [f"trial_{index}" for index in range(len(trials))]
    task_order, schedule = _counterfactual_arm_schedule(
        task_ids, ["baseline", *trial_arms], int(seed))
    schedule_sha256 = sha256_obj(schedule)
    base_snapshot_sha256 = sha256_obj(_state_snapshot(base_state))
    trial_snapshot_hashes = [sha256_obj(_state_snapshot(trial["state"]))
                             for trial in trials]
    protocol_sha256 = sha256_obj({
        "schema": "counterfactual-multiarm-rerun-v2-balanced-arms",
        "config_sha256": config_fingerprint(cfg),
        "base_state_sha256": base_snapshot_sha256,
        "trial_state_sha256s": trial_snapshot_hashes,
        "task_material_sha256": sha256_obj(
            sorted(tasks, key=lambda row: str(row["task_id"]))),
        "seed": int(seed),
        "arm_order_policy": COUNTERFACTUAL_ARM_ORDER_POLICY,
        "arm_order_schedule_sha256": schedule_sha256,
    })
    task_by_id = {str(task["task_id"]): task for task in tasks}
    observed: Dict[str, Dict[str, Any]] = {}
    states = {"baseline": base_state}
    states.update({arm: trial["state"]
                   for arm, trial in zip(trial_arms, trials)})
    for task_id in task_order:
        task = task_by_id[task_id]
        arms = observed[task_id] = {}
        for arm in schedule[task_id]:
            outcome = run_one(cfg, states[arm], task, int(seed))
            _validate_outcome(outcome, task, arm)
            arms[arm] = outcome

    out: List[Dict[str, Any]] = []
    for trial_index, tr in enumerate(trials):
        trial_arm = trial_arms[trial_index]
        intervention = describe_intervention(base_state, tr["state"])
        fixed = broke = base_correct = trial_correct = 0
        details: List[Dict[str, Any]] = []
        for task in tasks:
            task_id = str(task["task_id"])
            baseline = observed[task_id]["baseline"]
            candidate = observed[task_id][trial_arm]
            base_correct += int(bool(baseline.correct))
            trial_correct += int(bool(candidate.correct))
            if candidate.correct and not baseline.correct:
                fixed += 1
            elif baseline.correct and not candidate.correct:
                broke += 1
            base_trace = _trace_provenance(baseline)
            trial_trace = _trace_provenance(candidate)
            details.append({
                "task_id": task_id,
                "dataset": task.get("dataset", ""),
                "base_correct": bool(baseline.correct),
                "trial_correct": bool(candidate.correct),
                "base_answer": int(baseline.answer_index),
                "trial_answer": int(candidate.answer_index),
                "base_trace_run_ids": base_trace["trace_run_ids"],
                "base_trace_event_ids": base_trace["trace_event_ids"],
                "base_trace_artifact_paths": base_trace["trace_artifact_paths"],
                "trial_trace_run_ids": trial_trace["trace_run_ids"],
                "trial_trace_event_ids": trial_trace["trace_event_ids"],
                "trial_trace_artifact_paths": trial_trace["trace_artifact_paths"],
                "planned_arm_order": list(schedule[task_id]),
                "baseline_shared_with_candidate_arms": True,
                "gold_missing_in_corpus": not bool(baseline.gold_in_corpus),
            })
        rec = dict(tr)
        rec.pop("state", None)
        rec.update({
            "n": len(tasks),
            "base_correct": base_correct,
            "trial_correct": trial_correct,
            "n_fixed": fixed,
            "n_broke": broke,
            "net": fixed - broke,
            "details": details,
            "evidence_scope": "dev_optimize_candidate_intervention",
            "attribution_is_causal": False,
            "independent_confirmation": False,
            "intervention": intervention,
            "protocol_sha256": protocol_sha256,
            "arm_order_policy": COUNTERFACTUAL_ARM_ORDER_POLICY,
            "arm_order_schedule_sha256": schedule_sha256,
            "arm_order_schedule": schedule,
            "task_execution_order": task_order,
            "baseline_shared_across_trials": True,
            "arms_executed_within_task_blocks": True,
        })
        out.append(rec)
    return out


def rerun_many_seeds(cfg: Dict[str, Any], base_state: ArtifactState,
                     trials: Sequence[Dict[str, Any]],
                     tasks: Sequence[Dict[str, Any]],
                     seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Repeat paired candidate interventions across a frozen seed list.

    Each seed gets a separate baseline cache. The report summarizes task-seed
    pairs and retains per-task answer/correctness variation so stochastic
    instability is visible. This is still optimize-scope candidate diagnosis;
    repeated seeds do not by themselves establish a task-level cause or
    independent confirmation.
    """
    seed_values = list(seeds)
    if (not seed_values
            or any(type(seed) is not int or seed < 0 for seed in seed_values)
            or len(set(seed_values)) != len(seed_values)):
        raise ValueError("counterfactual requires unique non-negative integer seeds")
    if not trials:
        raise ValueError("counterfactual repeated-seed trials must be non-empty")
    names = [str(trial.get("name") or "").strip() for trial in trials]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("counterfactual repeated-seed trial names must be non-empty and unique")
    if any(not isinstance(trial.get("state"), ArtifactState) for trial in trials):
        raise ValueError("counterfactual repeated-seed trials require ArtifactState values")
    if not tasks:
        raise ValueError("counterfactual repeated-seed task scope must be non-empty")
    _preflight_counterfactual_batch(base_state, trials, tasks)

    per_trial_seed_rows: list[list[dict[str, Any]]] = [
        [] for _ in trials
    ]
    for seed in seed_values:
        rows = rerun_many(cfg, base_state, trials, tasks, seed=seed)
        if len(rows) != len(trials):
            raise RuntimeError("counterfactual trial count changed across seeds")
        for index, row in enumerate(rows):
            per_trial_seed_rows[index].append(row)

    output: list[dict[str, Any]] = []
    for index, seed_rows in enumerate(per_trial_seed_rows):
        first = seed_rows[0]
        is_aa_control = first["intervention"].get("state_changed") is False
        task_ids = [str(row.get("task_id") or "") for row in first["details"]]
        if not all(task_ids) or len(set(task_ids)) != len(task_ids):
            raise ValueError("counterfactual details must have unique task IDs")
        by_seed = []
        for seed, row in zip(seed_values, seed_rows):
            details = row.get("details") or []
            if [str(item.get("task_id") or "") for item in details] != task_ids:
                raise ValueError("counterfactual task order changed across seeds")
            if row.get("intervention") != first.get("intervention"):
                raise ValueError("counterfactual intervention changed across seeds")
            by_seed.append({
                "seed": seed,
                "base_correct": int(row["base_correct"]),
                "trial_correct": int(row["trial_correct"]),
                "n_fixed": int(row["n_fixed"]),
                "n_broke": int(row["n_broke"]),
                "net": int(row["net"]),
                "protocol_sha256": str(row["protocol_sha256"]),
                "arm_order_policy": str(row["arm_order_policy"]),
                "arm_order_schedule_sha256": str(
                    row["arm_order_schedule_sha256"]),
                "planned_arm_orders": {
                    str(item["task_id"]): list(item["planned_arm_order"])
                    for item in details
                },
                "details": details,
            })

        per_task = []
        for task_index, task_id in enumerate(task_ids):
            task_rows = [row["details"][task_index] for row in by_seed]
            base_answers = [int(row["base_answer"]) for row in task_rows]
            trial_answers = [int(row["trial_answer"]) for row in task_rows]
            base_correct_values = [bool(row["base_correct"]) for row in task_rows]
            trial_correct_values = [bool(row["trial_correct"]) for row in task_rows]
            fixed_seeds = [seed for seed, base, trial in zip(
                seed_values, base_correct_values, trial_correct_values)
                if trial and not base]
            broke_seeds = [seed for seed, base, trial in zip(
                seed_values, base_correct_values, trial_correct_values)
                if base and not trial]
            task_summary = {
                "task_id": task_id,
                "dataset": str(task_rows[0].get("dataset", "")),
                "base_correct_by_seed": base_correct_values,
                "trial_correct_by_seed": trial_correct_values,
                "base_answer_by_seed": base_answers,
                "trial_answer_by_seed": trial_answers,
                "base_trace_run_ids_by_seed": [
                    list(row.get("base_trace_run_ids") or []) for row in task_rows],
                "base_trace_event_ids_by_seed": [
                    list(row.get("base_trace_event_ids") or []) for row in task_rows],
                "base_trace_artifact_paths_by_seed": [
                    list(row.get("base_trace_artifact_paths") or []) for row in task_rows],
                "planned_arm_orders_by_seed": [
                    list(row.get("planned_arm_order") or []) for row in task_rows],
                "trial_trace_run_ids_by_seed": [
                    list(row.get("trial_trace_run_ids") or []) for row in task_rows],
                "trial_trace_event_ids_by_seed": [
                    list(row.get("trial_trace_event_ids") or []) for row in task_rows],
                "trial_trace_artifact_paths_by_seed": [
                    list(row.get("trial_trace_artifact_paths") or []) for row in task_rows],
                "fixed_seeds": fixed_seeds,
                "broke_seeds": broke_seeds,
                "base_answer_varied": len(set(base_answers)) > 1,
                "trial_answer_varied": len(set(trial_answers)) > 1,
            }
            if is_aa_control:
                answer_disagreement_seeds = [
                    seed for seed, base, trial in zip(
                        seed_values, base_answers, trial_answers)
                    if base != trial]
                correctness_disagreement_seeds = [
                    seed for seed, base, trial in zip(
                        seed_values, base_correct_values, trial_correct_values)
                    if base != trial]
                task_summary.update({
                    "aa_answer_disagreement_seeds": answer_disagreement_seeds,
                    "aa_correctness_disagreement_seeds": correctness_disagreement_seeds,
                })
            per_task.append(task_summary)

        paired_observations = len(task_ids) * len(seed_values)
        base_correct = sum(row["base_correct"] for row in by_seed)
        trial_correct = sum(row["trial_correct"] for row in by_seed)
        n_fixed = sum(row["n_fixed"] for row in by_seed)
        n_broke = sum(row["n_broke"] for row in by_seed)
        seed_protocols = [row["protocol_sha256"] for row in by_seed]
        protocol_sha = sha256_obj({
            "schema": REPEATED_SEED_REPORT_SCHEMA,
            "config_sha256": config_fingerprint(cfg),
            "intervention_sha256": first["intervention"]["intervention_sha256"],
            "task_ids": task_ids,
            "seeds": seed_values,
            "seed_protocol_sha256": seed_protocols,
        })
        candidate_row = dict(trials[index])
        candidate_row.pop("state", None)
        candidate_row.update({
            "schema": REPEATED_SEED_REPORT_SCHEMA,
            "comparison_role": ("aa_control" if is_aa_control
                                else "candidate_intervention"),
            "seeds": seed_values,
            "n_tasks": len(task_ids),
            "n_seeds": len(seed_values),
            "paired_observations": paired_observations,
            "base_correct": base_correct,
            "trial_correct": trial_correct,
            "base_accuracy": base_correct / paired_observations,
            "trial_accuracy": trial_correct / paired_observations,
            "mean_accuracy_delta": (
                trial_correct - base_correct) / paired_observations,
            "n_fixed": n_fixed,
            "n_broke": n_broke,
            "net": n_fixed - n_broke,
            "seed_results": [{key: value for key, value in row.items()
                              if key != "details"} for row in by_seed],
            "per_task": per_task,
            "aa_control_diagnostics": ({
                "same_effective_state": True,
                "paired_observations": paired_observations,
                "answer_disagreement_count": sum(
                    len(row["aa_answer_disagreement_seeds"]) for row in per_task),
                "correctness_disagreement_count": sum(
                    len(row["aa_correctness_disagreement_seeds"]) for row in per_task),
                "answer_disagreement_rate": sum(
                    len(row["aa_answer_disagreement_seeds"]) for row in per_task)
                    / paired_observations,
                "correctness_disagreement_rate": sum(
                    len(row["aa_correctness_disagreement_seeds"]) for row in per_task)
                    / paired_observations,
            } if is_aa_control else None),
            "evidence_scope": "dev_optimize_candidate_intervention_repeated_seeds",
            "attribution_is_causal": False,
            "independent_confirmation": False,
            "intervention": first["intervention"],
            "seed_protocol_sha256": seed_protocols,
            "protocol_sha256": protocol_sha,
        })
        output.append(candidate_row)
    return output
