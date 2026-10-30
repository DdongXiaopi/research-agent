"""一轮进化的编排（Spec §0 主线闭环）。

    baseline 评测 → 收集失败样本 → 4 类归因 → 生成候选 → 静态门
    → dev_optimize 评测 → selection_gate 聚合 → 判定（拟接受/拟拒绝）→ 写 review 请求

判定只给"拟接受 / 拟拒绝"，**真正的发布必须走人工 review**（FR-205）。
"""

from __future__ import annotations

import copy
import math
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from ..config import config_fingerprint
from ..contracts.types import ATTRIBUTION_TARGET, Candidate, content_sha256, new_id
from ..eval.harness import evaluate, gate_summary
from ..eval.leakage import assert_gate_is_aggregate
from ..eval.stats import paired_bootstrap_binary
from ..governance import release as rel
from ..governance import review as rv
from ..governance.budget import check_tolerances
from ..runtime.channels import globally_active_behaviors
from ..governance.policy import apply_gate, rejection_reason
from ..runtime.runner import ArtifactState
from ..util import (now_iso, artifacts_path, path, read_json, read_jsonl,
                    sha256_obj, write_json)
from .attribution import summarize
from .reflection import generate_candidate
from . import candidate_memory

ROUNDS_ROOT = artifacts_path("rounds")

# These labels nominate an artifact to test; they are not causal diagnoses.
# Share the mapping between target selection and reporting so a new experiment
# cannot silently inherit the attribution selection mode.
EXPLICIT_ARTIFACT_TARGETS = {
    "experiment.system_prompt": "system_prompt",
    "experiment.skill_description": "skill_description",
    "experiment.skill_body": "skill_body",
    "experiment.task_demos": "task_demos",
    "experiment.task_program": "task_program",
    "experiment.new_skill": "new_skill",
}


def _binary_answer_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    answered = sum(int(row.get("answer_index", -1)) >= 0 for row in rows)
    return {"n": n, "accuracy": correct / n if n else 0.0,
            "coverage": answered / n if n else 0.0,
            "answered_precision": correct / answered if answered else 0.0}


def evolution_eligible_failures(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep task-answer failures for artifact experiments; exclude system faults.

    Explicit Prompt/Skill/demo experiments test a chosen artifact dimension and
    therefore do not need the heuristic attribution class to nominate it. They
    can include unresolved ``unknown`` errors, but never infrastructure,
    evaluator, retrieval, or tool failures as reflection examples.
    """
    eligible = set(ATTRIBUTION_TARGET) | {"unknown"}
    return [row for row in failures
            if row.get("effective_attributed_class", row.get("attributed_class")) in eligible
            and row.get("outcome_status", row.get("status"))
            in (None, "ok", "abstain")]


def select_candidate_work_items(
        by_class: Dict[str, List[Dict[str, Any]]],
        all_failures: List[Dict[str, Any]],
        *, gepa_enabled: bool, gepa_target: str,
        fewshot_enabled: bool, joint_program_enabled: bool,
        max_candidates: int, input_mode: str,
        new_skill_enabled: bool = False) -> List[tuple[str, List[Dict[str, Any]]]]:
    """Select artifact experiments independently of uncertain causal labels."""
    evolvable = evolution_eligible_failures(all_failures)
    if new_skill_enabled:
        return [("experiment.new_skill", evolvable)] if evolvable else []
    if joint_program_enabled:
        return [("experiment.task_program", evolvable)] if evolvable else []
    if fewshot_enabled:
        return [("experiment.task_demos", evolvable)] if evolvable else []
    if gepa_enabled and gepa_target == "skill_description":
        return [("experiment.skill_description", evolvable)] if evolvable else []
    if gepa_enabled and gepa_target == "skill_body":
        return [("experiment.skill_body", evolvable)] if evolvable else []
    if gepa_enabled and gepa_target == "system_prompt":
        return [("experiment.system_prompt", evolvable)] if evolvable else []

    eligible_by_class: Dict[str, List[Dict[str, Any]]] = {}
    for row in evolvable:
        label = row.get("effective_attributed_class", row.get("attributed_class"))
        eligible_by_class.setdefault(str(label), []).append(row)
    work = actionable_failure_classes(eligible_by_class, max_candidates)
    # Retrieval-mode legacy/template Prompt evolution still needs an explicit
    # Prompt experiment for unresolved answer failures. Keep it out of the
    # causal taxonomy and label the selection as an artifact experiment.
    if (not fewshot_enabled and input_mode == "retrieval"
            and len(work) < max_candidates):
        unresolved = [row for row in evolvable
                      if row.get("effective_attributed_class",
                                 row.get("attributed_class")) == "unknown"]
        if unresolved:
            work.append(("experiment.system_prompt", unresolved))
    return work


def _nested_demo_confirmation_pair(report: Dict[str, Any], candidate: Candidate,
                                  seed: int, baseline_run_id: str,
                                  baseline_rows: List[Dict[str, Any]]) -> tuple[
                                       Any, Dict[str, Any], Dict[str, int], Dict[str, int]]:
    """Reuse one-shot confirmation results for task-demo optimize comparison.

    Task-demo examples are drawn from the `dev_optimize` train IDs. Re-scoring
    the candidate on the full dev split would put those same demonstrations
    back into its evaluation set and contaminate optimize metrics/CIs. The
    nested confirmation rows are already a persisted, disjoint candidate
    evaluation; pair them with baseline rows from precisely those IDs.
    """
    from .task_demo_search import SEARCH_PROTOCOL_VERSION

    if report.get("search_protocol_version") != SEARCH_PROTOCOL_VERSION:
        raise RuntimeError(
            "task-demo round requires class-aware nested-confirmation search results")
    confirmation = report.get("internal_confirmation") or {}
    if confirmation.get("status") != "passed":
        raise RuntimeError("task-demo round requires a passed one-winner confirmation")
    required_class_axes = {"per_gold_label.accuracy", "per_gold_label.coverage"}
    recorded_axes = set((report.get("selection_criteria") or {}).get(
        "all_seed_nonregression") or [])
    if not required_class_axes.issubset(recorded_axes):
        raise RuntimeError("task-demo confirmation report lacks class-level nonregression axes")
    if (candidate.generator.get("search_report") is None or
            candidate.generator.get("selected_idx") != report.get("selected_idx")):
        raise RuntimeError("task-demo candidate and confirmation report identity mismatch")
    if sha256_obj(candidate.new_content) != report.get("selected_sha256"):
        raise RuntimeError("task-demo candidate content does not match confirmed artifact")

    train_ids = {str(value) for value in report.get("train_ids") or []}
    confirmation_ids = {str(value) for value in report.get("confirmation_ids") or []}
    demo_ids = {str(value) for value in report.get("selected_demo_ids") or []}
    if not confirmation_ids or train_ids & confirmation_ids:
        raise RuntimeError("task-demo confirmation IDs overlap the training pool")
    if demo_ids & confirmation_ids:
        raise RuntimeError("a task-demo example appears in its confirmation evaluation")
    if sorted(confirmation_ids) != sorted(
            str(value) for value in candidate.generator.get("confirmation_ids") or []):
        raise RuntimeError("candidate confirmation IDs differ from search provenance")
    if candidate.generator.get("selected_demo_ids") is not None and \
            {str(value) for value in candidate.generator["selected_demo_ids"]} != demo_ids:
        raise RuntimeError("candidate demo IDs differ from confirmed artifact")

    seed_entries = list(report.get("baseline_metrics_by_seed") or [])
    run_ids = list(report.get("baseline_seed_run_ids") or [])
    seed_index = next((i for i, row in enumerate(seed_entries)
                       if int(row.get("seed", -1)) == int(seed)), None)
    if seed_index is None or seed_index >= len(run_ids) or \
            str(run_ids[seed_index]) != str(baseline_run_id):
        raise RuntimeError("task-demo confirmation baseline run does not match round baseline")

    candidate_rows = [row for row in confirmation.get("rows") or []
                      if int(row.get("seed", -1)) == int(seed)]
    candidate_by_id = {str(row.get("task_id") or ""): row for row in candidate_rows}
    baseline_by_id = {str(row.get("task_id") or ""): row for row in baseline_rows}
    if set(candidate_by_id) != confirmation_ids:
        raise RuntimeError("candidate confirmation rows do not exactly match confirmation IDs")
    if not confirmation_ids.issubset(baseline_by_id):
        raise RuntimeError("baseline run is missing task-demo confirmation IDs")

    selected_rows = [candidate_by_id[task_id] for task_id in sorted(confirmation_ids)]
    paired_baseline_rows = [baseline_by_id[task_id] for task_id in sorted(confirmation_ids)]
    candidate_metrics = _binary_answer_metrics(selected_rows)
    baseline_metrics = _binary_answer_metrics(paired_baseline_rows)
    candidate_correct = {task_id: int(bool(row.get("correct")))
                         for task_id, row in candidate_by_id.items()}
    baseline_correct = {task_id: int(bool(row.get("correct")))
                        for task_id, row in baseline_by_id.items()
                        if task_id in confirmation_ids}
    evaluation = SimpleNamespace(
        candidate_id=candidate.candidate_id, split="dev_optimize", seed=int(seed),
        metric_values=candidate_metrics, n_tasks=len(confirmation_ids),
        run_id=f"{report.get('search_protocol_sha256', 'search')}:{seed}:confirmation",
        per_task=[{"task_id": task_id, "correct": bool(candidate_correct[task_id]),
                   "attributed_class": None}
                  for task_id in sorted(candidate_correct)])
    return evaluation, baseline_metrics, baseline_correct, candidate_correct


def _build_reflection_backend(cfg: Dict[str, Any],
                              state: Any = None) -> Optional[Any]:
    """按 `evolution.reflection_backend` 选反思后端。

    - `deterministic-stub`（默认）→ None，走 `reflection.py` 的模板库；
    - `llm` → `LlmBackend`（真实模型反思，需要 models.reflection.provider=http）；
    - `replay` → `ReplayBackend`，用固化响应零 API 跑通链路。

    为什么要显式配置而不是"有 key 就用模型"：反思后端换了之后，候选文本的
    来源就变了 —— 这是**协议级**变更，必须写在配置里、能被指纹和报告看到，
    不能靠环境隐式决定。
    """
    ev = cfg.get("evolution", {}) or {}
    kind = str(ev.get("reflection_backend", "deterministic-stub")).lower()
    # 【2026-09-27】产出形态：`patch`（默认，追加一句）/ `rewrite`（GEPA 式整段改写）。
    # 只有 llm 类后端支持 rewrite —— 模板与回放的候选本来就是枚举出来的句子，
    # 谈不上"整段改写"。
    mode = str(ev.get("proposer", "patch")).lower()
    if kind in ("llm", "llm-backend", "http"):
        if mode == "rewrite":
            from .fewshot import bootstrap_demos, render_demos_block
            from .reflection_backend import RewriteBackend

            # few-shot 只在 rewrite 模式下引导：它是候选空间里另一个未被探索的轴
            # （对照 dspy MIPROv2：候选 = instruction 变体 × 示例组合）。
            demos = ""
            if bool(ev.get("fewshot_enabled", True)):
                ds = str((cfg.get("evaluation", {}) or {}).get("datasets", ["pubmedqa"])[0])
                got = bootstrap_demos(
                    cfg, state, ds,
                    k=int(ev.get("fewshot_k", 3)),
                    seed=int((cfg.get("evaluation", {}) or {}).get("seed", 20260924)),
                    limit=ev.get("fewshot_scan_limit"),
                )
                demos = render_demos_block(got)
            return RewriteBackend(cfg=cfg, max_attempts=int(
                ev.get("reflection_max_attempts", 2)), demos=demos)
        from .reflection_backend import LlmBackend

        return LlmBackend(cfg=cfg, max_attempts=int(ev.get("reflection_max_attempts", 2)))
    if kind == "replay":
        from .reflection_backend import ReplayBackend

        fixture = ev.get("reflection_fixture")
        if not fixture:
            raise ValueError("reflection_backend=replay 需要 evolution.reflection_fixture")
        return ReplayBackend(path(fixture))
    return None


def _assert_protocol_consistent(cfg: Dict[str, Any]) -> None:
    """【B / P5 fail-loud】进化轮次使用的协议必须与封存 final_holdout 基线一致。

    否则候选在新协议下被度量，却要用旧协议封存的基线做终评配对 ——
    Δ 不可解释、且静默过期（gepa api.py:286 范式：配了却测不到 / 对不齐就直接报错，
    绝不静默放行）。这里对齐该思想：config 的指纹与基线协议字段不一致 → 直接 raise。
    """
    try:
        from ..eval.final_eval import (ProtocolMismatch, check_protocol,
                                        load_baseline)
    except Exception:
        return
    # 【显式豁免，非静默】离线合成夹具（synthetic-v2）跑的是"闭环机制"，
    # 其产物按设计**不会**与封存基线配对（真实语料下的收益结论只能由 http 夹具给出）。
    # 这类运行必须显式声明 `evaluation.sealed_pairing: false`，且该声明会写进 round 报告，
    # 仍能事后审计 —— 不允许"什么都不写就悄悄跳过"。
    if cfg.get("evaluation", {}).get("sealed_pairing", True) is False:
        return
    try:
        baseline = load_baseline()
    except Exception:
        # 基线尚未封存：进化轮次无法被终评，但这是另一条规则，不在此拦截
        return
    proto = check_protocol(cfg, baseline)
    if not proto["ok"]:
        raise ProtocolMismatch(
            "进化轮次所用协议与封存 final_holdout 基线不一致，候选无法在同源协议下被终评配对："
            + str(proto["diffs"])
            + "。请先以当前夹具重跑 final 基线（final-baseline --force），"
            + "或修正 config 使其与基线协议一致后再进化。"
        )


def _baseline_metrics(cfg: Dict[str, Any], state: ArtifactState, dataset: str, split: str,
                      seed: int, limit: Optional[int], cfg_fp: str) -> Dict[str, Any]:
    res = evaluate(cfg, state, dataset, split, seed=seed, limit=limit,
                   run_tag="baseline", write_per_task=(split == "dev_optimize"), cfg_fp=cfg_fp)
    return {"metrics": res.metric_values, "run_id": res.run_id, "result": res,
            "class_metrics": res.class_metrics}


def _reviewed_baselines(cfg, state, datasets, seeds, limit, document):
    """Validate completed receipts and human edits before any fresh model call."""
    from ..eval.harness import prepare_tasks
    from ..eval.run_manifest import source_tree_fingerprint
    from ..util import sha256_file
    from .attribution_review import apply_attribution_corrections
    sources = document.get("source_run_ids") or []
    expected = {(dataset, seed) for dataset in datasets for seed in seeds}
    restored, failures = {}, []
    for run_id in sources:
        if not isinstance(run_id, str) or not re.fullmatch(r"run-[a-f0-9-]+", run_id):
            raise ValueError("invalid reviewed run identity")
        root = artifacts_path("runs", run_id)
        receipt = read_json(root / "receipt.json")
        unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if (receipt.get("run_id") != run_id or receipt.get("status") != "completed"
                or sha256_obj(unsigned) != receipt.get("receipt_sha256")
                or any(sha256_file(root / name) != value for name, value in receipt["files"].items())):
            raise ValueError("reviewed execution receipt changed")
        manifest = read_json(root / "manifest.json")
        summary = read_json(root / "summary.json")
        dataset, seed = manifest.get("dataset"), manifest.get("seed")
        key = (dataset, seed)
        if (key not in expected or key in restored or manifest.get("split") != "dev_optimize"
                or manifest.get("release_id") != state.release_id
                or manifest["configuration"]["config_sha256"] != sha256_obj(cfg)
                or manifest["configuration"]["protocol_sha256"] != config_fingerprint(cfg)
                or manifest["code"]["source_tree_sha256"] != source_tree_fingerprint()["sha256"]
                or manifest["artifacts"]["system_prompt_sha256"] != sha256_obj(state.system_prompt)
                or manifest["artifacts"]["skill_registry_fingerprint"] != state.registry.fingerprint()
                or manifest["artifacts"]["task_demos_sha256"] != sha256_obj(state.task_demos)
                or summary.get("manifest_sha256") != sha256_obj(manifest)):
            raise ValueError("reviewed baseline does not match current scope/protocol/artifacts")
        tasks, _ = prepare_tasks(cfg, dataset, "dev_optimize", limit)
        if sha256_obj(tasks) != manifest["data"]["evaluated_task_content_sha256"]:
            raise ValueError("reviewed baseline task content/scope changed")
        task_rows = read_jsonl(root / "per_task.jsonl")
        if {r["task_id"] for r in task_rows} != {r["task_id"] for r in tasks}:
            raise ValueError("reviewed baseline output cohort changed")
        restored[key] = {"metrics": summary["metrics"], "run_id": run_id,
            "class_metrics": summary["class_metrics"],
            "result": SimpleNamespace(per_task=task_rows)}
        run_failures = read_jsonl(root / "failures.jsonl")
        if (run_id in document.get("empty_source_run_ids", [])) != (not run_failures):
            raise ValueError("reviewed empty-run declaration differs from completed receipt")
        failures.extend(run_failures)
    if set(restored) != expected:
        raise ValueError("manual correction resume needs one completed baseline receipt for every dataset/seed")
    apply_attribution_corrections(failures, document)
    return restored


def _candidate_state(state: ArtifactState, cand: Candidate) -> ArtifactState:
    """按候选类型构造"候选状态"（实现见 `evolution.apply`，与消融/反事实共用一套）。"""
    from .apply import state_with

    return state_with(state, cand)


def gate_delta_threshold(delta_pp: float, min_delta_pp: float,
                         min_prob: float = 0.90) -> Dict[str, Any]:
    """selection_gate 仅暴露聚合值，因此只能按 gate 自身差值判定。"""
    values = (float(delta_pp), float(min_delta_pp), float(min_prob))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gate threshold values must be finite")
    delta_pp, min_delta_pp, min_prob = values
    if min_delta_pp < 0:
        raise ValueError("gate_min_delta_pp must be non-negative")
    if not 0 <= min_prob <= 1:
        raise ValueError("gate_min_prob_delta_gt_0 must be between 0 and 1")
    return {"passed": delta_pp >= min_delta_pp,
            "delta_pp": delta_pp, "min_delta_pp": min_delta_pp,
            "min_prob_delta_gt_0": min_prob,
            "prob_delta_gt_0": None,
            "probability_branch": "n_a: selection_gate exposes aggregate metrics only",
            "rule": "gate_delta_only",
            "reason": "selection_gate aggregate delta only"}


def task_input_secondary_gate(baselines: Dict[str, Dict[str, Any]],
                              candidates: Dict[str, Dict[str, Any]],
                              datasets: List[str]) -> Dict[str, Any]:
    """Fail closed on measurable task-answer regressions.

    Task-input experiments have benchmark answer labels, but no gold evidence-
    entailment labels. ``answer_evidence_consistency_rate`` comes from a fallible
    LLM judge and is diagnostic only until independently calibrated against
    explicitly validated AI or human entailment references. Requiring its value or imposing an
    absolute floor would let an unvalidated evaluator block the self-evolution
    loop. Keep it visible in diagnostics; gate on benchmark coverage and
    answered precision, whose labels are available.
    """
    checks: List[Dict[str, Any]] = []
    grounding_diagnostics: List[Dict[str, Any]] = []
    from ..eval.assessment_policy import classwise_non_regression, finite_rate
    metrics = ("coverage", "answered_precision")

    def record_grounding(ds: str, seed: Any, split: str,
                         before: Dict[str, Any], after: Dict[str, Any]) -> None:
        grounding_diagnostics.append({
            "dataset": ds, "seed": seed, "split": split,
            "baseline": before.get("answer_evidence_consistency_rate"),
            "candidate": after.get("answer_evidence_consistency_rate"),
            "status": "diagnostic_only_unvalidated_judge",
        })

    for ds in datasets:
        rows = candidates[ds].get("per_seed") or []
        if not rows:
            for split, after in (("optimize", candidates[ds]["optimize"]),
                                 ("gate", candidates[ds]["gate"]["metrics"])):
                before = baselines[ds][split]
                record_grounding(ds, None, split, before, after)
                before_classes = baselines[ds].get(split + "_class_metrics", {})
                after_classes = (candidates[ds].get("optimize_class_metrics", {})
                                 if split == "optimize" else
                                 candidates[ds]["gate"].get("class_metrics", {}))
                checks.append({"dataset": ds, "seed": None, "split": split,
                    "metric": "classwise_non_regression",
                    **classwise_non_regression(before_classes, after_classes)})
                for metric in metrics:
                    base_value, cand_value = before.get(metric), after.get(metric)
                    checks.append({"dataset": ds, "seed": None, "split": split,
                                   "metric": metric, "base": base_value,
                                   "candidate": cand_value,
                                   "passed": (finite_rate(base_value) and
                                              finite_rate(cand_value) and
                                              cand_value >= base_value - 1e-9)})
            continue
        for row in rows:
            seed = str(row["seed"])
            base_seed = baselines[ds].get("per_seed", {}).get(seed, baselines[ds])
            for split, cand_key in (("optimize", "cand_opt_metrics"),
                                    ("gate", "cand_gate_metrics")):
                base = (row.get("base_opt_metrics") or base_seed.get("optimize", {})
                        if split == "optimize" else base_seed.get("gate", {}))
                cand = row.get(cand_key, {})
                record_grounding(ds, row["seed"], split, base, cand)
                before_classes = base_seed.get(split + "_class_metrics", {})
                after_classes = row.get("cand_opt_class_metrics" if split == "optimize"
                                        else "cand_gate_class_metrics", {})
                checks.append({"dataset": ds, "seed": row["seed"], "split": split,
                    "metric": "classwise_non_regression",
                    **classwise_non_regression(before_classes, after_classes)})
                for metric in metrics:
                    before, after = base.get(metric), cand.get(metric)
                    passed = (finite_rate(before) and finite_rate(after)
                              and after >= before - 1e-9)
                    checks.append({"dataset": ds, "seed": row["seed"],
                                   "split": split, "metric": metric,
                                   "base": before, "candidate": after,
                                   "passed": passed})
    return {"passed": all(x["passed"] for x in checks), "checks": checks,
            "grounding_metric": "answer_evidence_consistency_rate",
            "grounding_interpretation": (
                "diagnostic-only fallible model-judge proxy over supplied task evidence; not a gold entailment label"),
            "grounding_gate_enabled": False,
            "grounding_diagnostics": grounding_diagnostics}


def _seed_requirement(cfg: Dict[str, Any], seeds: Sequence[int]) -> Dict[str, Any]:
    """Real task-input experiments need the Spec's three-seed repeatability check."""
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    real_task_input = (runtime.get("input_mode") == "task_input" and
                       (str(runtime.get("provider") or "").lower() == "pi" or
                        str(task_model.get("provider") or "").lower() not in
                        ("", "fake", "stub", "deterministic")))
    required = 3 if real_task_input else 1
    return {"required": required, "observed": len(set(seeds)),
            "passed": len(set(seeds)) >= required,
            "basis": "RSI SPEC v1.2 M2: repeat at least 3 seeds"
                     if required == 3 else "deterministic/synthetic workflow"}


def mean_gate_delta_pp(seed_rows: Sequence[Dict[str, Any]]) -> float:
    """Average paired aggregate-gate deltas across every configured seed."""
    if not seed_rows:
        return 0.0
    return sum((float(row["cand_gate"]) - float(row["base_gate"])) * 100.0
               for row in seed_rows) / len(seed_rows)


def evaluation_support_check(base_correct: Dict[str, int],
                             cand_correct: Dict[str, int],
                             optimize_n: int, baseline_gate_n: int,
                             candidate_gate_n: int) -> Dict[str, Any]:
    """Require a non-empty, identical optimize cohort and complete gate counts."""
    base_ids = {str(task_id) for task_id in base_correct}
    candidate_ids = {str(task_id) for task_id in cand_correct}
    paired_ids = base_ids & candidate_ids
    try:
        optimize_count = int(optimize_n)
        baseline_gate_count = int(baseline_gate_n)
        candidate_gate_count = int(candidate_gate_n)
    except (TypeError, ValueError):
        optimize_count = baseline_gate_count = candidate_gate_count = 0
    checks = {
        "optimize_nonempty": bool(base_ids and candidate_ids and paired_ids),
        "optimize_cohort_identical": base_ids == candidate_ids,
        "candidate_optimize_count_matches_ids": optimize_count == len(candidate_ids),
        "baseline_gate_nonempty": baseline_gate_count > 0,
        "candidate_gate_nonempty": candidate_gate_count > 0,
        "gate_count_matches_baseline": candidate_gate_count == baseline_gate_count,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "n_baseline_optimize": len(base_ids),
        "n_candidate_optimize": len(candidate_ids),
        "n_paired_optimize": len(paired_ids),
        "n_baseline_gate": baseline_gate_count,
        "n_candidate_gate": candidate_gate_count,
    }


def actionable_failure_classes(by_class: Dict[str, List[Dict[str, Any]]],
                               max_candidates: int) -> List[tuple[str, List[Dict[str, Any]]]]:
    """Reserve candidate budget for failures that map to evolvable artifacts."""
    return sorted(((cls, rows) for cls, rows in by_class.items()
                   if cls in ATTRIBUTION_TARGET),
                  key=lambda kv: (-len(kv[1]), kv[0]))[:max_candidates]


def run_round(cfg: Dict[str, Any], datasets: Optional[List[str]] = None,
              seed: Optional[int] = None, limit: Optional[int] = None,
              max_candidates: Optional[int] = None,
              seeds: Optional[List[int]] = None,
              gepa_replay_report: Optional[str] = None,
              gepa_replay_index: int = 1,
              fewshot_search: bool = False,
              candidate_pre_evaluation_gate=None,
              attribution_corrections: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_live_execution_plan(cfg)
    cfg_fp = config_fingerprint(cfg)
    # 【B / P5】协议指纹 fail-loud：进化轮次协议必须与封存 final_holdout 基线一致，
    # 否则候选度量与终评配对不在同源协议下，Δ 不可解释且静默过期。
    _assert_protocol_consistent(cfg)
    seed = int(seed if seed is not None else cfg.get("evaluation", {}).get("seed", 0))
    seeds_list = [int(s) for s in seeds] if seeds else [seed]
    seed_requirement = _seed_requirement(cfg, seeds_list)
    datasets = datasets or [str(cfg.get("evaluation", {}).get("primary_dataset", "litqa2"))]
    max_cands = int(max_candidates or cfg.get("evolution", {}).get("max_candidates_per_round", 5))

    state = rel.load_state()
    memory_context = candidate_memory.context(cfg, state, datasets, seeds_list, limit)
    round_id = new_id("round")
    allow = list(cfg.get("evolution", {}).get("allowed_artifacts", []))
    main_metric = str(cfg.get("evaluation", {}).get("main_metric", "accuracy"))
    min_delta = float(cfg.get("gates", {}).get("accept", {}).get("gate_min_delta_pp", 0.0))
    min_prob = float(cfg.get("gates", {}).get("accept", {}).get("gate_min_prob_delta_gt_0", 0.90))
    B = int(cfg.get("evaluation", {}).get("bootstrap", {}).get("B", 10000))
    rng_seed = int(cfg.get("evaluation", {}).get("bootstrap", {}).get("rng_seed", 20260924))

    # 基线：每个 seed 都跑一遍（optimize 留逐题结果用于配对，gate 只留聚合）
    baselines: Dict[str, Dict[str, Any]] = {}
    all_failures: List[Dict[str, Any]] = []
    reviewed_baselines = (_reviewed_baselines(cfg, state, datasets, seeds_list, limit,
        attribution_corrections) if attribution_corrections is not None else {})
    for ds in datasets:
        per_seed: Dict[str, Any] = {}
        for s in seeds_list:
            opt = (reviewed_baselines.get((ds, s)) or
                   _baseline_metrics(cfg, state, ds, "dev_optimize", s, limit, cfg_fp))
            gate = _baseline_metrics(cfg, state, ds, "selection_gate", s, limit, cfg_fp)
            per_seed[str(s)] = {
                "optimize": opt["metrics"],
                "gate": gate["metrics"],
                "optimize_correct": {r["task_id"]: int(bool(r["correct"]))
                                     for r in opt["result"].per_task},
                "optimize_run_id": opt["run_id"],
                "gate_run_id": gate["run_id"],
                "optimize_class_metrics": opt.get("class_metrics", {}),
                "gate_class_metrics": gate.get("class_metrics", {}),
            }
            failures = [dict(row) for row in _load_failures(opt["run_id"])]
            for failure in failures:
                failure.setdefault("source_run_id", opt["run_id"])
                failure["execution_seed"] = s
            all_failures.extend(failures)
        first = per_seed[str(seeds_list[0])]
        baselines[ds] = {
            "optimize": first["optimize"],
            "gate": first["gate"],
            "optimize_run_id": first["optimize_run_id"],
            "gate_run_id": first["gate_run_id"],
            "per_seed": per_seed,
            "optimize_class_metrics": first["optimize_class_metrics"],
            "gate_class_metrics": first["gate_class_metrics"],
            "seeds": seeds_list,
        }

    correction_audit = None
    if attribution_corrections is not None:
        from .attribution_review import apply_attribution_corrections
        all_failures = apply_attribution_corrections(all_failures, attribution_corrections)
        correction_audit = {
            "review_packet_id": attribution_corrections.get("review_packet_id"),
            "review_document_sha256": sha256_obj(attribution_corrections),
            "source_failures_sha256": attribution_corrections.get("source_failures_sha256"),
            "source_run_ids": attribution_corrections.get("source_run_ids"),
            "applied_count": sum(bool(row.get("human_correction", {}).get("applied"))
                                  for row in all_failures),
            "is_causal_evidence": False,
        }
    failure_summary: Dict[str, int] = {}
    for failure in all_failures:
        label = failure.get("effective_attributed_class", failure.get("attributed_class"))
        failure_summary[label] = failure_summary.get(label, 0) + 1
    failure_summary = dict(sorted(failure_summary.items(),
                                  key=lambda item: (-item[1], item[0])))
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for f in all_failures:
        label = f.get("effective_attributed_class", f.get("attributed_class"))
        by_class.setdefault(label, []).append(f)

    candidates: List[Dict[str, Any]] = []
    candidate_objects = {}
    gepa_cfg = (cfg.get("evolution", {}).get("gepa") or {})
    gepa_enabled = bool(gepa_cfg.get("enabled"))
    fewshot_enabled = bool(fewshot_search or
                           (cfg.get("evolution", {}).get("fewshot") or {}).get("enabled"))
    joint_program_enabled = gepa_enabled and fewshot_enabled
    gepa_target = str(gepa_cfg.get("target_artifact") or "system_prompt")
    if gepa_target not in ("system_prompt", "skill_description", "skill_body"):
        raise ValueError("unsupported GEPA target_artifact")
    if gepa_target in ("skill_description", "skill_body") and not gepa_enabled:
        raise ValueError("GEPA Skill-artifact experiment requires GEPA enabled")
    if gepa_replay_report and not gepa_enabled:
        raise ValueError("GEPA replay requires evolution.gepa.enabled")
    if gepa_replay_report and gepa_target != "system_prompt":
        raise ValueError("GEPA replay currently supports Prompt search reports only")
    if fewshot_enabled:
        if (cfg.get("runtime") or {}).get("input_mode") != "task_input" \
                or str((cfg.get("runtime") or {}).get("provider") or "").lower() != "pi":
            raise ValueError("task-demo search requires Pi with task_input")
        required_artifact = "task_program" if joint_program_enabled else "task_demos"
        if required_artifact not in allow:
            raise ValueError(f"task-demo search requires {required_artifact} in allowed_artifacts")
        if joint_program_enabled and "system_prompt" not in allow:
            raise ValueError("joint search also requires system_prompt for GEPA proposal validation")
        if joint_program_enabled and gepa_target != "system_prompt":
            raise ValueError("joint instruction/demo search requires GEPA target system_prompt")
        if joint_program_enabled and gepa_replay_report:
            raise ValueError("joint instruction/demo search does not accept GEPA replay reports")
    # GEPA and task-demo search both own candidate generation in their rounds.
    # The rewrite backend's bootstrap path performs an extra full visible-split
    # evaluation to assemble reflection examples; neither search consumes those
    # examples, and that pass does not share the round's explicit task limit.
    backend = None if (gepa_enabled or fewshot_enabled) else _build_reflection_backend(
        cfg, state=state)
    # 检索/工具/评分诊断要留在报告里，但不能占用每轮有限的可进化候选名额。
    # 否则一次大面积检索失败可能使 Prompt/Skill 根本没有机会被试验。
    diagnostic_failures = {cls: len(rows) for cls, rows in by_class.items()
                           if cls not in ATTRIBUTION_TARGET}
    work_items = select_candidate_work_items(
        by_class, all_failures, gepa_enabled=gepa_enabled, gepa_target=gepa_target,
        fewshot_enabled=fewshot_enabled, joint_program_enabled=joint_program_enabled,
        max_candidates=max_cands,
        input_mode=str((cfg.get("runtime") or {}).get("input_mode", "retrieval")),
        new_skill_enabled=bool((cfg.get("evolution", {}).get("crystallization")
                               or {}).get("nominate_unresolved", False)))
    for cls, rows in work_items:
        target = EXPLICIT_ARTIFACT_TARGETS.get(cls, ATTRIBUTION_TARGET.get(cls))
        selection_mode = ("explicit_artifact_experiment"
                          if cls in EXPLICIT_ARTIFACT_TARGETS else "attribution")
        if target is None or target not in allow:
            candidates.append({
                "candidate_id": None,
                "attributed_class": cls,
                "selection_mode": selection_mode,
                "target_artifact": target,
                "status": "deferred",
                "reason": f"目标工件 {target} 不在 allowed_artifacts={allow}",
                "n_failures": len(rows),
            })
            continue

        # 失败样本原文要交给候选生成器：description 进化需要题目文本抽关键词
        if gepa_enabled and cls in ("experiment.skill_description",
                                    "experiment.skill_body"):
            from .gepa_skill_search import (search_skill_body_candidate,
                                            search_skill_description_candidate)
            searcher = (search_skill_description_candidate
                        if cls == "experiment.skill_description"
                        else search_skill_body_candidate)
            cand = searcher(cfg, state, datasets, limit, seed,
                            [r["failure_id"] for r in rows],
                            str(gepa_cfg.get("target_skill") or ""))
        elif joint_program_enabled and cls == "experiment.task_program":
            from .gepa_search import search_prompt_candidate
            from .task_demo_search import search_task_demo_candidate

            instruction_candidate = search_prompt_candidate(
                cfg, state, datasets, limit, seed,
                [r["failure_id"] for r in rows])
            cand = search_task_demo_candidate(
                cfg, state, datasets, limit, seed,
                [r["failure_id"] for r in rows],
                {ds: {str(seed_key): str(row["optimize_run_id"])
                      for seed_key, row in baselines[ds]["per_seed"].items()}
                 for ds in datasets},
                instruction_candidate=instruction_candidate)
        elif fewshot_enabled and cls == "experiment.task_demos":
            from .task_demo_search import search_task_demo_candidate

            cand = search_task_demo_candidate(
                cfg, state, datasets, limit, seed,
                [r["failure_id"] for r in rows],
                {ds: {str(seed_key): str(row["optimize_run_id"])
                      for seed_key, row in baselines[ds]["per_seed"].items()}
                 for ds in datasets})
        elif gepa_enabled and cls in ("plan.failed", "experiment.system_prompt"):
            from .gepa_search import replay_prompt_candidate, search_prompt_candidate

            if gepa_replay_report:
                cand = replay_prompt_candidate(
                    cfg, state, datasets, limit, [r["failure_id"] for r in rows],
                    gepa_replay_report, gepa_replay_index)
            else:
                cand = search_prompt_candidate(
                    cfg, state, datasets, limit, seed,
                    [r["failure_id"] for r in rows])
        else:
            backend_cls = ("plan.failed" if cls == "experiment.system_prompt"
                           else "skill.missing" if cls == "experiment.new_skill" else cls)
            cand = generate_candidate(backend_cls, state, [r["failure_id"] for r in rows],
                                      cfg, failure_rows=rows, backend=backend)
        if cand is None:
            reason = ("Search returned no promotable candidate; see its search report"
                      if ((gepa_enabled and cls in ("plan.failed", "experiment.system_prompt",
                                                     "experiment.skill_body",
                                                     "experiment.skill_description"))
                      or (fewshot_enabled and cls in
                          ("experiment.task_demos", "experiment.task_program"))) else
                      _skip_reason(target, state, rows, cfg))
            skip: Dict[str, Any] = {
                "candidate_id": None,
                "attributed_class": cls,
                "selection_mode": selection_mode,
                "target_artifact": target,
                "status": "skipped",
                "reason": reason,
                "n_failures": len(rows),
            }
            # 真实模型反思被拒时，模板库的 skip_reason 会说"无可用补丁"，
            # 那是错的：模板没了不等于模型提不出。把模型的拒绝原因一并记下。
            # 只对 system_prompt 附 —— backend 是整轮共享的，别把上一类的
            # last_error 挂到无关类别上。
            err = getattr(backend, "last_error", None) if target == "system_prompt" else None
            if err:
                skip["reflection_error"] = err
                skip["reflection_attempts"] = list(getattr(backend, "attempts", []) or [])[:3]
            candidates.append(skip)
            continue

        cand.attributed_class = cls
        cand.source_failure_ids = [r["failure_id"] for r in rows]
        cand = apply_gate(cand, cfg, state)
        pre_evaluation_gate = None
        if (cand.status != "static_gate_rejected"
                and candidate_pre_evaluation_gate is not None):
            pre_evaluation_gate = candidate_pre_evaluation_gate(cand, cfg, state)
            cand.generator["pre_evaluation_gate"] = pre_evaluation_gate
            if not (isinstance(pre_evaluation_gate, dict)
                    and pre_evaluation_gate.get("passed") is True):
                cand.status = "pre_evaluation_gate_rejected"
        # 被门拒的候选同样要落盘：否则 review 时看不到"究竟提了什么被拒了"
        _save_candidate(cand)
        candidate_objects[cand.candidate_id] = cand
        rec: Dict[str, Any] = {
            "candidate_id": cand.candidate_id,
            # 【M1/M2 回顾】跨轮候选池要靠这个字段判断"该候选是相对哪个 release 度量的"。
            # 没有它，campaign 无法阻止"把度量于旧 release 的候选发布到当前 release 之后"
            # 那种静默回退（详见 campaign.py 的说明）。
            "parent_release_id": cand.parent_release_id,
            "attributed_class": cls,
            "selection_mode": selection_mode,
            "target_artifact": target,
            "n_failures": len(rows),
            "static_gate_passed": all(r["passed"] for r in cand.static_gate_results),
            "status": cand.status,
            "artifact_id": cand.artifact_id,
            "diff": cand.diff,
            "content_sha256": cand.content_sha256,
            "proposal_sha256": candidate_memory.proposal_sha256(cand),
        }
        remembered = candidate_memory.find(cand, memory_context)
        if remembered is not None:
            rec.update({"status": "previously_rejected", "reason": remembered["reason"],
                        "rejection_memory_sha256": remembered["memory_sha256"]})
            candidates.append(rec)
            continue
        if pre_evaluation_gate is not None:
            rec["pre_evaluation_gate"] = pre_evaluation_gate
        if cand.status == "static_gate_rejected":
            rec["reason"] = rejection_reason(cand.static_gate_results)
            candidates.append(rec)
            continue
        if cand.status == "pre_evaluation_gate_rejected":
            rec["reason"] = "pre_evaluation_gate_rejected"
            candidates.append(rec)
            continue

        cstate = _candidate_state(state, cand)
        cand_metrics: Dict[str, Dict[str, Any]] = {}
        ci_report: Dict[str, Any] = {}
        per_seed_rows_all: Dict[str, List[Dict[str, Any]]] = {}
        nested_demo_report = None
        if fewshot_enabled and cand.artifact_type in ("task_demos", "task_program"):
            report_path = str(cand.generator.get("search_report") or "")
            if not report_path:
                raise RuntimeError("task-demo candidate is missing its nested search report")
            nested_demo_report = read_json(report_path)
            if nested_demo_report.get("dataset") not in datasets:
                raise RuntimeError("task-demo search report dataset differs from this round")

        for ds in datasets:
            rows_by_seed: List[Dict[str, Any]] = []
            first_opt = None
            first_gate = None
            for s in seeds_list:
                if nested_demo_report is not None:
                    baseline_seed = baselines[ds]["per_seed"][str(s)]
                    baseline_run_id = str(baseline_seed["optimize_run_id"])
                    baseline_task_rows = read_jsonl(artifacts_path(
                        "runs", baseline_run_id, "per_task.jsonl"))
                    o, base_opt_metrics, base_correct, cand_correct = \
                        _nested_demo_confirmation_pair(
                            nested_demo_report, cand, s, baseline_run_id,
                            baseline_task_rows)
                else:
                    o = evaluate(cfg, cstate, ds, "dev_optimize", seed=s, limit=limit,
                                 run_tag=cand.candidate_id, cfg_fp=cfg_fp)
                    base_opt_metrics = baselines[ds]["per_seed"][str(s)]["optimize"]
                    base_correct = baselines[ds]["per_seed"][str(s)]["optimize_correct"]
                    cand_correct = {r["task_id"]: int(bool(r["correct"]))
                                    for r in o.per_task}
                g = evaluate(cfg, cstate, ds, "selection_gate", seed=s, limit=limit,
                             run_tag=cand.candidate_id, write_per_task=False, cfg_fp=cfg_fp)
                baseline_gate_metrics = baselines[ds]["per_seed"][str(s)]["gate"]
                support_check = evaluation_support_check(
                    base_correct, cand_correct,
                    optimize_n=getattr(o, "n_tasks", 0),
                    baseline_gate_n=baseline_gate_metrics.get("n_tasks", 0),
                    candidate_gate_n=getattr(g, "n_tasks", 0),
                )
                rows_by_seed.append({
                    "seed": s,
                    "base_opt": base_opt_metrics[main_metric],
                    "cand_opt": o.metric_values[main_metric],
                    "base_opt_metrics": base_opt_metrics,
                    "base_gate": baselines[ds]["per_seed"][str(s)]["gate"][main_metric],
                    "cand_gate": g.metric_values[main_metric],
                    "cand_opt_metrics": o.metric_values,
                    "cand_gate_metrics": g.metric_values,
                    "cand_opt_class_metrics": getattr(o, "class_metrics", {}),
                    "cand_gate_class_metrics": getattr(g, "class_metrics", {}),
                    "base_correct": base_correct,
                    "cand_correct": cand_correct,
                    "evaluation_support": support_check,
                })
                if s == seeds_list[0]:
                    first_opt, first_gate = o, g
            # 【M1/M2 回顾】铁律 2 之前只在 harness 里靠 write_per_task=False 间接保证，
            # assert_gate_is_aggregate 全仓库零调用 —— 等于没有守卫。这里把它接成活的。
            gate_agg = assert_gate_is_aggregate(first_gate)
            if not gate_agg["passed"]:
                raise RuntimeError("selection_gate 回传了逐题结果（Spec §6.1 铁律 2）："
                                   + gate_agg["detail"])
            cand_metrics[ds] = {"optimize": first_opt.metric_values,
                                "optimize_class_metrics": getattr(first_opt, "class_metrics", {}),
                                "gate_class_metrics": getattr(first_gate, "class_metrics", {}),
                                "optimize_pairing": ({
                                    "scope": "task_demo_internal_confirmation",
                                    "source_report": str(cand.generator["search_report"]),
                                    "task_ids": sorted(nested_demo_report[
                                        "confirmation_ids"]),
                                    "task_count_per_seed": len(nested_demo_report[
                                        "confirmation_ids"]),
                                    "baseline_metrics_by_seed": [
                                        {"seed": row["seed"],
                                         "metrics": row["base_opt_metrics"]}
                                        for row in rows_by_seed],
                                } if nested_demo_report is not None else {
                                    "scope": "full_dev_optimize_split",
                                    "task_count_per_seed": first_opt.n_tasks,
                                }),
                                "paired_optimize_baseline": rows_by_seed[0][
                                    "base_opt_metrics"],
                                "gate": gate_summary(first_gate),
                                "gate_aggregate_check": gate_agg,
                                "per_seed": [{k: r[k] for k in ("seed", "base_opt", "cand_opt",
                                                                "base_opt_metrics",
                                                                "base_gate", "cand_gate",
                                                                "cand_opt_metrics",
                                                                "cand_gate_metrics",
                                                                "cand_opt_class_metrics",
                                                                "cand_gate_class_metrics")}
                                             for r in rows_by_seed]}
            per_seed_rows_all[ds] = rows_by_seed

            # 配对 bootstrap CI（只可能在 dev_optimize 上做：gate 不回传逐题结果）
            cis = []
            for r in rows_by_seed:
                ids = sorted(set(r["base_correct"]) & set(r["cand_correct"]))
                cis.append({
                    "seed": r["seed"],
                    **paired_bootstrap_binary([r["base_correct"][i] for i in ids],
                                              [r["cand_correct"][i] for i in ids],
                                              B=B, rng_seed=rng_seed),
                })
            ci_report[ds] = {
                "optimize_paired_bootstrap": cis,
                "optimize_mean_delta_pp": round(sum(c["delta_pp"] for c in cis) / len(cis), 4),
                "optimize_min_p_delta_gt_0": round(min(c["p_delta_gt_0"] for c in cis), 4),
                "gate_delta_pp_per_seed": [
                    round((r["cand_gate"] - r["base_gate"]) * 100.0, 4) for r in rows_by_seed
                ],
            }

        # M2 增强接受规则（§6.3）：
        #   1) 每个 seed 上 dev_optimize 主指标都不降（非回归）
        #   2) gate 只有聚合指标，必须由 gate 自身的 Δ 达阈值。
        #      optimize 的配对概率不能充当 gate 上的证据。
        evaluation_support_checks = [
            {"dataset": ds, "seed": row["seed"], **row["evaluation_support"]}
            for ds in datasets for row in per_seed_rows_all[ds]
        ]
        evaluation_support_passed = bool(evaluation_support_checks) and all(
            row["passed"] for row in evaluation_support_checks)
        non_regress = evaluation_support_passed and all(
            r["cand_opt"] >= r["base_opt"] - 1e-9
            for ds in datasets for r in per_seed_rows_all[ds]
        )
        # gate 判定用**按题数加权的总体 Δ**：多数据集一起评时，某个已饱和的数据集
        # （如 protocolqa 召回率 96%，description 对它无增益）Δ=0 不应否决整体提升；
        # 但非回归仍要求**每个**数据集都不降，护栏没有放松。
        w_sum = 0.0
        d_sum = 0.0
        for ds in datasets:
            n = float(baselines[ds]["gate"].get("n_tasks", 1) or 1)
            ds_seed_rows = per_seed_rows_all[ds]
            dsum_gate = mean_gate_delta_pp(ds_seed_rows)
            d_sum += dsum_gate * n
            w_sum += n
        overall_gate_delta = round(d_sum / w_sum if w_sum else 0.0, 4)
        threshold = gate_delta_threshold(overall_gate_delta, min_delta, min_prob)
        gate_ok = bool(threshold["passed"])
        verdict = ("proposed_accept" if (non_regress and gate_ok
                                         and seed_requirement["passed"])
                   else "proposed_reject")

        # 【R1b】次指标容差门禁 —— 在此之前它**从未被任何代码读取过**
        # （配置里写了 cost_pct / p95_latency_pct / citation_precision_pp，但没有调用点）。
        #
        # 接上的同时必须先判"这个量到底测不测得了"：离线 fake 的 latency 是写死常数、
        # token 是 len/4 估算，直接比等于恒真通过。对齐 gepa
        # （`api.py:286` "the cost stopper would silently never fire"）：
        # 测不了就记 n_a 并点名，绝不允许在报告里写出"成本门禁：通过"。
        tol_report: Dict[str, Any] = {}
        tolerance_cfg = cfg
        runtime_provider = str((cfg.get("runtime") or {}).get("provider") or "").lower()
        if runtime_provider in ("http", "pi"):
            # Spec §6.3 treats the spend budget as a hard constraint. A real
            # model run with unknown USD pricing cannot be proposed for release.
            tolerance_cfg = copy.deepcopy(cfg)
            gates_cfg = tolerance_cfg.get("gates") or {}
            accept_cfg = gates_cfg.get("accept") or {}
            configured_tolerance = accept_cfg.get("tolerance") or {}
            if "cost_pct" in configured_tolerance:
                required = set(configured_tolerance.get("required", []) or [])
                required.add("cost")
                configured_tolerance["required"] = sorted(required)
        for ds in datasets:
            # 注意：`cand_metrics[ds]["optimize"]` 本身就是完整 metric 字典
            # （来自 `first_opt.metric_values`，见上文 line 179），
            # 不要再 `.get("metrics", {})` 二次取键 —— 否则候选字典变成空 {}，
            # 门禁永远只能看到 base 而 cand=None，所有次指标容差都静默退化为 n_a。
            cand_opt_metrics = cand_metrics.get(ds, {}).get("optimize", {}) or {}
            tol_report[ds] = check_tolerances(
                cand_metrics.get(ds, {}).get("paired_optimize_baseline") or
                baselines[ds]["optimize"],
                cand_opt_metrics,
                tolerance_cfg,
                base_effective=globally_active_behaviors(state, cfg),
                cand_effective=globally_active_behaviors(cstate, cfg),
            )
        tol_passed = all(v.get("passed", True) for v in tol_report.values())
        tol_skipped = sorted({q for v in tol_report.values()
                              for q in (v.get("not_measurable") or [])})
        if not tol_passed:
            verdict = "proposed_reject"
        secondary = (task_input_secondary_gate(baselines, cand_metrics, datasets)
                     if (cfg.get("runtime") or {}).get("input_mode") == "task_input"
                     else None)
        if secondary is not None and not secondary["passed"]:
            verdict = "proposed_reject"
        # 【M4+】逐题支持度：修好了几题、弄坏了几题。
        #
        # 为什么必须有：`delta_pp` 是**跨数据集平均**，而平均恰恰是最会掩盖回退的
        # 聚合方式 —— CS 转移集上 6 个回归里有 5 个，在 dev_optimize 的平均增益是正的，
        # 所以被放行了。逐题计数让"子集回退"在数字上看得见。
        # 对齐 GEPA 的 instance-wise Pareto front（`core/state.py:246`
        # `program_at_pareto_front_valset`）：front 按**每个验证样本**组织，
        # 而不是一个全局平均标量。
        #
        # 口径：用第一个 seed 的 dev_optimize 配对结果（与 delta_pp 同一口径、可复现）。
        imp = reg = paired = 0
        for ds in datasets:
            r0 = (per_seed_rows_all.get(ds) or [{}])[0]
            bc, cc = r0.get("base_correct") or {}, r0.get("cand_correct") or {}
            ids = sorted(set(bc) & set(cc))
            paired += len(ids)
            for i in ids:
                if cc[i] and not bc[i]:
                    imp += 1
                elif bc[i] and not cc[i]:
                    reg += 1
        rec["task_support"] = {
            "improved": imp, "regressed": reg, "net": imp - reg,
            "n_paired": paired, "seed": seeds_list[0] if seeds_list else seed,
            "note": "dev_optimize 逐题配对（第一个 seed）；平均增量看不见的子集回退在这里",
        }
        rec.update({
            "status": verdict,
            "evaluation_support": {
                "passed": evaluation_support_passed,
                "per_dataset_seed": evaluation_support_checks,
            },
            "candidate_metrics": cand_metrics,
            "baseline_metrics": {ds: {"optimize": baselines[ds]["optimize"],
                                      "gate": baselines[ds]["gate"]} for ds in datasets},
            "paired_optimize_baseline_metrics": {
                ds: cand_metrics[ds].get("paired_optimize_baseline")
                for ds in datasets},
            "delta_pp": {
                ds: {
                    "optimize": round(sum((r["cand_opt"] - r["base_opt"]) * 100.0
                                          for r in per_seed_rows_all[ds]) /
                                      max(1, len(per_seed_rows_all[ds])), 4),
                    "gate": round(sum((r["cand_gate"] - r["base_gate"]) * 100.0
                                      for r in per_seed_rows_all[ds]) /
                                  max(1, len(per_seed_rows_all[ds])), 4),
                } for ds in datasets
            },
            "ci": ci_report,
            "overall_gate_delta_pp": overall_gate_delta,
            "gate_threshold": threshold,
            # R1b：次指标容差 —— 逐项状态 + 哪些量根本测不了（报告里必须点名）
            "tolerance": {
                "per_dataset": tol_report,
                "passed": tol_passed,
                "not_measurable": tol_skipped,
                "note": ("测不了的量记为 n_a 且不参与判定；报告里不得把 n_a 写成'通过'"
                         if tol_skipped else "本次所有容差项均可测"),
            },
            "task_input_secondary_gate": secondary,
            "seed_requirement": seed_requirement,
            "seeds": seeds_list,
            "verdict_reason": _verdict_reason(
                evaluation_support_passed=evaluation_support_passed,
                non_regress=non_regress, gate_ok=gate_ok,
                seed_passed=seed_requirement["passed"],
                tolerance_passed=tol_passed,
                secondary_passed=(secondary is None or secondary["passed"]))
                              + f"；{threshold['rule']}"
                              + ("；真实 task_input 结果至少需要 3 个独立 seed"
                                 if not seed_requirement["passed"] else "")
                              + (f"（实测 {threshold['reason']}）" if threshold["reason"] else "")
                              + ("；task_input coverage/answered_precision 回退"
                                 if secondary is not None and not secondary["passed"] else ""),
        })

        req = rv.build_request(
            cand,
            {ds: cand_metrics[ds]["optimize"] for ds in datasets},
            {ds: cand_metrics[ds]["gate"] for ds in datasets},
            rollback_point=state.release_id,
            static_gate={"passed": True, "results": cand.static_gate_results},
        )
        req["verdict"] = verdict
        req["round_id"] = round_id
        # The exact-repeat control runs after every candidate. Writing a request
        # here would leave its verdict at proposed_accept if the control later invalidates
        # the measurement. Persist requests only after that final decision.
        rec["_review_request_payload"] = req
        rec["diff"] = cand.diff
        rec["content_sha256"] = cand.content_sha256
        candidates.append(rec)

    # Repeat the exact baseline prompt after candidates. A paraphrase is not a
    # valid placebo for an LLM: small wording changes can alter its behavior.
    measurement_control = _evaluate_repeat_control(
        cfg, state, datasets, baselines, seeds_list, limit, cfg_fp, main_metric)
    control_max_abs = int((cfg.get("evolution", {}) or {}).get(
        "repeat_control_max_abs_net",
        (cfg.get("evolution", {}) or {}).get("placebo_max_abs_net", 0)))
    repeat_drift = bool(measurement_control and
                        abs(int(measurement_control.get("net", 0))) > control_max_abs)
    # A single repeat is useful as a diagnostic, but real task-input decisions
    # rely on the Spec-required multi-seed evaluations rather than exact equality
    # between two remote-model calls.
    contaminated = bool(repeat_drift and not seed_requirement["passed"])
    measurement_control.update({"drift_detected": repeat_drift,
                                "contaminated": contaminated,
                                "multi_seed_decision_available":
                                    seed_requirement["passed"]})
    if contaminated:
        # fail-closed：测量本身不可信时，宁可什么都不发布
        for rec in candidates:
            if rec.get("status") == "proposed_accept":
                rec["status"] = "proposed_reject"
                rec["verdict_reason"] = (
                    f"同提示重测出现非零净变化（net={measurement_control['net']}），"
                    "本轮运行存在重复性偏差，候选一律降级为拟拒绝"
                )

    for rec in candidates:
        req = rec.pop("_review_request_payload", None)
        if req is None:
            continue
        req["verdict"] = rec["status"]
        req["evaluation_summary"] = {
            "baseline_metrics": rec.get("baseline_metrics"),
            "candidate_metrics": rec.get("candidate_metrics"),
            "paired_optimize_baseline_metrics": rec.get(
                "paired_optimize_baseline_metrics"),
            "delta_pp": rec.get("delta_pp"),
            "task_support": rec.get("task_support"),
            "ci": rec.get("ci"),
            "gate_threshold": rec.get("gate_threshold"),
            "tolerance": rec.get("tolerance"),
            "task_input_secondary_gate": rec.get("task_input_secondary_gate"),
            "seed_requirement": rec.get("seed_requirement"),
            "evaluation_support": rec.get("evaluation_support"),
            "measurement_control": measurement_control,
            "sealed_pairing": bool(cfg.get("evaluation", {}).get("sealed_pairing", True)),
            "gepa_replay": bool(gepa_replay_report),
            "evidence_scope": ("visible_dev_exploratory_replay" if gepa_replay_report
                               else "visible_dev_selection"),
        }
        rec["review_request"] = rv.write_request(req).as_posix()

    report = {
        "round_id": round_id,
        "created_at": now_iso(),
        "release_id": state.release_id,
        "config_fingerprint": cfg_fp,
        "datasets": datasets,
        "seed": seed,
        "seeds": seeds_list,
        "seed_requirement": seed_requirement,
        "main_metric": main_metric,
        "allowed_artifacts": allow,
        "gepa_target_artifact": gepa_target if gepa_enabled else None,
        "gepa_target_skill": (gepa_cfg.get("target_skill")
                              if gepa_target in ("skill_description", "skill_body")
                              else None),
        "gepa_replay": ({"search_report": gepa_replay_report,
                         "candidate_index": gepa_replay_index,
                         "exploratory": True} if gepa_replay_report else None),
        # 【P5】本轮产物是否允许与封存 final_holdout 基线配对。
        # false = 显式声明的离线机制跑（合成夹具），其 Δ 不得写成"终评结论"。
        "sealed_pairing": bool(cfg.get("evaluation", {}).get("sealed_pairing", True)),
        "baseline": {ds: {"optimize": baselines[ds]["optimize"], "gate": baselines[ds]["gate"]}
                     for ds in datasets},
        "n_failures": len(all_failures),
        "failure_class_counts": failure_summary,
        "attribution_corrections": correction_audit,
        "diagnostic_failure_counts": diagnostic_failures,
        "candidates": candidates,
        "rejection_memory_context": memory_context,
        # Exact-prompt repeat control is diagnostic when the Spec-required
        # multi-seed paired evaluation is available. It invalidates a round
        # only when that multi-seed decision is unavailable (fail closed).
        "measurement_control": {**measurement_control, "max_abs_net": control_max_abs,
                    "contaminated": contaminated},
    }
    out = path(ROUNDS_ROOT) / f"{round_id}.json"
    write_json(out, report)
    if not contaminated:
        for rec in candidates:
            # Caller-supplied gates may depend on changing human/reference input.
            # Their rejection is not a transferable, protocol-bound measurement.
            if rec.get("status") in {"proposed_reject", "static_gate_rejected"}:
                saved = candidate_objects[rec["candidate_id"]]
                candidate_memory.remember(saved, memory_context,
                    rec.get("verdict_reason") or rec.get("reason") or rec["status"], out)
    report["report_path"] = out.as_posix()
    return report


def _require_live_execution_plan(cfg: Dict[str, Any]) -> None:
    """Keep direct Python round/campaign callers behind the guarded CLI plan."""
    provider = str((cfg.get("runtime") or {}).get("provider") or "fake").lower()
    task_provider = str(((cfg.get("models") or {}).get("task") or {})
                        .get("provider") or provider).lower()
    if provider == "fake" or (provider == "pi" and task_provider == "fake"):
        return
    plan_sha = os.environ.get("RSI_ACTIVE_EXECUTION_PLAN_SHA256", "")
    ledger = os.environ.get("RSI_CALL_LEDGER_PATH", "")
    guard = os.environ.get("RSI_EXECUTION_GUARD_PATH", "")
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha):
        raise PermissionError("live round/campaign requires a verified execution plan")
    if not ledger or not guard:
        raise PermissionError("live round/campaign requires an execution ledger and one-shot guard")
    if any(int(os.environ.get(name, "0") or 0) < 1 for name in (
            "RSI_CALL_LEDGER_MAX_LOGICAL", "RSI_CALL_LEDGER_MAX_PHYSICAL")):
        raise PermissionError("live round/campaign requires positive request ceilings")
    try:
        import json
        with open(guard, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PermissionError("live round/campaign one-shot guard is unavailable") from exc
    if payload.get("plan_sha256") != plan_sha or payload.get("status") != "started":
        raise PermissionError("live round/campaign execution guard does not match the plan")


def _evaluate_repeat_control(cfg: Dict[str, Any], state: ArtifactState,
                             datasets: Sequence[str], baselines: Dict[str, Any],
                             seeds_list: Sequence[int], limit: Optional[int],
                             cfg_fp: str, main_metric: str) -> Dict[str, Any]:
    """Repeat the exact baseline state to measure run-to-run answer drift.

    This replaces the previous synonym-rewrite "placebo": paraphrases can
    change LLM behavior even when keyword-based behavior checks are unchanged.
    A distinct evaluation seed gives the repeat a new run id while keeping the
    release, prompt, Skills, examples, task cohort, and task protocol identical.
    """
    s = seeds_list[0] if seeds_list else 0
    control_seed = max(seeds_list or [s]) + 1
    imp = reg = paired = 0
    per_ds: Dict[str, Any] = {}
    for ds in datasets:
        o = evaluate(cfg, state, ds, "dev_optimize", seed=control_seed, limit=limit,
                     run_tag="exact-baseline-repeat", cfg_fp=cfg_fp)
        cc = {r["task_id"]: int(bool(r["correct"])) for r in o.per_task}
        bc = ((baselines.get(ds) or {}).get("per_seed", {}).get(str(s), {})
              .get("optimize_correct") or {})
        ids = sorted(set(bc) & set(cc))
        i = sum(1 for k in ids if cc[k] and not bc[k])
        r = sum(1 for k in ids if bc[k] and not cc[k])
        imp += i
        reg += r
        paired += len(ids)
        per_ds[ds] = {"improved": i, "regressed": r, "net": i - r,
                      "n_paired": len(ids), "replicate_seed": control_seed,
                      "run_id": o.run_id}
    return {
        "status": "ok",
        "control_type": "exact_baseline_repeat",
        "baseline_seed": s,
        "replicate_seed": control_seed,
        "improved": imp,
        "regressed": reg,
        "net": imp - reg,
        "n_paired": paired,
        "per_dataset": per_ds,
        "expectation": "same release and prompt reproduce the same per-task answers",
    }


def _skip_reason(target: Optional[str], state: ArtifactState, rows: List[Dict[str, Any]],
                 cfg: Dict[str, Any]) -> str:
    """候选生成器返回 None 时，给出**可复核**的原因（不同工件原因不同，不能一句话糊过去）。"""
    generic = "无可用补丁（指令已存在 / 抽不出新关键词 / 无可变 Skill）"
    if target == "new_skill":
        try:
            from .crystallize import plan_new_skill
            return plan_new_skill(state, rows, cfg).get("reason") or generic
        except Exception as exc:  # pragma: no cover - 诊断路径
            return f"结晶规划失败：{exc}"
    if target == "skill_body":
        return ("目标 Skill 正文已含全部候选步骤，或反事实重跑没有净增益"
                "（候选只在 net>0 时才提案）")
    if target == "skill_description":
        return "关键词已在 description/triggers 中，或没有可变 Skill"
    return generic


def _verdict_reason(*, evaluation_support_passed: bool, non_regress: bool,
                    gate_ok: bool, seed_passed: bool, tolerance_passed: bool,
                    secondary_passed: bool) -> str:
    reasons = []
    if not evaluation_support_passed:
        reasons.append("评测样本为空、配对题集不一致或 gate 题数与基线不一致")
    if evaluation_support_passed and not non_regress:
        reasons.append("dev_optimize 主指标下降，违反非回归门")
    if not gate_ok:
        reasons.append("selection_gate 指标低于基线 + 阈值")
    if not seed_passed:
        reasons.append("独立 seed 数不足")
    if not tolerance_passed:
        reasons.append("次指标容差门未通过")
    if not secondary_passed:
        reasons.append("task_input coverage/answered_precision 回退")
    return "；".join(reasons) if reasons else (
        "评测 cohort 完整；静态门通过；dev_optimize 主指标不降；"
        "selection_gate 达阈值；seed 与次指标约束通过")


def _load_failures(run_id: str) -> List[Dict[str, Any]]:
    from ..util import read_jsonl
    return read_jsonl(artifacts_path("runs", run_id, "failures.jsonl"))


def _save_candidate(cand: Candidate) -> str:
    p = artifacts_path("candidates", f"{cand.candidate_id}.json")
    write_json(p, cand.to_dict())
    return p.as_posix()
