"""评测执行（Spec §6）：只跑可见分区，严守数据隔离。

两条硬约束：
1. `final_holdout` 只能通过 evaluator 入口（`RSI_ALLOW_FINAL=1` + allow_final=True）加载
2. `selection_gate` 只回传聚合指标与失败类别，**不回传逐题答案**（write_per_task=False）
"""

from __future__ import annotations

import os
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import config_fingerprint
from ..contracts.types import EvaluationResult
from ..data.split import load_tasks
from ..evolution.attribution import attribute_all, summarize, validate_rows
from ..runtime.runner import ArtifactState, TaskRunner
from ..runtime.trace import Trace, run_dir
from ..util import ensure_dir, now_iso, artifacts_path, path, sha256_obj, write_json, write_jsonl
from . import leakage
from .metrics import aggregate, aggregate_by_gold_label
from .options import INSUFFICIENT_OPTION_DEFAULT, inject_all
from .run_manifest import build_run_manifest
from .task_rows import validate_task_answer_row

ARTIFACTS_ROOT = artifacts_path()


class EvaluationOutcomeError(RuntimeError):
    """An evaluation run cannot produce a score for a failed task outcome."""

    def __init__(self, split: str, task_id: str, status: str):
        self.split = split
        self.task_id = task_id
        self.status = status
        super().__init__(
            f"evaluation produced an unscorable outcome: split={split}, "
            f"task_id={task_id}, status={status}")


def _run_id(state: ArtifactState, dataset: str, split: str, seed: int, cfg_fp: str,
            limit: Optional[int] = None) -> str:
    """run 目录 id。

    【P9 修复】`limit` 必须进哈希。否则 `--limit 25` 与全量（46 题）会算出
    同一个 run_id，后者把前者的 trace/summary **静默覆盖** —— 同一个 id 在今天
    先记着 25 题的 acc=0.600、几小时后变成 46 题的 acc=0.717，任何引用旧 id
    的报告都会对不上号。run_id 本就是"这次评估是什么"的指纹，题量是它的一部分。
    """
    fp = sha256_obj(
        {
            "release": state.release_id,
            "prompt": state.system_prompt,
            "skills": state.registry.fingerprint(),
            "task_demos": sha256_obj(state.task_demos),
            "dataset": dataset,
            "split": split,
            "seed": seed,
            "cfg": cfg_fp,
            "limit": limit,
            # Gate 隔离升级后不能复用含逐题 trace 的旧 run 目录。
            "visibility_schema": 2 if split == "selection_gate" else 1,
        }
    )
    return "run-" + fp[:16]


def _task_offset(cfg: Dict[str, Any], split: str) -> int:
    evaluation = cfg.get("evaluation") or {}
    offset = int((evaluation.get("task_offsets") or {}).get(
        split, evaluation.get("task_offset", 0)))
    if offset < 0:
        raise ValueError("evaluation task offset must be nonnegative")
    if offset and split not in {"dev_optimize", "selection_gate"}:
        raise ValueError("task offsets are only permitted for visible development splits")
    return offset


def prepare_tasks(cfg: Dict[str, Any], dataset: str, split: str,
                  limit: Optional[int] = None, allow_final: bool = False) -> tuple[list, int]:
    offset = _task_offset(cfg, split)
    evaluation = cfg.get("evaluation") or {}
    tasks = sorted(load_tasks(dataset, splits=[split], allow_final=allow_final),
                   key=lambda row: row["task_id"])
    tasks = tasks[offset:offset + limit if limit is not None else None]
    if evaluation.get("insufficient_option", INSUFFICIENT_OPTION_DEFAULT):
        tasks = inject_all(tasks)
    if not tasks:
        raise RuntimeError(f"no tasks for {dataset}/{split}")
    return tasks, offset


def evaluate(cfg: Dict[str, Any], state: ArtifactState, dataset: str, split: str,
             seed: int = 0, limit: Optional[int] = None, run_tag: str = "eval",
             write_per_task: Optional[bool] = None, allow_final: bool = False,
             final_access_grant=None,
             cfg_fp: str = "", require_freeze: Optional[bool] = None,
             out_root: Optional[Path] = None, provider=None) -> EvaluationResult:
    # Required grounding metrics are a governance dependency, not merely a
    # post-hoc comparison. Enforce their calibration before any task cohort,
    # protected exposure, or provider request can be reached.
    from ..governance.grounding_attestation import require_grounding_calibration
    require_grounding_calibration(cfg, [dataset])
    _task_offset(cfg, split)  # Reject invalid scope before reserving protected exposure.
    if split == "final_holdout":
        from .unblind_budget import FinalAccessGrant
        if not allow_final or not isinstance(final_access_grant, FinalAccessGrant):
            raise PermissionError(
                "final_holdout evaluation requires a grant returned by the unblind-budget check")
        from .final_eval import protocol_fingerprint
        if (final_access_grant.dataset != dataset
                or final_access_grant.protocol_fingerprint != protocol_fingerprint(cfg)
                or final_access_grant.decision not in {
                    "within_budget", "repeat_allowed", "override_granted"}):
            raise PermissionError(
                "final_holdout budget grant does not match this dataset and protocol")
        from .unblind_budget import claim_final_seed
        claim_final_seed(final_access_grant, state, seed, run_tag, limit)
    persist_selection = False
    if split == "selection_gate":
        # Gate cohorts remain aggregate-only by default.  A frozen diagnostic
        # protocol may opt into per-task persistence explicitly when it needs
        # classwise/paired analysis; the opt-in is carried in the config hash
        # and therefore cannot be confused with an ordinary gate run.
        persist_selection = bool((cfg.get("evaluation") or {}).get("persist_selection_per_task"))
        if write_per_task is True and not persist_selection:
            raise ValueError("selection_gate 禁止导出逐题结果")
        write_per_task = bool(write_per_task and persist_selection)
    elif write_per_task is None:
        write_per_task = True
    if require_freeze is None:
        require_freeze = bool(cfg.get("evaluation", {}).get("require_split_freeze", True))
    if require_freeze:
        # M2：split 必须已冻结且与冻结记录一致，否则历史 run 不可比，拒绝跑
        from ..data.freeze import require_verified
        require_verified([dataset])

    tasks, offset = prepare_tasks(cfg, dataset, split, limit, allow_final)

    # 调用者提供的 cfg_fp 不能代替实际配置指纹；协议变更必须换 run_id。
    pi_source = ({"pi_runtime": config_fingerprint(cfg)}
                 if str((cfg.get("runtime") or {}).get("provider", "")).lower() == "pi"
                 else {})
    effective_cfg_fp = sha256_obj({"caller": cfg_fp, "config": cfg, **pi_source})
    experiment_id = _run_id(state, dataset, split, seed, effective_cfg_fp, limit)
    # A fingerprint identifies an experiment, not a physical execution. Even
    # temperature-zero providers can differ; never replace prior evidence.
    run_id = experiment_id + "-" + uuid4().hex
    rdir = run_dir(Path(out_root) if out_root else ARTIFACTS_ROOT, run_id)
    manifest = build_run_manifest(
        cfg=cfg, state=state, dataset=dataset, split=split, seed=seed,
        run_id=run_id, run_tag=run_tag, tasks=tasks)
    manifest["experiment_id"] = experiment_id
    write_json(rdir / "manifest.json", manifest)
    trace = Trace(run_id, out_path=(rdir / "trace.jsonl"
                  if split != "selection_gate" or persist_selection else None))

    outcomes = []
    for t in tasks:
        runner = TaskRunner(cfg, state, trace=trace, seed=seed, provider=provider)
        if split == "selection_gate" and hasattr(runner.provider, "cache"):
            # HTTP 响应缓存本身也是逐题答案的旁路。
            runner.provider.cache = False
        outcome = runner.run(t)
        status = str(getattr(outcome, "status", ""))
        if status not in {"ok", "abstain"}:
            # An explicit task-answer plan failure is a useful FR-102
            # diagnostic, even though FR-202 correctly makes the evaluation
            # unscorable. Preserve only its trace-backed attribution on the
            # visible optimize split; never turn this outcome into a metric or
            # expose a per-task diagnostic from aggregate-only/protected splits.
            if split == "dev_optimize":
                task_events = list(getattr(outcome, "events", []) or [])
                has_answer_plan_failure = any(
                    isinstance(event, dict)
                    and event.get("event_type") == "plan.failed"
                    and event.get("tool_name") == "task.answer"
                    and isinstance(event.get("payload"), dict)
                    and event["payload"].get("plan_kind") == "task_answer"
                    for event in task_events)
                if has_answer_plan_failure:
                    failure_rows = attribute_all([outcome])
                    validate_rows(failure_rows)
                    if len(failure_rows) != 1:
                        raise RuntimeError(
                            "invalid task-answer failure diagnostic cardinality")
                    failure = failure_rows[0]
                    write_json(rdir / "unscorable_attribution.json", {
                        "schema": "unscorable-task-attribution-v1",
                        "run_id": run_id,
                        "dataset": dataset,
                        "split": split,
                        "task_id": str(t["task_id"]),
                        "outcome_status": status or "missing",
                        "score_valid": False,
                        "attributed_class": failure["attributed_class"],
                        "target_artifact": failure["target_artifact"],
                        "mainline": failure["mainline"],
                        "attribution_basis": failure["attribution_basis"],
                        "attribution_is_causal": False,
                        "event_ids": failure["event_ids"],
                        "trace_event_ids": failure["trace_event_ids"],
                    })
            raise EvaluationOutcomeError(split, str(t["task_id"]), status or "missing")
        outcome_row = (outcome.to_dict() if callable(
            getattr(outcome, "to_dict", None)) else vars(outcome))
        validate_task_answer_row(
            outcome_row, t, context=f"evaluation:{split}:{t['task_id']}",
            require_reported_gold_index=True)
        if split == "selection_gate" and not persist_selection:
            outcome.events = []
            trace.events.clear()
        outcomes.append(outcome)

    metrics = aggregate(outcomes)
    class_metrics = aggregate_by_gold_label(outcomes, tasks)
    failures = attribute_all(outcomes)
    for failure in failures:
        failure["source_run_id"] = run_id
        failure["execution_seed"] = seed
        failure["task_key"] = sha256_obj({k: failure[k] for k in
                                         ("dataset", "split", "task_id")})
        failure["failure_id"] = "f-" + sha256_obj({
            "run_id": run_id, "task_key": failure["task_key"]})
        failure["failure_id_schema"] = "sha256-execution-task-v2"
    if split != "selection_gate" or persist_selection:
        # Visible task traces are required to support every mainline attribution.
        # selection_gate intentionally erases per-task traces and exposes only
        # aggregate metrics/class counts, so it cannot run this row-level check.
        validate_rows(failures)
    fail_summary = summarize(failures)

    write_json(rdir / "summary.json", {
        "run_id": run_id,
        "run_tag": run_tag,
        "experiment_id": experiment_id,
        "dataset": dataset,
        "split": split,
        "task_offset": offset,
        "seed": seed,
        "release_id": state.release_id,
        "manifest_path": "manifest.json",
        "manifest_sha256": sha256_obj(manifest),
        "system_prompt_sha256": sha256_obj(state.system_prompt),
        "skills_fingerprint": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(state.task_demos),
        "metrics": metrics,
        "class_metrics": class_metrics,
        "failure_class_counts": fail_summary,
        "created_at": now_iso(),
    })

    if write_per_task:
        write_jsonl(rdir / "per_task.jsonl", [
            {"task_id": o.task_id, "correct": o.correct, "status": o.status,
             "answer_index": o.answer_index, "gold_index": o.gold_index,
             "skill_recalled": o.skill_recalled, "gold_in_corpus": o.gold_in_corpus,
             "gold_in_topk": o.gold_in_topk, "gold_rank": o.gold_rank,
             "retrieval_query": o.retrieval_query,
             "retrieval_query_source": o.retrieval_query_source,
             "retrieved_doc_ids": o.retrieved_doc_ids,
             "retrieved_passage_hashes": o.retrieved_passage_hashes,
             "input_doc_ids": o.input_doc_ids,
             "input_passage_hashes": o.input_passage_hashes,
             "cited": o.cited,
             "citation_supported": o.citation_supported,
             "citation_support_detail": o.citation_support_detail,
             "answer_evidence_consistent": o.answer_evidence_consistent,
             "answer_evidence_detail": o.answer_evidence_detail}
            for o in outcomes
        ])
        write_jsonl(rdir / "failures.jsonl", failures)

    # 泄漏自检：任何一次**进化可见分区**的运行，产物里都不允许出现 final 题目 ID。
    # final 终评本身就在跑 final，产物里自然有这些 ID，因此跳过（且它的产物落在
    # artifacts/final_holdout/ 独立目录，不在 verify 的扫描范围内）。
    if split == "final_holdout":
        leak = {"ok": True, "skipped": True,
                "reason": "本次运行即 final_holdout 终评，产物独立存放，不参与进化"}
    else:
        leak = leakage.check_dir(rdir, dataset)
    write_json(rdir / "leakage.json", leak)
    from ..util import sha256_file
    receipt = {"schema": "evaluation-execution-receipt-v1", "run_id": run_id,
               "experiment_id": experiment_id, "status": "completed",
               "files": {name: sha256_file(rdir / name) for name in (
                   "manifest.json", "summary.json", "per_task.jsonl", "failures.jsonl",
                   "trace.jsonl", "leakage.json") if (rdir / name).exists()}}
    receipt["receipt_sha256"] = sha256_obj(receipt)
    write_json(rdir / "receipt.json", receipt)

    return EvaluationResult(
        candidate_id=run_tag,
        split=split,
        metric_values=metrics,
        seed=seed,
        cost={"usd": metrics.get("cost_usd"),
              "usd_nominal": metrics.get("cost_usd_nominal", 0.0),
              "cost_estimated": metrics.get("cost_estimated", True),
              "cost_basis": metrics.get("cost_basis", "")},
        n_tasks=len(outcomes),
        run_id=run_id,
        per_task=[{"task_id": o.task_id, "correct": o.correct,
                   "attributed_class": next((f["attributed_class"] for f in failures
                                             if f["task_id"] == o.task_id), None)}
                  for o in outcomes] if write_per_task else [],
        leakage_check=leak,
        failure_class_counts=fail_summary,
        class_metrics=class_metrics,
        task_cohort_sha256=sha256_obj(sorted(t["task_id"] for t in tasks)),
    )


def gate_summary(result: EvaluationResult) -> Dict[str, Any]:
    """selection_gate 对外只暴露聚合指标 + 失败类别（Spec §6.1 铁律 2）。"""
    return {
        "split": result.split,
        "n_tasks": result.n_tasks,
        "metrics": result.metric_values,
        "class_metrics": getattr(result, "class_metrics", {}),
        "failure_class_counts": result.failure_class_counts,
        "run_id": result.run_id,
        "leakage_ok": result.leakage_check.get("ok", False),
    }


def summarize_from_per_task(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        c = r.get("attributed_class")
        if c:
            out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
