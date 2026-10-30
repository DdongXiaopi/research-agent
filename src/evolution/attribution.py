"""Trace-based four-class failure attribution (Spec §5.2 / FR-102).

主线只用 4 类，每类都可追溯到 trace event，并直接决定"该改哪个工件"：

    skill.not_recalled  → description
    skill.misapplied    → SKILL.md 正文
    skill.missing       → 结晶新 Skill
    plan.failed         → System Prompt

只有 trace 明确支持时才分配四类。`plan.failed` 是保留的显式/历史标签；当前
trace 规则不会仅因 Skill 被执行就生成它。Skill 已执行但答错时，事件本身不能
区分 Skill 使用错误与 Prompt、模型或题目问题，必须标成 `unknown`；显式
工件实验仍可测试某个组件，但不得把它包装成已确认的因果归因。工具故障、
评测器错误、限流等也标为扩展类，**禁止用改 Prompt 掩盖基础设施故障**。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from ..contracts.types import ATTRIBUTION_CLASSES, ATTRIBUTION_TARGET
from ..contracts.validate import validate_attribution
from ..runtime.runner import TaskOutcome
from ..util import sha256_obj

EXTENDED = {
    "retrieval.no_result",
    "retrieval.low_quality",
    "tool.failed",
    "citation.unsupported",
    "model.capability_limit",
    "evaluator.error",
    "infrastructure.transient",
    "unknown",
}

# 【N3】归因判据版本 —— 进入 final 的"协议指纹"。
# 只要归因判据变了（例如 N1 从 gold_* 改成 evidence_*），版本号必须递增：
# 否则终评日志无法区分"同协议重复重跑"与"换了协议的新一次揭盲"，
# 揭盲次数就只能靠人回忆（本次就差点记错：实测 2 次、口头报成 1 次）。
ATTRIBUTION_RULE_VERSION = "trace-causal-uncertain-v14-empty-registry-missing-proof"


def _task_events(outcome: TaskOutcome) -> List[Dict[str, Any]]:
    """Scope shared run traces to this task; accept legacy synthetic fixtures."""
    events = list(outcome.events or [])
    scoped = [e for e in events if e.get("task_id") == outcome.task_id]
    if scoped:
        return scoped
    # Real Trace events always carry task_id. If present but none match, fail
    # closed instead of attributing this task from another task's event stream.
    if any(e.get("task_id") for e in events):
        return []
    return events


def validated_answer_trace_ids(outcome: Any,
                               events: Optional[List[Dict[str, Any]]] = None) -> set[str]:
    """Return answer-selection and parent-call IDs only for a coherent trace link.

    Parent IDs alone are not enough: router and other model calls share the same
    event type. Fail closed unless the one selected-answer event points to the
    unique task-answer model call and its raw/normalized indexes agree with the
    model event and TaskOutcome. This validates provenance only; it does not
    establish why an answer was wrong.
    """
    scoped = list(events if events is not None else _task_events(outcome))
    selected = [event for event in scoped
                if isinstance(event, dict)
                and event.get("event_type") == "task.answer.selected"]
    if len(selected) != 1:
        return set()
    selection = selected[0]
    selection_id = selection.get("event_id")
    parent_id = selection.get("parent_event_id")
    if (not isinstance(selection_id, str) or not selection_id
            or not isinstance(parent_id, str) or not parent_id):
        return set()
    if sum(1 for event in scoped
           if isinstance(event, dict) and event.get("event_id") == selection_id) != 1:
        return set()
    parents = [event for event in scoped
               if isinstance(event, dict) and event.get("event_id") == parent_id]
    if len(parents) != 1:
        return set()
    parent = parents[0]
    if parent.get("event_type") != "model.call" \
            or parent.get("tool_name") != "model.call.task":
        return set()
    call_payload = parent.get("payload")
    selection_payload = selection.get("payload")
    if not isinstance(call_payload, dict) or not isinstance(selection_payload, dict):
        return set()
    if (not isinstance(call_payload.get("mode"), str)
            or not call_payload.get("mode")
            or call_payload.get("parse_failed") is not False):
        return set()
    raw_index = selection_payload.get("raw_answer_index")
    call_index = call_payload.get("answer_index")
    normalized_index = selection_payload.get("answer_index")
    outcome_index = getattr(outcome, "answer_index", None)
    abstained = selection_payload.get("abstained")
    if (not isinstance(raw_index, int) or isinstance(raw_index, bool)
            or not isinstance(call_index, int) or isinstance(call_index, bool)
            or raw_index != call_index
            or not isinstance(normalized_index, int) or isinstance(normalized_index, bool)
            or not isinstance(outcome_index, int) or isinstance(outcome_index, bool)
            or normalized_index != outcome_index
            or not isinstance(abstained, bool)
            or abstained != (normalized_index < 0)):
        return set()
    return {selection_id, parent_id}


def _skill_routing_evidence(outcome: TaskOutcome) -> Optional[Dict[str, Any]]:
    """Return a route only when the event and outcome record the same decision."""
    routes = [event for event in _task_events(outcome)
              if event.get("event_type") == "skill.candidates"]
    if not routes:
        return None
    if len(routes) != 1:
        raise ValueError(f"task {outcome.task_id} has multiple skill.candidates events")
    payload = routes[0].get("payload")
    if not isinstance(payload, dict):
        return None
    candidates = payload.get("candidates")
    registry_skill_count = payload.get("registry_skill_count")
    mode = payload.get("routing_mode")
    selected = payload.get("selected_skill")
    if (not isinstance(candidates, list)
            or not all(isinstance(name, str) and name for name in candidates)
            or len(candidates) != len(set(candidates))
            or type(registry_skill_count) is not int
            or registry_skill_count < len(candidates)
            or not isinstance(mode, str)
            or mode not in {"lexical", "model", "model_frozen"}
            or selected is not None and (not isinstance(selected, str) or not selected)):
        return None
    if mode in {"model", "model_frozen"} and registry_skill_count != len(candidates):
        return None
    if list(outcome.skill_candidates or []) != candidates:
        raise ValueError(f"task {outcome.task_id} Skill candidates differ between trace and outcome")
    expected_selected = str(outcome.skill_recalled or "")
    trace_selected = selected or ""
    if expected_selected != trace_selected:
        raise ValueError(f"task {outcome.task_id} Skill selection differs between trace and outcome")
    return {"candidates": candidates, "selected_skill": trace_selected,
            "registry_skill_count": registry_skill_count,
            "event_id": routes[0].get("event_id")}


def _loaded_skill_behaviors(outcome: TaskOutcome, selected_skill: str) -> Optional[List[str]]:
    """Return explicit loaded behavior markers when trace and outcome agree."""
    loaded = [event for event in _task_events(outcome)
              if event.get("event_type") == "skill.loaded"]
    if len(loaded) != 1:
        return None
    payload = loaded[0].get("payload")
    if not isinstance(payload, dict) or payload.get("name") != selected_skill:
        return None
    behaviors = payload.get("behaviors")
    if not isinstance(behaviors, list) or not all(
            isinstance(behavior, str) and behavior for behavior in behaviors):
        return None
    recorded = list(outcome.skill_behaviors or [])
    if recorded != behaviors:
        raise ValueError(f"task {outcome.task_id} Skill behaviors differ between trace and outcome")
    return behaviors


def _loaded_skill_body_is_proven_empty(outcome: TaskOutcome,
                                      selected_skill: str) -> bool:
    """Require exact empty-body provenance before targeting a Skill body.

    No recognized behavior marker is not proof of an empty or misapplied Skill:
    the marker parser intentionally recognizes only a small local vocabulary.
    """
    loaded = [event for event in _task_events(outcome)
              if event.get("event_type") == "skill.loaded"]
    if len(loaded) != 1:
        return False
    payload = loaded[0].get("payload")
    if not isinstance(payload, dict) or payload.get("name") != selected_skill:
        return False
    char_count = payload.get("body_char_count")
    body_sha256 = payload.get("body_sha256")
    return (type(char_count) is int and char_count == 0
            and body_sha256 == hashlib.sha256(b"").hexdigest())


def _plan_failure_class(event: Dict[str, Any]) -> str:
    """Map only known planner contracts to an attribution target.

    Retrieval planning is an optional tool path, so its malformed plan must not
    nominate the mainline System Prompt artifact. Unknown/legacy event shapes
    stay unknown instead of inheriting the mainline label from event type alone.
    """
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return "unknown"
    identity = (payload.get("plan_kind"), event.get("tool_name"),
                payload.get("failure_code"))
    if identity == ("task_answer", "task.answer", "invalid_answer_format"):
        return "plan.failed"
    if identity in {
            ("retrieval_query", "retrieval.query_plan", "invalid_query_plan"),
            ("retrieval_source_selection", "retrieval.source_plan",
             "invalid_query_source_plan"),
    }:
        return "tool.failed"
    return "unknown"


def _retrieval_tool_failure(event: Dict[str, Any]) -> bool:
    """Recognize only the typed failure emitted for the academic-search boundary."""
    payload = event.get("payload")
    return (event.get("event_type") == "tool.call.failed"
            and event.get("tool_name") == "academic.search"
            and isinstance(payload, dict)
            and payload.get("failure_type") == "retrieval_error"
            and isinstance(payload.get("exception_type"), str)
            and bool(payload.get("exception_type")))


def _attribute_from_observed_trace(outcome: TaskOutcome) -> Optional[str]:
    """对一条失败样本判定归因；答对返回 None。

    【N1】判据全部**内容派生**，不读答案键
    -------------------------------------
    旧实现是 `if not outcome.gold_in_corpus` / `if not outcome.gold_in_topk` ——
    这两个字段直接读语料里的答案标记。真实自进化系统的反思模块只能看见 trace
    （召回没召回、引了哪些文档、模型输出了什么），**看不见 `is_gold`**；
    读它做归因，等于让归因器当 oracle router，"归因驱动进化"这条主张就不成立。

    `evidence_in_corpus` / `evidence_in_topk` 是内容派生的主题词代理，
    不是语义蕴含或正确答案。它们只能提示可能的检索瓶颈，不能证明
    缺 Skill，更不能把检索失败当作修改 Skill 正文的训练信号。

    `gold_*` 字段仍存在于 outcome 上，但**只允许**用于离线诊断
    （反事实估计、分析报告），归因路径一律不读 —— 由
    `tests/unit/test_validity_fixes.py::test_attribution_reads_no_answer_key` 静态强制。
    """
    if outcome.correct:
        return None
    # A plan failure is event-driven, but its tool/kind/code tuple also matters:
    # only the task-answer contract maps to the mainline Prompt class. Retrieval
    # query/source planning belongs to the optional tool path and stays extended.
    events = _task_events(outcome)
    plan_failures = [event for event in events
                     if event.get("event_type") == "plan.failed"]
    if plan_failures:
        if len(plan_failures) != 1:
            return "unknown"
        return _plan_failure_class(plan_failures[0])
    runtime_tool_failures = [event for event in events
                             if event.get("event_type") == "tool.call.failed"]
    if runtime_tool_failures:
        if (len(runtime_tool_failures) == 1
                and _retrieval_tool_failure(runtime_tool_failures[0])):
            return "tool.failed"
        return "unknown"
    if outcome.status == "error":
        return "infrastructure.transient"
    if outcome.input_mode == "retrieval":
        searched = any(e.get("event_type") == "tool.call"
                       and e.get("tool_name") == "academic.search"
                       for e in events)
        if searched and not outcome.retrieved_doc_ids:
            return "retrieval.no_result"
        if not outcome.evidence_in_corpus or not outcome.evidence_in_topk:
            # 仅为低质量/低覆盖的诊断线索；不交给 Prompt/Skill 优化器。
            return "retrieval.low_quality"
    route = _skill_routing_evidence(outcome)
    if route is None:
        return "unknown"
    selected = route["selected_skill"]
    candidates = route["candidates"]
    if not selected and candidates:
        return "skill.not_recalled"
    if not selected and not candidates:
        # An empty lexical candidate set only proves that the current lexical
        # matcher found no overlap. It does not prove that no registered Skill
        # could handle the task. Crystallize only when the trace proves the
        # entire registered inventory was empty; otherwise leave the cause open.
        if route["registry_skill_count"] == 0:
            return "skill.missing"
        return "unknown"
    behaviors = _loaded_skill_behaviors(outcome, selected)
    if behaviors == []:
        # The local behavior parser is deliberately incomplete. Only an exact
        # empty-body trace can support a mainline Skill-body label; nonempty
        # bodies with unrecognized wording remain unresolved.
        if _loaded_skill_body_is_proven_empty(outcome, selected):
            return "skill.misapplied"
    # A recalled Skill plus parsed behavior markers proves invocation, not
    # correct application. The trace alone cannot distinguish a bad Skill step,
    # a Prompt decision error, model limits, or ambiguous task evidence.
    return "unknown"


def attribute(outcome: TaskOutcome) -> Optional[str]:
    """Return a mainline class only when its trace provenance is addressable.

    GEPA consumes this function directly, before the evaluation harness calls
    ``validate_rows``. Require the same unique, nonempty event-ID invariant at
    this boundary so malformed/legacy traces cannot reach optimizer feedback
    with an apparently actionable class.
    """
    cls = _attribute_from_observed_trace(outcome)
    if cls not in ATTRIBUTION_CLASSES:
        return cls
    events = _task_events(outcome)
    if any(not isinstance(event, dict) for event in events):
        return "unknown"
    trace_ids = [event.get("event_id") for event in events]
    if (any(not isinstance(event_id, str) or not event_id for event_id in trace_ids)
            or len(trace_ids) != len(set(trace_ids))):
        return "unknown"
    supporting_ids = _event_ids_for_class(outcome, cls)
    if (not supporting_ids or len(supporting_ids) != len(set(supporting_ids))
            or not set(supporting_ids).issubset(trace_ids)):
        return "unknown"
    return cls


def _event_ids_for_class(outcome: TaskOutcome, cls: Optional[str]) -> List[str]:
    """Return only trace events that support this attribution label.

    Including every event in the task trace made a failure appear traceable even
    when the listed IDs did not support its class. This mapping keeps provenance
    specific. `plan.failed` must have its explicit event; generic model/task-end
    events are never substituted for it.
    """
    event_types = {
        "skill.not_recalled": {"skill.candidates"},
        "skill.missing": {"skill.candidates"},
        "skill.misapplied": {"skill.loaded"},
        "plan.failed": {"plan.failed"},
        "tool.failed": {"plan.failed", "tool.call.failed"},
        "retrieval.no_result": {"tool.call"},
        "retrieval.low_quality": {"tool.call", "input.evidence"},
        "infrastructure.transient": {"budget.exceeded", "task.end"},
        # A selected-answer event binds the normalized option/abstention to its
        # model call. It improves review provenance, but never names a root cause.
        "unknown": {"model.call", "task.answer.selected", "task.end"},
    }
    events = _task_events(outcome)
    allowed = event_types.get(cls or "", set())
    # For an unresolved task-answer failure, the selected-answer event is the
    # review anchor. Include only the model call it explicitly names as parent;
    # a task can contain other model calls (routing, retrieval planning, judges)
    # that are not evidence for the selected answer. Preserve task.end as the
    # task boundary and leave the complete trace available in trace_event_ids.
    answer_trace_ids = (validated_answer_trace_ids(outcome, events)
                        if cls == "unknown" else set())
    ids: List[str] = []
    for event in events:
        event_id = event.get("event_id")
        event_type = event.get("event_type")
        if not event_id:
            continue
        if cls == "unknown":
            if event_type == "model.call":
                if str(event_id) not in answer_trace_ids:
                    continue
            elif event_type == "task.answer.selected":
                if str(event_id) not in answer_trace_ids:
                    continue
            elif event_type != "task.end":
                continue
        elif event_type not in allowed:
            continue
        if cls == "retrieval.no_result" and event.get("tool_name") != "academic.search":
            continue
        if cls == "retrieval.low_quality" and event_type == "tool.call" \
                and event.get("tool_name") != "academic.search":
            continue
        if cls in {"plan.failed", "tool.failed"} \
                and not (_plan_failure_class(event) == cls
                         if event_type == "plan.failed"
                         else cls == "tool.failed" and _retrieval_tool_failure(event)):
                continue
        ids.append(str(event_id))
    return ids


def _attribution_basis(cls: Optional[str]) -> str:
    """Name the observable rule behind a label without claiming causality."""
    return {
        "skill.not_recalled": "candidate_skill_exists_but_no_skill_was_selected",
        "skill.missing": "registered_skill_inventory_is_empty",
        "skill.misapplied": "skill_loaded_with_verified_empty_body",
        "plan.failed": "explicit_task_answer_contract_failure",
        "tool.failed": "explicit_retrieval_tool_or_plan_contract_failure",
        "retrieval.no_result": "academic_search_trace_returned_no_documents",
        "retrieval.low_quality": "content_derived_evidence_proxy_missing",
        "infrastructure.transient": "task_ended_with_runtime_error_status",
        "unknown": "failure_trace_does_not_isolate_a_root_cause",
    }.get(cls or "", "versioned_trace_rule; heuristic, not independent causal proof")


def target_artifact(cls: str) -> Optional[str]:
    return ATTRIBUTION_TARGET.get(cls)


def is_mainline(cls: str) -> bool:
    return cls in ATTRIBUTION_CLASSES


def attribute_all(outcomes: List[TaskOutcome]) -> List[Dict[str, Any]]:
    """批量归因，返回可直接写盘的失败样本记录。"""
    rows: List[Dict[str, Any]] = []
    for o in outcomes:
        _validate_skill_selection_trace(o)
        if o.correct:
            continue
        cls = attribute(o)
        failure_identity = {
            "dataset": str(o.dataset),
            "split": str(o.split),
            "task_id": str(o.task_id),
        }
        rec = {
            # Task IDs from a dataset often share a long prefix (HealthVer's
            # selected 54-row cohort shares its first eight characters across
            # every row). Hash the complete task scope so candidate lineage
            # and trace references remain one-to-one and deterministic.
            "failure_id": f"f-{sha256_obj(failure_identity)}",
            "failure_id_schema": "sha256-dataset-split-task-v1",
            "task_id": o.task_id,
            "dataset": o.dataset,
            "split": o.split,
            "input_mode": o.input_mode,
            "outcome_status": str(o.status),
            "attributed_class": cls,
            "target_artifact": target_artifact(cls) if is_mainline(cls) else None,
            "mainline": is_mainline(cls),
            "attribution_basis": _attribution_basis(cls),
            "attribution_is_causal": False,
            # 归因判据（内容派生，可观测）—— 上面两个是判据本身
            "evidence_in_corpus": o.evidence_in_corpus,
            "evidence_in_topk": o.evidence_in_topk,
            # 以下两个读答案键，**仅作离线诊断**，归因不得依赖（见 attribute() 说明）
            "gold_in_corpus": o.gold_in_corpus,
            "gold_in_topk": o.gold_in_topk,
            "skill_recalled": o.skill_recalled,
            "skill_candidates": list(o.skill_candidates),
            "skill_behaviors": list(o.skill_behaviors),
            "retrieved_doc_ids": list(o.retrieved_doc_ids),
            "input_doc_ids": list(o.input_doc_ids),
            "answer_index": o.answer_index,
            "gold_index": o.gold_index,
            "event_ids": _event_ids_for_class(o, cls),
            # Human correction may consult the complete task-scoped trace. The
            # machine-supported event_ids above remain class-specific.
            "trace_event_ids": [str(event_id) for event_id in dict.fromkeys(
                e.get("event_id") for e in _task_events(o)
                if isinstance(e.get("event_id"), str) and e.get("event_id"))],
        }
        rows.append(rec)
    return rows


def _validate_skill_selection_trace(outcome: TaskOutcome) -> None:
    """Validate route identity before exporting any hypothesis as a failure row."""
    _skill_routing_evidence(outcome)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r["attributed_class"]] = out.get(r["attributed_class"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def validate_rows(rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        if r["mainline"]:
            validate_attribution(r["attributed_class"])
            event_ids = r.get("event_ids")
            trace_event_ids = r.get("trace_event_ids")
            if (not isinstance(trace_event_ids, list)
                    or any(not isinstance(event_id, str) or not event_id
                           for event_id in trace_event_ids)
                    or len(trace_event_ids) != len(set(trace_event_ids))):
                raise ValueError(
                    f"mainline attribution {r['attributed_class']} has invalid full trace event IDs")
            if (not isinstance(event_ids, list) or not event_ids
                    or any(not isinstance(event_id, str) or not event_id
                           for event_id in event_ids)
                    or len(event_ids) != len(set(event_ids))):
                raise ValueError(
                    f"mainline attribution {r['attributed_class']} lacks supporting trace event IDs")
            if not set(event_ids).issubset(trace_event_ids):
                raise ValueError(
                    f"mainline attribution {r['attributed_class']} cites an event ID "
                    "outside its task trace")
