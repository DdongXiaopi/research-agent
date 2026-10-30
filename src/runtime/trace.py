"""Trace 读写（Spec FR-101）。

主线不单独建 trace-store：直接写 JSONL，一行一个 TraceEvent，带 event_id / parent_event_id，
足以把单任务还原出来，不靠文本日志猜。
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..util import append_jsonl, ensure_dir, read_jsonl, short_hash, sha256_obj


class Trace:
    """一次运行的事件流（内存 + 可选落盘）。"""

    def __init__(self, run_id: str, task_id: str = "", out_path: Optional[Path] = None,
                 truncate_output: bool = True):
        self.run_id = run_id
        self.task_id = task_id
        self.out_path = out_path
        self.events: List[Dict[str, Any]] = []
        self._seq = itertools.count(1)
        if out_path:
            ensure_dir(out_path.parent)
            if truncate_output:
                # Evaluation run IDs are deterministic fingerprints. Re-running
                # the same run replaces its artifacts, so its trace must also
                # start fresh instead of appending a second event sequence.
                out_path.write_text("", encoding="utf-8")

    def emit(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        seq = next(self._seq)
        ev = {
            "run_id": self.run_id,
            "event_id": f"e{seq:04d}",
            "event_type": event_type,
            "task_id": self.task_id,
            "tool_name": fields.pop("tool_name", ""),
            "latency_ms": int(fields.pop("latency_ms", 0)),
            "token_usage": fields.pop("token_usage", {}),
            "error": fields.pop("error", ""),
            "parent_event_id": fields.pop("parent_event_id", ""),
            "payload": fields.pop("payload", {}),
        }
        ev.update({k: v for k, v in fields.items() if k not in ev})
        self.events.append(ev)
        if self.out_path:
            append_jsonl(self.out_path, ev)
        return ev

    def task_ids(self) -> List[str]:
        return sorted({e.get("task_id", "") for e in self.events if e.get("task_id")})


def run_dir(root: str | Path, run_id: str) -> Path:
    p = Path(root) / "runs" / run_id
    ensure_dir(p)
    return p


def read_trace(path: str | Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def new_run_id(tag: str = "run") -> str:
    import os
    import time
    import uuid

    return f"{tag}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


def audit_task_execution(events: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    """Reconstruct a visible runtime workflow; never infer model reasoning/cause.

    This validates recorded consistency, not authenticity against an adversary
    who can rewrite the entire trace. Legacy traces without a plan are rejected,
    rather than retrospectively inventing one for a historical experiment.
    """
    def require(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(message)

    rows = [event for event in events if event.get("task_id") == task_id]
    require(bool(rows), "task not found in trace")
    run_ids = {row.get("run_id") for row in rows}
    require(len(run_ids) == 1 and all(run_ids), "ambiguous run identity")
    by_id: Dict[str, Any] = {}
    for row in rows:
        event_id = row.get("event_id")
        require(isinstance(event_id, str) and bool(event_id) and event_id not in by_id,
                "duplicate or missing event identity")
        parent = row.get("parent_event_id")
        require(not parent or parent in by_id, "missing or forward parent event")
        by_id[event_id] = row

    def one(kind: str) -> Dict[str, Any]:
        matches = [row for row in rows if row.get("event_type") == kind]
        require(len(matches) == 1, f"expected one {kind}")
        return matches[0]

    start, plan, end = one("task.start"), one("task.plan"), one("task.end")
    require(rows[0] is start and rows[-1] is end, "invalid task boundaries")
    plan_payload = plan.get("payload") or {}
    require(plan.get("parent_event_id") == start["event_id"]
            and rows[1] is plan
            and plan_payload.get("schema") == "runtime-workflow-plan-v1"
            and plan_payload.get("planner") == "runtime_workflow"
            and plan_payload.get("causal_evidence") is False
            and plan_payload.get("model_reasoning_recorded") is False,
            "invalid visible workflow plan")
    require(isinstance(plan_payload.get("task_input_sha256"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", plan_payload["task_input_sha256"])),
            "missing plan input identity")
    require(end.get("parent_event_id") == start["event_id"], "terminal has wrong task")
    status = (end.get("payload") or {}).get("status")
    require(status in {"ok", "abstain", "error"}, "invalid terminal status")
    requests = [row for row in rows if row.get("event_type") == "task.answer.requested"]
    calls = [row for row in rows if row.get("tool_name") == "model.call.task"
             and row.get("event_type") in {"model.call", "model.call.failed"}]
    require(len(requests) <= 1 and len(calls) == len(requests), "unclosed answer dispatch")
    call = calls[0] if calls else None
    request = requests[0] if requests else None
    if request:
        payload = request.get("payload") or {}
        binding = payload.get("binding")
        require(isinstance(binding, dict)
                and binding.get("scope") == "provider_complete_arguments"
                and binding.get("task_input_sha256") == plan_payload.get("task_input_sha256")
                and payload.get("binding_sha256") == sha256_obj(binding),
                "answer input binding mismatch")
        require(all(isinstance(binding.get(key), str)
                    and bool(re.fullmatch(r"[0-9a-f]{64}", binding[key]))
                    for key in ("system_prompt_sha256", "skill_body_sha256",
                                "documents_sha256", "context_sha256", "channels_sha256")),
                "missing provider argument identity")
        require(request.get("parent_event_id") == plan["event_id"], "request has wrong plan")
        call_payload = call.get("payload") or {}
        require(call_payload.get("request_event_id") == request["event_id"]
                and call_payload.get("request_binding_sha256") == payload["binding_sha256"]
                and rows.index(request) < rows.index(call), "call has wrong request")
    selected = [row for row in rows if row.get("event_type") == "task.answer.selected"]
    require(len(selected) <= 1, "ambiguous answer selection")
    selection = selected[0] if selected else None
    if selection:
        payload = selection.get("payload") or {}
        raw, normalized = payload.get("raw_answer_index"), payload.get("answer_index")
        require(call is not None and call.get("event_type") == "model.call"
                and (call.get("payload") or {}).get("parse_failed") is False
                and selection.get("parent_event_id") == call["event_id"]
                and type(raw) is int and type(normalized) is int
                and raw == call["payload"].get("answer_index")
                and normalized >= -1 and (normalized == raw or normalized == -1)
                and payload.get("abstained") is (normalized < 0),
                "inconsistent selected answer")
        require(rows.index(call) < rows.index(selection), "answer precedes call")
    if status != "error":
        require(selection is not None, "completed task has no selected answer")
        require((status == "abstain") == selection["payload"]["abstained"],
                "answer and terminal status disagree")
    report = {
        "schema": "visible-task-execution-audit-v1",
        "run_id": next(iter(run_ids)), "task_id": task_id,
        "trace_sha256": sha256_obj(rows), "status": status,
        "plan_event_id": plan["event_id"],
        "answer_request_event_id": request["event_id"] if request else None,
        "answer_call_event_id": call["event_id"] if call else None,
        "answer_selection_event_id": selection["event_id"] if selection else None,
        "actions": [{key: row.get(key) for key in ("event_id", "event_type", "tool_name")}
                    for row in rows],
        "model_reasoning_recorded": False, "causal_evidence": False,
        "scope": "recorded runtime workflow and provider-boundary argument identity",
    }
    report["report_sha256"] = sha256_obj(report)
    return report
