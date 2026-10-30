"""跨模块数据契约（Spec §7）。

只严格实现标 [M] 的主线字段；[E] 增强字段预留但不强制。
所有结构都可序列化为稳定 JSON，用于 manifest / trace / review 留痕。
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# 主线 4 类归因（Spec §5.2）
ATTRIBUTION_CLASSES = (
    "skill.not_recalled",
    "skill.misapplied",
    "skill.missing",
    "plan.failed",
)

# 增强 taxonomy（附录 B），主线不进进化
EXTENDED_ATTRIBUTION_CLASSES = (
    "retrieval.no_result",
    "retrieval.low_quality",
    "tool.failed",
    "citation.unsupported",
    "model.capability_limit",
    "evaluator.error",
    "infrastructure.transient",
    "unknown",
)

ARTIFACT_TYPES = ("system_prompt", "skill_description", "skill_body", "new_skill",
                  "task_demos", "task_program")

# 归因 → 应改的工件（Spec §0.1）
ATTRIBUTION_TARGET = {
    "skill.not_recalled": "skill_description",
    "skill.misapplied": "skill_body",
    "skill.missing": "new_skill",
    "plan.failed": "system_prompt",
}

CANDIDATE_STATUS = (
    "draft",
    "static_gate_rejected",
    "deferred",
    "evaluated",
    "rejected",
    "approved",
    "released",
)


def new_id(prefix: str) -> str:
    """**仅用于磁盘目录/文件名防碰撞**的随机 ID。

    ⚠️ 绝不能拿它参与任何排序、裁决或平手判定 —— uuid 每次都不同，
    一旦进入排序键，"同一份代码重跑两次结果不同"就变成结构性问题。
    （GEPA `core/state.py:50-54` 也是同一分工：uuid 只作目录名。）
    需要"身份"时用 `content_id`。
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def content_id(prefix: str, *parts: Any) -> str:
    """内容派生 ID：同一输入 → 同一 ID（对齐 GEPA `core/state.py:37-39`，候选身份用内容 sha256）。

    用途：候选身份、平手裁决、以及**字节级重放比对** —— 有了它，
    "两次运行是否真的等价"就不需要人工去 diff 一堆 uuid。
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return f"{prefix}-{h.hexdigest()[:12]}"


def _d(obj: Any) -> Dict[str, Any]:
    return asdict(obj)


@dataclass
class TaskItem:
    """归一化后的基准题目（来自 parquet，经 scripts/make_splits.py 固化）。"""

    task_id: str
    dataset: str
    question: str
    options: List[str]
    correct_index: int
    split: str
    context: str = ""          # ProtocolQA 的 protocol 正文
    sources: List[str] = field(default_factory=list)
    domain: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _d(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskItem":
        return TaskItem(**{k: v for k, v in d.items() if k in TaskItem.__annotations__})


@dataclass
class TaskRequest:
    task_id: str
    query: str
    task_type: str = "mcq"
    domain_profile: str = "biomedical"
    tool_allowlist: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    project_id: str = "personal"
    session_id: str = "default"
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _d(self)


@dataclass
class Citation:
    source_id: str
    url: str = ""
    locator: str = ""
    retrieved_at: str = ""
    content_sha256: str = ""
    claim: str = ""
    evidence_excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _d(self)


@dataclass
class AgentResponse:
    status: str                      # ok | error | abstain
    answer: str = ""                 # 选项文本
    answer_index: int = -1
    citations: List[Citation] = field(default_factory=list)
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    trace_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = _d(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        return d


@dataclass
class TraceEvent:
    run_id: str
    event_id: str
    event_type: str
    task_id: str = ""
    tool_name: str = ""
    latency_ms: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)
    error: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    parent_event_id: str = ""        # [E]

    def to_dict(self) -> Dict[str, Any]:
        return _d(self)


@dataclass
class Candidate:
    candidate_id: str
    parent_release_id: str
    artifact_type: str
    artifact_id: str                 # system_prompt / skill name
    diff: str = ""
    rationale: str = ""
    status: str = "draft"
    new_content: str = ""
    attributed_class: str = ""
    source_failure_ids: List[str] = field(default_factory=list)
    static_gate_results: List[Dict[str, Any]] = field(default_factory=list)
    generator: Dict[str, Any] = field(default_factory=dict)   # backend / is_gepa
    content_sha256: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)       # 工件特有附加信息（如 skill triggers）

    def to_dict(self) -> Dict[str, Any]:
        return _d(self)


@dataclass
class EvaluationResult:
    candidate_id: str
    split: str
    metric_values: Dict[str, float] = field(default_factory=dict)
    seed: int = 0
    cost: Dict[str, float] = field(default_factory=dict)
    n_tasks: int = 0
    run_id: str = ""
    per_task: List[Dict[str, Any]] = field(default_factory=list)  # 仅 optimize 可见
    leakage_check: Dict[str, Any] = field(default_factory=dict)   # [E]
    failure_class_counts: Dict[str, int] = field(default_factory=dict)
    class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    task_cohort_sha256: str = ""  # task-ID 集合指纹；可核对队列而不暴露逐题答案

    def to_dict(self) -> Dict[str, Any]:
        return _d(self)


@dataclass
class ReviewRecord:
    action_type: str                 # release | rollback | final_unlock
    subject_sha256: str
    decision: str                    # approve | reject
    created_at: str = ""
    reason: str = ""
    reviewer: str = "human"
    subject_id: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_point: str = ""
    parent_release_id: str = ""
    diff: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _d(self)


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
