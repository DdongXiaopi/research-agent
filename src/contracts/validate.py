"""契约字段校验（轻量版，不引入 schema 依赖）。

主线只校验 [M] 必填字段存在且类型正确；增强字段缺失不报错。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from .types import (
    ATTRIBUTION_CLASSES,
    ARTIFACT_TYPES,
    CANDIDATE_STATUS,
    Candidate,
    EvaluationResult,
    ReviewRecord,
    TaskItem,
    TraceEvent,
)

REQUIRED: Dict[str, Dict[str, type | Tuple[type, ...]]] = {
    "TaskItem": {"task_id": str, "dataset": str, "question": str, "correct_index": int, "split": str},
    "TraceEvent": {"run_id": str, "event_id": str, "event_type": str},
    "Candidate": {
        "candidate_id": str,
        "parent_release_id": str,
        "artifact_type": str,
        "artifact_id": str,
        "status": str,
    },
    "EvaluationResult": {"candidate_id": str, "split": str, "metric_values": dict},
    "ReviewRecord": {"action_type": str, "subject_sha256": str, "decision": str},
}


class ContractError(ValueError):
    pass


def validate(name: str, obj: Dict[str, Any]) -> None:
    spec = REQUIRED.get(name)
    if spec is None:
        raise ContractError(f"unknown contract: {name}")
    missing = [k for k in spec if k not in obj or obj[k] is None]
    if missing:
        raise ContractError(f"{name} missing required fields: {missing}")
    for k, t in spec.items():
        v = obj[k]
        if isinstance(t, tuple):
            if not isinstance(v, t):
                raise ContractError(f"{name}.{k} type mismatch: {type(v).__name__}")
        elif not isinstance(v, t):
            raise ContractError(f"{name}.{k} type mismatch: {type(v).__name__}")


def validate_attribution(cls: str) -> None:
    if cls not in ATTRIBUTION_CLASSES:
        raise ContractError(
            f"attribution class '{cls}' not in mainline 4 classes {list(ATTRIBUTION_CLASSES)}"
        )


def validate_artifact_type(t: str) -> None:
    if t not in ARTIFACT_TYPES:
        raise ContractError(f"artifact type '{t}' not in {list(ARTIFACT_TYPES)}")


def validate_candidate_status(s: str) -> None:
    if s not in CANDIDATE_STATUS:
        raise ContractError(f"candidate status '{s}' not in {list(CANDIDATE_STATUS)}")
