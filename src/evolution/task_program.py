"""Task-input candidate envelope for a single parent-relative component change."""

from __future__ import annotations

import json
from typing import Any, Dict

from .fewshot import parse_demos_bundle, serialize_demos_bundle

TASK_PROGRAM_VERSION = 1


def serialize_task_program(system_prompt: str, task_demos: Any) -> str:
    """Serialize an instruction plus verified visible-dev examples as one artifact."""
    if isinstance(task_demos, str):
        demos = parse_demos_bundle(task_demos)
    else:
        demos = list(task_demos or [])
        # Reuse the demo bundle validator for IDs, splits, labels, and count.
        demos = parse_demos_bundle(serialize_demos_bundle(demos))
    prompt = str(system_prompt or "")
    if not prompt.strip():
        raise ValueError("task program system_prompt cannot be empty")
    return json.dumps({"version": TASK_PROGRAM_VERSION,
                       "system_prompt": prompt,
                       "task_demos": demos}, ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"))


def parse_task_program(content: str) -> Dict[str, Any]:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid task-program JSON: {exc}") from exc
    if not isinstance(value, dict) or value.get("version") != TASK_PROGRAM_VERSION:
        raise ValueError("unsupported task-program version")
    prompt = str(value.get("system_prompt") or "")
    if not prompt.strip():
        raise ValueError("task program system_prompt cannot be empty")
    demos = value.get("task_demos")
    if not isinstance(demos, list):
        raise ValueError("task program task_demos must be a list")
    # Canonical demo validation also rejects labels changed after selection.
    parsed = parse_demos_bundle(serialize_demos_bundle(demos))
    return {"system_prompt": prompt, "task_demos": parsed}


def changed_task_program_components(content: str, parent_state: Any) -> list[str]:
    """Return parent-relative runtime components changed by a task program.

    A task-program JSON envelope is only a storage format. FR-201 still applies
    to its effective Prompt and demo components, so callers must have the
    parent state and accept exactly one changed component before scoring or
    publication.
    """
    if (parent_state is None or not hasattr(parent_state, "system_prompt")
            or not hasattr(parent_state, "task_demos")):
        raise ValueError("task-program single-component check requires parent state")
    program = parse_task_program(content)
    prompt_changed = program["system_prompt"] != str(parent_state.system_prompt or "")
    candidate_demos = serialize_demos_bundle(program["task_demos"])
    parent_demos = serialize_demos_bundle(list(parent_state.task_demos or []))
    changed = []
    if prompt_changed:
        changed.append("system_prompt")
    if candidate_demos != parent_demos:
        changed.append("task_demos")
    return changed
