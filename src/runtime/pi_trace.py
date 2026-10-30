"""Read pinned Pi 0.87.1 JSONL sessions for task-level trace attribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class PiTraceError(ValueError):
    pass


def read_pi_session(path: str | Path) -> Dict[str, Any]:
    """Return committed messages, usage, and terminal results from one Pi session."""
    session_path = Path(path)
    try:
        lines = session_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise PiTraceError("empty Pi session")
        header = json.loads(lines[0])
        if not isinstance(header, dict) or header.get("v") != 4 \
                or header.get("kind") != "header" or not header.get("id"):
            raise PiTraceError("unsupported Pi JSONL session header")
        messages = []
        tool_calls = []
        tool_results = []
        manifests = []
        usages = []
        results = []
        seen_ids = set()
        for line in lines[1:]:
            writes = json.loads(line)
            if isinstance(writes, dict):
                writes = [writes]
            if not isinstance(writes, list):
                raise PiTraceError("invalid Pi JSONL transaction")
            for write in writes:
                if not isinstance(write, dict):
                    raise PiTraceError("invalid Pi JSONL write")
                if write.get("kind") == "entry" and write.get("type") == "message":
                    entry_id = write.get("id")
                    if not entry_id or entry_id in seen_ids:
                        raise PiTraceError("missing or duplicate Pi message ID")
                    seen_ids.add(entry_id)
                    message = write.get("message") or {}
                    if not isinstance(message, dict):
                        raise PiTraceError("invalid Pi message")
                    content = message.get("content") or []
                    text = "".join(str(part.get("text") or "")
                                   for part in content if isinstance(part, dict)
                                   and part.get("type") == "text")
                    message_calls = []
                    if isinstance(content, list):
                        for part in content:
                            if not isinstance(part, dict) or part.get("type") != "toolCall":
                                continue
                            call_id = part.get("id")
                            name = part.get("name")
                            arguments = part.get("arguments") or {}
                            if (not isinstance(call_id, str) or not call_id
                                    or not isinstance(name, str) or not name
                                    or not isinstance(arguments, dict)):
                                raise PiTraceError("invalid Pi tool call")
                            call = {"id": call_id, "name": name,
                                    "arguments": arguments,
                                    "entry_id": entry_id}
                            message_calls.append(call)
                            tool_calls.append(call)
                    if message.get("role") == "toolResult":
                        call_id = message.get("toolCallId")
                        name = message.get("toolName")
                        if (not isinstance(call_id, str) or not call_id
                                or not isinstance(name, str) or not name):
                            raise PiTraceError("invalid Pi tool result")
                        result_text = "".join(
                            str(part.get("text") or "") for part in content
                            if isinstance(part, dict) and part.get("type") == "text")
                        tool_results.append({
                            "tool_call_id": call_id, "name": name,
                            "text": result_text,
                            "is_error": bool(message.get("isError")),
                            "entry_id": entry_id,
                        })
                    messages.append({"entry_id": entry_id,
                                     "parent_id": write.get("parentId"),
                                     "role": message.get("role"), "text": text,
                                     "tool_calls": message_calls,
                                     "usage": message.get("usage")})
                elif write.get("kind") == "entry" and write.get("type") == "custom" \
                        and write.get("customType") == "rsi.task_manifest":
                    manifests.append({"entry_id": write.get("id"),
                                      "data": write.get("data")})
                elif write.get("kind") == "usage":
                    usages.append({"entry_id": write.get("entryId"),
                                   "usage": write.get("usage")})
                elif write.get("kind") == "value" and write.get("op") == "set" \
                        and write.get("namespace") == "pi.result":
                    results.append(write.get("value"))
        if not messages or not results or len(manifests) != 1:
            raise PiTraceError("Pi session lacks messages, result, or task manifest")
        return {"session_id": header["id"], "messages": messages,
                "tool_calls": tool_calls, "tool_results": tool_results,
                "usages": usages, "results": results,
                "task_manifest": manifests[0],
                "message_count": len(messages), "path": str(session_path)}
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise PiTraceError(f"cannot read Pi JSONL session: {exc}") from exc
