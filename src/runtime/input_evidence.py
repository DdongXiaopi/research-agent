"""Turn benchmark-supplied material into a stable task input, without search."""

from __future__ import annotations

from typing import Any, Dict, List

from ..util import sha256_text


def task_input_docs(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use only the text supplied with this task; never query an external corpus."""
    supplied = str(task.get("evidence") or task.get("context") or "").strip()
    if not supplied:
        return []
    sources = task.get("sources") or []
    source = str(sources[0]) if sources else ""
    return [{"doc_id": "input:0", "text": supplied, "url": source,
             "source_kind": "task_input", "content_sha256": sha256_text(supplied),
             "rank": 0}]
