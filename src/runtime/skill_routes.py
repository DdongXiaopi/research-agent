"""Experiment-scoped Skill route freezing for paired candidate evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Callable

from ..util import sha256_obj
from .input_evidence import task_input_docs


def route_input_sha256(task: dict, *, registry_sha256: str,
                       router_model_id: str, input_mode: str) -> str:
    """Fingerprint every task field that can affect a frozen route decision."""
    visible_evidence = task_input_docs(task) if input_mode == "task_input" else []
    return sha256_obj({
        "question": str(task.get("question") or ""),
        "options": [str(value) for value in task.get("options", [])],
        "visible_evidence": [
            {"doc_id": str(doc.get("doc_id") or ""),
             "content_sha256": str(doc.get("content_sha256") or "")}
            for doc in visible_evidence
        ],
        "registry_sha256": str(registry_sha256),
        "router_model_id": str(router_model_id),
    })

class FrozenSkillRouteCache:
    """Resolve a task's model-routed Skill once and reuse it across candidates.

    The cache is scoped to one frozen experiment plan. Entries are bound to a
    fingerprint of the router-visible task input, registry, and router model;
    reusing a task ID after any of those inputs change fails closed. Optional
    persistence makes route assignments survive an interrupted MIPRO run.
    """

    SCHEMA_VERSION = 2

    def __init__(self, *, path: str | Path | None = None,
                 scope_sha256: str = "") -> None:
        self.path = Path(path) if path is not None else None
        self.scope_sha256 = str(scope_sha256 or "")
        if self.path is not None and not self.scope_sha256:
            raise ValueError("persistent frozen Skill routes require a scope hash")
        self._routes: dict[str, dict[str, str | None]] = {}
        self._lock = threading.RLock()
        if self.path is not None and self.path.exists():
            self._load()

    def resolve(self, task_id: str, input_sha256: str,
                choose_route: Callable[[], object]
                ) -> tuple[str | None, bool, dict]:
        task_id = str(task_id or "").strip()
        input_sha256 = str(input_sha256 or "").strip()
        if not task_id or not input_sha256:
            raise ValueError("frozen Skill route requires task ID and input hash")
        with self._lock:
            existing = self._routes.get(task_id)
            if existing is not None:
                if existing["input_sha256"] != input_sha256:
                    raise ValueError(
                        f"frozen Skill route input changed for task {task_id}")
                return existing["skill_name"], True, dict(existing["metadata"])

            resolved = choose_route()
            if isinstance(resolved, tuple) and len(resolved) == 2:
                selected, metadata = resolved
            else:
                selected, metadata = resolved, {}
            if selected is not None:
                selected = str(selected).strip()
                if not selected:
                    raise ValueError("frozen Skill route cannot be an empty name")
            if not isinstance(metadata, dict):
                raise ValueError("frozen Skill route metadata must be an object")
            self._routes[task_id] = {
                "input_sha256": input_sha256,
                "skill_name": selected,
                "metadata": metadata,
            }
            self._persist()
            return selected, False, dict(metadata)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema_version": self.SCHEMA_VERSION,
                "scope_sha256": self.scope_sha256,
                "routes": {key: dict(value)
                           for key, value in sorted(self._routes.items())},
            }

    def contains(self, task_id: str) -> bool:
        with self._lock:
            return str(task_id) in self._routes

    def __len__(self) -> int:
        with self._lock:
            return len(self._routes)

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("frozen Skill route checkpoint is unreadable") from exc
        if raw.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported frozen Skill route checkpoint schema")
        if raw.get("scope_sha256") != self.scope_sha256:
            raise ValueError("frozen Skill route checkpoint belongs to a different scope")
        routes = raw.get("routes")
        if not isinstance(routes, dict):
            raise ValueError("frozen Skill route checkpoint has invalid routes")
        for task_id, route in routes.items():
            if (not str(task_id).strip() or not isinstance(route, dict)
                    or not str(route.get("input_sha256") or "").strip()
                    or (route.get("skill_name") is not None
                        and not str(route.get("skill_name") or "").strip())
                    or not isinstance(route.get("metadata"), dict)):
                raise ValueError("frozen Skill route checkpoint has an invalid entry")
        self._routes = routes

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
