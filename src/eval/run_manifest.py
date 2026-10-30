"""Reproducibility manifest for one evaluated cohort and release.

The manifest stores hashes and sanitized model identifiers, never task rows,
answer labels, API credentials, or full prompts/Skill bodies.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import config_fingerprint
from ..runtime.runner import ArtifactState
from ..util import (REPO_ROOT, now_iso, path, sha256_file, sha256_obj)

_SOURCE_ROOTS = ("src", "scripts", "configs", "skills", "tools/pi-bridge", "tools/docker-worker")
_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".txt", ".mjs", ".js", ".json", ".md"}
_SOURCE_EXCLUDED = {".git", "__pycache__", "node_modules", ".venv"}
def source_tree_fingerprint() -> dict[str, Any]:
    """Hash executable project sources, including dirty working-tree contents."""
    file_paths: dict[str, Path] = {}
    for relative_root in _SOURCE_ROOTS:
        root = path(relative_root)
        if not root.exists():
            continue
        for current, directories, filenames in os.walk(root):
            directories[:] = [name for name in directories
                              if name not in _SOURCE_EXCLUDED]
            for filename in filenames:
                file_path = Path(current) / filename
                if file_path.suffix.lower() not in _SOURCE_SUFFIXES:
                    continue
                relative = file_path.relative_to(REPO_ROOT).as_posix()
                file_paths[relative] = file_path
    for relative in ("requirements.txt", "requirements-mipro.txt", "requirements-optimizers.constraints.txt", "requirements-documents.txt", "requirements-paperqa.txt", "UPSTREAM_LOCK.yml", "tools/docker-worker/Dockerfile"):
        file_path = path(relative)
        if file_path.is_file():
            file_paths[relative] = file_path

    # Windows st_ctime is file creation time, so an in-place same-size edit
    # followed by restoring mtime can preserve the entire stat signature.
    # Plans bind actual bytes, therefore recompute hashes at every check.
    hashes = {relative: sha256_file(file_path)
              for relative, file_path in sorted(file_paths.items())}
    return {"sha256": sha256_obj(hashes), "file_count": len(hashes)}


@lru_cache(maxsize=1)
def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _model_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    models = cfg.get("models") or {}
    safe_fields = ("provider", "model_id", "base_url", "temperature",
                   "max_tokens", "enable_thinking")
    return {
        role: {key: model[key] for key in safe_fields if key in model}
        for role, model in models.items() if isinstance(model, dict)
    }


def build_run_manifest(*, cfg: dict[str, Any], state: ArtifactState,
                       dataset: str, split: str, seed: int, run_id: str,
                       run_tag: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a content-free record sufficient to identify an evaluation run."""
    task_ids = sorted(str(task.get("task_id") or "") for task in tasks)
    if not task_ids or any(not task_id for task_id in task_ids):
        raise ValueError("run manifest requires non-empty task IDs")
    manifest_path = path("data", "manifests", f"{dataset}.json")
    task_cohort_sha = sha256_obj(task_ids)
    task_content_sha = sha256_obj(tasks)
    source_tree = source_tree_fingerprint()
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "run_id": run_id,
        "run_tag": run_tag,
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "release_id": state.release_id,
        "code": {
            "git_revision": _git_revision(),
            "source_tree_sha256": source_tree["sha256"],
            "source_file_count": source_tree["file_count"],
            "upstream_lock_sha256": (
                sha256_file(path("UPSTREAM_LOCK.yml"))
                if path("UPSTREAM_LOCK.yml").is_file() else None),
        },
        "configuration": {
            "config_sha256": sha256_obj(cfg),
            "protocol_sha256": config_fingerprint(cfg),
            "domain_profile": cfg.get("domain_profile"),
            "runtime_provider": (cfg.get("runtime") or {}).get("provider"),
            "input_mode": (cfg.get("runtime") or {}).get("input_mode"),
            "models": _model_manifest(cfg),
        },
        "data": {
            "manifest_sha256": (sha256_file(manifest_path)
                                if manifest_path.is_file() else None),
            "evaluated_task_ids_sha256": task_cohort_sha,
            "evaluated_task_content_sha256": task_content_sha,
            "task_count": len(tasks),
        },
        "artifacts": {
            "system_prompt_sha256": sha256_obj(state.system_prompt),
            "skill_registry_fingerprint": state.registry.fingerprint(),
            "skill_count": len(getattr(state.registry, "skills", []) or []),
            "task_demos_sha256": sha256_obj(state.task_demos),
        },
    }
