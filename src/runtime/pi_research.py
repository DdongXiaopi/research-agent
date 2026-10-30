"""Open-research model calls through the pinned Pi harness.

Retrieval and quotation validation remain owned by the Python workflow. Pi
executes the selected Skill and model call, and its session is checked against
the exact caller-owned inputs before the answer is accepted.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ..config import ConfigError, verify_pi_runtime_identity
from ..util import sha256_file, sha256_text
from .call_ledger import record_operation_end, start_operation
from .pi_bridge import (_BRIDGE, _ROOT, _cost, _invoke_bridge, _snapshot_skills,
                        _validate_task_prompt_in_session)
from .pi_trace import PiTraceError, read_pi_session
from .provider import ChatOutput, ProviderError, load_project_dotenv


class PiResearchProvider:
    def __init__(self, cfg, state, task_id, selected_skill, sources, run_dir, *, ephemeral=False):
        self.cfg, self.state, self.task_id = cfg, state, task_id
        self.selected_skill, self.sources = selected_skill or "", sources
        self.run_dir, self.ephemeral = Path(run_dir), ephemeral
        self.trace = None
        if not _BRIDGE.exists() or not (_BRIDGE.parent / "node_modules").exists():
            raise ProviderError("Pi bridge dependency is missing; run npm ci in tools/pi-bridge")
        try:
            verify_pi_runtime_identity()
        except ConfigError as exc:
            raise ProviderError(str(exc)) from exc

    def chat_measured(self, system, user, max_tokens=None):
        runtime = self.cfg.get("runtime") or {}
        model = (self.cfg.get("models") or {}).get("task") or {}
        fake = runtime.get("pi_fake_response")
        model_id = str(model.get("model_id") or "")
        if fake is None and not model_id:
            raise ProviderError("Pi task model_id is required")
        skills = _snapshot_skills(self.state.registry)
        selected = next((skill for skill in skills if skill["name"] == self.selected_skill), None)
        if self.selected_skill and not selected:
            raise ProviderError("selected Pi Skill is missing from registry")
        response_format = (model.get("extra_body") or {}).get("response_format")
        response_hash = sha256_text(json.dumps(response_format, ensure_ascii=False,
                                              sort_keys=True, separators=(",", ":")))
        trace_id = uuid4().hex
        payload = {
            # Pi encodes cwd into the session directory name. The research
            # harness exposes no filesystem tools and uses absolute Skill
            # paths, so use the drive root to avoid Windows MAX_PATH failures
            # when the project itself lives in a deeply nested worktree.
            "cwd": str(Path(_ROOT.anchor)), "sessionsRoot": str(self.run_dir / "pi-sessions"),
            "taskId": self.task_id, "taskSplit": "independent_confirmation" if self.ephemeral else "dev_optimize",
            "releaseId": self.state.registry.release_id, "ephemeral": self.ephemeral,
            "systemPrompt": system, "userPrompt": user, "skills": skills,
            "selectedSkill": self.selected_skill, "providerTraceId": trace_id,
            "catalogCharBudget": int((self.cfg.get("skills") or {}).get("max_catalog_chars", 16000)),
            "selectedSkillCharBudget": int((self.cfg.get("skills") or {}).get("max_selected_chars", 20000)),
            "sources": [{"docId": row["source_id"], "contentSha256": row["content_sha256"],
                         "url": row.get("url", "")} for row in self.sources],
            "modelId": model_id, "baseUrl": model.get("base_url", "https://api.siliconflow.cn/v1"),
            "apiKeyEnv": model.get("api_key_env", "SILICONFLOW_API_KEY"),
            "maxTokens": int(max_tokens or model.get("max_tokens", 1800)),
            "temperature": float(model.get("temperature", 0)),
            "enableThinking": bool(model.get("enable_thinking", False)),
            "extraBody": {k: v for k, v in (model.get("extra_body") or {}).items() if k != "response_format"},
            "responseFormat": response_format, "responseFormatSha256": response_hash,
            "providerMaxRetries": max(0, min(2, int(runtime.get("pi_provider_max_retries", 0)),
                                              int(runtime.get("open_research_physical_limit", 1)) - 1)),
        }
        if fake is not None:
            payload["fakeResponse"] = str(fake)
        else:
            load_project_dotenv()
        operation = None if fake is not None else start_operation(None, "pi", model_id)
        if operation:
            payload["ledgerOperationId"] = operation
        try:
            proc, attempts = _invoke_bridge(payload, timeout_seconds=float(runtime.get("pi_timeout_seconds", 180)),
                                             max_attempts=1)
            if proc.returncode:
                # Bridge errors may contain provider messages: do not expose credentials or prompts.
                raise ProviderError("Pi research bridge failed; see bounded request ledger")
            result = json.loads(proc.stdout)
            content = result["assistantText"]
            if not isinstance(content, str) or not content:
                raise PiTraceError("Pi assistant response missing")
            if result.get("providerTraceId") != trace_id or result.get("selectedSkill") != self.selected_skill:
                raise PiTraceError("Pi result identity mismatch")
            manifest = result.get("taskManifest")
            expected = {"taskId": self.task_id, "taskSplit": payload["taskSplit"],
                        "releaseId": payload["releaseId"], "providerTraceId": trace_id,
                        "systemPrompt": system, "systemPromptSha256": sha256_text(system),
                        "userPromptSha256": sha256_text(user), "selectedSkill": self.selected_skill,
                        "sources": payload["sources"], "responseFormatSha256": response_hash}
            if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
                raise PiTraceError("Pi research manifest mismatch")
            if self.ephemeral:
                if not result.get("ephemeral") or result.get("sessionPath"):
                    raise PiTraceError("independent evaluation must use an ephemeral Pi session")
                session = {"messages": [{"role": "user", "text": text}
                                         for text in result.get("userMessages", [])]}
            else:
                session_path = Path(result["sessionPath"]).resolve()
                session_path.relative_to((self.run_dir / "pi-sessions").resolve())
                session = read_pi_session(session_path)
                if (session["session_id"] != result.get("sessionId")
                        or session["task_manifest"]["data"] != manifest
                        or session["message_count"] != result.get("messageCount")
                        or not any(r.get("operationId") == result.get("operationId") and r.get("status") == "completed"
                                   for r in session["results"] if isinstance(r, dict))
                        or not any(m["role"] == "assistant" and m["text"] == content for m in session["messages"])):
                    raise PiTraceError("Pi research session does not match returned result")
            _validate_task_prompt_in_session(session, user_prompt=user, selected_skill=self.selected_skill,
                skill_body=selected["content"] if selected else "", skill_file_path=selected["filePath"] if selected else "")
            usage = result.get("usage") or {}
            read, written = int(usage.get("cacheRead") or 0), int(usage.get("cacheWrite") or 0)
            input_tokens, output_tokens = int(usage.get("input") or 0) + read + written, int(usage.get("output") or 0)
            cost = _cost(input_tokens, output_tokens, model_id, fake=fake is not None,
                         cache_read_tokens=read, cache_write_tokens=written,
                         usage_available=usage.get("input") is not None and usage.get("output") is not None,
                         pi_usage_cost=usage.get("cost"), base_url=payload["baseUrl"])
            cost["usage_complete"] = bool(fake is None and usage.get("input") is not None
                and usage.get("output") is not None and len(result.get("providerAttempts") or []) == 1)
            self.trace = {"session_id": result.get("sessionId"), "operation_id": result.get("operationId"),
                          "session_path": result.get("sessionPath") or "", "ephemeral": self.ephemeral,
                          "session_sha256": None if self.ephemeral else sha256_file(session_path),
                          "provider_trace_id": trace_id, "bridge_attempts": attempts,
                          "provider_http_requests": len(result.get("providerAttempts") or [])}
            record_operation_end(operation, status="success")
            return ChatOutput(content=content, prompt_tokens=input_tokens, completion_tokens=output_tokens,
                              latency_ms=int(result.get("latencyMs") or 0), cost=cost,
                              http_attempts=self.trace["provider_http_requests"], model_id=model_id)
        except BaseException as exc:
            record_operation_end(operation, status="failed", error_type=type(exc).__name__)
            raise
