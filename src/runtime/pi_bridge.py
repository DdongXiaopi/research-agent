"""One-task Python boundary for the pinned Pi harness (task_input only).

Pi owns model execution, Skill invocation, and the durable JSONL session.  The
Python evaluator still owns benchmark input, scoring, and candidate governance.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Dict, Sequence
from uuid import uuid4

from ..util import artifacts_path, ensure_dir, sha256_text
from ..config import ConfigError, verify_pi_runtime_identity
from ..eval.options import explicit_information_gap_index
from .call_ledger import (record_operation_end, reserve_physical_attempts,
                         start_operation, timeout_http_attempt)
from .channels import effective_modes
from .provider import (HttpProvider, ModelOutput, PRICE_PER_1K, ProviderError,
                       _truncate, answer_json_schema_response_format,
                       parse_structured_answer, published_price_for_endpoint,
                       published_tariff_metadata, SILICONFLOW_PRICE_CHECKED_AT)
from .pi_trace import PiTraceError, read_pi_session
from .skills import Registry, load_body
from ..evolution.fewshot import (DEFAULT_MAX_BLOCK_CHARS, DEFAULT_MAX_DEMO_CHARS,
                                 render_demos_block_with_ids)


PI_BRIDGE_PROTOCOL_VERSION = "pi-harness-task-input-v3"
PI_ABSTAIN_POLICIES = frozenset(
    {"literal", "supported_inference", "evidence_qualified"})
_SAFE_SKILL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE = _ROOT / "tools" / "pi-bridge" / "run.mjs"
_PI_COOLDOWN_LOCK = threading.Lock()
_PI_LAST_REQUEST_AT = 0.0


def _apply_request_cooldown(seconds: float) -> None:
    """Throttle sequential Pi calls across provider instances in one process."""
    global _PI_LAST_REQUEST_AT
    seconds = float(seconds)
    if seconds < 0:
        raise ValueError("Pi request cooldown must be nonnegative")
    if seconds == 0:
        return
    with _PI_COOLDOWN_LOCK:
        remaining = seconds - (time.monotonic() - _PI_LAST_REQUEST_AT)
        if remaining > 0:
            time.sleep(remaining)
        _PI_LAST_REQUEST_AT = time.monotonic()


def _snapshot_skills(registry: Registry) -> list[Dict[str, str]]:
    result = []
    for skill in registry.skills:
        if skill.lifecycle != "active":
            continue
        if not _SAFE_SKILL_NAME.fullmatch(skill.name):
            raise ProviderError(f"invalid Pi Skill name: {skill.name}")
        body = load_body(skill)
        from .skill_bundle import render_references, materialize_bundle
        body += render_references(skill)
        if not body.strip():
            raise ProviderError(f"empty Pi Skill body: {skill.name}")
        content = ("---\nname: " + skill.name + "\ndescription: "
                   + json.dumps(skill.description, ensure_ascii=False)
                   + "\n---\n\n" + body.rstrip() + "\n")
        destination = artifacts_path("pi_skill_snapshots") / (
            skill.name + "-" + sha256_text(content + registry.fingerprint())) / "SKILL.md"
        ensure_dir(destination.parent)
        materialize_bundle(skill, destination.parent)
        if destination.exists():
            if destination.read_text(encoding="utf-8") != content:
                raise ProviderError(f"Pi Skill snapshot changed: {destination}")
        else:
            destination.write_text(content, encoding="utf-8")
        result.append({"name": skill.name, "description": skill.description,
                       "content": body, "filePath": str(destination)})
    return result


def _user_prompt(*, question: str, options: Sequence[str],
                 docs: Sequence[Dict[str, Any]], context: str,
                 channels: Any, system_prompt: str,
                 skill_body: str, max_context_chars: int,
                 max_doc_chars: int, abstain_policy: str = "literal",
                 structured_output: str = "") -> str:
    modes = effective_modes(system_prompt, skill_body, channels)
    blocks = []
    if context.strip():
        blocks.append("Context:\n" + _truncate(context, max_context_chars))
    if docs:
        label = ("Task-provided evidence" if all(
            d.get("source_kind") == "task_input" for d in docs)
            else "Retrieved passages")
        lines = [label + " (identifier in brackets):"]
        for doc in docs:
            lines.append(f"[{doc.get('doc_id', '')}] "
                         + _truncate(doc.get("text", ""), max_doc_chars))
        blocks.append("\n".join(lines))
    else:
        blocks.append("Evidence passages: (none)")
    blocks.append("Question: " + question.strip())
    blocks.append("Options:\n" + "\n".join(
        f"{i}. {option}" for i, option in enumerate(options)))
    if structured_output == "json_schema":
        fields = ['"answer_index": <integer option index>']
        if modes.get("citation"):
            fields.append('"cited_doc_ids": [<supporting passage identifiers>]')
        rules = ["Return one JSON object with exactly these fields and no prose:",
                 "{" + ", ".join(fields) + "}"]
    else:
        line_count = "two lines" if modes.get("citation") else "one line"
        rules = [f"Reply with EXACTLY the following {line_count} and nothing else:",
                 "ANSWER: <option number>"]
        if modes.get("citation"):
            rules.append(
                "CITED: <at most 3 comma-separated passage identifiers that support your answer>. "
                "Use only identifiers shown in square brackets in the evidence above; "
                "never invent identifiers or enumerate passages.")
    if modes.get("abstain") and explicit_information_gap_index(list(options)) is None:
        if structured_output == "json_schema":
            if explicit_information_gap_index(list(options)) is None:
                rules.append("Use answer_index -1 only when the supplied evidence cannot distinguish the options.")
        elif abstain_policy == "supported_inference":
            rules.append(
                "Choose an option when the supplied material supports it by "
                "a direct finding or a clear inference from reported results; "
                "the passage need not literally say the option text. "
                "If the material cannot distinguish the options, reply "
                "`ANSWER: INSUFFICIENT` instead of guessing.")
        elif abstain_policy == "evidence_qualified":
            rules.append(
                "Choose an option when the supplied material directly reports "
                "the relevant result or the answer follows clearly from the "
                "reported estimates. Match the exact outcome, comparison, and "
                "population in the question. For effect or association claims, "
                "prefer adjusted or multivariable analyses over unadjusted "
                "trends. A nonsignificant trend alone does not establish an "
                "effect; when the adjusted analysis reports no association or "
                "no significant difference, answer no to the positive effect "
                "claim. Abstain only when relevant findings remain conflicting "
                "or cannot distinguish the options. The passage need not "
                "literally repeat the option text.")
        else:
            rules.append("If no supplied passage directly states any option, "
                         "reply `ANSWER: INSUFFICIENT` instead of guessing.")
    blocks.append("Output format:\n" + "\n".join(rules))
    return "\n\n".join(blocks)


def _validate_task_prompt_in_session(session: Dict[str, Any], *,
                                     user_prompt: str, selected_skill: str,
                                     skill_body: str,
                                     skill_file_path: str = "") -> None:
    """Check the actual persisted/in-memory user message, not only its manifest hash.

    Pi formats a Skill invocation as the loaded Skill block followed by the
    caller's additional instructions. A prompt manifest alone is insufficient:
    it could claim the expected prompt hash while the message sent to the model
    contains different task material.
    """
    user_texts = [str(message.get("text") or "")
                  for message in session.get("messages") or []
                  if message.get("role") == "user"]
    if selected_skill:
        if not skill_file_path:
            raise PiTraceError("Pi Skill invocation path is missing")
        normalized_path = skill_file_path.rstrip("/\\")
        separator = max(normalized_path.rfind("/"), normalized_path.rfind("\\"))
        if separator == 2 and normalized_path[1] == ":":
            skill_directory = normalized_path[:3]
        else:
            skill_directory = "/" if separator <= 0 else normalized_path[:separator]
        expected_skill_block = (
            f'<skill name="{selected_skill}" location="{skill_file_path}">\n'
            f"References are relative to {skill_directory}.\n\n"
            f"{skill_body}\n</skill>")
        expected_message = expected_skill_block + "\n\n" + user_prompt
        if user_texts != [expected_message]:
            raise PiTraceError(
                "Pi Skill user message differs from the expected task prompt")
    elif user_texts != [user_prompt]:
        raise PiTraceError("Pi session user message differs from the expected task prompt")


def _cost(p_tokens: int, c_tokens: int, model: str,
          *, fake: bool, cache_read_tokens: int = 0,
          cache_write_tokens: int = 0,
          usage_available: bool = True,
          pi_usage_cost: Any = None,
          base_url: str = "https://api.siliconflow.cn/v1") -> Dict[str, Any]:
    token_details = {"prompt_tokens": p_tokens,
                     "completion_tokens": c_tokens,
                     "cache_read_tokens": cache_read_tokens,
                     "cache_write_tokens": cache_write_tokens}
    # Pi's usage.cost is computed from the model metadata passed to Pi. Our
    # dynamic model definition currently sets those rates to zero, so preserve
    # the value as an explicitly non-authoritative diagnostic and never use it
    # to populate the billable USD fields below.
    pi_cost_detail = None
    if isinstance(pi_usage_cost, dict):
        numeric = {key: value for key, value in pi_usage_cost.items()
                   if key in {"input", "output", "cacheRead", "cacheWrite", "total"}
                   and isinstance(value, (int, float))
                   and not isinstance(value, bool) and math.isfinite(float(value))
                   and float(value) >= 0}
        if "total" in numeric:
            pi_cost_detail = {key: round(float(value), 9)
                              for key, value in numeric.items()}
    if fake:
        return {"usd": None, "usd_nominal": 0.0,
                "cost_estimated": True, "cost_basis": "Pi faux provider test response",
                "usage_available": True,
                "pi_usage_cost_metadata": pi_cost_detail,
                "pi_usage_cost_authoritative": False, **token_details}
    if not usage_available:
        return {"usd": None, "usd_nominal": None,
                "cost_estimated": True,
                "cost_basis": "Pi API usage counters unavailable; cost not estimable",
                "usage_available": False,
                "pi_usage_cost_metadata": pi_cost_detail,
                "pi_usage_cost_authoritative": False, **token_details}
    price = published_price_for_endpoint(model, base_url)
    nominal = (p_tokens / 1000.0) * PRICE_PER_1K["prompt"] \
        + (c_tokens / 1000.0) * PRICE_PER_1K["completion"]
    cache_tariff_missing = (cache_read_tokens and (price or {}).get("cache_read") is None) \
        or (cache_write_tokens and (price or {}).get("cache_write") is None)
    if price is None or cache_tariff_missing:
        return {"usd": None, "usd_nominal": round(nominal, 6),
                "cost_estimated": True,
                "cost_basis": (f"Pi API usage tokens including cache; no applicable published "
                               f"input/output tariff for endpoint={base_url}, "
                               f"model={model}; Pi usage.cost is model metadata, not provider billing"),
                "usage_available": True,
                "pi_usage_cost_metadata": pi_cost_detail,
                "pi_usage_cost_authoritative": False, **token_details}
    uncached_tokens = p_tokens - cache_read_tokens - cache_write_tokens
    usd = (uncached_tokens * price["prompt"]
           + cache_read_tokens * price.get("cache_read", 0)
           + cache_write_tokens * price.get("cache_write", 0)
           + c_tokens * price["completion"]) / 1_000_000.0
    return {"usd": round(usd, 6), "usd_nominal": round(usd, 6),
            "cost_estimated": False,
            "cost_basis": (f"Pi API usage × SiliconFlow published list tariff; "
                           f"checked={SILICONFLOW_PRICE_CHECKED_AT}; not account invoice"),
            **published_tariff_metadata(model, base_url),
            "usage_available": True,
            "pi_usage_cost_metadata": pi_cost_detail,
            "pi_usage_cost_authoritative": False,
            **token_details}


def _invoke_bridge(payload: Dict[str, Any], *, timeout_seconds: float,
                   max_attempts: int) -> tuple[subprocess.CompletedProcess[str], int]:
    """Retry only a timed-out Pi subprocess, with an explicit attempt bound."""
    if timeout_seconds <= 0 or not 1 <= max_attempts <= 3:
        raise ProviderError("Pi bridge timeout must be positive and max attempts must be 1..3")
    for attempt in range(1, max_attempts + 1):
        try:
            proc = subprocess.run(
                ["node", str(_BRIDGE)], input=json.dumps(payload, ensure_ascii=False),
                text=True, encoding="utf-8", capture_output=True,
                timeout=timeout_seconds, check=False, cwd=str(_BRIDGE.parent))
            return proc, attempt
        except subprocess.TimeoutExpired as exc:
            # The child may have durably reserved its HTTP attempt before the
            # outer Pi process was killed.  Close that physical slot in the
            # parent ledger so a later retry is auditable rather than leaving
            # an apparent open request forever.
            timeout_http_attempt(payload.get("ledgerOperationId"), attempt=0)
            if attempt == max_attempts:
                raise ProviderError(
                    f"Pi bridge timed out after {attempt} attempts "
                    f"of {timeout_seconds:g} seconds") from exc
        except OSError as exc:
            raise ProviderError(f"Pi bridge invocation failed: {exc}") from exc
    raise AssertionError("Pi bridge attempt loop did not return")


class PiProvider:
    is_pi_provider = True

    def __init__(self, cfg: Dict[str, Any], channels: Any):
        if (cfg.get("runtime") or {}).get("input_mode") != "task_input":
            raise ProviderError("Pi bridge currently supports task_input only")
        routing_mode = str((cfg.get("skills") or {}).get("routing_mode", "lexical"))
        if routing_mode not in {"lexical", "model"}:
            raise ProviderError("Pi bridge supports lexical or model Skill routing")
        self.cfg = cfg
        self.channels = channels
        self.mcfg = dict((cfg.get("models") or {}).get("task") or {})
        self.model_id = str(self.mcfg.get("model_id") or "")
        self.fake_response = (cfg.get("runtime") or {}).get("pi_fake_response")
        if self.fake_response is None and not self.model_id:
            raise ProviderError("Pi task model_id is required")
        if not _BRIDGE.exists() or not (_BRIDGE.parent / "node_modules").exists():
            raise ProviderError("Pi bridge dependency is missing; run npm ci in tools/pi-bridge")
        try:
            verify_pi_runtime_identity()
        except ConfigError as exc:
            raise ProviderError(str(exc)) from exc

    def complete(self, *, system_prompt: str, question: str,
                 options: Sequence[str], docs: Sequence[Dict[str, Any]],
                 context: str = "", task_id: str = "", skill_body: str = "",
                 channels: Any = None, selected_skill: str = "",
                 registry: Registry | None = None, task_split: str = "",
                 task_demos: Sequence[Dict[str, Any]] = ()) -> ModelOutput:
        if registry is None:
            raise ProviderError("Pi provider requires the release Skill registry")
        if task_split not in {"dev_optimize", "selection_gate", "final_holdout",
                              "independent_confirmation"}:
            raise ProviderError("Pi task split must be explicitly classified")
        ephemeral = task_split != "dev_optimize"
        skills = _snapshot_skills(registry)
        if selected_skill and not any(s["name"] == selected_skill for s in skills):
            raise ProviderError("selected Pi Skill is missing from registry")
        if selected_skill:
            expected = load_body(registry.get(selected_skill))
            if expected != skill_body:
                raise ProviderError("selected Pi Skill body does not match runner state")
        runtime = self.cfg.get("runtime") or {}
        abstain_policy = str(runtime.get("pi_abstain_policy") or "literal")
        structured_output = str(self.mcfg.get("structured_output") or "").strip().lower()
        if structured_output not in {"", "json_schema"}:
            raise ProviderError("Pi structured_output must be empty or json_schema")
        if abstain_policy not in PI_ABSTAIN_POLICIES:
            raise ProviderError("invalid Pi abstain policy")
        configured_block = str(runtime.get("pi_task_demos_block") or "")
        configured_ids = list(runtime.get("pi_task_demo_ids") or [])
        if task_demos and (configured_block or configured_ids):
            raise ProviderError("task demos must come from either the release or runtime, not both")
        if task_demos:
            active_demos = [item for item in task_demos
                            if str(item.get("task_id")) != task_id]
            demo_ids = [str(item["task_id"]) for item in active_demos]
            try:
                demo_block_limit = int(runtime.get(
                    "pi_task_demos_max_block_chars", DEFAULT_MAX_BLOCK_CHARS))
                demo_char_limit = int(runtime.get(
                    "pi_task_demo_max_example_chars", DEFAULT_MAX_DEMO_CHARS))
            except (TypeError, ValueError) as exc:
                raise ProviderError("Pi task demo rendering limits must be integers") from exc
            if not 1 <= demo_block_limit <= 4000 or demo_char_limit <= 0:
                raise ProviderError("Pi task demo rendering limits are invalid")
            demos_block, rendered_demo_ids = render_demos_block_with_ids(
                active_demos, max_block_chars=demo_block_limit,
                max_demo_chars=demo_char_limit)
            if rendered_demo_ids != demo_ids:
                raise ProviderError(
                    "Pi task demo block truncated candidate examples; refusing to score "
                    f"candidate={demo_ids} rendered={rendered_demo_ids}")
            demo_split = "dev_optimize"
        else:
            active_demos = []
            demos_block = configured_block
            demo_ids = configured_ids
            rendered_demo_ids = list(configured_ids) if configured_block else []
            demo_split = runtime.get("pi_task_demo_split")
        if demos_block:
            if len(demos_block) > 4000 or not demo_ids \
                    or len(demo_ids) != len(set(demo_ids)) \
                    or demo_split != "dev_optimize":
                raise ProviderError("Pi task demos need distinct visible train IDs and a 4000-character budget")
            if configured_block and task_id in demo_ids:
                raise ProviderError("runtime task demos cannot contain the current evaluation task")
        elif demo_ids:
            raise ProviderError("Pi task demo IDs require a demos block")
        effective_prompt = (system_prompt + "\n\nVisible training examples "
                            "(examples only; decide the current task from its own evidence):\n"
                            + demos_block if demos_block else system_prompt)
        provider_trace_id = uuid4().hex
        max_doc_chars = max(int(self.mcfg.get("max_doc_chars", 1500)),
                            int(runtime.get("task_input_max_chars", 30000)))
        modes = effective_modes(system_prompt, skill_body, channels or self.channels)
        valid_doc_ids = [str(d.get("doc_id", "")) for d in docs if d.get("doc_id")]
        abstain_enabled = bool(modes.get("abstain")) and (
            explicit_information_gap_index(list(options)) is None)
        response_format = (answer_json_schema_response_format(
            options, valid_doc_ids, citation_enabled=bool(modes.get("citation")),
            abstain_enabled=abstain_enabled)
            if structured_output == "json_schema" else None)
        response_format_sha256 = (sha256_text(json.dumps(
            response_format, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"))) if response_format else "")
        payload = {
            "cwd": str(_ROOT),
            "sessionsRoot": str(artifacts_path("pi_sessions")),
            "taskId": task_id,
            "taskSplit": task_split,
            "releaseId": registry.release_id,
            "ephemeral": ephemeral,
            "systemPrompt": effective_prompt,
            "demoTaskIds": rendered_demo_ids,
            "candidateDemoTaskIds": ([str(item["task_id"]) for item in task_demos]
                                      if task_demos else configured_ids),
            "demosBlockSha256": sha256_text(demos_block) if demos_block else "",
            "userPrompt": _user_prompt(
                question=question, options=options, docs=docs, context=context,
                channels=channels or self.channels, system_prompt=system_prompt,
                skill_body=skill_body,
                max_context_chars=int(self.mcfg.get("max_context_chars", 6000)),
                max_doc_chars=max_doc_chars,
                abstain_policy=abstain_policy,
                structured_output=structured_output),
            "skills": skills,
            "selectedSkill": selected_skill,
            "catalogCharBudget": int((self.cfg.get("skills") or {}).get("max_catalog_chars", 16000)),
            "selectedSkillCharBudget": int((self.cfg.get("skills") or {}).get("max_selected_chars", 20000)),
            "sources": [{"docId": str(d.get("doc_id") or ""),
                         "contentSha256": str(d.get("content_sha256") or ""),
                         "url": str(d.get("url") or "")}
                        for d in docs],
            "modelId": self.model_id,
            "providerTraceId": provider_trace_id,
            "baseUrl": self.mcfg.get("base_url", "https://api.siliconflow.cn/v1"),
            "apiKeyEnv": self.mcfg.get("api_key_env", "SILICONFLOW_API_KEY"),
            "maxTokens": int(self.mcfg.get("max_tokens", 512)),
            "temperature": float(self.mcfg.get("temperature", 0)),
            "enableThinking": bool(self.mcfg.get("enable_thinking", False)),
            "extraBody": dict(self.mcfg.get("extra_body") or {}),
            "responseFormat": response_format,
            "responseFormatSha256": response_format_sha256,
            "providerMaxRetries": max(0, min(4, int(
                runtime.get("pi_provider_max_retries", 4)))),
            "provider429BackoffBaseMs": max(0, min(60_000, int(float(
                os.environ.get("RSI_PROVIDER_429_BACKOFF_BASE_SECONDS", "0")) * 1000))),
        }
        if self.fake_response is not None:
            payload["fakeResponse"] = str(self.fake_response)
        _apply_request_cooldown(float(runtime.get("pi_request_cooldown_seconds", 0)))
        # Explicit fake responses are fully offline and must remain usable in
        # unit/integration tests without an experiment execution plan.
        ledger_operation_id = (None if self.fake_response is not None else
                               start_operation(None, "pi", self.model_id))
        if ledger_operation_id:
            payload["ledgerOperationId"] = ledger_operation_id
        try:
            if (ledger_operation_id
                    and os.environ.get("RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL", "")
                    .strip().lower() in {"1", "true", "yes"}):
                runtime = self.cfg.get("runtime") or {}
                provider_retries = max(0, min(4, int(
                    runtime.get("pi_provider_max_retries", 4))))
                bridge_attempts = max(1, min(3, int(
                    runtime.get("pi_timeout_max_attempts", 2))))
                reserve_physical_attempts(
                    count=(provider_retries + 1) * bridge_attempts,
                    reason=("Pi bridge reserves provider retries for each possible "
                            "subprocess timeout attempt"),
                    reservation_key="pi-provider-attempts-" + ledger_operation_id,
                    operation_id=ledger_operation_id,
                )
            proc, bridge_attempts = _invoke_bridge(
                payload, timeout_seconds=float(runtime.get("pi_timeout_seconds", 180)),
                max_attempts=int(runtime.get("pi_timeout_max_attempts", 2)))
        except Exception as exc:
            record_operation_end(ledger_operation_id, status="failed",
                                 error_type=type(exc).__name__)
            raise
        if proc.returncode != 0:
            record_operation_end(ledger_operation_id, status="failed",
                                 error_type="ProviderError")
            details = {}
            try:
                candidate = json.loads(proc.stdout)
                if isinstance(candidate, dict):
                    details = candidate
            except (ValueError, TypeError):
                pass
            error_detail = details.get("error") or {}
            message = str(error_detail.get("message") or proc.stderr[-1000:])
            attempts = details.get("providerAttempts")
            if not isinstance(attempts, list):
                attempts = []
            statuses = [str(item.get("status")) for item in attempts
                        if isinstance(item, dict) and item.get("status") is not None]
            retry_after = [str(item.get("headers", {}).get("retry-after"))
                           for item in attempts if isinstance(item, dict)
                           and isinstance(item.get("headers"), dict)
                           and item.get("headers", {}).get("retry-after") is not None]
            provider_errors = [item.get("providerError") for item in attempts
                               if isinstance(item, dict)
                               and isinstance(item.get("providerError"), dict)]
            summary = (f"; provider_http_attempts={len(attempts)}"
                       f"; statuses={','.join(statuses) or 'unknown'}"
                       f"; retry_after={','.join(retry_after) or 'none'}")
            if provider_errors:
                latest = provider_errors[-1]
                error_parts = [str(latest[key]) for key in ("code", "type", "message")
                               if latest.get(key)]
                if error_parts:
                    summary += "; provider_error=" + " | ".join(error_parts)[:600]
            error = ProviderError(f"Pi bridge failed: {message}{summary}")
            error.provider_attempts = attempts
            error.provider_error = provider_errors[-1] if provider_errors else None
            error.provider_http_attempts = len(attempts)
            error.provider_statuses = [
                int(item["status"]) for item in attempts
                if isinstance(item, dict)
                and isinstance(item.get("status"), int)
            ]
            error.provider_session_path = str(details.get("sessionPath") or "")
            error.provider_session_id = str(details.get("sessionId") or "")
            raise error
        try:
            result = json.loads(proc.stdout)
            content = str(result["assistantText"])
            usage = result.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            usage_available = (
                usage.get("input") is not None
                and usage.get("output") is not None)
            cache_read_tokens = int(usage.get("cacheRead") or 0)
            cache_write_tokens = int(usage.get("cacheWrite") or 0)
            p_tokens = int(usage.get("input") or 0) + cache_read_tokens + cache_write_tokens
            c_tokens = int(usage.get("output") or 0)
            session_path = Path(result["sessionPath"]) if result.get("sessionPath") else None
            if ephemeral:
                if result.get("ephemeral") is not True or session_path is not None:
                    raise ValueError("gate/final Pi task persisted a session")
            elif session_path is None or not session_path.is_file():
                raise ValueError("Pi JSONL session missing")
        except (ValueError, KeyError, TypeError) as exc:
            record_operation_end(ledger_operation_id, status="failed",
                                 error_type=type(exc).__name__)
            raise ProviderError(f"invalid Pi bridge output: {exc}") from exc
        try:
            if ephemeral:
                user_messages = list(result.get("userMessages") or [])
                user_ids = list(result.get("userEntryIds") or [])
                if len(user_messages) != len(user_ids):
                    raise ValueError("ephemeral Pi user message manifest mismatch")
                entries = list(result.get("messageEntries") or [])
                entry_ids = [item.get("entryId") for item in entries
                             if isinstance(item, dict)]
                assistant_ids = [item.get("entryId") for item in entries
                                 if isinstance(item, dict)
                                 and item.get("role") == "assistant"]
                if (len(entries) != len(entry_ids) or not entries
                        or len(set(entry_ids)) != len(entry_ids)
                        or any(not isinstance(entry_id, str) or not entry_id
                               for entry_id in entry_ids)
                        or [item.get("entryId") for item in entries
                            if item.get("role") == "user"] != user_ids
                        or not assistant_ids
                        or assistant_ids[-1] != result.get("assistantEntryId")
                        or len(entries) != result.get("messageCount")):
                    raise PiTraceError("ephemeral Pi message manifest mismatch")
                user_text_by_id = dict(zip(user_ids, user_messages))
                ephemeral_messages = [
                    {"entry_id": item["entryId"], "role": item["role"],
                     "text": (user_text_by_id[item["entryId"]]
                              if item["role"] == "user" else
                              content if item["entryId"] == assistant_ids[-1]
                              else ""),
                     "usage": usage if item["entryId"] == assistant_ids[-1]
                     else None}
                    for item in entries]
                session = {
                    "session_id": result.get("sessionId"),
                    "message_count": len(ephemeral_messages),
                    "task_manifest": {"entry_id": "",
                                      "data": result.get("taskManifest")},
                    "messages": ephemeral_messages,
                    "results": [{"operationId": result.get("operationId"),
                                 "status": "completed"}],
                    "usages": [],
                }
                if not user_ids or not result.get("assistantEntryId"):
                    raise PiTraceError("ephemeral Pi message IDs missing")
            else:
                session = read_pi_session(session_path)
            if session["session_id"] != result.get("sessionId"):
                raise PiTraceError("Pi session ID mismatch")
            if session["message_count"] != result.get("messageCount"):
                raise PiTraceError("Pi message count mismatch")
            manifest = session["task_manifest"]["data"]
            if not isinstance(manifest, dict) or manifest.get("protocol") != PI_BRIDGE_PROTOCOL_VERSION:
                raise PiTraceError("Pi task protocol manifest missing")
            if manifest.get("taskId") != task_id \
                    or manifest.get("taskSplit") != task_split \
                    or manifest.get("releaseId") != registry.release_id:
                raise PiTraceError("Pi task or release identity mismatch")
            if manifest.get("providerTraceId") != provider_trace_id \
                    or result.get("providerTraceId") != provider_trace_id:
                raise PiTraceError("Pi provider trace ID mismatch")
            if manifest.get("systemPrompt") != effective_prompt \
                    or manifest.get("systemPromptSha256") != sha256_text(effective_prompt) \
                    or manifest.get("userPromptSha256") != sha256_text(payload["userPrompt"]):
                raise PiTraceError("Pi prompt manifest mismatch")
            if manifest.get("demoTaskIds") != demo_ids \
                    or manifest.get("candidateDemoTaskIds") != payload["candidateDemoTaskIds"] \
                    or manifest.get("demosBlockSha256") != payload["demosBlockSha256"]:
                raise PiTraceError("Pi task demo manifest mismatch")
            if manifest.get("selectedSkill") != selected_skill:
                raise PiTraceError("Pi selected Skill mismatch")
            if manifest.get("sources") != payload["sources"]:
                raise PiTraceError("Pi source manifest mismatch")
            if manifest.get("responseFormatSha256") != response_format_sha256:
                raise PiTraceError("Pi structured response format manifest mismatch")
            if manifest.get("skills") != [
                    {"name": s["name"], "description": s["description"],
                     "bodySha256": sha256_text(s["content"]),
                     "filePath": s["filePath"]} for s in skills]:
                raise PiTraceError("Pi Skill snapshot manifest mismatch")
            last_assistant = next((m for m in reversed(session["messages"])
                                   if m["role"] == "assistant"), None)
            if last_assistant is None or last_assistant["text"] != content:
                raise PiTraceError("Pi assistant text mismatch")
            operation = next((r for r in session["results"]
                              if isinstance(r, dict)
                              and r.get("operationId") == result.get("operationId")), None)
            if operation is None or operation.get("status") != "completed":
                raise PiTraceError("Pi completed result missing")
            _validate_task_prompt_in_session(
                session, user_prompt=payload["userPrompt"],
                selected_skill=selected_skill, skill_body=skill_body,
                skill_file_path=next(
                    (skill["filePath"] for skill in skills
                     if skill["name"] == selected_skill), ""))
            session_usage = last_assistant.get("usage") or {}
            if (int(session_usage.get("input") or 0)
                    + int(session_usage.get("cacheRead") or 0)
                    + int(session_usage.get("cacheWrite") or 0)) != p_tokens \
                    or int(session_usage.get("output") or 0) != c_tokens:
                raise PiTraceError("Pi session usage mismatch")
        except PiTraceError as exc:
            record_operation_end(ledger_operation_id, status="failed",
                                 error_type=type(exc).__name__)
            raise ProviderError(f"Pi trace contract failed: {exc}") from exc
        if structured_output == "json_schema":
            idx, cited, failed = parse_structured_answer(
                content, options, valid_doc_ids,
                citation_enabled=bool(modes.get("citation")),
                abstain_enabled=abstain_enabled)
        else:
            idx, cited, failed = HttpProvider._parse(
                self, content, options, valid_doc_ids, abstain_enabled=abstain_enabled)
        mode = ("pi-faux:" if self.fake_response is not None else "pi:") \
            + str(result.get("modelId") or self.model_id)
        if demos_block:
            mode += "+task-demos"
        if failed:
            mode += "+parse_fail"
        if idx < 0:
            mode += "+abstain"
        cost = _cost(p_tokens, c_tokens, self.model_id,
                     fake=self.fake_response is not None,
                     cache_read_tokens=cache_read_tokens,
                     cache_write_tokens=cache_write_tokens,
                     usage_available=usage_available,
                     pi_usage_cost=usage.get("cost"),
                     base_url=str(self.mcfg.get(
                         "base_url", "https://api.siliconflow.cn/v1")))
        if bridge_attempts > 1:
            # A timed-out request may have consumed tokens or been billed even
            # though it returned no usage. Never claim a complete dollar cost.
            cost["usd"] = None
            cost["cost_estimated"] = True
            cost["cost_basis"] += "; prior timed-out Pi attempt usage unknown"
        provider_attempts = result.get("providerAttempts")
        if not isinstance(provider_attempts, list):
            provider_attempts = []
        record_operation_end(ledger_operation_id, status="success")
        return ModelOutput(
            answer_index=idx,
            answer_text=options[idx] if 0 <= idx < len(options) else "",
            cited_doc_ids=cited,
            prompt_tokens=p_tokens, completion_tokens=c_tokens,
            latency_ms=int(result.get("latencyMs") or 0), mode=mode,
            raw=content[:2000],
            cost_override=cost,
            pi_session_path=str(session_path) if session_path else "",
            pi_operation_id=str(result.get("operationId") or ""),
            provider_http_attempts=len(provider_attempts),
            pi_trace={"session_id": session["session_id"],
                      "task_manifest_entry_id": session["task_manifest"]["entry_id"],
                      "user_entry_ids": [m["entry_id"] for m in session["messages"]
                                         if m["role"] == "user"],
                      "assistant_entry_id": last_assistant["entry_id"],
                      "provider_trace_id": provider_trace_id,
                      "provider_trace_id_source": (
                          "client_sent_to_provider" if self.fake_response is None
                          else "local_fake_not_sent"),
                      "provider_trace_id_confirmed_by_response": False,
                      "message_count": session["message_count"],
                      "tool_calls": session.get("tool_calls", []),
                      "tool_results": session.get("tool_results", []),
                      "usage_records": len(session["usages"]),
                      "bridge_attempts": bridge_attempts,
                      "bridge_timeouts": bridge_attempts - 1,
                      "provider_attempts": provider_attempts,
                      "provider_http_requests": len(provider_attempts),
                      "protocol": PI_BRIDGE_PROTOCOL_VERSION},
            parse_failed=failed,
        )
