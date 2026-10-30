"""One-shot runner for the selected-answer semantic calibration contract.

The runner is intentionally smaller than the general AI-job dispatcher: one
frozen plan, one physical attempt per case, exact prompt/response receipts, and
an exclusive ledger. It is diagnostic evidence until a separate runtime
attestation binds the scorer to production.
"""
from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from ..runtime import provider as provider_module
from ..util import sha256_file, write_json
from .grounding_selected_answer import selected_answer_support_messages
from .grounding_selected_answer_calibration import (
    build_selected_answer_plan, score_selected_answer_calibration,
)

ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"
SCHEMA = "g05-selected-answer-calibration-run-v1"
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "selected_answer_support",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "verdict": {"type": "string", "enum": ["supported", "contradicted", "insufficient"]},
                "basis": {"type": "string", "enum": [
                    "direct_finding_supports", "direct_finding_opposes",
                    "evidence_gap_or_uncertainty", "topic_only_or_scope_mismatch"]},
                "sentence_ids": {"type": "array", "items": {"type": "integer"},
                                 "minItems": 1, "maxItems": 8},
            },
            "required": ["verdict", "basis", "sentence_ids"],
        },
    },
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _persist_ledger(path: Path, value: dict[str, Any]) -> None:
    """Keep the last durable receipt if a process stops during a rewrite."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _nonnegative_decimal(value: Any, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid selected-answer {name}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"invalid selected-answer {name}")
    return number


def _cost_usd(prompt_tokens: int, completion_tokens: int,
              rates: dict[str, Any]) -> Decimal:
    # Aggregate usage may omit the cache split. Charge every input token at
    # the highest applicable rate; never treat missing cache usage as free.
    input_rate = max(_nonnegative_decimal(rates.get(key, rates["prompt"]), "tariff")
                     for key in ("prompt", "cache_read", "cache_write"))
    output_rate = _nonnegative_decimal(rates["completion"], "tariff")
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


def _post(body: dict[str, Any], *, api_key: str, timeout: int,
          endpoint: str = ENDPOINT) -> Any:
    import requests
    session = requests.Session()
    session.trust_env = False
    try:
        return session.post(endpoint, headers={"Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"}, json=body, timeout=timeout)
    finally:
        session.close()


def _protocol(plan: dict[str, Any]) -> dict[str, Any]:
    value = plan.get("model_protocol")
    if value is None:
        return {"provider": "siliconflow", "base_url": ENDPOINT.rsplit("/chat/completions", 1)[0],
                "api_key_env": DEFAULT_API_KEY_ENV, "price_policy": "zero_price_only",
                "paid_fallback": False, "response_format": RESPONSE_FORMAT,
                "max_cost_usd": 0.0}
    if not isinstance(value, dict) or not isinstance(value.get("base_url"), str):
        raise ValueError("selected-answer model_protocol is missing a valid base_url")
    if not isinstance(value.get("api_key_env", DEFAULT_API_KEY_ENV), str):
        raise ValueError("selected-answer model_protocol api_key_env is invalid")
    return value


def _request_options(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return provider-specific request controls without allowing body takeover.

    DeepSeek's current chat API uses ``thinking.type=disabled``; the older
    ``enable_thinking`` flag is not a portable OpenAI-compatible contract.
    Keep these controls in the frozen model protocol and allow only the small
    set of provider options that cannot alter the task or accounting fields.
    """
    value = protocol.get("request_options", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("selected-answer model_protocol request_options is invalid")
    allowed = {"thinking", "reasoning_effort", "enable_thinking"}
    if set(value) - allowed:
        raise ValueError("selected-answer model_protocol has unsupported request options")
    return dict(value)


def execute(packet: dict[str, Any], reference: dict[str, Any], plan: dict[str, Any],
            *, ledger_path: Path, output_path: Path, api_key: str | None = None,
            transport: Callable[..., Any] | None = None, timeout: int = 120,
            approval: dict[str, Any] | None = None, test_only: bool = False) -> dict[str, Any]:
    """Execute each exact plan request once and score the recorded receipts."""
    if plan.get("request_count") != len(packet.get("cases") or []):
        raise ValueError("calibration plan/request count mismatch")
    protocol = _protocol(plan)
    request_options = _request_options(protocol)
    endpoint = protocol["base_url"].rstrip("/") + "/chat/completions"
    api_key_env = protocol.get("api_key_env", DEFAULT_API_KEY_ENV)
    paid = protocol.get("price_policy") != "zero_price_only"
    expected = build_selected_answer_plan(packet, reference,
        model_id=plan["model_id"], model_protocol_sha256=plan["model_protocol_sha256"],
        created_at=plan["created_at"], max_doc_chars=plan["max_doc_chars"],
        max_tokens=plan["max_tokens"], model_protocol=plan.get("model_protocol"))
    if expected != plan:
        raise ValueError("calibration plan is stale or altered")
    if not isinstance(approval, dict):
        raise ValueError("selected-answer calibration requires a separate dispatch approval")
    approval_body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    if (approval.get("schema") != "g05-selected-answer-calibration-dispatch-approval-v1"
            or approval.get("plan_sha256") != plan["plan_sha256"]
            or approval.get("request_count") != plan["request_count"]
            or approval.get("max_physical_attempts") != plan["max_physical_attempts"]
            or approval.get("hard_cost_usd") != f"{float(protocol.get('max_cost_usd', 0.0)):.2f}"
            or approval.get("dispatch_authorized") is not True
            or approval.get("approval_sha256") != _sha(approval_body)):
        raise ValueError("selected-answer calibration approval is stale or malformed")
    if ledger_path.exists() or output_path.exists():
        raise FileExistsError("calibration run paths are write-once")
    if transport is not None and (not test_only or not os.environ.get("RSI_HOME")):
        raise ValueError("injected transport is test-only and requires isolated RSI_HOME")
    if test_only and transport is None:
        raise ValueError("test-only calibration requires an injected transport")
    if api_key is None:
        api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing {api_key_env}")
    price = provider_module.published_price_for_endpoint(plan["model_id"], protocol["base_url"])
    if price is None and test_only:
        price = {"prompt": 0.0, "completion": 0.0}
    if price is None:
        raise RuntimeError("selected-answer calibration requires a known endpoint tariff")
    if not paid and any(_nonnegative_decimal(value, "tariff") != 0 for value in price.values()):
        raise RuntimeError("selected-answer zero-price calibration requires zero tariff")
    max_cost = _nonnegative_decimal(protocol.get("max_cost_usd", 0.0), "max_cost_usd")
    # UTF-8 bytes plus framing is deliberately more conservative than the
    # common characters/4 estimate. It is a planning bound, not a provider
    # billing guarantee; actual usage and any excess are retained below.
    planned_upper = Decimal(0)
    for case in packet["cases"]:
        system, user = selected_answer_support_messages(case["question"], case["selected_answer"],
            case["evidence"], plan["max_doc_chars"])
        planned_upper += _cost_usd(len((system + user).encode("utf-8")) + 512,
                                  plan["max_tokens"], price)
    if planned_upper > max_cost:
        raise RuntimeError("selected-answer calibration planned cost exceeds approved budget")
    _exclusive(ledger_path, {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
        "status": "started", "physical_attempts": 0, "request_intents": [],
        "call_records": [], "tariff_usd_per_million": price,
        "planned_cost_upper_estimate_usd": str(planned_upper),
        "approved_max_cost_usd": str(max_cost), "cost_upper_estimate_usd": "0",
        "cost_is_provider_invoice": False, "prompt_tokens": 0, "completion_tokens": 0})
    cases = {case["case_id"]: case for case in packet["cases"]}
    requests_by_id = {row["case_id"]: row for row in plan["requests"]}
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    records = ledger["call_records"]
    running_cost = Decimal(0)
    for ordinal, request in enumerate(plan["requests"], start=1):
        case = cases[request["case_id"]]
        system, user = selected_answer_support_messages(case["question"], case["selected_answer"],
            case["evidence"], plan["max_doc_chars"])
        if hashlib.sha256((system + "\0" + user).encode()).hexdigest() != request["prompt_sha256"]:
            raise ValueError("request prompt hash mismatch")
        next_cost = _cost_usd(len((system + user).encode("utf-8")) + 512,
                              plan["max_tokens"], price)
        if running_cost + next_cost > max_cost:
            ledger["status"] = "failed_closed"
            ledger["error_type"] = "BudgetExceeded"
            _persist_ledger(ledger_path, ledger)
            raise RuntimeError("selected-answer calibration remaining budget is insufficient; no retry")
        intent = {"ordinal": ordinal, "case_id": request["case_id"],
                  "input_sha256": request["input_sha256"], "prompt_sha256": request["prompt_sha256"],
                  "state": "in_flight"}
        ledger["physical_attempts"] += 1
        ledger["request_intents"].append(intent)
        _persist_ledger(ledger_path, ledger)
        body = {"model": plan["model_id"], "messages": [{"role": "system", "content": system},
            {"role": "user", "content": user}], "temperature": 0.0,
            "max_tokens": plan["max_tokens"],
            "response_format": protocol.get("response_format", RESPONSE_FORMAT)}
        body.update(request_options)
        try:
            response = (transport(body, api_key=api_key, timeout=timeout)
                        if transport is not None
                        else _post(body, api_key=api_key, timeout=timeout, endpoint=endpoint))
            status = int(getattr(response, "status_code", 200))
            payload = response.json() if callable(getattr(response, "json", None)) else response
            if status != 200 or not isinstance(payload, dict):
                raise RuntimeError(f"provider_http_{status}")
            choices = payload.get("choices")
            message = choices[0].get("message") if isinstance(choices, list) and len(choices) == 1 else None
            raw = message.get("content") if isinstance(message, dict) else None
            usage = payload.get("usage")
            request_id = payload.get("id")
            if (not isinstance(raw, str) or not isinstance(request_id, str) or not request_id.strip()
                    or not isinstance(usage, dict)
                    or type(usage.get("prompt_tokens")) is not int
                    or type(usage.get("completion_tokens")) is not int
                    or usage["prompt_tokens"] < 0 or usage["completion_tokens"] < 0):
                raise ValueError("provider receipt is incomplete")
            record = {"case_id": request["case_id"], "plan_sha256": plan["plan_sha256"],
                "input_sha256": request["input_sha256"], "prompt_sha256": request["prompt_sha256"],
                "system": system, "user": user, "status": "succeeded", "raw_output": raw,
                "physical_attempts": 1,
                "receipt": {"plan_sha256": plan["plan_sha256"], "input_sha256": request["input_sha256"],
                    "prompt_sha256": request["prompt_sha256"], "model_protocol_sha256": plan["model_protocol_sha256"],
                    "physical_attempts": 1, "request_id": request_id,
                    # The scorer's receipt verifier hashes all JSON scalar values
                    # through its canonical serializer.  Keep the runner on the
                    # same protocol so a successful provider response is not
                    # rejected as an output/receipt mismatch.
                    "raw_output_sha256": _sha(raw), "usage": usage}}
            records.append(record)
            running_cost += _cost_usd(usage["prompt_tokens"], usage["completion_tokens"], price)
            ledger["prompt_tokens"] += usage["prompt_tokens"]
            ledger["completion_tokens"] += usage["completion_tokens"]
            ledger["cost_upper_estimate_usd"] = str(running_cost)
            intent["state"] = "succeeded"
            if running_cost > max_cost:
                # Preserve the paid response before stopping. This cannot
                # undo a provider's unexpected usage or justify another call.
                raise RuntimeError("provider usage exceeded approved cost budget")
        except Exception as exc:
            if intent["state"] != "succeeded":
                intent["state"] = "attempted_unknown"
            ledger["status"] = "failed_closed"
            ledger["error_type"] = type(exc).__name__
            _persist_ledger(ledger_path, ledger)
            raise RuntimeError("selected-answer calibration failed closed; no retry") from exc
        ledger["status"] = "request_terminal"
        _persist_ledger(ledger_path, ledger)
    ledger["status"] = "complete"
    _persist_ledger(ledger_path, ledger)
    report = score_selected_answer_calibration(packet, reference, plan, records)
    result = {"schema": SCHEMA, "status": "complete", "plan_sha256": plan["plan_sha256"],
              "ledger_sha256": sha256_file(ledger_path), "report": report,
              "model_calls": len(records), "network_calls": len(records),
              "cost_upper_estimate_usd": str(running_cost),
              "cost_is_provider_invoice": False,
              "calibration_pass": bool(report.get("calibration_pass")),
              "diagnostic_only": True}
    write_json(output_path, result)
    return result


__all__ = ["execute", "SCHEMA", "RESPONSE_FORMAT"]
