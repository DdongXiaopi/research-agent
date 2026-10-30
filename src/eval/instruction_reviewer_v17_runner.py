"""One-shot, role-approved SiliconFlow transport for the G06 v17 plan."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from ..runtime import provider as provider_module
from ..util import sha256_obj
from . import instruction_reviewer_v17_execution as execution

ROOT = Path(__file__).resolve().parents[2]
API_KEY_ENV = "SILICONFLOW_API_KEY"
ALLOWED_ROLES = {"reviewer_a", "reviewer_b"}


def _flush_directory(path: Path, *, windows: bool | None = None, win32: Any | None = None) -> None:
    """Durably flush a directory entry on the current platform or fail closed."""
    use_windows = os.name == "nt" if windows is None else windows
    if use_windows:
        import ctypes

        api = win32 or ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = api.CreateFileW
        flush_buffers = api.FlushFileBuffers
        close_handle = api.CloseHandle
        if win32 is None:
            create_file.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
            create_file.restype = ctypes.c_void_p
            flush_buffers.argtypes = [ctypes.c_void_p]
            flush_buffers.restype = ctypes.c_int
            close_handle.argtypes = [ctypes.c_void_p]
            close_handle.restype = ctypes.c_int
        handle = create_file(str(path), 0x80000000 | 0x40000000,
                             0x00000001 | 0x00000002 | 0x00000004,
                             None, 3, 0x02000000, None)
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            code = ctypes.get_last_error()
            raise OSError(code, "CreateFileW failed while opening directory for durable flush", str(path))
        try:
            if not flush_buffers(handle):
                code = ctypes.get_last_error()
                raise OSError(code, "FlushFileBuffers failed for directory", str(path))
        finally:
            close_handle(handle)
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    if exclusive:
        with path.open("xb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        _flush_directory(path.parent)
        return digest
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
        _flush_directory(path.parent)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
    return digest


def source_tree_fingerprint() -> str:
    files = {}
    for base in (ROOT / "src", ROOT / "scripts"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".yaml", ".yml"}:
                files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha256_obj(files)


def validate_approval(plan: dict[str, Any], role: str, approval: dict[str, Any], count: int) -> None:
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    required = {"schema", "execution_plan_sha256", "reference_plan_sha256", "role",
        "max_calls", "max_physical_attempts", "hard_cost_usd", "dispatch_authorized",
        "root_review_complete", "source_tree_sha256"}
    if (role not in ALLOWED_ROLES or set(body) != required
            or approval.get("schema") != "g06-v17-role-dispatch-approval-v1"
            or approval.get("approval_sha256") != sha256_obj(body)
            or approval.get("execution_plan_sha256") != plan["execution_plan_sha256"]
            or approval.get("reference_plan_sha256") != plan["reference_plan_sha256"]
            or approval.get("role") != role or approval.get("max_calls") != count
            or approval.get("max_physical_attempts") != count
            or approval.get("hard_cost_usd") != "0.00"
            or approval.get("dispatch_authorized") is not True
            or approval.get("root_review_complete") is not True
            or approval.get("source_tree_sha256") != source_tree_fingerprint()):
        raise ValueError("a separate exact role-specific G06 v17 approval is required")


def _provider_call(body: dict[str, Any], *, api_key: str, timeout: int) -> Any:
    import requests
    return provider_module._endpoint_post(requests, "https://api.siliconflow.cn/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body, timeout=timeout)


def _read_response(response: Any, model_id: str) -> tuple[str, dict[str, int], str]:
    status = int(getattr(response, "status_code", 200))
    if status != 200: raise RuntimeError(f"provider_http_{status}")
    payload = response.json() if callable(getattr(response, "json", None)) else response
    if not isinstance(payload, dict) or payload.get("model") != model_id:
        raise ValueError("provider model echo mismatch")
    choices = payload.get("choices")
    if (not isinstance(payload.get("id"), str) or not payload["id"].strip()
            or not isinstance(choices, list) or len(choices) != 1):
        raise ValueError("provider receipt identity/choice missing")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    usage = payload.get("usage")
    if (not isinstance(content, str) or choice.get("finish_reason") != "stop"
            or not isinstance(usage, dict)
            or type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
            or min(usage["prompt_tokens"], usage["completion_tokens"]) < 0
            or usage["completion_tokens"] > execution.MAX_TOKENS):
        raise ValueError("provider response is incomplete or usage invalid")
    return content, {"prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"]}, payload["id"]


def dispatch_role(*, packet: dict[str, Any], reference_plan: dict[str, Any],
                  execution_plan: dict[str, Any], zero_price_evidence: dict[str, Any],
                  role: str, approval: dict[str, Any], run_dir: Path,
                  transport: Callable[..., Any] | None = None, api_key: str | None = None,
                  test_only: bool = False, role_registry_dir: Path | None = None,
                  sleep_fn: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    execution.validate_execution_plan(packet, reference_plan, zero_price_evidence, execution_plan)
    if role not in ALLOWED_ROLES: raise ValueError("unknown G06 v17 role")
    requests = execution_plan["requests"][role]
    validate_approval(execution_plan, role, approval, len(requests))
    model_id = execution_plan["reviewers"][role]["model_id"]
    rates = provider_module.published_price_for_endpoint(model_id, execution.BASE_URL)
    if not rates or any(float(price) != 0.0 for price in rates.values()):
        raise RuntimeError("dispatch blocked: current endpoint tariff is not exactly zero")
    if transport is not None and (not test_only or not os.environ.get("RSI_HOME")):
        raise ValueError("injected transport is test-only and requires isolated RSI_HOME")
    api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
    if not api_key: raise RuntimeError(f"missing {API_KEY_ENV}; provide through process environment only")
    run_dir = Path(run_dir).resolve()
    try: run_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc: raise ValueError("G06 role run must be under .workbuddy") from exc
    if run_dir.exists(): raise FileExistsError("role run exists; replay/resume is unsupported")
    registry = Path(role_registry_dir or (ROOT / ".workbuddy" / "g06-v17-role-dispatch-once" /
        execution_plan["execution_plan_sha256"])).resolve()
    try: registry.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc: raise ValueError("role registry must be under .workbuddy") from exc
    registry.mkdir(parents=True, exist_ok=True)
    reservation = {"schema": "g06-v17-role-reservation-v1", "role": role,
        "execution_plan_sha256": execution_plan["execution_plan_sha256"],
        "approval_sha256": approval["approval_sha256"], "status": "reserved_before_first_call"}
    _atomic(registry / f"{role}.json", reservation, exclusive=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic(run_dir / "execution-plan.json", execution_plan, exclusive=True)
    _atomic(run_dir / "approval.json", approval, exclusive=True)
    ledger = {"schema": "g06-v17-write-ahead-ledger-v1", "role": role,
        "execution_plan_sha256": execution_plan["execution_plan_sha256"],
        "approval_sha256": approval["approval_sha256"], "status": "started",
        "physical_attempts": 0, "retries": 0, "intents": []}
    _atomic(run_dir / "write-ahead-ledger.json", ledger, exclusive=True)
    receipts = {}
    for index, request in enumerate(requests):
        intent = {"ordinal": index + 1, "case_id": request["case_id"],
            "call_sha256": request["call_sha256"], "state": "in_flight"}
        ledger["physical_attempts"] += 1; ledger["intents"].append(intent); ledger["status"] = "request_in_flight"
        _atomic(run_dir / "write-ahead-ledger.json", ledger)
        try:
            body = {"model": model_id, "messages": [{"role": "system", "content": request["system"]},
                {"role": "user", "content": request["user"]}], "temperature": 0.0,
                "max_tokens": request["max_tokens"], "response_format": request["response_format"],
                "enable_thinking": False}
            response = (transport or _provider_call)(body, api_key=api_key, timeout=240)
            content, usage, provider_id = _read_response(response, model_id)
            rec = {"case_id": request["case_id"], "packet_sha256": packet["packet_sha256"],
                "reference_plan_sha256": reference_plan["plan_sha256"], "call_sha256": request["call_sha256"],
                "role": role, "provider": request["model"]["provider"], "model_id": model_id,
                "model_family": request["model"]["model_family"], "provider_request_id": provider_id,
                "http_status": 200, "response_model": model_id, "finish_reason": "stop",
                "usage": usage, "cost_usd": "0.00", "raw_response": content,
                "raw_response_sha256": hashlib.sha256(content.encode()).hexdigest()}
            execution._receipt(rec, request)
            digest = _atomic(run_dir / f"receipt-{index:03d}.json", rec, exclusive=True)
            receipts[request["case_id"]] = rec
            intent.update({"state": "succeeded", "receipt_file_sha256": digest})
            ledger["status"] = "request_terminal"
            _atomic(run_dir / "write-ahead-ledger.json", ledger)
        except Exception as exc:
            intent.update({"state": "attempted_unknown", "error_type": type(exc).__name__})
            ledger["status"] = "failed_closed"; _atomic(run_dir / "write-ahead-ledger.json", ledger)
            _atomic(run_dir / "partial-receipts.json", {"schema": "g06-v17-partial-receipts-v1",
                "role": role, "completed": receipts, "failed_case_id": request["case_id"],
                "failure_type": type(exc).__name__, "retry_allowed": False}, exclusive=True)
            raise RuntimeError("G06 request failed or is ambiguous; role sealed without retry") from exc
        if index + 1 < len(requests): sleep_fn(1.0)
    ledger["status"] = "complete"; _atomic(run_dir / "write-ahead-ledger.json", ledger)
    bundle = {"schema": "g06-v17-role-receipts-v1", "role": role,
        "packet_sha256": packet["packet_sha256"], "reference_plan_sha256": reference_plan["plan_sha256"],
        "execution_plan_sha256": execution_plan["execution_plan_sha256"], "receipts": receipts}
    receipt_sha = _atomic(run_dir / "role-receipts.json", bundle, exclusive=True)
    return {"status": "complete", "role": role, "model_calls": len(receipts),
        "retries": 0, "role_receipts_sha256": receipt_sha,
        "run_dir": str(run_dir), "dispatch_authorized": True}


__all__ = ["source_tree_fingerprint", "validate_approval", "dispatch_role"]
