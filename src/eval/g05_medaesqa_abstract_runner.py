"""One-shot, approved G05 v5 SiliconFlow runner with durable write-ahead state.

The runner has no resume/retry path. An interrupted or ambiguous call consumes
its slot and seals the role run; only a separately reviewed new protocol may
continue with a fresh run that excludes the consumed case.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import ctypes
from pathlib import Path
from typing import Any, Callable

from ..runtime import provider as provider_module
from ..util import sha256_obj
from . import g05_medaesqa_abstract_execution as execution

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY_ENV = "SILICONFLOW_API_KEY"
MAX_TOKENS = 512
TIMEOUT_SECONDS = 240
ALLOWED_ROLES = {"reviewer_a", "reviewer_b", "adjudicator"}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _flush_directory(path: Path, *, windows: bool | None = None, win32: Any | None = None) -> None:
    """Persist directory-entry updates and fail closed when the platform cannot flush."""
    use_windows = os.name == "nt" if windows is None else windows
    if use_windows:
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
        if handle in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_last_error(), "CreateFileW failed for durable G05 directory flush", str(path))
        try:
            if not flush_buffers(handle):
                raise OSError(ctypes.get_last_error(), "FlushFileBuffers failed for G05 directory", str(path))
        finally:
            close_handle(handle)
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def source_tree_fingerprint() -> str:
    """Fingerprint executable project source used by a separately reviewed approval."""
    files = {}
    for base in (ROOT / "src", ROOT / "scripts"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".yaml", ".yml"}:
                files[path.relative_to(ROOT).as_posix()] = _sha_bytes(path.read_bytes())
    return sha256_obj(files)


def _atomic_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if exclusive:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _flush_directory(path.parent)
        return _sha_bytes(raw)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _flush_directory(path.parent)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return _sha_bytes(raw)


def validate_dispatch_approval(execution_plan: dict[str, Any], role: str,
                               approval: dict[str, Any], request_count: int) -> None:
    if role not in ALLOWED_ROLES or not isinstance(approval, dict):
        raise ValueError("a separate role-specific dispatch approval is required")
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    required = {"schema", "execution_plan_sha256", "reference_plan_sha256", "role",
        "max_calls", "max_physical_attempts", "hard_cost_usd", "dispatch_authorized",
        "root_review_complete", "source_tree_sha256"}
    if (set(body) != required
            or approval.get("schema") != "g05-medaesqa-v5-reference-dispatch-approval-v1"
            or approval.get("approval_sha256") != sha256_obj(body)
            or approval.get("execution_plan_sha256") != execution_plan["execution_plan_sha256"]
            or approval.get("reference_plan_sha256") != execution_plan["reference_plan_sha256"]
            or approval.get("role") != role
            or approval.get("dispatch_authorized") is not True
            or approval.get("hard_cost_usd") != "0.00"
            or approval.get("root_review_complete") is not True
            or approval.get("source_tree_sha256") != source_tree_fingerprint()
            or approval.get("max_calls") != request_count
            or approval.get("max_physical_attempts") != request_count):
        raise ValueError("G05 dispatch approval does not exactly authorize this role run")


class _WriteAheadLedger:
    def __init__(self, path: Path, *, execution_plan_sha256: str,
                 reference_plan_sha256: str, role: str, approval_sha256: str):
        self.path = path
        self.value = {"schema": "g05-v5-reference-write-ahead-ledger-v1",
            "execution_plan_sha256": execution_plan_sha256,
            "reference_plan_sha256": reference_plan_sha256, "role": role,
            "approval_sha256": approval_sha256, "status": "started",
            "physical_attempts": 0, "retries": 0, "request_intents": []}
        self._flush(exclusive=True)

    def _flush(self, *, exclusive: bool = False) -> None:
        body = {key: value for key, value in self.value.items() if key != "ledger_sha256"}
        self.value["ledger_sha256"] = sha256_obj(body)
        _atomic_json(self.path, self.value, exclusive=exclusive)

    def before(self, request: dict[str, Any]) -> dict[str, Any]:
        item = {"ordinal": len(self.value["request_intents"]) + 1,
            "case_id": request["case_id"], "role": request["role"],
            "model_id": request["model_id"], "call_sha256": request["call_sha256"],
            "prompt_sha256": request["prompt_sha256"], "state": "in_flight"}
        self.value["physical_attempts"] += 1
        self.value["request_intents"].append(item)
        self.value["status"] = "request_in_flight"
        self._flush()
        return item

    def after(self, item: dict[str, Any], *, state: str, receipt_sha256: str | None = None,
              error_type: str | None = None) -> None:
        item["state"] = state
        if receipt_sha256:
            item["receipt_sha256"] = receipt_sha256
        if error_type:
            item["error_type"] = error_type
        self.value["status"] = "failed_closed" if state != "succeeded" else "request_terminal"
        self._flush()


def _request_body(request: dict[str, Any]) -> dict[str, Any]:
    return {"model": request["model_id"],
        "messages": [{"role": "system", "content": request["system"]},
                     {"role": "user", "content": request["user"]}],
        "temperature": 0.0, "max_tokens": MAX_TOKENS,
        "response_format": request["response_format"], "enable_thinking": False}


def _http_transport(body: dict[str, Any], *, api_key: str, timeout: int):
    import requests
    return provider_module._endpoint_post(requests, f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body, timeout=timeout)


def _read_provider_response(response: Any, *, model_id: str) -> tuple[str, dict[str, int], str]:
    status = int(getattr(response, "status_code", 200))
    if status != 200:
        raise RuntimeError(f"provider_http_{status}")
    payload = response.json() if callable(getattr(response, "json", None)) else response
    if not isinstance(payload, dict) or payload.get("model") != model_id:
        raise ValueError("provider model echo mismatch")
    request_id = payload.get("id")
    choices = payload.get("choices")
    if (not isinstance(request_id, str) or not request_id.strip()
            or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict)):
        raise ValueError("provider receipt identity/choice missing")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    finish_reason = choices[0].get("finish_reason")
    if not isinstance(content, str) or finish_reason != "stop":
        raise ValueError("provider output is absent or truncated/unfinished")
    usage = payload.get("usage")
    if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
            or usage["prompt_tokens"] < 0 or not 0 <= usage["completion_tokens"] <= MAX_TOKENS):
        raise ValueError("provider usage is missing or invalid")
    return content, {"prompt_tokens": usage["prompt_tokens"],
                     "completion_tokens": usage["completion_tokens"]}, request_id


def _role_requests(packet: dict[str, Any], execution_plan: dict[str, Any],
                   reference_plan: dict[str, Any], role: str,
                   reviewer_receipts: dict[str, dict[str, Any]] | None) -> list[dict[str, Any]]:
    if role in {"reviewer_a", "reviewer_b"}:
        return execution_plan["reviewer_requests"][role]
    if not isinstance(reviewer_receipts, dict) or set(reviewer_receipts) != {"reviewer_a", "reviewer_b"}:
        raise ValueError("adjudicator requires both complete reviewer receipt bundles")
    by_case = {case["case_id"]: case for case in packet["cases"]}
    parsed = {}
    for slot in ("reviewer_a", "reviewer_b"):
        planned = {row["case_id"]: row for row in execution_plan["reviewer_requests"][slot]}
        if set(reviewer_receipts[slot]) != set(by_case):
            raise ValueError("adjudicator input reviewer receipts are incomplete")
        parsed[slot] = {case_id: execution._validate_receipt(receipt, planned[case_id])
                        for case_id, receipt in reviewer_receipts[slot].items()}
    requests = []
    for case_id, case in by_case.items():
        a, b = parsed["reviewer_a"][case_id], parsed["reviewer_b"][case_id]
        if a["verdict"] == b["verdict"]:
            continue
        a["raw_response_sha256"] = reviewer_receipts["reviewer_a"][case_id]["raw_response_sha256"]
        b["raw_response_sha256"] = reviewer_receipts["reviewer_b"][case_id]["raw_response_sha256"]
        requests.append(execution.build_adjudication_request(case, a, b,
            execution_plan, reference_plan))
    return requests


def dispatch_role(*, packet: dict[str, Any], reviewer_packets: dict[str, dict[str, Any]],
                  reference_plan: dict[str, Any], execution_plan: dict[str, Any],
                  zero_price_evidence: dict[str, Any], zero_price_evidence_sha256: str,
                  role: str, approval: dict[str, Any], run_dir: Path,
                  transport: Callable[..., Any] | None = None, api_key: str | None = None,
                  reviewer_receipts: dict[str, dict[str, Any]] | None = None,
                  role_registry_dir: Path | None = None, test_only: bool = False,
                  sleep_fn: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Execute exactly one approved role once; never retry or resume."""
    execution.validate_execution_plan(packet, reviewer_packets, reference_plan,
        zero_price_evidence=zero_price_evidence,
        zero_price_evidence_sha256=zero_price_evidence_sha256,
        execution_plan=execution_plan)
    if role not in ALLOWED_ROLES:
        raise ValueError("unknown G05 role")
    requests = _role_requests(packet, execution_plan, reference_plan, role, reviewer_receipts)
    if not requests:
        return {"schema": "g05-v5-role-run-v1", "status": "no_disagreement_calls",
            "role": role, "physical_attempts": 0, "model_calls": 0}
    validate_dispatch_approval(execution_plan, role, approval, len(requests))
    model = execution_plan["role_models"][role]
    current_price = provider_module.published_price_for_endpoint(model["model_id"], BASE_URL)
    if current_price is None or not current_price or any(float(value) != 0.0 for value in current_price.values()):
        raise RuntimeError("dispatch blocked: current endpoint tariff is not exactly zero")
    if transport is not None and (not test_only or not os.environ.get("RSI_HOME")):
        raise ValueError("injected transport is test-only and requires isolated RSI_HOME")
    if api_key is None:
        api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"missing {API_KEY_ENV}; provide through process environment only")
    run_dir = Path(run_dir).resolve()
    try:
        run_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("role run must be stored under project .workbuddy") from exc
    if run_dir.exists():
        raise FileExistsError("G05 role run already exists; replay/resume is not supported")
    registry = Path(role_registry_dir or (ROOT / ".workbuddy" /
        "g05-medaesqa-v5-role-dispatch-once" / execution_plan["execution_plan_sha256"])).resolve()
    try:
        registry.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("role registry must stay under project .workbuddy") from exc
    registry.mkdir(parents=True, exist_ok=True)
    reservation = {"schema": "g05-v5-role-dispatch-reservation-v1",
        "execution_plan_sha256": execution_plan["execution_plan_sha256"],
        "role": role, "approval_sha256": approval["approval_sha256"],
        "source_tree_sha256": source_tree_fingerprint(),
        "run_dir": str(run_dir), "status": "reserved_before_first_call"}
    try:
        _atomic_json(registry / f"{role}.json", reservation, exclusive=True)
    except FileExistsError as exc:
        raise FileExistsError("this role is already consumed for the frozen G05 plan") from exc
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "execution-plan.json", execution_plan, exclusive=True)
    _atomic_json(run_dir / "approval.json", approval, exclusive=True)
    ledger = _WriteAheadLedger(run_dir / "write-ahead-ledger.json",
        execution_plan_sha256=execution_plan["execution_plan_sha256"],
        reference_plan_sha256=reference_plan["plan_sha256"], role=role,
        approval_sha256=approval["approval_sha256"])
    receipts: dict[str, dict[str, Any]] = {}
    for index, request in enumerate(requests):
        intent = ledger.before(request)
        raw_response = None
        request_id = None
        usage = None
        try:
            body = _request_body(request)
            sender = transport or _http_transport
            response = sender(body, api_key=api_key, timeout=TIMEOUT_SECONDS)
            raw, usage, request_id = _read_provider_response(response, model_id=model["model_id"])
            raw_response = raw
            execution._strict_object(raw)
            receipt = {"case_id": request["case_id"],
                "plan_sha256": request["plan_sha256"],
                "call_sha256": request["call_sha256"],
                "prompt_sha256": request["prompt_sha256"],
                "provider": model["provider"], "model_id": model["model_id"],
                "model_family": model["model_family"], "provider_request_id": request_id,
                "http_status": 200, "response_model": model["model_id"],
                "finish_reason": "stop",
                "usage": usage, "cost_usd": "0.00", "raw_response": raw,
                "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
            execution._validate_receipt(receipt, request)
            receipt_sha = _atomic_json(run_dir / f"receipt-{index:03d}.json", receipt, exclusive=True)
            receipts[request["case_id"]] = receipt
            ledger.after(intent, state="succeeded", receipt_sha256=receipt_sha)
        except Exception as exc:
            ledger.after(intent, state="attempted_unknown", error_type=type(exc).__name__)
            ledger.value["status"] = "failed_closed"
            ledger._flush()
            _atomic_json(run_dir / "partial-receipts.json", {
                "schema": "g05-v5-partial-receipts-v1", "role": role,
                "execution_plan_sha256": execution_plan["execution_plan_sha256"],
                "completed": receipts, "failed_case_id": request["case_id"],
                "failure_type": type(exc).__name__, "retry_allowed": False,
                "failed_response": ({"provider_request_id": request_id,
                    "usage": usage, "raw_response": raw_response,
                    "raw_response_sha256": (hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
                        if isinstance(raw_response, str) else None)}
                    if isinstance(raw_response, str) else None)}, exclusive=True)
            raise RuntimeError("G05 call failed or is ambiguous; this role run is sealed without retry") from exc
        if index + 1 < len(requests):
            sleep_fn(1.0)
    ledger.value["status"] = "complete"
    ledger._flush()
    receipt_bundle = {"schema": "g05-v5-role-receipts-v1", "role": role,
        "execution_plan_sha256": execution_plan["execution_plan_sha256"],
        "reference_plan_sha256": reference_plan["plan_sha256"], "receipts": receipts}
    bundle_sha = _atomic_json(run_dir / "role-receipts.json", receipt_bundle, exclusive=True)
    return {"schema": "g05-v5-role-run-v1", "status": "complete",
        "role": role, "physical_attempts": len(requests), "retries": 0,
        "role_receipts_sha256": bundle_sha, "ledger_sha256": ledger.value["ledger_sha256"],
        "run_dir": str(run_dir), "model_calls": len(requests), "dispatch_authorized": True}


__all__ = ["source_tree_fingerprint", "validate_dispatch_approval", "dispatch_role"]
