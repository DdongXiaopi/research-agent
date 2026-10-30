"""Append-only request ledger for bounded experiment runs.

The ledger stores metadata only. It never stores prompts, completions, URLs,
headers, or credentials. A start event is flushed before work begins so a
crash cannot make a dispatched request disappear from the logical budget.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_LOCK = threading.Lock()
_ACTIVE_OPERATION: ContextVar[str | None] = ContextVar("rsi_active_call_operation", default=None)
_ALLOWED_COMPONENTS = {
    "ai_evaluation",
    "open_research_answer", "skill_crystallization_proposer", "feedback_candidate_proposer",
    "task_answer", "skill_router", "mipro_proposer", "gepa_reflection",
    "grounding", "provider_http",
}
_COMPONENT = ContextVar("rsi_call_component", default="provider_http")


@contextmanager
def call_component(component: str):
    if component not in _ALLOWED_COMPONENTS:
        raise ValueError(f"unknown request-ledger component: {component}")
    token = _COMPONENT.set(component)
    try:
        yield
    finally:
        _COMPONENT.reset(token)


def set_active_operation(operation_id: str | None):
    return _ACTIVE_OPERATION.set(operation_id)


def reset_active_operation(token) -> None:
    _ACTIVE_OPERATION.reset(token)


def active_operation_id() -> str | None:
    return _ACTIVE_OPERATION.get()


def next_http_attempt_number(operation_id: str | None) -> int:
    path = _ledger_path()
    if path is None or operation_id is None or not path.exists():
        return 1
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if (item.get("event") == "http_attempt_start"
                    and item.get("operation_id") == operation_id):
                count += 1
    return count + 1


def _ledger_path() -> Path | None:
    value = os.environ.get("RSI_CALL_LEDGER_PATH", "").strip()
    return Path(value) if value else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _read_counts(path: Path) -> tuple[int, int]:
    logical = 0
    actual_by_operation: dict[str, int] = {}
    unreserved_actual = 0
    reserved_by_operation: dict[str, int] = {}
    unscoped_reservations = 0
    if not path.exists():
        return logical, 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if item.get("event") == "logical_start":
                logical += 1
            elif item.get("event") == "http_attempt_start":
                operation_id = str(item.get("operation_id") or "")
                if operation_id:
                    actual_by_operation[operation_id] = (
                        actual_by_operation.get(operation_id, 0) + 1)
                else:
                    unreserved_actual += 1
            elif item.get("event") == "physical_reservation":
                count = max(0, int(item.get("count", 0)))
                operation_id = str(item.get("operation_id") or "")
                if operation_id:
                    reserved_by_operation[operation_id] = (
                        reserved_by_operation.get(operation_id, 0) + count)
                else:
                    unscoped_reservations += count
    # An operation-level reservation is a ceiling which its observed attempts
    # consume. Count max(reserved, actual), not both; otherwise a correctly
    # reserved request would spend its budget twice.
    physical = (sum(max(reserved_by_operation.get(operation_id, 0), actual)
                    for operation_id, actual in actual_by_operation.items())
                + sum(count for operation_id, count in reserved_by_operation.items()
                      if operation_id not in actual_by_operation)
                + unreserved_actual + unscoped_reservations)
    return logical, physical


def _read_component_logical_count(path: Path, component: str) -> int:
    count = 0
    if not path.exists():
        return count
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if (item.get("event") == "logical_start"
                    and item.get("component") == component):
                count += 1
    return count


def _append(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
    with _LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def start_operation(component: str | None, provider: str, model_id: str) -> str | None:
    """Reserve one logical provider operation before making a network call."""
    path = _ledger_path()
    if path is None:
        raise RuntimeError(
            "real provider requests require a bounded request ledger")
    component = component or _COMPONENT.get()
    if component not in _ALLOWED_COMPONENTS:
        raise ValueError(f"unknown request-ledger component: {component}")
    logical_limit = int(os.environ.get("RSI_CALL_LEDGER_MAX_LOGICAL", "0") or 0)
    physical_limit = int(os.environ.get("RSI_CALL_LEDGER_MAX_PHYSICAL", "0") or 0)
    if logical_limit <= 0 or physical_limit <= 0:
        raise RuntimeError(
            "real provider requests require positive logical and physical request ceilings")
    plan_sha256 = os.environ.get("RSI_ACTIVE_EXECUTION_PLAN_SHA256", "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha256):
        raise RuntimeError(
            "real provider requests require a verified 64-character active execution-plan hash")
    component_limit = int(os.environ.get(
        "RSI_CALL_LEDGER_MAX_COMPONENT_" + component.upper(), "0") or 0)
    with _LOCK:
        logical, _ = _read_counts(path)
        if logical_limit > 0 and logical >= logical_limit:
            raise RuntimeError("run logical model-request budget exhausted before dispatch")
        if (component_limit > 0
                and _read_component_logical_count(path, component) >= component_limit):
            raise RuntimeError(
                f"{component} component model-request budget exhausted before dispatch")
        operation_id = uuid.uuid4().hex
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "logical_start", "operation_id": operation_id,
            "component": component, "provider": str(provider)[:80],
            "model_id": str(model_id)[:160], "plan_sha256": plan_sha256,
            "timestamp": _now(),
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return operation_id


def seed_legacy_usage(path: str | Path, *, logical_by_component: dict[str, int],
                      known_http_statuses: list[int],
                      unknown_physical_reservation: int) -> bool:
    """Backfill a pre-ledger run once, conservatively reserving unknown attempts."""
    target = Path(path)
    if target.exists():
        return False
    invalid = set(logical_by_component) - _ALLOWED_COMPONENTS
    if invalid or any(int(value) < 0 for value in logical_by_component.values()):
        raise ValueError("invalid legacy logical request counts")
    if unknown_physical_reservation < 0:
        raise ValueError("unknown physical reservation must be non-negative")
    events: list[dict[str, Any]] = []
    operations_by_component: dict[str, list[str]] = {}
    for component, count in sorted(logical_by_component.items()):
        operations_by_component[component] = []
        for _ in range(int(count)):
            operation_id = "legacy-" + uuid.uuid4().hex
            operations_by_component[component].append(operation_id)
            events.extend([
                {"event": "logical_start", "operation_id": operation_id,
                 "component": component, "provider": "legacy_backfill",
                 "model_id": "unknown", "timestamp": _now(),
                 "source": "legacy_backfill"},
                {"event": "logical_end", "operation_id": operation_id,
                 "status": "unknown", "error_type": "",
                 "timestamp": _now(), "source": "legacy_backfill"},
            ])
    task_operations = operations_by_component.get("task_answer", [])
    if known_http_statuses and not task_operations:
        raise ValueError("known task HTTP attempts require legacy task-answer operations")
    for index, status in enumerate(known_http_statuses):
        group = index // 3
        operation_id = task_operations[max(0, len(task_operations) - group - 1)]
        attempt = index % 3 + 1
        events.extend([
            {"event": "http_attempt_start", "operation_id": operation_id,
             "attempt": attempt, "timestamp": _now(),
             "source": "legacy_backfill"},
            {"event": "http_attempt_end", "operation_id": operation_id,
             "attempt": attempt, "status": int(status), "error_type": "",
             "timestamp": _now(), "source": "legacy_backfill"},
        ])
    if unknown_physical_reservation:
        events.append({"event": "physical_reservation",
                       "count": int(unknown_physical_reservation),
                       "reason": "legacy physical attempt count unavailable",
                       "timestamp": _now(), "source": "legacy_backfill"})
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if target.exists():
            return False
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.writelines(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                              for event in events)
            handle.flush()
            os.fsync(handle.fileno())
    return True


def record_operation_end(operation_id: str | None, *, status: str,
                         error_type: str = "") -> None:
    path = _ledger_path()
    if path is None or operation_id is None:
        return
    _append(path, {
        "event": "logical_end", "operation_id": operation_id,
        "status": "success" if status == "success" else "failed",
        "error_type": str(error_type)[:100], "timestamp": _now(),
    })


def start_http_attempt(operation_id: str | None, *, attempt: int) -> None:
    """Reserve one physical HTTP attempt before dispatching it."""
    path = _ledger_path()
    if path is None or operation_id is None:
        return
    physical_limit = int(os.environ.get("RSI_CALL_LEDGER_MAX_PHYSICAL", "0") or 0)
    with _LOCK:
        _, physical = _read_counts(path)
        actual_for_operation = 0
        reserved_for_operation = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if item.get("operation_id") != operation_id:
                        continue
                    if item.get("event") == "http_attempt_start":
                        actual_for_operation += 1
                    elif item.get("event") == "physical_reservation":
                        reserved_for_operation += max(0, int(item.get("count", 0)))
        consumes_reserved_capacity = actual_for_operation < reserved_for_operation
        if (physical_limit > 0 and physical >= physical_limit
                and not consumes_reserved_capacity):
            raise RuntimeError("run physical HTTP-attempt budget exhausted before dispatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "http_attempt_start", "operation_id": operation_id,
            "attempt": int(attempt), "timestamp": _now(),
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def end_http_attempt(operation_id: str | None, *, attempt: int,
                     status: int | None, error_type: str = "") -> None:
    path = _ledger_path()
    if path is None or operation_id is None:
        return
    _append(path, {
        "event": "http_attempt_end", "operation_id": operation_id,
        "attempt": int(attempt),
        "status": int(status) if isinstance(status, int) else None,
        "error_type": str(error_type)[:100], "timestamp": _now(),
    })


def timeout_http_attempt(operation_id: str | None, *, attempt: int,
                         error_type: str = "PiBridgeTimeout") -> None:
    """Close a started HTTP attempt whose Pi subprocess was killed by timeout."""
    path = _ledger_path()
    if path is None or operation_id is None:
        return
    _append(path, {
        "event": "http_attempt_timeout", "operation_id": operation_id,
        "attempt": int(attempt), "error_type": str(error_type)[:100],
        "timestamp": _now(),
    })


def reserve_physical_attempts(*, count: int, reason: str,
                              reservation_key: str,
                              operation_id: str | None = None) -> bool:
    """Reserve a conservative physical-attempt ceiling once before dispatch."""
    path = _ledger_path()
    if path is None:
        return False
    count = int(count)
    if count < 0 or not reservation_key:
        raise ValueError("physical reservation needs a non-negative count and key")
    physical_limit = int(os.environ.get("RSI_CALL_LEDGER_MAX_PHYSICAL", "0") or 0)
    with _LOCK:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if (event.get("event") == "physical_reservation"
                        and event.get("reservation_key") == reservation_key):
                    return False
        _, physical = _read_counts(path)
        if physical_limit > 0 and physical + count > physical_limit:
            raise RuntimeError("run physical HTTP-attempt budget exhausted before reservation")
        event = {
            "event": "physical_reservation", "count": count,
            "reservation_key": str(reservation_key)[:160],
            "operation_id": str(operation_id or ""),
            "reason": str(reason)[:300], "source": "conservative_upper_bound",
            "timestamp": _now(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return True


def summarize_ledger(path: str | Path) -> dict[str, Any]:
    """Summarize starts and terminal events; open starts remain counted."""
    source = Path(path)
    starts: dict[str, dict[str, Any]] = {}
    physical_starts: dict[tuple[str, int], dict[str, Any]] = {}
    physical_reservation = 0
    legacy_physical_reservation = 0
    conservative_physical_reservation = 0
    actual_by_operation: dict[str, int] = {}
    unreserved_actual = 0
    reserved_by_operation: dict[str, int] = {}
    unscoped_reservations = 0
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            operation_id = str(item.get("operation_id") or "")
            if item.get("event") == "logical_start":
                starts[operation_id] = {**item, "status": "incomplete"}
            elif item.get("event") == "logical_end" and operation_id in starts:
                starts[operation_id]["status"] = item.get("status", "failed")
            elif item.get("event") == "http_attempt_start":
                attempt = int(item.get("attempt", 0))
                physical_starts[(operation_id, attempt)] = {
                    "operation_id": operation_id, "attempt": int(item.get("attempt", 0)),
                    "status": "incomplete",
                }
                if operation_id:
                    actual_by_operation[operation_id] = (
                        actual_by_operation.get(operation_id, 0) + 1)
                else:
                    unreserved_actual += 1
            elif item.get("event") == "http_attempt_end":
                key = (operation_id, int(item.get("attempt", 0)))
                if key in physical_starts:
                    physical_starts[key]["status"] = item.get("status")
            elif item.get("event") == "http_attempt_timeout":
                requested_attempt = int(item.get("attempt", 0))
                key = (operation_id, requested_attempt)
                if requested_attempt == 0:
                    open_keys = [candidate for candidate, value in physical_starts.items()
                                 if candidate[0] == operation_id
                                 and value["status"] == "incomplete"]
                    if open_keys:
                        key = max(open_keys, key=lambda candidate: candidate[1])
                # A normal terminal end, if already present, remains
                # authoritative. Otherwise timeout closes the physical slot
                # with an explicit unknown outcome.
                if (key in physical_starts
                        and physical_starts[key]["status"] == "incomplete"):
                    physical_starts[key]["status"] = "timeout"
            elif item.get("event") == "physical_reservation":
                count = max(0, int(item.get("count", 0)))
                physical_reservation += count
                reserved_operation_id = str(item.get("operation_id") or "")
                if reserved_operation_id:
                    reserved_by_operation[reserved_operation_id] = (
                        reserved_by_operation.get(reserved_operation_id, 0) + count)
                else:
                    unscoped_reservations += count
                if item.get("source") == "legacy_backfill":
                    legacy_physical_reservation += count
                else:
                    conservative_physical_reservation += count
    by_component: dict[str, int] = {}
    plan_hashes = sorted({
        str(operation.get("plan_sha256"))
        for operation in starts.values()
        if re.fullmatch(r"[0-9a-f]{64}", str(operation.get("plan_sha256") or ""))
    })
    for operation in starts.values():
        component = str(operation.get("component") or "unknown")
        by_component[component] = by_component.get(component, 0) + 1
    statuses: dict[str, int] = {}
    logical_statuses: dict[str, int] = {}
    for attempt in physical_starts.values():
        status = str(attempt.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    for operation in starts.values():
        status = str(operation.get("status"))
        logical_statuses[status] = logical_statuses.get(status, 0) + 1
    physical_budget = (
        sum(max(reserved_by_operation.get(operation_id, 0), actual)
            for operation_id, actual in actual_by_operation.items())
        + sum(count for operation_id, count in reserved_by_operation.items()
              if operation_id not in actual_by_operation)
        + unreserved_actual + unscoped_reservations)
    return {
        "logical_operations": len(starts),
        "execution_plan_sha256s": plan_hashes,
        "logical_by_component": dict(sorted(by_component.items())),
        "logical_status_counts": dict(sorted(logical_statuses.items())),
        "logical_incomplete": sum(1 for item in starts.values()
                                   if item.get("status") == "incomplete"),
        "physical_http_attempts": len(physical_starts),
        "legacy_unknown_physical_reserved": legacy_physical_reservation,
        "conservative_physical_reserved": conservative_physical_reservation,
        "physical_budget_consumed_or_reserved": physical_budget,
        "physical_status_counts": dict(sorted(statuses.items())),
        "physical_attempts_exact": physical_reservation == 0,
        "physical_incomplete": sum(1 for item in physical_starts.values()
                                    if item.get("status") == "incomplete"),
        "physical_timeouts": sum(1 for item in physical_starts.values()
                                  if item.get("status") == "timeout"),
    }
