"""Hash-bound, approved and budgeted G05 task-authoring execution.

Planning is offline and dry-run is the default. Provider calls require a
separate approval artifact for the exact execution plan. A started request is
never replayed after interruption because the remote side may have accepted it.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Any

from .g05_task_authoring import (
    MAX_ANSWER_CHARS,
    MAX_COMPLETION_TOKENS,
    MAX_QUESTION_CHARS,
    PLAN_SCHEMA as AUTHORING_PLAN_SCHEMA,
    _validate_sample_plan_for_authoring,
)
from .g05_task_packet import _canonical_identity, _canonical_sha, _require_sha
from ..runtime.provider import (
    SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL,
    SILICONFLOW_CN_PRICE_SOURCE,
    published_price_for_endpoint,
)

EXECUTION_PLAN_SCHEMA = "g05-task-authoring-execution-plan-v1"
APPROVAL_SCHEMA = "g05-task-authoring-execution-approval-v1"
RUN_SCHEMA = "g05-task-authoring-execution-run-v1"
APPROVAL_STATEMENT = "I_APPROVE_G05_AUTHORING_MODEL_CALLS"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_IDENTITY = ("siliconflow", "qwen3-8b", "qwen")
BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY_ENV = "SILICONFLOW_API_KEY"
PRICE_SOURCE = "https://siliconflow.cn/pricing"
PRICE_EVIDENCE_DEFAULT = Path(
    ".workbuddy/g06-instruction-reviewer-v3-ai-reference-20261016/pricing-evidence.json")
MAX_CALLS = 120
MAX_RETRIES = 0
MAX_TOKENS_PER_CALL = MAX_COMPLETION_TOKENS
MAX_COMPLETION_TOKENS_TOTAL = MAX_CALLS * MAX_TOKENS_PER_CALL
MAX_INPUT_CHARS_PER_CALL = 8_000
MAX_INPUT_CHARS_TOTAL = MAX_CALLS * MAX_INPUT_CHARS_PER_CALL
MAX_COST_USD = 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: Any, name: str) -> tuple[str, str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} model identity is invalid")
    return _canonical_identity(*value)


def _verified_authoring_plan(authoring_plan: dict, sample_plan: dict) -> list[dict]:
    if not isinstance(authoring_plan, dict) or authoring_plan.get("schema") != AUTHORING_PLAN_SCHEMA:
        raise ValueError("unsupported G05 task-authoring plan")
    authoring_sha = _require_sha(authoring_plan.get("plan_sha256"), "authoring plan hash is invalid")
    if _canonical_sha({key: value for key, value in authoring_plan.items()
                       if key != "plan_sha256"}) != authoring_sha:
        raise ValueError("authoring plan hash mismatch")
    if (authoring_plan.get("status") != "draft_requires_root_review"
            or authoring_plan.get("calls_allowed") is not False
            or authoring_plan.get("dispatch_authorized") is not False
            or authoring_plan.get("task_contract") != "selected_answer_support_v1"
            or authoring_plan.get("scope") != "single_doc"
            or authoring_plan.get("sample_plan_sha256") != sample_plan.get("plan_sha256")
            or authoring_plan.get("source_case_count") != MAX_CALLS
            or authoring_plan.get("preflight_eligible_count") != MAX_CALLS
            or authoring_plan.get("preflight_excluded_count") != 0
            or authoring_plan.get("max_physical_attempts") != MAX_CALLS
            or authoring_plan.get("max_attempts_per_case") != 1
            or authoring_plan.get("retries_per_case") != MAX_RETRIES
            or authoring_plan.get("max_completion_tokens_per_case") != MAX_TOKENS_PER_CALL
            or authoring_plan.get("max_total_completion_tokens") != MAX_COMPLETION_TOKENS_TOTAL):
        raise ValueError("authoring plan is not the complete review-only G05 v5 draft")
    shells = _validate_sample_plan_for_authoring(sample_plan)
    if len(shells) != MAX_CALLS:
        raise ValueError("G05 execution requires exactly 120 frozen sample shells")
    author = _identity(authoring_plan.get("author_model_identity"), "author")
    target = _identity(authoring_plan.get("target_model_identity"), "target")
    references = authoring_plan.get("reference_model_identities")
    if not isinstance(references, list) or len(references) != 2:
        raise ValueError("G05 plan requires two trusted reference identities")
    refs = [_identity(value, "reference") for value in references]
    if (author != MODEL_IDENTITY or target[2] in {refs[0][2], refs[1][2]}
            or refs[0][2] == refs[1][2] or author[2] == target[2]):
        raise ValueError("author/target/reference identities violate the frozen role contract")

    rows = authoring_plan.get("cases")
    if not isinstance(rows, list) or len(rows) != MAX_CALLS:
        raise ValueError("G05 plan must bind exactly 120 authoring prompts")
    shell_by_id = {row["case_id"]: row for row in shells}
    if len(shell_by_id) != MAX_CALLS or len({row.get("case_id") for row in rows}) != MAX_CALLS:
        raise ValueError("G05 sample and authoring plan case identities are not unique")
    if len({row.get("source_group_id") for row in rows}) != MAX_CALLS:
        raise ValueError("G05 sample requires 120 independent source groups")
    for row in rows:
        shell = shell_by_id.get(row.get("case_id"))
        system, user = row.get("system"), row.get("user")
        if (shell is None or row.get("source_group_id") != shell.get("source_group_id")
                or row.get("source_content_sha256") != shell.get("source_content_sha256")
                or row.get("preflight_status") != "eligible"
                or row.get("max_completion_tokens") != MAX_TOKENS_PER_CALL
                or not isinstance(system, str) or not isinstance(user, str)
                or len(system) + len(user) > MAX_INPUT_CHARS_PER_CALL
                or row.get("prompt_sha256") != hashlib.sha256(
                    (system + "\0" + user).encode("utf-8")).hexdigest()):
            raise ValueError("authoring prompt/source binding or per-call cap is invalid")
    return rows


def _verify_price_evidence(path: Path, workspace: Path) -> dict:
    root, report_path = workspace.resolve(), path.resolve()
    try:
        report_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("price evidence must be stored inside the workspace") from exc
    if not report_path.is_file():
        raise ValueError("current official zero-price evidence is missing")
    evidence = json.loads(report_path.read_text(encoding="utf-8"))
    if (evidence.get("schema") != "siliconflow-cn-live-pricing-evidence-v1"
            or evidence.get("checked_at") != date.today().isoformat()
            or evidence.get("http_status") != 200
            or evidence.get("source_url") != PRICE_SOURCE
            or evidence.get("actual_account_invoice_verified") is not False):
        raise ValueError("official price evidence is invalid, stale, or not point-in-time verified")
    raw_value = evidence.get("raw_html_path")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("official price source body receipt is missing")
    raw_path = (root / raw_value).resolve()
    try:
        raw_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("official price source body path escapes workspace") from exc
    if (not raw_path.is_file()
            or _file_sha256(raw_path) != evidence.get("raw_html_sha256")):
        raise ValueError("official price source body hash mismatch")
    rows = evidence.get("model_rows")
    row = next((item for item in rows or []
                if isinstance(item, dict) and item.get("model_id") == MODEL_ID), None)
    if (not isinstance(row, dict)
            or row.get("token_price_cny_per_million") != "0"
            or row.get("currency") != "¥"
            or row.get("status") != "normal"
            or row.get("need_auth") is not False
            or row.get("only_by_charge_balance") is not False):
        raise ValueError("Qwen3-8B is not explicitly verified as zero-price on the CN endpoint")
    rates = published_price_for_endpoint(MODEL_ID, BASE_URL)
    if not rates or any(value != 0 for value in rates.values()):
        raise ValueError("runtime tariff registry does not prove a zero price for the frozen endpoint")
    from ..runtime.provider import SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL
    return {"report_path": report_path.relative_to(root).as_posix(),
        "report_sha256": _file_sha256(report_path),
        "raw_html_path": raw_path.relative_to(root).as_posix(),
        "raw_html_sha256": evidence["raw_html_sha256"],
        "checked_at": evidence["checked_at"], "model_id": MODEL_ID,
        "base_url": BASE_URL, "currency": "CNY", "price_per_million_tokens": "0",
        "provider_price_evidence_sha256": SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL[MODEL_ID],
        "account_invoice_verified": False}


def freeze_g05_task_authoring_execution_plan(
    authoring_plan: dict, sample_plan: dict, *,
    pricing_evidence_path: Path, workspace: Path,
) -> dict:
    """Freeze a provider-ready execution plan without authorizing dispatch."""
    rows = _verified_authoring_plan(authoring_plan, sample_plan)
    price = _verify_price_evidence(Path(pricing_evidence_path), Path(workspace))
    authoring_path = Path(__file__).with_name("g05_task_authoring.py")
    provider_path = Path(__file__).resolve().parents[1] / "runtime" / "provider.py"
    body = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "status": "frozen_requires_hash_bound_approval",
        "authoring_plan_sha256": authoring_plan["plan_sha256"],
        "sample_plan_sha256": sample_plan["plan_sha256"],
        "model_identity": list(MODEL_IDENTITY),
        "provider_model_id": MODEL_ID,
        "provider": "http",
        "base_url": BASE_URL,
        "api_key_env": API_KEY_ENV,
        "paid_fallback": False,
        "price_evidence": price,
        "execution_source_sha256": _file_sha256(Path(__file__)),
        "authoring_source_sha256": _file_sha256(authoring_path),
        "provider_source_sha256": _file_sha256(provider_path),
        "case_ids_sha256": _canonical_sha([row["case_id"] for row in rows]),
        "sources_sha256": _canonical_sha([
            [row["case_id"], row["source_group_id"], row["source_content_sha256"]]
            for row in rows]),
        "prompts_sha256": _canonical_sha([
            [row["case_id"], row["prompt_sha256"]] for row in rows]),
        "max_logical_calls": MAX_CALLS,
        "max_physical_attempts": MAX_CALLS,
        "max_retries_per_call": MAX_RETRIES,
        "max_tokens_per_call": MAX_TOKENS_PER_CALL,
        "max_total_completion_tokens": MAX_COMPLETION_TOKENS_TOTAL,
        "max_input_chars_per_call": MAX_INPUT_CHARS_PER_CALL,
        "max_total_input_chars": MAX_INPUT_CHARS_TOTAL,
        "max_cost_usd": MAX_COST_USD,
        "cache_enabled": False,
        "planning_calls": 0,
        "dispatch_authorized": False,
        "case_count": MAX_CALLS,
    }
    return {**body, "plan_sha256": _canonical_sha(body)}


def verify_execution_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != EXECUTION_PLAN_SCHEMA:
        raise ValueError("unsupported G05 execution plan")
    plan_sha = _require_sha(plan.get("plan_sha256"), "execution plan hash is invalid")
    if _canonical_sha({key: value for key, value in plan.items()
                       if key != "plan_sha256"}) != plan_sha:
        raise ValueError("execution plan hash mismatch")
    if (plan.get("status") != "frozen_requires_hash_bound_approval"
            or plan.get("dispatch_authorized") is not False
            or plan.get("paid_fallback") is not False
            or plan.get("model_identity") != list(MODEL_IDENTITY)
            or plan.get("provider_model_id") != MODEL_ID
            or plan.get("provider") != "http"
            or plan.get("base_url") != BASE_URL
            or plan.get("api_key_env") != API_KEY_ENV
            or plan.get("max_logical_calls") != MAX_CALLS
            or plan.get("max_physical_attempts") != MAX_CALLS
            or plan.get("max_retries_per_call") != MAX_RETRIES
            or plan.get("max_tokens_per_call") != MAX_TOKENS_PER_CALL
            or plan.get("max_total_completion_tokens") != MAX_COMPLETION_TOKENS_TOTAL
            or plan.get("max_input_chars_per_call") != MAX_INPUT_CHARS_PER_CALL
            or plan.get("max_total_input_chars") != MAX_INPUT_CHARS_TOTAL
            or plan.get("max_cost_usd") != MAX_COST_USD
            or plan.get("cache_enabled") is not False
            or plan.get("planning_calls") != 0
            or plan.get("case_count") != MAX_CALLS):
        raise ValueError("execution plan violates model identity or frozen hard caps")


def approve_g05_task_authoring_execution(
    plan: dict, *, approved_by: str, approval_statement: str,
    approved_at: str | None = None,
) -> dict:
    """Create a human-reviewed approval artifact; this function never dispatches."""
    verify_execution_plan(plan)
    if approval_statement != APPROVAL_STATEMENT:
        raise ValueError("explicit approval statement is required")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("approval reviewer identity is required")
    timestamp = approved_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("approval timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("approval timestamp must include a timezone")
    body = {"schema": APPROVAL_SCHEMA, "approved": True,
        "approved_by": approved_by.strip(), "approved_at": timestamp,
        "approval_statement": APPROVAL_STATEMENT,
        "execution_plan_sha256": plan["plan_sha256"],
        "authoring_plan_sha256": plan["authoring_plan_sha256"],
        "sample_plan_sha256": plan["sample_plan_sha256"],
        "model_identity": plan["model_identity"],
        "provider_model_id": plan["provider_model_id"], "base_url": plan["base_url"],
        "price_evidence_sha256": plan["price_evidence"]["report_sha256"],
        "max_logical_calls": MAX_CALLS, "max_physical_attempts": MAX_CALLS,
        "max_retries_per_call": MAX_RETRIES, "max_tokens_per_call": MAX_TOKENS_PER_CALL,
        "max_total_completion_tokens": MAX_COMPLETION_TOKENS_TOTAL,
        "max_input_chars_per_call": MAX_INPUT_CHARS_PER_CALL,
        "max_total_input_chars": MAX_INPUT_CHARS_TOTAL,
        "cache_enabled": False,
        "max_cost_usd": MAX_COST_USD, "paid_fallback": False}
    return {**body, "approval_sha256": _canonical_sha(body)}


def verify_g05_authoring_approval(plan: dict, approval: dict) -> dict:
    verify_execution_plan(plan)
    if not isinstance(approval, dict) or approval.get("schema") != APPROVAL_SCHEMA:
        raise ValueError("G05 execution approval is missing or malformed")
    approval_sha = _require_sha(approval.get("approval_sha256"), "approval hash is invalid")
    if _canonical_sha({key: value for key, value in approval.items()
                       if key != "approval_sha256"}) != approval_sha:
        raise ValueError("approval hash mismatch")
    expected = approve_g05_task_authoring_execution(plan,
        approved_by=approval.get("approved_by"),
        approval_statement=approval.get("approval_statement"),
        approved_at=approval.get("approved_at"))
    if approval != expected or approval.get("approved") is not True:
        raise ValueError("approval does not match exact frozen plan and all hard caps")
    return {"approved": True, "approval_sha256": approval_sha,
        "approved_by": approval["approved_by"], "approved_at": approval["approved_at"]}


def _strict_task_json(text: Any) -> tuple[dict | None, str | None]:
    if not isinstance(text, str):
        return None, "response_not_text"
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "invalid_json_or_duplicate_key"
    if not isinstance(value, dict) or set(value) != {"question", "selected_answer"}:
        return None, "response_fields_mismatch"
    question, answer = value["question"], value["selected_answer"]
    if (not isinstance(question, str) or not isinstance(answer, str)
            or not question.strip() or not answer.strip()
            or len(question.strip()) > MAX_QUESTION_CHARS
            or len(answer.strip()) > MAX_ANSWER_CHARS
            or not question.strip().endswith("?")
            or question.strip().casefold() == answer.strip().casefold()):
        return None, "response_content_contract_invalid"
    return {"question": question.strip(), "selected_answer": answer.strip()}, None


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _write_once(path: Path, row: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _write_status(path: Path, row: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    data = (json.dumps(row, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def _run_lock(run_dir: Path):
    """Cross-process exclusive lock; an OS lock releases automatically on crash."""
    lock_path = run_dir / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if lock_path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another G05 executor holds the run lock") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another G05 executor holds the run lock") from exc
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"G05 call ledger is corrupt at line {number}; refusing resume") from exc
        if not isinstance(row, dict):
            raise ValueError("G05 call ledger contains a non-object event")
        result.append(row)
    return result


def _ledger_states(events: list[dict]) -> tuple[dict, set[str], set[str]]:
    starts: dict[str, dict] = {}
    ends: dict[str, dict] = {}
    for row in events:
        event = row.get("event")
        case_id = row.get("case_id")
        if event == "g05_authoring_case_start":
            if not isinstance(case_id, str) or case_id in starts:
                raise ValueError("duplicate or malformed case start; refusing resume")
            starts[case_id] = row
        elif event == "g05_authoring_case_end":
            if case_id not in starts or case_id in ends:
                raise ValueError("orphan or duplicate case terminal; refusing resume")
            ends[case_id] = row
    return ends, set(starts), set(starts) - set(ends)


def _save_once_json(path: Path, value: Any) -> str:
    return _write_once(path, value)


def _check_frozen_sources(plan: dict) -> None:
    if (_file_sha256(Path(__file__)) != plan.get("execution_source_sha256")
            or _file_sha256(Path(__file__).with_name("g05_task_authoring.py"))
                != plan.get("authoring_source_sha256")
            or _file_sha256(Path(__file__).resolve().parents[1] / "runtime" / "provider.py")
                != plan.get("provider_source_sha256")):
        raise ValueError("execution, authoring, or provider source changed after plan freeze")


def _verify_resume_anchor(run_dir: Path, ledger_path: Path, plan: dict,
                          authoring_plan: dict, sample_plan: dict,
                          approval: dict) -> None:
    """Reject ledger edits or crash windows that lack a durable status anchor."""
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise ValueError("G05 resume status anchor is missing")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (ValueError, TypeError) as exc:
        raise ValueError("G05 resume status anchor is corrupt") from exc
    if not isinstance(status, dict):
        raise ValueError("G05 resume status anchor is malformed")
    if (status.get("status") == "started" and not ledger_path.exists()
            and status.get("execution_plan_sha256") == plan["plan_sha256"]
            and status.get("model_calls") == 0):
        # A crash before the first start event is safe to resume.
        return
    status_sha = status.get("run_sha256")
    if (not isinstance(status_sha, str)
            or _canonical_sha({key: value for key, value in status.items()
                               if key != "run_sha256"}) != status_sha):
        raise ValueError("G05 resume status anchor hash is missing or invalid")
    if (status.get("execution_plan_sha256") != plan["plan_sha256"]
            or status.get("approval_sha256") != approval.get("approval_sha256")
            or status.get("authoring_plan_sha256") != authoring_plan["plan_sha256"]
            or status.get("sample_plan_sha256") != sample_plan["plan_sha256"]):
        raise ValueError("G05 resume status anchor is bound to different frozen artifacts")
    current_ledger_sha = _file_sha256(ledger_path) if ledger_path.is_file() else None
    if current_ledger_sha != status.get("call_ledger_sha256"):
        raise ValueError("G05 ledger hash does not match the last run status")


def execute_g05_task_authoring(
    plan: dict, authoring_plan: dict, sample_plan: dict, approval: dict | None,
    run_dir: Path, *, execute: bool = False, resume: bool = False,
    provider=None, workspace: Path | None = None,
    stop_after_first_error: bool = False,
    stop_after_hard_violation: bool = False,
) -> dict:
    """Validate by default; dispatch only with explicit execute and exact approval.

    Injected providers are available only to isolated unit tests. Their data is
    always marked test-only and can never yield a production task packet.
    """
    workspace = Path(workspace or Path(__file__).resolve().parents[2]).resolve()
    cases = _verified_authoring_plan(authoring_plan, sample_plan)
    verify_execution_plan(plan)
    if (plan.get("authoring_plan_sha256") != authoring_plan["plan_sha256"]
            or plan.get("sample_plan_sha256") != sample_plan["plan_sha256"]
            or plan.get("case_ids_sha256") != _canonical_sha([row["case_id"] for row in cases])
            or plan.get("sources_sha256") != _canonical_sha([
                [row["case_id"], row["source_group_id"], row["source_content_sha256"]]
                for row in cases])
            or plan.get("prompts_sha256") != _canonical_sha([
                [row["case_id"], row["prompt_sha256"]] for row in cases])):
        raise ValueError("execution plan does not bind these authoring and sample artifacts")
    current_price = _verify_price_evidence(
        workspace / plan["price_evidence"]["report_path"], workspace)
    if current_price != plan["price_evidence"]:
        raise ValueError("pricing evidence changed or expired after plan freeze")
    _check_frozen_sources(plan)
    approval_info = verify_g05_authoring_approval(plan, approval) if approval is not None else None
    if not execute:
        return {"schema": RUN_SCHEMA, "status": "dry_run_validated",
            "execution_plan_sha256": plan["plan_sha256"],
            "approval_validated": approval_info is not None,
            "planned_calls": len(cases), "model_calls": 0, "dispatch_authorized": False}
    if approval_info is None:
        raise ValueError("execution requires a separate hash-bound approval file")
    if provider is not None and not os.environ.get("RSI_HOME"):
        raise ValueError("injected providers are test-only and require an isolated RSI_HOME")

    run_dir = Path(run_dir).resolve()
    try:
        run_dir.relative_to((workspace / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("G05 run artifacts must stay under workspace .workbuddy") from exc
    ledger_path = run_dir / "call-ledger.jsonl"
    if not resume:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError("G05 run already exists; use resume without overwriting history") from exc
        for name, row in (("execution-plan.json", plan), ("approval.json", approval),
                          ("authoring-plan.json", authoring_plan), ("sample-plan.json", sample_plan)):
            _save_once_json(run_dir / name, row)
        _save_once_json(run_dir / "status.json", {"status": "started",
            "execution_plan_sha256": plan["plan_sha256"], "model_calls": 0})
    else:
        if not run_dir.is_dir():
            raise FileNotFoundError("cannot resume a G05 run that does not exist")
        for name, expected in (("execution-plan.json", plan), ("approval.json", approval),
                              ("authoring-plan.json", authoring_plan), ("sample-plan.json", sample_plan)):
            path = run_dir / name
            if (not path.is_file() or _canonical_sha(
                    json.loads(path.read_text(encoding="utf-8"))) != _canonical_sha(expected)):
                raise ValueError("resume artifacts differ from the original approved run")

    # The advisory lock prevents two concurrent resumes from dispatching the
    # same unstarted case. The OS releases it automatically if the process dies.
    with _run_lock(run_dir):
        return _execute_locked(plan, authoring_plan, sample_plan, approval,
            cases, run_dir, ledger_path, execute=True, resume=resume,
            provider=provider, workspace=workspace,
            stop_after_first_error=stop_after_first_error,
            stop_after_hard_violation=stop_after_hard_violation)


def _execute_locked(plan, authoring_plan, sample_plan, approval, cases,
                    run_dir, ledger_path, *, execute, resume, provider, workspace,
                    stop_after_first_error, stop_after_hard_violation):

    if resume:
        _verify_resume_anchor(run_dir, ledger_path, plan, authoring_plan,
                              sample_plan, approval)
    events = _read_ledger(ledger_path)
    terminal, started, orphaned = _ledger_states(events)
    by_case = {row["case_id"]: row for row in cases}
    if (set(started) - set(by_case) or set(terminal) - set(by_case)
            or len(started) > plan["max_physical_attempts"]):
        raise ValueError("G05 run ledger violates plan identity or physical cap")
    records: list[dict] = []
    for case_id, end in terminal.items():
        if end.get("result_sha256") is None:
            continue
        result_path = run_dir / "cases" / f"{case_id}.json"
        if not result_path.is_file() or _file_sha256(result_path) != end["result_sha256"]:
            raise ValueError("completed case artifact is missing or changed")
        receipt = json.loads(result_path.read_text(encoding="utf-8"))
        if receipt.get("call_record"):
            records.append(receipt["call_record"])
    unknown = len(orphaned)

    if provider is None:
        if not os.environ.get(API_KEY_ENV):
            raise RuntimeError(f"required credential environment variable is missing: {API_KEY_ENV}")
        from ..runtime.provider import build_provider
        actual_provider = build_provider({
            "runtime": {"provider": "http", "http_no_cache_required": True},
            "models": {"task": {"provider": "http", "model_id": MODEL_ID,
                "base_url": BASE_URL, "api_key_env": API_KEY_ENV,
                "max_tokens": MAX_TOKENS_PER_CALL, "max_retries": 1,
                "timeout": 120, "temperature": 0, "cache": False,
                "enable_thinking": False}}})
    else:
        actual_provider = provider

    env_names = ("RSI_ACTIVE_EXECUTION_PLAN_SHA256", "RSI_CALL_LEDGER_PATH",
        "RSI_CALL_LEDGER_MAX_LOGICAL", "RSI_CALL_LEDGER_MAX_PHYSICAL",
        "RSI_CALL_LEDGER_MAX_COMPONENT_AI_EVALUATION", "RSI_HTTP_NO_CACHE")
    saved_env = {key: os.environ.get(key) for key in env_names}
    bindings = {"RSI_ACTIVE_EXECUTION_PLAN_SHA256": plan["plan_sha256"],
        "RSI_CALL_LEDGER_PATH": str(ledger_path),
        "RSI_CALL_LEDGER_MAX_LOGICAL": str(MAX_CALLS),
        "RSI_CALL_LEDGER_MAX_PHYSICAL": str(MAX_CALLS),
        "RSI_CALL_LEDGER_MAX_COMPONENT_AI_EVALUATION": str(MAX_CALLS),
        "RSI_HTTP_NO_CACHE": "1"}
    os.environ.update(bindings)
    total_completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in records)
    total_prompt_tokens = sum(int(row.get("prompt_tokens", 0)) for row in records)
    stopped = False
    interrupted_by_error = False
    try:
        from ..runtime.call_ledger import call_component
        for case in cases:
            case_id = case["case_id"]
            if case_id in started:
                continue  # Completed and ambiguous starts are both non-repeatable.
            if len(started) >= plan["max_physical_attempts"]:
                raise RuntimeError("physical request cap reached before dispatch")
            if total_completion_tokens > plan["max_total_completion_tokens"]:
                raise RuntimeError("total completion-token cap reached before dispatch")
            _append_jsonl(ledger_path, {"event": "g05_authoring_case_start",
                "case_id": case_id, "attempt": 1, "prompt_sha256": case["prompt_sha256"],
                "execution_plan_sha256": plan["plan_sha256"],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            started.add(case_id)
            try:
                with call_component("ai_evaluation"):
                    output = actual_provider.chat_measured(
                        case["system"], case["user"], max_tokens=plan["max_tokens_per_call"])
                prompt_tokens = getattr(output, "prompt_tokens", None)
                completion_tokens = getattr(output, "completion_tokens", None)
                output_model_id = getattr(output, "model_id", None)
                cost = getattr(output, "cost", None) or {}
                usage_valid = (getattr(output, "usage_available", False) is True
                    and type(prompt_tokens) is int and 0 <= prompt_tokens <= MAX_INPUT_CHARS_PER_CALL
                    and type(completion_tokens) is int and 0 <= completion_tokens <= MAX_TOKENS_PER_CALL
                    and total_prompt_tokens + prompt_tokens <= MAX_INPUT_CHARS_TOTAL
                    and total_completion_tokens + completion_tokens <= MAX_COMPLETION_TOKENS_TOTAL)
                tariff_valid = (cost.get("usd") == 0.0
                    and cost.get("cost_estimated") is False
                    and cost.get("usage_complete") is True
                    and cost.get("pricing_region") == "CN"
                    and cost.get("pricing_source") == PRICE_SOURCE
                    and cost.get("pricing_evidence_report_sha256") ==
                        SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL[MODEL_ID])
                identity_valid = (output_model_id == MODEL_ID
                    and getattr(output, "http_attempts", None) == 1)
                parsed, parse_error = _strict_task_json(getattr(output, "content", None))
                reason = (parse_error or (None if usage_valid and tariff_valid and identity_valid
                    else "usage_cost_model_or_http_identity_contract_failed"))
                hard_violation = not (usage_valid and tariff_valid and identity_valid)
                state = "valid" if reason is None else "invalid"
                raw_response = parsed if parsed is not None else getattr(output, "content", None)
                call_record = {"case_id": case_id,
                    "status": "succeeded" if state == "valid" else "invalid",
                    "physical_attempts": 1, "provider": "siliconflow",
                    "model_id": MODEL_ID, "model_family": "qwen",
                    "request_sha256": case["prompt_sha256"],
                    "raw_response": raw_response,
                    "raw_response_sha256": _canonical_sha(raw_response),
                    "completion_tokens": completion_tokens if type(completion_tokens) is int else 0,
                    "prompt_tokens": prompt_tokens if type(prompt_tokens) is int else 0,
                    "response_model_id": output_model_id, "cost": cost}
                receipt = {"schema": "g05-task-authoring-call-receipt-v1",
                    "case_id": case_id, "prompt_sha256": case["prompt_sha256"],
                    "source_group_id": case["source_group_id"],
                    "source_content_sha256": case["source_content_sha256"],
                    "authoring_route": case["authoring_route"],
                    "status": state, "reason": reason,
                    "provider_model_id": MODEL_ID, "response_model_id": output_model_id,
                    "http_attempts": getattr(output, "http_attempts", None),
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "cost": cost, "raw_response": raw_response,
                    "raw_output": getattr(output, "content", None),
                    "raw_output_sha256": _canonical_sha(getattr(output, "content", None)),
                    "raw_response_sha256": _canonical_sha(raw_response),
                    "parsed_response": parsed, "call_record": call_record,
                    "test_only": provider is not None}
                result_path = run_dir / "cases" / f"{case_id}.json"
                result_sha = _write_once(result_path, receipt)
                _append_jsonl(ledger_path, {"event": "g05_authoring_case_end",
                    "case_id": case_id, "status": state, "reason": reason,
                    "result_sha256": result_sha, "physical_attempts_reserved": 1,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "cost_usd": cost.get("usd"), "timestamp": datetime.now(
                        timezone.utc).isoformat(timespec="seconds")})
                terminal[case_id] = {"status": state, "result_sha256": result_sha}
                records.append(call_record)
                if type(prompt_tokens) is int and prompt_tokens >= 0:
                    total_prompt_tokens += prompt_tokens
                if type(completion_tokens) is int and completion_tokens >= 0:
                    total_completion_tokens += completion_tokens
                if (stop_after_hard_violation and reason) or hard_violation:
                    stopped = True
                    break
            except BaseException as exc:
                # Unknown dispatch outcomes are reserved and not retried.
                _append_jsonl(ledger_path, {"event": "g05_authoring_case_end",
                    "case_id": case_id, "status": "failed_unknown_outcome",
                    "reason": type(exc).__name__, "physical_attempts_reserved": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                terminal[case_id] = {"status": "failed_unknown_outcome"}
                unknown += 1
                if stop_after_first_error or stop_after_hard_violation:
                    stopped = True
                    interrupted_by_error = True
                    break
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    events = _read_ledger(ledger_path)
    terminal, started, orphaned = _ledger_states(events)
    unknown = len(orphaned) + sum(row.get("status") == "failed_unknown_outcome"
                                  for row in terminal.values())
    valid_receipts = []
    invalid_count = 0
    for case_id, end in terminal.items():
        if end.get("status") != "valid" or not end.get("result_sha256"):
            if end.get("status") == "invalid":
                invalid_count += 1
            continue
        receipt_path = run_dir / "cases" / f"{case_id}.json"
        if not receipt_path.is_file() or _file_sha256(receipt_path) != end["result_sha256"]:
            raise ValueError("case output is missing or changed; refusing acceptance")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("test_only") is not (provider is not None)
                or receipt.get("status") != "valid"
                or receipt.get("parsed_response") is None):
            raise ValueError("case receipt integrity or test-only boundary failed")
        valid_receipts.append(receipt)
    # Add pre-existing valid receipts from prior process invocations.
    valid_count = len(valid_receipts)
    all_done = len(terminal) == MAX_CALLS and len(started) == MAX_CALLS
    complete = all_done and valid_count == MAX_CALLS and invalid_count == 0 and unknown == 0 and not stopped
    test_only = provider is not None
    if test_only and complete:
        status = "test_only_not_promotable"
    elif complete:
        status = "complete"
    elif stopped and interrupted_by_error:
        status = "interrupted_incomplete"
    else:
        status = "incomplete_or_invalid"
    authoring_input = None
    if complete and not test_only:
        case_by_id = {row["case_id"]: row for row in valid_receipts}
        authored_cases = []
        for case in cases:
            parsed = case_by_id[case["case_id"]]["parsed_response"]
            authored_cases.append({"case_id": case["case_id"],
                "source_group_id": case["source_group_id"],
                "question": parsed["question"],
                "selected_answer": parsed["selected_answer"]})
        authoring_input = {"schema": "g05-task-authoring-input-v1",
            "sample_plan_sha256": sample_plan["plan_sha256"], "cases": authored_cases}
    run_summary = {"schema": RUN_SCHEMA, "status": status,
        "execution_plan_sha256": plan["plan_sha256"],
        "approval_sha256": approval["approval_sha256"],
        "authoring_plan_sha256": authoring_plan["plan_sha256"],
        "sample_plan_sha256": sample_plan["plan_sha256"],
        "case_count": MAX_CALLS, "attempted_case_count": len(started),
        "physical_attempts": len(started), "max_physical_attempts": MAX_CALLS,
        "valid_count": valid_count, "invalid_count": invalid_count,
        "unknown_outcome_count": unknown, "model_calls": len(started),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "max_total_completion_tokens": MAX_COMPLETION_TOKENS_TOTAL,
        "max_cost_usd": MAX_COST_USD, "paid_fallback": False,
        "test_only": test_only, "authoring_input": authoring_input,
        "call_ledger_path": "call-ledger.jsonl",
        "call_ledger_sha256": _file_sha256(ledger_path) if ledger_path.is_file() else None}
    run_summary["run_sha256"] = _canonical_sha(run_summary)
    _write_status(run_dir / "status.json", run_summary)
    if authoring_input is not None:
        _save_once_json(run_dir / "authoring-input.json", authoring_input)
    return run_summary
