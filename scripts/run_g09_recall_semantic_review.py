"""One-shot G09 staged-candidate semantic review.

The command has an offline plan/approval mode and an explicitly approved
execute mode.  It never creates source groups or labels; its output is only a
semantic review receipt that must be rechecked by the existing strict source
acceptance adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Direct script execution sets ``sys.path[0]`` to ``scripts``; bind imports to
# this repository explicitly so the approved CLI and module tests use the same
# source tree.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.open_research_recall import (
    build_semantic_review_plan,
    parse_semantic_review_output,
    validate_semantic_review_plan,
)
from src.runtime.provider import SILICONFLOW_CN_FREE_MODELS, SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL
from src.util import sha256_obj

BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY_ENV = "SILICONFLOW_API_KEY"
MAX_TOKENS = 512


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes().decode("utf-8"))


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if exclusive:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
    return _sha_bytes(raw)


def _approval_body(plan: dict[str, Any], *, plan_file_sha256: str) -> dict[str, Any]:
    return {
        "schema": "g09-recall-semantic-review-approval-v1",
        "status": "approved",
        "approved_by": "root",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "dispatch_authorized": True,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": plan_file_sha256,
        "ledger_sha256": plan["ledger_sha256"],
        "model_id": plan["model_id"],
        "max_calls": plan["max_calls"],
        "max_physical_attempts": plan["max_physical_attempts"],
        "hard_cost_usd": "0.00",
        "price_evidence_sha256": SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL[plan["model_id"]],
        "scope": "G09 staged-candidate semantic review only; no source groups, labels, scores, answers, cohort, or final access",
    }


def make_approval(*, plan_path: Path, approval_path: Path) -> dict[str, Any]:
    raw = plan_path.read_bytes()
    plan = json.loads(raw.decode("utf-8"))
    body = _approval_body(plan, plan_file_sha256=_sha_bytes(raw))
    approval = {**body, "approval_sha256": sha256_obj(body)}
    _write(approval_path, approval, exclusive=True)
    return {"status": "approval_written", "approval_sha256": approval["approval_sha256"],
        "plan_sha256": plan["plan_sha256"], "max_calls": plan["max_calls"],
        "network_calls": 0, "model_calls": 0}


def _validate_approval(approval: dict[str, Any], plan: dict[str, Any], *, plan_file_sha256: str) -> None:
    if approval.get("schema") != "g09-recall-semantic-review-approval-v1" or approval.get("status") != "approved":
        raise ValueError("semantic review approval is not approved")
    if approval.get("approved_by") != "root" or approval.get("dispatch_authorized") is not True:
        raise ValueError("semantic review approval must be root-authorized")
    if approval.get("plan_sha256") != plan["plan_sha256"] or approval.get("plan_file_sha256") != plan_file_sha256:
        raise ValueError("semantic review approval plan binding mismatch")
    if approval.get("ledger_sha256") != plan["ledger_sha256"] or approval.get("model_id") != plan["model_id"]:
        raise ValueError("semantic review approval ledger/model binding mismatch")
    if approval.get("max_calls") != plan["max_calls"] or approval.get("max_physical_attempts") != plan["max_physical_attempts"]:
        raise ValueError("semantic review approval budget mismatch")
    if approval.get("hard_cost_usd") != "0.00" or approval.get("price_evidence_sha256") != SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL[plan["model_id"]]:
        raise ValueError("semantic review approval is not zero-price bound")
    if approval.get("approval_sha256") != sha256_obj({key: value for key, value in approval.items() if key != "approval_sha256"}):
        raise ValueError("semantic review approval hash mismatch")


def _request(plan_case: dict[str, Any], *, api_key: str, requests_module: Any) -> tuple[dict[str, Any], bytes]:
    body = {"model": plan_case["model_id"], "messages": [
        {"role": "system", "content": "You are a strict scientific source-relevance reviewer. Return only the requested JSON."},
        {"role": "user", "content": plan_case["prompt"]}],
        "temperature": 0.0, "max_tokens": MAX_TOKENS,
        "response_format": plan_case["response_format"], "enable_thinking": False}
    response = requests_module.post(BASE_URL + "/chat/completions",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        json=body, timeout=240)
    raw = response.content
    if int(response.status_code) != 200:
        raise RuntimeError(f"provider_http_{response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("provider payload is not an object")
    return payload, raw


def execute(*, plan_path: Path, approval_path: Path, ledger_path: Path, run_dir: Path) -> dict[str, Any]:
    import requests
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw.decode("utf-8"))
    ledger = _load(ledger_path)
    validate_semantic_review_plan(plan, ledger, ledger_file_sha256=_sha_bytes(ledger_path.read_bytes()))
    approval = _load(approval_path)
    _validate_approval(approval, plan, plan_file_sha256=_sha_bytes(plan_raw))
    if plan["model_id"] not in SILICONFLOW_CN_FREE_MODELS:
        raise ValueError("semantic review model is outside the verified CN free set")
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse semantic review run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    state: dict[str, Any] = {"schema": "g09-recall-semantic-review-ledger-v1",
        "plan_sha256": plan["plan_sha256"], "approval_sha256": approval["approval_sha256"],
        "status": "started", "physical_attempts": 0, "request_intents": [], "reviews": []}
    ledger_file = run_dir / "write-ahead-ledger.json"
    def persist_state(*, exclusive: bool = False) -> None:
        state["ledger_sha256"] = sha256_obj({key: value for key, value in state.items()
            if key != "ledger_sha256"})
        _write(ledger_file, state, exclusive=exclusive)
    persist_state(exclusive=True)
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is required only in the execution environment")
    for ordinal, candidate in enumerate(plan["candidates"], 1):
        request = {"ordinal": ordinal, "case_id": candidate["case_id"],
            "model_id": plan["model_id"], "prompt_sha256": candidate["prompt_sha256"],
            "state": "in_flight"}
        state["physical_attempts"] += 1
        state["request_intents"].append(request)
        persist_state()
        try:
            case = {**candidate, "model_id": plan["model_id"], "response_format": plan["response_format"]}
            payload, raw = _request(case, api_key=api_key, requests_module=requests)
            response_file = run_dir / f"response-{ordinal:02d}.json"
            _write(response_file, payload, exclusive=True)
            choices = payload.get("choices") or []
            content = ((choices[0].get("message") or {}).get("content") if choices else None)
            review = parse_semantic_review_output(content, candidate=candidate)
            receipt = {"case_id": candidate["case_id"], "response_file": response_file.name,
                "response_sha256": _sha_bytes(raw), "review": review,
                "usage": payload.get("usage"), "provider_request_id": payload.get("id")}
            (run_dir / f"review-{ordinal:02d}.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            request.update({"state": "succeeded", "receipt_sha256": sha256_obj(receipt)})
            state["reviews"].append(receipt)
            persist_state()
        except Exception as exc:
            request.update({"state": "failed_closed", "error_type": type(exc).__name__, "error": str(exc)[:300]})
            state["status"] = "failed_closed"
            persist_state()
            raise
    state["status"] = "complete_semantic_review_only"
    state["network_calls"] = len(plan["candidates"])
    state["model_calls"] = len(plan["candidates"])
    persist_state()
    report = {"schema": "g09-recall-semantic-review-report-v1", "status": state["status"],
        "plan_sha256": plan["plan_sha256"], "approval_sha256": approval["approval_sha256"],
        "reviews": state["reviews"], "accepted_source_group_count": 0,
        "labels_created": False, "scores_created": False, "cohort_created": False,
        "network_calls": state["network_calls"], "model_calls": state["model_calls"]}
    report["report_sha256"] = sha256_obj(report)
    _write(run_dir / "report.json", report, exclusive=True)
    return {"status": report["status"], "report_sha256": report["report_sha256"],
        "review_count": len(report["reviews"]), "accepted_source_group_count": 0,
        "network_calls": report["network_calls"], "model_calls": report["model_calls"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--write-approval", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.write_approval:
        if args.approval is None:
            raise SystemExit("--approval is required with --write-approval")
        print(json.dumps(make_approval(plan_path=args.plan, approval_path=args.approval), indent=2))
        return 0
    if args.execute:
        if args.approval is None:
            raise SystemExit("--approval is required with --execute")
        print(json.dumps(execute(plan_path=args.plan, approval_path=args.approval,
            ledger_path=args.ledger, run_dir=args.run_dir), indent=2))
        return 0
    raise SystemExit("choose --write-approval or --execute")


if __name__ == "__main__":
    raise SystemExit(main())

