"""Plan, execute, and report the synthetic-only v14 instruction-reviewer calibration."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_synthetic_calibration_v14 as calibration  # noqa: E402
from src.util import artifacts_root, read_json, sha256_file, sha256_obj  # noqa: E402


def _write_new_json(path: Path, value: dict) -> None:
    """Create a JSON artifact exactly once; never truncate a prior run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(path), flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _using_isolated_rsi_home(value: Path):
    home = value.expanduser().resolve()
    if home == (ROOT / "artifacts").resolve():
        raise ValueError("v14 calibration requires an isolated RSI_HOME")
    previous = os.environ.get("RSI_HOME")
    os.environ["RSI_HOME"] = str(home)
    try:
        if artifacts_root().resolve() != home:
            raise ValueError("RSI_HOME did not resolve to the requested isolated directory")
        yield home
    finally:
        if previous is None:
            os.environ.pop("RSI_HOME", None)
        else:
            os.environ["RSI_HOME"] = previous


def _read_json_document(path: Path, name: str) -> dict:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} must be an existing regular JSON file")
    value = read_json(resolved)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _input_bundle(args):
    paths = {
        "packet": args.packet,
        "frozen": args.frozen,
        "schedule_a": args.schedule_a,
        "aggregate_a": args.aggregate_a,
        "schedule_b": args.schedule_b,
        "aggregate_b": args.aggregate_b,
    }
    docs = {name: _read_json_document(path, name) for name, path in paths.items()}
    state_a = args.state_a.expanduser().resolve()
    state_b = args.state_b.expanduser().resolve()
    if not state_a.exists() or not state_b.exists():
        raise ValueError("both reviewer state paths must exist")
    return docs, state_a, state_b


def _add_review_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--schedule-a", type=Path, required=True)
    parser.add_argument("--state-a", type=Path, required=True)
    parser.add_argument("--aggregate-a", type=Path, required=True)
    parser.add_argument("--schedule-b", type=Path, required=True)
    parser.add_argument("--state-b", type=Path, required=True)
    parser.add_argument("--aggregate-b", type=Path, required=True)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--rsi-home", type=Path, required=True,
                        help="isolated artifact root; never defaults to repository artifacts/")


def _call_plan(args, docs, state_a, state_b):
    return calibration.calibration_plan_v14(
        docs["packet"], docs["frozen"], docs["schedule_a"], state_a,
        docs["aggregate_a"], docs["schedule_b"], state_b,
        docs["aggregate_b"], args.target_model_id, workspace=ROOT)


def _write_plan(args) -> int:
    docs, state_a, state_b = _input_bundle(args)
    out = args.output_dir.expanduser().resolve()
    if any(path.resolve() == out / "plan.json" for path in
           (args.packet, args.frozen, args.schedule_a, args.aggregate_a,
            args.schedule_b, args.aggregate_b)):
        raise ValueError("plan output must not overwrite an input document")
    with _using_isolated_rsi_home(args.rsi_home):
        plan = _call_plan(args, docs, state_a, state_b)
        _write_new_json(out / "plan.json", plan)
    print(json.dumps({"action": "plan", "plan": str(out / "plan.json"),
        "plan_sha256": plan["plan_sha256"], "packet_sha256": plan["packet_sha256"],
        "case_count": plan["max_calls"], "provider_calls": 0,
        "synthetic_only": True, "promotion_authorized": False}, ensure_ascii=False))
    return 0


def _execute(args) -> int:
    docs, state_a, state_b = _input_bundle(args)
    out = args.output_dir.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    if plan_path != (out / "plan.json").resolve():
        raise ValueError("execute must use the write-once plan.json in its output directory")
    saved_plan = _read_json_document(plan_path, "plan")
    if args.expected_plan_sha256 != saved_plan.get("plan_sha256"):
        raise ValueError("execute requires the exact frozen plan SHA-256")
    plan_body = {key: value for key, value in saved_plan.items() if key != "plan_sha256"}
    if saved_plan.get("plan_sha256") != sha256_obj(plan_body):
        raise ValueError("saved v14 plan hash is invalid")
    execution_path = out / "execution.json"
    report_path = out / "report.json"
    claim_path = out / "execution.started.json"
    if execution_path.exists() or report_path.exists() or claim_path.exists():
        raise FileExistsError("a consumed v14 calibration plan cannot be replayed")
    with _using_isolated_rsi_home(args.rsi_home):
        current_plan = _call_plan(args, docs, state_a, state_b)
        if current_plan != saved_plan:
            raise ValueError("saved v14 plan differs from current packet/frozen/config bindings")
        _write_new_json(claim_path, {"schema": "g06-v14-cli-execution-claim-v1",
            "plan_sha256": saved_plan["plan_sha256"], "replay_allowed": False})
        result = calibration.calibration_execute_v14(
            docs["packet"], docs["frozen"], docs["schedule_a"], state_a,
            docs["aggregate_a"], docs["schedule_b"], state_b,
            docs["aggregate_b"], args.target_model_id, saved_plan,
            workspace=ROOT)
        envelope = {"schema": "g06-v14-calibration-execution-envelope-v1",
            "status": result.get("measurement_status", "measured_zero_price"),
            "plan_sha256": saved_plan["plan_sha256"],
            "cli_execution_claim_sha256": sha256_file(claim_path),
            "execution_result": result,
            "execution_result_sha256": sha256_obj(result),
            "cli_source_sha256": sha256_file(Path(__file__)),
            "synthetic_only": True, "promotion_authorized": False,
            "production_gate_compatible": False}
        _write_new_json(execution_path, envelope)
    print(json.dumps({"action": "execute", "execution": str(execution_path),
        "plan_sha256": saved_plan["plan_sha256"],
        "case_count": result.get("case_count"),
        "measured_usd": result.get("ai_job_receipt", {}).get("measured_usd"),
        "synthetic_only": True, "promotion_authorized": False}, ensure_ascii=False))
    return 0


def _report(args) -> int:
    execution = _read_json_document(args.execution, "execution")
    if (execution.get("schema") != "g06-v14-calibration-execution-envelope-v1"
            or execution.get("synthetic_only") is not True
            or execution.get("promotion_authorized") is not False
            or execution.get("execution_result_sha256")
                != sha256_obj(execution.get("execution_result"))):
        raise ValueError("report requires an intact synthetic v14 execution envelope")
    result = execution["execution_result"]
    receipt = result.get("ai_job_receipt", {})
    accounting = receipt.get("request_accounting", {})
    report = {
        "schema": "g06-v14-synthetic-diagnostic-report-v1",
        "scope": "synthetic_diagnostic_only",
        "status": execution.get("status"),
        "packet_sha256": result.get("packet_sha256"),
        "frozen_reference_sha256": result.get("frozen_reference_sha256"),
        "target_protocol_version": result.get("target_protocol_version"),
        "target_protocol_sha256": result.get("target_protocol_sha256"),
        "reference_protocol_hash": result.get("reference_protocol_hash"),
        "target_model_id": result.get("target_model_id"),
        "case_count": result.get("case_count"),
        "dimensions": result.get("dimensions", {}),
        "ledger_summary": {
            "call_count": receipt.get("call_count"),
            "measured_usd": receipt.get("measured_usd"),
            "usage_complete": receipt.get("usage_complete"),
            "logical_operations": accounting.get("logical_operations"),
            "physical_http_attempts": accounting.get("physical_http_attempts"),
            "physical_status_counts": accounting.get("physical_status_counts"),
            "physical_attempts_exact": accounting.get("physical_attempts_exact"),
        },
        "execution_sha256": sha256_obj(execution),
        "synthetic_only": True,
        "natural_distribution": False,
        "production_gate_compatible": False,
        "candidate_gate_eligible": False,
        "promotion_authorized": False,
    }
    output = args.output.expanduser().resolve()
    if output == args.execution.expanduser().resolve():
        raise ValueError("report output must not overwrite its execution input")
    _write_new_json(output, report)
    print(json.dumps({"action": "report", "report": str(output),
        "case_count": report["case_count"], "scope": report["scope"],
        "promotion_authorized": False}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="validate bindings and freeze a zero-call plan")
    _add_review_inputs(plan)
    plan.add_argument("--output-dir", type=Path, required=True)
    execute = sub.add_parser("execute", help="revalidate and execute the exact frozen plan")
    _add_review_inputs(execute)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--expected-plan-sha256", required=True)
    report = sub.add_parser("report", help="write a synthetic-only metrics and ledger summary")
    report.add_argument("--execution", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "plan":
        return _write_plan(args)
    if args.action == "execute":
        return _execute(args)
    return _report(args)


if __name__ == "__main__":
    raise SystemExit(main())
