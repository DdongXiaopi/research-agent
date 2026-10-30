"""Create a local-only G05 MedAESQA plan and no-dispatch preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_medaesqa_adapter import (
    DATASET_ID,
    build_medaesqa_calibration_bundle,
    build_medaesqa_zero_call_plan,
)  # noqa: E402
from src.eval.grounding_selected_answer_calibration import (
    validate_selected_answer_packet,
    validate_selected_answer_reference,
)  # noqa: E402


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def build_bundle(source_path: Path, history_path: Path, protocol_path: Path,
                 output_path: Path, *, sample_seed: str, model_id: str,
                 created_at: str) -> dict[str, object]:
    root = Path.cwd().resolve()
    output = output_path.resolve()
    workbuddy = (root / ".workbuddy").resolve()
    if workbuddy not in output.parents:
        raise ValueError("output directory must be a fresh child of the workspace .workbuddy directory")
    if output.exists():
        raise FileExistsError("output directory must be new; existing evidence is immutable")
    source_bytes = source_path.read_bytes()
    history_report = json.loads(history_path.read_text(encoding="utf-8"))
    protocol_sha256 = _sha_bytes(protocol_path.read_bytes())
    packet, reference, audit = build_medaesqa_calibration_bundle(
        source_bytes,
        history_report=history_report,
        protocol_sha256=protocol_sha256,
        sample_seed=sample_seed,
    )
    plan, preflight = build_medaesqa_zero_call_plan(
        packet, reference, model_id=model_id, created_at=created_at)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "packet.json", packet)
    _write_json(output / "reference.json", reference)
    _write_json(output / "adapter-audit.json", audit)
    _write_json(output / "target-plan.json", plan)
    _write_json(output / "zero-call-preflight.json", preflight)
    files = {}
    for path in sorted(output.iterdir()):
        files[path.name] = _sha_bytes(path.read_bytes())
    manifest = {
        "schema": "g05-medaesqa-zero-call-bundle-manifest-v1",
        "files": files,
        "bundle_sha256": hashlib.sha256(_canonical(files).encode("utf-8")).hexdigest(),
        "model_calls": 0,
        "target_scorer_calls": 0,
        "dispatch_authorized": False,
    }
    _write_json(output / "manifest.json", manifest)
    return {"output": str(output), "packet_sha256": packet["packet_sha256"],
            "reference_sha256": reference["reference_sha256"],
            "plan_sha256": plan["plan_sha256"], "bundle_sha256": manifest["bundle_sha256"],
            "case_count": len(packet["cases"]), "model_calls": 0,
            "target_scorer_calls": 0, "dispatch_authorized": False}


def preflight_bundle(bundle_path: Path) -> dict[str, object]:
    """Reload and verify a frozen bundle without credentials or network access."""
    root = Path.cwd().resolve()
    bundle = bundle_path.resolve()
    workbuddy = (root / ".workbuddy").resolve()
    if workbuddy not in bundle.parents:
        raise ValueError("bundle must be below the workspace .workbuddy directory")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    expected_files = {"packet.json", "reference.json", "adapter-audit.json",
                      "target-plan.json", "zero-call-preflight.json"}
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "g05-medaesqa-zero-call-bundle-manifest-v1"
            or set(manifest.get("files", {})) != expected_files):
        raise ValueError("G05 bundle manifest schema or inventory mismatch")
    actual_files = {name: _sha_bytes((bundle / name).read_bytes())
                    for name in sorted(expected_files)}
    if actual_files != manifest["files"]:
        raise ValueError("G05 bundle file hash mismatch")
    bundle_sha256 = hashlib.sha256(_canonical(actual_files).encode("utf-8")).hexdigest()
    if (manifest.get("bundle_sha256") != bundle_sha256
            or manifest.get("model_calls") != 0
            or manifest.get("target_scorer_calls") != 0
            or manifest.get("dispatch_authorized") is not False):
        raise ValueError("G05 bundle manifest call-safety or hash mismatch")
    packet = json.loads((bundle / "packet.json").read_text(encoding="utf-8"))
    reference = json.loads((bundle / "reference.json").read_text(encoding="utf-8"))
    plan = json.loads((bundle / "target-plan.json").read_text(encoding="utf-8"))
    preflight = json.loads((bundle / "zero-call-preflight.json").read_text(encoding="utf-8"))
    validate_selected_answer_packet(packet)
    validate_selected_answer_reference(packet, reference)
    expected_plan, expected_preflight = build_medaesqa_zero_call_plan(
        packet, reference, model_id=plan.get("model_id"),
        created_at=plan.get("created_at"))
    if plan != expected_plan or preflight != expected_preflight:
        raise ValueError("G05 plan/preflight is stale or altered")
    if packet.get("dataset") != DATASET_ID:
        raise ValueError("unexpected G05 calibration dataset")
    return {
        "schema": "g05-medaesqa-reload-preflight-v1",
        "status": "passed_zero_call_plan_integrity",
        "bundle_path": str(bundle),
        "bundle_sha256": bundle_sha256,
        "packet_sha256": packet["packet_sha256"],
        "reference_sha256": reference["reference_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "case_count": len(packet["cases"]),
        "request_count": plan["request_count"],
        "model_calls": 0,
        "target_scorer_calls": 0,
        "dispatch_authorized": False,
        "cost_usd": "0.00",
        "upstream_data_license_status": "unknown_unverified_osf_original",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-bundle", type=Path,
                        help="reload and verify an existing plan bundle without dispatch")
    parser.add_argument("--preflight-output", type=Path,
                        help="optional new .workbuddy path for the reload-preflight report")
    parser.add_argument("--source", type=Path,
                        help="pinned local MedAESQA JSON snapshot")
    parser.add_argument("--history-report", type=Path,
                        help="hash-matched exact-history scan report")
    parser.add_argument("--protocol", type=Path,
                        help="frozen G05 MedAESQA calibration protocol document")
    parser.add_argument("--output", type=Path,
                        help="new output directory below workspace .workbuddy")
    parser.add_argument("--sample-seed")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B",
                        choices=("Qwen/Qwen3-8B", "XingChenAGI/Xing4.0-29B"))
    parser.add_argument("--created-at")
    args = parser.parse_args(argv)
    if args.preflight_bundle:
        report = preflight_bundle(args.preflight_bundle)
        if args.preflight_output:
            output = args.preflight_output.resolve()
            if (ROOT / ".workbuddy").resolve() not in output.parents:
                raise ValueError("preflight report must stay below project .workbuddy")
            if output.exists():
                raise FileExistsError("refusing to overwrite an existing preflight report")
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(output, report)
            report["preflight_report_path"] = str(output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    required = {"source": args.source, "history-report": args.history_report,
                "protocol": args.protocol, "output": args.output,
                "sample-seed": args.sample_seed, "created-at": args.created_at}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("plan mode requires: " + ", ".join("--" + name for name in missing))
    summary = build_bundle(
        args.source, args.history_report, args.protocol, args.output,
        sample_seed=args.sample_seed, model_id=args.model_id,
        created_at=args.created_at,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
