"""Plan-bound, one-shot runner for the G05 v5 full-abstract AI references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import g05_medaesqa_abstract_execution as execution  # noqa: E402
from src.eval import g05_medaesqa_abstract_runner as runner  # noqa: E402

SOURCE_BUNDLE = ROOT / ".workbuddy/g05-medaesqa-calibration-20261017-v5"
EXECUTION_BUNDLE = ROOT / ".workbuddy/g05-medaesqa-reference-execution-plan-only-20261018"


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_verified_json_bundle(path: Path, manifest_name: str) -> tuple[dict, dict]:
    manifest = json.loads((path / manifest_name).read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("G05 bundle manifest has no files")
    actual = {name: _sha_bytes((path / name).read_bytes()) for name in sorted(files)}
    if actual != files:
        raise ValueError("G05 bundle file hash mismatch")
    canonical = json.dumps(actual, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    if _sha_bytes(canonical) != manifest.get("bundle_sha256"):
        raise ValueError("G05 bundle aggregate hash mismatch")
    return manifest, {name: json.loads((path / name).read_text(encoding="utf-8"))
                      for name in files if name.endswith(".json")}


def load_inputs() -> tuple[dict, dict, dict, dict, str, dict]:
    source_manifest, source = _load_verified_json_bundle(SOURCE_BUNDLE, "bundle-manifest.json")
    execution_manifest, prepared = _load_verified_json_bundle(EXECUTION_BUNDLE, "bundle-manifest.json")
    packet = source["packet.json"]
    reviewers = {slot: source[f"{slot.replace('_', '-')}-blind.json"]
                 for slot in ("reviewer_a", "reviewer_b")}
    ref_plan = source["ai-reference-plan.json"]
    prices = source["zero-price-evidence.json"]
    price_sha = source_manifest["files"]["zero-price-evidence.json"]
    plan = prepared["execution-plan.json"]
    if execution_manifest.get("source_bundle_sha256") != source_manifest.get("bundle_sha256"):
        raise ValueError("execution schedule is bound to a different v5 source bundle")
    execution.validate_execution_plan(packet, reviewers, ref_plan,
        zero_price_evidence=prices, zero_price_evidence_sha256=price_sha,
        execution_plan=plan)
    return packet, reviewers, ref_plan, prices, price_sha, plan


def preflight(role: str | None = None) -> dict[str, object]:
    packet, reviewers, ref_plan, prices, price_sha, plan = load_inputs()
    return {"schema": "g05-v5-reference-runner-preflight-v1",
        "status": "passed_zero_call_plan_integrity",
        "role": role, "packet_sha256": packet["packet_sha256"],
        "reference_plan_sha256": ref_plan["plan_sha256"],
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "source_bundle_sha256": json.loads((SOURCE_BUNDLE / "bundle-manifest.json").read_text(
            encoding="utf-8"))["bundle_sha256"],
        "source_tree_sha256": runner.source_tree_fingerprint(),
        "model_calls": 0, "network_calls": 0, "dispatch_authorized": False,
        "role_call_caps": plan["planned_calls"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="validate all frozen v5 inputs; no calls")
    parser.add_argument("--execute", action="store_true", help="run one role after external root approval")
    parser.add_argument("--role", choices=("reviewer_a", "reviewer_b", "adjudicator"))
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-bundle", type=Path,
        help="frozen source bundle directory (defaults to the v5 bundle)")
    parser.add_argument("--execution-bundle", type=Path,
        help="write-once execution plan bundle directory")
    parser.add_argument("--reviewer-a-receipts", type=Path)
    parser.add_argument("--reviewer-b-receipts", type=Path)
    args = parser.parse_args()
    global SOURCE_BUNDLE, EXECUTION_BUNDLE
    if args.source_bundle is not None:
        SOURCE_BUNDLE = args.source_bundle.resolve()
    if args.execution_bundle is not None:
        EXECUTION_BUNDLE = args.execution_bundle.resolve()
    if args.execute and args.preflight:
        parser.error("choose either --preflight or --execute")
    if not args.execute:
        print(json.dumps(preflight(args.role), ensure_ascii=False, indent=2))
        return 0
    if not args.role or not args.approval or not args.run_dir:
        parser.error("--execute requires --role, --approval and --run-dir")
    packet, reviewers, ref_plan, prices, price_sha, plan = load_inputs()
    approval = json.loads(args.approval.read_text(encoding="utf-8"))
    reviewer_receipts = None
    if args.role == "adjudicator":
        if not args.reviewer_a_receipts or not args.reviewer_b_receipts:
            parser.error("adjudicator requires both reviewer receipt bundles")
        reviewer_receipts = {
            "reviewer_a": json.loads(args.reviewer_a_receipts.read_text(encoding="utf-8"))["receipts"],
            "reviewer_b": json.loads(args.reviewer_b_receipts.read_text(encoding="utf-8"))["receipts"],
        }
    result = runner.dispatch_role(packet=packet, reviewer_packets=reviewers,
        reference_plan=ref_plan, execution_plan=plan,
        zero_price_evidence=prices, zero_price_evidence_sha256=price_sha,
        role=args.role, approval=approval, run_dir=args.run_dir,
        reviewer_receipts=reviewer_receipts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
