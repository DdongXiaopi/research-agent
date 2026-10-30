"""Prepare a plan-bound G05 full-abstract reference execution schedule offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.g05_medaesqa_abstract_execution import build_execution_plan  # noqa: E402

DEFAULT_BUNDLE = ROOT / ".workbuddy/g05-medaesqa-calibration-20261017-v5"
DEFAULT_OUTPUT = ROOT / ".workbuddy/g05-medaesqa-reference-execution-plan-only-20261018"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_verified_bundle(bundle: Path) -> dict[str, object]:
    manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("G05 v5 bundle manifest is missing file hashes")
    actual = {name: _sha_bytes((bundle / name).read_bytes()) for name in sorted(files)}
    if actual != files or _sha_bytes(_canonical(actual)) != manifest.get("bundle_sha256"):
        raise ValueError("G05 v5 bundle does not match its frozen manifest")
    required = {"packet.json", "reviewer-a-blind.json", "reviewer-b-blind.json",
        "ai-reference-plan.json", "zero-price-evidence.json"}
    if not required <= set(files):
        raise ValueError("G05 v5 bundle lacks plan-bound execution inputs")
    return manifest


def prepare(bundle: Path = DEFAULT_BUNDLE, output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    bundle, output = bundle.resolve(), output.resolve()
    if ROOT / ".workbuddy" not in bundle.parents or ROOT / ".workbuddy" not in output.parents:
        raise ValueError("bundle and output must be under project .workbuddy")
    if output.exists():
        raise FileExistsError("plan-only output is write-once; choose a new path")
    bundle_manifest = _read_verified_bundle(bundle)
    packet = json.loads((bundle / "packet.json").read_text(encoding="utf-8"))
    reviewer_packets = {slot: json.loads((bundle / f"{slot.replace('_', '-')}-blind.json")
        .read_text(encoding="utf-8")) for slot in ("reviewer_a", "reviewer_b")}
    ref_plan = json.loads((bundle / "ai-reference-plan.json").read_text(encoding="utf-8"))
    price_bytes = (bundle / "zero-price-evidence.json").read_bytes()
    prices = json.loads(price_bytes)
    price_sha = _sha_bytes(price_bytes)
    schedule = build_execution_plan(packet, reviewer_packets, ref_plan,
        zero_price_evidence=prices, zero_price_evidence_sha256=price_sha)
    output.mkdir(parents=True, exist_ok=False)
    payloads = {
        "execution-plan.json": schedule,
        "zero-call-preflight.json": {
            "schema": "g05-v5-execution-zero-call-preflight-v1",
            "status": "passed_zero_call_plan_integrity",
            "source_bundle_sha256": bundle_manifest["bundle_sha256"],
            "execution_plan_sha256": schedule["execution_plan_sha256"],
            "reviewer_call_intents": 180, "adjudication_max_call_intents": 90,
            "physical_attempt_cap": 270, "model_calls": 0,
            "network_calls": 0, "dispatch_authorized": False,
        },
    }
    file_hashes = {}
    for name, value in payloads.items():
        data = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        path = output / name
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
        file_hashes[name] = _sha_bytes(path.read_bytes())
    manifest = {"schema": "g05-v5-execution-plan-only-bundle-v1",
        "source_bundle_sha256": bundle_manifest["bundle_sha256"],
        "files": file_hashes, "bundle_sha256": _sha_bytes(_canonical(file_hashes)),
        "model_calls": 0, "network_calls": 0, "dispatch_authorized": False}
    manifest_path = output / "bundle-manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
    return {"output": str(output), "source_bundle_sha256": bundle_manifest["bundle_sha256"],
        "execution_plan_sha256": schedule["execution_plan_sha256"],
        "bundle_sha256": manifest["bundle_sha256"], "model_calls": 0,
        "network_calls": 0, "dispatch_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
