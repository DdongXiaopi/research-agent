"""Create a metadata-only audit manifest for a failed G06 reference job."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.instruction_reviewer_partial_recovery import (  # noqa: E402
    audit_partial_reference_run,
)
from src.util import read_json, sha256_file  # noqa: E402


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_new_manifest(path: Path, manifest: dict, *, run_dir: Path) -> str:
    path = path.resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("audit output must be outside the immutable failed-job directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    # Exclusive creation makes reruns non-destructive and prevents silent overwrite.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
    return sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-plan", type=Path, required=True,
                        help="frozen top-level AI reference plan JSON")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="failed ai_jobs directory (read-only)")
    parser.add_argument("--slot", choices=("reviewer_a", "reviewer_b"), required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new manifest path outside --run-dir; must not already exist")
    args = parser.parse_args(argv)
    cohort_plan_path = _resolve(args.cohort_plan)
    run_dir = _resolve(args.run_dir)
    output = _resolve(args.output)
    cohort_plan = read_json(cohort_plan_path)
    # inputs.json contains the blind packet passed to ai_jobs. It is read only;
    # neither candidate text nor reference labels are emitted by this command.
    inputs = read_json(run_dir / "inputs.json")
    packet = inputs.get("packet") if isinstance(inputs, dict) else None
    if not isinstance(packet, dict):
        raise ValueError("failed-job inputs do not contain a reference packet")
    manifest = audit_partial_reference_run(packet, cohort_plan, args.slot, run_dir)
    digest = _write_new_manifest(output, manifest, run_dir=run_dir)
    print(json.dumps({"status": manifest["status"], "manifest_sha256": digest,
        "output": str(output), "case_state_counts": manifest["case_state_counts"],
        "mergeable": manifest["mergeable"], "model_calls": 0},
        sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
