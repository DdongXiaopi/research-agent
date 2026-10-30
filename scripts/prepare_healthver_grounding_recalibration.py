"""Freeze a source-native recalibration packet with verified history exclusions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.healthver_calibration_cohort import build_packet
from src.eval.healthver_reference import build_reference
from src.util import write_json


def prepare(output_dir: Path, history_paths: list[Path], *, per_label=20, seed=20261016,
            policy="exact_task_disjoint"):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing calibration artifacts: {output_dir}")
    packet = build_packet(history_paths, per_label=per_label, seed=seed, policy=policy)
    reference = build_reference(packet)
    summary = {"schema": "healthver-recalibration-intake-v1",
               "packet_sha256": packet["packet_sha256"],
               "protocol_sha256": packet["protocol_sha256"],
               "reference_sha256": reference["reference_sha256"],
               "case_count": len(packet["cases"]), "model_requests_made": 0,
               "source_label_counts": dict(Counter(
                   row["resolved_verdict"] for row in reference["per_case"])),
               "independent_confirmation": False, "promotion_authorized": False,
               "history_exclusion": packet["protocol_context"]["history_exclusion"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "packet.json", packet)
    write_json(output_dir / "source-reference.json", reference)
    write_json(output_dir / "intake.json", summary)
    return {k: v for k, v in summary.items() if k != "history_exclusion"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--policy", choices=["exact_task_disjoint", "component_disjoint"],
                        default="exact_task_disjoint")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.history, per_label=args.per_label,
                             seed=args.seed, policy=args.policy), indent=2))
