"""Freeze a fresh, label-balanced dev-only cohort for the task-input Skill v2 study.

The script excludes every PubMedQA three-way task ID mentioned in existing
artifacts, then hashes the remaining IDs into disjoint train/discovery/confirm
partitions. It never reads final or selection-gate data and makes no model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASK_ID_RE = re.compile(r"pubmedqa_threeway_probe-[0-9]+")
DEFAULT_SEED = "pubmedqa-v2-skill-routing-20261005"


def _rows() -> list[dict]:
    p = ROOT / "data" / "normalized" / "pubmedqa_threeway_probe.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _previously_mentioned_ids(output_dir: Path) -> set[str]:
    command = ["rg", "-o", "--no-filename", TASK_ID_RE.pattern,
               str(ROOT / "artifacts"), "-g", f"!{output_dir.relative_to(ROOT / 'artifacts').as_posix()}/**"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("could not conservatively scan existing artifacts for used task IDs: "
                           + result.stderr[-1000:])
    dataset_ids = {str(row["task_id"]) for row in _rows()}
    return {value for value in result.stdout.splitlines() if value in dataset_ids}


def _rank(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).hexdigest()


def allocate(rows: list[dict], excluded_ids: set[str], *, seed: str,
             train_per_label: int, discovery_per_label: int,
             confirmation_per_label: int) -> dict:
    groups: dict[str, list[dict]] = {"yes": [], "no": [], "maybe": []}
    for row in rows:
        if row["task_id"] in excluded_ids:
            continue
        label = str(row["options"][int(row["correct_index"])]).strip().casefold()
        if label not in groups:
            raise ValueError(f"unexpected native PubMedQA label: {label}")
        groups[label].append(row)
    requested = train_per_label + discovery_per_label + confirmation_per_label
    if any(len(items) < requested for items in groups.values()):
        raise ValueError("not enough unused tasks in every label for the requested cohort")

    parts = {"train_ids": [], "discovery_ids": [], "confirmation_ids": []}
    by_partition = {"train_ids": train_per_label,
                    "discovery_ids": discovery_per_label,
                    "confirmation_ids": confirmation_per_label}
    for label, items in groups.items():
        ranked = sorted(items, key=lambda row: _rank(seed, str(row["task_id"])))
        cursor = 0
        for partition, count in by_partition.items():
            parts[partition].extend(str(row["task_id"])
                                    for row in ranked[cursor:cursor + count])
            cursor += count
    for ids in parts.values():
        ids.sort()
    flattened = [task_id for ids in parts.values() for task_id in ids]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("cohort partitions overlap")
    return {
        "seed": seed,
        **parts,
        "label_counts": {
            partition: {
                label: sum(str(row["options"][int(row["correct_index"])]).casefold() == label
                           for row in rows if row["task_id"] in set(ids))
                for label in groups
            }
            for partition, ids in parts.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--train-per-label", type=int, default=15)
    parser.add_argument("--discovery-per-label", type=int, default=4)
    parser.add_argument("--confirmation-per-label", type=int, default=3)
    parser.add_argument("--release-id", default="task-input-baseline-v2",
                        help="immutable release used by the frozen cohort")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if min(args.train_per_label, args.discovery_per_label,
           args.confirmation_per_label) < 1:
        raise ValueError("each partition must include at least one task per label")
    output_dir = ROOT / "artifacts" / "sandboxes" / "pubmedqa-threeway-v2-skill-routing"
    output_path = Path(args.output) if args.output else output_dir / "cohort-plan.json"
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    manifest_path = ROOT / "data/manifests/pubmedqa_threeway_probe.json"
    normalized_path = ROOT / "data/normalized/pubmedqa_threeway_probe.jsonl"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (existing.get("dataset_manifest_sha256") != hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()
                or existing.get("dataset_normalized_sha256") != hashlib.sha256(
                    normalized_path.read_bytes()).hexdigest()):
            raise RuntimeError("frozen cohort source data changed; create a new protocol/output")
        print(json.dumps({"output": str(output_path), "status": "already_frozen",
                          "train": len(existing.get("train_ids", [])),
                          "discovery": len(existing.get("discovery_ids", [])),
                          "confirmation": len(existing.get("confirmation_ids", []))},
                         ensure_ascii=False))
        return 0
    excluded = _previously_mentioned_ids(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dev_ids = set(manifest["splits"]["dev_optimize"]["ids"])
    cohort = allocate(
        _rows(), excluded, seed=args.seed,
        train_per_label=args.train_per_label,
        discovery_per_label=args.discovery_per_label,
        confirmation_per_label=args.confirmation_per_label)
    all_ids = set(cohort["train_ids"] + cohort["discovery_ids"]
                  + cohort["confirmation_ids"])
    if not all_ids <= dev_ids:
        raise ValueError("cohort included a task outside dev_optimize")
    from src.util import sha256_file, write_json

    result = {
        "protocol": "pubmedqa-threeway-v2-skill-routing-cohort-v1",
        "dataset": "pubmedqa_threeway_probe",
        "release_id": str(args.release_id),
        "skill_routing_mode": "model",
        "train_only_candidate_generation": True,
        "discovery_accessed": False,
        "confirmation_accessed": False,
        "selection_gate_accessed": False,
        "final_holdout_accessed": False,
        "retrieval_used": False,
        "published": False,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_normalized_sha256": sha256_file(normalized_path),
        "previously_mentioned_task_ids_excluded": len(excluded),
        "unused_candidate_pool_size": len(dev_ids - excluded),
        **cohort,
    }
    write_json(output_path, result)
    print(json.dumps({"output": str(output_path),
                      "previously_mentioned_excluded": len(excluded),
                      "unused_pool": len(dev_ids - excluded),
                      "train": len(cohort["train_ids"]),
                      "discovery": len(cohort["discovery_ids"]),
                      "confirmation": len(cohort["confirmation_ids"]),
                      "label_counts": cohort["label_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
