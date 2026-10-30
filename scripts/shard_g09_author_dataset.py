#!/usr/bin/env python3
"""Write deterministic task-only shards for bounded G09 rubric authoring."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.open_research_author_shards import shard_task_dataset
from src.util import sha256_obj


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=250)
    args = parser.parse_args(argv)
    dataset = json.loads(args.dataset.resolve().read_text(encoding="utf-8"))
    shards = shard_task_dataset(dataset, max_tasks=args.max_tasks)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, shard in enumerate(shards):
        path = out / f"shard-{index:02d}.json"
        path.write_text(json.dumps(shard, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        entries.append({"index": index, "path": path.name,
                        "sha256": sha256_obj(shard),
                        "task_count": sum(len(rows) for rows in shard["partitions"].values()),
                        "groups_by_split": {split: len({case["group_id"] for case in rows})
                                             for split, rows in shard["partitions"].items()}})
    manifest = {"schema": "g09-author-shard-manifest-v1", "source_sha256": sha256_obj(dataset),
                "max_tasks": args.max_tasks, "shard_count": len(entries), "shards": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "shard_count": len(entries),
                      "manifest_sha256": sha256_obj(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
