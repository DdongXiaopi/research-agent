"""Write the source/checksum manifest for the component-split HealthVer pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.healthver_task_input import DATASET, MANIFEST_PATH, load_manifest  # noqa: E402
from src.util import commit_new_files  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="write data/manifests/healthver_task_input.json")
    args = parser.parse_args(argv)
    if args.write:
        from src.data.freeze import require_unfrozen
        require_unfrozen(DATASET)
        if MANIFEST_PATH.exists():
            raise FileExistsError("HealthVer manifest already exists")
    manifest = load_manifest()
    if args.write:
        content = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        commit_new_files([(MANIFEST_PATH, content)])
        result = {"status": "written", "manifest": str(MANIFEST_PATH),
                  "n_tasks": manifest["n_tasks"],
                  "task_ids_sha256": manifest["task_ids_sha256"]}
    else:
        result = {"status": "validated_not_written", "n_tasks": manifest["n_tasks"],
                  "task_ids_sha256": manifest["task_ids_sha256"],
                  "label_counts": manifest["label_counts"]}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
