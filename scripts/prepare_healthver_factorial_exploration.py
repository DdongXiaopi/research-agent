"""Materialize the frozen exploration-only HealthVer factorial manifest."""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.healthver_factorial_exploration import (  # noqa: E402
    DATASET, MANIFEST_PATH, build_manifest,
)
from src.util import commit_new_files  # noqa: E402


def main() -> int:
    from src.data.freeze import require_unfrozen
    require_unfrozen(DATASET)
    if MANIFEST_PATH.exists():
        raise FileExistsError("HealthVer factorial manifest already exists")
    manifest = build_manifest()
    content = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    commit_new_files([(MANIFEST_PATH, content)])
    print(f"wrote {MANIFEST_PATH.as_posix()} ({manifest['n_questions']} rows; "
          "exploratory only, no final holdout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
