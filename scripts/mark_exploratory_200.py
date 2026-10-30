"""给重复查看的 PubMedQA 200 题生成不可覆盖的探索集清单。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.util import sha256_obj

OUT = ROOT / "artifacts/probe/exploratory_pubmedqa_200.json"


def main() -> None:
    path = ROOT / "artifacts/runs/run-4eb02d4643130dc3/per_task.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    ids = [str(r["task_id"]) for r in rows[30:230]]
    if len(ids) != 200 or len(set(ids)) != 200:
        raise RuntimeError("historical 200 task IDs are not unique")
    payload = {
        "status": "exploratory_reused",
        "dataset": "pubmedqa", "split": "dev_optimize",
        "n": 200, "task_ids_sha256": sha256_obj(ids),
        "task_ids": ids,
        "excluded_proposal_prefix": 30,
        "historical_base_run": "run-4eb02d4643130dc3",
        "historical_r2c0_run": "run-4e3862bf60e0e394",
        "reason": "Repeated r1c0/r2c1/r2c0 confirmation and earlier pilot selection exposed part of this cohort; it is no longer independent.",
        "interpretation": "No confirmatory p-value or clean-set claim may be made from this cohort.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(OUT)


if __name__ == "__main__":
    main()
