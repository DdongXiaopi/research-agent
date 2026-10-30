"""四臂实验的配对探索统计与预先固定的晋级规则。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.stats import mcnemar_exact, paired_bootstrap_binary


def _rows(run_id: str) -> dict:
    p = ROOT / "artifacts/runs" / run_id / "per_task.jsonl"
    return {str(r["task_id"]): r for r in
            (json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[30:])}


def main() -> int:
    files = sorted(p for p in (ROOT / "artifacts/probe/real_factorial").glob("factorial-*.json")
                   if not p.stem.endswith("-analysis"))
    if len(files) != 1:
        raise RuntimeError(f"Specify experiment explicitly; found {len(files)}")
    report = json.loads(files[0].read_text(encoding="utf-8"))
    arms = report["full"]
    if set(arms) != {"base", "prompt_only", "skill_only", "joint"}:
        raise RuntimeError(f"not all arms complete: {list(arms)}")
    baseline = arms["base"]["exploratory_after_30"]
    base_rows = _rows(arms["base"]["run_id"])
    comparisons = {}
    eligible = []
    for arm in ("prompt_only", "skill_only", "joint"):
        cand = arms[arm]["exploratory_after_30"]
        rows = _rows(arms[arm]["run_id"])
        ids = sorted(set(base_rows) & set(rows))
        before = [int(bool(base_rows[i]["correct"])) for i in ids]
        after = [int(bool(rows[i]["correct"])) for i in ids]
        deltas = {k: round(100 * (cand[k] - baseline[k]), 4)
                  for k in ("accuracy", "coverage", "answered_precision")}
        def unsupported(sample: dict) -> float:
            no_gold = [r for r in sample.values() if not r.get("gold_in_corpus")]
            return (sum(int(r.get("answer_index", -1) >= 0) for r in no_gold) /
                    len(no_gold)) if no_gold else 0.0
        base_uns = unsupported(base_rows)
        cand_uns = unsupported(rows)
        uns_delta = round(100 * (cand_uns - base_uns), 4)
        passed = (deltas["accuracy"] >= 1.0 and
                  deltas["answered_precision"] >= 0 and
                  deltas["coverage"] >= -1.0 and uns_delta <= 5.0)
        comparisons[arm] = {
            "n": len(ids), "deltas_pp": deltas,
            "unsupported_delta_pp_exploratory_200": uns_delta,
            "mcnemar": mcnemar_exact(before, after),
            "paired_accuracy_bootstrap": paired_bootstrap_binary(
                before, after, B=10000, rng_seed=20260929),
            "advance_rule_passed": passed,
        }
        if passed:
            eligible.append(arm)
    # 规则在联合臂完成前已写入 docs/reliability-reset-2026-09-29.md。
    eligible.sort(key=lambda arm: (-comparisons[arm]["deltas_pp"]["accuracy"],
                                   arm == "joint", arm))
    out = {"experiment_id": report["run_id"],
           "cohort_status": "exploratory_reused_200",
           "selection_rule": "accuracy >= +1pp; precision >= 0pp; coverage >= -1pp; unsupported <= +5pp; max accuracy then single artifact",
           "comparisons": comparisons,
           "preselected_for_new_cohort": eligible[0] if eligible else None}
    path = files[0].with_name(files[0].stem + "-analysis.json")
    with path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
