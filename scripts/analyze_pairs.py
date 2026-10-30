"""新基线的题级配对检验（baseline-v0 vs 进化后 release）。

协议升级后准确率普遍下降（多了弃权出口），单看 accuracy 会得出"进化变差了"。
这里用**题级配对 bootstrap**给出 Δ / 95% CI / P(Δ>0)，并同时配对三个指标：

    accuracy            全体题（弃权算错，官方口径）
    answered_precision  只在两个 run 都作答的题上比（看"答得准不准"）
    coverage            作答率（看"敢不敢答"）

用法：
    python scripts/analyze_pairs.py artifacts/real_probe/new_baseline_<ts>.json
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

sys.path.insert(0, ".")

from src.eval.final_eval import pair_seed  # noqa: E402
from src.util import artifacts_path, read_jsonl, write_json  # noqa: E402


def _load(run_id: str) -> Dict[str, Dict[str, Any]]:
    p = artifacts_path("runs", run_id, "per_task.jsonl")
    return {r["task_id"]: r for r in read_jsonl(p)}


def _pair(base: Dict[str, Dict[str, Any]], cur: Dict[str, Dict[str, Any]],
          key: str) -> Dict[str, Any]:
    ids = sorted(set(base) & set(cur))
    b = [{"task_id": i, "correct": bool(base[i][key])} for i in ids]
    c = [{"task_id": i, "correct": bool(cur[i][key])} for i in ids]
    return pair_seed(b, c)


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if not src:
        print("usage: python scripts/analyze_pairs.py <new_baseline_*.json>")
        return 1
    d = json.load(open(src, encoding="utf-8"))
    rids = list(d["releases"].keys())
    if len(rids) < 2:
        print("需要至少两个 release 才能配对")
        return 1
    base_id, cur_id = rids[0], rids[1]
    out: Dict[str, Any] = {
        "source": src, "baseline": base_id, "current": cur_id,
        "corpus_version": d.get("corpus_version"),
        "insufficient_option": d.get("insufficient_option"),
        "pairs": {},
    }
    for name in d["releases"][base_id]["results"]:
        ds, split = name.split("/")
        br = d["releases"][base_id]["results"][name]["run_id"]
        cr = d["releases"][cur_id]["results"][name]["run_id"]
        base, cur = _load(br), _load(cr)
        ids = sorted(set(base) & set(cur))
        # accuracy：全体题（弃权 = 错）
        acc = _pair(base, cur, "correct")
        # precision：只在两边都作答的题上比
        both = [i for i in ids if int(base[i].get("answer_index", -1)) >= 0
                and int(cur[i].get("answer_index", -1)) >= 0]
        prec = pair_seed(
            [{"task_id": i, "correct": bool(base[i]["correct"])} for i in both],
            [{"task_id": i, "correct": bool(cur[i]["correct"])} for i in both])
        base_cov = sum(1 for i in ids if int(base[i].get("answer_index", -1)) >= 0) / len(ids)
        cur_cov = sum(1 for i in ids if int(cur[i].get("answer_index", -1)) >= 0) / len(ids)
        out["pairs"][name] = {
            "n_paired": len(ids),
            "accuracy": {"base": round(sum(1 for i in ids if base[i]["correct"]) / len(ids), 4),
                         "cur": round(sum(1 for i in ids if cur[i]["correct"]) / len(ids), 4),
                         "delta_pp": acc["delta_pp"], "ci95": acc["ci95"],
                         "p_delta_gt_0": acc["p_delta_gt_0"]},
            "precision_on_both_answered": {
                "n": len(both),
                "base": round(sum(1 for i in both if base[i]["correct"]) / len(both), 4) if both else None,
                "cur": round(sum(1 for i in both if cur[i]["correct"]) / len(both), 4) if both else None,
                "delta_pp": prec["delta_pp"], "ci95": prec["ci95"],
                "p_delta_gt_0": prec["p_delta_gt_0"]},
            "coverage": {"base": round(base_cov, 4), "cur": round(cur_cov, 4),
                         "delta_pp": round((cur_cov - base_cov) * 100, 4)},
        }
        print(f"{name}: {out['pairs'][name]}", flush=True)
    p = artifacts_path("real_probe", "paired_new_baseline.json")
    write_json(p, out)
    print(f"\n>>> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
