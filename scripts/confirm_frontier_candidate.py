"""从历史迭代产物取前沿成员，在已复用题集上探索性复查。

为什么单独写一个：重跑整个迭代要 25 分钟，而只想再确认一次某个候选。
前提是 `pilot_rewrite_iterative.json` 里已落盘 prompt 原文
（此前只存 sha256，导致想复验就得重跑 —— 已修）。

用法：
    python scripts/confirm_frontier_candidate.py --cand r2c1 --n 230
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config  # noqa: E402
from src.governance import release as rel  # noqa: E402
from src.util import sha256_obj  # noqa: E402

PILOT = ROOT / "artifacts" / "probe" / "pilot_rewrite_iterative.json"
OUT_DIR = ROOT / "artifacts" / "probe" / "exploratory_confirm"


def per_task_full(run_id):
    """⚠️ 必须读文件：内存里的 res.per_task 是简版（无 status）。"""
    p = ROOT / "artifacts" / "runs" / str(run_id) / "per_task.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    cor = sum(1 for r in rows if r.get("correct"))
    abst = sum(1 for r in rows if str(r.get("status", "")).lower() == "abstain")
    answered = n - abst
    return {"n_tasks": n, "accuracy": cor / n, "abstain_rate": abst / n,
            "coverage": answered / n,
            "answered_precision": (cor / answered) if answered else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    ap.add_argument("--pilot", default=str(PILOT))
    ap.add_argument("--cand", default=None, help="不指定则取 accuracy 最高的前沿成员")
    ap.add_argument("--dataset", default="pubmedqa")
    ap.add_argument("--split", default="dev_optimize")
    ap.add_argument("--release", default="baseline-v1-cited")
    ap.add_argument("--scan", type=int, default=30, help="提案样本题数（这些题要排除）")
    ap.add_argument("--n", type=int, default=230)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    raise SystemExit(
        "该历史复查脚本已停用：候选来自在 selection_gate 上迭代选择的旧前沿，"
        "不能作为独立复核或新的候选评分。历史产物仅供审计；请使用当前受控入口，"
        "并先生成新的冻结计划。此入口不会加载配置、任务或 provider，也不会评分。"
    )

    cfg = load_config(args.config)
    state = rel.load_state(args.release)
    pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
    front = pilot.get("frontier") or []
    if not front:
        raise SystemExit("产物里没有前沿成员")

    item = None
    if args.cand:
        item = next((x for x in front if x["id"] == args.cand), None)
        if item is None:
            raise SystemExit(f"没有 id={args.cand}；可选：{[x['id'] for x in front]}")
    else:
        item = max(front, key=lambda x: (x.get("metrics") or {}).get("accuracy") or 0)
    prompt = item.get("prompt") or ""
    if not prompt:
        raise SystemExit(f"{item['id']} 的 prompt 未落盘（旧产物？需重跑迭代）")
    print(f"[cand] {item['id']} chars={len(prompt)}", flush=True)

    from src.eval.harness import evaluate
    from src.governance.release import ArtifactState

    # 提案样本的题 id：用同样的 limit/seed 再跑一次即可复现同一批题
    scan = evaluate(cfg, state, args.dataset, args.split, seed=args.seed,
                    limit=args.scan, run_tag="confirm-scan", write_per_task=True,
                    cfg_fp="confirm")
    scan_ids = {str(r.get("task_id")) for r in per_task_full(scan.run_id)}
    print(f"[scan] 排除 {len(scan_ids)} 题提案样本", flush=True)

    cb = evaluate(cfg, state, args.dataset, args.split, seed=args.seed,
                  limit=args.n, run_tag="confirm-base", write_per_task=True,
                  cfg_fp="confirm")
    cst = ArtifactState(release_id=f"confirm-{item['id']}", system_prompt=prompt,
                        registry=state.registry)
    cc = evaluate(cfg, cst, args.dataset, args.split, seed=args.seed,
                  limit=args.n, run_tag=f"confirm-{item['id']}",
                  write_per_task=True, cfg_fp="confirm")

    b_rows = [r for r in per_task_full(cb.run_id)
              if str(r.get("task_id")) not in scan_ids]
    c_rows = [r for r in per_task_full(cc.run_id)
              if str(r.get("task_id")) not in scan_ids]
    base, cand = summarize(b_rows), summarize(c_rows)
    bm = {str(r.get("task_id")): int(bool(r.get("correct"))) for r in b_rows}
    cm = {str(r.get("task_id")): int(bool(r.get("correct"))) for r in c_rows}
    ids = sorted(set(bm) & set(cm))
    from src.eval.stats import mcnemar_exact
    mc = mcnemar_exact([bm[i] for i in ids], [cm[i] for i in ids]) if ids else {}
    d = {k: (cand.get(k, 0) - base.get(k, 0)) * 100
         for k in ("accuracy", "coverage", "answered_precision", "abstain_rate")}

    print("\n=== 已复用题集：探索性复查 ===", flush=True)
    print("base:", json.dumps(base, ensure_ascii=False), flush=True)
    print("cand:", json.dumps(cand, ensure_ascii=False), flush=True)
    print("Δpp :", json.dumps({k: round(v, 2) for k, v in d.items()},
                              ensure_ascii=False), flush=True)
    print("McNemar p:", mc.get("p_value"), "| n =", len(ids), flush=True)

    out = {"cand_id": item["id"], "candidate_prompt_sha256": sha256_obj(prompt),
           "pilot_file_sha256": sha256_obj(pilot),
           "n_exploratory": len(ids),
           "cohort_status": "exploratory_reused",
           "excluded_proposal_ids": len(scan_ids), "base": base, "cand": cand,
           "delta_pp": d, "mcnemar_p": mc.get("p_value"),
           "iter_metrics": item.get("metrics"),
           "note": "iter_metrics 是迭代集(selection_gate)上的数字，与这里不可比；"
                   "两者差异正是判断有没有过拟合的依据。"}
    out_path = OUT_DIR / f"confirm-{uuid.uuid4().hex[:12]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n产物: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
