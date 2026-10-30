"""历史整段改写候选探索性复查；默认 gate 仅返回聚合。

为什么需要这一步：pilot 的失败样本来自 dev_optimize 前 30 题，而评测跑的是
前 120 题 —— 实测两者**重叠 30/120（25%）**，所以那 +5.24pp 可能含过拟合成分。

`selection_gate` 与 `dev_optimize` 是**不同 split**，pilot 的提案过程从未看到它
⇒ 该集已经被反复查看，不能再作为独立 holdout 确认。

用法：
  python scripts/verify_rewrite_holdout.py --eval 190
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

PILOT = ROOT / "artifacts" / "probe" / "pilot_rewrite_fewshot.json"
OUT_DIR = ROOT / "artifacts" / "probe" / "rewrite_recheck"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    ap.add_argument("--pilot", default=str(PILOT))
    ap.add_argument("--dataset", default="pubmedqa")
    ap.add_argument("--split", default="selection_gate")
    ap.add_argument("--release", default="baseline-v1-cited")
    ap.add_argument("--eval", type=int, default=190, help="0=跑满该 split")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--cand-idx", type=int, default=None, help="默认取 pilot 里最优候选")
    args = ap.parse_args()

    raise SystemExit(
        "历史改写候选复查已停用：候选来自缺少静态门的旧 few-shot pilot，且默认在已复用的"
        " selection_gate 上评分。历史结果仅供审计；请使用当前受控的固定候选流程。"
        "此入口不会加载配置、历史候选或评测数据。"
    )

    cfg = load_config(args.config)
    state = rel.load_state(args.release)
    pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
    cands = pilot.get("candidates") or []
    if not cands:
        raise SystemExit("pilot 产物里没有候选，先跑 pilot_rewrite_fewshot.py")

    if args.cand_idx is not None:
        cand = [c for c in cands if c["idx"] == args.cand_idx]
        if not cand:
            raise SystemExit(f"pilot 产物里没有 idx={args.cand_idx}")
        cand = cand[0]
    else:
        best = pilot.get("best") or {}
        idx = best.get("idx", cands[0]["idx"])
        cand = [c for c in cands if c["idx"] == idx][0]
    print(f"[cand] idx={cand['idx']} chars={len(cand['prompt'])} "
          f"demos={cand.get('demos_used')}", flush=True)

    limit = args.eval if args.eval > 0 else None
    from src.eval.harness import evaluate
    from src.eval.verdict import pair_triple
    from src.governance.release import ArtifactState

    b = evaluate(cfg, state, args.dataset, args.split, seed=args.seed,
                 limit=limit, run_tag="holdout-base",
                 write_per_task=(args.split != "selection_gate"),
                 cfg_fp="holdout")
    cstate = ArtifactState(release_id=f"holdout-cand-{cand['idx']}",
                           system_prompt=cand["prompt"], registry=state.registry)
    c = evaluate(cfg, cstate, args.dataset, args.split, seed=args.seed,
                 limit=limit, run_tag=f"holdout-cand{cand['idx']}",
                 write_per_task=(args.split != "selection_gate"), cfg_fp="holdout")

    bm, cm = b.metric_values, c.metric_values
    vd = pair_triple(bm, cm, cfg)
    for tag, m in (("base", bm), ("cand", cm)):
        print(f"[{tag}] n={m.get('n_tasks')} acc={m.get('accuracy'):.4f} "
              f"prec={m.get('answered_precision'):.4f} cov={m.get('coverage'):.4f} "
              f"abstain={m.get('abstain_rate'):.4f} "
              f"unsupported={m.get('unsupported_answer_rate')}", flush=True)
    print(f"[Δ] {json.dumps(vd.get('deltas_pp'), ensure_ascii=False)} "
          f"verdict={vd.get('verdict')}", flush=True)

    # 题级配对：McNemar + 配对 bootstrap CI
    def correct_map(res):
        return {str(r.get("task_id")): int(bool(r.get("correct")))
                for r in (getattr(res, "per_task", None) or [])}
    bmap, cmap = correct_map(b), correct_map(c)
    ids = sorted(set(bmap) & set(cmap))
    before = [bmap[i] for i in ids]
    after = [cmap[i] for i in ids]
    stat = {}
    if ids:
        from src.eval.stats import mcnemar_exact, paired_bootstrap_binary
        stat["mcnemar"] = mcnemar_exact(before, after)
        stat["paired_ci"] = paired_bootstrap_binary(before, after, B=10000,
                                                    rng_seed=args.seed)
        print(f"[paired] n={len(ids)} mcnemar_p={stat['mcnemar'].get('p_value')} "
              f"delta_pp={stat['paired_ci'].get('delta_pp')} "
              f"ci={stat['paired_ci'].get('ci')}", flush=True)

    out = {"purpose": "历史候选探索性复查",
           "cohort_status": "exploratory_reused",
           "split": args.split, "dataset": args.dataset, "n": len(ids),
           "cand_idx": cand["idx"], "cand_chars": len(cand["prompt"]),
           "base_metrics": {k: bm.get(k) for k in ("accuracy", "coverage",
                                                   "answered_precision", "abstain_rate",
                                                   "unsupported_answer_rate")},
           "cand_metrics": {k: cm.get(k) for k in ("accuracy", "coverage",
                                                   "answered_precision", "abstain_rate",
                                                   "unsupported_answer_rate")},
           "deltas_pp": vd.get("deltas_pp"), "verdict": vd.get("verdict"),
           "primary_axis": vd.get("primary_axis"),
           "primary_delta_pp": vd.get("primary_delta_pp"),
           "stats": stat,
           "dev_pilot_delta_pp": (pilot.get("best") or {}).get("primary_delta_pp"),
           "note": ("dev 上的 Δ 与 holdout 上的 Δ 若接近 ⇒ 不是过拟合；"
                    "若 holdout 显著小于 dev ⇒ dev 那个数含过拟合成分。")}
    out_path = OUT_DIR / f"recheck-{uuid.uuid4().hex[:12]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n产物: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
