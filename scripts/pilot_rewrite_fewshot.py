"""Pilot：GEPA 式整段改写 + few-shot，能不能造出比"追加一句"更好的候选？

背景：本项目的 prompt 候选一直是 `old + 一句指令`（追加形态），候选空间约 2^4。
对照 vendor（gepa / dspy MIPROv2）后补齐两件事：
  1. 产出形态改为**整段改写**（`src/evolution/prompt_rewrite.py`）；
  2. few-shot 只引导提案模型，**不会进入答题执行**；真实执行见
     `scripts/run_real_candidate_factorial.py`。

本脚本先在**进化可见区**（dev_optimize）上验证，不碰 final_holdout，
**不消耗揭盲预算**。

用法：
  # 第一步（便宜）：只验证提案器能产出合格候选
  python scripts/pilot_rewrite_fewshot.py --scan 30 --propose 3

  # 第二步（贵）：把候选拉去评测，与起点做同批次配对比较
  python scripts/pilot_rewrite_fewshot.py --scan 30 --propose 3 --eval 120
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

OUT_DIR = ROOT / "artifacts" / "probe" / "fewshot_pilot"

NOISE_FLOOR_PP = {"pubmedqa": 1.6, "litqa2": 4.3, "protocolqa": 3.3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    ap.add_argument("--dataset", default="pubmedqa")
    ap.add_argument("--split", default="dev_optimize")
    ap.add_argument("--release", default="baseline-v1-cited")
    ap.add_argument("--scan", type=int, default=30, help="扫描多少题以取得失败样本与 demos")
    ap.add_argument("--propose", type=int, default=3, help="产出几个候选")
    ap.add_argument("--eval", type=int, default=0, help=">0 时把候选拉去评测（同题数）")
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    raise SystemExit(
        "该历史探索脚本已停用：它对生成的候选直接调用任务评测，缺少 FR-203 静态门"
        "及当前冻结计划/预算护栏。历史产物仅供审计；请改用受控的 task-input 搜索入口，"
        "并先生成新的冻结计划。此入口不会加载配置、任务或 provider，也不会评分。"
    )
    if args.split != "dev_optimize":
        raise SystemExit("few-shot 提案与逐题反馈只允许 dev_optimize")
    experiment_id = "fewshot-" + uuid.uuid4().hex[:12]

    cfg = load_config(args.config)
    state = rel.load_state(args.release)

    # 1) 跑起点，拿失败样本 + demos
    from src.eval.harness import evaluate
    from src.evolution.fewshot import bootstrap_demos, render_demos_block

    print(f"[scan] {args.dataset}/{args.split} n={args.scan} release={args.release}",
          flush=True)
    base = evaluate(cfg, state, args.dataset, args.split, seed=args.seed,
                    limit=args.scan, run_tag="pilot-scan", write_per_task=True,
                    cfg_fp="pilot")
    bm = base.metric_values
    print(f"[scan] acc={bm.get('accuracy'):.4f} prec={bm.get('answered_precision'):.4f} "
          f"cov={bm.get('coverage'):.4f} abstain={bm.get('abstain_rate'):.4f}", flush=True)

    demos = bootstrap_demos(cfg, state, args.dataset, args.split, k=3,
                            seed=args.seed, limit=args.scan,
                            run_tag="pilot-fewshot")
    demos_block = render_demos_block(demos)
    print(f"[fewshot] demos={len(demos)} block_chars={len(demos_block)}", flush=True)

    # 2) 失败样本（带反馈）
    from src.data.split import load_tasks
    tasks = {str(t.get("task_id")): t for t in load_tasks(args.dataset, [args.split])}
    failure_rows = []
    for row in (getattr(base, "per_task", None) or []):
        if row.get("correct"):
            continue
        t = tasks.get(str(row.get("task_id")), {})
        failure_rows.append({
            "question": t.get("question", ""),
            "options": t.get("options", []),
            "correct_index": row.get("gold_index", t.get("correct_index")),
            "answer_index": row.get("answer_index"),
            "feedback": f"attributed={row.get('attributed_class', 'n/a')}",
        })
    print(f"[failures] n={len(failure_rows)}", flush=True)

    # 3) 整段改写，产出多个候选
    from src.evolution.reflection_backend import RewriteBackend
    backend = RewriteBackend(cfg=cfg, max_attempts=2, demos=demos_block)
    cands = []
    seen = set()
    for i in range(args.propose):
        patch = backend.propose("plan.failed", state.system_prompt, failure_rows,
                                cfg, state=state)
        if patch is None or not getattr(patch, "full_prompt", ""):
            print(f"[propose {i}] 未产出合格候选：{getattr(backend,'last_error',None)}",
                  flush=True)
            continue
        new = patch.full_prompt
        if new in seen:
            print(f"[propose {i}] 与已有候选重复，跳过", flush=True)
            continue
        seen.add(new)
        cands.append({"idx": i, "global_id": f"{experiment_id}:{i}:{sha256_obj(new)[:12]}",
                      "prompt_sha256": sha256_obj(new),
                      "prompt": new, "rationale": patch.rationale,
                      "demos_used": patch.demos_used})
        print(f"[propose {i}] ok, chars={len(new)} demos={patch.demos_used}", flush=True)

    out = {"experiment_id": experiment_id, "cohort_status": "exploratory",
           "fewshot_execution": "proposal_only; demos not embedded in answer prompt",
           "config": args.config, "config_sha256": sha256_obj(cfg),
           "dataset": args.dataset, "split": args.split,
           "release": args.release, "scan_n": args.scan,
           "start_metrics": {k: bm.get(k) for k in
                             ("accuracy", "coverage", "answered_precision",
                              "abstain_rate", "unsupported_answer_rate")},
           "fewshot": {"n_demos": len(demos), "block_chars": len(demos_block)},
           "n_failures": len(failure_rows),
           "candidates": cands}

    # 4) 评测（可选）
    if args.eval > 0 and cands:
        # ⚠️ 起点必须用与候选**同样的题数**重跑一遍。
        # 第一版直接拿 --scan(30 题) 的指标去比 --eval(120 题) 的候选，
        # 两边不是同一批题 ⇒ 那个 Δ 无效（曾据此报出 +13.6pp，已作废）。
        from src.eval.verdict import pair_triple
        from src.governance.release import ArtifactState

        base_e = evaluate(cfg, state, args.dataset, args.split, seed=args.seed,
                          limit=args.eval, run_tag="pilot-base-eval",
                          write_per_task=True, cfg_fp="pilot")
        bm_e = base_e.metric_values
        print(f"[base-eval] n={args.eval} acc={bm_e.get('accuracy'):.4f} "
              f"prec={bm_e.get('answered_precision'):.4f} "
              f"cov={bm_e.get('coverage'):.4f}", flush=True)
        out["start_metrics_eval_n"] = {
            k: bm_e.get(k) for k in ("accuracy", "coverage", "answered_precision",
                                     "abstain_rate", "unsupported_answer_rate")}

        evals = []
        for c in cands:
            cstate = ArtifactState(release_id=f"pilot-cand-{c['idx']}",
                                   system_prompt=c["prompt"],
                                   registry=state.registry)
            r = evaluate(cfg, cstate, args.dataset, args.split, seed=args.seed,
                         limit=args.eval, run_tag=f"pilot-cand{c['idx']}",
                         write_per_task=True, cfg_fp="pilot")
            cm = r.metric_values
            vd = pair_triple(bm_e, cm, cfg)
            evals.append({
                "idx": c["idx"],
                "metrics": {k: cm.get(k) for k in
                            ("accuracy", "coverage", "answered_precision",
                             "abstain_rate", "unsupported_answer_rate")},
                "deltas_pp": vd.get("deltas_pp"),
                "primary_axis": vd.get("primary_axis"),
                "primary_delta_pp": vd.get("primary_delta_pp"),
                "verdict": vd.get("verdict"),
            })
            print(f"[eval {c['idx']}] acc={cm.get('accuracy'):.4f} "
                  f"prec={cm.get('answered_precision'):.4f} "
                  f"Δprec={vd.get('primary_delta_pp')} verdict={vd.get('verdict')}",
                  flush=True)
        out["eval_n"] = args.eval
        out["noise_floor_pp"] = NOISE_FLOOR_PP.get(args.dataset, 3.3)
        out["evals"] = evals
        best = max(evals, key=lambda e: (e["primary_delta_pp"] or -99))
        out["best"] = best
        print(f"\n[best] cand#{best['idx']} Δprec={best['primary_delta_pp']}pp "
              f"(噪声地板 {out['noise_floor_pp']}pp)", flush=True)

    out_path = OUT_DIR / f"{experiment_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n产物: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
