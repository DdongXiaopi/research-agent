"""自研 RewriteBackend 迭代式整段改写 + Pareto 前沿（探索脚本）。

## 与上一版 pilot 的三点区别（对应发现的三个问题）

| 问题 | 上一版 | 本版 |
|---|---|---|
| 单次改写不够 | 单次提案 → 挑主判据最优 | **多轮迭代**，每轮从 Pareto 前沿选种子，产出新候选并入前沿 |
| 偏科生被当成最优 | 按主判据挑单一最优 | **Pareto 前沿**保留互补解（coverage 型 / precision 型都在） |
| 提案提示词缺约束 | 只要求不丢 ANSWER | 新增"不得用精度换覆盖率"+"弃权必须用 INSUFFICIENT" |

## 数据隔离（关键，上一版就栽在这里）

上一版的失败样本与评测题**重叠 25%**，于是 dev 上报出 +5.24pp、
holdout 上却是 −2.33pp。本版三块数据互不相交：

- **提案输入**：`dev_optimize` 前 N 题（失败样本 + few-shot 示范）
- **迭代评测**：`selection_gate`（**不同 split**）
- **可选探索性复查**：`dev_optimize` 的第 N+1..M 题；这些题已多次复用

## 用法

    python scripts/pilot_rewrite_iterative.py --rounds 2 --per-round 2
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

OUT_DIR = ROOT / "artifacts" / "probe" / "iterative"


def _eval_prompt(cfg, prompt, registry, release_id, dataset, split, seed,
                 limit, tag):
    from src.eval.harness import evaluate
    from src.governance.release import ArtifactState

    st = ArtifactState(release_id=release_id, system_prompt=prompt,
                       registry=registry)
    return evaluate(cfg, st, dataset, split, seed=seed, limit=limit,
                    run_tag=tag, write_per_task=(split != "selection_gate"),
                    cfg_fp="pilot-iter-v2")


def _task_ids(res):
    return [str(r.get("task_id")) for r in (getattr(res, "per_task", None) or [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    ap.add_argument("--dataset", default="pubmedqa")
    ap.add_argument("--release", default="baseline-v1-cited")
    ap.add_argument("--propose-split", default="dev_optimize",
                    help="失败样本与 few-shot 的来源")
    ap.add_argument("--eval-split", default="selection_gate", help="迭代评测集")
    ap.add_argument("--scan", type=int, default=30, help="提案用的题数")
    ap.add_argument("--eval", type=int, default=190, help="迭代评测题数（0=跑满）")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--per-round", type=int, default=2)
    ap.add_argument("--confirm", type=int, default=0,
                    help="仅探索性复查已多次查看的 dev 题；默认不执行")
    ap.add_argument("--confirm-cand", default=None,
                    help="指定确认哪个前沿成员（默认取 accuracy 最高的）")
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    raise SystemExit(
        "该历史探索脚本已停用：它曾用 selection_gate 迭代选择候选，"
        "不符合 FR-203/FR-204。历史产物仅供审计；请改用受控的 task-input 搜索入口，"
        "并先生成新的冻结计划。此入口不会加载配置、任务或 provider，也不会评分。"
    )
    experiment_id = "iter-" + uuid.uuid4().hex[:12]
    out_path = OUT_DIR / f"{experiment_id}.json"

    cfg = load_config(args.config)
    state = rel.load_state(args.release)
    ev_limit = args.eval if args.eval > 0 else None

    from src.data.split import load_tasks
    from src.evolution.fewshot import bootstrap_demos, render_demos_block
    from src.evolution.pareto import objectives_of, pareto_frontier, select_seed
    from src.evolution.reflection_backend import RewriteBackend

    tasks_by_id = {str(t.get("task_id")): t
                   for t in load_tasks(args.dataset, [args.propose_split])}

    # ---- 起点 ----
    base = _eval_prompt(cfg, state.system_prompt, state.registry, "iter-start",
                        args.dataset, args.eval_split, args.seed, ev_limit, "iter-start")
    bm = base.metric_values
    print(f"[start] n={bm.get('n_tasks')} acc={bm.get('accuracy'):.4f} "
          f"prec={bm.get('answered_precision'):.4f} cov={bm.get('coverage'):.4f} "
          f"uns={bm.get('unsupported_answer_rate')}", flush=True)

    frontier = [{"id": "start", "round": 0, "prompt": state.system_prompt,
                 "objectives": objectives_of(bm), "metrics": bm,
                 "run_id": base.run_id,
                 "rationale": "起点（baseline-v1-cited）"}]
    all_candidates = list(frontier)
    history: list = []

    # ---- 迭代 ----
    seen_prompts = {state.system_prompt}
    for rnd in range(1, args.rounds + 1):
        seed_item = select_seed(frontier, strategy="round_robin" if rnd > 1
                                else "best_primary", round_index=rnd - 2)
        if seed_item is None:
            break
        print(f"\n=== round {rnd}: seed={seed_item['id']} ===", flush=True)

        # 失败样本：只在 propose_split 上跑"种子 prompt"，且只取前 scan 题
        seed_state = rel.load_state(args.release)  # registry 复用
        from src.governance.release import ArtifactState
        sst = ArtifactState(release_id=seed_item["id"],
                            system_prompt=seed_item["prompt"],
                            registry=state.registry)
        scan = _eval_prompt(cfg, seed_item["prompt"], state.registry,
                            f"iter-r{rnd}-scan", args.dataset, args.propose_split,
                            args.seed, args.scan, f"iter-r{rnd}-scan")
        scan_ids = set(_task_ids(scan))
        failures = []
        for row in (getattr(scan, "per_task", None) or []):
            if row.get("correct"):
                continue
            t = tasks_by_id.get(str(row.get("task_id")), {})
            failures.append({
                "question": t.get("question", ""),
                "options": t.get("options", []),
                "correct_index": row.get("gold_index", t.get("correct_index")),
                "answer_index": row.get("answer_index"),
                "feedback": f"attributed={row.get('attributed_class', 'n/a')}",
            })
        demos = bootstrap_demos(cfg, sst, args.dataset, args.propose_split, k=3,
                                seed=args.seed, limit=args.scan,
                                run_tag=f"iter-r{rnd}-fewshot")
        demos_block = render_demos_block(demos)
        print(f"[round {rnd}] failures={len(failures)} demos={len(demos)}", flush=True)

        backend = RewriteBackend(cfg=cfg, max_attempts=2, demos=demos_block)
        for i in range(args.per_round):
            patch = backend.propose("plan.failed", seed_item["prompt"], failures,
                                    cfg, state=sst)
            if patch is None or not getattr(patch, "full_prompt", ""):
                print(f"[round {rnd}.{i}] 未产出合格候选："
                      f"{getattr(backend, 'last_error', None)}", flush=True)
                continue
            new = patch.full_prompt
            if new in seen_prompts:
                print(f"[round {rnd}.{i}] 重复，跳过", flush=True)
                continue
            seen_prompts.add(new)
            res = _eval_prompt(cfg, new, state.registry, f"iter-r{rnd}c{i}",
                               args.dataset, args.eval_split, args.seed,
                               ev_limit, f"iter-r{rnd}c{i}")
            m = res.metric_values
            obj = objectives_of(m)
            if obj is None:
                continue
            cand = {"id": f"r{rnd}c{i}", "round": rnd, "prompt": new,
                    "objectives": obj, "metrics": m,
                    "rationale": patch.rationale, "seed": seed_item["id"],
                    "run_id": res.run_id}
            frontier.append(cand)
            all_candidates.append(cand)
            print(f"[round {rnd}.{i}] acc={m.get('accuracy'):.4f} "
                  f"prec={m.get('answered_precision'):.4f} "
                  f"cov={m.get('coverage'):.4f} uns={m.get('unsupported_answer_rate')}",
                  flush=True)

        frontier = pareto_frontier(frontier)
        history.append({"round": rnd, "seed": seed_item["id"],
                        "frontier_ids": [it["id"] for it in frontier]})
        print(f"[round {rnd}] 前沿 = {[it['id'] for it in frontier]}", flush=True)

    # ---- 探索性复查：这 200 题已被多次查看，不能再称独立确认 ----
    confirm = None
    if args.confirm > 0 and len(frontier) > 1:
        # 选谁去确认：默认按**主判据最优**，但主判据（precision）最高的那个
        # 未必是 accuracy 最高的 —— 实测 r1c0 的 prec 最高却在干净题上 −3.5pp。
        # 故支持显式指定，并默认改用 accuracy 最高的前沿成员。
        if args.confirm_cand:
            pick = [it for it in frontier if it["id"] == args.confirm_cand]
            if not pick:
                raise SystemExit(f"前沿里没有 id={args.confirm_cand}")
            best = pick[0]
        else:
            best = max(frontier, key=lambda it: it["metrics"].get("accuracy") or 0)
        print(f"\n=== 探索性复查：best={best['id']} on {args.propose_split} "
              f"(排除 {len(scan_ids)} 题提案样本) ===", flush=True)
        cb = _eval_prompt(cfg, state.system_prompt, state.registry, "confirm-base",
                          args.dataset, args.propose_split, args.seed,
                          args.confirm, "confirm-base")
        cc = _eval_prompt(cfg, best["prompt"], state.registry, f"confirm-{best['id']}",
                          args.dataset, args.propose_split, args.seed,
                          args.confirm, f"confirm-{best['id']}")

        def per_task_full(res):
            """从**文件**读逐题结果。

            ⚠️ 不能用内存里的 `res.per_task`：那是简版（只有
            `attributed_class / correct / task_id`），没有 `status`，
            于是弃权率算成 0、coverage 算成 1.0（实测踩过）。
            完整字段只落在 `artifacts/runs/<run_id>/per_task.jsonl`。
            """
            p = ROOT / "artifacts" / "runs" / str(res.run_id) / "per_task.jsonl"
            if not p.exists():
                return []
            return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                    if l.strip()]

        def clean(res):
            rows = [r for r in per_task_full(res)
                    if str(r.get("task_id")) not in scan_ids]
            n = len(rows)
            if not n:
                return {}
            cor = sum(1 for r in rows if r.get("correct"))
            # 弃权用 status 判定（与 metrics.py 一致）；作答的题里答对的比例 = precision
            abst = sum(1 for r in rows
                       if str(r.get("status", "")).lower() == "abstain")
            answered = n - abst
            return {"n_tasks": n, "accuracy": cor / n,
                    "abstain_rate": abst / n, "coverage": answered / n,
                    "answered_precision": (cor / answered) if answered else 0.0}

        b_clean, c_clean = clean(cb), clean(cc)
        from src.eval.stats import mcnemar_exact
        bm2 = {str(r.get("task_id")): int(bool(r.get("correct")))
               for r in per_task_full(cb)
               if str(r.get("task_id")) not in scan_ids}
        cm2 = {str(r.get("task_id")): int(bool(r.get("correct")))
               for r in per_task_full(cc)
               if str(r.get("task_id")) not in scan_ids}
        ids = sorted(set(bm2) & set(cm2))
        mc = mcnemar_exact([bm2[i] for i in ids], [cm2[i] for i in ids]) if ids else {}
        confirm = {
            "excluded_proposal_ids": len(scan_ids),
            "n_exploratory": len(ids),
            "cohort_status": "exploratory_reused",
            "cand_id": best["id"],
            "base": b_clean, "cand": c_clean,
            "delta_pp": {k: (c_clean.get(k, 0) - b_clean.get(k, 0)) * 100
                         for k in ("accuracy", "coverage", "answered_precision",
                                   "abstain_rate")},
            "mcnemar_p": mc.get("p_value"),
        }
        print(f"[confirm] n={len(ids)} cand={best['id']} "
              f"base_acc={b_clean.get('accuracy'):.4f} "
              f"cand_acc={c_clean.get('accuracy'):.4f} "
              f"Δacc={(confirm['delta_pp'].get('accuracy') or 0):+.2f}pp "
              f"Δprec={(confirm['delta_pp'].get('answered_precision') or 0):+.2f}pp "
              f"Δcov={(confirm['delta_pp'].get('coverage') or 0):+.2f}pp "
              f"p={mc.get('p_value')}", flush=True)

    out = {
        "purpose": "迭代式整段改写 + Pareto 前沿；可选复查仅属探索",
        "experiment_id": experiment_id,
        "cohort_status": "exploratory",
        "config": args.config, "dataset": args.dataset,
        "config_sha256": sha256_obj(cfg),
        "propose_split": args.propose_split, "eval_split": args.eval_split,
        "rounds": args.rounds, "per_round": args.per_round,
        "start_metrics": {k: bm.get(k) for k in
                          ("accuracy", "coverage", "answered_precision",
                           "abstain_rate", "unsupported_answer_rate")},
        "frontier": [{"id": it["id"], "round": it["round"],
                      "global_id": f"{experiment_id}:{it['id']}:{sha256_obj(it.get('prompt', ''))[:12]}",
                      "prompt_sha256": sha256_obj(it.get("prompt", "")),
                      "run_id": it.get("run_id"),
                      # 必须落盘 prompt 原文：否则事后无法复现/确认某个前沿成员
                      # （只存 sha256 时想再验一次就得把整个迭代重跑一遍）。
                      "prompt": it.get("prompt", ""),
                      "objectives": it["objectives"],
                      "metrics": {k: it["metrics"].get(k) for k in
                                  ("accuracy", "coverage", "answered_precision",
                                   "abstain_rate", "unsupported_answer_rate")},
                      "rationale": it.get("rationale")}
                     for it in frontier],
        "history": history,
        "all_candidates": [{**it, "global_id": f"{experiment_id}:{it['id']}"}
                           for it in all_candidates],
        "confirm": confirm,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n产物: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
