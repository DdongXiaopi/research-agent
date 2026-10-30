#!/usr/bin/env python
"""【P8】证据分块的**真实**消融：分块关 vs 预算中性 vs 扩预算。

## 为什么要跑

离线诊断（`scripts/probe_chunking.py`）只能测"词面信号"，测不到模型行为。
它给出的结论是：**分块几乎没有 headroom**

- 预算中性（1500）：gold 信号 27→26，干扰项 24→25（略变差）
- 长证据在预算 1500 时已捕获全量信号的 84%（中位），3000 才有 95%
- 而且 protocolqa 的全文 protocol 本来就已经通过 `context` 全部进了 prompt

但词面信号不是模型。本脚本做模型层面的判定。

## 预注册判据（写死在脚本里，跑之前就定好）

单数据集、单批次、题级配对，三个条件两两比较：

1. **方向**：只看 answered_precision（三指标主判据）的 Δ；
2. **超噪**：|Δ| 必须 > 该数据集的噪声地板（protocolqa dev 61 题 ≈ 1.64pp/题，
   实测跨 seed 极差约 1 题 ⇒ 地板取 **3.3pp**，即 2 题）；
3. **接地性**：无支撑主张率 Δ ≤ +5.0pp（与 gate 容差一致）。

只有同时满足 1（正）+ 2 + 3，才可以说"分块有效"。
不满足 2 的结论是 **inconclusive**（测不出来），**不得**解读成"没有效果"。

## 用法

    python scripts/probe_chunking_real.py --dataset protocolqa --limit 0
    python scripts/probe_chunking_real.py --dataset litqa2 --limit 30 --conditions off,b3000
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.harness import evaluate  # noqa: E402
from src.eval.stats import mcnemar_exact  # noqa: E402
from src.governance import release as rel  # noqa: E402
from src.util import path  # noqa: E402

SPLIT = "dev_optimize"          # 诊断用 dev；selection_gate 留给真正的判定
# 【D5 2026-09-27】加 --split：pubmedqa 的 dev_optimize 有 525 题（跑全量太贵），
# 而 selection_gate 190 题才是「不被进化污染 + 功效足够」的判定面
# （噪声地板 0.89pp，见 caveat #10）。故本脚本也支持在 gate 上做对照。

# 条件名 -> (retrieval.chunking, models.task.max_doc_chars)
CONDITIONS = {
    "off": (None, 1500),
    "b1500": ({"enabled": True, "budget_chars": 1500}, 1500),
    "b3000": ({"enabled": True, "budget_chars": 3000}, 3000),
}

# 噪声地板（pp）：|Δ| 低于它就是"测不出来"，不是"没效果"。
# 依据 2026-09-27 实测：champion 自身跨 seed 的极差在三个集上都是 1 题。
NOISE_FLOOR_PP = {"protocolqa": 3.3, "litqa2": 4.3, "pubmedqa": 1.6}
UNSUPPORTED_TOLERANCE_PP = 5.0


def load_config(rel_path: str) -> dict:
    import yaml

    return yaml.safe_load(path(rel_path).read_text(encoding="utf-8")) or {}


def per_task_of(run_id: str) -> dict:
    """从 trace 里读回逐题结果（correct / abstain）。"""
    f = path(f"artifacts/runs/{run_id}/trace.jsonl")
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event_type") != "task.end":
            continue
        d = r.get("payload")
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except Exception:
                d = {}
        out[r["task_id"]] = {"correct": bool(d.get("correct")),
                             "status": d.get("status", "")}
    return out


def triple(rows: dict) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    ans = [r for r in rows.values() if r["status"] != "abstain"]
    cov = len(ans) / n
    prec = (sum(1 for r in ans if r["correct"]) / len(ans)) if ans else 0.0
    return {"n": n, "accuracy": round(sum(1 for r in rows.values() if r["correct"]) / n, 4),
            "coverage": round(cov, 4), "answered_precision": round(prec, 4),
            "abstain_rate": round(1 - cov, 4)}


def main() -> int:
    raise SystemExit(
        "real chunking probe is retired: it scores provider-backed task cohorts without a "
        "current frozen request plan and shared call ledger; preserve historical reports "
        "and use a separately approved plan-bound experiment before reopening it"
    )
    ap = argparse.ArgumentParser(description="P8 证据分块真实消融")
    ap.add_argument("--config", default="configs/experiment.http.yaml")
    ap.add_argument("--dataset", default="protocolqa")
    ap.add_argument("--limit", type=int, default=0, help="0=跑满该 split")
    ap.add_argument("--split", default=SPLIT,
                    help="dev_optimize（默认，诊断用）/ selection_gate（判定面）")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--conditions", default="off,b1500,b3000")
    ap.add_argument("--out", default="artifacts/probe/chunking_real.json")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    state = rel.load_state()
    limit = args.limit if args.limit > 0 else None
    names = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in names if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"未知条件 {unknown}；可用：{list(CONDITIONS)}")

    runs, per_task, tokens = {}, {}, {}
    for name in names:
        chunk_cfg, max_doc = CONDITIONS[name]
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("models", {}).setdefault("task", {})["max_doc_chars"] = max_doc
        if chunk_cfg is None:
            cfg.setdefault("retrieval", {}).pop("chunking", None)
        else:
            cfg.setdefault("retrieval", {})["chunking"] = dict(chunk_cfg)
        t0 = time.time()
        res = evaluate(cfg, state, args.dataset, args.split, seed=args.seed, limit=limit,
                       run_tag=f"p8-{name}", write_per_task=True,
                       cfg_fp=f"p8-{name}")
        m = res.metric_values
        runs[name] = {"run_id": res.run_id, "metrics": m}
        per_task[name] = per_task_of(res.run_id)
        tokens[name] = m.get("prompt_tokens")
        t = triple(per_task[name])
        print(f"[{name:6s}] n={t['n']:3d} acc={t['accuracy']:.3f} "
              f"cov={t['coverage']:.3f} prec={t['answered_precision']:.3f} "
              f"弃权={t['abstain_rate']:.3f} "
              f"unsupported={m.get('unsupported_answer_rate')} "
              f"prompt_tok={tokens[name]} ({time.time()-t0:.0f}s)", flush=True)

    # 两两比较
    ids = sorted(set.intersection(*[set(v) for v in per_task.values()]))
    floor = NOISE_FLOOR_PP.get(args.dataset, 3.3)
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ta, tb = triple(per_task[a]), triple(per_task[b])
            d_prec = round((tb["answered_precision"] - ta["answered_precision"]) * 100, 2)
            d_cov = round((tb["coverage"] - ta["coverage"]) * 100, 2)
            d_acc = round((tb["accuracy"] - ta["accuracy"]) * 100, 2)
            reg = sum(1 for k in ids if per_task[a][k]["correct"] and not per_task[b][k]["correct"])
            imp = sum(1 for k in ids if not per_task[a][k]["correct"] and per_task[b][k]["correct"])
            mc = mcnemar_exact([float(per_task[a][k]["correct"]) for k in ids],
                               [float(per_task[b][k]["correct"]) for k in ids])
            if abs(d_prec) <= floor:
                verdict = "inconclusive_below_noise"
            elif d_prec > 0:
                verdict = "improved"
            else:
                verdict = "regressed"
            pairs[f"{a}->{b}"] = {"delta_precision_pp": d_prec, "delta_coverage_pp": d_cov,
                                  "delta_accuracy_pp": d_acc, "n_improved": imp,
                                  "n_regressed": reg, "mcnemar_p": mc["p_value"],
                                  "noise_floor_pp": floor, "verdict": verdict}

    report = {
        "dataset": args.dataset, "split": args.split, "seed": args.seed,
        "conditions": {k: {"chunking": CONDITIONS[k][0],
                           "max_doc_chars": CONDITIONS[k][1]} for k in names},
        "runs": runs, "triples": {k: triple(v) for k, v in per_task.items()},
        "prompt_tokens": tokens, "pairwise": pairs,
        "preregistered": {
            "main_metric": "answered_precision",
            "noise_floor_pp": floor,
            "unsupported_tolerance_pp": UNSUPPORTED_TOLERANCE_PP,
            "rule": ("|Δprecision| ≤ 噪声地板 ⇒ inconclusive（测不出来，不等于没效果）；"
                     "同批次题级配对跑，Δ 的共同漂移做差时大部分抵消"),
        },
    }
    outp = path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    for k, v in pairs.items():
        print(f"{k:12s} Δprec={v['delta_precision_pp']:+.2f}pp "
              f"Δcov={v['delta_coverage_pp']:+.2f}pp Δacc={v['delta_accuracy_pp']:+.2f}pp "
              f"| 好{ v['n_improved']} 坏{v['n_regressed']} "
              f"McNemar p={v['mcnemar_p']:.3f} | {v['verdict']}")
    print(f"\n-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
