#!/usr/bin/env python
"""protocolqa 长证据**截断效应**的消融探针（走真实链路）。

## 要回答的问题

protocolqa 在三个判定集里基线最低（gate acc 0.310 vs litqa2 0.717 / pubmedqa 0.663），
候选一"多答"就伴随无支撑主张率暴涨。表面看像"模型能力不够"，但按证据长度
分层后看到一个更具体的现象：

    protocolqa selection_gate 29 题（champion，run-cfe60506d962bf64）
      证据  >6000 字符（11 题）  acc 0.182  弃权率 0.545
      证据 <=6000 字符（18 题）  acc 0.389  弃权率 0.389

而 `HttpProvider.__init__` 默认 `max_context_chars = 6000`，`_truncate()` 的实现是
`t[:6000] + " …[truncated]"` —— **从中间硬切，尾部直接消失**。丢得最狠的 5 题
丢掉了 57.6% 的 protocol 正文，且这 5 题 0 对（4 弃权 + 1 错）。

要区分两个假设：
  H1 硬截断 —— 证据被物理切掉，提高阈值就能恢复；
  H2 长上下文迷失 —— 证据全给了但模型读不出来，提高阈值没用，得改压缩策略。

## 为什么走 harness 而不是直接调 provider

直接调 provider 时 `docs=[]` 会让 prompt 变成 "Retrieved passages: (none)"，
champion 的 rerank / citation / abstain 通道全部空转（实测弃权率 0.000，
纯粹瞎猜），生态效度不够。这里走项目自己的 `harness.evaluate`，唯一变量是
`models.task.max_context_chars`。

## 协议影响

本脚本**不修改仓库里任何配置文件** —— cfg 在内存里改，不落盘。
因此不影响 `configs/` 的协议指纹。它也不碰 final_holdout，只跑 dev_optimize
（判定集 selection_gate 保持干净，留给真正的 gate-verdict）。

用法
----
    python scripts/probe_context_truncation.py --limit 0     # 全量 61 题
    python scripts/probe_context_truncation.py --limit 12    # 抽样试跑
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
from src.governance import release as rel  # noqa: E402
from src.util import path  # noqa: E402

DATASET = "protocolqa"
SPLIT = "dev_optimize"          # 用 dev 做诊断，selection_gate 保持干净
DEFAULT_MAX_CTX = 6000          # HttpProvider 的现状默认值

# 条件名 -> models.task.max_context_chars
CONDITIONS = [
    ("current", DEFAULT_MAX_CTX),
    ("full", 30000),            # protocolqa 最长 14188 字符，30000 等于不截断
]


def load_config(rel_path: str) -> dict:
    import yaml

    return yaml.safe_load(path(rel_path).read_text(encoding="utf-8")) or {}


def main() -> int:
    raise SystemExit(
        "real context-truncation probe is retired: it scores provider-backed task cohorts "
        "without a current frozen request plan and shared call ledger; preserve historical "
        "reports and use a separately approved plan-bound experiment before reopening it"
    )
    ap = argparse.ArgumentParser(description="protocolqa 截断效应消融探针")
    ap.add_argument("--config", default="configs/experiment.http.yaml")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", default="artifacts/real_probe/context_truncation.json")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    state = rel.load_state()
    limit = args.limit if args.limit > 0 else None

    results = {}
    for name, mx in CONDITIONS:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("models", {}).setdefault("task", {})["max_context_chars"] = mx
        t0 = time.time()
        res = evaluate(cfg, state, DATASET, SPLIT, seed=args.seed, limit=limit,
                       run_tag=f"probe-{name}", write_per_task=True,
                       cfg_fp=f"probe-{name}")
        results[name] = {
            "run_id": res.run_id,
            "n_tasks": res.n_tasks,
            "metrics": res.metric_values,
            "failure_classes": res.failure_class_counts if hasattr(res, "failure_class_counts") else None,
        }
        m = res.metric_values
        print(f"[{name:8s}] max_context_chars={mx:6d} n={res.n_tasks} "
              f"acc={m.get('accuracy'):.3f} cov={m.get('coverage'):.3f} "
              f"prec={m.get('answered_precision'):.3f} "
              f"unsupported={m.get('unsupported_answer_rate')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    def per_task(run_id: str) -> dict:
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
                                 "status": d.get("status")}
        return out

    a = per_task(results["current"]["run_id"])
    b = per_task(results["full"]["run_id"])
    from src.util import read_jsonl

    tasks = {t["task_id"]: t for t in read_jsonl(path(f"data/normalized/{DATASET}.jsonl"))
             if t.get("split") == SPLIT}

    rows = []
    for tid in sorted(set(a) & set(b)):
        L = len((tasks.get(tid) or {}).get("context") or "")
        rows.append({"task_id": tid, "ctx_len": L,
                     "truncated": L > DEFAULT_MAX_CTX,
                     "current": ("correct" if a[tid]["correct"]
                                 else ("abstain" if a[tid]["status"] == "abstain" else "wrong")),
                     "full": ("correct" if b[tid]["correct"]
                              else ("abstain" if b[tid]["status"] == "abstain" else "wrong"))})

    def stat(sub, key):
        n = len(sub)
        if not n:
            return {"n": 0, "acc": None, "abstain": None}
        c = sum(1 for r in sub if r[key] == "correct")
        ab = sum(1 for r in sub if r[key] == "abstain")
        return {"n": n, "acc": round(c / n, 4), "abstain": round(ab / n, 4)}

    cut = [r for r in rows if r["truncated"]]
    keep = [r for r in rows if not r["truncated"]]
    flipped = [r for r in rows if r["current"] != r["full"]]

    report = {
        "dataset": DATASET, "split": SPLIT, "seed": args.seed,
        "conditions": {k: v for k, v in CONDITIONS},
        "runs": results,
        "overall": {k: stat(rows, k) for k in ("current", "full")},
        "truncated_subset": {k: stat(cut, k) for k in ("current", "full")},
        "intact_subset": {k: stat(keep, k) for k in ("current", "full")},
        "n_truncated": len(cut), "n_intact": len(keep),
        "n_flipped": len(flipped),
        "flipped_rows": flipped,
        "rows": rows,
        "note": ("唯一变量是 models.task.max_context_chars；cfg 只在内存里改，"
                 "不落盘，不改 configs/ 的协议指纹。dev_optimize 用于诊断，"
                 "selection_gate 未被触碰。"),
    }
    outp = path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"{'':22s} {'n':>4s}   current(6000)      full(30000)")
    for label, sub in (("全部题", rows), ("被截断的题(>6000)", cut), ("未被截断的题", keep)):
        if not sub:
            continue
        x, y = stat(sub, "current"), stat(sub, "full")
        print(f"{label:22s} {x['n']:4d}   acc={x['acc']:.3f} 弃权{x['abstain']:.3f}"
              f"   acc={y['acc']:.3f} 弃权{y['abstain']:.3f}")
    print(f"\n翻转题数 {len(flipped)}")
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
