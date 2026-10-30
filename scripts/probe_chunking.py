#!/usr/bin/env python
"""【P8】证据分块的**离线**诊断（零 API 调用）。

## 要回答的三个问题

1. **有没有 headroom**：现役做法（一条 passage = 一个最相关滑动窗口）捕获了多少
   与"问题+选项"相关的信号？分块筛选又能捕获多少？差多少？
2. **是不是只在长证据上生效**：短证据（≤ 预算）必须**逐字符 no-op**，否则
   litqa2 / pubmedqa 就当不了阴性对照。
3. **副作用有多大**：干扰项 passage 也会被分块 —— 它们的相关度是升是降？
   降了是好事（少喂噪声），升了要警惕（分块把干扰项也变强了）。

## 方法

对同一道题，用同一套 `RetrievalGateway` 跑两遍（分块关 / 开），逐 passage 对比：
  - 字符数
  - `chunk_score`（题干词 ×1 + 选项词 ×3；内容派生，不读答案键）
  - gold passage 是否逐字符不变

## 用法

    python scripts/probe_chunking.py
    python scripts/probe_chunking.py --budget 3000 --dataset protocolqa
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import load_tasks  # noqa: E402
from src.retrieval import chunking  # noqa: E402
from src.retrieval import corpus as rc  # noqa: E402
from src.retrieval.gateway import REAL_CORPUS, RetrievalGateway  # noqa: E402
from src.util import path  # noqa: E402

SPLITS = ["dev_optimize", "selection_gate"]


def _med(xs):
    return round(st.median(xs), 3) if xs else None


def run(ds: str, budget: int, target: int, max_chars: int, max_chunks: int,
        limit: int | None, dump_rows: bool = False) -> dict:
    chunk_cfg = {"enabled": True, "budget_chars": budget, "target_chars": target,
                 "max_chars": max_chars, "max_chunks": max_chunks,
                 "option_weight": chunking.DEFAULT_OPTION_WEIGHT}
    gw_off = RetrievalGateway(corpus_version=REAL_CORPUS, top_k=3)
    gw_on = RetrievalGateway(corpus_version=REAL_CORPUS, top_k=3, chunking=chunk_cfg)

    tasks = load_tasks(ds, splits=SPLITS)
    if limit:
        tasks = tasks[:limit]

    rows = []
    for t in tasks:
        qtok = rc.tokens(t.get("question", ""))
        opttok = rc.tokens(" ".join(t.get("options", [])))
        try:
            off = gw_off.search(t, skill_recalled=True, rerank=True)
            on = gw_on.search(t, skill_recalled=True, rerank=True)
        except Exception:
            continue
        ev = rc.evidence_of(t)
        g_off = next((d for d in off if d.get("is_gold")), None)
        g_on = next((d for d in on if d.get("is_gold")), None)
        if g_off is None or g_on is None:
            continue                       # gold_missing 桶：语料里本就没有答案
        rows.append({
            "task_id": t["task_id"],
            "split": t.get("split"),
            "ev_len": len(ev),
            "long": len(ev) > budget,
            "gold_off_chars": len(g_off["text"]),
            "gold_on_chars": len(g_on["text"]),
            "sig_off": chunking.chunk_score(g_off["text"], qtok, opttok),
            "sig_on": chunking.chunk_score(g_on["text"], qtok, opttok),
            "identical": g_off["text"] == g_on["text"],
            # 干扰项（非 gold）的平均相关度：越低 = 喂进去的噪声越少
            "distract_off": _med([chunking.chunk_score(d["text"], qtok, opttok)
                                  for d in off if not d.get("is_gold")]),
            "distract_on": _med([chunking.chunk_score(d["text"], qtok, opttok)
                                 for d in on if not d.get("is_gold")]),
            "docs_off_chars": sum(len(d["text"]) for d in off),
            "docs_on_chars": sum(len(d["text"]) for d in on),
        })

    def agg(sub, key):
        return _med([r[key] for r in sub])

    long_rows = [r for r in rows if r["long"]]
    short_rows = [r for r in rows if not r["long"]]
    out = {
        "dataset": ds, "budget_chars": budget, "n": len(rows),
        "n_long(>budget)": len(long_rows), "n_short": len(short_rows),
        "evidence_len_median": _med([r["ev_len"] for r in rows]),
        "gold_signal": {
            "off_median": agg(rows, "sig_off"), "on_median": agg(rows, "sig_on"),
            "long_off_median": agg(long_rows, "sig_off"),
            "long_on_median": agg(long_rows, "sig_on"),
        },
        "gold_chars": {"off_median": agg(rows, "gold_off_chars"),
                       "on_median": agg(rows, "gold_on_chars"),
                       "long_off_median": agg(long_rows, "gold_off_chars"),
                       "long_on_median": agg(long_rows, "gold_on_chars")},
        "distractor_signal": {"off_median": agg(rows, "distract_off"),
                              "on_median": agg(rows, "distract_on")},
        "docs_total_chars": {"off_median": agg(rows, "docs_off_chars"),
                             "on_median": agg(rows, "docs_on_chars")},
        "no_op_rate_short": (round(sum(1 for r in short_rows if r["identical"])
                                   / len(short_rows), 4) if short_rows else None),
        "changed_rate_long": (round(sum(1 for r in long_rows if not r["identical"])
                                    / len(long_rows), 4) if long_rows else None),
        "n_gold_signal_up": sum(1 for r in rows if r["sig_on"] > r["sig_off"]),
        "n_gold_signal_down": sum(1 for r in rows if r["sig_on"] < r["sig_off"]),
        "n_gold_signal_same": sum(1 for r in rows if r["sig_on"] == r["sig_off"]),
    }
    if long_rows:
        gains = [(r["sig_on"] - r["sig_off"]) for r in long_rows]
        rel = [(r["sig_on"] / r["sig_off"]) for r in long_rows if r["sig_off"] > 0]
        out["long_signal_gain"] = {
            "median_abs": _med(gains),
            "median_relative": _med(rel),
            "pct_improved": round(sum(1 for g in gains if g > 0) / len(gains), 4),
            "pct_degraded": round(sum(1 for g in gains if g < 0) / len(gains), 4),
        }
    # 逐题明细默认**不写进产物**：803 题 × 15 字段 ≈ 290KB，会把仓库撑大且
    # 没人逐行读。要看明细加 --dump-rows。
    if dump_rows:
        out["rows"] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="P8 证据分块离线诊断")
    ap.add_argument("--dataset", default="all",
                    help="数据集名或 all（litqa2 / protocolqa / pubmedqa）")
    ap.add_argument("--budget", type=int, default=1500, help="每条 passage 字符预算")
    ap.add_argument("--target", type=int, default=chunking.DEFAULT_TARGET_CHARS)
    ap.add_argument("--max-chars", type=int, default=chunking.DEFAULT_MAX_CHARS)
    ap.add_argument("--max-chunks", type=int, default=chunking.DEFAULT_MAX_CHUNKS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-rows", action="store_true", help="把逐题明细写进产物（默认不写）")
    ap.add_argument("--out", default="artifacts/probe/chunking_offline.json")
    args = ap.parse_args()

    dss = ["litqa2", "protocolqa", "pubmedqa"] if args.dataset == "all" \
        else [args.dataset]
    report = {"budget_chars": args.budget, "datasets": {}}
    for ds in dss:
        try:
            r = run(ds, args.budget, args.target, args.max_chars, args.max_chunks,
                    args.limit or None, dump_rows=args.dump_rows)
        except Exception as exc:
            print(f"[{ds}] 跳过：{exc}")
            continue
        report["datasets"][ds] = r
        g = r["gold_signal"]
        print(f"[{ds:11s}] n={r['n']:4d} 长证据={r['n_long(>budget)']:4d} "
              f"| gold 信号 off={g['off_median']} on={g['on_median']} "
              f"(长证据子集 off={g['long_off_median']} on={g['long_on_median']}) "
              f"| 短证据 no-op 率={r['no_op_rate_short']} "
              f"| 长证据被改动率={r['changed_rate_long']}")
        if "long_signal_gain" in r:
            lg = r["long_signal_gain"]
            print(f"{'':13s} 长证据信号增益 中位 +{lg['median_abs']} "
                  f"(×{lg['median_relative']})，变好 {lg['pct_improved']:.0%} / "
                  f"变差 {lg['pct_degraded']:.0%}")

    outp = path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
