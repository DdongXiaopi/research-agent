#!/usr/bin/env python
"""protocolqa 的**定性画像**（不用于定量判定）。

## 为什么要有这个脚本

protocolqa 在三个判定集里分数最低、噪声地板最高（gate 29 题，1 题 = 3.45pp）。
2026-09-27 的 3-seed 实测确认：它**结构上不适合**做定量判定 —— 全量只有 90 题，
再怎么跑也压不下噪声地板。于是它被正式改作**接地性守门集（定性）**。

但"定性"不能只是"反正分低，我们看看就好"。这个脚本把它**到底在测什么**
钉成可复算的数字：

1. **正确选项在证据里被字面陈述的比例** —— protocolqa 的选项是"该怎么改"的
   干预建议，protocol 原文从来不会写出"把 NaCl 改成 0.85g"。若这个比例接近 0，
   那么任何"证据里有没有这句话"式的接地判据在它上面都必然失效；
2. **弃权 × gold 是否在 top_k 的交叉表** —— 区分"该弃权而弃权"与"有证据却弃权"；
3. **证据长度与选项类型** —— 说明它为什么与 litqa2 / pubmedqa 不同质。

用法
----
    python scripts/protocolqa_profile.py
    python scripts/protocolqa_profile.py --run-id run-xxxxxxxx  # 指定某次运行
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import load_tasks  # noqa: E402
from src.retrieval import corpus as rc  # noqa: E402
from src.runtime.provider import STATED_MIN_CHARS, _norm  # noqa: E402
from src.util import path, read_jsonl  # noqa: E402

DATASETS = ["litqa2", "protocolqa", "pubmedqa"]


def stated(opt: str, text: str) -> bool:
    n = _norm(opt)
    return len(n) >= STATED_MIN_CHARS and n in _norm(text)


def literal_support(ds: str) -> dict:
    """正确选项被证据**字面陈述**的比例 —— 接地判据能否生效的前提。"""
    out = {}
    for sp in ["dev_optimize", "selection_gate"]:
        try:
            tasks = load_tasks(ds, splits=[sp])
        except Exception:
            continue
        rows = []
        for t in tasks:
            ev = rc.evidence_of(t)
            opts = list(t.get("options", []))
            gi = int(t.get("correct_index", -1))
            gold = opts[gi] if 0 <= gi < len(opts) else ""
            rows.append({
                "gold_stated": bool(gold) and stated(gold, ev),
                "n_stated": sum(1 for o in opts if stated(o, ev)),
                "gold_len": len(_norm(gold)),
                "ev_len": len(ev),
                "n_opts": len(opts),
            })
        n = len(rows)
        if not n:
            continue
        out[sp] = {
            "n": n,
            "gold_option_literally_stated": round(sum(r["gold_stated"] for r in rows) / n, 4),
            "mean_options_stated": round(sum(r["n_stated"] for r in rows) / n, 3),
            "mean_options": round(sum(r["n_opts"] for r in rows) / n, 2),
            "gold_option_chars_median": st.median([r["gold_len"] for r in rows]),
            "evidence_chars_median": st.median([r["ev_len"] for r in rows]),
        }
    return out


def abstain_crosstab(run_id: str) -> dict | None:
    """弃权 × gold_in_topk 交叉表：区分"该弃权"与"误弃权"。"""
    f = path(f"artifacts/runs/{run_id}/per_task.jsonl")
    if not f.exists():
        return None
    rows = read_jsonl(f)
    cells = {}
    for key, sub in (("gold_in_topk", [r for r in rows if r.get("gold_in_topk")]),
                     ("gold_missing", [r for r in rows if not r.get("gold_in_topk")])):
        if not sub:
            continue
        ab = [r for r in sub if int(r.get("answer_index", -1)) < 0]
        ans = [r for r in sub if int(r.get("answer_index", -1)) >= 0]
        cells[key] = {
            "n": len(sub),
            "abstained": len(ab),
            "answered": len(ans),
            "answered_correct": sum(1 for r in ans if r.get("correct")),
            "answered_precision": round(sum(1 for r in ans if r.get("correct")) / len(ans), 4)
            if ans else None,
        }
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description="protocolqa 定性画像")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--run-id", default="", help="可选：某次运行的 run_id（出交叉表）")
    ap.add_argument("--out", default="artifacts/probe/protocolqa_profile.json")
    args = ap.parse_args()

    report = {"datasets": {}, "run_id": args.run_id or None}
    print(f"{'数据集':<12s}{'分区':<17s}{'n':>5s} {'正确选项被字面陈述':>18s} "
          f"{'平均被陈述选项数':>14s} {'证据长度中位':>10s}")
    for ds in args.datasets.split(","):
        d = literal_support(ds.strip())
        report["datasets"][ds] = d
        for sp, v in d.items():
            print(f"{ds:<12s}{sp:<17s}{v['n']:>5d} {v['gold_option_literally_stated']:>18.1%} "
                  f"{v['mean_options_stated']:>14.2f} {v['evidence_chars_median']:>10.0f}")

    if args.run_id:
        ct = abstain_crosstab(args.run_id)
        report["abstain_crosstab"] = ct
        print()
        if ct is None:
            print(f"[warn] {args.run_id} 没有 per_task.jsonl（selection_gate 按设计不落逐题结果）")
        else:
            for k, v in ct.items():
                print(f"{k:<14s} n={v['n']:>3d} 弃权={v['abstained']:>3d} "
                      f"作答={v['answered']:>3d} 答对={v['answered_correct']:>3d} "
                      f"精度={v['answered_precision']}")

    outp = path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
