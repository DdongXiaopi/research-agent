#!/usr/bin/env python
"""A1 重选的**配对显著性检验**。

背景：selection_gate 上 25 题的 acc 从 0.600 涨到 0.640 看着像提升，
实际只是 25 题里多对 1 题。这个脚本把两个 run 的**逐题对错**拉出来配对，
跑 McNemar 精确检验 + 配对 bootstrap CI，回答"这个差异是不是噪声"。

用法
----
    # 直接给两个 run_id
    python scripts/a1_pairwise.py --a run-xxx --b run-yyy

    # 或者对 A1 重选汇总里的每个候选批量跑（自动取报告里的 run_id）
    python scripts/a1_pairwise.py --from-summary \
        artifacts/gate_verdicts/a1_reselect_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.stats import mcnemar_exact, paired_bootstrap_binary  # noqa: E402

RUNS = Path("artifacts/runs")


def _parse_payload(raw: Any) -> Dict[str, Any]:
    """trace.jsonl 的 payload 可能是 dict，也可能是 Python repr 字符串。"""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    s2 = (s.replace("True", "true").replace("False", "false")
            .replace("None", "null").replace("'", '"'))
    try:
        return json.loads(s2)
    except Exception:
        return {}


def per_task(run_id: str) -> Dict[str, int]:
    """从 trace.jsonl 抽出 {task_id: 0/1}。"""
    f = RUNS / run_id / "trace.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"run 目录不存在: {f}")
    out: Dict[str, int] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("event_type") != "task.end":
            continue
        d = _parse_payload(r.get("payload", ""))
        if "correct" in d:
            out[r["task_id"]] = int(bool(d["correct"]))
    return out


def compare(a: str, b: str) -> Dict[str, Any]:
    """a = champion，b = challenger。返回配对统计。"""
    ta, tb = per_task(a), per_task(b)
    ids = [t for t in ta if t in tb]
    before = [ta[t] for t in ids]
    after = [tb[t] for t in ids]
    mc = mcnemar_exact(before, after)
    ci = paired_bootstrap_binary(before, after)
    return {
        "champion_run": a,
        "challenger_run": b,
        "n_paired": len(ids),
        "champion_correct": int(sum(before)),
        "challenger_correct": int(sum(after)),
        "acc_champion": round(sum(before) / len(ids), 4) if ids else None,
        "acc_challenger": round(sum(after) / len(ids), 4) if ids else None,
        "delta_pp": round((sum(after) - sum(before)) / len(ids) * 100, 2) if ids else None,
        "mcnemar": mc,
        "bootstrap_ci_pp": ci,
        "significant": bool(mc.get("p_value") is not None and mc["p_value"] < 0.05),
    }


def _from_summary(path: str) -> List[Tuple[str, str, str]]:
    """从重选汇总里取 (challenger, champion_run, challenger_run)。

    汇总里没存 run_id，靠 `report_path` 回到完整报告里取 —— 报告是
    gate_verdict 落的原始数据，字段最全。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[Tuple[str, str, str]] = []
    seen = set()

    def _walk(report: Dict[str, Any], label: str) -> None:
        for _ds, blk in (report.get("per_dataset") or {}).items():
            sd = str(blk.get("primary_seed"))
            r = (blk.get("per_seed") or {}).get(sd, {})
            key = (r.get("champion_run_id"), r.get("challenger_run_id"))
            if all(key) and key not in seen:
                seen.add(key)
                out.append((label, key[0], key[1]))

    for row in d.get("rows", []):
        rp = row.get("report_path")
        if rp and Path(rp).exists():
            _walk(json.loads(Path(rp).read_text(encoding="utf-8")), row["challenger"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="A1 重选的配对显著性检验")
    ap.add_argument("--a", help="champion run_id")
    ap.add_argument("--b", help="challenger run_id")
    ap.add_argument("--from-summary", help="重选汇总 json（批量跑）")
    ap.add_argument("--out", default="artifacts/gate_verdicts/a1_pairwise.json")
    args = ap.parse_args()

    raise SystemExit(
        "A1 历史配对分析已停用：旧 selection_gate run 可能包含逐题答案，"
        "不得再读取或重分析。历史汇总仅供审计；此入口不会打开 run trace。"
    )

    if args.from_summary:
        jobs = _from_summary(args.from_summary)
    elif args.a and args.b:
        jobs = [("manual", args.a, args.b)]
    else:
        ap.error("需要 --a/--b 或 --from-summary")
        return 2

    rows = []
    for name, a, b in jobs:
        try:
            r = compare(a, b)
            r["label"] = name
        except Exception as exc:
            r = {"label": name, "champion_run": a, "challenger_run": b, "error": str(exc)}
        rows.append(r)
        if "error" in r:
            print(f"[ERR ] {name}: {r['error'][:120]}")
            continue
        mc = r["mcnemar"]
        ci = r["bootstrap_ci_pp"]
        lo = ci.get("ci_low_pp") if isinstance(ci, dict) else None
        hi = ci.get("ci_high_pp") if isinstance(ci, dict) else None
        print(f"[OK  ] {name:24s} n={r['n_paired']} "
              f"{r['acc_champion']:.3f}->{r['acc_challenger']:.3f} "
              f"Δ={r['delta_pp']:+.1f}pp | McNemar p={mc.get('p_value'):.3f} "
              f"| CI=[{lo}, {hi}] | {'显著' if r['significant'] else '不显著'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
