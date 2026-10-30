#!/usr/bin/env python
"""A1 重选扫描：在 selection_gate 上用**真实模型 + 新协议**把一批历史 release
挨个与当前 champion 比较，输出三指标并列的对照表。

为什么要有这个脚本
------------------
决策 1（R1）之后封存集已退役，判定下沉到 ``selection_gate``。而 A1 的本意是
"把 ACTIVE 重选到新协议下" —— 它不等于"再跑一轮进化"：进化器在真实模型下
可能已饱和（实测 3 轮 0 候选），此时**从已有版本里重选**才是唯一能出结论的
通路，并且它不消耗封存揭盲预算。

用法
----
    python scripts/a1_reselect.py --config configs/experiment.http.yaml \
        --datasets litqa2 --split selection_gate --limit 25

    # 显式指定候选（默认自动：champion 的谱系祖先 + 最近发布的若干版本）
    python scripts/a1_reselect.py --candidates baseline-v0 rel-cand-xxxx

产出
----
* 每个候选一份完整报告 → ``artifacts/gate_verdicts/``（由 gate_verdict 落盘）
* 汇总表 → ``artifacts/gate_verdicts/a1_reselect_summary.json``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 允许 `python scripts/xxx.py` 直跑

from src.config import load_config  # noqa: E402
from src.eval.gate_verdict import run_gate_verdict  # noqa: E402
from src.governance import release as rel  # noqa: E402


def _lineage(anchor: str, max_depth: int = 8) -> List[str]:
    """沿 parent 指针回溯 anchor 的祖先链（不含 anchor 自身）。"""
    root = rel.RELEASES_ROOT
    out: List[str] = []
    cur = anchor
    for _ in range(max_depth):
        f = root / cur / "release.json"
        if not f.exists():
            break
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        p = j.get("parent")
        if not p or not (root / p).exists():
            break
        out.append(p)
        cur = p
    return out


def _recent(n: int, exclude: set) -> List[str]:
    """按 created_at 取最近发布的 n 个 release。"""
    root = rel.RELEASES_ROOT
    rows = []
    for d in root.iterdir():
        f = d / "release.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        rid = j.get("release_id")
        if rid and rid not in exclude and rid != "baseline-v0":
            rows.append((j.get("created_at") or "", rid))
    rows.sort(reverse=True)
    return [r for _, r in rows[:n]]


def pick_candidates(champion: str, n_recent: int = 4, extra: Optional[List[str]] = None
                    ) -> List[str]:
    """默认候选集 = 谱系祖先 ∪ 最近发布的 n 个 ∪ baseline-v0（保底有对照）。"""
    seen: List[str] = []
    for c in list(extra or []) + _lineage(champion) + _recent(n_recent, {champion}):
        if c and c != champion and c not in seen:
            seen.append(c)
    if "baseline-v0" not in seen and "baseline-v0" != champion:
        seen.append("baseline-v0")
    return seen


def _row(rep: Dict[str, Any]) -> Dict[str, Any]:
    """把一份 gate-verdict 报告压成一行汇总。"""
    out: Dict[str, Any] = {
        "challenger": rep["challenger_release"],
        "champion": rep["champion_release"],
        "split": rep["split"],
    }
    per_ds: Dict[str, Any] = {}
    for ds, blk in rep["per_dataset"].items():
        sd = str(blk["primary_seed"])
        r = blk["per_seed"][sd]
        row = {
            "n_tasks": r.get("n_tasks"),
            "champion": {k: r["baseline"].get(k) for k in
                         ("accuracy", "coverage", "answered_precision",
                          "unsupported_answer_rate", "abstention_f1", "citation_precision")},
            "challenger": {k: r["candidate"].get(k) for k in
                           ("accuracy", "coverage", "answered_precision",
                            "unsupported_answer_rate", "abstention_f1", "citation_precision")},
            "deltas_pp": r["deltas_pp"],
            "primary_axis": r["primary_axis"],
            "primary_delta_pp": r["primary_delta_pp"],
            "verdict": r["verdict"],
            "floor_effect": r["floor_effect"]["triggered"],
            "caveats": r["caveats"],
        }
        per_ds[ds] = row
    out["per_dataset"] = per_ds
    out["promotion"] = rep.get("promotion")
    out["report_path"] = rep.get("report_path")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="A1 重选扫描（selection_gate，真实模型）")
    ap.add_argument("--config", default="configs/experiment.http.yaml")
    ap.add_argument("--candidates", nargs="*", default=None,
                    help="显式指定候选 release；默认 = 谱系祖先 + 最近发布 + baseline-v0")
    ap.add_argument("--n-recent", type=int, default=4)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--split", default="selection_gate")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--no-protocol-check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raise SystemExit(
        "A1 批量重选已停用：它会用同一 selection_gate 在多个已发布版本中挑优，"
        "造成 gate 适应性选择。历史报告仅供审计；固定候选应走受控的单候选 gate 流程。"
        "此入口不会加载配置或 release，也不会启动评测。"
    )

    cfg = load_config(args.config)
    champion = rel.current_release_id()
    cands = args.candidates or pick_candidates(champion, args.n_recent)
    datasets = args.datasets or list(cfg.get("evaluation", {}).get("datasets", ["litqa2"]))
    # `--limit 0` = 不限（跑满该 split 的全部题）
    limit = args.limit if (args.limit and args.limit > 0) else None
    print(f"champion = {champion}")
    print(f"candidates = {cands}")
    print(f"split = {args.split} | datasets = {datasets} | limit = {limit or 'ALL'}")

    rows: List[Dict[str, Any]] = []
    for c in cands:
        try:
            rep = run_gate_verdict(cfg, challenger_id=c, datasets=datasets,
                                   split=args.split, seeds=args.seeds or None,
                                   limit=limit,
                                   check_protocol=not args.no_protocol_check)
            row = _row(rep)
        except Exception as exc:                      # 单个候选失败不应中断整轮扫描
            row = {"challenger": c, "error": str(exc)}
        rows.append(row)
        if "error" in row:
            print(f"[ERR ] {c}: {row['error'][:120]}")
            continue
        for ds, b in row["per_dataset"].items():
            print(f"[OK  ] {c} | {ds} n={b['n_tasks']} "
                  f"acc {b['champion']['accuracy']:.3f}->{b['challenger']['accuracy']:.3f} "
                  f"cov {b['champion']['coverage']:.3f}->{b['challenger']['coverage']:.3f} "
                  f"prec {b['champion']['answered_precision']:.3f}->{b['challenger']['answered_precision']:.3f} "
                  f"| {b['verdict']}")

    out_path = Path(args.out or "artifacts/gate_verdicts/a1_reselect_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "created_at": rel.now_iso(),
        "champion": champion,
        "split": args.split,
        "datasets": datasets,
        "limit": limit,
        "protocol_checked": not args.no_protocol_check,
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"summary -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
