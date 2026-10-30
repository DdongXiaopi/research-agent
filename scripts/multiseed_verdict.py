"""多 seed 判定聚合器：把「端点不可复现噪声」变成可测量的量。

## 为什么需要它

2026-09-27 发现：同一请求 body 逐字符相同，**跨天**重跑 pubmedqa 190 题有
8 题翻转（4.2%），方向对称（4 好 4 坏，acc 净变化 0.0pp）。含义是：
**单次运行下小于约 5pp 的效应测不出来。**

多 seed 能不能解决？取决于噪声的时间尺度：

| 噪声类型 | 同批次多 seed 能否捕获 |
|---|---|
| 即时非确定性（同一次调用内的采样/批处理抖动） | ✅ 能，取平均可压下去 |
| 跨时段漂移（服务端模型/路由状态变了） | ❌ 不能，同批次内重跑完全一致 |

本脚本**同时测这两件事**：

1. **噪声地板** = champion 自身跨 seed 的 accuracy / precision 极差。
   这就是"同样的输入反复问，答案能差多少"的直接测量。
2. **效应** = challenger − champion 的 delta，跨 seed 取均值。

判据（**预注册**，跑之前写死，禁止事后调整）：

- **一致性**：所有 seed 的 primary delta 符号必须一致（都 ≥ −tolerance 或都 < 0）。
  符号打架 → `inconclusive_inconsistent`，**不是**"通过"。
- **超噪**：|delta 均值| 必须 > 噪声地板，否则 `inconclusive_below_noise`。
- **接地性**：unsupported delta 均值 ≤ tolerance（默认 5.0pp）。

前两条是硬门槛：不通过就是"测不出来"，不得解读成"没有提升"。

## 用法

    # 必须禁缓存！否则多 seed 100% 命中本地 HTTP 缓存 = 白跑
    RSI_HTTP_NO_CACHE=1 python scripts/multiseed_verdict.py \
        --challenger rel-cand-4bf3fba380ec --seeds 20260924 20260925 20260926

    # 只在 protocolqa 上跑（省钱，且它是接地性守门集）
    RSI_HTTP_NO_CACHE=1 python scripts/multiseed_verdict.py \
        --challenger rel-cand-4bf3fba380ec --datasets protocolqa --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.eval.gate_verdict import compare_on_split  # noqa: E402
from src.governance import release as rel  # noqa: E402
from src.util import artifacts_path, now_iso, write_json  # noqa: E402

OUT_ROOT = artifacts_path("gate_verdicts")

# 预注册容差（与 gate-verdict 保持一致）
DEFAULT_TOL_PRIMARY = 0.0     # 主判据不容忍回退
DEFAULT_TOL_UNSUPPORTED = 5.0  # 无支撑主张率容差（pp）


def _span(values: Sequence[float]) -> float:
    """极差：max − min。样本 <2 时为 0。"""
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return 0.0
    return round(max(v) - min(v), 4)


def _mean(values: Sequence[float]) -> Optional[float]:
    v = [x for x in values if x is not None]
    return round(statistics.fmean(v), 4) if v else None


def aggregate(rep: Dict[str, Any], tol_primary: float,
              tol_unsup: float) -> Dict[str, Any]:
    """把 compare_on_split 的多 seed 结果聚成一条带噪声地板的结论。"""
    per_ds = {}
    for ds, blk in rep["per_dataset"].items():
        seeds = sorted(blk["per_seed"].keys())
        rows = [blk["per_seed"][s] for s in seeds]

        axis = rows[0].get("primary_axis") or "answered_precision"
        deltas = [r["primary_delta_pp"] for r in rows]
        # champion / challenger 自身跨 seed 的波动 = 噪声地板
        champ_prec = [(r["baseline"] or {}).get(axis) for r in rows]
        cand_prec = [(r["candidate"] or {}).get(axis) for r in rows]
        champ_acc = [(r["baseline"] or {}).get("accuracy") for r in rows]

        noise_floor_pp = round(max(_span(champ_prec), _span(cand_prec)) * 100.0, 4)
        champ_acc_span_pp = round(_span(champ_acc) * 100.0, 4)

        unsup = []
        for r in rows:
            base = (r["baseline"] or {}).get("unsupported_answer_rate")
            cand = (r["candidate"] or {}).get("unsupported_answer_rate")
            unsup.append(None if (base is None or cand is None)
                         else round((cand - base) * 100.0, 4))

        d_mean = _mean(deltas)
        u_mean = _mean(unsup)

        # ---- 预注册判据 ----
        reasons: List[str] = []
        # (1) 一致性：符号必须一致（都以 -tol 为界同侧）
        if d_mean is None:
            verdict = "inconclusive_no_data"
            reasons.append("主判据无数据")
        else:
            above = [d for d in deltas if d >= -tol_primary - 1e-9]
            signs_consistent = (len(above) == len(deltas)) or (len(above) == 0)
            if not signs_consistent:
                verdict = "inconclusive_inconsistent"
                reasons.append(
                    f"各 seed 主判据符号不一致 {deltas} → 无法判定方向")
            elif d_mean < -tol_primary - 1e-9:
                verdict = "regressed"
                reasons.append(f"主判据均值 {d_mean:+.2f}pp < 容差 {-tol_primary}pp")
            else:
                # (2) 超噪：效应必须大于噪声地板
                if abs(d_mean) <= noise_floor_pp:
                    verdict = "inconclusive_below_noise"
                    reasons.append(
                        f"|Δ|={abs(d_mean):.2f}pp ≤ 噪声地板 {noise_floor_pp:.2f}pp "
                        f"（champion 自身跨 seed 极差）→ 测不出来，不等于没有提升")
                else:
                    verdict = "improved"
                    reasons.append(
                        f"Δ={d_mean:+.2f}pp > 噪声地板 {noise_floor_pp:.2f}pp")

            # (3) 接地性：无支撑主张率不得超容差
            if u_mean is not None and u_mean > tol_unsup + 1e-9:
                if verdict == "improved":
                    verdict = "blocked_unsupported"
                reasons.append(
                    f"无支撑主张率均值 {u_mean:+.2f}pp > 容差 {tol_unsup}pp → 接地性回退")

        per_ds[ds] = {
            "seeds": seeds,
            "n_tasks": rows[0].get("n_tasks"),
            "primary_axis": axis,
            "primary_delta_per_seed_pp": deltas,
            "primary_delta_mean_pp": d_mean,
            "primary_delta_span_pp": _span(deltas),
            "noise_floor_pp": noise_floor_pp,          # 主判据上的噪声地板
            "champion_acc_span_pp": champ_acc_span_pp,  # accuracy 上的噪声地板
            "unsupported_delta_per_seed_pp": unsup,
            "unsupported_delta_mean_pp": u_mean,
            "verdict": verdict,
            "reasons": reasons,
            "verdicts_per_seed": [r.get("verdict") for r in rows],
        }

    # ---- 总判定：任一主集被拦/回退则整体不晋升 ----
    blocking = [d for d, v in per_ds.items()
                if v["verdict"] in ("regressed", "blocked_unsupported")]
    inconclusive = [d for d, v in per_ds.items()
                    if v["verdict"].startswith("inconclusive")]
    if blocking:
        overall = "not_promotable"
    elif inconclusive:
        overall = "inconclusive"
    else:
        overall = "promotable_directionally"

    return {
        "created_at": now_iso(),
        "champion": rep["champion_release"],
        "challenger": rep["challenger_release"],
        "split": rep["split"],
        "seeds": rep["seeds"],
        "cache_disabled": os.environ.get("RSI_HTTP_NO_CACHE") == "1",
        "tolerances": {"primary_axis_tolerance_pp": tol_primary,
                       "unsupported_answer_rate_pp": tol_unsup},
        "per_dataset": per_ds,
        "overall": overall,
        "blocking_datasets": blocking,
        "inconclusive_datasets": inconclusive,
        "note": ("总判定只称『方向性』。selection_gate 按设计不给 CI；"
                 "且本项目的端点存在约 4pp 的跨时段不可复现噪声，"
                 "单次或单批次的结论都不得写成『显著改善』。"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="多 seed 判定 + 噪声地板测量")
    ap.add_argument("--config", default="configs/experiment.http.pqa.yaml")
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--champion", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--split", default="selection_gate")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tol-primary", type=float, default=DEFAULT_TOL_PRIMARY)
    ap.add_argument("--tol-unsupported", type=float, default=DEFAULT_TOL_UNSUPPORTED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    raise SystemExit(
        "Historical multi-seed challenger scoring is disabled: this entrypoint "
        "can compare candidates on selection_gate without a current hash-bound "
        "execution plan and bounded request ledger. Preserve old reports for audit; "
        "use a controlled plan-bound confirmation path. No config, release, task, "
        "or evaluator is loaded here."
    )

    if os.environ.get("RSI_HTTP_NO_CACHE") != "1":
        print("!! 必须设 RSI_HTTP_NO_CACHE=1 再跑：")
        print("   HTTP 缓存键由请求 body 决定，而 seed 不进 body（temperature=0），")
        print("   不禁缓存的话第 2 个 seed 起会 100% 命中缓存，等于白跑。")
        return 2

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seeds = args.seeds or [int(s) for s in
                           (cfg.get("evaluation", {}).get("seeds") or [20260924])]
    datasets = args.datasets or list(cfg.get("evaluation", {}).get("datasets", []))
    limit = args.limit if (args.limit and args.limit > 0) else None

    champ_id = args.champion or rel.current_release_id()
    champion = rel.load_state(champ_id)
    challenger = rel.load_state(args.challenger)

    print(f"champion  : {champ_id}")
    print(f"challenger: {args.challenger}")
    print(f"datasets  : {datasets}")
    print(f"seeds     : {seeds}  (每 seed 各跑 champion+challenger 一次)")
    print(f"limit     : {limit or '全部'}")
    print(flush=True)

    from src.eval.harness import evaluate  # noqa: E402
    from src.util import sha256_obj  # noqa: E402
    cfg_fp = sha256_obj(cfg)

    # 直接复用 compare_on_split（它只调 evaluate，不碰封存集）
    rep = compare_on_split(cfg, champion, challenger, datasets, split=args.split,
                           seeds=seeds, limit=limit)

    agg = aggregate(rep, args.tol_primary, args.tol_unsupported)

    out = Path(args.out) if args.out else (
        OUT_ROOT / f"multiseed-{args.challenger}-{len(seeds)}seed.json")
    write_json(out, agg)

    print()
    hdr = (f"{'数据集':12s} {'n':>4s} {'Δ均值':>9s} {'Δ极差':>8s} "
           f"{'噪声地板':>9s} {'acc极差':>8s} {'无支撑Δ':>8s}  判定")
    print(hdr)
    print("-" * len(hdr))
    for ds, v in agg["per_dataset"].items():
        dm = v["primary_delta_mean_pp"]
        um = v["unsupported_delta_mean_pp"]
        print(f"{ds:12s} {v['n_tasks'] or 0:4d} "
              f"{(f'{dm:+.2f}pp' if dm is not None else 'n/a'):>9s} "
              f"{v['primary_delta_span_pp']:7.2f}pp "
              f"{v['noise_floor_pp']:8.2f}pp "
              f"{v['champion_acc_span_pp']:7.2f}pp "
              f"{(f'{um:+.2f}pp' if um is not None else 'n/a'):>8s}  "
              f"{v['verdict']}")
    print()
    print(f"总判定: {agg['overall']}")
    for ds, v in agg["per_dataset"].items():
        for r in v["reasons"]:
            print(f"  [{ds}] {r}")
    print()
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
