"""D6：成本门容差标定（2026-09-27）。

问题：`gates.accept.tolerance.cost_pct = 15` 是 fake 时代拍的，真实模型下从未标定
（caveat #3：champion 在 litqa2 上 cost_pct +22.92%，会被拦；当时靠"维持并披露"
绕过，没有降级）。于是这个门处在两难：既可能太紧（拦掉真实可接受的波动），
也可能太松（放过真实涨价）。

做法：同一对 release（champion vs baseline-v1-cited）在同一 split 上重复跑 B 批，
直接测 **cost_pct 本身的跨批波动**，容差取 k×σ。
  * 单侧成本的相对波动为 σ，则 cost_pct = (C−B)/B 的波动约为 √2·σ
    （两侧独立）。建议容差 = 3·√2·σ_single，或直接用实测 cost_pct 的 3σ。
  * 本报告两种都给，取保守者。

已知缺口（如实登记）：噪声的时间尺度是**跨时段**的（caveat #10），
同批次内重复抓不到主噪声源。本脚本跑的是同批次多批 ⇒ 给出的是**下界**，
真实 σ 可能更大。跨时段采样需要在不同时间各跑一次（见输出的 caveat 字段）。

用法（必须禁缓存，否则第 2 批起 100% 命中缓存 = 白跑）：
  RSI_HTTP_NO_CACHE=1 python scripts/probe_cost_gate.py --batches 3 --limit 30
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config  # noqa: E402
from src.governance import release as rel  # noqa: E402
from src.eval.harness import evaluate  # noqa: E402

OUT = ROOT / "artifacts" / "probe" / "cost_gate_calibration.json"

DEFAULT_CONFIG = "configs/experiment.http.pubmedqa-final.yaml"
BASE_RELEASE = "baseline-v1-cited"
CHALLENGER = "rel-cand-34bfab62fe04"


def _one(cfg, release_id: str, dataset: str, split: str, limit: int, seed: int,
         tag: str):
    st = rel.load_state(release_id)
    res = evaluate(cfg, st, dataset, split, seed=seed, limit=limit,
                   run_tag=tag, write_per_task=False, cfg_fp="cost-calib")
    m = res.metric_values
    return {
        "run_id": res.run_id,
        "n_tasks": m.get("n_tasks"),
        "prompt_tokens": m.get("prompt_tokens"),
        "completion_tokens": m.get("completion_tokens"),
        "cost_usd_nominal": m.get("cost_usd_nominal"),
        "cost_estimated": m.get("cost_estimated"),
        "p95_latency_ms": m.get("p95_latency_ms"),
    }


def main() -> int:
    raise SystemExit(
        "cost-gate calibration probe is retired: it scores repeated live release pairs "
        "without a current frozen request plan and shared call ledger; preserve historical "
        "reports and use a separately approved plan-bound experiment before reopening it"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--dataset", default="pubmedqa")
    ap.add_argument("--split", default="dev_optimize")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=20260924)
    ap.add_argument("--base-release", default=BASE_RELEASE)
    ap.add_argument("--challenger", default=CHALLENGER)
    args = ap.parse_args()

    cfg = load_config(args.config)
    batches: list[dict] = []
    for i in range(args.batches):
        seed = args.base_seed + i
        b = _one(cfg, args.base_release, args.dataset, args.split,
                 args.limit, seed, f"cost-calib-base{i}")
        c = _one(cfg, args.challenger, args.dataset, args.split,
                 args.limit, seed, f"cost-calib-cand{i}")
        bb = float(b["cost_usd_nominal"] or 0.0)
        cc = float(c["cost_usd_nominal"] or 0.0)
        pct = ((cc - bb) / bb * 100.0) if bb > 0 else None
        batches.append({"batch": i, "seed": seed, "baseline": b, "candidate": c,
                        "cost_pct": pct})
        print(f"batch {i}: base={bb:.6f} cand={cc:.6f} cost_pct="
              f"{'n/a' if pct is None else f'{pct:.2f}'}", flush=True)

    pcts = [x["cost_pct"] for x in batches if x["cost_pct"] is not None]
    base_costs = [float(x["baseline"]["cost_usd_nominal"] or 0) for x in batches]
    cand_costs = [float(x["candidate"]["cost_usd_nominal"] or 0) for x in batches]

    def rel_sd(xs: list[float]) -> float | None:
        if len(xs) < 2:
            return None
        mu = statistics.fmean(xs)
        if mu <= 0:
            return None
        return statistics.stdev(xs) / mu * 100.0

    sd_base = rel_sd(base_costs)
    sd_cand = rel_sd(cand_costs)
    sd_single = (statistics.fmean([s for s in (sd_base, sd_cand) if s is not None])
                 if any(s is not None for s in (sd_base, sd_cand)) else None)
    sd_pct = statistics.stdev(pcts) if len(pcts) >= 2 else None

    rec_from_single = (3.0 * math.sqrt(2.0) * sd_single) if sd_single else None
    rec_from_pct = (3.0 * sd_pct) if sd_pct else None
    cands = [r for r in (rec_from_single, rec_from_pct) if r is not None]
    recommended = max(cands) if cands else None  # 取保守（更宽）

    out = {
        "created_purpose": "D6 成本门容差标定（caveat #3）",
        "config": args.config,
        "dataset": args.dataset,
        "split": args.split,
        "limit_per_batch": args.limit,
        "batches": batches,
        "cost_pct_samples": pcts,
        "sd_single_side_pct": sd_single,
        "sd_single_side_detail": {"baseline": sd_base, "candidate": sd_cand},
        "sd_cost_pct": sd_pct,
        "recommended_tolerance_pct": recommended,
        "recommended_from": {
            "3*sqrt2*sd_single": rec_from_single,
            "3*sd_cost_pct": rec_from_pct,
        },
        "current_tolerance_pct": 15,
        "caveat": [
            "同批次多批抓不到跨时段噪声（caveat #10）：本读数是 σ 的**下界**，"
            "真实 σ 可能更大；严谨标定需在不同时段各跑一次。",
            f"样本仅 {len(pcts)} 批 × {args.limit} 题，σ 估计本身不稳定。",
            "cost_usd_nominal 用真实 token 计数但未登记单价（cost_estimated=true），"
            "此处用 nominal 值作相对比较，不用于金额结论。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 标定结果 ===")
    print(f"cost_pct 样本: {[None if p is None else round(p, 2) for p in pcts]}")
    print(f"单侧成本相对 σ: {None if sd_single is None else round(sd_single, 3)}%")
    print(f"cost_pct 的 σ:  {None if sd_pct is None else round(sd_pct, 3)}pp")
    print(f"当前容差 15% → 建议容差: "
          f"{None if recommended is None else round(recommended, 2)}%")
    print(f"产物: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
