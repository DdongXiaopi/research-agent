"""新协议基线重跑（协议升级 2026-09-26：real-v3 语料 + 官方"信息不足"选项）。

为什么单独写这个脚本
    `scripts/rsi.py baseline` 只能跑 **当前 ACTIVE** release，而新基线需要同时给出
    - `baseline-v0`（进化起点）与
    - 当前 ACTIVE（进化后）
    两个 release 的数字，才能回答"换了语料和选项口径之后，进化收益还在不在"。
    直接 `rollback` 改 ACTIVE 指针会触发已知的 P1 问题（测试会重置指针），
    所以这里用 `load_state(release_id)` 显式加载，不动指针。

用法（真实模型，串行，禁用 HTTP 缓存保证每次都是真实调用）
    SILICONFLOW_API_KEY=xxx RSI_HTTP_NO_CACHE=1 \
      python scripts/rerun_baseline.py --config configs/experiment.http.yaml \
             --releases baseline-v0 rel-cand-34bfab62fe04 \
             --datasets litqa2 protocolqa

产物
    artifacts/real_probe/new_baseline_<ts>.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.eval.harness import evaluate  # noqa: E402
from src.governance import release as rel  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.util import artifacts_path, write_json  # noqa: E402

METRIC_KEYS = ["accuracy", "coverage", "answered_precision", "citation_precision",
               "abstain_rate", "skill_recall"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.http.yaml")
    ap.add_argument("--releases", nargs="+", default=["baseline-v0"])
    ap.add_argument("--datasets", nargs="+", default=["litqa2", "protocolqa"])
    ap.add_argument("--splits", nargs="+", default=["dev_optimize", "selection_gate"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--tag", default="new-baseline")
    args = ap.parse_args()

    raise SystemExit(
        "retired_unplanned_entrypoint: rerun_baseline compares arbitrary releases "
        "on configurable splits without a frozen execution plan, caller hash "
        "challenge, or bounded request ledger; use `rsi baseline` for a plan-bound "
        "current ACTIVE release run"
    )

    cfg = load_config(args.config)
    fp = config_fingerprint(cfg)
    provider = build_provider(cfg)

    out: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "config_fingerprint": fp,
        "corpus_version": cfg.get("retrieval", {}).get("corpus_version"),
        "insufficient_option": cfg.get("evaluation", {}).get("insufficient_option"),
        "seed": args.seed,
        "limit": args.limit,
        "releases": {},
    }

    for rid in args.releases:
        state = rel.load_state(rid)
        out["releases"][rid] = {"n_skills": len(state.registry.skills), "results": {}}
        for ds in args.datasets:
            for split in args.splits:
                res = evaluate(cfg, state, ds, split, seed=args.seed, limit=args.limit,
                               run_tag=args.tag, write_per_task=True, cfg_fp=fp,
                               provider=provider)
                out["releases"][rid]["results"][f"{ds}/{split}"] = {
                    "run_id": res.run_id,
                    "n_tasks": res.n_tasks,
                    "metrics": {k: res.metric_values.get(k)
                                for k in METRIC_KEYS if k in res.metric_values},
                    "cost": dict(res.cost or {}),
                }
                print(f"[{rid}] {ds}/{split} n={res.n_tasks} "
                      f"{out['releases'][rid]['results'][f'{ds}/{split}']['metrics']}",
                      flush=True)

    out["provider_stats"] = getattr(provider, "stats", {})
    p = artifacts_path("real_probe", f"new_baseline_{datetime.now():%Y%m%d-%H%M%S}.json")
    write_json(p, out)
    print(f"\n>>> {p}")
    print(json.dumps({"provider_stats": out["provider_stats"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
