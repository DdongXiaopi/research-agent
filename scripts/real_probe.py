"""真实模型跑测探针（不参与离线回归，不写 ACTIVE，不碰 final_holdout）。

用法：
    set SILICONFLOW_API_KEY=...
    python scripts/real_probe.py --config configs/experiment.http.yaml ^
        --datasets litqa2 protocolqa --splits dev_optimize selection_gate ^
        --releases baseline-v0 --limit 20

产出：artifacts/real_probe/<tag>.json + stdout 摘要。
只跑可见分区；`final_holdout` 会被显式拒绝（suppqa 已决定不揭盲）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config_fingerprint, load_config          # noqa: E402
from src.eval.harness import evaluate                            # noqa: E402
from src.governance import release as rel                        # noqa: E402
from src.runtime.provider import build_provider                  # noqa: E402
from src.util import artifacts_path, ensure_dir, now_iso, write_json  # noqa: E402

FORBIDDEN_SPLITS = {"final_holdout"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.http.yaml")
    ap.add_argument("--datasets", nargs="+", default=["litqa2", "protocolqa"])
    ap.add_argument("--splits", nargs="+", default=["dev_optimize", "selection_gate"])
    ap.add_argument("--releases", nargs="+", default=["baseline-v0"])
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--limit", type=int, default=0, help="0 = 不限")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    raise SystemExit(
        "retired_unplanned_entrypoint: real_probe can score arbitrary releases and "
        "splits without a frozen execution plan, caller hash challenge, or bounded "
        "request ledger; use a plan-bound evaluation entrypoint"
    )

    bad = FORBIDDEN_SPLITS.intersection(args.splits)
    if bad:
        print(f"拒绝执行：真实跑测不得触碰 {sorted(bad)}（suppqa 已决定不揭盲）", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    fp = config_fingerprint(cfg)
    provider = build_provider(cfg, seed=args.seed)
    limit = args.limit or None

    report: Dict[str, Any] = {
        "created_at": now_iso(),
        "config": args.config,
        "config_fingerprint": fp,
        "model": cfg.get("models", {}).get("task", {}).get("model_id"),
        "seed": args.seed,
        "limit": args.limit,
        "runs": [],
    }
    t0 = time.time()
    for release_id in args.releases:
        state = rel.load_state(release_id)
        for ds in args.datasets:
            for split in args.splits:
                res = evaluate(cfg, state, ds, split, seed=args.seed, limit=limit,
                               run_tag=f"real:{release_id}", write_per_task=(split == "dev_optimize"),
                               cfg_fp=fp, provider=provider,
                               out_root=artifacts_path("real_probe", "runs"))
                m = res.metric_values
                report["runs"].append({
                    "release_id": release_id,
                    "dataset": ds,
                    "split": split,
                    "run_id": res.run_id,
                    "n_tasks": res.n_tasks,
                    "accuracy": m.get("accuracy"),
                    "abstain_rate": m.get("abstain_rate"),
                    "citation_precision": m.get("citation_precision"),
                    "p95_latency_ms": m.get("p95_latency_ms"),
                    "prompt_tokens": m.get("prompt_tokens"),
                    "completion_tokens": m.get("completion_tokens"),
                    "cost": res.cost,
                })
                print(f"[{release_id}] {ds}/{split}: acc={m.get('accuracy')} "
                      f"abstain={m.get('abstain_rate')} n={res.n_tasks} "
                      f"p95={m.get('p95_latency_ms')}ms")

    report["provider_stats"] = getattr(provider, "stats", {})
    report["elapsed_s"] = round(time.time() - t0, 1)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    out = artifacts_path("real_probe", f"probe-{tag}.json")
    ensure_dir(out.parent)
    write_json(out, report)

    print("\n=== provider stats ===")
    print(json.dumps(report["provider_stats"], ensure_ascii=False))
    print(f"elapsed: {report['elapsed_s']}s")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
