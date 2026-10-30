"""N4 协议变更后的基线重跑（selection_gate，三数据集 = 265 题）。

为什么必须重跑：证据呈现规则从 `hard-truncate-v1` 变成 `boundary-truncate-v2`，
protocolqa 的输入**不再是残缺的**。旧基线（证据被切掉最多 57.6%）与新基线
不同协议，系统已经通过指纹把它们判为不可比 —— 这正是我们要的效果，但代价是
历史数字不能直接拿来判定，必须在新协议下重立基线。

预期（也是一次可证伪的预测）：
- protocolqa 应**明显变化**（dev 消融预测 acc +11.1pp、弃权 −33.4pp）；
- litqa2 / pubmedqa **一题都不会变** —— 它们从没被截断过，输入逐字符一致。
  这两条是阴性对照：如果它们也变了，说明改动波及了不该波及的地方。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.harness import evaluate  # noqa: E402
from src.governance import release as rel  # noqa: E402
from src.util import path  # noqa: E402

DATASETS = ["litqa2", "protocolqa", "pubmedqa"]
SPLIT = "selection_gate"
SEED = 20260924


def load_config(rel_path: str) -> dict:
    import yaml

    return yaml.safe_load(path(rel_path).read_text(encoding="utf-8")) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description="N4 协议变更后重立 selection_gate 基线")
    ap.add_argument("--config", default="configs/experiment.http.pqa.yaml")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="n4-baseline")
    ap.add_argument("--out", default="artifacts/gate_verdicts/n4_baseline.json")
    args = ap.parse_args()

    raise SystemExit(
        "retired_unplanned_entrypoint: protocol-change baseline directly scores "
        "selection_gate without a frozen execution plan, caller hash challenge, "
        "or bounded request ledger; use a plan-bound baseline entrypoint"
    )

    cfg = load_config(args.config)
    state = rel.load_state()
    limit = args.limit if args.limit > 0 else None

    print(f"config        : {args.config}")
    print(f"max_context   : {cfg['models']['task'].get('max_context_chars')}")
    print(f"release(ACTIVE): {state.release_id}")
    print(f"datasets      : {args.datasets} @ {SPLIT}")
    print(flush=True)

    out = {"config": args.config, "split": SPLIT, "seed": args.seed,
           "release": state.release_id, "tag": args.tag, "runs": {}}
    for ds in args.datasets:
        t0 = time.time()
        res = evaluate(cfg, state, ds, SPLIT, seed=args.seed, limit=limit,
                       run_tag=args.tag, write_per_task=True, cfg_fp=args.tag)
        m = res.metric_values
        out["runs"][ds] = {"run_id": res.run_id, "n_tasks": res.n_tasks,
                           "metrics": m, "seconds": round(time.time() - t0, 1)}
        print(f"[{ds:11s}] n={res.n_tasks:3d} acc={m.get('accuracy'):.3f} "
              f"cov={m.get('coverage'):.3f} prec={m.get('answered_precision'):.3f} "
              f"unsup={m.get('unsupported_answer_rate')} "
              f"abstain_f1={m.get('abstention_f1')} ({time.time()-t0:.0f}s) "
              f"run={res.run_id}", flush=True)

    p = path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
