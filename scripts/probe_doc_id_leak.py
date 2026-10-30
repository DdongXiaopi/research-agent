"""诊断：语料里的 passage 标识符是否泄漏答案（对照实验）。

## 怀疑什么

`gateway` 给每篇文档编的 `doc_id` 是**带语义前缀**的：

    gold::<task_id>          ← 支撑正确答案的那篇
    cand::<task_id>::0/1/2   ← 干扰项
    filler::<task_id>::0/1   ← 背景

而 `HttpProvider._messages` 把标识符原样写进 prompt：

    Retrieved passages (identifier in brackets):
    [gold::0a9d...] <text>
    [filler::0a9d...::0] <text>

也就是说 **模型不用读正文，看一眼 `gold::` 就知道该引哪篇**。
真实检索系统里不存在这种字段 —— 这会让准确率与引用精度同时虚高。

（顺带排除了另一个怀疑：`title` 字段虽然也带语义（`Evidence for: <题干>`），
但它**不进 prompt**（`_messages` 只用 `doc_id` + `text`），所以不是泄漏通道。
把 title 改成中性后所有指标一字不变、且全部命中 HTTP 缓存，可证。）

## 怎么测

同一批题、同一 release、同一真实模型，只改一件事：把 doc_id 换成中性编号
（`P0/P1/...`），正文与顺序一字不动。差值即"标识符泄漏"的贡献。

用法（真实模型，串行）：

    SILICONFLOW_API_KEY=xxx python scripts/probe_doc_id_leak.py --limit 40
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.eval.harness import evaluate  # noqa: E402
from src.governance import release as rel  # noqa: E402
from src.retrieval import gateway as gw  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.util import artifacts_path, write_json  # noqa: E402

_ORIGINAL = gw.build_fixture_corpus


def _neutral_ids(task: Dict[str, Any], corpus_version: str = gw.DEFAULT_CORPUS_VERSION
                 ) -> List[Dict[str, Any]]:
    docs = _ORIGINAL(task, corpus_version)
    for i, d in enumerate(docs):
        d["doc_id"] = f"P{i}"
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.http.yaml")
    ap.add_argument("--release", default="baseline-v0")
    ap.add_argument("--compare-release", default=None,
                    help="同时跑第二个 release（通常是进化后的 ACTIVE）做对照")
    ap.add_argument("--datasets", nargs="+", default=["litqa2", "protocolqa"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()

    raise SystemExit(
        "unplanned document-id leakage scoring is retired: this historical "
        "diagnostic has no frozen execution-plan hash, request caps, or shared call ledger"
    )

    # Retained for historical review only. Restore execution only after adding
    # a caller-challenged plan, bounded request ledger, and frozen task checks.
    cfg = load_config(args.config)
    fp = config_fingerprint(cfg)
    provider = build_provider(cfg)
    releases = [args.release] + ([args.compare_release] if args.compare_release else [])
    out: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config, "limit": args.limit, "seed": args.seed, "results": {},
    }

    for rid in releases:
        state = rel.load_state(rid)
        for cond in ("as_is", "neutral_ids"):
            gw.build_fixture_corpus = _neutral_ids if cond == "neutral_ids" else _ORIGINAL
            out["results"].setdefault(rid, {}).setdefault(cond, {})
            for ds in args.datasets:
                res = evaluate(cfg, state, ds, "dev_optimize", seed=args.seed,
                               limit=args.limit, run_tag=f"docid-{cond}", cfg_fp=fp,
                               provider=provider)
                out["results"][rid][cond][ds] = {
                    "run_id": res.run_id, "n_tasks": res.n_tasks,
                    "metrics": {k: v for k, v in res.metric_values.items()
                                if k in ("accuracy", "coverage", "answered_precision",
                                         "citation_precision")},
                }
                print(f"[{rid}/{cond}] {ds} n={res.n_tasks} "
                      f"{out['results'][rid][cond][ds]['metrics']}", flush=True)
    gw.build_fixture_corpus = _ORIGINAL

    out["provider_stats"] = getattr(provider, "stats", {})
    p = artifacts_path("real_probe", f"doc_id_leak_{datetime.now():%Y%m%d-%H%M%S}.json")
    write_json(p, out)
    print(f"\n>>> {p}")
    print(json.dumps(out["results"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
