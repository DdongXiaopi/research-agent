#!/usr/bin/env python
"""R3：真实模型下的 2^3 全因子消融。

为什么要单独写这个脚本
    行为通道（prompt→rerank/citation、body→evidence/abstain、description→recall）
    的判据是**关键词**：提示词里出现 "cite" 就认为开了引用通道。
    真实模型读的是语义不是关键词，所以这条判据在真实模型上不成立 ——
    离线模拟器里"三路可分"的结论无法外推。R3 的做法是：**不看规则，只看数据**。

    直接用 `rsi.py ablate --prompt-release …` 有个坑：
    `build_factorial_conditions` 是**整个 registry 替换**，
    而现有 release 是混杂的（rel-cand-535716eb92b6 同时改了 description 和 body），
    于是 "只开 D" 的条件会偷偷把 body 也换掉，因子效应被污染。
    本脚本先把三个因子**拆干净**，再交给同一套全因子数学。

用法
    SILICONFLOW_API_KEY=... python scripts/real_factorial.py \
        --config configs/experiment.real-v3.yaml --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import load_config                       # noqa: E402
from src.eval import ablation                            # noqa: E402
from src.governance import release as rel                # noqa: E402
from src.runtime.runner import ArtifactState             # noqa: E402

# 三个因子各自的来源（用"改动最少"的那个 release，避免带入别的因子）
PROMPT_SRC = "rel-cand-34bfab62fe04"    # 唯一改了 system_prompt 的
DESC_SRC = "rel-cand-535716eb92b6"      # 改了 description（也改了 body，这里只取 description）
BODY_SRC = "rel-cand-912577c87582"      # 只改了 body —— 最干净
BASE = "baseline-v0"


def _clone_with(base: ArtifactState, *, descriptions_from: Any = None,
                bodies_from: Any = None) -> Any:
    """复制 baseline 的 registry，只替换指定的那一个维度。"""
    import copy

    src_d = {s.name: s for s in descriptions_from.skills} if descriptions_from else {}
    src_b = {s.name: s for s in bodies_from.skills} if bodies_from else {}
    reg = copy.deepcopy(base.registry)
    new_skills = []
    for s in reg.skills:
        ns = copy.deepcopy(s)
        if s.name in src_d:
            ns.description = src_d[s.name].description
        if s.name in src_b:
            ns.body_path = src_b[s.name].body_path
        new_skills.append(ns)
    reg.skills = new_skills
    return reg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.real-v3.yaml")
    ap.add_argument("--datasets", nargs="+", default=["litqa2", "protocolqa"])
    ap.add_argument("--splits", nargs="+", default=["dev_optimize", "selection_gate"])
    ap.add_argument("--limit", type=int, default=60, help="每数据集每分区的题数；0=全量")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    raise SystemExit(
        "historical real-factorial scoring is retired: use the current plan-bound "
        "task-input factorial or an explicit plan-governed ablation; this legacy "
        "entrypoint will not load releases, config, or providers"
    )

    cfg = load_config(args.config)
    base = rel.load_state(BASE)
    p_src = rel.load_state(PROMPT_SRC)
    d_src = rel.load_state(DESC_SRC)
    b_src = rel.load_state(BODY_SRC)

    desc_only = _clone_with(base, descriptions_from=d_src.registry)
    body_only = _clone_with(base, bodies_from=b_src.registry)
    both = _clone_with(base, descriptions_from=d_src.registry,
                       bodies_from=b_src.registry)

    conditions = ablation.build_factorial_conditions(
        base, p_src.system_prompt, desc_only, body_only, both)

    limit = args.limit or None
    rep = ablation.run_ablation(
        cfg, conditions, args.datasets, tuple(args.splits),
        seeds=(args.seed,), limit=limit)

    rep["factor_sources"] = {"P": PROMPT_SRC, "D": DESC_SRC, "B": BODY_SRC,
                             "base": BASE, "pure_factors": True}
    p = ablation.save(rep)
    print(json.dumps({"path": p, "conditions": {
        c: rep["conditions"][c]["label"] for c in rep["conditions"]}},
        ensure_ascii=False, indent=1))

    print("\n=== 主效应（accuracy, pp）===")
    for ds in args.datasets:
        for sp in args.splits:
            blk = rep["factorial"][ds][sp]
            eff = blk.get("effects") or {}
            main = {k: v for k, v in eff.items() if len(k) == 1}
            print(f"  {ds}/{sp}: " + "  ".join(
                f"{k}={v*100:+.2f}pp" for k, v in sorted(main.items())))
            print(f"      条件均值: " + " ".join(
                f"{c}={blk['conditions'][c]['aggregate']['accuracy']:.3f}"
                for c in sorted(blk.get("conditions", {}))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
