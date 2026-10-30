#!/usr/bin/env python
"""路线 B：把 PubMedQA（pqa_labeled）接成项目可用的数据集。

## 为什么是 PubMedQA

selection_gate 现在只有 litqa2 46 + protocolqa 29 = **75 题**。这个规模下
任何真实的候选差异都测不出来（McNemar p≥0.25，CI 宽 ±10pp）—— 判定功效
不足是"所有进化都看不出效果"的**方法学**原因，不是进化器不行。

PubMedQA 的三个特点正好对症：
1. **自带证据段落**（`context.contexts` 是 PubMed 摘要原文），不需要额外的
   检索语料即可评估"接地"能力 —— 项目其它数据集的证据池也是这么来的；
2. **1000 题、MIT 许可**，许可风险最低（LAB-Bench 那三份子集至今未确认）；
3. 领域（biomedical）与现有一致，不会引入跨域混淆。

## 刻意的取舍

- **丢掉 `maybe` 题**：`maybe` 与运行时注入的 "Insufficient information" 选项
  语义重叠，混在一起会让"弃权"这一轴无法解释。只保留 yes/no。
- **选项按题 ID 稳定打乱**：与 `make_splits.py` 一致，避免"正确答案总在第一个"。
- split 沿用同一套 `sha256(salt|dataset|task_id)` 算法，比例 60/20/20。

用法：
    python scripts/make_pubmedqa.py
    python scripts/make_pubmedqa.py --freeze     # 顺带固化
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.split import (  # noqa: E402
    DEFAULT_RATIOS,
    DEFAULT_SALT,
    MANIFEST_DIR,
    NORMALIZED_DIR,
    SEALED_DIR,
    SPLIT_ORDER,
    assign_split,
)
from src.util import (  # noqa: E402
    ensure_dir,
    now_iso,
    sha256_file,
    sha256_obj,
    sha256_text,
    write_json,
    write_jsonl,
)

DATASET = "pubmedqa"
PARQUET = Path("vendor/PubMedQA/train-00000-of-00001.parquet")
# 只保留 yes / no：maybe 与运行时注入的 "Insufficient information" 语义重叠
KEEP = ("yes", "no")
DOMAIN = "biomedical"
SOURCE_NOTE = ("qiaojin/PubMedQA (pqa_labeled)，MIT 许可；"
               "题面与选项来自原始标注，证据段落取自 PubMed 摘要原文。")


def _contexts_of(row: Any) -> List[str]:
    ctx = row.get("context")
    if isinstance(ctx, dict):
        raw = ctx.get("contexts")
        if raw is None:
            return []
        # numpy 数组：`or []` 会触发"真值不明确"，必须显式判空
        return [str(c) for c in list(raw)]
    if isinstance(ctx, (list, tuple)):
        return [str(c) for c in ctx]
    return []


def normalize(df: "pd.DataFrame", salt: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dropped = 0
    for _, r in df.iterrows():
        decision = str(r.get("final_decision", "")).strip().lower()
        if decision not in KEEP:
            dropped += 1
            continue
        qid = str(r.get("pubid"))
        options = ["yes", "no"]
        rng = random.Random(sha256_text(f"options|{DATASET}|{qid}"))
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_index = shuffled.index(decision)
        evidence = " ".join(_contexts_of(r)).strip()
        if not evidence:
            dropped += 1
            continue
        rows.append({
            "task_id": qid,
            "dataset": DATASET,
            "question": str(r.get("question", "")).strip(),
            "options": shuffled,
            "correct_index": correct_index,
            "split": assign_split(DATASET, qid, salt),
            "context": "",
            "evidence": evidence,
            "sources": [f"https://pubmed.ncbi.nlm.nih.gov/{qid}/"],
            "domain": DOMAIN,
            "task_type": "mcq",
        })
    print(f"[info] 归一化 {len(rows)} 题，丢弃 {dropped} 题（maybe / 无证据段落）")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="接入 PubMedQA 数据集")
    ap.add_argument("--freeze", action="store_true", help="生成后顺带固化 split")
    args = ap.parse_args()

    if not PARQUET.exists():
        print(f"[fail] parquet 不存在：{PARQUET}")
        print("       下载：https://hf-mirror.com/datasets/qiaojin/PubMedQA/resolve/main/"
              "pqa_labeled/train-00000-of-00001.parquet")
        return 1

    df = pd.read_parquet(PARQUET)
    rows = normalize(df, DEFAULT_SALT)
    if not rows:
        return 1

    by_split: Dict[str, List[str]] = {s: [] for s in SPLIT_ORDER}
    for r in rows:
        by_split[r["split"]].append(r["task_id"])

    manifest = {
        "dataset": DATASET,
        "source_file": str(PARQUET).replace("\\", "/"),
        "source_sha256": sha256_file(PARQUET),
        "source_url": "https://huggingface.co/datasets/qiaojin/PubMedQA",
        "license": "MIT（数据集声明）；PubMed 摘要原文的使用条款另行适用，"
                   "对外发布前仍需确认",
        "note": SOURCE_NOTE,
        "n_questions": len(rows),
        "salt": DEFAULT_SALT,
        "ratios": DEFAULT_RATIOS,
        "split_algorithm": "sha256(salt|dataset|task_id)[:8] % 100 -> cumulative ratios",
        "task_type": "mcq",
        "created_at": now_iso(),
        "splits": {
            s: {"n": len(ids), "ids": sorted(ids), "ids_sha256": sha256_obj(sorted(ids))}
            for s, ids in by_split.items()
        },
    }
    visible = [r for r in rows if r["split"] != "final_holdout"]
    sealed = [r for r in rows if r["split"] == "final_holdout"]
    from scripts.make_splits import _commit_prepared
    _commit_prepared([(DATASET, manifest, visible, sealed)])

    print(f"[ok] {DATASET}: total={len(rows)} "
          + " ".join(f"{s}={len(by_split[s])}" for s in SPLIT_ORDER)
          + f"  (visible={len(visible)}, sealed={len(sealed)})")

    if args.freeze:
        # 【事故复盘】`freeze()` 是**整体覆盖**而不是增量追加。只传新数据集会把
        # litqa2/protocolqa 等从冻结记录里挤掉，随后任何评估都会被
        # `require_verified` 拦下（SplitIntegrityError）。所以必须带上既有条目。
        from src.data.freeze import freeze, load_freeze, verify

        try:
            existing = sorted((load_freeze().get("datasets") or {}).keys())
        except FileNotFoundError:
            existing = []
        targets = sorted(set(existing) | {DATASET})
        out = freeze(targets, operator="make_pubmedqa")
        print(f"[ok] freeze: 冻结 {sorted(out['datasets'].keys())}（含既有 {existing}）")
        rep = verify()
        print(f"[{'ok' if rep['ok'] else 'FAIL'}] verify: {rep['ok']} "
              f"mismatches={rep.get('mismatches')}")
        if not rep["ok"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
