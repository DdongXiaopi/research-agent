"""把 SuppQA 封存成**全新的 final_holdout**（A3 / Spec §6.1 铁律 3）。

为什么必须重封一份
------------------
`artifacts/final_access.log` 里已经有 43 条访问记录、其中 **6 轮是针对候选的终评**
（M4 初版 / M4 role 修正 / M4+ / M4+ 含 CS / M1M2 回顾 / M1M2 回滚修复）。
每一轮的理由（协议变了必须同协议重跑）都成立，但客观结果是：
现在报的数是**第 6 次揭盲**。序贯窥视会系统性抬高假阳性，
Spec「final 只跑一次」这条铁律事实上已经失效。

asta-bench / LitQA2 的应对纪律是：公开集可以反复调，**held-out 预先注册、只揭一次**。
要恢复这条纪律，唯一干净的办法是换一份从未被看过的封存集 —— 于是选 SuppQA：

  - 82 题，`question` + `ideal`（论文里的真实答案）+ 3 个**人工撰写**的干扰项；
  - 标签来源与本项目任何启发式**无关**（这同时治好了 CS 集那种
    「gold = 与问题实词重叠最高的句子」的循环论证，见 A2）；
  - 自带 canary GUID，可进泄漏检查的第二条通道；
  - 干扰项是**近义数值/术语**（如 `0.5 × S ≤ L ≤ 1.5 × S` vs `0.8 × S ≤ L ≤ 1.2 × S`），
    语义上正是"看着像但无证据支撑"，与 LitQA2 的设计意图一致。

适配规则（预先注册、确定性、可审计）
------------------------------------
1. 源：`vendor/SuppQA/train-00000-of-00001.parquet`（82 行）。
2. 选项 = `[ideal] + distractors`（4 项），位置用
   `random.Random(sha256(dataset|id))` 稳定打乱，避免 gold 恒在末位。
3. `evidence` 字段只放**论文题名与出处**（`paper-title` / `source`），
   **不放答案**；真实语料检索需要另行准备可用的合法全文。
4. canary 字符串**不进题面**（否则会被当成题干噪声），只写进 manifest。
5. 全量封存为 final_holdout；`data/normalized/suppqa.jsonl` 为空
   （进化器永不看见，物理保证）。

产物
----
    data/normalized/suppqa.jsonl              空
    data/sealed/suppqa.final_holdout.jsonl    82 题
    data/manifests/suppqa.json                manifest（含 canary、冻结校验）
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import MANIFEST_DIR, NORMALIZED_DIR, SEALED_DIR  # noqa: E402
from src.util import ensure_dir, now_iso, sha256_file, sha256_obj, write_json, write_jsonl  # noqa: E402

ADAPTATION_RULE_VERSION = "suppqa-final-v1"
DATASET = "suppqa"
DOMAIN = "biomedical"

SOURCE = "vendor/SuppQA/train-00000-of-00001.parquet"
CANARY_FIELD = "canary"      # 只入 manifest，不入题面
N_DISTRACTORS = 3


def _clean(s: Any) -> str:
    return " ".join(str(s or "").split()).strip()


def build_rows(src: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    import pandas as pd

    df = pd.read_parquet(src)
    canaries: List[str] = []
    tasks: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for i in range(len(df)):
        r = df.iloc[i].to_dict()
        qid = str(r.get("id") or f"row-{i}")
        question = _clean(r.get("question"))
        ideal = _clean(r.get("ideal"))
        raw_d = r.get("distractors")
        dists = [_clean(x) for x in (list(raw_d) if raw_d is not None else [])]
        canary = _clean(r.get(CANARY_FIELD))
        if canary:
            canaries.append(canary)

        if not question or not ideal:
            dropped.append({"id": qid, "reason": "question 或 ideal 为空"})
            continue
        dists = [d for d in dists if d and d.lower() != ideal.lower()]
        if len(dists) < N_DISTRACTORS:
            dropped.append({"id": qid, "reason": f"可用干扰项不足 {N_DISTRACTORS} 个"})
            continue

        options = [ideal] + dists[:N_DISTRACTORS]
        rng = random.Random(int(sha256_obj(f"{DATASET}|{qid}")[:16], 16))
        shuffled = list(options)
        rng.shuffle(shuffled)
        correct_index = shuffled.index(ideal)

        title = _clean(r.get("paper-title"))
        source = _clean(r.get("source"))
        evidence = ". ".join(x for x in (title, source) if x) or title or question

        tasks.append({
            "task_id": f"suppqa-{i:03d}",
            "dataset": DATASET,
            "domain": DOMAIN,
            "split": "final_holdout",
            "task_type": "mcq",
            "question": question,
            "options": shuffled,
            "correct_index": correct_index,
            "evidence": evidence,
            "context": "",
            "sources": [source] if source else [],
            "source_row_id": qid,
        })

    stats = {
        "n_source": int(len(df)),
        "n_tasks": len(tasks),
        "n_dropped": len(dropped),
        "dropped_examples": dropped[:10],
        "n_options": N_DISTRACTORS + 1,
        "n_canary_markers": len(set(canaries)),
        "note": "gold 来自数据集的 ideal 字段（论文真实答案），与本项目任何启发式无关",
    }
    return tasks, stats, sorted(set(canaries))


def main() -> int:
    src = Path(SOURCE)
    if not src.exists():
        print(f"[suppqa] 源数据不存在：{src}")
        return 1

    tasks, stats, canaries = build_rows(src)
    if not tasks:
        print("[suppqa] 没有构造出任何题目")
        return 1

    tasks.sort(key=lambda t: t["task_id"])
    ids = sorted(t["task_id"] for t in tasks)
    manifest = {
        "dataset": DATASET,
        "created_at": now_iso(),
        "source_file": str(src).replace("\\", "/"),
        "source_sha256": sha256_file(src),
        "domain": DOMAIN,
        "task_type": "mcq",
        "n_questions": len(tasks),
        "salt": "not-applicable",
        "ratios": {"dev_optimize": 0, "selection_gate": 0, "final_holdout": 100},
        "split_algorithm": "full_sealed_holdout（全量封存，进化器零可见，Spec §6.1）",
        "role": "primary_final",
        "replaces": {
            "datasets": ["litqa2", "protocolqa", "scholarqa_cs"],
            "reason": ("既有 final_holdout 已被 6 轮候选终评消费（43 条审计记录），"
                       "序贯揭盲抬高假阳性；三者的 final 降级为 dev2，"
                       "只用于分析、不得再作为最终结论。"),
        },
        "adaptation": {
            "rule_version": ADAPTATION_RULE_VERSION,
            "options": "ideal + 3 个人工干扰项，sha256(dataset|id) 稳定打乱",
            "evidence": "仅论文题名与出处；真实语料需另行准备合法全文",
            "canary": "只写进 manifest，不进题面",
        },
        "canary_markers": canaries[:3],
        "stats": stats,
        "caveat": ("SuppQA 的 evidence 由论文题名/出处构造（离线无全文），"
                   "绝对准确率低于真实全文检索下的表现；只用于同协议前后对比。"),
        "splits": {
            "dev_optimize": {"n": 0, "ids": [], "ids_sha256": sha256_obj([])},
            "selection_gate": {"n": 0, "ids": [], "ids_sha256": sha256_obj([])},
            "final_holdout": {"n": len(ids), "ids": ids, "ids_sha256": sha256_obj(ids)},
        },
    }
    from scripts.make_splits import _commit_prepared
    _commit_prepared([(DATASET, manifest, [], tasks)])

    print(json.dumps({
        "ok": True,
        "dataset": DATASET,
        "manifest": f"data/manifests/{DATASET}.json",
        "sealed": f"data/sealed/{DATASET}.final_holdout.jsonl",
        "stats": stats,
        "caveat": manifest["caveat"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
