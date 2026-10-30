"""数据集三层划分与加载（Spec §6.1）。

数据集在 HF 上只有 train split，因此按 `sha256(salt|dataset|question_id)` 稳定划分，
salt 与划分结果固化进 manifest，保证任何机器上重现同一份 split。
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional

from ..util import path, read_json, read_jsonl, sha256_text

SPLIT_ORDER = ("dev_optimize", "selection_gate", "final_holdout")
DEFAULT_RATIOS = {"dev_optimize": 60, "selection_gate": 20, "final_holdout": 20}
DEFAULT_SALT = "rsi-2026-09-24"

MANIFEST_DIR = path("data", "manifests")
NORMALIZED_DIR = path("data", "normalized")
SEALED_DIR = path("data", "sealed")


def assign_split(dataset: str, question_id: str, salt: str = DEFAULT_SALT,
                 ratios: Dict[str, int] | None = None) -> str:
    """按稳定 hash 划分。同一 (dataset, qid, salt) 永远得到同一分区。"""
    _validate_dataset_name(dataset)
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("question_id must be nonempty text")
    if not isinstance(salt, str) or not salt.strip():
        raise ValueError("split salt must be nonempty text")
    ratios = DEFAULT_RATIOS if ratios is None else ratios
    if (not isinstance(ratios, dict) or set(ratios) != set(SPLIT_ORDER)
            or any(type(value) is not int or value < 0 for value in ratios.values())
            or sum(ratios.values()) != 100):
        raise ValueError("split ratios must be three non-negative integer percentages totaling 100")
    digest = sha256_text(f"{salt}|{dataset}|{question_id}")
    bucket = int(digest[:8], 16) % 100
    acc = 0
    for name in SPLIT_ORDER:
        acc += int(ratios.get(name, 0))
        if bucket < acc:
            return name
    return SPLIT_ORDER[-1]


def assign_all(dataset: str, question_ids: Iterable[str], salt: str = DEFAULT_SALT,
               ratios: Dict[str, int] | None = None) -> Dict[str, str]:
    return {qid: assign_split(dataset, qid, salt, ratios) for qid in question_ids}


class FinalHoldoutSealed(RuntimeError):
    """final_holdout 在 M4 前不可被进化器读取（Spec 红线）。"""


def _validate_dataset_name(dataset: str) -> None:
    if not isinstance(dataset, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset):
        raise ValueError("dataset must be a simple dataset identifier")


def load_tasks(dataset: str,
               splits: Optional[Iterable[str]] = None,
               allow_final: bool = False,
               limit: Optional[int] = None) -> List[Dict]:
    """加载归一化题目。

    final_holdout 默认拒绝加载；仅 evaluator（显式 allow_final=True 且
    环境变量 RSI_ALLOW_FINAL=1）可读取。
    """
    _validate_dataset_name(dataset)
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a non-negative integer")
    if isinstance(splits, (str, bytes)):
        raise ValueError("splits must be an iterable of split names, not a string")
    splits = tuple(splits) if splits is not None else None
    if dataset == "pubmedqa_threeway_probe":
        from .pubmedqa_threeway_probe import load_tasks as load_threeway_probe
        return load_threeway_probe(
            splits=list(splits) if splits is not None else None, limit=limit)
    if dataset == "evidence_inference_v2_abstract":
        requested = set(splits) if splits is not None else None
        if allow_final or (requested and "final_holdout" in requested):
            raise FinalHoldoutSealed(
                "Evidence Inference official test has no project final_holdout mapping and remains sealed"
            )
        from .evidence_inference_v2 import load_tasks as load_evidence_inference
        return load_evidence_inference(
            splits=list(splits) if splits is not None else None,
            limit=limit,
        )
    if dataset == "clinifact_fixed_abstract":
        if allow_final:
            raise FinalHoldoutSealed(
                "CliniFact official test has no project final_holdout mapping and remains sealed")
        from .clinifact_task_input import load_tasks as load_clinifact
        return load_clinifact(
            splits=list(splits) if splits is not None else None,
            limit=limit,
        )
    if dataset == "healthver_entailment_task_input":
        from .healthver_task_input import load_tasks as load_healthver
        return load_healthver(
            splits=list(splits) if splits is not None else None,
            limit=limit,
        )
    if dataset == "scifact_claim_verification":
        from .scifact_task_input import (
            SealedSciFactTestError, load_tasks as load_scifact,
        )
        requested = list(splits) if splits is not None else None
        if allow_final:
            raise SealedSciFactTestError(
                "SciFact has no project final holdout; official test remains sealed")
        return load_scifact(splits=requested, limit=limit)
    if dataset == "healthver_factorial_exploration":
        from .healthver_factorial_exploration import load_tasks as load_healthver_factorial
        return load_healthver_factorial(
            splits=list(splits) if splits is not None else None,
            limit=limit,
        )
    want = set(splits) if splits is not None else None
    if want and "final_holdout" in want and not (allow_final and os.environ.get("RSI_ALLOW_FINAL") == "1"):
        raise FinalHoldoutSealed(
            "final_holdout 处于封存状态；进化器配置中不得出现其路径。"
            "仅 evaluator 入口可设置 RSI_ALLOW_FINAL=1 解锁。"
        )
    p = NORMALIZED_DIR / f"{dataset}.jsonl"
    from .freeze import is_frozen, load_freeze, require_verified
    if is_frozen() and dataset in (load_freeze().get("datasets") or {}):
        require_verified([dataset])
    rows = read_jsonl(p)
    if want and "final_holdout" in want:
        # 封存区与进化可见区物理分离：只有 evaluator 解锁时才读 data/sealed/
        rows = rows + read_jsonl(SEALED_DIR / f"{dataset}.final_holdout.jsonl")
    out = [r for r in rows if (want is None or r.get("split") in want)]
    if limit is not None:
        out = out[:limit]
    return out


def load_manifest(dataset: str) -> Dict:
    """读取 split manifest（题目 ID、split 归属、salt、数据集 revision）。"""
    _validate_dataset_name(dataset)
    if dataset == "evidence_inference_v2_abstract":
        from .evidence_inference_v2 import load_manifest as load_evidence_manifest
        return load_evidence_manifest()
    if dataset == "healthver_entailment_task_input":
        from .healthver_task_input import load_manifest as load_healthver_manifest
        return load_healthver_manifest()
    if dataset == "healthver_factorial_exploration":
        from .healthver_factorial_exploration import load_manifest as load_healthver_factorial_manifest
        return load_healthver_factorial_manifest()
    return read_json(MANIFEST_DIR / f"{dataset}.json")
