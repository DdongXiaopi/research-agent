"""三层 split 冻结（M2 / Spec §4-M2 第 1 条：三层 split 冻结并哈希）。

为什么需要"冻结"这一步：
    M1 里 split 只是生成出来就用了。只要有人（或一次误操作）重跑 `make_splits.py`
    改了 salt / 比例 / 题目集合，历史 run 的可比性就全废了——
    而"gate 与 final 严格分离"这条结论恰恰建立在 split 不变之上。

冻结做什么：
    data/manifests/freeze.json 记录每个数据集的
      - manifest 全文 sha256
      - 每个 split 的题目 ID 列表 sha256（与 manifest 内 ids_sha256 交叉校验）
      - 源数据 parquet 的 sha256
      - salt / ratios / 算法
    以及一个整体 fingerprint。此后**每次加载题目都校验**，不一致即拒绝运行。

不做什么：
    - 不加密、不做权限系统。这是研究项目的完整性校验，不是安全边界。
    - final_holdout 的解锁仍由 split.load_tasks 的 RSI_ALLOW_FINAL 控制（M4 前不放开）。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .split import DEFAULT_SALT, MANIFEST_DIR, NORMALIZED_DIR, SEALED_DIR, SPLIT_ORDER, load_manifest
from ..util import now_iso, path, sha256_file, sha256_obj, write_json

FREEZE_FILE = MANIFEST_DIR / "freeze.json"


class SplitNotFrozen(RuntimeError):
    """尚未冻结。M2 起要求先冻结再跑进化。"""


class SplitIntegrityError(RuntimeError):
    """冻结后 split 被改动 —— 历史 run 不再可比，拒绝运行。"""


def require_unfrozen(dataset: str) -> None:
    """Refuse dataset preparation when its current identity has been frozen."""
    from .split import _validate_dataset_name
    _validate_dataset_name(dataset)
    if FREEZE_FILE.exists() and dataset in (load_freeze().get("datasets") or {}):
        raise SplitIntegrityError(f"cannot rewrite frozen dataset: {dataset}")


def _dataset_fingerprint(dataset: str) -> Dict[str, Any]:
    m = load_manifest(dataset)
    splits = m.get("splits", {})
    per_split = {}
    all_ids = []
    for name in SPLIT_ORDER:
        ids = splits.get(name, {}).get("ids", [])
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value.strip()
                                            for value in ids):
            raise SplitIntegrityError(f"{dataset}/{name}: invalid task IDs")
        all_ids.extend(ids)
        ids = sorted(ids)
        per_split[name] = {
            "n": len(ids),
            "ids_sha256": splits.get(name, {}).get("ids_sha256", ""),
            "recomputed_ids_sha256": sha256_obj(ids),
        }
    if len(all_ids) != len(set(all_ids)):
        raise SplitIntegrityError(f"{dataset}: duplicate task IDs across split manifest")
    return {
        "dataset": dataset,
        "source_file": m.get("source_file", ""),
        "source_sha256": m.get("source_sha256", ""),
        "salt": m.get("salt", ""),
        "ratios": m.get("ratios", {}),
        "split_algorithm": m.get("split_algorithm", ""),
        "n_questions": m.get("n_questions", 0),
        "manifest_sha256": sha256_obj(m),
        "splits": per_split,
        "task_content": _task_content_fingerprint(dataset),
    }


def _task_content_fingerprint(dataset: str) -> Dict[str, Any]:
    """Hash actual visible bytes without opening protected questions/labels."""
    normalized = NORMALIZED_DIR / f"{dataset}.jsonl"
    if normalized.exists():
        return {"schema": "normalized-task-jsonl-v1", "sha256": sha256_file(normalized)}
    if dataset == "healthver_factorial_exploration":
        # This adapter has no materialized JSONL. Bind its frozen cohort and
        # verified raw archive; the loader independently verifies the archive.
        from .healthver_factorial_exploration import COHORT_PATH
        from .healthver_task_input import SOURCE_ARCHIVE
        return {"schema": "healthver-factorial-adapter-v1", "files": {
            "cohort": sha256_file(COHORT_PATH), "source": sha256_file(SOURCE_ARCHIVE)}}
    raise FileNotFoundError(f"visible normalized task content missing: {dataset}")


def freeze(datasets: Iterable[str], operator: str = "rsi-cli",
           note: str = "") -> Dict[str, Any]:
    """冻结当前 manifest 状态，写入 freeze.json。重复冻结且内容一致 → 幂等。"""
    ds_list = list(datasets)
    parts: Dict[str, Any] = {}
    for ds in ds_list:
        fp = _dataset_fingerprint(ds)
        # 冻结时立刻自检：manifest 内自带的 ids_sha256 必须与重算值一致
        for name, info in fp["splits"].items():
            if info["ids_sha256"] and info["ids_sha256"] != info["recomputed_ids_sha256"]:
                raise SplitIntegrityError(
                    f"{ds}/{name}: manifest 内 ids_sha256 与实际 ID 列表不一致，拒绝冻结"
                )
        parts[ds] = fp

    if FREEZE_FILE.exists():
        prior = load_freeze()
        old_parts = prior.get("datasets") or {}
        if prior.get("schema") == "split-freeze-content-v2":
            for ds, previous in old_parts.items():
                actual = parts.get(ds) or _dataset_fingerprint(ds)
                if actual != previous:
                    raise SplitIntegrityError(f"{ds}: cannot overwrite an existing content freeze")
                parts[ds] = previous
            if set(parts) == set(old_parts):
                require_verified()
                return prior
            ds_list = sorted(parts)

    sealed = {}
    for ds in ds_list:
        p = SEALED_DIR / f"{ds}.final_holdout.jsonl"
        final_count = parts[ds]["splits"]["final_holdout"]["n"]
        if final_count and (not p.is_file() or p.stat().st_size == 0):
            raise SplitIntegrityError(f"{ds}: final_holdout IDs require a nonempty sealed file")
        if not final_count and p.exists() and p.stat().st_size:
            raise SplitIntegrityError(f"{ds}: sealed content exists without final_holdout IDs")
        sealed[ds] = sha256_file(p) if p.exists() else ""

    payload: Dict[str, Any] = {
        "schema": "split-freeze-content-v2",
        "frozen_at": now_iso(),
        "operator": operator,
        "note": note or "M2：三层 split 冻结；final_holdout 解锁里程碑 = M4",
        "policy": {
            "final_holdout_unlock_milestone": "M4",
            "final_holdout_access": "evaluator_only",
            "salt_frozen": True,
        },
        "datasets": parts,
        "sealed_sha256": sealed,
    }
    payload["freeze_sha256"] = sha256_obj(parts)
    previous = FREEZE_FILE.read_bytes() if FREEZE_FILE.exists() else None
    try:
        write_json(FREEZE_FILE, payload)
        report = verify()
        if not report["ok"]:
            raise SplitIntegrityError("new split freeze failed verification: "
                                      + ", ".join(f"{row['dataset']}.{row['field']}"
                                                  for row in report["mismatches"]))
    except BaseException:
        if previous is None:
            FREEZE_FILE.unlink(missing_ok=True)
        else:
            FREEZE_FILE.write_bytes(previous)
        raise
    return payload


def load_freeze() -> Dict[str, Any]:
    """读取冻结记录；不存在则抛 SplitNotFrozen（M2 起进化入口强制要求）。"""
    import json

    if not FREEZE_FILE.exists():
        raise SplitNotFrozen(
            "尚未冻结 split。请先运行 `scripts/rsi.py freeze`（M2 起为强制步骤）。"
        )
    with open(FREEZE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_frozen() -> bool:
    return FREEZE_FILE.exists()


def upgrade_content_freeze(operator: str = "rsi-cli") -> Dict[str, Any]:
    """Start a new content-bound epoch; preserve the legacy receipt verbatim.

    This migration cannot retroactively prove historical question identity.
    It hashes sealed files as bytes and never parses protected labels.
    """
    prior = load_freeze()
    if all(row.get("task_content") for row in prior.get("datasets", {}).values()):
        require_verified()
        return prior
    report = verify(require_content=False)
    if not report["ok"]:
        raise SplitIntegrityError("legacy manifest/sealed identity must verify before upgrade")
    prior_hash = sha256_file(FREEZE_FILE)
    archive = FREEZE_FILE.with_name("freeze-legacy-" + prior_hash + ".json")
    if not archive.exists():
        archive.write_bytes(FREEZE_FILE.read_bytes())
    payload = freeze(prior["datasets"], operator=operator,
                     note="Content-bound epoch begins here; historical content identity remains unverified")
    payload["supersedes"] = {"file": archive.name, "sha256": prior_hash,
                              "historical_task_content_verified": False}
    write_json(FREEZE_FILE, payload)
    return payload


def verify(datasets: Optional[Iterable[str]] = None, *, require_content: bool = True) -> Dict[str, Any]:
    """比对当前 manifest / sealed 文件与冻结记录。返回结构化报告（不抛异常）。"""
    if not FREEZE_FILE.exists():
        return {"ok": False, "frozen": False,
                "reason": "no freeze.json（未冻结）", "mismatches": []}
    frozen = load_freeze()
    want = list(datasets) if datasets else sorted(frozen.get("datasets", {}).keys())
    mismatches: List[Dict[str, Any]] = []

    for ds in want:
        if ds not in frozen.get("datasets", {}):
            mismatches.append({"dataset": ds, "field": "dataset", "detail": "冻结记录中没有该数据集"})
            continue
        expect = frozen["datasets"][ds]
        try:
            actual = _dataset_fingerprint(ds)
        except (FileNotFoundError, SplitIntegrityError) as exc:
            mismatches.append({"dataset": ds, "field": "manifest", "detail": str(exc)})
            continue
        for field in ("salt", "ratios", "split_algorithm", "n_questions",
                      "source_sha256", "manifest_sha256"):
            # Compare structured manifest values directly. Stringifying dicts
            # makes equivalent mappings depend on insertion order after JSON
            # round-tripping (notably the three split ratios).
            if actual.get(field) != expect.get(field):
                mismatches.append({
                    "dataset": ds, "field": field,
                    "expected": expect.get(field), "actual": actual.get(field),
                })
        if require_content and (not expect.get("task_content")
                                or actual.get("task_content") != expect.get("task_content")):
            mismatches.append({"dataset": ds, "field": "task_content",
                "detail": "actual visible task bytes changed or legacy freeze lacks content identity"})
        for name in SPLIT_ORDER:
            a = actual["splits"].get(name, {})
            e = expect["splits"].get(name, {})
            if a.get("n") != e.get("n") or a.get("recomputed_ids_sha256") != e.get("recomputed_ids_sha256"):
                mismatches.append({
                    "dataset": ds, "field": f"splits.{name}",
                    "expected": {"n": e.get("n"), "ids_sha256": e.get("recomputed_ids_sha256")},
                    "actual": {"n": a.get("n"), "ids_sha256": a.get("recomputed_ids_sha256")},
                })

        sp = SEALED_DIR / f"{ds}.final_holdout.jsonl"
        expected_final_count = int(
            (expect.get("splits", {}).get("final_holdout") or {}).get("n", 0))
        expected_sealed = (frozen.get("sealed_sha256") or {}).get(ds, "")
        if sp.exists():
            actual_sealed = sha256_file(sp)
            if expected_sealed and actual_sealed != expected_sealed:
                mismatches.append({
                    "dataset": ds, "field": "sealed_file",
                    "expected": expected_sealed, "actual": actual_sealed,
                })
            elif not expected_sealed and expected_final_count > 0:
                mismatches.append({
                    "dataset": ds, "field": "sealed_file",
                    "detail": "final_holdout IDs exist but the frozen record has no sealed hash",
                })
            elif not expected_sealed and expected_final_count == 0 and sp.stat().st_size > 0:
                mismatches.append({
                    "dataset": ds, "field": "sealed_file",
                    "detail": "sealed content exists although final_holdout has zero IDs",
                })
        elif expected_final_count > 0 or expected_sealed:
            mismatches.append({"dataset": ds, "field": "sealed_file", "detail": "封存文件缺失"})

    return {
        "ok": not mismatches,
        "frozen": True,
        "frozen_at": frozen.get("frozen_at"),
        "freeze_sha256": frozen.get("freeze_sha256"),
        "checked": want,
        "mismatches": mismatches,
    }


def require_verified(datasets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """评测入口调用：未冻结或校验失败直接抛错，拒绝带着脏数据跑进化。"""
    rep = verify(datasets)
    if not rep.get("frozen"):
        raise SplitNotFrozen(str(rep.get("reason")))
    if not rep["ok"]:
        raise SplitIntegrityError(
            "split 冻结校验失败（历史 run 已不可比）："
            + "; ".join(f"{m['dataset']}.{m['field']}" for m in rep["mismatches"])
        )
    return rep
