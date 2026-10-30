"""固化数据集与三层 split（Spec 附录 D 第 2 步）。

用法：
    python -m pip install -r requirements-data.txt
    python scripts/make_splits.py --litqa2-parquet <licensed-file> --protocolqa-parquet <licensed-file>

产物：
    data/manifests/<dataset>.json   本地清单；确认许可与隐私后才可显式提交
    data/normalized/<dataset>.jsonl dev_optimize + selection_gate（进化器可见）
    data/sealed/<dataset>.final_holdout.jsonl   M4 前封存，仅 evaluator 可加载
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import (  # noqa: E402
    DEFAULT_RATIOS,
    DEFAULT_SALT,
    MANIFEST_DIR,
    NORMALIZED_DIR,
    SEALED_DIR,
    SPLIT_ORDER,
    assign_split,
)
from src.util import ensure_dir, now_iso, sha256_file, sha256_obj, sha256_text, write_json, write_jsonl  # noqa: E402

SOURCES = {
    "litqa2": {
        "parquet": "vendor/LitQA2/train-00000-of-00001.parquet",
        "domain": "biomedical",
        "context_field": None,
        "evidence_field": "key-passage",
        "task_type": "mcq",
    },
    "protocolqa": {
        "parquet": "vendor/ProtocolQA/train-00000-of-00001.parquet",
        "domain": "biomedical",
        "context_field": "protocol",
        "evidence_field": "protocol",
        "task_type": "mcq",
    },
}


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    try:
        return [str(x) for x in list(v) if x is not None]
    except TypeError:
        return []


def normalize(dataset: str, df: Any, salt: str) -> List[Dict[str, Any]]:
    cfg = SOURCES[dataset]
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        qid = str(r["id"]).strip()
        ideal = str(r["ideal"]).strip()
        distractors = [str(d).strip() for d in _as_list(r["distractors"])]
        options = [ideal] + distractors
        # 选项顺序按题目 ID 稳定打乱（不依赖 pandas 行序），保证可重放
        rng = random.Random(sha256_text(f"options|{dataset}|{qid}"))
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_index = shuffled.index(ideal)

        ctx = ""
        if cfg["context_field"] and cfg["context_field"] in df.columns:
            v = r[cfg["context_field"]]
            ctx = "" if v is None else str(v)
        ev = ""
        if cfg["evidence_field"] and cfg["evidence_field"] in df.columns:
            v = r[cfg["evidence_field"]]
            ev = "" if v is None else str(v)

        rows.append(
            {
                "task_id": qid,
                "dataset": dataset,
                "question": str(r["question"]).strip(),
                "options": shuffled,
                "correct_index": correct_index,
                "split": assign_split(dataset, qid, salt),
                "context": ctx,
                "evidence": ev,
                "sources": _as_list(r["sources"]) if "sources" in df.columns else [],
                "domain": cfg["domain"],
                "task_type": cfg["task_type"],
            }
        )
    return rows


def validate_input(dataset: str, df: Any) -> None:
    """Reject ambiguous task identities before any split artifact is written."""
    import pandas as pd

    required = {"id", "question", "ideal", "distractors", SOURCES[dataset]["evidence_field"]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{dataset}: missing Parquet columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError(f"{dataset}: Parquet contains no tasks")
    ids = []
    for index, row in df.iterrows():
        for field in ("id", "question", "ideal"):
            value = row[field]
            if not pd.api.types.is_scalar(value) or pd.isna(value) or not str(value).strip():
                raise ValueError(f"{dataset}: row {index} has an empty or invalid {field}")
        qid = str(row["id"]).strip()
        ids.append(qid)
        if not _as_list(row["distractors"]):
            raise ValueError(f"{dataset}: row {index} has no distractors")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: duplicate task IDs in Parquet")


def _install_file(pending: Path, target: Path) -> None:
    os.replace(pending, target)


def _commit_prepared(prepared: list[tuple[str, dict, list, list]]) -> None:
    """Stage all outputs and restore prior files if a replacement fails."""
    from src.data.freeze import FREEZE_FILE
    if FREEZE_FILE.exists():
        frozen = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
        blocked = {dataset for dataset, _, _, _ in prepared} & set(frozen.get("datasets") or {})
        if blocked:
            raise ValueError("cannot rewrite frozen datasets: " + ", ".join(sorted(blocked)))
    data_root = MANIFEST_DIR.parent
    ensure_dir(data_root)
    with tempfile.TemporaryDirectory(prefix=".split-stage-", dir=data_root) as directory:
        stage = Path(directory)
        entries = []
        for dataset, manifest, visible, sealed in prepared:
            for target, value, writer in (
                (MANIFEST_DIR / f"{dataset}.json", manifest, write_json),
                (NORMALIZED_DIR / f"{dataset}.jsonl", visible, write_jsonl),
                (SEALED_DIR / f"{dataset}.final_holdout.jsonl", sealed, write_jsonl),
            ):
                relative = target.relative_to(data_root)
                pending = stage / "pending" / relative
                writer(pending, value)
                backup = stage / "backup" / relative
                if target.exists():
                    ensure_dir(backup.parent)
                    shutil.copy2(target, backup)
                entries.append((pending, target, backup if target.exists() else None))
        replaced = []
        try:
            for pending, target, backup in entries:
                ensure_dir(target.parent)
                _install_file(pending, target)
                replaced.append((target, backup))
        except BaseException:
            for target, backup in reversed(replaced):
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            raise


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SOURCES),
                        default=list(SOURCES), help="datasets to prepare")
    parser.add_argument("--litqa2-parquet", type=Path,
                        default=Path("vendor/LitQA2/train-00000-of-00001.parquet"))
    parser.add_argument("--protocolqa-parquet", type=Path,
                        default=Path("vendor/ProtocolQA/train-00000-of-00001.parquet"))
    args = parser.parse_args(argv)
    inputs = {"litqa2": args.litqa2_parquet,
              "protocolqa": args.protocolqa_parquet}
    selected = list(dict.fromkeys(args.datasets))
    missing = [f"{dataset}: {inputs[dataset]}" for dataset in selected
               if not inputs[dataset].is_file()]
    if missing:
        parser.error("Parquet input missing; supply licensed local files: " + "; ".join(missing))
    try:
        import pandas as pd
    except ImportError as exc:
        parser.error("Parquet support missing; install requirements-data.txt")
        raise AssertionError("unreachable") from exc
    from src.data.freeze import FREEZE_FILE
    if FREEZE_FILE.exists():
        frozen = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
        blocked = set(selected) & set(frozen.get("datasets") or {})
        if blocked:
            parser.error("cannot rewrite frozen datasets: " + ", ".join(sorted(blocked)))
    prepared = []
    for dataset in selected:
        cfg = SOURCES[dataset]
        pq = inputs[dataset]
        df = pd.read_parquet(pq)
        validate_input(dataset, df)
        rows = normalize(dataset, df, DEFAULT_SALT)

        by_split: Dict[str, List[str]] = {s: [] for s in SPLIT_ORDER}
        for r in rows:
            by_split[r["split"]].append(r["task_id"])

        manifest = {
            "dataset": dataset,
            "source_file": str(pq).replace("\\", "/"),
            "source_sha256": sha256_file(pq),
            "n_questions": len(rows),
            "salt": DEFAULT_SALT,
            "ratios": DEFAULT_RATIOS,
            "split_algorithm": "sha256(salt|dataset|task_id)[:8] % 100 -> cumulative ratios",
            "task_type": cfg["task_type"],
            "created_at": now_iso(),
            "splits": {
                s: {
                    "n": len(ids),
                    "ids": sorted(ids),
                    "ids_sha256": sha256_obj(sorted(ids)),
                }
                for s, ids in by_split.items()
            },
        }
        visible = [r for r in rows if r["split"] != "final_holdout"]
        sealed = [r for r in rows if r["split"] == "final_holdout"]
        prepared.append((dataset, manifest, visible, sealed))

    _commit_prepared(prepared)
    for dataset, manifest, visible, sealed in prepared:
        by_split = {name: manifest["splits"][name]["ids"] for name in SPLIT_ORDER}
        print(
            f"[ok] {dataset}: total={manifest['n_questions']} "
            + " ".join(f"{s}={len(by_split[s])}" for s in SPLIT_ORDER)
            + f"  (visible={len(visible)}, sealed={len(sealed)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
