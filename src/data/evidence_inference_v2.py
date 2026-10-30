"""Load frozen, license-filtered Evidence Inference task-input splits.

The official training split is exposed as ``dev_optimize`` and official
validation as ``selection_gate``. The official test split is deliberately
absent and this loader has no final/unseal escape hatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..util import read_json, read_jsonl, sha256_file, sha256_obj


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data/manifests/evidence_inference_v2_abstract.json"
DATASET = "evidence_inference_v2_abstract"
PROTOCOL_VERSION = "evidence-inference-v2-abstract-task-input-v2-annotation-support"
SPLITS = ("dev_optimize", "selection_gate")


class SealedEvidenceInferenceTestError(PermissionError):
    """The official test split is not available through the evolution loader."""


def load_manifest() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    if manifest.get("dataset") != DATASET or manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Evidence Inference manifest identity/protocol mismatch")
    if manifest.get("eligible_for_confirmation") is not False:
        raise ValueError("Evidence Inference development data cannot be marked confirmation eligible")
    if manifest.get("candidate_runtime_uses_live_retrieval") is not False:
        raise ValueError("Evidence Inference candidate inputs must remain frozen task evidence")
    return manifest


def load_tasks(
    splits: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
    *,
    allow_final: bool = False,
) -> list[dict[str, Any]]:
    """Load train-derived optimize/gate tasks; refuse any final/test request."""
    requested = {"dev_optimize"} if splits is None else set(splits)
    if "final_holdout" in requested or "test" in requested:
        raise SealedEvidenceInferenceTestError(
            "Evidence Inference official test is sealed and is not materialized by this adapter"
        )
    if not requested or not requested.issubset(SPLITS):
        raise ValueError(f"unsupported Evidence Inference split request: {sorted(requested)}")
    if allow_final:
        raise SealedEvidenceInferenceTestError(
            "allow_final cannot unlock Evidence Inference test through the evolution adapter"
        )
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        if split not in requested:
            continue
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, dict):
            raise FileNotFoundError(f"Evidence Inference manifest lacks materialized split {split}")
        path = ROOT / str(entry.get("path") or "")
        if not path.is_file():
            raise FileNotFoundError(f"prepared Evidence Inference task input missing: {path}")
        if sha256_file(path) != entry.get("file_sha256"):
            raise ValueError(f"Evidence Inference {split} file hash differs from frozen manifest")
        split_rows = read_jsonl(path)
        task_ids = [str(row.get("task_id") or "") for row in split_rows]
        if len(task_ids) != int(entry.get("n", -1)) or sha256_obj(task_ids) != entry.get("ids_sha256"):
            raise ValueError(f"Evidence Inference {split} IDs differ from frozen manifest")
        project_split = split
        for row in split_rows:
            if row.get("dataset") != DATASET or row.get("split") != entry.get("source_split"):
                raise ValueError(f"Evidence Inference row escaped its frozen {split} mapping")
            rows.append({**row, "split": project_split})
    rows.sort(key=lambda row: (SPLITS.index(row["split"]), str(row["task_id"])))
    return rows if limit is None else rows[:limit]
