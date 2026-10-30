"""Join completed HealthVer prediction changes to gold rationales offline.

Rationale snippets are read only after scoring. This script never changes task
inputs or sends gold rationale text to a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.healthver_task_input import (  # noqa: E402
    SOURCE_ARCHIVE, SOURCE_SHA256, SOURCE_SPLITS, load_tasks, verify_source_archive,
)
from src.util import read_json, sha256_file, write_json  # noqa: E402

DEFAULT_REPORT = ROOT / (
    "artifacts/sandboxes/healthver-mipro-fresh-components-20261007/probe/"
    "healthver_mipro/05a1f50d2fc0437d/report.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts/probe/healthver_mipro_fresh_error_attribution.json"


def _jsonl_member(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    stream = archive.extractfile(name)
    if stream is None:
        raise FileNotFoundError(f"HealthVer archive member missing: {name}")
    return [json.loads(line) for line in stream.read().decode("utf-8").splitlines()
            if line.strip()]


def build_audit(report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    report = read_json(report_path)
    if report.get("protocol") != "healthver-component-disjoint-mipro-v2":
        raise ValueError("expected the completed component-disjoint HealthVer MIPRO report")
    baseline_path = Path(report["internal_baseline_checkpoint"])
    candidate_path = Path(report["internal_candidate_checkpoint"])
    baseline_rows = [json.loads(line) for line in baseline_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    candidate_rows = [json.loads(line) for line in candidate_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    baseline = {(int(row["seed"]), str(row["task_id"])): row for row in baseline_rows}
    candidate = {(int(row["seed"]), str(row["task_id"])): row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate checkpoints are not paired")

    changed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(baseline):
        old, new = baseline[key], candidate[key]
        if int(old.get("answer_index", -1)) == int(new.get("answer_index", -1)):
            continue
        changed[key[1]].append({
            "execution_label": key[0],
            "gold_label": old.get("gold_label"),
            "baseline_answer_index": old.get("answer_index"),
            "candidate_answer_index": new.get("answer_index"),
            "baseline_correct": bool(old.get("correct")),
            "candidate_correct": bool(new.get("correct")),
        })

    task_by_id = {str(row["task_id"]): row for row in load_tasks(["dev_optimize"])}
    verify_source_archive()
    corpus: dict[int, dict[str, Any]] = {}
    claims: dict[int, dict[str, Any]] = {}
    with tarfile.open(SOURCE_ARCHIVE, mode="r:gz") as archive:
        for row in _jsonl_member(archive, "data/healthver/corpus.jsonl"):
            corpus[int(row["doc_id"])] = row
        for split in SOURCE_SPLITS:
            for row in _jsonl_member(archive, f"data/healthver/claims_{split}.jsonl"):
                claims[int(row["id"])] = row

    audited = []
    for task_id, changes in sorted(changed.items()):
        task = task_by_id.get(task_id)
        if task is None:
            raise ValueError(f"changed prediction task missing from HealthVer adapter: {task_id}")
        metadata = task.get("metadata") or {}
        claim_id, abstract_id = int(metadata["claim_id"]), int(metadata["abstract_id"])
        claim_row = claims.get(claim_id)
        document = corpus.get(abstract_id)
        if claim_row is None or document is None:
            raise ValueError(f"source rationale lookup failed for {task_id}")
        raw_annotations = (claim_row.get("evidence") or {}).get(str(abstract_id), [])
        rationales = []
        for annotation in raw_annotations:
            indices = [int(index) for index in annotation.get("sentences") or []]
            sentences = document.get("abstract") or []
            if any(index < 0 or index >= len(sentences) for index in indices):
                raise ValueError(f"invalid gold rationale sentence index in {task_id}")
            rationales.append({
                "source_label": str(annotation.get("label") or "").upper(),
                "sentence_indices": indices,
                "text": " ".join(str(sentences[index]) for index in indices),
            })
        audited.append({
            "task_id": task_id,
            "claim_id": claim_id,
            "abstract_id": abstract_id,
            "gold_document_label": task["options"][int(task["correct_index"])],
            "claim": task["question"],
            "source_title": document.get("title") or "",
            "prediction_changes": changes,
            "source_gold_rationales": rationales,
        })

    transitions = Counter()
    for row in audited:
        for change in row["prediction_changes"]:
            if change["candidate_correct"] and not change["baseline_correct"]:
                transitions["improved"] += 1
            elif change["baseline_correct"] and not change["candidate_correct"]:
                transitions["regressed"] += 1
            else:
                transitions["changed_but_still_wrong"] += 1
    return {
        "schema_version": 1,
        "protocol": "healthver-offline-rationale-error-attribution-v1",
        "mipro_report_sha256": sha256_file(report_path),
        "candidate_id": report.get("candidate_id"),
        "source_archive_sha256": SOURCE_SHA256,
        "source_archive_file_sha256": sha256_file(SOURCE_ARCHIVE),
        "changed_task_count": len(audited),
        "changed_prediction_row_count": sum(len(row["prediction_changes"]) for row in audited),
        "transition_counts": dict(sorted(transitions.items())),
        "rationale_text_used_as_runtime_input": False,
        "rationale_text_used_for_candidate_selection": False,
        "confirmation_eligible": False,
        "tasks": audited,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    audit = build_audit(args.report)
    if args.write:
        write_json(args.output, audit)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output),
        "candidate_id": audit["candidate_id"],
        "changed_tasks": audit["changed_task_count"],
        "changed_prediction_rows": audit["changed_prediction_row_count"],
        "transition_counts": audit["transition_counts"],
        "rationales_were_model_inputs": False,
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
