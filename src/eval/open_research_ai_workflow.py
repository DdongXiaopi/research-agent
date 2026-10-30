"""Isolated AI rubric authoring for the open-research benchmark.

This is a thin, protocol-bound adapter around the existing task-first AI
annotation job.  It deliberately does not answer tasks or read the final
holdout during authoring; it only converts the frozen task-only cohort into a
dataset with independently authored, source-quoted rubrics.
"""
from __future__ import annotations

import copy
from pathlib import Path

from . import ai_annotation, ai_jobs

AUTHOR_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "g09_task_rubric",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array", "minItems": 1, "maxItems": 1,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "item_id": {"type": "string"},
                            "criterion": {"type": "string"},
                            "evidence_quotes": {
                                "type": "array", "minItems": 1, "maxItems": 1,
                                "items": {
                                    "type": "object", "additionalProperties": False,
                                    "properties": {
                                        "source_id": {"type": "string"},
                                        "quote": {"type": "string"},
                                    },
                                    "required": ["source_id", "quote"],
                                },
                            },
                        },
                        "required": ["item_id", "criterion", "evidence_quotes"],
                    },
                },
            },
            "required": ["items"],
        },
    },
}


def build_author_plan(dataset: dict, cfg: dict, *, max_calls: int | None = None,
                      max_tokens: int = 4096, max_cost_usd: float = 0.25) -> dict:
    """Freeze a zero-call authoring plan for every task in ``dataset``."""
    count = sum(len(rows) for rows in dataset.get("partitions", {}).values())
    # Reserve one bounded correction slot for up to 200 tasks while staying
    # within the shared AI-job maximum of 500 logical calls.
    frozen_calls = min(500, max(count, count + min(200, count))) if max_calls is None else max_calls
    return ai_annotation.author_plan(dataset, cfg,
        max_calls=frozen_calls,
        tokens=max_tokens, max_cost_usd=max_cost_usd,
        response_format=AUTHOR_RESPONSE_FORMAT)


def execute_author(dataset: dict, cfg: dict, plan: dict, *, provider=None) -> dict:
    """Execute the frozen authoring job and verify its private receipt."""
    result = ai_annotation.author(dataset, cfg, plan, provider=provider)
    verified = ai_jobs.verify_completed(result["run_dir"], kind="rubric_author")
    if verified["receipt"]["output_sha256"] != result["receipt"]["output_sha256"]:
        raise ValueError("AI author output receipt changed after verification")
    output = copy.deepcopy(result["output"])
    for split, cases in output["partitions"].items():
        for case in cases:
            if "rubric" not in case:
                raise ValueError(f"AI author omitted rubric for {split}")
    return {**result, "verified": verified, "output": output}


def write_run_artifacts(work_dir: Path, plan: dict, result: dict) -> dict:
    """Write a small public pointer without copying private raw model calls."""
    from ..util import write_json, sha256_obj

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(work_dir / "author-plan.json", plan)
    dataset_path = work_dir / "authored-dataset.json"
    write_json(dataset_path, result["output"])
    pointer = {
        "schema": "g09-open-research-ai-authoring-v1",
        "run_dir": result["run_dir"],
        "plan_sha256": plan["plan_sha256"],
        "dataset_sha256": sha256_obj(result["output"]),
        "dataset_path": str(dataset_path),
        "receipt": result["receipt"],
        "verified": True,
    }
    write_json(work_dir / "authoring-report.json", pointer)
    return pointer

