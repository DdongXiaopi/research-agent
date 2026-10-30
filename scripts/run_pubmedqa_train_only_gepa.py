"""Run bounded prompt-only GEPA using only an explicit exploratory train cohort.

This command creates a draft candidate and never reads discovery, confirmation,
selection-gate, or final-holdout labels. Use an isolated RSI_HOME.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.evolution.gepa_search import search_prompt_candidate, visible_examples  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.util import artifacts_path, read_json, sha256_file, write_json  # noqa: E402
from scripts.pubmedqa_threeway_protocol import (  # noqa: E402
    gold_label_counts, verify_dataset_snapshot,
)


def resolve_source_protocol(source: dict, requested_release: str | None,
                            requested_routing_mode: str | None) -> tuple[str, str]:
    release_id = str(requested_release or source.get("release_id")
                     or "task-input-baseline-v1")
    source_release = source.get("release_id")
    if source_release is not None and release_id != str(source_release):
        raise ValueError("requested release differs from the frozen source protocol")
    routing_mode = str(requested_routing_mode or source.get(
        "skill_routing_mode", "lexical"))
    if routing_mode not in {"lexical", "model"}:
        raise ValueError("source report has an invalid Skill routing mode")
    if (requested_routing_mode is not None
            and source.get("skill_routing_mode") is not None
            and routing_mode != str(source["skill_routing_mode"])):
        raise ValueError("requested routing mode differs from the frozen source protocol")
    return release_id, routing_mode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True,
                        help="prior exploratory report that records train IDs")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--release", default=None,
                        help="defaults to the release frozen in --source-report")
    parser.add_argument("--seed", type=int, default=202610051)
    parser.add_argument("--metric-calls", type=int, default=90)
    parser.add_argument("--proposals", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--skill-routing-mode", choices=["lexical", "model"],
                        default=None,
                        help="defaults to the routing mode frozen in --source-report")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate source protocol and release without model calls")
    args = parser.parse_args()
    raise SystemExit(
        "retired_unplanned_entrypoint: this legacy train-only GEPA CLI has no "
        "caller-challenged execution plan or request ledger; use a controlled, "
        "source-bound runner after its exact plan and request cap are approved")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("train-only GEPA requires an explicit isolated RSI_HOME")

    source_path = Path(args.source_report).resolve()
    source = read_json(source_path)
    if source.get("dataset") == "pubmedqa_threeway_probe":
        verify_dataset_snapshot(source)
    task_ids = list(source.get("train_ids") or [])
    if len(task_ids) < 9 or len(task_ids) != len(set(task_ids)):
        raise ValueError("source report must contain at least nine unique train_ids")
    forbidden = set(source.get("discovery_ids", [])) | set(source.get("confirmation_ids", []))
    if set(task_ids) & forbidden:
        raise ValueError("source train_ids overlap discovery or confirmation IDs")
    release_id, skill_routing_mode = resolve_source_protocol(
        source, args.release, args.skill_routing_mode)

    cfg = copy.deepcopy(load_config(args.config))
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input", "task_input_max_chars": 30000,
        "pi_request_cooldown_seconds": 6,
    })
    cfg.setdefault("evaluation", {}).update({
        "insufficient_option": False, "main_metric": "accuracy", "primary_axis": "accuracy",
        "sealed_pairing": False,
    })
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("skills", {})["routing_mode"] = skill_routing_mode
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg.get("tools", {}).get("active", [])
        if name != "academic.search"]
    gp = cfg.setdefault("evolution", {}).setdefault("gepa", {})
    gp.update({
        "enabled": True,
        "visible_task_ids": task_ids,
        "reflection_label_semantics": (
            "PubMedQA's native labels are not a generic confidence scale. Following "
            "Jin et al. (EMNLP-IJCNLP 2019), label YES when the experiments/results "
            "indicate the answer under the studied context, even though it may not "
            "hold universally; label NO when they do not. Label MAYBE when the paper "
            "supports the answer as true in some conditions and false in others, or "
            "when a compound question covers multiple interventions/observations with "
            "mixed true/false results. For a broad question, inspect whether the "
            "reported strata or conditions yield different answers; do not collapse "
            "them into a single YES/NO and do not use MAYBE solely for low confidence "
            "or missing evidence. This rubric describes benchmark labels; it does not "
            "authorize adding facts absent from the supplied task material."
        ),
        "reflection_label_semantics_source": (
            "Jin et al., PubMedQA, Appendix C (Annotation Criteria), EMNLP-IJCNLP 2019"
        ),
        "task_offset": 0,
        "stratify_by_gold_label": True,
        "max_metric_calls": args.metric_calls,
        "max_candidate_proposals": args.proposals,
        "reflection_minibatch_size": args.minibatch,
        "acceptance_criterion": "strict_improvement",
        "feedback_evidence_signals": True,
        "multi_objective": False,
    })

    state = load_state(release_id)
    outer_train, outer_validation = visible_examples(
        cfg, [str(source.get("dataset", "pubmedqa_threeway_probe"))], len(task_ids))
    if args.validate_only:
        print(json.dumps({
            "status": "valid",
            "source_report": str(source_path),
            "source_report_sha256": sha256_file(source_path),
            "release_id": state.release_id,
            "skill_count": len(state.registry.skills),
            "skill_routing_mode": skill_routing_mode,
            "train_id_count": len(task_ids),
            "outer_train_count": len(outer_train),
            "outer_validation_count": len(outer_validation),
            "outer_train_label_counts": gold_label_counts(outer_train),
            "outer_validation_label_counts": gold_label_counts(outer_validation),
            "discovery_confirmation_accessed": False,
            "model_calls": 0,
            "config_sha256": config_fingerprint(cfg),
        }, ensure_ascii=False))
        return 0
    before = {p.resolve() for p in artifacts_path("search", "gepa").glob("*/report.json")}
    error = None
    candidate = None
    try:
        candidate = search_prompt_candidate(
            cfg, state, [str(source.get("dataset", "pubmedqa_threeway_probe"))],
            limit=len(task_ids), seed=args.seed, failure_ids=[])
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    reports = [p for p in artifacts_path("search", "gepa").glob("*/report.json")
               if p.resolve() not in before]
    search_report_path = max(reports, key=lambda p: p.stat().st_mtime) if reports else None
    report = {
        "protocol": "pubmedqa-threeway-train-only-gepa-v3-label-contract-exploratory",
        "source_report": str(source_path),
        "source_report_sha256": sha256_file(source_path),
        "source_protocol": source.get("protocol"),
        "dataset": str(source.get("dataset", "pubmedqa_threeway_probe")),
        "train_ids": task_ids,
        "outer_train_ids": [str(row["task_id"]) for row in outer_train],
        "outer_validation_ids": [str(row["task_id"]) for row in outer_validation],
        "outer_train_label_counts": gold_label_counts(outer_train),
        "outer_validation_label_counts": gold_label_counts(outer_validation),
        "discovery_ids_accessed": False,
        "confirmation_ids_accessed": False,
        "selection_gate_accessed": False,
        "final_holdout_accessed": False,
        "retrieval_used": False,
        "published": False,
        "release_id": state.release_id,
        "skill_routing_mode": skill_routing_mode,
        "target_artifact": "system_prompt",
        "config_sha256": config_fingerprint(cfg),
        "requested_metric_calls": args.metric_calls,
        "requested_proposals": args.proposals,
        "reflection_label_semantics_source": gp["reflection_label_semantics_source"],
        "reflection_label_semantics": gp["reflection_label_semantics"],
        "candidate_id": candidate.candidate_id if candidate else None,
        "candidate_status": candidate.status if candidate else "no_candidate",
        "candidate": candidate.to_dict() if candidate else None,
        "search_report_path": str(search_report_path) if search_report_path else None,
        "search_error": error,
        "eligible_for_confirmation": False,
    }
    output = artifacts_path("probe", "pubmedqa_threeway_v4",
                            "train-only-gepa-" + uuid.uuid4().hex[:12] + ".json")
    write_json(output, report)
    print(json.dumps({
        "report": str(output),
        "search_report": str(search_report_path) if search_report_path else None,
        "candidate_id": report["candidate_id"],
        "candidate_status": report["candidate_status"],
        "search_error": error,
        "train_id_count": len(task_ids),
        "discovery_confirmation_accessed": False,
    }, ensure_ascii=False))
    return 2 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
