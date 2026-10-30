"""Run train-only GEPA against one Skill body in the frozen v2 cohort."""

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
from src.evolution.gepa_search import visible_examples  # noqa: E402
from src.evolution.gepa_skill_search import search_skill_body_candidate  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.util import artifacts_path, read_json, sha256_file, write_json  # noqa: E402
from scripts.pubmedqa_threeway_protocol import (  # noqa: E402
    gold_label_counts, verify_dataset_snapshot,
)
from scripts.run_pubmedqa_train_only_gepa import resolve_source_protocol  # noqa: E402


LABEL_SEMANTICS = (
    "PubMedQA's native labels are not a generic confidence scale. Following "
    "Jin et al. (EMNLP-IJCNLP 2019), label YES when experiments/results indicate "
    "the answer under the studied context, even if it may not hold universally; "
    "label NO when they do not. Label MAYBE when the paper supports the answer "
    "as true in some conditions and false in others, or a compound question has "
    "mixed true/false results. For a broad question, inspect reported strata or "
    "conditions; do not collapse them into one YES/NO and do not use MAYBE solely "
    "for low confidence or missing evidence. This rubric describes benchmark labels; "
    "it does not authorize adding facts absent from task material."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True,
                        help="frozen v2 discovery baseline report")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--skill", default="mixed-results-synthesis")
    parser.add_argument("--release", default=None)
    parser.add_argument("--seed", type=int, default=202610051)
    parser.add_argument("--metric-calls", type=int, default=90)
    parser.add_argument("--proposals", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--skill-routing-mode", choices=["lexical", "model"],
                        default=None)
    parser.add_argument("--validate-only", action="store_true",
                        help="validate frozen train cohort and target Skill without model calls")
    args = parser.parse_args()
    raise SystemExit(
        "retired_unplanned_entrypoint: this legacy train-only Skill GEPA CLI has "
        "no caller-challenged execution plan or request ledger; use a controlled, "
        "source-bound runner after its exact plan and request cap are approved")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("train-only Skill GEPA requires an explicit isolated RSI_HOME")

    source_path = Path(args.source_report).resolve()
    source = read_json(source_path)
    if source.get("dataset") != "pubmedqa_threeway_probe":
        raise ValueError("Skill GEPA source must be the frozen PubMedQA three-way cohort")
    verify_dataset_snapshot(source)
    train_ids = [str(value) for value in source.get("train_ids") or []]
    if len(train_ids) < 9 or len(train_ids) != len(set(train_ids)):
        raise ValueError("source report must contain at least nine unique train IDs")
    forbidden = set(source.get("discovery_ids") or []) | set(
        source.get("confirmation_ids") or [])
    if set(train_ids) & forbidden:
        raise ValueError("train IDs overlap discovery or confirmation IDs")
    release_id, routing_mode = resolve_source_protocol(
        source, args.release, args.skill_routing_mode)

    cfg = copy.deepcopy(load_config(args.config))
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input", "task_input_max_chars": 30000,
        "pi_request_cooldown_seconds": 6,
    })
    cfg["runtime"].pop("pi_fake_response", None)
    cfg.setdefault("evaluation", {}).update({
        "insufficient_option": False, "main_metric": "accuracy",
        "primary_axis": "accuracy", "sealed_pairing": False,
    })
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    cfg.setdefault("skills", {})["routing_mode"] = routing_mode
    cfg["evolution"]["allowed_artifacts"] = ["skill_body"]
    gepa = cfg["evolution"].setdefault("gepa", {})
    gepa.update({
        "enabled": True,
        "visible_task_ids": train_ids,
        "stratify_by_gold_label": True,
        "target_artifact": "skill_body",
        "target_skill": args.skill,
        "max_metric_calls": args.metric_calls,
        "max_candidate_proposals": args.proposals,
        "reflection_minibatch_size": args.minibatch,
        "acceptance_criterion": "strict_improvement",
        "feedback_evidence_signals": True,
        "multi_objective": False,
        "reflection_label_semantics": LABEL_SEMANTICS,
        "reflection_label_semantics_source": (
            "Jin et al., PubMedQA, Appendix C, EMNLP-IJCNLP 2019"),
    })
    state = load_state(release_id)
    target = state.registry.get(args.skill)
    if target is None or target.frozen:
        raise ValueError(f"target Skill missing or frozen in release {release_id}: {args.skill}")
    if source.get("release_id") != state.release_id:
        raise ValueError("loaded Skill release differs from frozen source report")
    outer_train, outer_validation = visible_examples(
        cfg, [str(source["dataset"])], len(train_ids))

    if args.validate_only:
        print(json.dumps({
            "status": "valid",
            "source_report_sha256": sha256_file(source_path),
            "release_id": state.release_id,
            "skill": args.skill,
            "skill_count": len(state.registry.skills),
            "skill_routing_mode": routing_mode,
            "train_id_count": len(train_ids),
            "outer_train_count": len(outer_train),
            "outer_validation_count": len(outer_validation),
            "outer_train_label_counts": gold_label_counts(outer_train),
            "outer_validation_label_counts": gold_label_counts(outer_validation),
            "visible_task_ids_exact": gepa["visible_task_ids"] == train_ids,
            "discovery_confirmation_accessed": False,
            "model_calls": 0,
            "config_sha256": config_fingerprint(cfg),
        }, ensure_ascii=False))
        return 0

    before = {path.resolve() for path in
              artifacts_path("search", "gepa_skill").glob("*/report.json")}
    error = None
    candidate = None
    try:
        candidate = search_skill_body_candidate(
            cfg, state, [str(source["dataset"])], len(train_ids), args.seed,
            failure_ids=[], skill_name=args.skill)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    reports = [path for path in artifacts_path("search", "gepa_skill").glob("*/report.json")
               if path.resolve() not in before]
    search_report_path = max(reports, key=lambda path: path.stat().st_mtime) if reports else None
    static = static_gate(candidate, cfg, state) if candidate else None
    report = {
        "protocol": "pubmedqa-threeway-v2-train-only-skill-gepa-v1-exploratory",
        "source_report": str(source_path),
        "source_report_sha256": sha256_file(source_path),
        "dataset": source["dataset"],
        "dataset_manifest_sha256": source["dataset_manifest_sha256"],
        "dataset_normalized_sha256": source["dataset_normalized_sha256"],
        "train_ids": train_ids,
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
        "skill_routing_mode": routing_mode,
        "target_artifact": "skill_body",
        "target_skill": args.skill,
        "label_semantics_source": gepa["reflection_label_semantics_source"],
        "label_semantics": LABEL_SEMANTICS,
        "config_sha256": config_fingerprint(cfg),
        "candidate": candidate.to_dict() if candidate else None,
        "candidate_status": candidate.status if candidate else "no_candidate",
        "candidate_static_gate": static,
        "search_report_path": str(search_report_path) if search_report_path else None,
        "search_error": error,
        "eligible_for_discovery": bool(candidate and static and static.get("passed")),
    }
    output = artifacts_path("probe", "pubmedqa_v2_skill_routing",
                            "train-only-skill-gepa-" + uuid.uuid4().hex[:12] + ".json")
    write_json(output, report)
    print(json.dumps({
        "report": str(output),
        "search_report": str(search_report_path) if search_report_path else None,
        "candidate_id": candidate.candidate_id if candidate else None,
        "candidate_status": report["candidate_status"],
        "candidate_static_gate_passed": bool(static and static.get("passed")),
        "train_id_count": len(train_ids),
        "discovery_confirmation_accessed": False,
        "search_error": error,
    }, ensure_ascii=False))
    return 2 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
