"""Personal research entry point; benchmark/evolution CLI remains scripts/rsi.py."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.runtime.research_assistant import build_plan, execute, request_from_dict
from src.runtime.research_tools import audit_open_research_trace
from src.evolution.session_feedback import export_development, record_feedback
from src.eval.open_research import assess_annotations, validate_group_partition
from src.util import artifacts_path, read_json, write_json


def _check_execution_plan(args, plan):
    expected = args.expected_plan_sha256
    if args.execute and not expected:
        raise ValueError(f"{args.command} execution requires an explicit frozen plan hash")
    if expected and expected != plan["plan_sha256"]:
        raise ValueError(f"{args.command} caller plan differs from current inputs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    author_ai = commands.add_parser("benchmark-ai-rubric")
    author_ai.add_argument("--input", type=Path, required=True)
    author_ai.add_argument("--config", required=True)
    author_ai.add_argument("--max-calls", type=int)
    author_ai.add_argument("--max-cost-usd", type=float, default=0.25)
    author_ai.add_argument("--expected-plan-sha256")
    author_ai.add_argument("--execute", action="store_true")
    annotate_ai = commands.add_parser("benchmark-ai-annotate")
    annotate_ai.add_argument("--evaluation-id", required=True)
    annotate_ai.add_argument("--config", required=True)
    annotate_ai.add_argument("--max-calls", type=int)
    annotate_ai.add_argument("--max-tokens", type=int, default=1200)
    annotate_ai.add_argument("--max-cost-usd", type=float, default=0.25)
    annotate_ai.add_argument("--expected-plan-sha256")
    annotate_ai.add_argument("--execute", action="store_true")
    for operation in ("discover", "lookup"):
        command = commands.add_parser(operation)
        command.add_argument("--query" if operation == "discover" else "--doi", required=True)
        command.add_argument("--providers", nargs="+")
        command.add_argument("--limit", type=int, default=5)
        command.add_argument("--max-requests", type=int, default=8)
        command.add_argument("--retries", type=int, default=1)
        command.add_argument("--timeout", type=int, default=15)
        command.add_argument("--expected-plan-sha256")
        command.add_argument("--execute", action="store_true")
    for operation in ("answer-discovered", "answer-documents"):
        discovered = commands.add_parser(operation)
        discovered.add_argument("--request", type=Path, required=True)
        discovered.add_argument("--snapshot", type=Path, required=True)
        discovered.add_argument("--config", default="configs/experiment.yaml")
        discovered.add_argument("--expected-plan-sha256")
        discovered.add_argument("--execute", action="store_true")
        if operation == "answer-documents":
            discovered.add_argument("--document-backend", choices=["local", "paperqa"], default="local")
            discovered.add_argument("--top-k", type=int, default=5)
    fulltext = commands.add_parser("fetch-fulltext")
    fulltext.add_argument("--snapshot", type=Path, required=True)
    fulltext.add_argument("--source-id", required=True)
    fulltext.add_argument("--max-bytes", type=int, default=8_000_000)
    fulltext.add_argument("--max-chars", type=int, default=500000)
    fulltext.add_argument("--timeout", type=int, default=20)
    fulltext.add_argument("--expected-plan-sha256")
    fulltext.add_argument("--execute", action="store_true")
    reference = commands.add_parser("verify-reference")
    reference.add_argument("--reference", type=Path, required=True)
    reference.add_argument("--snapshot", type=Path, required=True)
    citation = commands.add_parser("verify-citations")
    citation.add_argument("--run-dir", type=Path, required=True)
    trace_audit = commands.add_parser("audit-trace")
    trace_audit.add_argument("--run-dir", type=Path, required=True)
    tool_verify = commands.add_parser("verify-tool")
    tool_verify.add_argument("--run-dir", type=Path, required=True)
    tool = commands.add_parser("tool")
    tool.add_argument("--input", type=Path, required=True)
    tool.add_argument("--config", default="configs/experiment.yaml")
    tool.add_argument("--expected-plan-sha256")
    tool.add_argument("--execute", action="store_true")
    answer = commands.add_parser("answer")
    answer.add_argument("--request", type=Path, required=True)
    answer.add_argument("--config", default="configs/experiment.yaml")
    answer.add_argument("--execute", action="store_true")
    answer.add_argument("--expected-plan-sha256")
    feedback = commands.add_parser("feedback")
    feedback.add_argument("--run-dir", type=Path, required=True)
    feedback.add_argument("--category", choices=["preference", "evidence_fact", "procedure"], required=True)
    feedback.add_argument("--correction", required=True)
    feedback.add_argument("--reviewer", required=True)
    feedback.add_argument("--quotes", type=Path)
    feedback.add_argument("--use-for-evolution", action="store_true")
    export = commands.add_parser("export-dev")
    export.add_argument("--project", required=True)
    export.add_argument("--output", type=Path, required=True)
    assess = commands.add_parser("assess")
    assess.add_argument("--response", type=Path, required=True)
    assess.add_argument("--rubric", type=Path, required=True)
    assess.add_argument("--annotations", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    partitions = commands.add_parser("verify-partitions")
    partitions.add_argument("--input", type=Path, required=True)
    proposal = commands.add_parser("propose-skill")
    proposal.add_argument("--development-set", type=Path, required=True)
    proposal.add_argument("--config", required=True)
    proposal.add_argument("--expected-plan-sha256")
    proposal.add_argument("--execute", action="store_true")
    promote = commands.add_parser("promotion-request")
    promote.add_argument("--candidate", type=Path, required=True)
    promote.add_argument("--config", required=True)
    promote.add_argument("--development-id", required=True)
    promote.add_argument("--selection-id", required=True)
    promote.add_argument("--engineering-only", action="store_true")
    promotion_review = commands.add_parser("promotion-review")
    promotion_review.add_argument("--candidate-id", required=True)
    promotion_review.add_argument("--expected-request-sha256", required=True)
    promotion_review.add_argument("--decision", choices=["approve", "reject"], required=True)
    promotion_review.add_argument("--reviewer", required=True)
    promotion_review.add_argument("--reason", required=True)
    promotion_publish = commands.add_parser("promotion-publish")
    promotion_publish.add_argument("--candidate-id", required=True)
    promotion_publish.add_argument("--expected-request-sha256", required=True)
    curate = commands.add_parser("curate")
    curate.add_argument("--actions", type=Path, required=True)
    curate.add_argument("--output", type=Path, required=True)
    maintenance = commands.add_parser("migration-eval")
    maintenance.add_argument("--proposal", type=Path, required=True)
    maintenance.add_argument("--config", required=True)
    maintenance.add_argument("--negative-datasets", nargs="+", required=True)
    maintenance.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 71])
    maintenance.add_argument("--limit", type=int)
    maintenance.add_argument("--expected-plan-sha256")
    maintenance.add_argument("--execute", action="store_true")
    review = commands.add_parser("migration-review")
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--config", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--decision", choices=["approve", "reject"], required=True)
    review.add_argument("--expected-report-sha256", required=True)
    publish = commands.add_parser("migration-publish")
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--config", required=True)
    dataset = commands.add_parser("benchmark-freeze")
    dataset.add_argument("--input", type=Path, required=True)
    bench = commands.add_parser("benchmark-run")
    bench.add_argument("--dataset", type=Path, required=True)
    bench.add_argument("--config", required=True)
    bench.add_argument("--split", choices=["dev_optimize", "selection_gate", "final_holdout"], default="dev_optimize")
    bench.add_argument("--candidate-release")
    bench.add_argument("--candidate", type=Path)
    bench.add_argument("--expected-plan-sha256")
    bench.add_argument("--execute", action="store_true")
    packet = commands.add_parser("benchmark-packet")
    packet.add_argument("--evaluation-id", required=True)
    packet.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("benchmark-assess")
    score.add_argument("--evaluation-id", required=True)
    score.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {"benchmark-ai-rubric", "benchmark-ai-annotate"}:
        from src.eval import ai_annotation, open_benchmark
        cfg = load_config(args.config)
        inputs = read_json(args.input) if args.command == "benchmark-ai-rubric" else open_benchmark.annotation_packet(args.evaluation_id)
        budget = {"max_cost_usd": args.max_cost_usd}
        if args.max_calls is not None:
            budget["max_calls"] = args.max_calls
        if args.command == "benchmark-ai-annotate":
            budget["tokens"] = args.max_tokens
        plan = (ai_annotation.author_plan if args.command == "benchmark-ai-rubric" else ai_annotation.annotation_plan)(inputs, cfg, **budget)
        write_json(artifacts_path("research_plans", plan["plan_sha256"] + ".json"), plan)
        _check_execution_plan(args, plan)
        if args.execute:
            result = (ai_annotation.author if args.command == "benchmark-ai-rubric" else ai_annotation.annotate)(inputs, cfg, plan)
            if args.command == "benchmark-ai-annotate":
                result["assessment"] = open_benchmark.assess(args.evaluation_id, result["output"])
        else:
            result = {"status": "validated_not_run", **plan}
    elif args.command in {"discover", "lookup"}:
        from src.retrieval import literature
        plan = literature.build_plan(args.command, query=getattr(args, "query", ""), doi=getattr(args, "doi", ""),
            providers=args.providers, limit=args.limit, max_requests=args.max_requests, retries=args.retries, timeout=args.timeout)
        write_json(artifacts_path("research_plans", plan["plan_sha256"] + ".json"), plan)
        _check_execution_plan(args, plan)
        result = literature.execute(plan) if args.execute else {"status": "validated_not_run", **plan}
    elif args.command == "fetch-fulltext":
        from src.retrieval import documents
        snapshot = read_json(args.snapshot)
        plan = documents.build_plan(snapshot, args.source_id, max_bytes=args.max_bytes,
                                    max_chars=args.max_chars, timeout=args.timeout)
        write_json(artifacts_path("research_plans", plan["plan_sha256"] + ".json"), plan)
        _check_execution_plan(args, plan)
        result = (documents.execute(snapshot, plan, args.expected_plan_sha256) if args.execute else
                  {"status": "validated_not_run", **plan})
    elif args.command in {"answer", "answer-discovered", "answer-documents"}:
        value = read_json(args.request)
        retrieval = None
        if args.command in {"answer-discovered", "answer-documents"}:
            from src.retrieval.literature import evidence_from_snapshot
            retrieval = read_json(args.snapshot)
            if args.command == "answer-documents":
                from src.retrieval.documents import retrieval_snapshot
                retrieval = retrieval_snapshot(retrieval, value.get("query", ""),
                    backend=args.document_backend, top_k=args.top_k)
            if value.get("evidence"):
                raise ValueError("answer-discovered requires an empty evidence list; source merge must be explicit")
            value = {**value, "evidence": evidence_from_snapshot(retrieval)}
        request = request_from_dict(value)
        cfg = load_config(args.config)
        plan = build_plan(request, cfg, retrieval_snapshot=retrieval)
        plan_path = artifacts_path("research_plans", plan["plan_sha256"] + ".json")
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("saved research plan changed")
        write_json(plan_path, plan)
        _check_execution_plan(args, plan)
        if args.execute:
            result = execute(request, cfg, plan["plan_sha256"], retrieval_snapshot=retrieval)
        else:
            result = {"status": "validated_not_run", "plan_path": str(plan_path), **plan}
        if retrieval is not None:
            result["literature_snapshot_sha256"] = retrieval["snapshot_sha256"]
            result["evidence_scope"] = ("licensed full-text excerpts" if retrieval.get("schema") in {
                "fulltext-snapshot-v1", "document-evidence-snapshot-v1"} else
                "retrieved abstracts only; metadata and OA links excluded")
    elif args.command == "verify-reference":
        from src.retrieval.literature import verify_reference
        result = verify_reference(read_json(args.reference), read_json(args.snapshot))
    elif args.command == "verify-citations":
        from src.runtime.research_tools import verify_citations
        result = verify_citations(args.run_dir)
    elif args.command == "audit-trace":
        result = audit_open_research_trace(args.run_dir)
    elif args.command == "verify-tool":
        from src.runtime.research_tools import verify_tool_run
        result = verify_tool_run(args.run_dir)
    elif args.command == "tool":
        from src.runtime.research_tools import tool_plan, execute_tool
        cfg, value = load_config(args.config), read_json(args.input)
        plan = tool_plan(value, cfg)
        write_json(artifacts_path("research_plans", plan["plan_sha256"] + ".json"), plan)
        _check_execution_plan(args, plan)
        result = execute_tool(value, cfg, plan["plan_sha256"]) if args.execute else {"status": "validated_not_run", **plan}
    elif args.command == "feedback":
        result = {"feedback_path": str(record_feedback(args.run_dir, category=args.category,
            correction=args.correction, reviewer=args.reviewer, use_for_evolution=args.use_for_evolution,
            evidence_quotes=read_json(args.quotes) if args.quotes else None))}
    elif args.command == "export-dev":
        dataset = export_development(args.project, args.output)
        result = {"output": str(args.output), "n_examples": len(dataset["examples"]),
                  "dataset_sha256": dataset["dataset_sha256"], "heldout_use_allowed": False}
    elif args.command == "assess":
        result = assess_annotations(read_json(args.response), read_json(args.rubric), read_json(args.annotations))
        write_json(args.output, result)
    elif args.command == "propose-skill":
        from src.evolution.feedback_skill import build_feedback_skill_plan, execute_feedback_skill
        dataset, cfg = read_json(args.development_set), load_config(args.config)
        plan = build_feedback_skill_plan(dataset, cfg)
        write_json(artifacts_path("research_plans", plan["plan_sha256"] + ".json"), plan)
        _check_execution_plan(args, plan)
        result = execute_feedback_skill(dataset, cfg, plan["plan_sha256"]) if args.execute else plan
    elif args.command == "promotion-request":
        from src.governance.open_promotion import create_request
        from src.governance import review
        result = create_request(read_json(args.candidate), load_config(args.config),
            development_id=args.development_id, selection_id=args.selection_id,
            engineering_only=args.engineering_only)
        result["review_request_sha256"] = review.request_sha256(review.read_request(result["candidate_id"]))
    elif args.command in {"promotion-review", "promotion-publish"}:
        from src.governance import review, release
        from src.contracts.types import Candidate
        req = review.read_request(args.candidate_id)
        if (req.get("evidence_type") != "open_research_evaluation_v1"
                or args.expected_request_sha256 != review.request_sha256(req)):
            raise ValueError("promotion requires the exact open-research review request")
        if args.command == "promotion-review":
            if not args.reason.strip():
                raise ValueError("promotion review requires an explicit reason")
            result = review.decide(args.candidate_id, args.decision,
                                   reviewer=args.reviewer, reason=args.reason)
        else:
            candidate = Candidate(**read_json(artifacts_path("candidates", args.candidate_id + ".json")))
            result = {"release_id": release.publish(req["parent_release_id"], candidate,
                {"optimize": req["optimize_metrics"], "gate": req["gate_metrics"]})}
    elif args.command == "curate":
        from src.evolution.skill_curation import propose_curation
        from src.governance.release import load_state
        actions = read_json(args.actions)
        if set(actions) - {"retire", "merges", "max_catalog_chars"}:
            raise ValueError("unsupported curation action field")
        result = propose_curation(load_state().registry, **actions)
        write_json(args.output, result)
    elif args.command == "migration-eval":
        from src.governance import skill_migration
        proposal, cfg = read_json(args.proposal), load_config(args.config)
        plan = skill_migration.build_plan(proposal, cfg, negatives=args.negative_datasets,
                                         seeds=args.seeds, limit=args.limit)
        write_json(artifacts_path("research_plans", plan["plan_sha256"] + ".json"), plan)
        _check_execution_plan(args, plan)
        result = skill_migration.evaluate(proposal, cfg, plan) if args.execute else plan
    elif args.command == "migration-review":
        from src.governance import skill_migration
        result = skill_migration.review(args.run_dir, load_config(args.config), reviewer=args.reviewer,
            reason=args.reason, decision=args.decision, expected_report_sha256=args.expected_report_sha256)
    elif args.command == "migration-publish":
        from src.governance import skill_migration
        result = {"release_id": skill_migration.publish(args.run_dir, load_config(args.config))}
    elif args.command == "benchmark-freeze":
        from src.eval import open_benchmark
        result = {"dataset_dir": str(open_benchmark.freeze(read_json(args.input)))}
    elif args.command == "benchmark-run":
        from src.eval import open_benchmark
        cfg = load_config(args.config)
        plan = open_benchmark.build_plan(args.dataset, cfg, split=args.split,
                                         candidate_release=args.candidate_release,
                                         candidate=read_json(args.candidate) if args.candidate else None)
        write_json(artifacts_path("research_plans", plan["plan_sha256"] + ".json"), plan)
        _check_execution_plan(args, plan)
        result = open_benchmark.run(args.dataset, cfg, plan) if args.execute else plan
    elif args.command == "benchmark-packet":
        from src.eval import open_benchmark
        args.output.resolve().relative_to(artifacts_path("evaluation_only").resolve())
        result = open_benchmark.annotation_packet(args.evaluation_id)
        write_json(args.output, result)
        result = {"packet_path": str(args.output), "n_cases": len(result["cases"])}
    elif args.command == "benchmark-assess":
        from src.eval import open_benchmark
        result = open_benchmark.assess(args.evaluation_id, read_json(args.annotations))
    else:
        result = validate_group_partition(read_json(args.input))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
