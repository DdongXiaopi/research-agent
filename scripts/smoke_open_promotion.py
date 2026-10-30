"""Isolated synthetic feedback -> independent benchmark -> review -> publish -> rollback."""
from __future__ import annotations
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_fixture(*, live=False, model_id="Qwen/Qwen3-8B", authored_candidate=False, kind="new_skill"):
    from src.config import load_config
    from src.contracts.types import Candidate
    from src.evolution.feedback_skill import build_feedback_skill_plan, execute_feedback_skill
    from src.evolution import feedback_candidate
    from src.evolution.session_feedback import record_feedback, export_development
    from src.eval import open_benchmark as benchmark
    from src.governance import release, review
    from src.governance.open_promotion import create_request
    from src.runtime import research_assistant as research
    from src.runtime.call_ledger import summarize_ledger
    from src.util import artifacts_path, sha256_obj, read_json, write_json
    import yaml

    assets = artifacts_path("engineering-fixture")
    assets.mkdir(parents=True, exist_ok=False)
    if live:
        from src.runtime.free_model import verify_current
        from src.runtime.provider import load_project_dotenv
        load_project_dotenv()
        if not os.environ.get("SILICONFLOW_API_KEY"):
            raise ValueError("SILICONFLOW_API_KEY is unavailable; no live request dispatched")
        verify_current(model_id, assets / "zero-price-verification")
    prompt = assets / "prompt.txt"
    prompt.write_text('Synthetic governance acceptance fixture. Return status abstain, claims [], limitations [] '
        'unless the Research procedure section contains the literal token CONTROL_ACCEPTANCE. '
        'When that token is present, answer the research question from the supplied sources with an exact quote.', encoding="utf-8")
    registry = assets / "registry.yaml"
    existing_skills = [] if kind in {"new_skill", "system_prompt"} else [{
        "name": "protocol-troubleshooting", "group": "G-main", "license": "SELF-AUTHORED",
        "description": "Diagnose laboratory protocol control problems.",
        "triggers": ["assay", "control"], "path": "skills/main/protocol-troubleshooting.md"}]
    registry.write_text(yaml.safe_dump({"release_id": "public-open-base", "skills": existing_skills}), encoding="utf-8")
    release.init_baseline(prompt_path=prompt, registry_path=registry, release_id="public-open-base")
    cfg = load_config(str(ROOT / "configs/experiment.public-task-input.yaml"))
    model = {"provider": "http", "model_id": model_id if live else "fixture-model",
             "base_url": "https://api.siliconflow.cn/v1", "api_key_env": "SILICONFLOW_API_KEY",
             "temperature": 0, "max_tokens": 1200, "max_retries": 1, "timeout": 120,
             "enable_thinking": False, "cache": False}
    cfg["models"]["task"] = copy.deepcopy(model)
    cfg["models"]["reflection"] = copy.deepcopy(model)
    cfg["runtime"].update(provider="http", http_no_cache_required=True,
                          open_research_structured_parse_failures=True)
    cfg["skills"]["recall_min_score"] = 1
    cfg["evolution"]["behavior_channels"]["enforce_in_evolution"] = False
    feedback_request = research.request_from_dict({"task_id": "feedback-case", "task_type": "open_research",
        "query": "Which synthetic assay control does code K7Q4 denote?", "project_id": "public-evolution",
        "session_id": "feedback", "evidence": [{"source_id": "training-note",
        "text": "Synthetic training codebook: missing-control code K7Q4 denotes a control group.",
        "url": "synthetic://feedback-only"}]})
    offline = copy.deepcopy(cfg)
    offline["runtime"]["provider"] = "fake"
    result = research.execute(feedback_request, offline, research.build_plan(feedback_request, offline)["plan_sha256"])
    record_feedback(Path(result["run_dir"]), category="procedure", reviewer="engineering-fixture-owner",
        correction="Create an assay control decoding procedure with the literal marker CONTROL_ACCEPTANCE in its body; code K7Q4 denotes a control group. Read the supplied code, decode it, and cite the exact code source.",
        use_for_evolution=True)
    development = export_development("public-evolution", assets / "development.json")
    skill = {"name": "synthetic-control-check", "description": "Decode missing controls in synthetic assay evidence",
             "triggers": ["assay", "control", "synthetic"],
             "body": "## Procedure\n1. CONTROL_ACCEPTANCE: Read the missing-control code in the supplied assay evidence.\n2. Apply the learned codebook: K7Q4 denotes a control group.\n3. State that a control group was missing and cite the exact source quote that contains K7Q4."}
    if kind == "new_skill":
        proposal_provider = (None if live and not authored_candidate else
                             SimpleNamespace(chat=lambda *args, **kwargs: json.dumps(skill)))
        proposal_plan = build_feedback_skill_plan(development, cfg)
        proposal = execute_feedback_skill(development, cfg, proposal_plan["plan_sha256"], provider=proposal_provider)
    else:
        if live:
            raise ValueError("multi-artifact acceptance uses synthetic engineering fixtures")
        from src.runtime.skills import load_body
        state = release.load_state()
        target = "system_prompt" if kind == "system_prompt" else "protocol-troubleshooting"
        old = (state.system_prompt if kind == "system_prompt" else
               state.registry.get(target).description if kind == "skill_description" else load_body(state.registry.get(target)))
        content = (" ".join(old.split()) + " Check the study population." if kind == "skill_description" else
                   old.rstrip() + ("\n4. Check the study population." if kind == "skill_body" else "\nCheck the study population."))
        provider = SimpleNamespace(chat_measured=lambda *args, **kwargs:
            SimpleNamespace(content=json.dumps({"content": content}), cost={"usd": 0}))
        proposal_plan = feedback_candidate.build_plan(development, cfg, kind=kind, artifact_id=target, state=state)
        proposal = feedback_candidate.execute(development, cfg, proposal_plan["plan_sha256"],
                                              kind=kind, artifact_id=target, state=state, provider=provider)
        if proposal["status"] != "proposed_pending_independent_evaluation":
            raise ValueError("fixture candidate did not pass static gate")
    candidate_path = Path(proposal["proposal_dir"]) / "candidate.json"
    candidate = read_json(candidate_path)
    partitions = {}
    for split in sorted(benchmark.SPLITS):
        cases = []
        for index in range(3):
            source_text = (f"Synthetic independent {split} case {index}: "
                           "The assay's missing-control code is K7Q4.")
            request = research.request_from_dict({"task_id": f"case-{split}-{index}", "task_type": "open_research",
                "query": "Which synthetic assay control was missing? Decode the source code using the active procedure.",
                "project_id": "public-evolution",
                "session_id": split, "evidence": [{"source_id": "paper-one", "text": source_text,
                "url": f"synthetic://independent/{split}/{index}"}],
                "budget": {"max_completion_tokens": 250, "max_physical_requests": 1}})
            rubric = {"schema": "open-research-task-rubric-v1", "author": "synthetic-reference-author",
                "authorship": "engineering_fixture", "request_sha256": sha256_obj(request.to_dict()),
                "items": [{"item_id": "missing-control", "criterion":
                           "Decode K7Q4 and state that the assay lacked a control group.",
                           "evidence_quotes": [{"source_id": "paper-one",
                                                "quote": "The assay's missing-control code is K7Q4."}]}]}
            cases.append({"request": request.to_dict(), "rubric": rubric, "group_id": f"article-{split}-{index}"})
        partitions[split] = cases
    dataset = benchmark.freeze({"schema": "open-research-dataset-v1", "partitions": partitions,
        "provenance": {"source": "synthetic engineering acceptance fixture", "revision": "public-v1",
                       "license": "SELF-AUTHORED", "reference_origin": "engineering_fixture"}})

    def factory(arm, request):
        content = ({"status": "abstain", "claims": [], "limitations": []} if arm == "baseline" else {
            "status": "ok", "claims": [{"text": "The assay lacked a control group (code K7Q4).",
            "citations": [{"source_id": "paper-one", "quote":
                           "The assay's missing-control code is K7Q4."}]}],
            "limitations": []})
        return SimpleNamespace(chat_measured=lambda *args, **kwargs: SimpleNamespace(content=json.dumps(content), cost={"usd": 0}))

    evaluations = {}
    for split in ("dev_optimize", "selection_gate"):
        plan = benchmark.build_plan(dataset, cfg, split=split, candidate=candidate)
        result = benchmark.run(dataset, cfg, plan, provider_factory=None if live else factory)
        packet = benchmark.annotation_packet(result["evaluation_id"])
        annotations = {"evaluation_id": result["evaluation_id"], "cases": []}
        for case in packet["cases"]:
            response = case["response"]
            answer_text = response["answer"]
            passed = "control group" in answer_text.lower()
            values = []
            for reviewer in ("synthetic-rule-a", "synthetic-rule-b"):
                for annotation_kind, identity in [("rubric", "missing-control")] + [
                        ("citation", str(i)) for i in range(len(response["citations"]))]:
                    values.append({"reviewer": reviewer, "source": "engineering_fixture", "kind": annotation_kind,
                        "item_id": identity, "supported": passed if annotation_kind == "rubric" else True,
                        "rationale": "Deterministic synthetic control fixture; no human or calibrated semantic judgment.",
                        "response_sha256": case["response_sha256"], "rubric_sha256": case["rubric_sha256"]})
            annotations["cases"].append({"case_id": case["case_id"], "annotations": values})
        benchmark.assess(result["evaluation_id"], annotations)
        evaluations[split] = result["evaluation_id"]
    request = create_request(candidate, cfg, development_id=evaluations["dev_optimize"],
                             selection_id=evaluations["selection_gate"], engineering_only=True)
    if request["verdict"] != "proposed_accept":
        write_json(assets / "failed-gates.json", request)
        raise ValueError("synthetic live candidate did not meet the unchanged engineering acceptance gates")
    req = review.read_request(candidate["candidate_id"])
    try:
        release.publish(candidate["parent_release_id"], Candidate(**candidate),
                        {"optimize": req["optimize_metrics"], "gate": req["gate_metrics"]})
    except release.ReleaseError:
        pass
    else:
        raise ValueError("publication occurred without an approved review")
    review.decide(candidate["candidate_id"], "approve", reviewer="auto-sim",
                  reason="Isolated synthetic engineering acceptance; no production or scientific claim.")
    req = review.read_request(candidate["candidate_id"])
    published = release.publish(candidate["parent_release_id"], Candidate(**candidate),
                                {"optimize": req["optimize_metrics"], "gate": req["gate_metrics"]})
    if release.current_release_id() != published:
        raise ValueError("published release did not activate")
    release.rollback(candidate["parent_release_id"])
    if not review.request_effect_gate_ok(req):
        raise ValueError("unchanged reviewed evidence did not reverify after rollback")
    changed = copy.deepcopy(req)
    changed["gate_metrics"]["candidate"]["rubric"] = 0
    if review.request_effect_gate_ok(changed):
        raise ValueError("modified promotion metrics were accepted")
    private = benchmark._evaluation(evaluations["selection_gate"])
    first_run = Path(read_json(private / "cases.json")[0]["run_dir"])
    raw_response = first_run / "response.json"
    original = raw_response.read_bytes()
    try:
        raw_response.write_bytes(original + b"\n")
        if review.request_effect_gate_ok(req):
            raise ValueError("modified execution evidence was accepted")
    finally:
        raw_response.write_bytes(original)
    try:
        create_request(candidate, cfg, development_id=evaluations["dev_optimize"],
                       selection_id=evaluations["selection_gate"], engineering_only=False)
    except ValueError:
        pass
    else:
        raise ValueError("engineering fixture was accepted for production publication")
    accounting = [summarize_ledger(path) for path in artifacts_path().rglob("call-ledger.jsonl")]
    summary = {"status": "ok", "engineering_only": True, "live_http": live,
        "candidate_origin": "authored_engineering_fixture" if authored_candidate or not live else "live_model_proposal",
        "feedback_candidate": candidate["candidate_id"], "candidate_kind": candidate["artifact_type"], "published_release": published,
        "rollback_release": release.current_release_id(), "evaluations": evaluations,
        "physical_http_attempts": sum(x["physical_http_attempts"] for x in accounting),
        "unknown_requests": sum(x["logical_incomplete"] + x["physical_incomplete"] for x in accounting),
        "scientific_gain_claim": False, "tamper_guards_verified": True, "model_id": model_id if live else "fixture-model"}
    write_json(assets / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--authored-candidate", action="store_true",
                        help="Use a labeled authored feedback proposal while task answers use the live model")
    parser.add_argument("--kind", choices=["new_skill", "system_prompt", "skill_description", "skill_body"], default="new_skill")
    parser.add_argument("--model-id", choices=["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414"], default="Qwen/Qwen3-8B")
    parser.add_argument("--run-home", type=Path)
    args = parser.parse_args()
    if args.worker:
        if os.environ.get("RSI_OPEN_ACCEPTANCE_WORKER") != "1":
            parser.error("worker must be started by the isolated launcher")
        print(json.dumps(run_fixture(live=args.live, model_id=args.model_id,
                                     authored_candidate=args.authored_candidate, kind=args.kind), sort_keys=True))
        return 0
    if args.live and args.run_home is None:
        parser.error("live acceptance requires a new persistent --run-home")
    with tempfile.TemporaryDirectory(prefix="rsi-open-acceptance-") as tmp:
        home = (args.run_home or Path(tmp) / "runtime").resolve()
        if home.exists():
            parser.error("acceptance run-home must not exist")
        env = {**os.environ, "RSI_HOME": str(home), "RSI_OPEN_ACCEPTANCE_WORKER": "1",
               "PYTHONIOENCODING": "utf-8"}
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model-id", args.model_id, "--kind", args.kind]
        if args.live:
            command.append("--live")
        if args.authored_candidate:
            command.append("--authored-candidate")
        result = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError("open acceptance failed: " + result.stderr[-1800:])
        print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
