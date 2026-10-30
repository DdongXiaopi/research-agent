"""Consent-bound personal feedback to a new Skill proposal, not a publication."""
import json
import os
from pathlib import Path
from uuid import uuid4

from ..config import config_fingerprint
from ..contracts.types import Candidate, content_id, content_sha256
from ..runtime.call_ledger import summarize_ledger
from ..governance import release
from ..util import artifacts_path, read_json, sha256_file, sha256_obj, write_json
from .crystallize import propose_skill_from_failures
from .session_feedback import verify_research_run


def _validated_examples(dataset):
    from ..runtime.research_assistant import _ID
    import re
    if (dataset.get("schema") != "open-research-development-set-v1"
            or dataset.get("heldout_use_allowed") is not False or dataset.get("scientific_gold") is not False
            or sha256_obj(dataset.get("examples")) != dataset.get("dataset_sha256")):
        raise ValueError("feedback development set is invalid")
    if not isinstance(dataset.get("project_id"), str) or not _ID.fullmatch(dataset["project_id"]):
        raise ValueError("invalid feedback project identity")
    examples = dataset["examples"]
    if not examples:
        raise ValueError("no consented development feedback")
    for example in examples:
        if (example.get("category") not in {"procedure", "evidence_fact"}
                or not isinstance(example.get("feedback_id"), str)
                or not re.fullmatch(r"[a-f0-9]{32}", example["feedback_id"])):
            raise ValueError("invalid feedback source identity")
        feedback = read_json(artifacts_path("research_feedback", dataset["project_id"],
            example["category"], example["feedback_id"] + ".json"))
        if feedback.get("feedback_id") != example["feedback_id"]:
            raise ValueError("feedback identity changed")
        if (feedback.get("use_for_evolution") is not True or feedback.get("category") == "preference"
                or sha256_obj({k: v for k, v in feedback.items() if k != "feedback_sha256"}) != feedback.get("feedback_sha256")
                or feedback.get("correction") != example.get("expected_behavior")
                or feedback.get("reviewer") != example.get("reviewer")):
            raise ValueError("feedback export no longer matches user consent or correction")
        request, _ = verify_research_run(Path(feedback["run_dir"]))
        if (sha256_obj(request) != example.get("source_request_sha256")
                or feedback.get("run_receipt_sha256") != sha256_file(Path(feedback["run_dir"]) / "receipt.json")
                or feedback.get("evidence_quotes") != example.get("evidence_quotes")
                or request["query"] != example.get("query") or request["evidence"] != example.get("evidence")
                or example.get("split") != "dev_optimize"):
            raise ValueError("feedback source request changed or is not development-only")
    return examples


def build_feedback_skill_plan(dataset, cfg, state=None):
    examples = _validated_examples(dataset)
    state = state or release.load_state()
    from ..eval.run_manifest import source_tree_fingerprint
    plan = {"schema": "feedback-skill-proposal-plan-v1", "dataset_sha256": sha256_obj(dataset),
            "source_tree_sha256": source_tree_fingerprint()["sha256"],
            "config_fingerprint": config_fingerprint(cfg), "configuration_sha256": sha256_obj(cfg),
            "release_id": state.release_id, "registry_fingerprint": state.registry.fingerprint(),
            "selected_feedback_ids": [row["feedback_id"] for row in examples[:4]],
            "logical_request_cap": 1, "physical_request_cap": 1,
            "model_calls": 0, "final_holdout_read": False, "publication_authorized": False}
    plan["plan_sha256"] = sha256_obj(plan)
    return plan


def execute_feedback_skill(dataset, cfg, expected_plan_sha256, *, state=None, provider=None):
    import copy
    state = state or release.load_state()
    plan = build_feedback_skill_plan(dataset, cfg, state)
    if expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("personal feedback proposal differs from its frozen plan")
    examples = _validated_examples(dataset)[:4]
    root = artifacts_path("research_proposals", "feedback-" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", {"status": "started"})
    ledger = root / "call-ledger.jsonl"
    bindings = {"RSI_ACTIVE_EXECUTION_PLAN_SHA256": plan["plan_sha256"], "RSI_CALL_LEDGER_PATH": str(ledger),
                "RSI_CALL_LEDGER_MAX_LOGICAL": "1", "RSI_CALL_LEDGER_MAX_PHYSICAL": "1"}
    before = {name: os.environ.get(name) for name in bindings}
    try:
        os.environ.update(bindings)
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("evolution", {}).setdefault("crystallization", {})["proposal_max_attempts"] = 1
        cases = [{"query": example["query"], "human_correction": example["expected_behavior"],
                  "source_quotes": example["evidence_quotes"], "source": "consented_user_development_feedback"}
                 for example in examples]
        skill = propose_skill_from_failures(state, [{"split": "dev_optimize"} for _ in examples], cfg,
                                           provider=provider, cases_text=json.dumps(cases, ensure_ascii=False),
                                           audit_dir=root / "proposal-audit")
        candidate = Candidate(content_id("cand", state.release_id, "feedback-new-skill", skill.to_dict(), skill.body),
            state.release_id, "new_skill", skill.name, new_content=skill.body,
            diff="new Skill proposed from consented human feedback", content_sha256=content_sha256(skill.body),
            rationale="Personal procedural hypothesis; no causal attribution or efficacy claim.",
            extra={"skill": skill.to_dict(), "feedback_ids": plan["selected_feedback_ids"]},
            generator={"backend": "llm" if provider is None else "injected_provider_fixture",
                       "proposal_plan_sha256": plan["plan_sha256"],
                       "requires_independent_open_research_evaluation": True})
        write_json(root / "candidate.json", candidate.to_dict())
        write_json(root / "status.json", {"status": "proposed_pending_independent_evaluation",
            "request_accounting": summarize_ledger(ledger), "candidate_sha256": sha256_file(root / "candidate.json")})
        return {"proposal_dir": str(root), "candidate_id": candidate.candidate_id,
                "status": "proposed_pending_independent_evaluation", "publication_authorized": False}
    except BaseException as exc:
        write_json(root / "status.json", {"status": "failed", "error_type": type(exc).__name__,
                                        "request_accounting": summarize_ledger(ledger)})
        raise
    finally:
        for name, previous in before.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
