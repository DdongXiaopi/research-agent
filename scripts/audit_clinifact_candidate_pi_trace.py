"""Offline, fail-closed audit of a CliniFact MIPRO candidate's paired Pi traces.

This reads already recorded sessions. It never calls a model or opens a held-out set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.contracts.types import content_id, content_sha256
from src.eval.paired_candidate_audit import audit_paired_rows
from src.evolution.evidence_inference_promotion import compute_mipro_plan_sha256
from src.evolution.task_program import serialize_task_program
from src.runtime.pi_trace import read_pi_session
from src.util import sha256_file, sha256_obj, sha256_text


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), f"invalid rows: {path}")
    return rows


def audit(report_path: Path) -> dict:
    report_path = report_path.resolve(strict=True)
    report = load_json(report_path)
    require(report.get("status") == "completed" and report.get("candidate_status") == "internal_gate_passed",
            "source report has no completed internally passing candidate")
    require(report.get("protocol") == "clinifact-fixed-abstract-grouped-v1", "unexpected protocol")
    require(all(report.get(key) is False for key in
                ("official_validation_read", "official_test_read", "selection_gate_read")),
            "held-out or selection material was read")
    plan = report["plan"]
    plan_sha = compute_mipro_plan_sha256(plan)
    require(plan_sha == plan.get("plan_sha256"), "plan hash mismatch")
    cohort_path = Path(report["cohort_path"]).resolve(strict=True)
    cohort = load_json(cohort_path)
    cohort_sha = cohort.get("cohort_sha256")
    require(cohort_sha == sha256_obj({key: value for key, value in cohort.items() if key != "cohort_sha256"}),
            "cohort hash mismatch")
    require(cohort_sha == plan.get("cohort_sha256"), "plan/cohort mismatch")
    require(cohort.get("project_selection_gate_read") is False
            and cohort.get("official_test_downloaded_or_read") is False, "cohort opened a sealed set")

    program = serialize_task_program(report["candidate"]["system_prompt"], report["candidate"]["task_demos"])
    candidate_sha = content_sha256(program)
    candidate_id = content_id("cand", report["release_id"], "task_program", program)
    require(candidate_id == report["candidate_id"], "candidate content ID mismatch")
    selected_gates = [gate for gate in report["candidate_program_gates"]
                      if gate.get("candidate_id") == candidate_id and gate.get("content_sha256") == candidate_sha]
    require(len(selected_gates) == 1 and selected_gates[0].get("passed") is True
            and selected_gates[0].get("prompt_grounding_reviewed") is True,
            "selected candidate has no unique passing program gate")
    prompt_sha = sha256_text(report["candidate"]["system_prompt"])
    instruction_gates = [gate for gate in report["candidate_instruction_gates"]
                         if gate.get("content_sha256") == prompt_sha]
    require(len(instruction_gates) == 1 and instruction_gates[0].get("passed") is True
            and instruction_gates[0].get("grounding_review", {}).get("passed") is True
            and instruction_gates[0].get("task_contract_gate", {}).get("passed") is True,
            "selected prompt has no unique passing grounding/contract review")
    baseline_sha = report["incumbent_gate"]["grounding_review"]["candidate_sha256"]
    require(report["incumbent_gate"]["grounding_review"].get("passed") is True,
            "baseline grounding gate did not pass")

    baseline_path = Path(report["internal_baseline_checkpoint"]).resolve(strict=True)
    candidate_path = Path(report["internal_candidate_checkpoint"]).resolve(strict=True)
    baseline_rows, candidate_rows = load_jsonl(baseline_path), load_jsonl(candidate_path)
    expected_tasks = set(cohort["scopes"]["internal_candidate"]["task_ids"])
    require(len(expected_tasks) == 18 and len(baseline_rows) == len(candidate_rows) == 18,
            "unexpected fixed cohort size")
    require({row["task_id"] for row in baseline_rows} == expected_tasks
            and {row["task_id"] for row in candidate_rows} == expected_tasks,
            "checkpoint tasks differ from the frozen cohort")
    summary, changed = audit_paired_rows(baseline_rows, candidate_rows)
    require(summary["paired_row_count"] == summary["unique_task_count"] == 18,
            "repeated or missing task pairs")
    require(summary["improved_unique_tasks"] == report["internal_paired_analysis"]
            ["task_transition_consistency"]["improved_without_regression"], "improvement count mismatch")
    require(summary["regressed_unique_tasks"] == report["internal_paired_analysis"]
            ["task_transition_consistency"]["regressed_without_improvement"], "regression count mismatch")

    sandbox = report_path.parents[3].resolve(strict=True)
    seen_sessions: set[Path] = set()
    session_hashes: list[dict] = []
    for arm, rows, expected_sha in (("baseline", baseline_rows, baseline_sha),
                                    ("candidate", candidate_rows, prompt_sha)):
        for row in rows:
            require(row.get("status") == "ok" and row.get("input_mode") == "task_input",
                    f"{arm} task did not complete in task_input mode")
            if arm == "candidate":
                require(row.get("candidate_id") == candidate_id, "candidate row identity mismatch")
            session_path = Path(row["pi_session_path"]).resolve(strict=True)
            require(session_path.is_relative_to(sandbox) and session_path not in seen_sessions,
                    "session escaped sandbox or was reused")
            seen_sessions.add(session_path)
            session = read_pi_session(session_path)
            manifest_entry = session["task_manifest"]
            manifest = manifest_entry["data"]
            require(manifest["taskId"] == row["task_id"]
                    and manifest["releaseId"] == report["release_id"]
                    and manifest["inputMode"] == "task_input"
                    and manifest["systemPromptSha256"] == expected_sha
                    and sha256_text(manifest["systemPrompt"]) == expected_sha
                    and manifest["selectedSkill"] == row["skill_recalled"],
                    f"{arm} manifest binding mismatch: {row['task_id']}")
            messages = session["messages"]
            require(len(messages) == 2 and messages[0]["role"] == "user"
                    and messages[1]["role"] == "assistant"
                    and messages[0]["parent_id"] == manifest_entry["entry_id"]
                    and messages[1]["parent_id"] == messages[0]["entry_id"],
                    f"{arm} Pi message ancestry mismatch: {row['task_id']}")
            user_text = messages[0]["text"]
            if manifest["selectedSkill"]:
                skill = next((item for item in manifest["skills"]
                              if item["name"] == manifest["selectedSkill"]), None)
                require(skill is not None, f"{arm} selected Skill absent from manifest")
                skill_path = Path(skill["filePath"])
                skill_directory = str(skill_path.parent)
                # Pi prepends the loaded Skill body to the caller's task prompt.
                prefix = (f'<skill name="{skill["name"]}" location="{skill["filePath"]}">\n'
                          f'References are relative to {skill_directory}.\n\n')
                marker = "\n</skill>\n\n"
                require(user_text.startswith(prefix) and user_text.count(marker) == 1,
                        f"{arm} Pi Skill wrapper mismatch: {row['task_id']}")
                skill_body, task_prompt = user_text[len(prefix):].split(marker, 1)
                require(sha256_text(skill_body) == skill["bodySha256"],
                        f"{arm} Pi Skill body hash mismatch: {row['task_id']}")
            else:
                task_prompt = user_text
            require(sha256_text(task_prompt) == manifest["userPromptSha256"],
                    f"{arm} Pi user prompt hash mismatch: {row['task_id']}")
            require(not session["tool_calls"] and not session["tool_results"],
                    f"{arm} fixed-material Pi task called an unexpected tool")
            require(len(session["usages"]) == 1
                    and session["usages"][0]["entry_id"] == messages[1]["entry_id"]
                    and session["results"][-1]["status"] == "completed"
                    and session["results"][-1]["tipId"] == messages[1]["entry_id"],
                    f"{arm} Pi usage/result mismatch: {row['task_id']}")
            calls = [event["payload"] for event in row["events"] if event["event_type"] == "model.call"]
            require(len(calls) == 1, f"{arm} task lacks one model.call event")
            call = calls[0]
            trace = call["pi_trace"]
            require(Path(call["pi_session_path"]).resolve() == session_path
                    and trace["session_id"] == session["session_id"]
                    and trace["task_manifest_entry_id"] == manifest_entry["entry_id"]
                    and trace["user_entry_ids"] == [messages[0]["entry_id"]]
                    and trace["assistant_entry_id"] == messages[1]["entry_id"]
                    and trace["provider_trace_id"] == manifest["providerTraceId"]
                    and trace["protocol"] == manifest["protocol"]
                    and trace["message_count"] == 2 and trace["usage_records"] == 1
                    and trace["bridge_attempts"] == row["pi_bridge_attempts"] == 1
                    and trace["bridge_timeouts"] == row["pi_bridge_timeouts"] == 0
                    and trace["provider_http_requests"] == len(trace["provider_attempts"]) == 1
                    and trace["provider_attempts"][0]["status"] == 200,
                    f"{arm} row/Pi/provider binding mismatch: {row['task_id']}")
            session_hashes.append({"arm": arm, "task_id": row["task_id"],
                                   "session_sha256": sha256_file(session_path)})

    return {"schema": "clinifact-candidate-pi-trace-audit-v1", "status": "passed",
            "audit_source_sha256": sha256_file(__file__),
            "source_report_path": str(report_path), "source_report_sha256": sha256_file(report_path),
            "plan_sha256": plan_sha, "cohort_sha256": cohort_sha,
            "baseline_checkpoint_sha256": sha256_file(baseline_path),
            "candidate_checkpoint_sha256": sha256_file(candidate_path),
            "candidate_id": candidate_id, "candidate_program_sha256": candidate_sha,
            "candidate_prompt_sha256": prompt_sha, "baseline_prompt_sha256": baseline_sha,
            "selected_program_gate_passed": True, "selected_instruction_review_passed": True,
            "paired_audit": summary,
            "changed_task_ids": [row["task_id"] for row in changed],
            "session_count": len(session_hashes), "session_hashes": session_hashes,
            "official_validation_read": False, "official_test_read": False,
            "selection_gate_read": False, "independent_confirmation": False,
            "production_approval": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "candidate_id": result["candidate_id"],
                      "session_count": result["session_count"],
                      "improved": result["paired_audit"]["improved_unique_tasks"],
                      "regressed": result["paired_audit"]["regressed_unique_tasks"]}))


if __name__ == "__main__":
    main()
