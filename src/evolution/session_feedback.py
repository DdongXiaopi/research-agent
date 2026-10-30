"""User-authored corrections are optional development fuel, never held-out gold.

Adapt Hermes' session/golden source separation and Pi's session identity. Keep
preferences, source facts and procedural corrections in separate records.
"""
from pathlib import Path
from uuid import uuid4

from ..util import artifacts_path, now_iso, read_json, sha256_file, sha256_obj, write_json

CATEGORIES = {"preference", "evidence_fact", "procedure"}


def verify_research_run(root: Path):
    root = Path(root).resolve()
    root.relative_to(artifacts_path("research").resolve())
    receipt = read_json(root / "receipt.json")
    if (receipt.get("schema") != "open-research-receipt-v1"
            or read_json(root / "status.json").get("status") != "completed"
            or receipt.get("request_sha256") != sha256_file(root / "request.json")
            or receipt.get("response_sha256") != sha256_file(root / "response.json")
            or receipt.get("plan_sha256") != read_json(root / "plan.json").get("plan_sha256")):
        raise ValueError("research run receipt is incomplete or changed")
    plan = read_json(root / "plan.json")
    request = read_json(root / "request.json")
    if (plan.get("plan_sha256") != sha256_obj({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan.get("request_sha256") != sha256_obj(request)):
        raise ValueError("research plan content identity changed")
    if plan.get("retrieval_snapshot_sha256") or receipt.get("retrieval_sha256"):
        from ..retrieval.literature import evidence_from_snapshot
        snapshot = read_json(root / "retrieval.json")
        if (receipt.get("retrieval_sha256") != sha256_file(root / "retrieval.json")
                or snapshot.get("snapshot_sha256") != plan.get("retrieval_snapshot_sha256")
                or evidence_from_snapshot(snapshot) != request.get("evidence")):
            raise ValueError("research retrieval provenance changed")
    return read_json(root / "request.json"), receipt


def record_feedback(run_dir: Path, *, category: str, correction: str, reviewer: str,
                    use_for_evolution: bool = False, evidence_quotes: list | None = None) -> Path:
    request, receipt = verify_research_run(run_dir)
    if category not in CATEGORIES or not isinstance(use_for_evolution, bool):
        raise ValueError("invalid feedback category or consent")
    if (not isinstance(correction, str) or not correction.strip() or len(correction) > 8000
            or not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 120):
        raise ValueError("human feedback requires a bounded correction and reviewer identity")
    if category == "preference" and use_for_evolution:
        raise ValueError("preferences are personal memory, not scientific evolution labels")
    quotes = evidence_quotes or []
    sources = {source["source_id"]: source for source in request["evidence"]}
    for quote in quotes:
        if (not isinstance(quote, dict) or set(quote) != {"source_id", "quote"}
                or quote["source_id"] not in sources or not isinstance(quote["quote"], str)
                or not quote["quote"].strip() or quote["quote"] not in sources[quote["source_id"]]["text"]):
            raise ValueError("feedback evidence quote does not match the input source")
    if category == "evidence_fact" and not quotes:
        raise ValueError("research facts require source-bound evidence")
    body = {"schema": "research-session-feedback-v1", "feedback_id": uuid4().hex,
        "project_id": request["project_id"], "session_id": request["session_id"], "task_id": request["task_id"],
        "run_dir": str(Path(run_dir).resolve()), "run_receipt_sha256": sha256_file(Path(run_dir) / "receipt.json"),
        "category": category, "correction": correction.strip(), "reviewer": reviewer.strip(),
        "source": "user_authored", "evidence_quotes": quotes, "use_for_evolution": use_for_evolution,
        "data_classification": "private_development" if use_for_evolution else "private_session_only",
        "eligible_scope": "dev_only" if use_for_evolution else None, "created_at": now_iso()}
    body["feedback_sha256"] = sha256_obj(body)
    out = artifacts_path("research_feedback", request["project_id"], category, body["feedback_id"] + ".json")
    write_json(out, body)
    return out


def export_development(project_id: str, output: Path) -> dict:
    from ..runtime.research_assistant import _ID
    if not _ID.fullmatch(project_id):
        raise ValueError("invalid project identity")
    examples = []
    for file in sorted(artifacts_path("research_feedback", project_id).glob("*/*.json")):
        row = read_json(file)
        if sha256_obj({key: value for key, value in row.items() if key != "feedback_sha256"}) != row.get("feedback_sha256"):
            raise ValueError("feedback identity changed")
        if row.get("use_for_evolution") is not True or row.get("category") == "preference":
            continue
        root = Path(row["run_dir"])
        request, _ = verify_research_run(root)
        if sha256_file(root / "receipt.json") != row["run_receipt_sha256"]:
            raise ValueError("feedback source run changed")
        examples.append({"task_id": row["feedback_id"], "source_task_id": row["task_id"], "project_id": project_id,
            "query": request["query"], "evidence": request["evidence"],
            "expected_behavior": row["correction"], "evidence_quotes": row["evidence_quotes"],
            "source": "user_authored", "reviewer": row["reviewer"], "feedback_id": row["feedback_id"],
            "category": row["category"],
            "group_id": sha256_obj(sorted(sha256_obj(source["text"]) for source in request["evidence"])),
            "split": "dev_optimize", "requires_independent_evaluation": True,
            "source_request_sha256": sha256_obj(request)})
    result = {"schema": "open-research-development-set-v1", "project_id": project_id,
        "examples": examples, "heldout_use_allowed": False,
        "scientific_gold": False, "dataset_sha256": sha256_obj(examples)}
    write_json(output, result)
    return result
