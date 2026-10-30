"""Content-bound rejection memory for completed, same-protocol rounds (FR-207).

Only aggregate rejection reasons are retained. Unknown provider executions and
unreconciled failures are never evidence that a proposal was evaluated.
"""
from ..util import artifacts_path, read_json, sha256_file, sha256_obj, write_json
from ..config import config_fingerprint
from ..eval.run_manifest import source_tree_fingerprint


def proposal_sha256(candidate):
    return sha256_obj({"artifact_type": candidate.artifact_type,
        "artifact_id": candidate.artifact_id, "new_content": candidate.new_content,
        "extra": candidate.extra})


def context(cfg, state, datasets, seeds, limit):
    return {"config_sha256": config_fingerprint(cfg), "parent_release_id": state.release_id,
        "parent_prompt_sha256": sha256_obj(state.system_prompt),
        "parent_registry_sha256": state.registry.fingerprint(),
        "source_tree_sha256": source_tree_fingerprint()["sha256"],
        "datasets": list(datasets), "seeds": list(seeds), "limit": limit}


def _path(candidate, scope):
    return artifacts_path("candidates", "rejections", sha256_obj(scope), proposal_sha256(candidate) + ".json")


def _validate(record, candidate, scope):
    unsigned = {k: v for k, v in record.items() if k != "memory_sha256"}
    if (record.get("memory_sha256") != sha256_obj(unsigned)
            or record.get("context") != scope
            or record.get("proposal_sha256") != proposal_sha256(candidate)):
        raise ValueError("candidate rejection memory identity changed")
    proof = read_json(record["round_report_path"])
    if (sha256_file(record["round_report_path"]) != record["round_report_sha256"]
            or proof.get("rejection_memory_context") != scope
            or (proof.get("measurement_control") or {}).get("contaminated") is True
            or not any(c.get("candidate_id") == record["candidate_id"]
                       and c.get("proposal_sha256") == record["proposal_sha256"]
                       and c.get("status") in {"proposed_reject", "static_gate_rejected"}
                       for c in proof.get("candidates", []))):
        raise ValueError("candidate rejection memory lacks its completed decision receipt")
    return record


def find(candidate, scope):
    path = _path(candidate, scope)
    if not path.exists():
        return None
    return _validate(read_json(path), candidate, scope)


def remember(candidate, scope, reason, report_path):
    existing = find(candidate, scope)
    if existing is not None:
        return existing
    record = {"schema": "candidate-rejection-memory-v1", "context": scope,
        "proposal_sha256": proposal_sha256(candidate), "candidate_id": candidate.candidate_id,
        "reason": str(reason)[:2000], "round_report_path": str(report_path),
        "round_report_sha256": sha256_file(report_path)}
    record["memory_sha256"] = sha256_obj(record)
    _validate(record, candidate, scope)
    write_json(_path(candidate, scope), record)
    return find(candidate, scope)
