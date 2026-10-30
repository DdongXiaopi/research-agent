"""Shared AI/human annotation provenance, separate from publication approval."""
import re

ORIGINS = {"independent_ai", "independent_human", "engineering_fixture"}


def validate_actor(identity, origin, provenance=None):
    if not isinstance(identity, str) or not identity.strip() or origin not in ORIGINS:
        raise ValueError("annotation requires a declared AI/human origin and actor identity")
    if origin == "engineering_fixture":
        import os
        from ..util import artifacts_root, repo_root
        if not os.environ.get("RSI_HOME") or artifacts_root() in {repo_root(), repo_root() / "artifacts"}:
            raise ValueError("engineering annotation fixtures require an isolated RSI_HOME")
        return {"origin": origin, "identity": identity, "review_run_id": None}
    if origin == "independent_human":
        return {"origin": origin, "identity": identity, "review_run_id": None}
    if (not isinstance(provenance, dict)
            or {"identity", "origin"} & set(provenance)
            or not isinstance(provenance.get("review_run_id"), str) or not provenance["review_run_id"].strip()
            or provenance.get("blind") is not True
            or provenance.get("other_review_outputs_consulted") is not False
            or not isinstance(provenance.get("model_id"), (str, type(None)))
            or not isinstance(provenance.get("rubric_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", provenance["rubric_sha256"])):
        raise ValueError("AI annotations require separate-run, blind, model and frozen rubric provenance")
    return {"origin": origin, "identity": identity, **provenance}


def validate_pair(actors, author, author_provenance=None):
    if len(actors) != 2 or len({a["identity"].strip().casefold() for a in actors}) != 2:
        raise ValueError("two distinct reviewers are required")
    if any(a["identity"].strip().casefold() == author.strip().casefold() for a in actors):
        raise ValueError("reference author cannot review their own answer")
    if any(a["origin"] == "engineering_fixture" for a in actors):
        if not all(a["origin"] == "engineering_fixture" for a in actors):
            raise ValueError("engineering fixtures cannot impersonate independent human/AI evidence")
        return "engineering_fixture"
    runs = [a["review_run_id"] for a in actors if a["origin"] == "independent_ai"]
    if len(runs) != len(set(runs)):
        raise ValueError("AI reviewers must use separate run contexts")
    if author_provenance and author_provenance.get("review_run_id") in runs:
        raise ValueError("AI rubric author run cannot also be a reviewer run")
    return "ai" if all(a["origin"] == "independent_ai" for a in actors) else "mixed" if runs else "human"
