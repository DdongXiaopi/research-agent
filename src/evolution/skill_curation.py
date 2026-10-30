"""Reviewable lifecycle proposals; no fifth automatic optimizer dimension.

Archived Skills retain their bodies and dependencies for rollback. A merge
requires a named survivor and a supplied merged procedure; duplicate detection
does not silently delete capabilities. Applying a proposal requires a separately
evaluated manual registry migration, never a description/body candidate bypass.
"""
import copy
from difflib import SequenceMatcher

from ..runtime.skills import load_body, parse_procedure, render_catalog
from ..util import sha256_obj


def propose_curation(registry, *, retire=None, merges=None, max_catalog_chars=16000):
    if not isinstance(retire or [], list) or not isinstance(merges or [], list):
        raise ValueError("curation actions must be lists")
    retire, merges = retire or [], merges or []
    trial = copy.deepcopy(registry)
    touched, actions, redirects = set(), [], []
    for name in retire:
        skill = trial.get(name)
        if skill is None or skill.frozen or name in touched or skill.lifecycle != "active":
            raise ValueError("retirement must target one active mutable Skill")
        skill.lifecycle = "retired"
        touched.add(name)
        actions.append({"action": "retire", "skill": name})
    for merge in merges:
        if not isinstance(merge, dict) or set(merge) != {"sources", "survivor", "merged_body"}:
            raise ValueError("merge requires sources, survivor and a concrete merged procedure")
        sources, survivor, body = merge["sources"], merge["survivor"], merge["merged_body"]
        if (not isinstance(sources, list) or len(set(sources)) != len(sources) or len(sources) < 2
                or survivor not in sources or not isinstance(body, str) or not 3 <= len(parse_procedure(body)) <= 12
                or len(body) > 6000):
            raise ValueError("invalid merge procedure or participant identities")
        skills = [trial.get(name) for name in sources]
        if any(skill is None or skill.frozen or skill.lifecycle != "active" or skill.name in touched for skill in skills):
            raise ValueError("merge participants must be active mutable unmodified Skills")
        if len({skill.group for skill in skills}) != 1:
            raise ValueError("cross-domain/group merge requires a separately reviewed migration")
        if len({sha256_obj({"files": skill.bundle_files, "root": skill.bundle_root,
                           "license": skill.license, "dependencies": skill.dependencies,
                           "permissions": skill.permissions}) for skill in skills}) != 1:
            raise ValueError("merge dependency/license conflicts must be resolved explicitly before migration")
        trial.get(survivor).body = body
        for skill in skills:
            if skill.name != survivor:
                skill.lifecycle, skill.replaced_by = "merged", survivor
        for archived in trial.skills:
            if archived.lifecycle == "merged" and archived.replaced_by in set(sources) - {survivor}:
                if archived.frozen:
                    raise ValueError("merge cannot change a frozen predecessor replacement")
                redirects.append({"skill": archived.name, "from": archived.replaced_by, "to": survivor})
                archived.replaced_by = survivor
        touched.update(sources)
        actions.append({"action": "merge", **merge})
    if not actions:
        raise ValueError("curation proposal has no changes")
    for skill in trial.skills:
        if skill.replaced_by and (trial.get(skill.replaced_by) is None or
                                  trial.get(skill.replaced_by).lifecycle != "active"):
            raise ValueError("retirement would leave an archived replacement without an active survivor")
    catalog = render_catalog(trial, max_catalog_chars)
    duplicates = [{"left": left.name, "right": right.name,
        "description_similarity": SequenceMatcher(None, left.description, right.description).ratio()}
        for i, left in enumerate(registry.skills) for right in registry.skills[i + 1:]
        if SequenceMatcher(None, left.description, right.description).ratio() >= .8]
    proposal = {"schema": "manual-skill-curation-proposal-v1", "status": "requires_independent_evaluation",
        "parent_registry_fingerprint": registry.fingerprint(), "actions": actions,
        "replacement_redirects": redirects,
        "trial_registry": trial.to_dict(), "trial_bodies": {skill.name: load_body(skill) for skill in trial.skills},
        "catalog_chars": len(catalog), "potential_duplicates": duplicates,
        "max_catalog_chars": max_catalog_chars,
        "required_gates": ["same visible task scope and all execution seeds",
            "classwise accuracy/coverage/precision non-regression", "independent negative-domain tasks",
            "human approval of exact registry migration", "immutable snapshot and rollback"],
        "publication_authorized": False}
    proposal["proposal_sha256"] = sha256_obj(proposal)
    return proposal


def verified_trial(registry, proposal):
    """Re-derive the complete proposal; serialized statuses cannot bypass rules."""
    retire, merges = [], []
    for action in proposal.get("actions") or []:
        if action.get("action") == "retire" and set(action) == {"action", "skill"}:
            retire.append(action["skill"])
        elif action.get("action") == "merge" and set(action) == {"action", "sources", "survivor", "merged_body"}:
            merges.append({key: action[key] for key in ("sources", "survivor", "merged_body")})
        else:
            raise ValueError("unsupported migration action")
    canonical = propose_curation(registry, retire=retire, merges=merges,
                                  max_catalog_chars=proposal.get("max_catalog_chars", 16000))
    if canonical != proposal:
        raise ValueError("curation proposal differs from re-derived parent/actions")
    trial = copy.deepcopy(registry)
    rows = {row["name"]: row for row in proposal["trial_registry"]["skills"]}
    for skill in trial.skills:
        skill.lifecycle = rows[skill.name].get("lifecycle", "active")
        skill.replaced_by = rows[skill.name].get("replaced_by", "")
        skill.body = proposal["trial_bodies"][skill.name]
    return trial
