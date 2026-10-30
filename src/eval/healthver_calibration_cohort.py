"""Source-bound HealthVer recalibration with explicit historical exposure.

This creates a repeated-calibration cohort, not independent confirmation.
Historical files are re-read on validation; their finite coverage is disclosed.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..data.healthver_task_input import SOURCE_SHA256, load_tasks
from ..util import sha256_file, sha256_obj
from .healthver_reference import QUESTION_PREFIX, SOURCE_TO_GROUNDING

ROOT = Path(__file__).resolve().parents[2]
RUBRIC = "healthver-upstream-relation-reference-v1"
POLICIES = {"exact_task_disjoint", "component_disjoint"}


def _source_hashes() -> dict[str, str]:
    from ..data import healthver_task_input

    source_hash = sha256_file(Path(__file__))
    return {
        "calibration_selection_source_sha256": source_hash,
        "task_input_evidence_source_sha256": sha256_file(Path(healthver_task_input.__file__)),
        "reference_packet_builder_source_sha256": source_hash,
        "human_reference_scorer_source_sha256": sha256_file(
            ROOT / "scripts/score_task_input_grounding_human_reference.py"),
    }


def _input(task: dict[str, Any]) -> dict[str, Any]:
    return {"question": QUESTION_PREFIX + str(task["question"]).strip(),
            "evidence": [{"doc_id": task["task_id"], "text": task["evidence"]}]}


def _text_hash(task: dict[str, Any]) -> str:
    # A renamed task/document ID must not make identical claim/evidence new.
    return sha256_obj({"claim": str(task["question"]).strip().rstrip("."),
                       "evidence_text": task["evidence"]})


def _read_history(paths: list[Path], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    if not paths:
        raise ValueError("explicit historical packets or source references are required")
    task_map = {t["task_id"]: t for t in tasks}
    artifacts, exposed_tasks, exposed_inputs = [], set(), set()
    resolved_paths = sorted({Path(p).resolve() for p in paths}, key=lambda p: p.as_posix())
    for path in resolved_paths:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"history must be an object: {path}")
        schema = value.get("schema")
        if schema == "task-input-grounding-human-reference-v2":
            if (value.get("packet_sha256") != sha256_obj(
                    {k: v for k, v in value.items() if k != "packet_sha256"})
                    or value.get("protocol_sha256") != sha256_obj(value.get("protocol_context"))
                    or value.get("protocol_context", {}).get("dataset") != "healthver"):
                raise ValueError(f"invalid historical packet identity: {path}")
            cases = value.get("cases")
            if not isinstance(cases, list) or not cases:
                raise ValueError(f"historical packet has no cases: {path}")
            for case in cases:
                docs = case.get("evidence", [])
                payload = {"question": case.get("question"), "evidence": docs}
                if (not docs or case.get("input_sha256") != sha256_obj(payload)
                        or not isinstance(case.get("question"), str)):
                    raise ValueError(f"historical packet input hash mismatch: {path}")
                matching_claim = False
                for doc in docs:
                    task = task_map.get(doc.get("doc_id"))
                    if task is None or task["evidence"] != doc.get("text"):
                        raise ValueError(f"historical document differs from source: {path}")
                    claim = case["question"].split("Claim:", 1)
                    matching_claim |= (len(claim) == 2 and claim[1].strip().rstrip(".")
                                       == str(task["question"]).strip().rstrip("."))
                    # Distractors were visible too, so their source tasks enter
                    # the exclusion set even though they were not target claims.
                    exposed_tasks.add(task["task_id"])
                if not matching_claim:
                    raise ValueError(f"historical claim differs from source: {path}")
                exposed_inputs.add(case["input_sha256"])
        elif schema == "healthver-source-grounding-reference-v1":
            if (value.get("reference_sha256") != sha256_obj(
                    {k: v for k, v in value.items() if k != "reference_sha256"})
                    or value.get("annotation_origin") != "upstream_manual"
                    or value.get("source_archive_sha256") != SOURCE_SHA256
                    or value.get("dataset") != "healthver"):
                raise ValueError(f"invalid historical source reference identity: {path}")
            rows = value.get("per_case")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"historical reference has no cases: {path}")
            for row in rows:
                task = task_map.get(row.get("task_id"))
                if task is None or sha256_obj(_input(task)) != row.get("input_sha256"):
                    raise ValueError(f"historical reference input differs from source: {path}")
                label = task["options"][task["correct_index"]]
                if row.get("resolved_verdict") != SOURCE_TO_GROUNDING[label]:
                    raise ValueError(f"historical reference label differs from source: {path}")
                exposed_tasks.add(task["task_id"])
                exposed_inputs.add(row["input_sha256"])
        else:
            raise ValueError(f"unsupported historical artifact schema: {path}")
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = path.as_posix()
        artifacts.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "schema": schema})
    return {"artifacts": artifacts,
            "coverage": "explicitly supplied artifacts only; not a complete project history claim",
            "source_archive_sha256": SOURCE_SHA256,
            "excluded_task_ids": sorted(exposed_tasks),
            "excluded_input_sha256s": sorted(exposed_inputs),
            "excluded_claim_evidence_text_sha256s": sorted({_text_hash(task_map[t]) for t in exposed_tasks}),
            "exposed_component_ids": sorted({str(task_map[t]["group_id"]) for t in exposed_tasks})}


def _select(tasks, history, per_label, seed, policy):
    if (type(per_label) is not int or not 10 <= per_label <= 100
            or type(seed) is not int or policy not in POLICIES):
        raise ValueError("per_label must be 10..100, seed an integer, and policy supported")
    excluded_ids = set(history["excluded_task_ids"])
    excluded_texts = set(history["excluded_claim_evidence_text_sha256s"])
    excluded_inputs = set(history["excluded_input_sha256s"])
    old_components = set(history["exposed_component_ids"])
    by_component = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        if (task["task_id"] in excluded_ids or _text_hash(task) in excluded_texts
                or sha256_obj(_input(task)) in excluded_inputs
                or policy == "component_disjoint" and str(task["group_id"]) in old_components):
            continue
        label = str(task["options"][task["correct_index"]])
        if label not in SOURCE_TO_GROUNDING:
            raise ValueError("HealthVer source label is outside the frozen mapping")
        by_component[str(task["group_id"])][label].append(task)
    candidates = {label: sorted(
        [c for c in by_component if label in by_component[c]],
        key=lambda c: (len(by_component[c]), sha256_obj([seed, label, c])))
        for label in SOURCE_TO_GROUNDING}
    if len(by_component) < per_label * 3:
        raise ValueError(f"insufficient eligible components after {policy} exclusions: "
                         f"{len(by_component)} available, {per_label * 3} required")
    owner = {}

    def assign(label, visited):
        for component in candidates[label]:
            if component in visited:
                continue
            visited.add(component)
            if component not in owner or assign(owner[component], visited):
                owner[component] = label
                return True
        return False

    slots = [label for label in candidates for _ in range(per_label)]
    slots.sort(key=lambda label: (len(candidates[label]), sha256_obj([seed, "slot", label])))
    for label in slots:
        if not assign(label, set()):
            raise ValueError("eligible components cannot supply a class-balanced cohort")
    selected = [min(by_component[c][label], key=lambda t: sha256_obj([seed, t["task_id"]]))
                for c, label in owner.items()]
    return sorted(selected, key=lambda t: sha256_obj([seed, "case", t["task_id"]]))


def build_packet(history_paths: list[Path], *, per_label: int = 20,
                 seed: int = 20261016, policy: str = "exact_task_disjoint") -> dict[str, Any]:
    tasks = load_tasks(["dev_optimize"])
    history = _read_history(history_paths, tasks)
    selected = _select(tasks, history, per_label, seed, policy)
    payloads = [(t, _input(t)) for t in selected]
    inputs = sorted(sha256_obj(payload) for _, payload in payloads)
    cohort_hash = sha256_obj({"task_ids": sorted(t["task_id"] for t in selected),
                              "input_sha256s": inputs, "history": history,
                              "seed": seed, "policy": policy})
    reused = len({str(t["group_id"]) for t in selected} & set(history["exposed_component_ids"]))
    exclusion = {**history, "policy": policy, "selected_prior_component_overlap_count": reused,
                 "selected_exact_task_overlap_count": 0, "selected_exact_input_overlap_count": 0}
    context = {
        "schema": "task-input-grounding-reference-protocol-v4",
        "dataset": "healthver", "split": "dev_optimize",
        "sampling": "class-balanced-one-case-per-component-with-explicit-history-exclusions",
        "per_label": per_label, "offset_per_label": 0, "sampling_seed": seed,
        "judge_evidence_character_limit_per_document": 6000,
        "rubric_version": RUBRIC, "case_input_sha256s": inputs,
        "source_hashes": _source_hashes(), "case_count": len(selected),
        "labels": ["supported", "contradicted", "insufficient"],
        "cohort_sha256": cohort_hash, "history_exclusion": exclusion,
    }
    protocol_hash = sha256_obj(context)
    cases = [{"case_id": "HEALTHVER-GND-" + hashlib.sha256(
                  f"{protocol_hash}\0{t['task_id']}".encode()).hexdigest()[:14],
              **payload, "input_sha256": sha256_obj(payload)} for t, payload in payloads]
    packet = {
        "schema": "task-input-grounding-human-reference-v2", "reviewer_slot": "reviewer_a",
        "protocol_sha256": protocol_hash, "protocol_context": context,
        "scope": {"dataset": "healthver", "split": "dev_optimize", "case_count": len(cases),
                  "use": "user-authorized noncommercial internal source-label recalibration",
                  "independent_confirmation": False,
                  "limitations": ["public source with prior project exposure",
                                  f"{reused} selected components have prior exposure in supplied history",
                                  "history coverage limited to explicitly supplied artifacts",
                                  "do not redistribute source data"]},
        "instructions": "Classify the relation between the supplied claim and paired abstract as supported, "
                        "contradicted, or insufficient. Reference labels are reconstructed separately from "
                        "the audited upstream HealthVer archive; they express text relation, not external truth.",
        "cases": cases,
    }
    packet["packet_sha256"] = sha256_obj(packet)
    return packet


def validate_history_packet(packet: dict[str, Any]) -> None:
    context = packet["protocol_context"]
    history = context.get("history_exclusion")
    if not isinstance(history, dict) or not isinstance(history.get("artifacts"), list):
        raise ValueError("HealthVer recalibration history is malformed")
    paths = []
    for artifact in history["artifacts"]:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "schema"}:
            raise ValueError("HealthVer recalibration history artifact is malformed")
        path = ROOT / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ValueError("HealthVer recalibration history artifact changed or is missing")
        paths.append(path)
    rebuilt = build_packet(paths, per_label=context["per_label"], seed=context["sampling_seed"],
                           policy=history.get("policy"))
    if packet != rebuilt:
        raise ValueError("HealthVer recalibration selection or history does not reproduce from source")
