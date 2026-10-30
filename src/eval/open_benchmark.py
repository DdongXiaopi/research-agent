"""Frozen open-research tasks, private evaluator records and declared annotations.

Adapts ScholarQABench's independent rubric/citation scoring. Rubrics are frozen
against TASKS before answers, not authored to fit model responses. This module
never creates human labels; final exposure is reserved before sealed parsing.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from uuid import uuid4

from ..config import config_fingerprint
from ..runtime import research_assistant as research
from ..governance import release
from ..governance.policy import static_gate
from ..contracts.types import Candidate, content_sha256
from ..evolution.apply import state_with
from ..util import artifacts_path, read_json, write_json, sha256_file, sha256_obj, sha256_text, now_iso
from .open_research import (assess_annotations, citation_statistics, paired_group_bootstrap,
                            validate_group_partition)
from .run_manifest import source_tree_fingerprint
from .annotation_policy import validate_actor, ORIGINS

SPLITS = {"dev_optimize", "selection_gate", "final_holdout"}


def validate_rubric(request, rubric):
    if (not {"schema", "request_sha256", "author", "authorship", "items"} <= set(rubric)
            or set(rubric) - {"schema", "request_sha256", "author", "authorship", "items", "author_provenance"} or
            rubric["schema"] != "open-research-task-rubric-v1" or
            rubric["request_sha256"] != sha256_obj(request.to_dict()) or
            rubric["authorship"] not in ORIGINS or not isinstance(rubric["author"], str) or not rubric["author"].strip() or
            not isinstance(rubric["items"], list) or not rubric["items"]):
        raise ValueError("benchmark requires an independent task-bound rubric frozen before answering")
    validate_actor(rubric["author"], rubric["authorship"], rubric.get("author_provenance"))
    seen, sources = set(), {s["source_id"]: s["text"] for s in request.evidence}
    for row in rubric["items"]:
        if (set(row) != {"item_id", "criterion", "evidence_quotes"} or
                not isinstance(row["item_id"], str) or not row["item_id"].strip() or row["item_id"] in seen or
                not isinstance(row["criterion"], str) or not row["criterion"].strip() or
                not isinstance(row["evidence_quotes"], list) or not row["evidence_quotes"]):
            raise ValueError("rubric items need unique identities, criteria and original evidence")
        seen.add(row["item_id"])
        for quote in row["evidence_quotes"]:
            if (set(quote) != {"source_id", "quote"} or quote["source_id"] not in sources or
                    not isinstance(quote["quote"], str) or not quote["quote"].strip() or
                    quote["quote"] not in sources[quote["source_id"]]):
                raise ValueError("rubric evidence is not an original source quotation")


def freeze(value):
    if set(value) != {"schema", "provenance", "partitions"} or value["schema"] != "open-research-dataset-v1":
        raise ValueError("unsupported open dataset schema")
    provenance = value["provenance"]
    if (set(provenance) - {"source", "revision", "license", "reference_origin", "license_policy"}
            or not {"source", "revision", "license", "reference_origin"} <= set(provenance) or
            any(not isinstance(item, str) or not item.strip() for item in provenance.values()) or
            provenance["license"].upper() in {"UNKNOWN", "UNVERIFIED"}):
        raise ValueError("open dataset requires explicit source/revision/license/reference provenance")
    partitions, owners, identities, groups = value["partitions"], {}, {}, {}
    for split, cases in partitions.items():
        identities[split] = []
        for case in cases:
            if (set(case) - {"request", "group_id", "rubric", "evaluation_role"}
                    or not {"request", "group_id", "rubric"} <= set(case)):
                raise ValueError("open case requires request, group_id and task rubric")
            if "evaluation_role" in case and case["evaluation_role"] not in {"primary", "diagnostic_only"}:
                raise ValueError("open case evaluation role is invalid")
            request = research.request_from_dict(case["request"])
            validate_rubric(request, case["rubric"])
            identities[split].append({"task_id": request.task_id, "group_id": case["group_id"]})
            # Same text or URL cannot be relabeled as a different topic to leak
            # an article across splits. Evidence within one group is allowed.
            for source in request.evidence:
                keys = ["text:" + sha256_text(source["text"])]
                if source.get("url"):
                    keys.append("url:" + source["url"].strip().casefold())
                for key in keys:
                    if key in owners and owners[key] != split:
                        raise ValueError("same article/source crosses open benchmark splits")
                    owners[key] = split
            groups[request.task_id] = case["group_id"]
    validate_group_partition(identities)
    file_digests = {split: sha256_text(json.dumps(partitions[split], ensure_ascii=False, indent=2,
                                                  sort_keys=True) + "\n") for split in SPLITS}
    metadata = {split: {"n": len(partitions[split]),
        "cases": [{"task_id_sha256": sha256_text(c["request"]["task_id"]),
                   "group_sha256": sha256_text(c["group_id"]),
                   "request_sha256": sha256_obj(research.request_from_dict(c["request"]).to_dict()),
                   "rubric_sha256": sha256_obj(c["rubric"]), "budget": c["request"].get("budget", {})}
                  for c in partitions[split]]} for split in SPLITS}
    identity = sha256_obj({"schema": value["schema"], "provenance": provenance,
                          "files": file_digests, "splits": metadata})
    root = artifacts_path("open_benchmarks", identity)
    if root.exists():
        verify_dataset(root)
        return root
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in sorted(SPLITS):
        location = root / ("development" if split == "dev_optimize" else "evaluator_only") / (split + ".json")
        write_json(location, partitions[split])
        files[split] = {"path": location.relative_to(root).as_posix(), "sha256": sha256_file(location)}
    manifest = {"schema": "open-research-freeze-v1", "dataset_sha256": identity,
        "provenance": provenance, "files": files, "splits": metadata, "frozen_at": now_iso()}
    manifest["manifest_sha256"] = sha256_obj(manifest)
    write_json(root / "manifest.json", manifest)
    return root


def verify_dataset(root):
    root = Path(root).resolve()
    root.relative_to(artifacts_path("open_benchmarks").resolve())
    manifest = read_json(root / "manifest.json")
    if manifest.get("manifest_sha256") != sha256_obj({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("open benchmark manifest changed")
    if set(manifest["files"]) != SPLITS or root.name != manifest["dataset_sha256"]:
        raise ValueError("invalid frozen open dataset identity")
    if manifest["dataset_sha256"] != sha256_obj({"schema": "open-research-dataset-v1",
            "provenance": manifest["provenance"],
            "files": {split: entry["sha256"] for split, entry in manifest["files"].items()},
            "splits": manifest["splits"]}):
        raise ValueError("open benchmark content identity changed")
    for split, entry in manifest["files"].items():
        expected = ("development" if split == "dev_optimize" else "evaluator_only") + "/" + split + ".json"
        if entry["path"] != expected or sha256_file(root / expected) != entry["sha256"]:
            raise ValueError("frozen open dataset file changed")
    return manifest


def _finish_final(root, plan, evaluation_id, status):
    root = Path(root)
    lock = root / "final-reservation.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        ledger = read_json(root / "final-access.json")
        rows = [row for row in ledger["reservations"] if row["plan_sha256"] == plan["plan_sha256"]]
        if len(rows) != 1 or rows[0]["status"] != "reserved":
            raise ValueError("final completion does not match its one-use reservation")
        rows[0].update(status=status, evaluation_id=evaluation_id, finished_at=now_iso())
        write_json(root / "final-access.json", ledger)
    finally:
        lock.unlink(missing_ok=True)


def _candidate_state(parent, cfg, candidate):
    candidate = Candidate(**candidate)
    if (candidate.parent_release_id != parent.release_id or candidate.artifact_type not in {
            "system_prompt", "skill_description", "skill_body", "new_skill"} or
            candidate.content_sha256 != content_sha256(candidate.new_content)):
        raise ValueError("open candidate does not match its current parent/content/four-artifact contract")
    gate = static_gate(candidate, cfg, parent)
    if gate["passed"] is not True:
        raise ValueError("open candidate failed static gate")
    return state_with(parent, candidate)


def build_plan(root, cfg, *, split="dev_optimize", candidate_release=None, candidate=None):
    if split not in SPLITS:
        raise ValueError("invalid benchmark split")
    manifest = verify_dataset(root)
    parent = release.load_state()
    states = {"baseline": parent}
    if candidate and candidate_release:
        raise ValueError("select one candidate snapshot or one unpublished candidate")
    if candidate:
        states["candidate"] = _candidate_state(parent, cfg, candidate)
    if candidate_release:
        if candidate_release == parent.release_id:
            raise ValueError("open comparison arms must differ")
        states["candidate"] = release.load_state(candidate_release)
    caps = []
    for item in manifest["splits"][split]["cases"]:
        limits = research._limits(type("Limits", (), {"budget": item["budget"]})())
        caps.append(limits["max_physical_requests"])
    provider = str((cfg.get("runtime") or {}).get("provider", "fake"))
    if provider not in {"http", "fake"}:
        raise ValueError("open benchmark uses the measured HTTP/free-form adapter")
    plan = {"schema": "open-benchmark-execution-plan-v1", "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": manifest["manifest_sha256"], "split": split, "n_tasks": manifest["splits"][split]["n"],
        "config_sha256": sha256_obj(cfg), "protocol_sha256": config_fingerprint(cfg),
        "source_tree_sha256": source_tree_fingerprint()["sha256"], "provider": provider,
        "unpublished_candidate": copy.deepcopy(candidate),
        "arms": {name: {"release_id": state.release_id, "prompt_sha256": sha256_obj(state.system_prompt),
                         "registry_sha256": state.registry.fingerprint()} for name, state in states.items()},
        "logical_request_cap": len(caps) * len(states) if provider == "http" else 0,
        "physical_request_cap": sum(caps) * len(states) if provider == "http" else 0,
        "protected_rows_read_during_preflight": False, "preflight_model_calls": 0,
        "score_policy": "frozen_task_rubric_and_two_context_separated_ai_or_human_reviewers", "final_batch_limit": 2}
    return {**plan, "plan_sha256": sha256_obj(plan)}


def _reserve_final(root, plan):
    root = Path(root)
    lock = root / "final-reservation.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        file = root / "final-access.json"
        ledger = read_json(file) if file.exists() else {"reservations": []}
        if len(ledger["reservations"]) >= plan["final_batch_limit"] or any(
                row["plan_sha256"] == plan["plan_sha256"] for row in ledger["reservations"]):
            raise PermissionError("open final batch exhausted or this plan already exposed")
        ledger["reservations"].append({"batch_id": uuid4().hex, "plan_sha256": plan["plan_sha256"],
                                       "status": "reserved", "reserved_at": now_iso()})
        write_json(file, ledger)
    finally:
        lock.unlink(missing_ok=True)


def run(root, cfg, plan, *, provider_factory=None):
    expected = build_plan(root, cfg, split=plan["split"],
                          candidate_release=(None if plan.get("unpublished_candidate") else
                                             plan["arms"].get("candidate", {}).get("release_id")),
                          candidate=plan.get("unpublished_candidate"))
    if expected != plan:
        raise ValueError("open benchmark execution differs from frozen plan")
    if plan["split"] == "final_holdout":
        _reserve_final(root, plan)  # Before reading ANY sealed request/rubric.
    batch_id = "open-" + uuid4().hex
    private = artifacts_path("evaluation_only", "open_benchmarks", batch_id)
    private.mkdir(parents=True, exist_ok=False)
    write_json(private / "plan.json", plan)
    write_json(private / "status.json", {"status": "started"})
    cases = []
    try:
        manifest = verify_dataset(root)
        rows = read_json(Path(root) / manifest["files"][plan["split"]]["path"])
        metadata_by_task = {entry["task_id_sha256"]: entry
                            for entry in manifest["splits"][plan["split"]]["cases"]}
        for case in rows:
            request = research.request_from_dict(case["request"])
            task_sha = sha256_text(request.task_id)
            metadata = metadata_by_task.get(task_sha)
            group_sha = sha256_text(case["group_id"])
            if metadata is not None and metadata.get("group_sha256") != group_sha:
                raise ValueError("frozen task group differs from manifest identity")
            for arm, identity in plan["arms"].items():
                state = release.load_state(identity["release_id"])
                if arm == "candidate" and plan.get("unpublished_candidate"):
                    state = _candidate_state(state, cfg, plan["unpublished_candidate"])
                request_plan = research.build_plan(request, cfg, state)
                result = research.execute(request, cfg, request_plan["plan_sha256"], state=state,
                    provider=provider_factory(arm, request) if provider_factory else None, evaluator_output=private)
                case_id = sha256_obj({"task": request.task_id, "arm": arm, "batch": batch_id})
                cases.append({"case_id": case_id, "task_sha256": task_sha,
                    "group_sha256": group_sha, "arm": arm,
                    "run_dir": result["run_dir"], "request_sha256": sha256_obj(request.to_dict()),
                    "response_sha256": sha256_obj(result["response"]), "rubric_sha256": sha256_obj(case["rubric"]),
                    "rubric": case["rubric"]})
                write_json(private / "cases.json", cases)
        write_json(private / "status.json", {"status": "awaiting_independent_annotations"})
        if plan["split"] == "final_holdout":
            _finish_final(root, plan, batch_id, "executed_awaiting_annotations")
    except BaseException as exc:
        write_json(private / "status.json", {"status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                                             "error_type": type(exc).__name__})
        if plan["split"] == "final_holdout":
            _finish_final(root, plan, batch_id, "failed_exposure_consumed")
        raise
    public = {"evaluation_id": batch_id, "plan_sha256": plan["plan_sha256"], "n_responses": len(cases),
              "status": "awaiting_independent_annotations", "scientific_scores": None, "promotion_authorized": False}
    write_json(artifacts_path("open_benchmark_reports", batch_id + ".json"), public)
    return public


def _evaluation(evaluation_id):
    if not evaluation_id.startswith("open-") or len(evaluation_id) != 37 or not all(c in "0123456789abcdef" for c in evaluation_id[5:]):
        raise ValueError("invalid open evaluation identity")
    return artifacts_path("evaluation_only", "open_benchmarks", evaluation_id)


def annotation_packet(evaluation_id):
    private = _evaluation(evaluation_id)
    cases, plan = read_json(private / "cases.json"), read_json(private / "plan.json")
    if len(cases) != plan["n_tasks"] * len(plan["arms"]):
        raise ValueError("incomplete benchmark execution cannot be annotated as complete")
    if plan.get("plan_sha256") != sha256_obj({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("open evaluation plan changed")
    manifest = verify_dataset(artifacts_path("open_benchmarks", plan["dataset_sha256"]))
    if manifest["manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("open evaluation no longer matches its frozen task dataset")
    expected = {(row["task_id_sha256"], arm): row for row in manifest["splits"][plan["split"]]["cases"]
                for arm in plan["arms"]}
    seen = set()
    packet = []
    for case in cases:
        key = (case["task_sha256"], case["arm"])
        if (key not in expected or key in seen or
                case["request_sha256"] != expected[key]["request_sha256"] or
                case["rubric_sha256"] != expected[key]["rubric_sha256"] or
                ("group_sha256" in expected[key] and
                 case.get("group_sha256") != expected[key]["group_sha256"])):
            raise ValueError("open evaluation omitted, duplicated or substituted a frozen task/rubric")
        seen.add(key)
        root = Path(case["run_dir"]).resolve()
        root.relative_to(private.resolve())
        receipt = read_json(root / "receipt.json")
        response, request = read_json(root / "response.json"), read_json(root / "request.json")
        if (sha256_file(root / "response.json") != receipt["response_sha256"] or
                sha256_file(root / "request.json") != receipt["request_sha256"] or
                sha256_obj(response) != case["response_sha256"] or sha256_obj(request) != case["request_sha256"] or
                sha256_obj(case["rubric"]) != case["rubric_sha256"]):
            raise ValueError("open answer/request/rubric evidence changed")
        validate_rubric(research.request_from_dict(request), case["rubric"])
        if (sha256_text(request["task_id"]) != case["task_sha256"] or
                case["case_id"] != sha256_obj({"task": request["task_id"], "arm": case["arm"], "batch": evaluation_id})):
            raise ValueError("open annotation case identity changed")
        visible_response = {key: response[key] for key in ("status", "answer", "citations")}
        packet.append({"case_id": case["case_id"], "request": request, "response": visible_response,
                       "task_rubric": case["rubric"], "response_sha256": sha256_obj(visible_response),
                       "receipt_response_sha256": case["response_sha256"],
                       "rubric_sha256": case["rubric_sha256"], "annotations": []})
    # Arm identities are intentionally withheld from human-facing copies.
    return {"schema": "open-benchmark-annotation-packet-v1", "evaluation_id": evaluation_id,
            "plan_sha256": plan["plan_sha256"], "cases": sorted(packet, key=lambda c: c["case_id"])}


def assess(evaluation_id, annotations):
    packet = annotation_packet(evaluation_id)
    if set(annotations) != {"evaluation_id", "cases"} or annotations["evaluation_id"] != evaluation_id:
        raise ValueError("annotations differ from benchmark identity")
    rows = annotations["cases"]
    indexed = {row["case_id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {case["case_id"] for case in packet["cases"]}:
        raise ValueError("human annotations must cover every planned response exactly once")
    results = {}
    for case in packet["cases"]:
        raw_rows = indexed[case["case_id"]]["annotations"]
        from .ai_annotation import _parse_items
        ai_reviewers = {row.get("reviewer") for row in raw_rows if row.get("source") == "independent_ai"}
        expected_items = {("rubric", r["item_id"]) for r in case["task_rubric"]["items"]}
        expected_items |= {("citation", str(i)) for i in range(len(case["response"]["citations"]))}
        for reviewer in ai_reviewers:
            ai_rows = [row for row in raw_rows if row.get("reviewer") == reviewer]
            if any(not {"kind", "item_id", "supported", "rationale", "evidence_quotes"} <= set(row) for row in ai_rows):
                raise ValueError("AI annotation requires original evidence and complete item fields")
            _parse_items({"items": [{k: row[k] for k in ("kind", "item_id", "supported", "rationale", "evidence_quotes")}
                                   for row in ai_rows]}, case, expected_items)
        rubric = {"schema": "open-research-rubric-v1", "author": case["task_rubric"]["author"],
            "authorship": case["task_rubric"]["authorship"], "response_sha256": case["response_sha256"],
            "items": case["task_rubric"]["items"]}
        if "author_provenance" in case["task_rubric"]:
            rubric["author_provenance"] = case["task_rubric"]["author_provenance"]
        bound = []
        for row in indexed[case["case_id"]]["annotations"]:
            if row.get("rubric_sha256") != case["rubric_sha256"]:
                raise ValueError("human annotation does not bind the pre-answer task rubric")
            converted = {**row, "rubric_sha256": sha256_obj(rubric)}
            if row.get("source") == "independent_ai":
                provenance = row.get("reviewer_provenance") or {}
                if provenance.get("rubric_sha256") != case["rubric_sha256"]:
                    raise ValueError("AI provenance differs from the pre-answer task rubric")
                converted["reviewer_provenance"] = {**provenance,
                    "original_task_rubric_sha256": case["rubric_sha256"], "rubric_sha256": sha256_obj(rubric)}
            bound.append(converted)
        results[case["case_id"]] = assess_annotations(case["response"], rubric, bound)
        from .ai_annotation import apply_adjudications
        results[case["case_id"]] = apply_adjudications(results[case["case_id"]], case,
            indexed[case["case_id"]]["annotations"], indexed[case["case_id"]].get("adjudications", []))
    private = _evaluation(evaluation_id)
    cases = read_json(private / "cases.json")
    arms, citation_metrics = {}, {}
    packet_cases = {case["case_id"]: case for case in packet["cases"]}
    for arm in read_json(private / "plan.json")["arms"]:
        arm_results = [results[c["case_id"]] for c in cases if c["arm"] == arm]
        arms[arm] = {kind: sum(item["adjudicated_scores"][kind] if "adjudicated_scores" in item
                              else min(item["scores_by_reviewer"][kind].values())
                                 for item in arm_results) / len(arm_results)
                     if all((item["adjudicated_scores"][kind] is not None if "adjudicated_scores" in item
                             else all(v is not None for v in item["scores_by_reviewer"][kind].values()))
                            for item in arm_results) else None for kind in ("rubric", "citation")}
        citation_metrics[arm] = citation_statistics([
            (packet_cases[c["case_id"]]["response"], indexed[c["case_id"]]["annotations"],
             indexed[c["case_id"]].get("adjudications", [])) for c in cases if c["arm"] == arm])
        # Empty denominators belong in coverage. They must not discard valid
        # semantic judgments on the arm's other, citation-bearing responses.
        arms[arm]["citation"] = citation_metrics[arm]["citation_precision_macro"]
    disagreement = any(result["disagreements_need_adjudication"] for result in results.values())
    # A third reviewer can resolve one scoring axis while leaving another
    # unresolved (for example, utility is settled but a citation claim needs
    # an abstaining adjudicator).  Keep those axes separate: an unresolved
    # citation must not erase a valid utility estimate or its paired CI.
    unresolved_metrics = set()
    for case_id, result in results.items():
        if not result["disagreements_need_adjudication"]:
            continue
        scores = result.get("adjudicated_scores")
        if isinstance(scores, dict):
            unresolved_metrics.update(kind for kind in ("rubric", "citation")
                                      if scores.get(kind) is None and
                                      (kind != "citation" or packet_cases[case_id]["response"]["citations"]))
        else:
            # Without an adjudication object there is no safe way to identify
            # the disputed axis from the aggregate reviewer scores.
            unresolved_metrics.update(("rubric", "citation"))
    paired_analysis = None
    if ("baseline" in arms and "candidate" in arms and
            "rubric" not in unresolved_metrics and
            arms["baseline"].get("rubric") is not None and
            arms["candidate"].get("rubric") is not None):
        paired_analysis = paired_group_bootstrap(cases, results,
            baseline="baseline", candidate="candidate", metrics=("rubric",))
    origins = {item["annotation_origin"] for item in results.values()}
    origin = next(iter(origins)) if len(origins) == 1 else "mixed"
    if disagreement and unresolved_metrics == {"citation"}:
        status = "assessed_partial"
    elif disagreement:
        status = "needs_ai_adjudication" if origin == "ai" else "needs_human_adjudication"
    else:
        status = "assessed"
    report = {"evaluation_id": evaluation_id, "status": status,
        "n_responses": len(results), "arm_scores": arms, "citation_metrics": citation_metrics,
        "aggregation": "third-AI item adjudication; minimum reviewer score for unadjudicated items" if origin == "ai" else "minimum reviewer score per response",
        "plan_sha256": packet["plan_sha256"], "annotations_sha256": sha256_obj(annotations),
        "provider": read_json(private / "plan.json")["provider"],
        "annotation_origin": origin, "paired_group_analysis": paired_analysis,
        "metric_status": {kind: ("unresolved_ai_adjudication" if kind in unresolved_metrics
                                  and origin == "ai" else
                                  "unresolved_human_adjudication" if kind in unresolved_metrics
                                  else "resolved") for kind in ("rubric", "citation")},
        "unresolved_metrics": sorted(unresolved_metrics),
        "promotion_authorized": False, "generalization_claim_authorized": False}
    write_json(private / "assessment.json", {"cases": results, "annotations": annotations})
    write_json(artifacts_path("open_benchmark_reports", evaluation_id + ".json"), report)
    return report
