"""Common cohort identity/group/label validation, separated from MIPRO orchestration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from .evidence_inference_v2 import DATASET as EI_DATASET
from .healthver_task_input import DATASET as HEALTHVER_DATASET
from .clinifact_task_input import DATASET as CLINIFACT_DATASET
from .scifact_task_input import DATASET as SCIFACT_DATASET, SOURCE_SHA256 as SCIFACT_SOURCE_SHA256
from .healthfc_task_input import DATASET as HEALTHFC_DATASET, SOURCE_SHA256 as HEALTHFC_SOURCE_SHA256


@dataclass(frozen=True)
class CohortContext:
    dataset: str
    manifest_path: Path
    root: Path
    labels: Sequence[str]
    healthfc_history_report_path: Path
    read_json: Callable
    file_hash: Callable
    object_hash: Callable


def validate_cohort(context: CohortContext,
        cohort_path: Path,
        train_rows: list[dict[str, Any]],
        *,
        clinifact_history_report: Path | None = None):
    DATASET, MANIFEST_PATH, ROOT, LABELS = context.dataset, context.manifest_path, context.root, context.labels
    HEALTHFC_HISTORY_REPORT_PATH = context.healthfc_history_report_path
    read_json, sha256_file, sha256_obj = context.read_json, context.file_hash, context.object_hash
    cohort = read_json(cohort_path)
    expected_sha = str(cohort.get("cohort_sha256") or "")
    unhashed = dict(cohort)
    unhashed.pop("cohort_sha256", None)
    if not expected_sha or sha256_obj(unhashed) != expected_sha:
        raise ValueError("nested cohort content hash mismatch")
    if cohort.get("dataset") != DATASET or cohort.get("eligible_for_confirmation") is not False:
        raise ValueError("nested cohort identity/eligibility mismatch")
    if DATASET == EI_DATASET:
        if cohort.get("official_validation_or_test_read") is not False:
            raise ValueError("EI MIPRO cohort must attest that official validation/test were not read")
    elif DATASET == HEALTHVER_DATASET:
        if (cohort.get("source_upstream_test_materialized") is not True
                or cohort.get("source_test_is_blind_confirmation") is not False
                or cohort.get("project_selection_gate_or_confirmation_read") is not False):
            raise ValueError("HealthVer MIPRO cohort must disclose materialized upstream test and exploratory-only scope")
    elif DATASET == SCIFACT_DATASET:
        if (cohort.get("official_test_read") is not False
                or cohort.get("source_test_is_blind_confirmation") is not False
                or cohort.get("project_selection_gate_or_confirmation_read") is not False
                or cohort.get("selection_or_confirmation_evidence") is not False):
            raise ValueError("SciFact MIPRO cohort must keep the unlabeled test and project gate sealed")
    elif DATASET == HEALTHFC_DATASET:
        if (cohort.get("official_test_exists") is not False
                or cohort.get("official_test_read") is not False
                or cohort.get("source_cv_reused") is not False
                or cohort.get("history_scan_complete") is not True
                or cohort.get("eligible_for_confirmation") is not False
                or cohort.get("selection_or_confirmation_evidence") is not False):
            raise ValueError("HealthFC cohort must be history-screened exploratory data only")
    elif DATASET == CLINIFACT_DATASET:
        if (cohort.get("official_test_downloaded_or_read") is not False
                or cohort.get("eligible_for_confirmation") is not False
                or cohort.get("selection_or_confirmation_evidence") is not False
                or cohort.get("project_selection_gate_read") is not False):
            raise ValueError("CliniFact cohort must keep official test sealed and remain exploratory")
    if cohort.get("candidate_runtime_uses_live_retrieval") is not False:
        raise ValueError("MIPRO cohort cannot use live retrieval")
    minimum_annotation_votes = int(cohort.get("minimum_annotation_votes", 1))
    if minimum_annotation_votes < 1:
        raise ValueError("nested cohort annotation vote threshold is invalid")
    if cohort.get("source_manifest_sha256") != sha256_file(MANIFEST_PATH):
        raise ValueError("nested cohort source manifest changed")
    manifest = read_json(MANIFEST_PATH)
    if DATASET == EI_DATASET:
        if cohort.get("source_train_ids_sha256") != manifest["splits"]["dev_optimize"]["ids_sha256"]:
            raise ValueError("nested cohort is not bound to current train task IDs")
    elif DATASET == HEALTHVER_DATASET and cohort.get("source_task_ids_sha256") != manifest["task_ids_sha256"]:
        raise ValueError("HealthVer cohort is not bound to current source task IDs")
    elif DATASET == CLINIFACT_DATASET:
        expected_hashes = {
            split: manifest["splits"][split]["sha256"]
            for split in ("train", "validation")}
        if cohort.get("source_hashes") != expected_hashes:
            raise ValueError("CliniFact cohort is not bound to audited train/validation bytes")
        if cohort.get("source_task_ids_sha256") != sha256_obj(sorted(
                str(row["task_id"]) for row in train_rows)):
            raise ValueError("CliniFact cohort is not bound to current source task IDs")
        history_path = (clinifact_history_report or
                        ROOT / "artifacts/probe/clinifact_history_scan_20261008.json")
        history = read_json(history_path)
        if (history.get("complete") is not True
                or history.get("source_sha256") != expected_hashes
                or cohort.get("history_scan_sha256") != sha256_obj(history)):
            raise ValueError("CliniFact history scan is missing, stale, or incomplete")

    by_id = {str(row["task_id"]): row for row in train_rows}
    if len(by_id) != len(train_rows):
        raise ValueError(f"{DATASET} source task rows contain duplicate task IDs")
    if DATASET == SCIFACT_DATASET:
        if cohort.get("source_archive_sha256") != SCIFACT_SOURCE_SHA256:
            raise ValueError("SciFact cohort is not bound to the audited source archive")
        if cohort.get("source_task_ids_sha256") != sha256_obj(sorted(by_id)):
            raise ValueError("SciFact cohort is not bound to current labeled train/dev task IDs")
    elif DATASET == HEALTHFC_DATASET:
        if cohort.get("source_archive_sha256") != HEALTHFC_SOURCE_SHA256:
            raise ValueError("HealthFC cohort is not bound to audited source CSV bytes")
        if cohort.get("source_task_ids_sha256") != sha256_obj([
                f"healthfc:{HEALTHFC_SOURCE_SHA256[:12]}:r{index:04d}"
                for index in range(len(train_rows))]):
            raise ValueError("HealthFC cohort is not bound to current source task IDs")
        history = read_json(HEALTHFC_HISTORY_REPORT_PATH)
        if (history.get("complete") is not True
                or history.get("source_sha256") != HEALTHFC_SOURCE_SHA256
                or cohort.get("history_scan_sha256") != sha256_obj(history)):
            raise ValueError("HealthFC history screen is missing, stale, or incomplete")
    scopes: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    group_scopes: dict[str, str] = {}
    for name, entry in (cohort.get("scopes") or {}).items():
        ids = [str(value) for value in entry.get("task_ids") or []]
        if len(ids) != int(entry.get("task_count", -1)) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid task IDs/count in cohort scope {name}")
        if all_ids.intersection(ids):
            raise ValueError("nested cohort task IDs overlap")
        missing = set(ids) - by_id.keys()
        if missing:
            raise ValueError(f"cohort scope {name} escaped dev_optimize: {sorted(missing)[:3]}")
        selected = [by_id[task_id] for task_id in ids]
        labels = {label: 0 for label in LABELS}
        articles: set[str] = set()
        for row in selected:
            if DATASET == EI_DATASET:
                try:
                    vote_count = int((row.get("metadata") or {}).get(
                        "valid_abstract_label_vote_count", 1))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid annotation vote metadata in {row['task_id']}") from exc
                if vote_count < minimum_annotation_votes:
                    raise ValueError(
                        f"cohort task {row['task_id']} is below its annotation vote threshold"
                    )
            label = str(row["options"][int(row["correct_index"])])
            if label not in labels:
                raise ValueError(f"unexpected Evidence Inference label in {row['task_id']}")
            labels[label] += 1
            group_id = str(row.get("group_id") or "")
            if not group_id:
                raise ValueError(f"missing task grouping ID for {row['task_id']}")
            if group_id in group_scopes and group_scopes[group_id] != name:
                raise ValueError("one article/component appears in more than one MIPRO scope")
            group_scopes[group_id] = name
            articles.add(group_id)
        group_count = int(entry.get("component_count" if DATASET in {HEALTHVER_DATASET, SCIFACT_DATASET, CLINIFACT_DATASET}
                                    else "article_count", -1))
        if labels != entry.get("class_counts") or len(articles) != group_count:
            raise ValueError(f"cohort labels/group counts changed in {name}")
        if not all(labels.values()):
            raise ValueError(f"MIPRO scope {name} does not cover all three labels")
        scopes[name] = selected
        all_ids.update(ids)

    required = ({"mipro_train", "search_validation", "internal_confirmation"}
                if DATASET == EI_DATASET else
                {"mipro_train", "search_validation", "internal_candidate"})
    if set(scopes) != required:
        raise ValueError("cohort does not define the required nested MIPRO scopes")
    return cohort, scopes, by_id
