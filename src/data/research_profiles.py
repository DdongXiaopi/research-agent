"""Immutable source profiles for the common MIPRO runner.

Source-specific evidence/label validation remains in its data adapter. Profiles
bind entry-point defaults; they do not grant blind-test status to public data.
"""
from dataclasses import dataclass
from typing import Callable, Sequence

from ..util import path
from . import evidence_inference_v2 as ei, evidence_inference_cohort as ei_cohort
from . import healthver_task_input as healthver, scifact_task_input as scifact
from . import healthfc_task_input as healthfc, clinifact_task_input as clinifact


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    dataset: str
    labels: Sequence[str]
    loader: Callable
    release_id: str
    cohort_path: object
    manifest_path: object
    study_namespace: str


PROFILES = {
    "evidence_inference": DatasetProfile("evidence_inference", ei.DATASET, ei_cohort.LABELS, ei.load_tasks,
        "task-input-evidence-inference-v1", path("artifacts/probe/evidence_inference_nested_cohort.json"),
        path("data/manifests", ei.DATASET + ".json"), "evidence_inference_mipro"),
    "healthver": DatasetProfile("healthver", healthver.DATASET, healthver.LABELS, healthver.load_tasks,
        "task-input-healthver-v1", path("artifacts/probe/healthver_mipro_cohort_candidate.json"),
        healthver.MANIFEST_PATH, "healthver_mipro"),
    "scifact": DatasetProfile("scifact", scifact.DATASET, scifact.LABELS, scifact.load_tasks,
        "task-input-scifact-v1", path("artifacts/probe/scifact_mipro_exploratory_cohort_20261008_v4.json"),
        path("artifacts/probe/scifact_self_evolution_source_audit_20261008.json"), "scifact_mipro"),
    "healthfc": DatasetProfile("healthfc", healthfc.DATASET, healthfc.LABELS_ORDERED, healthfc.load_tasks,
        "task-input-healthfc-v2", path("artifacts/probe/healthfc_mipro_exploratory_cohort_20261008.json"),
        healthfc.AUDIT_PATH, "healthfc_mipro"),
    "clinifact": DatasetProfile("clinifact", clinifact.DATASET, clinifact.LABELS, clinifact.load_tasks,
        "task-input-clinifact-v3", path("artifacts/probe/clinifact_mipro_exploratory_cohort_20261008.json"),
        path("artifacts/probe/clinifact_source_audit_20261008.json"), "clinifact_mipro"),
}


def profile_for(name: str) -> DatasetProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError("unknown MIPRO dataset adapter: " + str(name)) from exc
