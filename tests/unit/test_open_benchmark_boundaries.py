"""Cross-split grouping must not be bypassed by presentation differences."""
import copy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval import open_benchmark, open_research
from src.runtime.research_assistant import request_from_dict
from src.util import read_json, sha256_obj, write_json


def dataset():
    partitions = {}
    for split in sorted(open_benchmark.SPLITS):
        request = request_from_dict({"task_id": "case-" + split, "task_type": "open_research",
            "query": "What did the experiment report?", "project_id": "fixture", "session_id": "session",
            "evidence": [{"source_id": "paper", "text": "Control evidence for " + split}]})
        rubric = {"schema": "open-research-task-rubric-v1", "author": "fixture-author",
            "authorship": "independent_human", "request_sha256": sha256_obj(request.to_dict()),
            "items": [{"item_id": "control", "criterion": "Report the control.",
                "evidence_quotes": [{"source_id": "paper", "quote": "Control evidence"}]}]}
        partitions[split] = [{"request": request.to_dict(), "group_id": "article-" + split, "rubric": rubric}]
    return {"schema": "open-research-dataset-v1", "provenance": {"source": "test fixture",
        "revision": "v1", "license": "SELF-AUTHORED", "reference_origin": "synthetic"}, "partitions": partitions}


def test_group_aliases_cannot_cross_partitions():
    groups = {"dev_optimize": [{"task_id": "a", "group_id": "article 1"}],
              "selection_gate": [{"task_id": "b", "group_id": " article\t１ "}],
              "final_holdout": [{"task_id": "c", "group_id": "other"}]}
    with pytest.raises(ValueError, match="crosses"):
        open_research.validate_group_partition(groups)


def test_same_evidence_with_unicode_and_whitespace_changes_is_rejected():
    value = dataset()
    source = value["partitions"]["dev_optimize"][0]
    target = value["partitions"]["selection_gate"][0]
    target["request"]["evidence"][0]["text"] = source["request"]["evidence"][0]["text"].replace("Control ", "Ｃontrol\n  ")
    target["rubric"]["request_sha256"] = sha256_obj(request_from_dict(target["request"]).to_dict())
    target["rubric"]["items"][0]["evidence_quotes"][0]["quote"] = "evidence"
    with pytest.raises(ValueError, match="same article/source"):
        open_benchmark.freeze(copy.deepcopy(value))


def test_equivalent_groups_within_a_partition_are_stored_as_one_group():
    value = dataset()
    first = value["partitions"]["dev_optimize"][0]
    first["group_id"] = "article 1"
    second = copy.deepcopy(first)
    second["group_id"] = " article\t１ "
    second["request"]["task_id"] = "second-dev-case"
    second["rubric"]["request_sha256"] = sha256_obj(request_from_dict(second["request"]).to_dict())
    value["partitions"]["dev_optimize"].append(second)
    original = copy.deepcopy(value)
    manifest = open_benchmark.verify_dataset(open_benchmark.freeze(value))
    rows = manifest["splits"]["dev_optimize"]["cases"]
    assert rows[0]["group_sha256"] == rows[1]["group_sha256"]
    assert value == original


def test_same_source_in_one_partition_cannot_be_counted_as_two_groups():
    value = dataset()
    first = value["partitions"]["dev_optimize"][0]
    second = copy.deepcopy(first)
    second["group_id"] = "different-claimed-group"
    second["request"]["task_id"] = "second-dev-case"
    second["rubric"]["request_sha256"] = sha256_obj(request_from_dict(second["request"]).to_dict())
    value["partitions"]["dev_optimize"].append(second)
    with pytest.raises(ValueError, match="multiple groups"):
        open_benchmark.freeze(value)


def test_failed_freeze_does_not_leave_an_unusable_partial_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(open_benchmark, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    original_write = open_benchmark.write_json
    calls = []
    def fail_second(path, value):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated disk failure")
        return original_write(path, value)
    monkeypatch.setattr(open_benchmark, "write_json", fail_second)
    with pytest.raises(OSError, match="simulated disk failure"):
        open_benchmark.freeze(dataset())
    assert list((tmp_path / "open_benchmarks").iterdir()) == []


def test_force_terminated_freeze_can_be_retried(tmp_path, monkeypatch):
    code = """
import os, sys
from pathlib import Path
from src.eval import open_benchmark
from tests.unit.test_open_benchmark_boundaries import dataset
base = Path(sys.argv[1])
open_benchmark.artifacts_path = lambda *parts: base.joinpath(*parts)
original = open_benchmark.write_json
def terminate_after_first(path, value):
    original(path, value)
    os._exit(71)
open_benchmark.write_json = terminate_after_first
open_benchmark.freeze(dataset())
"""
    root = Path(__file__).resolve().parents[2]
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=root, capture_output=True)
    assert child.returncode == 71
    pending = list((tmp_path / "open_benchmarks").iterdir())
    assert len(pending) == 1 and ".pending-" in pending[0].name
    monkeypatch.setattr(open_benchmark, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    frozen = open_benchmark.freeze(dataset())
    assert open_benchmark.verify_dataset(frozen)["partition_policy"] == open_benchmark.PARTITION_POLICY


def test_concurrent_complete_freeze_is_verified_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(open_benchmark, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    def concurrent_publish(staging, destination):
        shutil.copytree(staging, destination)
        raise FileExistsError("another writer published first")
    monkeypatch.setattr(open_benchmark.os, "rename", concurrent_publish)
    frozen = open_benchmark.freeze(dataset())
    assert open_benchmark.verify_dataset(frozen)["partition_policy"] == open_benchmark.PARTITION_POLICY
    assert not any(".pending-" in path.name for path in frozen.parent.iterdir())


def test_feedback_source_identity_is_stable_across_equivalent_text(tmp_path, monkeypatch):
    from src.governance import open_promotion
    from src.evolution import session_feedback
    monkeypatch.setattr(open_promotion, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    feedback = {"feedback_id": "f1", "use_for_evolution": True, "category": "procedure",
                "run_dir": str(tmp_path / "run"), "run_receipt_sha256": "receipt"}
    feedback["feedback_sha256"] = sha256_obj(feedback)
    write_json(tmp_path / "research_feedback/p/s/feedback.json", feedback)
    monkeypatch.setattr(open_promotion, "sha256_file", lambda _: "receipt")
    source = {"text": "Ａ  control\nresult"}
    monkeypatch.setattr(session_feedback, "verify_research_run", lambda _: ({"evidence": [source]}, {}))
    candidate = SimpleNamespace(extra={"feedback_ids": ["f1"]})
    first = open_promotion._feedback_sources(candidate)
    source["text"] = "A control result"
    assert open_promotion._feedback_sources(candidate) == first


def test_promotion_rejects_feedback_text_reformatted_in_evaluation(tmp_path, monkeypatch):
    from src.governance import open_promotion
    cfg = {"runtime": {"provider": "fake"}}
    candidate = SimpleNamespace(to_dict=lambda: {"candidate_id": "candidate"})
    monkeypatch.setattr(open_promotion.benchmark, "_evaluation", lambda _: tmp_path)
    monkeypatch.setattr(open_promotion.benchmark, "assess", lambda *_: {"status": "assessed", "unresolved_metrics": []})
    monkeypatch.setattr(open_promotion.benchmark, "_candidate_state", lambda *_: object())
    monkeypatch.setattr(open_promotion.release, "load_state", lambda: object())
    monkeypatch.setattr(open_promotion.release, "current_release_id", lambda: "parent")
    monkeypatch.setattr(open_promotion, "config_fingerprint", lambda _: "protocol")
    monkeypatch.setattr(open_promotion, "source_tree_fingerprint", lambda: {"sha256": "source"})
    write_json(tmp_path / "plan.json", {"split": "dev_optimize", "unpublished_candidate": candidate.to_dict(),
        "config_sha256": sha256_obj(cfg), "protocol_sha256": "protocol", "source_tree_sha256": "source",
        "arms": {"baseline": {"release_id": "parent"}, "candidate": {}}})
    write_json(tmp_path / "assessment.json", {"annotations": []})
    request = request_from_dict({"task_id": "case", "task_type": "open_research", "query": "What is missing?",
        "project_id": "fixture", "session_id": "session", "evidence": [{"source_id": "paper", "text": "A control result"}]})
    write_json(tmp_path / "run/request.json", request.to_dict())
    write_json(tmp_path / "cases.json", [{"run_dir": str(tmp_path / "run"), "arm": "baseline"}])
    feedback_sources = open_research.source_identity_keys({"text": "Ａ  control\nresult"})
    with pytest.raises(ValueError, match="feedback source leaked"):
        open_promotion._assessment("fixture", "dev_optimize", candidate, cfg, feedback_sources)


def test_legacy_freeze_is_auditable_but_cannot_create_new_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(open_benchmark, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    current = open_benchmark.freeze(dataset())
    manifest = read_json(current / "manifest.json")
    assert manifest["partition_policy"] == open_benchmark.PARTITION_POLICY
    manifest.pop("partition_policy")
    legacy_identity = sha256_obj({"schema": "open-research-dataset-v1", "provenance": manifest["provenance"],
        "files": {split: item["sha256"] for split, item in manifest["files"].items()},
        "splits": manifest["splits"]})
    legacy = current.parent / legacy_identity
    shutil.copytree(current, legacy)
    manifest["dataset_sha256"] = legacy_identity
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256_obj(manifest)
    write_json(legacy / "manifest.json", manifest)
    assert open_benchmark.verify_dataset(legacy)["dataset_sha256"] == legacy_identity
    with pytest.raises(ValueError, match="legacy open benchmark needs a fresh freeze"):
        open_benchmark.build_plan(legacy, {"runtime": {"provider": "fake"}})
