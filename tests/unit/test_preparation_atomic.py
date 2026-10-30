"""Preparation entrypoints must not replace frozen or partially written outputs."""
import hashlib
import json
import sys

import pytest

from src.data import freeze
from src.util import commit_new_files, read_json, sha256_file, write_json


def test_exclusive_file_transaction_rolls_back_on_second_install(tmp_path, monkeypatch):
    import src.util as util
    original = util.os.link
    calls = []
    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("simulated failure")
        return original(source, destination)
    monkeypatch.setattr(util.os, "link", fail_second)
    with pytest.raises(OSError, match="simulated failure"):
        commit_new_files([(tmp_path / "one.json", b"one"), (tmp_path / "two.json", b"two")])
    assert list(tmp_path.iterdir()) == []


def test_bioasq_preparation_creates_a_bound_pair_only_once(tmp_path, monkeypatch):
    from src.data import bioasq_unitor_task_input as adapter
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    source = tmp_path / "source.json"
    source.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(adapter, "SOURCE_TRAIN", source)
    monkeypatch.setattr(freeze, "FREEZE_FILE", tmp_path / "data/manifests/freeze.json")
    monkeypatch.setattr(adapter, "_all_tasks", lambda: [{"task_id": "1", "split": "dev_optimize"}])
    manifest = adapter.write_dataset()
    normalized = tmp_path / "data/normalized" / f"{adapter.DATASET}.jsonl"
    assert manifest["normalized_sha256"] == sha256_file(normalized)
    before = normalized.read_bytes()
    with pytest.raises(FileExistsError):
        adapter.write_dataset()
    assert normalized.read_bytes() == before


def test_pubmed_probe_refuses_frozen_output_before_writing(tmp_path, monkeypatch):
    from src.data import pubmedqa_threeway_probe as adapter
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setattr(adapter, "NORMALIZED_PATH", tmp_path / "data/normalized/probe.jsonl")
    monkeypatch.setattr(adapter, "MANIFEST_PATH", tmp_path / "data/manifests/probe.json")
    monkeypatch.setattr(freeze, "FREEZE_FILE", tmp_path / "data/manifests/freeze.json")
    write_json(freeze.FREEZE_FILE, {"datasets": {adapter.DATASET: {}}})
    row = {"task_id": "1", "dataset": adapter.DATASET, "split": adapter.SPLIT,
           "options": ["yes"], "correct_index": 0}
    monkeypatch.setattr(adapter, "normalize_rows", lambda *args: [row])
    with pytest.raises(freeze.SplitIntegrityError, match="frozen"):
        adapter.write_dataset([], {"splits": {"dev_optimize": {"ids": []}}})
    assert not adapter.NORMALIZED_PATH.exists()
    assert not adapter.MANIFEST_PATH.exists()


def test_evidence_inference_preparation_checks_path_and_commits_once(tmp_path, tmp_path_factory, monkeypatch):
    from scripts import prepare_evidence_inference_task_input as adapter
    monkeypatch.setattr(adapter, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(freeze, "FREEZE_FILE", tmp_path / "data/manifests/freeze.json")
    rows = {"train": [{"task_id": "t1"}], "validation": [{"task_id": "v1"}]}
    manifest = {"source_archive_sha256": "a", "abstracts_jsonl_sha256": "b", "license_allowlist": [],
                "label_policy": "fixture", "known_noise_prompt_ids_excluded": [],
                "class_counts": {"train": {}, "validation": {}},
                "article_counts": {"train": 1, "validation": 1},
                "annotation_support": {"train": {}, "validation": {}}}
    monkeypatch.setattr(adapter, "prepare_rows", lambda *args, **kwargs: (rows, dict(manifest)))
    outside = tmp_path_factory.mktemp("outside-ei")
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(outside)])
    with pytest.raises(ValueError, match="inside the project"):
        adapter.main()
    assert list(outside.iterdir()) == []
    destination = tmp_path / "data/normalized/ei"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(destination), "--splits", "train", "validation"])
    assert adapter.main() == 0
    dataset_manifest = read_json(tmp_path / "data/manifests/evidence_inference_v2_abstract.json")
    assert dataset_manifest["splits"]["dev_optimize"]["file_sha256"] == sha256_file(destination / "train.jsonl")
    before = (destination / "train.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        adapter.main()
    assert (destination / "train.jsonl").read_bytes() == before


@pytest.mark.parametrize("module_name,entry_args", [
    ("scripts.prepare_healthver_task_input", ["prepare", "--write"]),
    ("scripts.prepare_healthver_factorial_exploration", ["prepare"]),
])
def test_healthver_manifest_writers_refuse_frozen_data(tmp_path, monkeypatch, module_name, entry_args):
    import importlib
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "MANIFEST_PATH", tmp_path / "healthver.json")
    monkeypatch.setattr(freeze, "FREEZE_FILE", tmp_path / "freeze.json")
    write_json(freeze.FREEZE_FILE, {"datasets": {module.DATASET: {}}})
    monkeypatch.setattr(sys, "argv", entry_args)
    with pytest.raises(freeze.SplitIntegrityError, match="frozen"):
        module.main()
    assert not module.MANIFEST_PATH.exists()


def test_pubmedqa_writer_refuses_frozen_data(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    from scripts import make_pubmedqa as adapter
    source = tmp_path / "source.parquet"
    source.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(adapter, "PARQUET", source)
    monkeypatch.setattr(freeze, "FREEZE_FILE", tmp_path / "freeze.json")
    write_json(freeze.FREEZE_FILE, {"datasets": {adapter.DATASET: {}}})
    monkeypatch.setattr(adapter.pd, "read_parquet", lambda _: object())
    monkeypatch.setattr(adapter, "normalize", lambda *_: [{"task_id": "t1", "split": "dev_optimize"}])
    monkeypatch.setattr(sys, "argv", ["prepare"])
    with pytest.raises(ValueError, match="cannot rewrite frozen datasets"):
        adapter.main()
