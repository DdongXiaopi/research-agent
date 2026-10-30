"""Historical cohort preparation must never overwrite or leave partial data."""
import hashlib
import json

import pytest

from scripts import freeze_fresh_evidence_inference as preparation
from src.util import write_json, write_jsonl


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(preparation, "TARGETS", {"dev_optimize": 1, "selection_gate": 1, "final_holdout": 1})
    files = (tmp_path / "input/train.jsonl", tmp_path / "input/validation.jsonl")
    monkeypatch.setattr(preparation, "SOURCE_FILES", files)
    rows = []
    for label in preparation.LABELS:
        selected = set()
        for index in range(1000):
            group = f"{label}-{index}"
            bucket = int(hashlib.sha256((preparation.SALT + "|" + group).encode()).hexdigest()[:8], 16) % 100
            split = "dev_optimize" if bucket < 50 else "selection_gate" if bucket < 75 else "final_holdout"
            if split not in selected:
                selected.add(split)
                rows.append({"task_id": group, "group_id": group, "gold_label": label})
            if len(selected) == 3:
                break
    write_jsonl(files[0], rows)
    write_jsonl(files[1], [])
    return tmp_path


def test_fresh_writer_creates_once_and_never_replaces_existing_data(workspace):
    output = workspace / "data/normalized"
    result = preparation.freeze(output)
    assert result["historical_exclusion_inventory_complete"] is False
    targets = [output / f"{preparation.DATASET}.jsonl",
               workspace / "data/sealed" / f"{preparation.DATASET}.final_holdout.jsonl",
               workspace / "data/manifests" / f"{preparation.DATASET}.json",
               workspace / "data/manifests" / f"{preparation.DATASET}.excluded.json"]
    original = [path.read_bytes() for path in targets]
    with pytest.raises(FileExistsError):
        preparation.freeze(output)
    assert [path.read_bytes() for path in targets] == original


def test_fresh_writer_rejects_frozen_or_external_output(workspace, tmp_path_factory):
    write_json(workspace / "data/manifests/freeze.json", {"datasets": {preparation.DATASET: {}}})
    with pytest.raises(FileExistsError, match="frozen"):
        preparation.freeze(workspace / "data/normalized")
    outside = tmp_path_factory.mktemp("outside-fresh-cohort")
    with pytest.raises(ValueError, match="inside the project"):
        preparation.freeze(outside)
    assert list(outside.iterdir()) == []


def test_fresh_writer_rolls_back_partial_install(workspace, monkeypatch):
    real_link = preparation.os.link
    calls = []
    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("synthetic install failure")
        return real_link(source, destination)
    monkeypatch.setattr(preparation.os, "link", fail_second)
    with pytest.raises(OSError, match="synthetic install failure"):
        preparation.freeze(workspace / "data/normalized")
    assert not (workspace / "data/normalized" / f"{preparation.DATASET}.jsonl").exists()
    assert not (workspace / "data/manifests" / f"{preparation.DATASET}.json").exists()


def test_fresh_writer_rejects_unreadable_history(workspace):
    receipt = workspace / "artifacts/evidence-cohort-invalid.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="historical cohort receipt"):
        preparation.freeze(workspace / "data/normalized")
    assert not (workspace / "data/normalized" / f"{preparation.DATASET}.jsonl").exists()
