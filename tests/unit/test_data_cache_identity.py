"""Source adapters must not reuse stale or caller-mutated cached tasks."""
import json
import os

import pytest


def _replace_preserving_time(path, content):
    stat = path.stat()
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def _bioasq_row(evidence):
    return {"id": "1", "type": "yesno", "output": "yes", "input_1": "Does it work?",
            "input_2": evidence, "input_3": "", "input_4": ""}


def test_bioasq_task_input_cache_tracks_source_bytes_and_is_defensive(tmp_path, monkeypatch):
    from src.data import bioasq_unitor_task_input as adapter
    source = tmp_path / "train.json"
    evidence = "alpha evidence " * 8
    source.write_text(json.dumps([_bioasq_row(evidence)]), encoding="utf-8")
    monkeypatch.setattr(adapter, "SOURCE_TRAIN", source)
    first = adapter._all_tasks()
    first[0]["evidence"] = "caller modification"
    assert adapter._all_tasks()[0]["evidence"] == evidence.strip()
    changed = "bravo evidence " * 8
    _replace_preserving_time(source, json.dumps([_bioasq_row(changed)]))
    assert adapter._all_tasks()[0]["evidence"] == changed.strip()
    with pytest.raises(PermissionError):
        adapter.load_tasks([])


def test_bioasq_retrieval_caches_follow_source_bytes(tmp_path, monkeypatch):
    from src.data import bioasq_unitor as adapter
    source = tmp_path / "train_set_answGEN.json"
    source.write_text(json.dumps([_bioasq_row("alpha evidence " * 8)]), encoding="utf-8")
    monkeypatch.setattr(adapter, "SOURCE", tmp_path)
    first = adapter._rows("train")
    first[0]["input_2"] = "caller modification"
    assert "alpha" in adapter._rows("train")[0]["input_2"]
    assert "alpha" in adapter._train_passages()[0]
    _replace_preserving_time(source, json.dumps([_bioasq_row("bravo evidence " * 8)]))
    assert "bravo" in adapter._rows("train")[0]["input_2"]
    assert "bravo" in adapter._train_passages()[0]


def test_healthver_cache_keys_include_source_bytes_and_return_copies(tmp_path, monkeypatch):
    from src.data import healthver_task_input as adapter
    archive, split = tmp_path / "archive", tmp_path / "split"
    archive.write_text("archive", encoding="utf-8")
    split.write_text("version-a", encoding="utf-8")
    monkeypatch.setattr(adapter, "SOURCE_ARCHIVE", archive)
    monkeypatch.setattr(adapter, "SPLIT_CANDIDATE", split)
    cache = {}
    monkeypatch.setattr(adapter, "_load_all_tasks_cached",
                        lambda identity: cache.setdefault(identity, [{"identity": identity}]))
    first = adapter._load_all_tasks()
    expected = first[0]["identity"]
    first[0]["identity"] = "caller modification"
    assert adapter._load_all_tasks()[0]["identity"] == expected
    _replace_preserving_time(split, "version-b")
    assert adapter._load_all_tasks()[0]["identity"] != expected


def test_healthver_factorial_cache_tracks_cohort_and_source(tmp_path, monkeypatch):
    from src.data import healthver_factorial_exploration as adapter
    cohort, archive, split = (tmp_path / name for name in ("cohort", "archive", "split"))
    for file in (cohort, archive, split):
        file.write_text("version-a", encoding="utf-8")
    monkeypatch.setattr(adapter, "COHORT_PATH", cohort)
    monkeypatch.setattr(adapter.source_adapter, "SOURCE_ARCHIVE", archive)
    monkeypatch.setattr(adapter.source_adapter, "SPLIT_CANDIDATE", split)
    cache = {}
    monkeypatch.setattr(adapter, "_cohort_and_tasks_cached",
                        lambda identity: cache.setdefault(identity, ({"identity": identity}, {"dev_optimize": []})))
    first, _ = adapter._cohort_and_tasks()
    expected = first["identity"]
    first["identity"] = "caller modification"
    assert adapter._cohort_and_tasks()[0]["identity"] == expected
    _replace_preserving_time(cohort, "version-b")
    assert adapter._cohort_and_tasks()[0]["identity"] != expected
