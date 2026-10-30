"""Public data preparation must fail before writing partial split artifacts."""

import subprocess
import sys
import json
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_split_loader_preserves_generator_scope_for_evidence_inference():
    from src.data import split
    with patch("src.data.evidence_inference_v2.load_tasks", return_value=[]) as loader:
        split.load_tasks("evidence_inference_v2_abstract", splits=(s for s in ["selection_gate"]))
    loader.assert_called_once_with(splits=["selection_gate"], limit=None)


def test_split_assignment_rejects_invalid_ratio_contracts():
    from src.data.split import assign_split
    for ratios in ({}, {"dev_optimize": 60, "selection_gate": 20, "final_holdout": 10},
                   {"dev_optimize": -1, "selection_gate": 21, "final_holdout": 80},
                   {"dev_optimize": True, "selection_gate": 19, "final_holdout": 80}):
        with pytest.raises(ValueError, match="ratio"):
            assign_split("public_test", "task-1", ratios=ratios)


@pytest.mark.parametrize("module_name", ["scripts.make_cs_transfer", "scripts.make_suppqa_final"])
def test_sealed_dataset_writers_refuse_to_replace_frozen_data(tmp_path, monkeypatch, module_name):
    import importlib
    from src.data import freeze
    from src.util import write_json
    module = importlib.import_module(module_name)
    source = tmp_path / "source"
    source.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(module, "SOURCE", str(source))
    monkeypatch.setattr(freeze, "FREEZE_FILE", tmp_path / "freeze.json")
    write_json(freeze.FREEZE_FILE, {"datasets": {module.DATASET: {}}})
    task = {"task_id": "synthetic-1", "split": "final_holdout"}
    result = ([task], {}) if module_name.endswith("make_cs_transfer") else ([task], {}, [])
    monkeypatch.setattr(module, "build_rows", lambda _: result)
    with pytest.raises(ValueError, match="cannot rewrite frozen datasets"):
        module.main()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["freeze.json", "source"]


def test_split_loader_rejects_unsafe_names_and_invalid_limits():
    from src.data import split
    with pytest.raises(ValueError, match="dataset"):
        split.load_tasks("../outside")
    with pytest.raises(ValueError, match="dataset"):
        split.load_manifest("../outside")
    with pytest.raises(ValueError, match="limit"):
        split.load_tasks("public_test", limit=-1)
    with pytest.raises(ValueError, match="splits"):
        split.load_tasks("public_test", splits="dev_optimize")


def test_empty_split_request_does_not_expand_to_all_visible_tasks(tmp_path, monkeypatch):
    from src.data import freeze, split
    from src.util import write_jsonl
    monkeypatch.setattr(split, "NORMALIZED_DIR", tmp_path)
    monkeypatch.setattr(freeze, "FREEZE_FILE", tmp_path / "absent-freeze.json")
    write_jsonl(tmp_path / "public_test.jsonl", [{"task_id": "x", "split": "dev_optimize"}])
    assert split.load_tasks("public_test", splits=[]) == []


def test_missing_lawful_inputs_fail_without_optional_dependencies(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/make_splits.py"),
               "--litqa2-parquet", str(tmp_path / "missing-litqa2.parquet"),
               "--protocolqa-parquet", str(tmp_path / "missing-protocolqa.parquet")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Parquet input missing" in result.stderr


def test_optimizer_bootstrap_explains_missing_sources():
    if (ROOT / "vendor/dspy-main/dspy-main").exists():
        return
    result = subprocess.run([sys.executable, str(ROOT / "scripts/bootstrap_project.py"),
                             "--install-optimizers"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "--restore-vendor" in result.stderr


def test_pi_install_does_not_require_optimizer_vendor():
    from scripts import bootstrap_project

    with patch.object(bootstrap_project.shutil, "which", return_value="npm"), \
         patch.object(bootstrap_project.subprocess, "run") as run:
        assert bootstrap_project.main(["--install-pi"]) == 0
    run.assert_called_once()


def _prepared_inputs(tmp_path, monkeypatch, *, duplicate=False, invalid_second=False):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from scripts import make_splits
    from src.data import freeze, split

    data = tmp_path / "data"
    for module in (make_splits, freeze, split):
        monkeypatch.setattr(module, "MANIFEST_DIR", data / "manifests")
        monkeypatch.setattr(module, "NORMALIZED_DIR", data / "normalized")
        monkeypatch.setattr(module, "SEALED_DIR", data / "sealed")
    monkeypatch.setattr(freeze, "FREEZE_FILE", data / "manifests/freeze.json")
    paths = {}
    for dataset in ("litqa2", "protocolqa"):
        rows = [{"id": f"{dataset}-{0 if duplicate and dataset == 'litqa2' and i == 1 else i}",
                 "question": f"Synthetic question {i}?", "ideal": "alpha",
                 "distractors": ["beta", "gamma"],
                 "key-passage": f"Synthetic evidence {i}",
                 "protocol": f"Synthetic protocol {i}"} for i in range(40)]
        if invalid_second and dataset == "protocolqa":
            for row in rows:
                row.pop("id")
        paths[dataset] = tmp_path / f"{dataset}.parquet"
        pd.DataFrame(rows).to_parquet(paths[dataset], index=False)
    args = ["--litqa2-parquet", str(paths["litqa2"]),
            "--protocolqa-parquet", str(paths["protocolqa"])]
    return make_splits, freeze, data, args


def test_invalid_second_parquet_leaves_no_partial_outputs(tmp_path, monkeypatch):
    make_splits, _, data, args = _prepared_inputs(tmp_path, monkeypatch, invalid_second=True)
    with pytest.raises(ValueError, match="missing Parquet columns"):
        make_splits.main(args)
    assert not (data / "manifests/litqa2.json").exists()
    assert not (data / "normalized/litqa2.jsonl").exists()


def test_duplicate_task_ids_fail_before_writing(tmp_path, monkeypatch):
    make_splits, _, data, args = _prepared_inputs(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match="duplicate task IDs"):
        make_splits.main(args)
    assert not data.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_commit_failure_restores_previous_outputs(tmp_path, monkeypatch, existing):
    make_splits, _, data, args = _prepared_inputs(tmp_path, monkeypatch)
    targets = [data / folder / name for folder, name in (
        ("manifests", "litqa2.json"), ("normalized", "litqa2.jsonl"),
        ("sealed", "litqa2.final_holdout.jsonl"),
        ("manifests", "protocolqa.json"), ("normalized", "protocolqa.jsonl"),
        ("sealed", "protocolqa.final_holdout.jsonl"))]
    if existing:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"previous artifact")
    original = make_splits._install_file
    calls = 0

    def fail_on_second(pending, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic commit failure")
        original(pending, target)

    monkeypatch.setattr(make_splits, "_install_file", fail_on_second)
    with pytest.raises(OSError, match="synthetic commit failure"):
        make_splits.main(args)
    if existing:
        assert all(target.read_bytes() == b"previous artifact" for target in targets)
    else:
        assert not any(target.exists() for target in targets)
    assert not list(data.glob(".split-stage-*"))


def test_freeze_rejects_missing_sealed_file_without_writing_record(tmp_path, monkeypatch):
    make_splits, freeze, data, args = _prepared_inputs(tmp_path, monkeypatch)
    assert make_splits.main(args) == 0
    (data / "sealed/litqa2.final_holdout.jsonl").unlink()
    with pytest.raises(freeze.SplitIntegrityError, match="nonempty sealed file"):
        freeze.freeze(["litqa2", "protocolqa"])
    assert not freeze.FREEZE_FILE.exists()


def test_freeze_rejects_duplicate_manifest_ids_even_with_updated_hash(tmp_path, monkeypatch):
    from src.util import sha256_obj

    make_splits, freeze, data, args = _prepared_inputs(tmp_path, monkeypatch)
    assert make_splits.main(args) == 0
    manifest_path = data / "manifests/litqa2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = next(row for row in manifest["splits"].values() if row["ids"])
    split["ids"].append(split["ids"][0])
    split["n"] = len(split["ids"])
    split["ids_sha256"] = sha256_obj(sorted(split["ids"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(freeze.SplitIntegrityError, match="duplicate task IDs"):
        freeze.freeze(["litqa2", "protocolqa"])
    assert not freeze.FREEZE_FILE.exists()


def test_preparation_cannot_overwrite_frozen_datasets(tmp_path, monkeypatch):
    make_splits, freeze, data, args = _prepared_inputs(tmp_path, monkeypatch)
    assert make_splits.main(args) == 0
    freeze.freeze(["litqa2", "protocolqa"])
    before = {path: path.read_bytes() for path in data.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit):
        make_splits.main(args)
    assert all(path.read_bytes() == contents for path, contents in before.items())
