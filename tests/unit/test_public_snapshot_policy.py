"""The public exporter must not accept private paths even in a forged manifest."""

from scripts.build_public_snapshot import allowed_path, source_files


def test_public_allowlist_keeps_runtime_modules_and_baselines():
    assert allowed_path("src/data/split.py")
    assert allowed_path("artifacts/releases/baseline-v1/release.json")
    assert allowed_path("docs/PUBLIC-EVIDENCE-2026-10-19.json")
    assert allowed_path("tests/unit/test_review_publish_guard.py")


def test_public_allowlist_rejects_private_and_escape_paths():
    for path in (
        ".env", ".workbuddy/memory/private.md", "data/sealed/holdout.jsonl",
        "vendor/dspy/secret.py", "artifacts/http_cache/response.json",
        "artifacts/releases/rel-cand-secret/release.json",
        "configs/experiment.docker-tools.yaml", "configs/experiment.docker-tools-v2.yaml",
        "../.env", "src/../.env", "C:\\Users\\someone\\.env",
        "tests/unit/test_bioasq_unitor.py",
    ):
        assert not allowed_path(path), path


def test_exported_source_paths_obey_the_same_policy():
    from scripts.build_public_snapshot import ROOT

    assert all(allowed_path(path.relative_to(ROOT).as_posix()) for path in source_files())


def test_source_fingerprint_detects_same_size_edit_with_restored_timestamp(tmp_path, monkeypatch):
    import os
    from src.eval import run_manifest

    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(run_manifest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_manifest, "path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(run_manifest, "_SOURCE_ROOTS", ("src",))
    old_stat = source.stat()
    original = run_manifest.source_tree_fingerprint()["sha256"]
    source.write_text("value = 2\n", encoding="utf-8")
    os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert run_manifest.source_tree_fingerprint()["sha256"] != original
