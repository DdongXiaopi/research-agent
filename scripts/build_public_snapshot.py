"""Build a fresh, allowlisted source tree for a public GitHub repository.

The research checkout contains sealed evaluation data and private run history,
including files already present in its Git history. Never push that history as
the public repository. This command copies reviewed source paths into a new
directory and records their hashes. It never removes or modifies source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".env.example", ".gitattributes", ".gitignore", "LICENSE", "README.md", "THIRD_PARTY.yml",
    "THIRD_PARTY_NOTICES.md",
    "UPSTREAM_LOCK.yml", "pytest.ini", "requirements.txt",
    "requirements-data.txt",
    "requirements-documents.txt", "requirements-paperqa.txt",
    "requirements-mipro.txt", "requirements-optimizers.constraints.txt",
}
PUBLIC_TEST_FILES = {
    "tests/__init__.py", "tests/_sandbox.py", "tests/conftest.py",
    "tests/unit/__init__.py", "tests/unit/test_config_schema.py",
    "tests/unit/test_public_data_preflight.py",
    "tests/unit/test_public_capabilities.py", "tests/unit/test_open_promotion_lifecycle.py",
    "tests/unit/test_docker_render_worker.py",
    "tests/unit/test_audit_promotions.py",
    "tests/unit/test_public_smoke_encoding.py",
    "tests/unit/test_data_cache_identity.py",
    "tests/unit/test_fresh_evidence_inference_writer.py",
    "tests/unit/test_preparation_atomic.py",
    "tests/unit/test_metrics.py", "tests/unit/test_stats.py",
    "tests/unit/test_natural_instruction_adjudication.py",
    "tests/unit/test_open_research_sources.py", "tests/unit/test_open_research_v6_continuation.py",
    "tests/unit/test_open_research_v7_capture.py",
    "tests/unit/test_research_cli_plan_guard.py",
    "tests/unit/test_public_snapshot_policy.py",
    "tests/unit/test_release_aliases.py", "tests/unit/test_research_assistant.py",
    "tests/unit/test_review_publish_guard.py",
    "tests/unit/test_rsi_e2e_provider_guard.py",
}
DOC_FILES = {
    "docs/GETTING-STARTED.md",
    "docs/PUBLIC-DOCKER-WORKER.md",
    "docs/PUBLIC-CAPABILITY-EVIDENCE-2026-10-29.json",
    "docs/PUBLIC-EVIDENCE-2026-10-19.json",
    "docs/PUBLIC-RELEASE-PLAN-2026-10-28.md",
    "docs/PROJECT-REPAIR-PLAN-2026-10-29.md",
    "docs/PUBLIC-CAPABILITY-COMPLETION-2026-10-29.md",
    "docs/PUBLIC-RELEASE-SCOPE.md",
    "docs/PUBLIC-STATUS.md",
    "docs/vendor-license-audit-2026-10-10.md",
}
EXCLUDED = {
    "configs/experiment.docker-tools.yaml",  # bound to this machine's deployment
    "scripts/snapshot.py",  # local backup destination, outside public runtime
}
SOURCE_SUFFIXES = {
    "src": {".py"},
    "scripts": {".py"},
    "tests": {".py"},
    "configs": {".yaml", ".json", ".txt", ".md"},
    "skills": {".yaml", ".md", ".txt"},
    "patches": {".patch"},
    "tools": {".py", ".mjs", ".js", ".json", ""},
    ".github": {".yml", ".yaml"},
}
FORBIDDEN_PARTS = {
    ".git", ".workbuddy", ".venv", "__pycache__", "node_modules",
    "outputs", ".pytest_cache",
}
TEXT_SUFFIXES = {".py", ".mjs", ".js", ".json", ".yaml", ".yml", ".txt", ".md", ".patch", ""}
SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
PERSONAL_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\|E:\\program_resume\\)")
LOCAL_LINK = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed_path(relative: str) -> bool:
    """Keep the export policy independent of the manifest being verified."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or relative.startswith("/") or ":" in relative):
        return False
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    if relative in ROOT_FILES | DOC_FILES:
        return True
    if parts[0] == "tests":
        return relative in PUBLIC_TEST_FILES
    if (len(parts) == 2 and parts[0] == "configs"
            and parts[1].startswith("experiment.docker-tools")
            and parts[1].endswith(".yaml")):
        return False  # Deployment profiles bind a local image and receipt path.
    if relative in EXCLUDED or set(parts) & FORBIDDEN_PARTS:
        return False
    if relative.startswith("artifacts/releases/"):
        if relative in {"artifacts/releases/ACTIVE", "artifacts/releases/ALIASES.json"}:
            return True
        return (len(parts) >= 4 and parts[2] in {"baseline-v0", "baseline-v1"}
                and Path(relative).suffix in {".json", ".yaml", ".txt", ".md"})
    if len(parts) < 2 or parts[0] not in SOURCE_SUFFIXES:
        return False
    return Path(relative).suffix in SOURCE_SUFFIXES[parts[0]]


def source_files() -> list[Path]:
    files = {ROOT / name for name in ROOT_FILES | DOC_FILES}
    for folder, suffixes in SOURCE_SUFFIXES.items():
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                rel = path.relative_to(ROOT)
                if allowed_path(rel.as_posix()):
                    files.add(path)
    for release_id in ("baseline-v0", "baseline-v1"):
        base = ROOT / "artifacts" / "releases" / release_id
        for path in base.rglob("*"):
            if path.is_file() and allowed_path(path.relative_to(ROOT).as_posix()):
                files.add(path)
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise ValueError("required public source files missing: " + ", ".join(sorted(missing)))
    linked = [str(path.relative_to(ROOT)) for path in files if path.is_symlink()]
    if linked:
        raise ValueError("source symlinks require explicit review: " + ", ".join(sorted(linked)))
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def scan_text(path: Path, relative: str) -> None:
    if path.suffix not in TEXT_SUFFIXES:
        return
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"oversized public text file requires review: {relative}")
    value = path.read_text(encoding="utf-8")
    if SECRET.search(value):
        raise ValueError(f"credential-like string in {relative}")
    if PERSONAL_PATH.search(value):
        raise ValueError(f"personal absolute path in {relative}")


def verify(directory: Path) -> dict:
    directory = directory.resolve()
    manifest_path = directory / "PUBLIC_SNAPSHOT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rsi-public-snapshot-v1":
        raise ValueError("unexpected public manifest schema")
    expected = manifest["files"]
    required = ROOT_FILES | DOC_FILES | PUBLIC_TEST_FILES | {
        "artifacts/releases/ACTIVE", "artifacts/releases/ALIASES.json",
        "artifacts/releases/baseline-v1/release.json",
        "artifacts/releases/baseline-v1/system_prompt.txt",
        "artifacts/releases/baseline-v1/skills/registry.yaml",
    }
    if (not isinstance(expected, dict) or not required.issubset(expected)
            or any(not allowed_path(relative) or not isinstance(recorded_hash, str)
                   or not SHA256.fullmatch(recorded_hash)
                   for relative, recorded_hash in expected.items())):
        raise ValueError("public manifest contains disallowed paths or hashes")
    for relative, recorded_hash in expected.items():
        path = directory / relative
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(directory)
                or sha256(path) != recorded_hash):
            raise ValueError(f"missing or modified public file: {relative}")
        scan_text(path, relative)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file()
        and not ({".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"} & set(path.relative_to(directory).parts))
    }
    extra = actual - set(expected) - {"PUBLIC_SNAPSHOT_MANIFEST.json"}
    if extra:
        raise ValueError("unlisted public files: " + ", ".join(sorted(extra)[:10]))
    if (directory / ".git").is_dir():
        tracked = subprocess.run(["git", "ls-files", "-z"], cwd=directory,
                                 capture_output=True, check=True).stdout
        tracked_files = {item.decode("utf-8").replace("\\", "/")
                         for item in tracked.split(b"\0") if item}
        if tracked_files != set(expected) | {"PUBLIC_SNAPSHOT_MANIFEST.json"}:
            raise ValueError("Git tracked files differ from the public manifest")
    for relative in ("README.md", *sorted(DOC_FILES)):
        path = directory / relative
        if path.suffix != ".md":
            continue
        for target in LOCAL_LINK.findall(path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("mailto:", "#")):
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(directory)
            except ValueError as exc:
                raise ValueError(f"link escapes public snapshot: {relative} -> {target}") from exc
            if not resolved.exists():
                raise ValueError(f"broken public document link: {relative} -> {target}")
    return {"status": "ok", "file_count": len(expected), "directory": str(directory)}


def build(directory: Path) -> dict:
    directory = directory.resolve()
    allowed_parent = (ROOT / "outputs").resolve()
    if directory == ROOT.resolve() or (
        ROOT.resolve() in directory.parents
        and not (directory.parent == allowed_parent and directory.name.startswith("public-release-"))
    ):
        raise ValueError("output inside source tree must be outputs/public-release-<version>")
    if directory.exists():
        raise FileExistsError(f"output already exists: {directory}")
    directory.mkdir(parents=True)
    records: dict[str, str] = {}
    for source in source_files():
        relative = source.relative_to(ROOT).as_posix()
        scan_text(source, relative)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith("artifacts/releases/"):
            # Release files are immutable, content-addressed protocol assets.
            shutil.copy2(source, target)
        elif source.suffix in TEXT_SUFFIXES:
            target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        else:
            shutil.copy2(source, target)
        records[relative] = sha256(target)
    release_root = directory / "artifacts" / "releases"
    (release_root / "ACTIVE").write_text("baseline-v1\n", encoding="utf-8")
    (release_root / "ALIASES.json").write_text(
        json.dumps({"champion": "baseline-v1", "challenger": None,
                    "archived": ["baseline-v0", "baseline-v1"]}, indent=2) + "\n", encoding="utf-8")
    for name in ("ACTIVE", "ALIASES.json"):
        records[f"artifacts/releases/{name}"] = sha256(release_root / name)
    (directory / "PUBLIC_SNAPSHOT_MANIFEST.json").write_text(
        json.dumps({"schema": "rsi-public-snapshot-v1", "files": records},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return verify(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="new empty public snapshot directory")
    mode.add_argument("--verify", type=Path, help="verify an existing public snapshot")
    args = parser.parse_args()
    print(json.dumps(build(args.output) if args.output else verify(args.verify),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
