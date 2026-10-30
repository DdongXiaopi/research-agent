"""Explicit Skill dependencies; no directory crawl and no implicit execution.

Pi resolves relative references from SKILL.md. Persist the declared files with
the body so a release can restore them. Only reference text may be disclosed;
stored scripts remain inert and require a separate controlled execution adapter.
"""
from pathlib import Path, PurePosixPath
import re
import shutil

from ..util import resolve_artifact, sha256_file, sha256_obj

MAX_BUNDLE_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024


def bundle_files(skill) -> dict[str, Path]:
    declared = skill.bundle_files
    if not declared:
        return {}
    if not skill.bundle_root or len(declared) > 64:
        raise ValueError("invalid Skill bundle root or file count")
    if set(skill.permissions) - {"references.read"}:
        raise ValueError("Skill bundle permissions exceed controlled reference reading")
    root = resolve_artifact(skill.bundle_root).resolve()
    files, total = {}, 0
    for name, expected_sha in sorted(declared.items()):
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
                or not relative.parts or relative.parts[0] not in
                {"references", "scripts", "assets", "LICENSE", "requirements.txt"}
                or not re.fullmatch(r"[a-f0-9]{64}", expected_sha)):
            raise ValueError("invalid Skill bundle dependency identity")
        file = root.joinpath(*relative.parts)
        if file.is_symlink() or not file.is_file():
            raise ValueError("Skill bundle dependency is missing or a symbolic link")
        file.resolve().relative_to(root)
        size = file.stat().st_size
        total += size
        if size > MAX_FILE_BYTES or total > MAX_BUNDLE_BYTES:
            raise ValueError("Skill bundle exceeds byte budget")
        if sha256_file(file) != expected_sha:
            raise ValueError("Skill bundle dependency changed: " + name)
        files[name] = file
    return files


def bundle_identity(skill) -> str:
    bundle_files(skill)
    return sha256_obj({"files": skill.bundle_files, "dependencies": skill.dependencies,
                       "permissions": skill.permissions, "license": skill.license})


def materialize_bundle(skill, destination: Path) -> None:
    files = bundle_files(skill)
    for name, source in files.items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != skill.bundle_files[name]:
                raise ValueError("immutable Skill snapshot dependency differs")
        else:
            shutil.copyfile(source, target)


def render_references(skill, max_chars: int = 12000) -> str:
    files = bundle_files(skill)
    if "references.read" not in skill.permissions:
        return ""
    parts = []
    for name, file in files.items():
        if name.startswith("references/"):
            parts.append(f"\n\nBundle reference {name}:\n" + file.read_text(encoding="utf-8"))
    rendered = "".join(parts)
    if len(rendered) > max_chars:
        raise ValueError("Skill references exceed disclosure budget; curate the bundle")
    return rendered
