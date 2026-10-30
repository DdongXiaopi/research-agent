"""Restore optional optimizer sources from immutable locks; verify before install.

Does not replace existing vendor checkouts or download research datasets.
Pi's executing npm runtime is restored separately by npm ci from package-lock.
"""
from __future__ import annotations
import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml
from src.config import _source_tree_sha256
from src.util import sha256_file


def verify(entry, directory):
    if _source_tree_sha256(directory / entry["source_root"]) != entry["source_sha256"]:
        raise ValueError("locked package source differs: " + entry["name"])
    if sha256_file(directory / entry["license_source"]) != entry["license_sha256"]:
        raise ValueError("locked source license differs: " + entry["name"])


def unpack_archive(content: bytes, destination: Path, *, strip_root=False):
    with zipfile.ZipFile(BytesIO(content)) as archive:
        if sum(info.file_size for info in archive.infolist()) > 200 * 1024 * 1024:
            raise ValueError("upstream archive exceeds size ceiling")
        roots = set()
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if not name.parts or name.is_absolute() or ".." in name.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("unsafe upstream archive path")
            roots.add(name.parts[0])
        if len(roots) != 1:
            raise ValueError("upstream archive must contain one source root")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("unsafe upstream archive path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("upstream archive contains a symbolic link")
            relative = name.parts[1:] if strip_root else name.parts
            if not relative:
                if info.is_dir():
                    continue
                raise ValueError("archive root must be a directory")
            target = destination.joinpath(*relative)
            target.resolve().relative_to(destination.resolve())
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        return destination if strip_root else destination / roots.pop()


def restore(entry):
    import requests
    destination = ROOT / entry["path"]
    destination.resolve().relative_to((ROOT / "vendor").resolve())
    if destination.exists():
        verify(entry, destination)
        return "existing_verified"
    revision, repository = entry.get("upstream_revision", ""), entry.get("upstream_repository", "")
    if (not re.fullmatch(r"[a-f0-9]{40}", revision) or
            not re.fullmatch(r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)):
        raise ValueError("restore requires an immutable GitHub source revision")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(repository + "/archive/" + revision + ".zip", timeout=60, stream=True) as response:
        response.raise_for_status()
        pieces, total = [], 0
        for piece in response.iter_content(1024 * 1024):
            total += len(piece)
            if total > 100 * 1024 * 1024:
                raise ValueError("upstream download exceeds size ceiling")
            pieces.append(piece)
    with tempfile.TemporaryDirectory(prefix="rsi-upstream-", dir=destination.parent) as temp:
        # Strip the redundant commit-named root so nested Windows checkouts
        # don't exceed legacy path limits. Package-relative bytes stay pinned.
        source = unpack_archive(b"".join(pieces), Path(temp), strip_root=True)
        for patch in entry.get("patches") or []:
            patch_file = ROOT / patch["path"]
            patch_file.resolve().relative_to((ROOT / "patches").resolve())
            if sha256_file(patch_file) != patch["sha256"]:
                raise ValueError("upstream compatibility patch differs from lock")
            source.joinpath(patch["target"]).resolve().relative_to(source.resolve())
            # The lock pins LF bytes. User/global autocrlf settings must not
            # rewrite an applied compatibility patch into another source tree.
            subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(patch_file)], cwd=source, check=True,
                           capture_output=True, text=True)
            if sha256_file(source / patch["target"]) != patch["patched_sha256"]:
                raise ValueError("patched upstream target differs from lock")
        verify(entry, source)
        # Both paths stay inside vendor; existing trees are never overwritten.
        if destination.exists():
            raise ValueError("vendor destination appeared during restore")
        source.rename(destination)
    return "restored_verified"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-vendor", action="store_true")
    parser.add_argument("--install-optimizers", action="store_true")
    parser.add_argument("--install-pi", action="store_true")
    args = parser.parse_args(argv)
    if not (args.restore_vendor or args.install_optimizers or args.install_pi):
        parser.print_help()
        return 2
    lock = yaml.safe_load((ROOT / "UPSTREAM_LOCK.yml").read_text(encoding="utf-8"))
    if args.restore_vendor or args.install_optimizers:
        for entry in lock["upstreams"]:
            if entry["name"] not in {"dspy", "gepa"}:
                continue
            if args.restore_vendor:
                status = restore(entry)
            else:
                if not (ROOT / entry["path"]).exists():
                    parser.error("optimizer sources are absent; add --restore-vendor before --install-optimizers")
                verify(entry, ROOT / entry["path"])
                status = "existing_verified"
            print(entry["name"] + ": " + status)
    if args.install_optimizers:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-mipro.txt"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "vendor/gepa-main/gepa-main"], cwd=ROOT, check=True)
    if args.install_pi:
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("npm is required for the locked Pi runtime")
        subprocess.run([npm, "ci"], cwd=ROOT / "tools/pi-bridge", check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
