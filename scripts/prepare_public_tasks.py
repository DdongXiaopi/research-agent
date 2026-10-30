"""Prepare licensed supplied-evidence JSONL tasks for the public protocol."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.public_tasks import prepare
from src.data.freeze import FREEZE_FILE
from src.util import read_json, read_jsonl, sha256_file
from scripts.make_splits import _commit_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--salt", default="public-task-input-v1")
    args = parser.parse_args(argv)
    if FREEZE_FILE.exists() and args.dataset in (read_json(FREEZE_FILE).get("datasets") or {}):
        parser.error("cannot rewrite a frozen public dataset")
    if not args.input.is_file():
        parser.error("public JSONL input does not exist")
    manifest, visible, sealed = prepare(args.dataset, read_jsonl(args.input), source=args.source,
                                        license=args.license, salt=args.salt)
    manifest["raw_input_sha256"] = sha256_file(args.input)
    _commit_prepared([(args.dataset, manifest, visible, sealed)])
    print(f"prepared {args.dataset}: {len(visible)} visible, {len(sealed)} sealed; run rsi freeze next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
