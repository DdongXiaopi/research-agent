"""Exact phrase scan for prior exposure of CliniFact claims and abstracts."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .healthfc_history_scan import (
    Pattern, _AhoCorasick, _iter_text_files, normalize_text,
)


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
SPLITS = ("train", "validation")


def _read_patterns(paths: dict[str, Path]) -> tuple[list[Pattern], list[set[tuple[str, int]]]]:
    row_index = 0
    references: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for split in SPLITS:
        with paths[split].open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"claim", "article_title", "article_abstract"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise ValueError(f"CliniFact {split} file lacks history-scan fields")
            for local_index, row in enumerate(reader):
                ref = (split, local_index)
                claim = normalize_text(str(row.get("claim") or ""))
                title = normalize_text(str(row.get("article_title") or ""))
                abstract = normalize_text(str(row.get("article_abstract") or ""))
                for kind, text in (("claim", claim), ("title", title),
                                   ("abstract", abstract)):
                    if len(text) >= 32:
                        references[(kind, text)].add(ref)
                for sentence in SENTENCE_BOUNDARY.split(abstract):
                    sentence = sentence.strip()
                    if len(sentence) >= 48:
                        references[("abstract_sentence", sentence)].add(ref)
                row_index += 1
    patterns = []
    pattern_refs = []
    for (kind, text), refs in sorted(references.items()):
        patterns.append(Pattern(row_index=len(patterns), kind=kind, text=text))
        pattern_refs.append(refs)
    return patterns, pattern_refs


def scan_clinifact_history(
    train_path: Path,
    validation_path: Path,
    history_root: Path | list[Path],
) -> dict[str, Any]:
    """Return exact-match counts and row indexes; never emit source text/hashes."""
    paths = {"train": Path(train_path), "validation": Path(validation_path)}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    history_roots = ([Path(history_root)] if isinstance(history_root, (str, Path))
                     else [Path(p) for p in history_root])
    if not history_roots or any(not root.is_dir() for root in history_roots):
        missing = next((root for root in history_roots if not root.is_dir()), None)
        raise FileNotFoundError(missing or "history roots are empty")
    patterns, pattern_refs = _read_patterns(paths)
    if not patterns:
        raise ValueError("CliniFact source produced no non-empty history patterns")
    automaton = _AhoCorasick(patterns)
    matched: dict[str, set[int]] = {split: set() for split in SPLITS}
    matched_kinds: dict[str, dict[str, set[int]]] = {
        split: defaultdict(set) for split in SPLITS}
    hit_files: set[str] = set()
    scanned_files = scanned_bytes = fallback_files = unreadable_files = 0
    seen_paths: set[Path] = set()
    for root in history_roots:
        for path in _iter_text_files(root):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                raw = path.read_bytes()
            except OSError:
                unreadable_files += 1
                continue
            if b"\x00" in raw:
                continue
            try:
                text = normalize_text(raw.decode("utf-8"))
            except UnicodeDecodeError:
                fallback_files += 1
                text = normalize_text(raw.decode("cp1252", errors="replace"))
            scanned_files += 1
            scanned_bytes += len(raw)
            indexes = automaton.scan(text, patterns)
            if not indexes:
                continue
            hit_files.add(path.as_posix())
            for pattern_index in indexes:
                pattern = patterns[pattern_index]
                for split, row_index in pattern_refs[pattern_index]:
                    matched[split].add(row_index)
                    matched_kinds[split][pattern.kind].add(row_index)
    return {
        "schema": "clinifact-history-scan-v1",
        "source_sha256": {
            split: hashlib.sha256(path.read_bytes()).hexdigest()
            for split, path in paths.items()},
        "history_root": str(history_roots[0]) if len(history_roots) == 1 else [str(p) for p in history_roots],
        "pattern_count": len(patterns),
        "files_scanned": scanned_files,
        "bytes_scanned": scanned_bytes,
        "fallback_encoding_files": fallback_files,
        "unreadable_files": unreadable_files,
        "matched_row_indexes": {
            split: sorted(indexes) for split, indexes in matched.items()},
        "matched_rows_by_kind": {
            split: {kind: len(indexes) for kind, indexes in kinds.items()}
            for split, kinds in matched_kinds.items()},
        "matched_rows_total": sum(len(indexes) for indexes in matched.values()),
        "files_with_matches": len(hit_files),
        "source_text_or_match_hashes_emitted": False,
        "complete": unreadable_files == 0,
        "exact_phrase_only_not_paraphrase_proof": True,
    }
