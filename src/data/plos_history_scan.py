"""Exact historical exposure screen for frozen PLOS source candidates."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .healthfc_history_scan import Pattern, _AhoCorasick, _iter_text_files, normalize_text


DOI_RE = re.compile(r"10\.1371/journal\.[a-z0-9]+\.[0-9]{7}", re.I)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[a-z0-9])")
SENSITIVE_NAMES = {".env", ".env.local", ".env.production", "secrets.json", "credentials.json"}


def scan_plos_history(
    records: list[dict[str, Any]],
    history_roots: list[Path],
    *,
    exclude_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Find exact DOI/title/abstract phrase reuse without returning source text.

    This is a conservative lexical screen, not proof of semantic independence.
    Excluded roots are useful for keeping the current capture from matching itself.
    """
    roots = [Path(root).resolve() for root in history_roots]
    excluded = [Path(root).resolve() for root in (exclude_roots or [])]
    if not roots or any(not root.is_dir() for root in roots):
        raise FileNotFoundError("all PLOS history roots must exist")

    pattern_to_dois: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        doi = str(record.get("doi") or record.get("id") or "").strip().casefold()
        if not doi:
            continue
        patterns = [("doi", normalize_text(doi))]
        title = normalize_text(str(record.get("title") or record.get("title_display") or ""))
        abstract = normalize_text(str(record.get("abstract") or ""))
        if len(title) >= 24:
            patterns.append(("title", title))
        if len(abstract) >= 80:
            patterns.append(("abstract", abstract))
            patterns.extend(("abstract_sentence", sentence.strip())
                            for sentence in SENTENCE_BOUNDARY.split(abstract)
                            if len(sentence.strip()) >= 64)
        for kind, text in patterns:
            pattern_to_dois[(kind, text)].add(doi)

    patterns: list[Pattern] = []
    pattern_dois: list[set[str]] = []
    for index, ((kind, text), dois) in enumerate(sorted(pattern_to_dois.items())):
        patterns.append(Pattern(row_index=index, kind=kind, text=text))
        pattern_dois.append(dois)
    if not patterns:
        raise ValueError("PLOS candidate set contains no searchable DOI/title/abstract")

    automaton = _AhoCorasick(patterns)
    matches: dict[str, set[str]] = defaultdict(set)
    seen_paths: set[Path] = set()
    scanned_files = scanned_bytes = unreadable_files = fallback_files = 0
    for root in roots:
        for path in _iter_text_files(root):
            if path.name.casefold() in SENSITIVE_NAMES:
                continue
            resolved = path.resolve()
            if any(resolved == excluded_root or excluded_root in resolved.parents
                   for excluded_root in excluded):
                continue
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
            for pattern_index in automaton.scan(text, patterns):
                pattern = patterns[pattern_index]
                for doi in pattern_dois[pattern_index]:
                    matches[doi].add(pattern.kind)

    return {
        "schema": "plos-history-scan-v1",
        "history_roots": [str(root) for root in roots],
        "excluded_roots": [str(root) for root in excluded],
        "candidate_count": len({str(row.get("doi") or row.get("id") or "").casefold()
                                 for row in records}),
        "pattern_count": len(patterns),
        "files_scanned": scanned_files,
        "bytes_scanned": scanned_bytes,
        "fallback_encoding_files": fallback_files,
        "unreadable_files": unreadable_files,
        "overlap_by_kind": {
            kind: sorted(doi for doi, kinds in matches.items() if kind in kinds)
            for kind in ("doi", "title", "abstract", "abstract_sentence")
        },
        "overlap_candidates": {
            doi: sorted(kinds) for doi, kinds in sorted(matches.items())},
        "complete": unreadable_files == 0,
        "exact_phrase_only_not_paraphrase_proof": True,
        "secret_file_names_excluded": True,
    }
