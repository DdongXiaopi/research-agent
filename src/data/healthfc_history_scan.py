"""Exact phrase scan for prior exposure of HealthFC claims and fixed evidence."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / "artifacts/probe/healthfc_history_scan_20261008.json"

SKIP_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "node_modules", "vendor"}
SKIP_SUFFIXES = {
    ".7z", ".avi", ".bmp", ".doc", ".docx", ".exe", ".gif", ".gz", ".ico",
    ".jpeg", ".jpg", ".mp3", ".mp4", ".pdf", ".png", ".ppt", ".pptx", ".pyc",
    ".bin", ".pkl", ".so", ".sqlite", ".tar", ".tmp", ".webp", ".xls",
    ".xlsx", ".zip", ".parquet", ".arrow", ".npy", ".npz",
}
EVIDENCE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[a-z0-9])")


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True)
class Pattern:
    row_index: int
    kind: str
    text: str


def _source_patterns(csv_path: Path) -> list[Pattern]:
    patterns: dict[tuple[str, str], set[int]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"en_claim", "en_top_sentences"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("HealthFC source lacks English claim/evidence columns")
        for row_index, row in enumerate(reader):
            claim = normalize_text(str(row.get("en_claim") or ""))
            evidence = normalize_text(str(row.get("en_top_sentences") or ""))
            if claim:
                patterns.setdefault(("claim", claim), set()).add(row_index)
            if evidence:
                patterns.setdefault(("evidence_document", evidence), set()).add(row_index)
                # Search individual top sentences too, because previous traces may
                # have serialized only one evidence sentence from a multi-sentence field.
                for sentence in EVIDENCE_SENTENCE_BOUNDARY.split(evidence):
                    sentence = sentence.strip()
                    if len(sentence) >= 40:
                        patterns.setdefault(("evidence_sentence", sentence), set()).add(row_index)
    return [
        Pattern(row_index, kind, text)
        for (kind, text), row_indexes in patterns.items()
        for row_index in row_indexes
    ]


class _AhoCorasick:
    def __init__(self, patterns: list[Pattern]) -> None:
        self.next: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.output: list[list[int]] = [[]]
        for pattern_index, pattern in enumerate(patterns):
            state = 0
            for char in pattern.text:
                child = self.next[state].get(char)
                if child is None:
                    child = self._new_state()
                    self.next[state][char] = child
                state = child
            self.output[state].append(pattern_index)
        queue: deque[int] = deque()
        for state in self.next[0].values():
            queue.append(state)
        while queue:
            state = queue.popleft()
            for char, child in self.next[state].items():
                queue.append(child)
                fallback = self.fail[state]
                while fallback and char not in self.next[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.next[fallback].get(char, 0)
                self.output[child].extend(self.output[self.fail[child]])

    def _new_state(self) -> int:
        state = len(self.next)
        self.next.append({})
        self.fail.append(0)
        self.output.append([])
        return state

    @staticmethod
    def _word_char(char: str) -> bool:
        return char == "_" or char.isalnum()

    def scan(self, text: str, patterns: list[Pattern]) -> set[int]:
        hits: set[int] = set()
        state = 0
        for index, char in enumerate(text):
            while state and char not in self.next[state]:
                state = self.fail[state]
            state = self.next[state].get(char, 0)
            for pattern_index in self.output[state]:
                pattern = patterns[pattern_index].text
                start = index - len(pattern) + 1
                end = index + 1
                if (start > 0 and self._word_char(pattern[0])
                        and self._word_char(text[start - 1])):
                    continue
                if (end < len(text) and self._word_char(pattern[-1])
                        and self._word_char(text[end])):
                    continue
                hits.add(pattern_index)
        return hits


def _iter_text_files(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        for name in files:
            path = Path(current) / name
            if path.suffix.casefold() not in SKIP_SUFFIXES:
                yield path


def scan_healthfc_history(csv_path: Path, history_root: Path) -> dict[str, object]:
    """Scan text files for source claims/evidence; return counts, never phrases."""
    csv_path = Path(csv_path)
    history_root = Path(history_root)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not history_root.is_dir():
        raise FileNotFoundError(history_root)
    patterns = _source_patterns(csv_path)
    if not patterns:
        raise ValueError("HealthFC source produced no non-empty search patterns")
    automaton = _AhoCorasick(patterns)
    hit_claim_rows: set[int] = set()
    hit_evidence_rows: set[int] = set()
    hit_paths: set[str] = set()
    files_scanned = 0
    bytes_scanned = 0
    ignored_binary_files = 0
    fallback_encoding_files = 0
    unreadable_files = 0
    for path in _iter_text_files(history_root):
        try:
            raw = path.read_bytes()
        except OSError:
            unreadable_files += 1
            continue
        if b"\x00" in raw:
            ignored_binary_files += 1
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            fallback_encoding_files += 1
            try:
                decoded = raw.decode("cp1252")
            except UnicodeDecodeError:
                decoded = raw.decode("latin-1")
        text = normalize_text(decoded)
        files_scanned += 1
        bytes_scanned += len(raw)
        pattern_hits = automaton.scan(text, patterns)
        if pattern_hits:
            try:
                hit_paths.add(path.resolve().relative_to(history_root.resolve()).as_posix())
            except ValueError:
                hit_paths.add(hashlib.sha256(str(path).encode("utf-8")).hexdigest())
            for pattern_index in pattern_hits:
                pattern = patterns[pattern_index]
                if pattern.kind == "claim":
                    hit_claim_rows.add(pattern.row_index)
                else:
                    hit_evidence_rows.add(pattern.row_index)
    return {
        "history_root": str(history_root),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "pattern_count": len(patterns),
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "ignored_binary_files": ignored_binary_files,
        "fallback_encoding_files": fallback_encoding_files,
        "unreadable_files": unreadable_files,
        "claim_rows_with_exact_prior_phrase": len(hit_claim_rows),
        "evidence_rows_with_exact_prior_phrase": len(hit_evidence_rows),
        "rows_with_any_exact_prior_phrase": len(hit_claim_rows | hit_evidence_rows),
        "matched_row_indexes": sorted(hit_claim_rows | hit_evidence_rows),
        "files_with_any_exact_prior_phrase": len(hit_paths),
        "matched_file_paths": sorted(hit_paths),
        "claim_or_evidence_text_emitted": False,
        "complete": unreadable_files == 0,
    }
