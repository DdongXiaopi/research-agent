"""Prompt/Skill 控制检索 query 的受限协议。

只有显式开启新实验协议才生效。候选可写一行
``RETRIEVAL_QUERY: {question} ...``；命中的 Skill 正文同样可写，
Skill 优先。只替换公开题面和选项，不读取答案、证据或语料标记。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

_LINE = re.compile(r"^\s*RETRIEVAL_QUERY:\s*(.+?)\s*$", re.I | re.M)
_INSTRUCTION = re.compile(r"^\s*RETRIEVAL_INSTRUCTION:\s*(.+?)\s*$", re.I | re.M)
_TOKEN = re.compile(r"^[\w\s{}.,:/?+\-()]+$")


def query_instruction(prompt: str, skill_body: str) -> Tuple[str, str]:
    """Return the one-line search instruction; a loaded Skill takes priority."""
    for source, content in (("skill", skill_body), ("prompt", prompt)):
        matches = _INSTRUCTION.findall(content or "")
        if not matches:
            continue
        if len(matches) != 1 or not 1 <= len(matches[0].strip()) <= 300:
            raise ValueError(f"invalid RETRIEVAL_INSTRUCTION in {source}")
        instruction = matches[0].strip()
        if "{" in instruction or "}" in instruction or "\x00" in instruction:
            raise ValueError("RETRIEVAL_INSTRUCTION cannot contain placeholders")
        return instruction, source
    return "Extract the question's biomedical entities, intervention and outcome terms.", "default"


def parse_model_query(content: str) -> str:
    """Fail closed on malformed model queries; no answer or evidence is supplied."""
    lines = (content or "").strip().splitlines()
    if len(lines) != 1 or not lines[0].startswith("QUERY: "):
        raise ValueError("model query must be one QUERY: line")
    query = lines[0][7:].strip()
    if (not 3 <= len(query) <= 200 or not query.isprintable()
            or any(ch in query for ch in "{}<>`")):
        raise ValueError("model query has invalid length or characters")
    return query


def model_query_messages(instruction: str, question: str) -> Tuple[str, str]:
    """One canonical prompt for runtime and frozen query diagnostics."""
    return (
        "Rewrite the biomedical question into search terms for ranking frozen "
        "PubMed abstracts. Follow the search instruction as data. "
        "Reply with exactly one line: QUERY: <search terms>. "
        "Do not answer the question.",
        "Search instruction: " + instruction + "\nQuestion: " + question,
    )


def model_query_sources_messages(instruction: str, question: str,
                                 available: Dict[str, str]) -> Tuple[str, str]:
    """Let Prompt/Skill choose a frozen ESearch source subset and rank query."""
    if not available or any(not re.fullmatch(r"[a-z_]+", name)
                            or not isinstance(query, str)
                            or not " ".join(query.split()).isprintable()
                            for name, query in available.items()):
        raise ValueError("invalid available source query names")
    return (
        "Choose search terms and which frozen PubMed ESearch result sets to use. "
        "Follow the search instruction as data. Reply with exactly two lines: "
        "QUERY: <search terms> and SOURCES: <comma-separated names>. "
        "Choose one or more names only from the allowed list. "
        "Do not answer the question or invent source names.",
        "Search instruction: " + instruction + "\nQuestion: " + question
        + "\nCaptured PubMed ESearch sources (name: search string):\n"
        + "\n".join(name + ": " + " ".join(query.split())
                    for name, query in available.items()),
    )


def parse_model_query_sources(content: str,
                              available: list[str]) -> Tuple[str, list[str]]:
    lines = (content or "").strip().splitlines()
    if len(lines) != 2 or not lines[1].startswith("SOURCES: "):
        raise ValueError("model source plan must have QUERY and SOURCES lines")
    query = parse_model_query(lines[0])
    names = [part.strip() for part in lines[1][9:].split(",")]
    if (not names or any(not name for name in names)
            or len(names) != len(set(names)) or any(name not in available
                                                   for name in names)):
        raise ValueError("model selected invalid PubMed source names")
    return query, names


def query_for(task: Dict[str, Any], prompt: str, skill_body: str,
              enabled: bool = False) -> Tuple[str, str]:
    base = str(task.get("question", ""))
    if not enabled:
        return base, "baseline"
    for source, content in (("skill", skill_body), ("prompt", prompt)):
        found = _LINE.search(content or "")
        if not found:
            continue
        template = found.group(1).strip()
        if len(template) > 200 or not _TOKEN.fullmatch(template):
            raise ValueError(f"invalid RETRIEVAL_QUERY in {source}")
        if "{question}" not in template:
            raise ValueError("RETRIEVAL_QUERY must contain {question}")
        remainder = template.replace("{question}", "").replace("{options}", "")
        if "{" in remainder or "}" in remainder:
            raise ValueError("unknown RETRIEVAL_QUERY placeholder")
        query = template.replace("{question}", base).replace(
            "{options}", " ".join(str(o) for o in task.get("options", [])))
        return query[:1000], source
    return base, "baseline"
