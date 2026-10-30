"""Structure-only audit of the downloaded UniTor BioASQ source.

Do not print question text, answer values, or per-question IDs: the dev yes/no
slice is reserved as a possible future independent confirmation cohort.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "vendor/BioASQ2025-UNITOR-main/BioASQ2025-UNITOR-main"
          / "data/dataset/answer_generation")
sys.path.insert(0, str(ROOT))


def _read(name: str) -> tuple[list[dict], dict]:
    path = SOURCE / name
    raw = path.read_bytes()
    rows = json.loads(raw.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{name} is not a JSON list")
    yesno = [r for r in rows if r.get("type") == "yesno"]
    label_domain_ok = all(str(r.get("output", "")).strip().lower()
                          in {"yes", "no"} for r in yesno)
    has_marked_context = sum(any("[BS]" in str(r.get(k, ""))
                                 for k in ("input_2", "input_3", "input_4"))
                             for r in yesno)
    info = {
        "file": name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "records": len(rows),
        "type_counts": dict(sorted(collections.Counter(r.get("type") for r in rows).items())),
        "yesno_unique_ids": len({r.get("id") for r in yesno}),
        "yesno_labels_valid": label_domain_ok,
        "yesno_with_marked_context": has_marked_context,
        "schema_keys": sorted(set().union(*(r.keys() for r in rows))),
    }
    return rows, info


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def main() -> int:
    train, train_info = _read("train_set_answGEN.json")
    dev, dev_info = _read("dev_set_answGEN.json")
    train_yes = {r["id"] for r in train if r.get("type") == "yesno"}
    dev_yes = {r["id"] for r in dev if r.get("type") == "yesno"}
    train_questions = {_norm(str(r.get("input_1", ""))) for r in train
                       if r.get("type") == "yesno"}
    dev_questions = {_norm(str(r.get("input_1", ""))) for r in dev
                     if r.get("type") == "yesno"}
    report = {
        "source_root": str(SOURCE),
        "train": train_info,
        "dev": dev_info,
        "train_dev_yesno_id_overlap": len(train_yes & dev_yes),
        "train_dev_yesno_question_overlap": len(train_questions & dev_questions),
        "confirmation_source_status": "unselected_unscored_structure_audit_only",
        "notes": [
            "Dev answers and question text were not printed or used for candidate selection.",
            "Contexts are truncated and mark gold evidence; this is a new protocol.",
            "Code MIT license does not establish dataset redistribution terms.",
        ],
    }
    dest = ROOT / "artifacts/probe/bioasq_source_audit_2026-09-29.json"
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
