#!/usr/bin/env python3
"""Audit G06 reference class coverage by source; never emit acceptance labels.

The frozen merged reference is an AI proxy. Aggregate counts alone cannot
show whether a reviewer distinguishes compatibility rather than source style.
This offline audit preserves that distinction without changing runtime gates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


LABELS = {"assertion": ("claim_free", "contains_claims"),
          "contract": ("compatible", "incompatible")}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def audit(reference: dict, *, min_per_class: int = 30) -> dict:
    if min_per_class < 1:
        raise ValueError("min_per_class must be positive")
    if reference.get("schema") != "g06-natural-ai-reference-merge-v1":
        raise ValueError("unsupported reference schema")
    if reference.get("annotation_origin") != "ai":
        raise ValueError("reference annotation origin must remain ai")
    body = {k: v for k, v in reference.items() if k != "reference_sha256"}
    if reference.get("reference_sha256") != digest(body):
        raise ValueError("reference hash mismatch")
    rows = reference.get("cases")
    if not isinstance(rows, list) or reference.get("case_count") != len(rows):
        raise ValueError("reference case count mismatch")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate reference case IDs")
    if len({row["candidate_sha256"] for row in rows}) != len(rows):
        raise ValueError("duplicate candidate identities")
    dimensions = {}
    for dim, labels in LABELS.items():
        counts = defaultdict(Counter)
        for row in rows:
            source = row.get("source_kind")
            value = row.get("verdicts", {}).get(dim)
            if not isinstance(source, str) or not source or value not in (*labels, "uncertain"):
                raise ValueError("invalid source or verdict")
            counts[source][value] += 1
        totals = sum(counts.values(), Counter())
        resolved = sum(totals[label] for label in labels)
        # Descriptive in-sample source-only classifier, not held-out accuracy.
        source_majority_correct = sum(max(c[labels[0]], c[labels[1]]) for c in counts.values())
        dimensions[dim] = {
            "aggregate": dict(totals),
            "aggregate_30_per_class": all(totals[label] >= min_per_class for label in labels),
            "by_source": {source: {label: c[label] for label in (*labels, "uncertain")}
                          for source, c in sorted(counts.items())},
            "sources_with_both_classes": [source for source, c in sorted(counts.items())
                                          if all(c[label] > 0 for label in labels)],
            "sources_with_30_per_class": [source for source, c in sorted(counts.items())
                                         if all(c[label] >= min_per_class for label in labels)],
            "resolved_cases": resolved,
            "in_sample_source_only_accuracy": source_majority_correct / resolved if resolved else None,
        }
    result = {
        "schema": "g06-reference-readiness-audit-v1",
        "reference_sha256": reference["reference_sha256"],
        "annotation_origin": "ai", "case_count": len(rows),
        "min_per_class": min_per_class, "dimensions": dimensions,
        "production_gate_eligible": False,
        "target_evaluation_authorized": False,
        "model_calls": 0, "network_calls": 0,
        "limitations": [
            "source-only accuracy is descriptive and in-sample; not reviewer accuracy",
            "aggregate capacity does not establish within-source discrimination or current-candidate validity",
            "this audit does not reinterpret optimizer rejection as a semantic reference label",
        ],
    }
    return {**result, "audit_sha256": digest(result)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.reference.read_text(encoding="utf-8-sig")))
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != text:
            raise ValueError("refusing to replace an existing audit")
    else:
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit_sha256": result["audit_sha256"],
                      "model_calls": 0, "network_calls": 0}))


if __name__ == "__main__":
    main()
