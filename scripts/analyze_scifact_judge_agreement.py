"""Check whether two independent citation verifiers agree on SciFact gold pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUOTE_VALID = {"proposition_true_with_exact_quote",
               "proposition_false_with_exact_quote",
               "proposition_supports_opposite"}


def decide(first: dict, second: dict) -> dict:
    if (first["claim_id"], first["doc_id"], first["gold"]) != (
            second["claim_id"], second["doc_id"], second["gold"]):
        raise ValueError("calibration pairs differ")
    relation = first.get("relation")
    accepted = (relation in {"true", "false"}
                and relation == second.get("relation")
                and first.get("reason") in QUOTE_VALID
                and second.get("reason") in QUOTE_VALID)
    expected = "true" if first["gold"] == "SUPPORT" else "false"
    return {"claim_id": first["claim_id"], "doc_id": first["doc_id"],
            "gold": first["gold"], "first_relation": relation,
            "second_relation": second.get("relation"),
            "first_reason": first.get("reason"),
            "second_reason": second.get("reason"),
            "accepted": accepted,
            "correct_if_accepted": accepted and relation == expected}


def analyze(first_report: dict, second_report: dict,
            first_model: str | None = None,
            second_model: str | None = None) -> dict:
    if first_report["archive_sha256"] != second_report["archive_sha256"]:
        raise ValueError("dataset archives differ")
    first_rows = [row for row in first_report["rows"]
                  if first_model is None or row["model_id"] == first_model]
    second_rows = [row for row in second_report["rows"]
                   if second_model is None or row["model_id"] == second_model]
    first = {(row["claim_id"], row["doc_id"]): row for row in first_rows}
    second = {(row["claim_id"], row["doc_id"]): row for row in second_rows}
    if (not first or len(first) != len(first_rows)
            or len(second) != len(second_rows)
            or set(first) != set(second)):
        raise ValueError("reports must contain the same unique pairs")
    rows = [decide(first[key], second[key]) for key in sorted(first)]
    accepted = [row for row in rows if row["accepted"]]
    summary = {"n": len(rows), "accepted": len(accepted),
               "accepted_correct": sum(row["correct_if_accepted"]
                                       for row in accepted),
               "accepted_incorrect": sum(not row["correct_if_accepted"]
                                         for row in accepted),
               "abstained": len(rows) - len(accepted)}
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-model", default="")
    parser.add_argument("--second-model", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    result = analyze(first, second, args.first_model or None,
                     args.second_model or None)
    result.update({"status": "public_scifact_dev_verifier_agreement",
                   "first_model": args.first_model or first["rows"][0]["model_id"],
                   "second_model": args.second_model or second["rows"][0]["model_id"],
                   "archive_sha256": first["archive_sha256"],
                   "limit": "Agreement may reduce false accepts by rejecting "
                            "more pairs. SciFact to BioASQ transfer is unproven."})
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
