"""Score declared AI blind references without editing legacy human templates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.ai_grounding_reference import score_references
from src.util import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet-a", "packet-b", "responses-a", "responses-b", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = score_references(read_json(args.packet_a), read_json(args.packet_b), read_json(args.responses_a), read_json(args.responses_b))
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_case", "reviewer_provenance"}}, indent=2))


if __name__ == "__main__":
    main()
