"""Write a zero-call G05 task-authoring plan draft for review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_task_authoring import build_g05_task_authoring_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-plan", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    for role in ("author", "target", "reference-a", "reference-b"):
        parser.add_argument(f"--{role}-provider")
        parser.add_argument(f"--{role}-model-id")
        parser.add_argument(f"--{role}-model-family")
    args = parser.parse_args(argv)

    sample_plan = json.loads(args.sample_plan.read_text(encoding="utf-8"))
    role_values = {}
    for role in ("author", "target", "reference-a", "reference-b"):
        role_values[role] = {
            "provider": getattr(args, f"{role.replace('-', '_')}_provider"),
            "model_id": getattr(args, f"{role.replace('-', '_')}_model_id"),
            "model_family": getattr(args, f"{role.replace('-', '_')}_model_family"),
        }
    supplied = [value is not None for identity in role_values.values()
                for value in identity.values()]
    if any(supplied) and not all(supplied):
        parser.error("bind all provider/model-id/model-family fields for author, target, reference-a, and reference-b")
    plan = build_g05_task_authoring_plan(
        sample_plan, seed=args.seed,
        author_model=role_values["author"] if any(supplied) else None,
        target_model=role_values["target"] if any(supplied) else None,
        reference_models=([role_values["reference-a"], role_values["reference-b"]]
                          if any(supplied) else None),
        # This CLI freezes plans only; it never authorizes provider dispatch.
        allow_dispatch=False,
    )
    output = args.output.resolve()
    workbuddy = (ROOT / ".workbuddy").resolve()
    try:
        output.relative_to(workbuddy)
    except ValueError as exc:
        raise ValueError("G05 authoring plan output must stay under project .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite G05 authoring plan draft: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": plan["status"],
        "output": str(output),
        "plan_sha256": plan["plan_sha256"],
        "request_count": plan["request_count"],
        "route_counts": plan["route_counts"],
        "model_calls": plan["model_calls_during_planning"],
        "calls_allowed": plan["calls_allowed"],
        "dispatch_authorized": plan["dispatch_authorized"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
