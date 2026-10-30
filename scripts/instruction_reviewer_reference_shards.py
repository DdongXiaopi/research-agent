#!/usr/bin/env python3
"""Plan, execute, and aggregate isolated G06 reference shards."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_pair_gate as pair_gate  # noqa: E402
from src.eval import instruction_reviewer_reference_shards as shards  # noqa: E402
from src.eval.instruction_reviewer_cohort_reference import (  # noqa: E402
    validate_reference_packet, validate_reference_packet_integrity,
)
from src.eval.instruction_reviewer_cohort_v5 import SCHEMA as V5_SCHEMA  # noqa: E402
from src.eval.instruction_reviewer_cohort_v6 import SCHEMA as V6_SCHEMA  # noqa: E402
from src.eval.instruction_reviewer_cohort_v7 import SCHEMA as V7_SCHEMA  # noqa: E402
from src.eval.instruction_reviewer_cohort_v8 import SCHEMA as V8_SCHEMA  # noqa: E402
from src.eval.instruction_reviewer_cohort_v9 import SCHEMA as V9_SCHEMA  # noqa: E402
from src.eval.instruction_reviewer_cohort_v10 import SCHEMA as V10_SCHEMA  # noqa: E402
from src.eval.instruction_reviewer_cohort_v11 import (  # noqa: E402
    SCHEMA as V11_SCHEMA, validate_packet as validate_v11_packet,
    validate_packet_integrity as validate_v11_packet_integrity,
)
from src.eval.instruction_reviewer_cohort_v12 import (  # noqa: E402
    SCHEMA as V12_SCHEMA, validate_packet as validate_v12_packet,
    validate_packet_integrity as validate_v12_packet_integrity,
)
from src.eval.instruction_reviewer_cohort_v13 import (  # noqa: E402
    SCHEMA as V13_SCHEMA, validate_packet as validate_v13_packet,
    validate_packet_integrity as validate_v13_packet_integrity,
)
from src.eval.instruction_reviewer_cohort_v14 import (  # noqa: E402
    SCHEMA as V14_SCHEMA, validate_packet as validate_v14_packet,
    validate_packet_integrity as validate_v14_packet_integrity,
)
from src.eval.instruction_reviewer_cohort_v15 import (  # noqa: E402
    SCHEMA as V15_SCHEMA, validate_packet as validate_v15_packet,
    validate_packet_integrity as validate_v15_packet_integrity,
)
from src.eval.instruction_reviewer_cohort_v16 import (  # noqa: E402
    SCHEMA as V16_SCHEMA, validate_packet as validate_v16_packet,
    validate_packet_integrity as validate_v16_packet_integrity,
)
from src.eval.instruction_reviewer_rubric_v8 import RUBRIC_VERSION as V11_REFERENCE_RUBRIC_VERSION  # noqa: E402
V12_REFERENCE_RUBRIC_VERSION = V11_REFERENCE_RUBRIC_VERSION
V13_REFERENCE_RUBRIC_VERSION = V11_REFERENCE_RUBRIC_VERSION
V14_REFERENCE_RUBRIC_VERSION = "instruction-reviewer-reference-v6"
V15_REFERENCE_RUBRIC_VERSION = "v7"
V16_REFERENCE_RUBRIC_VERSION = "v7"
SUPPORTED_PACKET_HELP = (
    "the shard CLI accepts frozen v5/rubric-v3, v6/rubric-v4, "
    "v7/rubric-v5, v8/rubric-v6, v9/rubric-v7, v10/rubric-v7, "
    "v11-v13/rubric-v8, v14/rubric-v6, or v15-v16/rubric-v7 packets")


def _default_artifact_paths(work_dir: Path, slot: str, schema: str) -> tuple[Path, Path, Path]:
    """Return stable v14+slot-specific defaults without changing legacy paths."""
    if schema == V14_SCHEMA:
        prefix = f"v14-{slot}"
        return (work_dir / f"{prefix}-schedule.json",
                work_dir / f"{prefix}-state",
                work_dir / f"{prefix}-aggregate.json")
    if schema == V15_SCHEMA:
        prefix = f"v15-{slot}"
        return (work_dir / f"{prefix}-schedule.json",
                work_dir / f"{prefix}-state",
                work_dir / f"{prefix}-aggregate.json")
    if schema == V16_SCHEMA:
        prefix = f"v16-{slot}"
        return (work_dir / f"{prefix}-schedule.json",
                work_dir / f"{prefix}-state",
                work_dir / f"{prefix}-aggregate.json")
    return work_dir / "schedule.json", work_dir / "state", work_dir / "aggregate.json"
from src.util import artifacts_root, read_json  # noqa: E402


DEFAULT_MODEL = "THUDM/GLM-4-9B-0414"
# This v11 packet was built and exposure-audited before v12 was added. Its
# frozen audit is immutable; later cohorts must not invalidate its plan-only
# path merely by appearing in the workspace. Integrity validation still
# verifies all packet/source/content hashes and its embedded v1-v10 audit.
LEGACY_V11_PREFLIGHT_PACKET_SHA256 = "ca0e96bf50b51270ff7fafd3b173ad4b73d242746c2db20ef10676a76361b969"
LEGACY_V11_BUILDER_SOURCE_SHA256 = "82087658efb50c17e2fea8203ff5903a8033baf68973ec4b8827cb6ec975e9ea"


def _required_rubric(schema: str | None) -> str | None:
    return {V5_SCHEMA: "v3", V6_SCHEMA: "v4", V7_SCHEMA: "v5",
            V8_SCHEMA: "v6", V9_SCHEMA: "v7", V10_SCHEMA: "v7",
            V11_SCHEMA: "v8", V12_SCHEMA: "v8", V13_SCHEMA: "v8",
            V14_SCHEMA: "v6", V15_SCHEMA: "v7", V16_SCHEMA: "v7"}.get(schema)


def _validate_packet(packet: dict, *, integrity: bool = False) -> None:
    if packet.get("schema") == V16_SCHEMA:
        (validate_v16_packet_integrity if integrity else validate_v16_packet)(packet, ROOT)
        return
    if packet.get("schema") == V15_SCHEMA:
        (validate_v15_packet_integrity if integrity else validate_v15_packet)(packet, ROOT)
        return
    if packet.get("schema") == V14_SCHEMA:
        (validate_v14_packet_integrity if integrity else validate_v14_packet)(packet, ROOT)
        return
    if packet.get("schema") == V13_SCHEMA:
        (validate_v13_packet_integrity if integrity else validate_v13_packet)(packet, ROOT)
        return
    if packet.get("schema") == V12_SCHEMA:
        (validate_v12_packet_integrity if integrity else validate_v12_packet)(packet, ROOT)
        return
    if packet.get("schema") == V11_SCHEMA:
        frozen_v11_snapshot = (packet.get("packet_sha256") == LEGACY_V11_PREFLIGHT_PACKET_SHA256
            and packet.get("builder_source_sha256") == LEGACY_V11_BUILDER_SOURCE_SHA256)
        validator = (validate_v11_packet_integrity
                     if integrity or frozen_v11_snapshot else validate_v11_packet)
        validator(packet, ROOT)
        return
    (validate_reference_packet_integrity if integrity else validate_reference_packet)(packet)


def _validate_packet_rubric_binding(packet: dict) -> None:
    """Check route family and the full packet rubric identity independently."""
    schema = packet.get("schema") if isinstance(packet, dict) else None
    required_route = _required_rubric(schema)
    if required_route is None:
        raise ValueError("unsupported instruction-reviewer shard packet schema")
    actual_route = shards._rubric_version(packet)
    if actual_route != required_route:
        raise ValueError(f"{schema} shard packet does not route to rubric {required_route}")
    expected_reference_rubric = (V11_REFERENCE_RUBRIC_VERSION
        if schema in {V11_SCHEMA, V12_SCHEMA, V13_SCHEMA} else
        V16_REFERENCE_RUBRIC_VERSION if schema == V16_SCHEMA else
        V15_REFERENCE_RUBRIC_VERSION if schema == V15_SCHEMA else required_route)
    protocol = packet.get("reference_protocol")
    if (not isinstance(protocol, dict)
            or protocol.get("rubric_version") != expected_reference_rubric):
        label = ("reference rubric" if schema in {
                    V11_SCHEMA, V12_SCHEMA, V13_SCHEMA, V15_SCHEMA, V16_SCHEMA}
                 else "rubric")
        raise ValueError(f"{schema} shard packet must bind {label} {expected_reference_rubric}")


def _isolated() -> None:
    if (not os.environ.get("RSI_HOME")
            or artifacts_root().resolve() == (ROOT / "artifacts").resolve()):
        raise ValueError("G06 shard execution requires an isolated RSI_HOME")


def _controlled(path: Path) -> Path:
    return shards._controlled_work_path(path)


def _read_schedule(path: Path, expected_sha256: str | None) -> dict:
    schedule_path = _controlled(path)
    if not schedule_path.is_file() or schedule_path.is_symlink():
        raise ValueError("frozen shard schedule is missing or unsafe")
    schedule = read_json(schedule_path)
    if (not isinstance(schedule, dict)
            or not isinstance(expected_sha256, str)
            or schedule.get("schedule_sha256") != expected_sha256):
        raise ValueError("operation requires the exact frozen schedule SHA-256")
    return schedule


def _aggregate_rsi_home(value: str | None, state_dir: Path,
                        schedule: dict) -> Path:
    """Resolve the isolated artifact root and prove it owns recorded AI runs.

    Aggregation calls ``ai_jobs.verify_completed``, which resolves run paths
    relative to the *current* ``RSI_HOME``. Never let an absent or unrelated
    environment silently fall back to the repository's default artifacts.
    Each reviewer may have used a different root, so this is resolved per
    slot/schedule rather than shared globally.
    """
    selected = value or os.environ.get("RSI_HOME")
    if not selected:
        raise ValueError(
            "aggregate requires the isolated RSI_HOME used for this reviewer; "
            "pass --rsi-home (reviewer A and B may require different roots)")
    home = Path(selected).expanduser().resolve()
    if home == (ROOT / "artifacts").resolve():
        raise ValueError("aggregate refuses the project-default artifacts root; provide isolated --rsi-home")

    annotation_root = (home / "evaluation_only" / "ai_annotations").resolve()
    cases_dir = state_dir / "cases"
    if cases_dir.exists():
        if cases_dir.is_symlink() or not cases_dir.is_dir():
            raise ValueError("aggregate shard case directory is unsafe")
        for shard in schedule.get("shards", []):
            record_path = cases_dir / f"case-{shard['index']:03d}.json"
            if not record_path.exists():
                continue  # The aggregator will reject incomplete cohorts.
            if record_path.is_symlink() or not record_path.is_file():
                raise ValueError("aggregate shard record is unsafe")
            record = read_json(record_path)
            run_dir = record.get("run_dir") if isinstance(record, dict) else None
            if not isinstance(run_dir, str) or not run_dir:
                raise ValueError("aggregate shard record lacks its isolated run path")
            try:
                Path(run_dir).resolve().relative_to(annotation_root)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "aggregate --rsi-home does not contain this reviewer's recorded AI runs; "
                    "provide the RSI_HOME used to execute this slot") from exc
    return home


@contextmanager
def _using_rsi_home(home: Path):
    previous = os.environ.get("RSI_HOME")
    os.environ["RSI_HOME"] = str(home)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RSI_HOME", None)
        else:
            os.environ["RSI_HOME"] = previous


def _config(model_id: str) -> dict:
    cfg, _ = pair_gate.load_g06_config(
        model_id, ROOT, pair_gate.DEFAULT_CONFIG_PATH)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute", "aggregate"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="controlled directory below .workbuddy")
    parser.add_argument("--slot", choices=("reviewer_a", "reviewer_b"), required=True)
    parser.add_argument("--model-id", choices=tuple(sorted(pair_gate.FAMILY_REGISTRY)),
                        default=DEFAULT_MODEL)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--expected-schedule-sha256")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--aggregate-path", type=Path)
    parser.add_argument("--rsi-home", type=Path,
                        help="isolated artifacts root used when this reviewer slot was executed")
    args = parser.parse_args(argv)

    packet_path = args.packet.resolve()
    packet = read_json(packet_path)
    schema = packet.get("schema") if isinstance(packet, dict) else None
    expected_rubric = _required_rubric(schema)
    if expected_rubric is None:
        raise ValueError(SUPPORTED_PACKET_HELP)
    _validate_packet_rubric_binding(packet)
    if args.action == "plan":
        # Full source and historical exposure validation is required before a
        # schedule can be frozen. Runner actions use the bound integrity path.
        _validate_packet(packet)
    else:
        _validate_packet(packet, integrity=True)

    work_dir = _controlled(args.work_dir)
    default_schedule, default_state, default_aggregate = _default_artifact_paths(
        work_dir, args.slot, schema)
    schedule_path = _controlled(args.schedule_path or default_schedule)
    state_dir = _controlled(args.state_dir or default_state)
    aggregate_path = _controlled(args.aggregate_path or default_aggregate)

    if args.action == "plan":
        cfg = _config(args.model_id)
        schedule = shards.build_reference_shard_schedule(
            packet, args.slot, cfg, ROOT)
        digest = shards._write_once(schedule_path, schedule)
        print(json.dumps({"schedule_path": str(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "schedule_file_sha256": digest, "packet_sha256": packet["packet_sha256"],
            "slot": args.slot, "model_id": schedule["model_id"],
            "planned_calls": 0, "paid_fallback": False}, ensure_ascii=False))
        return 0

    schedule = _read_schedule(schedule_path, args.expected_schedule_sha256)
    shards.validate_shard_schedule(packet, schedule, args.slot)
    if args.action == "execute":
        _isolated()
        cfg = _config(schedule["model_id"])
        if args.model_id != DEFAULT_MODEL and args.model_id != schedule["model_id"]:
            raise ValueError("requested model differs from the frozen schedule")
        result = shards.run_reference_shards(
            packet, args.slot, cfg, schedule, state_dir)
        print(json.dumps({"state_dir": str(state_dir),
            "schedule_sha256": schedule["schedule_sha256"],
            "slot": args.slot, "model_id": schedule["model_id"],
            "status": result["status"],
            "completed_cases": result["completed_cases"],
            "unknown_cases": result["unknown_cases"],
            "logical_calls": result["logical_calls"],
            "physical_calls": result["physical_calls"],
            "paid_fallback": False}, ensure_ascii=False))
        return 0

    aggregate_home = _aggregate_rsi_home(
        str(args.rsi_home) if args.rsi_home else None, state_dir, schedule)
    with _using_rsi_home(aggregate_home):
        aggregate = shards.aggregate_verified_shards(packet, schedule, state_dir)
    file_sha256 = shards.write_aggregate_once(aggregate_path, aggregate)
    print(json.dumps({"aggregate_path": str(aggregate_path),
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "aggregate_file_sha256": file_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "slot": args.slot, "model_id": schedule["model_id"],
        "case_count": aggregate["case_count"],
        "status": aggregate["status"], "paid_fallback": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
