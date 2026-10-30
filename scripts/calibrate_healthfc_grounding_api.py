"""Run a frozen, configuration-driven HealthFC grounding calibration.

The protocol is model/provider agnostic: the endpoint, model, credential
environment variable, request limits, and optional caller-frozen tariff come
from a JSON config. The config is copied into the plan (without secrets) and
execution refuses any drift. Unknown prices are rejected unless the config
provides explicit rates and provenance, so arbitrary OpenAI-compatible models
remain usable without silently losing cost accounting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.healthfc_cohort import _components
from src.data.healthfc_source_audit import LABELS
from src.eval.grounding_calibration import THRESHOLDS, summarize
from src.eval.grounding_v11 import evidence_selection_messages_v11
from src.eval.grounding_v16 import (claim_relation_messages_v16,
                                    claim_relation_response_format_v16,
                                    prediction_v16)
from src.runtime.call_ledger import call_component, summarize_ledger
from src.runtime.provider import (HttpProvider, published_price_for_endpoint,
                                  published_tariff_metadata)
from src.util import sha256_file, sha256_obj, write_json


SOURCE = ROOT / "data/raw/healthfc/Datensatz.csv"
PRIOR_COHORT = ROOT / "artifacts/probe/healthfc_mipro_exploratory_cohort_20261008.json"
LABELS_ORDERED = ("SUPPORTED", "NEI", "REFUTED")
LABEL_TO_RELATION = {"SUPPORTED": "supported", "NEI": "insufficient", "REFUTED": "contradicted"}
MAX_DOC_CHARS = 6000
DEFAULT_CONFIG = ROOT / "configs/healthfc_calibration.deepseek-flash.json"


@dataclass(frozen=True)
class CalibrationConfig:
    model_id: str
    base_url: str
    api_key_env: str
    prompt_version: str
    seed: str
    max_http_attempts: int = 3
    cooldown_seconds: float = 1.2
    max_tokens: int = 500
    timeout_seconds: float = 180.0
    temperature: float = 0.0
    enable_thinking: bool = False
    sample_per_class: int = 26
    max_cost_usd: float = 0.50
    pricing: dict[str, Any] | None = None
    # Provider capabilities are part of the frozen protocol, rather than
    # being inferred from a vendor/model name.  An empty body is valid for
    # endpoints that do not implement JSON mode; the prompt and local parser
    # still enforce the JSON contract fail-closed.
    context_tokens: int = 40960
    extra_body: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        pricing = dict(self.pricing or {})
        return {
            "model_id": self.model_id, "base_url": self.base_url,
            "api_key_env": self.api_key_env, "prompt_version": self.prompt_version,
            "seed": self.seed, "max_http_attempts": self.max_http_attempts,
            "cooldown_seconds": self.cooldown_seconds, "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds, "temperature": self.temperature,
            "enable_thinking": self.enable_thinking, "sample_per_class": self.sample_per_class,
            "max_cost_usd": self.max_cost_usd, "pricing": pricing,
            "context_tokens": self.context_tokens,
            "extra_body": dict(self.extra_body),
        }


def load_config(path: Path) -> CalibrationConfig:
    raw = _load_json(path)
    required = ("model_id", "base_url", "api_key_env", "prompt_version", "seed")
    missing = [key for key in required if not str(raw.get(key) or "").strip()]
    if missing:
        raise ValueError("calibration config missing required fields: " + ", ".join(missing))
    model_id, base_url, api_key_env = (str(raw[key]).strip() for key in ("model_id", "base_url", "api_key_env"))
    parsed = urlparse(base_url)
    if not parsed.netloc or parsed.scheme not in {"https", "http"}:
        raise ValueError("base_url must be an absolute HTTP(S) OpenAI-compatible endpoint")
    if parsed.scheme == "http" and (parsed.hostname or "").lower() not in {
            "localhost", "127.0.0.1", "::1"}:
        raise ValueError("plain HTTP is allowed only for a local model endpoint")
    if not api_key_env.replace("_", "").isalnum() or api_key_env[0].isdigit():
        raise ValueError("api_key_env must be a shell-safe environment variable name")
    pricing_raw = raw.get("pricing")
    pricing = dict(pricing_raw) if isinstance(pricing_raw, dict) else None
    if pricing is None:
        from src.runtime.provider import published_price_for_endpoint
        if published_price_for_endpoint(model_id, base_url) is None:
            raise ValueError("unknown model tariff; provide explicit pricing rates and provenance")
    else:
        rates = ("prompt", "completion", "cache_read", "cache_write")
        if any(key not in pricing for key in rates):
            raise ValueError("explicit pricing must contain prompt/completion/cache_read/cache_write")
        if any(type(pricing[key]) not in (int, float) or float(pricing[key]) < 0 for key in rates):
            raise ValueError("explicit pricing rates must be non-negative numbers")
        if not str(pricing.get("pricing_source") or pricing.get("source") or "").strip():
            raise ValueError("explicit pricing requires pricing_source/source provenance")
    extra_body_raw = raw.get("extra_body", {})
    if not isinstance(extra_body_raw, dict):
        raise ValueError("extra_body must be a JSON object")
    # These fields are owned by the calibrator.  Allowing a provider-specific
    # config to replace them would make the frozen prompt/usage identity lie
    # about what was actually dispatched.
    forbidden_extra = {"model", "messages", "temperature", "max_tokens", "enable_thinking"}
    if forbidden_extra & set(extra_body_raw):
        raise ValueError("extra_body cannot override: " + ", ".join(sorted(forbidden_extra & set(extra_body_raw))))
    context_tokens = int(raw.get("context_tokens", 40960))
    cfg = CalibrationConfig(
        model_id=model_id, base_url=base_url.rstrip("/"), api_key_env=api_key_env,
        prompt_version=str(raw["prompt_version"]).strip(), seed=str(raw["seed"]).strip(),
        max_http_attempts=int(raw.get("max_http_attempts", 3)),
        cooldown_seconds=float(raw.get("cooldown_seconds", 1.2)),
        max_tokens=int(raw.get("max_tokens", 500)),
        timeout_seconds=float(raw.get("timeout_seconds", 180)),
        temperature=float(raw.get("temperature", 0)),
        enable_thinking=bool(raw.get("enable_thinking", False)),
        sample_per_class=int(raw.get("sample_per_class", 26)),
        max_cost_usd=float(raw.get("max_cost_usd", 0.50)), pricing=pricing,
        context_tokens=context_tokens, extra_body=dict(extra_body_raw),
    )
    if cfg.max_http_attempts < 1 or cfg.max_tokens < 1 or cfg.sample_per_class < 1:
        raise ValueError("request limits must be positive")
    if cfg.cooldown_seconds < 0 or cfg.timeout_seconds <= 0 or cfg.max_cost_usd < 0:
        raise ValueError("cooldown, timeout, and max_cost_usd are out of range")
    if cfg.context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    return cfg
SPEC = ROOT / "RSI自进化项目_SPEC_v1.2.md"
PREVIOUS_GROUNDING_PACKET = (ROOT
    / ".workbuddy/healthfc-grounding-local-20261016/run-v1/packet.json")
V15_API_PACKET = (ROOT
    / ".workbuddy/healthfc-grounding-api-20261016/run-v2/packet.json")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_source() -> tuple[list[dict[str, str]], list[str]]:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_sha = sha256_file(SOURCE)
    if source_sha != "6b832bd2a73af47f7a8d9dcd3c307defa962cff3746c26b33866170a406721c8":
        raise ValueError("HealthFC source differs from the audited archive")
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 750:
        raise ValueError("HealthFC source row count differs from the audited source")
    return rows, _components(rows, source_sha)


def build_materials(history_path: Path, cfg: CalibrationConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    history = _load_json(history_path)
    rows, components = _read_source()
    if history.get("complete") is not True or history.get("source_sha256") != sha256_file(SOURCE):
        raise ValueError("current HealthFC history scan is incomplete or source-mismatched")
    prior = _load_json(PRIOR_COHORT)
    if prior.get("cohort_sha256") != sha256_obj(
            {k: v for k, v in prior.items() if k != "cohort_sha256"}):
        raise ValueError("prior HealthFC MIPRO cohort self-hash is invalid")

    already_used: set[str] = set()
    for scope in ("mipro_train", "search_validation", "internal_candidate"):
        already_used.update(prior["scopes"][scope]["component_ids"])
    prior_grounding = _load_json(PREVIOUS_GROUNDING_PACKET)
    previous_grounding_components = {
        str(case["group_id"]) for case in prior_grounding.get("cases", [])
    }
    if len(previous_grounding_components) != len(prior_grounding.get("cases", [])):
        raise ValueError("previous grounding cohort is not group-disjoint")
    already_used.update(previous_grounding_components)
    prior_v15_api = _load_json(V15_API_PACKET)
    v15_api_components = {str(case["group_id"]) for case in prior_v15_api.get("cases", [])}
    if len(v15_api_components) != len(prior_v15_api.get("cases", [])):
        raise ValueError("previous v15 API cohort is not group-disjoint")
    already_used.update(v15_api_components)
    matched_rows = {int(index) for index in history.get("matched_row_indexes", [])}
    if any(index < 0 or index >= len(rows) for index in matched_rows):
        raise ValueError("history scan contains an out-of-range source row")
    already_used.update(components[index] for index in matched_rows)

    by_label: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        label = LABELS.get(str(row.get("label") or "").strip())
        if label not in LABELS_ORDERED:
            raise ValueError(f"unknown HealthFC source label at row {index}")
        by_label[label][components[index]].append(index)
    picked: list[tuple[str, int]] = []
    picked_components: set[str] = set()
    # Select the rarest relation first while enforcing article/claim component
    # disjointness across all three label strata.
    for label in ("REFUTED", "SUPPORTED", "NEI"):
        candidates = [component for component in by_label[label]
                      if component not in already_used and component not in picked_components]
        candidates.sort(key=lambda component: hashlib.sha256(
            f"{cfg.seed}\0{label}\0{component}".encode("utf-8")).hexdigest())
        if len(candidates) < cfg.sample_per_class:
            raise ValueError(f"only {len(candidates)} eligible {label} components remain")
        for component in candidates[:cfg.sample_per_class]:
            choices = by_label[label][component]
            row_index = min(choices, key=lambda index: hashlib.sha256(
                f"{cfg.seed}\0{label}\0{component}\0{index}".encode("utf-8")).hexdigest())
            picked.append((label, row_index))
            picked_components.add(component)

    cases = []
    reference_rows = []
    request_hashes = []
    for label, row_index in sorted(picked, key=lambda item: item[1]):
        row = rows[row_index]
        claim = str(row.get("en_claim") or "").strip()
        evidence_text = str(row.get("en_top_sentences") or "").strip()
        if not claim or not evidence_text:
            raise ValueError(f"HealthFC source row {row_index} has empty claim/evidence")
        task_id = f"healthfc:{sha256_file(SOURCE)[:12]}:r{row_index:04d}"
        case_id = "HF-G05-" + hashlib.sha256(
            f"{cfg.seed}\0{task_id}".encode("utf-8")).hexdigest()[:14]
        question = "Medical claim: " + claim
        doc_id = f"healthfc-article-{row_index:04d}"
        evidence = [{"doc_id": doc_id, "text": evidence_text[:MAX_DOC_CHARS]}]
        input_sha = sha256_obj({"question": question, "evidence": evidence})
        docs = evidence
        stage1 = evidence_selection_messages_v11(question, docs, max_doc_chars=MAX_DOC_CHARS)
        # Stage two receives the exact full document plus a placeholder hint. The
        # actual measured request is rebuilt after stage one returns.
        relation_schema = claim_relation_response_format_v16(docs, max_doc_chars=MAX_DOC_CHARS)
        requests = {
            "stage1": [hashlib.sha256(x.encode()).hexdigest() for x in stage1],
            "relation_response_format_sha256": sha256_obj(relation_schema),
        }
        cases.append({"case_id": case_id, "task_id": task_id, "question": question,
                      "evidence": evidence, "input_sha256": input_sha,
                      "group_id": components[row_index], "source_row_index": row_index,
                      "reference": LABEL_TO_RELATION[label]})
        reference_rows.append({"case_id": case_id, "input_sha256": input_sha,
                               "resolved_verdict": LABEL_TO_RELATION[label]})
        request_hashes.append({"case_id": case_id, **requests})

    unsigned_packet = {
        "schema": "healthfc-api-grounding-packet-v2",
        "dataset": "healthfc_fixed_evidence",
        "split": "exploratory_group_disjoint",
        "sample_seed": cfg.seed,
        "sample_per_label": cfg.sample_per_class,
        "source_sha256": sha256_file(SOURCE),
        "history_report_sha256": sha256_obj(history),
        "prior_mipro_cohort_sha256": prior["cohort_sha256"],
        "prior_grounding_packet_sha256": prior_grounding["packet_sha256"],
        "excluded_history_row_count": len(matched_rows),
        "excluded_history_component_count": len({components[i] for i in matched_rows}),
        "excluded_prior_mipro_component_count": sum(
            len(prior["scopes"][scope]["component_ids"])
            for scope in ("mipro_train", "search_validation", "internal_candidate")),
        "excluded_prior_grounding_component_count": len(previous_grounding_components),
        "scope_limitations": [
            "configuration-selected OpenAI-compatible API; bounded internal calibration",
            "diagnostic only; HealthFC source is public and the project previously used other components",
            "not an independent confirmation or production acceptance gate",
            "does not validate multi-document top-3 aggregation",
        ],
        "cases": [{k: v for k, v in case.items() if k != "reference"} for case in cases],
    }
    packet = {**unsigned_packet, "packet_sha256": sha256_obj(unsigned_packet)}
    reference_unsigned = {
        "schema": "healthfc-source-grounding-reference-v1",
        "dataset": "healthfc_fixed_evidence",
        "annotation_origin": "upstream_medical_team",
        "source_sha256": sha256_file(SOURCE),
        "packet_sha256": packet["packet_sha256"],
        "per_case": reference_rows,
    }
    reference = {**reference_unsigned, "reference_sha256": sha256_obj(reference_unsigned)}
    source_files = {
        "spec_sha256": sha256_file(SPEC),
        "calibration_script_sha256": sha256_file(Path(__file__)),
        "grounding_calibration_sha256": sha256_file(ROOT / "src/eval/grounding_calibration.py"),
        "grounding_v11_sha256": sha256_file(ROOT / "src/eval/grounding_v11.py"),
        "grounding_v14_sha256": sha256_file(ROOT / "src/eval/grounding_v14.py"),
        "grounding_v16_sha256": sha256_file(ROOT / "src/eval/grounding_v16.py"),
        "healthfc_adapter_sha256": sha256_file(ROOT / "src/data/healthfc_task_input.py"),
        "provider_sha256": sha256_file(ROOT / "src/runtime/provider.py"),
        "call_ledger_sha256": sha256_file(ROOT / "src/runtime/call_ledger.py"),
    }
    model_identity = {
        "model_id": cfg.model_id,
        "model_sha256": None,
        "provider": (str((cfg.pricing or {}).get("pricing_provider") or
                          (cfg.pricing or {}).get("provider") or
                          "OpenAI-compatible API")),
        "base_url": cfg.base_url,
        "endpoint_host": urlparse(cfg.base_url).hostname,
        "api_key_env": cfg.api_key_env,
        "temperature": cfg.temperature,
        "reasoning": "on" if cfg.enable_thinking else "off",
        "context_tokens": cfg.context_tokens,
        "max_http_attempts_per_call": cfg.max_http_attempts,
        "cooldown_seconds": cfg.cooldown_seconds,
        "max_tokens_per_call": cfg.max_tokens,
        "response_format": cfg.extra_body.get("response_format"),
        "relation_response_format": "healthfc_claim_relation_v16",
        "extra_body": dict(cfg.extra_body),
        "pricing": (dict(cfg.pricing) if cfg.pricing is not None
                    else published_tariff_metadata(cfg.model_id, cfg.base_url)),
        "config_public": cfg.public_dict(),
        "config_sha256": sha256_obj(cfg.public_dict()),
    }
    resolved_price = (cfg.pricing if cfg.pricing is not None else
                      published_price_for_endpoint(cfg.model_id, cfg.base_url))
    if resolved_price is None:
        raise ValueError("model tariff became unknown while freezing the plan")
    # Conservative pre-dispatch ceiling: two logical calls per case, a generous
    # prompt-token allowance for the 6000-character evidence and the configured
    # completion cap. This prevents an arbitrary high-priced model from being
    # dispatched under a budget that cannot cover its worst case.
    prompt_token_allowance = (MAX_DOC_CHARS // 3) + 1200
    estimated_upper_bound = (len(cases) * 2 * (
        prompt_token_allowance * float(resolved_price["prompt"])
        + cfg.max_tokens * float(resolved_price["completion"])) / 1_000_000.0)
    if estimated_upper_bound > cfg.max_cost_usd:
        raise ValueError(
            f"frozen max_cost_usd ${cfg.max_cost_usd:.6f} is below conservative "
            f"dispatch upper bound ${estimated_upper_bound:.6f}")
    protocol = {
        "schema": "healthfc-api-grounding-calibration-plan-v2",
        "dataset": packet["dataset"],
        "prompt_version": cfg.prompt_version,
        "sample_seed": cfg.seed,
        "sample_per_label": cfg.sample_per_class,
        "calibration_thresholds": THRESHOLDS,
        "packet_sha256": packet["packet_sha256"],
        "reference_sha256": reference["reference_sha256"],
        "model": model_identity,
        "source_files": source_files,
        "request_hashes": request_hashes,
        "logical_call_count": len(cases) * 2,
        "max_physical_attempts": len(cases) * 2 * cfg.max_http_attempts,
        "max_http_attempts_per_logical_call": cfg.max_http_attempts,
        "remote_calls": len(cases) * 2,
        "selection_gate_read": False,
        "confirmation_read": False,
        "official_test_read": False,
        "max_cost_usd": cfg.max_cost_usd,
        "estimated_cost_upper_bound_usd": round(estimated_upper_bound, 6),
        "config_sha256": sha256_obj(cfg.public_dict()),
    }
    plan = {**protocol, "plan_sha256": sha256_obj(protocol)}
    return packet, reference, plan


def _ledger_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _post(provider: HttpProvider, messages: tuple[str, str], call_index: int,
          ledger_path: Path, max_tokens: int,
          response_format: dict[str, Any] | None = None) -> dict[str, Any]:
    system, user = messages
    started = datetime.now(timezone.utc).isoformat()
    before_ids = {str(row.get("operation_id")) for row in _ledger_events(ledger_path)
                  if row.get("event") == "logical_start"}
    try:
        with call_component("grounding"):
            response = provider.chat_measured(system, user, max_tokens=max_tokens,
                                              response_format=response_format)
    except Exception as exc:
        events = _ledger_events(ledger_path)
        operation = next((row for row in reversed(events)
                          if row.get("event") == "logical_start"
                          and str(row.get("operation_id")) not in before_ids), {})
        operation_id = str(operation.get("operation_id") or "")
        statuses = [row.get("status") for row in events
                    if row.get("event") == "http_attempt_end"
                    and row.get("operation_id") == operation_id]
        failed = {
            "call_index": call_index,
            "stage": "evidence_selection" if call_index % 2 == 0 else "claim_relation",
            "started_at_utc": started,
            "http_statuses": statuses,
            "http_attempts": len(statuses),
            "operation_id": operation_id,
            "error_type": type(exc).__name__,
            "usage": None,
            "usage_complete": False,
            "prompt_sha256": hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest(),
        }
        raise _CallFailure(failed) from exc
    events = _ledger_events(ledger_path)
    operation = next((row for row in reversed(events)
                      if row.get("event") == "logical_start"
                      and str(row.get("operation_id")) not in before_ids), {})
    operation_id = str(operation.get("operation_id") or "")
    statuses = [row.get("status") for row in events
                if row.get("event") == "http_attempt_end"
                and row.get("operation_id") == operation_id]
    raw_content = response.content
    try:
        parsed = json.loads(raw_content, object_pairs_hook=_strict_json_pairs)
    except (ValueError, TypeError) as exc:
        parsed = {"_invalid_json": str(exc)}
    return {
        "call_index": call_index,
        "stage": "evidence_selection" if call_index % 2 == 0 else "claim_relation",
        "started_at_utc": started,
        "http_statuses": statuses,
        "http_attempts": response.http_attempts,
        "operation_id": operation_id,
        "prompt_sha256": hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest(),
        "raw_output": raw_content,
        "parsed_output": parsed,
        "usage": {"prompt_tokens": response.prompt_tokens,
                  "completion_tokens": response.completion_tokens},
        "usage_complete": response.usage_available and response.cost.get("usage_complete") is True,
        "cost": response.cost,
        "latency_ms": response.latency_ms,
        "cache_hit": response.cache_hit,
    }


class _CallFailure(RuntimeError):
    def __init__(self, record: dict[str, Any]):
        self.record = record
        super().__init__(str(record.get("error_type") or "provider call failed"))


def prepare(out_dir: Path, history_path: Path, cfg: CalibrationConfig) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite local calibration artifacts: {out_dir}")
    packet, reference, plan = build_materials(history_path, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "packet.json", packet)
    write_json(out_dir / "source-reference.json", reference)
    write_json(out_dir / "plan.json", plan)
    return {"packet_sha256": packet["packet_sha256"],
            "reference_sha256": reference["reference_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "task_count": len(packet["cases"]), "logical_calls": plan["logical_call_count"],
            "class_counts": {label: sum(1 for row in reference["per_case"]
                                       if row["resolved_verdict"] == LABEL_TO_RELATION[label])
                             for label in LABELS_ORDERED},
            "remote_calls": plan["remote_calls"]}


def execute(out_dir: Path, history_path: Path, expected_plan_sha256: str, cfg: CalibrationConfig) -> dict[str, Any]:
    packet, reference, current_plan = build_materials(history_path, cfg)
    saved_plan = _load_json(out_dir / "plan.json")
    if (saved_plan != current_plan or expected_plan_sha256 != saved_plan.get("plan_sha256")
            or saved_plan["plan_sha256"] != sha256_obj(
                {k: v for k, v in saved_plan.items() if k != "plan_sha256"})):
        raise ValueError("execution requires the exact unchanged API plan SHA-256")
    if (any((out_dir / name).exists() for name in ("status.json", "report.json", "execution.json"))
            or (out_dir / "calls").exists()):
        raise FileExistsError("a consumed API plan cannot be replayed")
    if not os.environ.get(cfg.api_key_env):
        raise RuntimeError(f"missing API credential in environment variable {cfg.api_key_env}")
    calls_dir = out_dir / "calls"
    calls_dir.mkdir()
    ledger_path = out_dir / "request-ledger.jsonl"
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path.resolve())
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(saved_plan["logical_call_count"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(saved_plan["max_physical_attempts"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_GROUNDING"] = str(saved_plan["logical_call_count"])
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = saved_plan["plan_sha256"]
    os.environ["RSI_HTTP_NO_CACHE"] = "1"
    os.environ["RSI_HTTP_COOLDOWN"] = str(cfg.cooldown_seconds)
    provider = HttpProvider(
        model_id=cfg.model_id, base_url=cfg.base_url, api_key_env=cfg.api_key_env,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens, timeout=cfg.timeout_seconds,
        max_retries=cfg.max_http_attempts, enable_thinking=cfg.enable_thinking,
        pricing_override=({k: float(cfg.pricing[k]) for k in ("prompt", "completion", "cache_read", "cache_write")}
                          if cfg.pricing is not None else None),
        pricing_metadata=(dict(cfg.pricing) if cfg.pricing is not None else {}),
        extra_body=dict(cfg.extra_body),
        cache=False, cooldown=cfg.cooldown_seconds,
    )
    execution_state = {
        "schema": "healthfc-api-grounding-execution-state-v2",
        "status": "running",
        "plan_sha256": saved_plan["plan_sha256"],
        "logical_call_cap": saved_plan["logical_call_count"],
        "physical_http_attempt_cap": saved_plan["max_physical_attempts"],
        "remote_provider": saved_plan["model"]["endpoint_host"],
        "credential_value_persisted": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    execution_path = out_dir / "execution.json"
    fd = os.open(str(execution_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(execution_state, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    write_json(out_dir / "status.json", {"status": "running", "plan_sha256": saved_plan["plan_sha256"]})
    cases = packet["cases"]
    reference_by_id = {row["case_id"]: row for row in reference["per_case"]}
    scored_rows = []
    measured_cost = 0.0
    try:
        for index, case in enumerate(cases):
            docs = case["evidence"]
            first = evidence_selection_messages_v11(
                case["question"], docs, max_doc_chars=MAX_DOC_CHARS)
            write_json(calls_dir / f"{index * 2:04d}.dispatch.json", {
                "status": "reserved_before_dispatch", "call_index": index * 2,
                "prompt_sha256": hashlib.sha256((first[0] + "\0" + first[1]).encode("utf-8")).hexdigest(),
                "plan_sha256": saved_plan["plan_sha256"],
            })
            try:
                call1 = _post(provider, first, index * 2, ledger_path, cfg.max_tokens)
            except _CallFailure as failure:
                write_json(calls_dir / f"{index * 2:04d}.json", failure.record)
                raise
            write_json(calls_dir / f"{index * 2:04d}.json", call1)
            relation_messages = claim_relation_messages_v16(
                case["question"], docs, call1["parsed_output"], max_doc_chars=MAX_DOC_CHARS)
            write_json(calls_dir / f"{index * 2 + 1:04d}.dispatch.json", {
                "status": "reserved_before_dispatch", "call_index": index * 2 + 1,
                "prompt_sha256": hashlib.sha256((relation_messages[0] + "\0" + relation_messages[1]).encode("utf-8")).hexdigest(),
                "plan_sha256": saved_plan["plan_sha256"],
            })
            try:
                call2 = _post(provider, relation_messages, index * 2 + 1, ledger_path,
                              cfg.max_tokens,
                              response_format=claim_relation_response_format_v16(
                                  docs, max_doc_chars=MAX_DOC_CHARS))
            except _CallFailure as failure:
                write_json(calls_dir / f"{index * 2 + 1:04d}.json", failure.record)
                raise
            write_json(calls_dir / f"{index * 2 + 1:04d}.json", call2)
            for call in (call1, call2):
                usd = call.get("cost", {}).get("usd") if isinstance(call.get("cost"), dict) else None
                if isinstance(usd, (int, float)):
                    measured_cost += float(usd)
            if measured_cost > cfg.max_cost_usd:
                raise RuntimeError(
                    f"measured cost ${measured_cost:.6f} exceeded frozen max_cost_usd "
                    f"${cfg.max_cost_usd:.6f} before next dispatch")
            raw = {"evidence_selection": call1["parsed_output"],
                   "claim_relation": call2["parsed_output"]}
            prediction = prediction_v16(raw["claim_relation"], docs, max_doc_chars=MAX_DOC_CHARS)
            scored_rows.append({
                "case_id": case["case_id"],
                "task_id": case["task_id"],
                "source_row_index": case["source_row_index"],
                "input_sha256": case["input_sha256"],
                "reference": reference_by_id[case["case_id"]]["resolved_verdict"],
                "prediction": prediction,
                "raw_verdict": raw,
                "call_indexes": [index * 2, index * 2 + 1],
            })
            write_json(out_dir / "progress.json", {
                "completed_cases": len(scored_rows), "total_cases": len(cases),
                "completed_calls": len(scored_rows) * 2,
                "total_calls": len(cases) * 2,
            })
        summary = summarize(scored_rows)
        call_records = [_load_json(calls_dir / f"{i:04d}.json")
                        for i in range(len(cases) * 2)]
        usage = [read.get("usage") for read in call_records]
        ledger_summary = summarize_ledger(ledger_path)
        if (ledger_summary["logical_operations"] != len(cases) * 2
                or ledger_summary["physical_budget_consumed_or_reserved"] > saved_plan["max_physical_attempts"]
                or ledger_summary["execution_plan_sha256s"] != [saved_plan["plan_sha256"]]):
            raise RuntimeError("request ledger does not match the frozen API plan")
        total_cost = (sum(float(row["cost"]["usd"]) for row in call_records)
                      if all(isinstance(row.get("cost"), dict)
                             and isinstance(row["cost"].get("usd"), (int, float))
                             for row in call_records) else None)
        if total_cost is not None and total_cost > cfg.max_cost_usd:
            raise RuntimeError(f"measured cost ${total_cost:.6f} exceeds frozen max_cost_usd ${cfg.max_cost_usd:.6f}")
        report_unsigned = {
            "schema": "healthfc-api-grounding-calibration-report-v2",
            "status": "completed",
            "dataset": packet["dataset"],
            "split": packet["split"],
            "annotation_origin": "upstream_medical_team",
            "prompt_version": cfg.prompt_version,
            "model": saved_plan["model"],
            "plan_sha256": saved_plan["plan_sha256"],
            "packet_sha256": packet["packet_sha256"],
            "reference_sha256": reference["reference_sha256"],
            "task_count": len(scored_rows),
            "logical_calls": len(cases) * 2,
            "http_successes": ledger_summary["physical_status_counts"].get("200", 0),
            "request_ledger": ledger_summary,
            "request_ledger_sha256": sha256_file(ledger_path),
            "usage_complete": all(isinstance(item, dict)
                                  and type(item.get("prompt_tokens")) is int
                                  and type(item.get("completion_tokens")) is int for item in usage),
            "total_cost_usd": total_cost,
            "estimated_cost_upper_bound_usd": saved_plan["estimated_cost_upper_bound_usd"],
            "remote_calls": ledger_summary["logical_operations"],
            "thresholds": {
                "minimum_usable": 30, "minimum_per_class": 3,
                "macro_f1": 0.80, "minimum_class_recall": 0.75,
                "false_support_rate": 0.10,
            },
            "limitations": packet["scope_limitations"],
            "per_case": scored_rows,
            **summary,
            "promotion_authorized": False,
        }
        report = {**report_unsigned, "report_sha256": sha256_obj(report_unsigned)}
        write_json(out_dir / "report.json", report)
        execution_state.update({"status": "completed", "report_sha256": report["report_sha256"],
                                "request_ledger_sha256": report["request_ledger_sha256"]})
        write_json(execution_path, execution_state)
        write_json(out_dir / "status.json", {
            "status": "completed", "plan_sha256": saved_plan["plan_sha256"],
            "report_sha256": report["report_sha256"],
        })
        return {k: v for k, v in report.items() if k not in {"per_case"}}
    except Exception as exc:
        ledger_summary = summarize_ledger(ledger_path)
        execution_state.update({"status": "failed_consumed",
                                "completed_cases": len(scored_rows),
                                "logical_operations": ledger_summary["logical_operations"],
                                "physical_http_attempts": ledger_summary["physical_http_attempts"],
                                "physical_incomplete": ledger_summary["physical_incomplete"],
                                "error_type": type(exc).__name__})
        write_json(execution_path, execution_state)
        write_json(out_dir / "status.json", {
            "status": "failed_consumed", "plan_sha256": saved_plan["plan_sha256"],
            "completed_cases": len(scored_rows),
            "logical_operations": ledger_summary["logical_operations"],
            "physical_http_attempts": ledger_summary["physical_http_attempts"],
            "physical_incomplete": ledger_summary["physical_incomplete"],
            "error_type": type(exc).__name__,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        result = prepare(args.output_dir, args.history, cfg)
    else:
        if not args.expected_plan_sha256:
            raise SystemExit("execute requires --expected-plan-sha256")
        result = execute(args.output_dir, args.history, args.expected_plan_sha256, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
