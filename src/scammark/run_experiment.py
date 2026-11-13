"""Resumable ScamMark Qwen3 pilot and full-experiment runner."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import stat
import sys
import threading
import time
import tomllib
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .bertscore_eval import BERTScoreConfig, default_model_dir, get_bertscore_evaluator
from .dataset_paths import DatasetPathMapper
from .evaluate import evaluate_prediction_records
from .evidence_loader import load_jsonl, load_samples
from .experiment_state import (
    ExperimentContractError,
    ExperimentState,
    project_request_metadata,
    scan_call_ledger,
    scan_checkpoints,
    summarize_sample_progress,
)
from .frozen_assets import verify_frozen_assets
from .image_preprocessing import ImagePolicy
from .model_client import (
    OpenAIAPIModelClient,
    _load_local_env,
    endpoint_identity,
    selected_model_ids,
)
from .observability import (
    ApiBudgetStop,
    AuthBillingStop,
    BreakerAcknowledgementRequired,
    CircuitBreakerStop,
    ContractDriftStop,
    GracefulStopRequested,
    FailureCircuitBreaker,
    HeartbeatWorker,
    ObservabilitySettings,
    ProgressTracker,
    RunControlError,
    RunEventLogger,
    SafeEventSink,
    StopController,
    SummaryScheduler,
    classify_failure,
    project_remaining_stage_counts,
    sanitize_payload,
    sanitize_text,
)
from .pilot_manifest import select_pilot_records
from .pipeline import ScamMarkPipeline
from .prediction_schema import INVALID_BINARY, normalize_final_prediction
from .reliability import RequestPolicy
from .run_status import _project_events, read_jsonl_tail
from .taxonomy import CANONICAL_LABELS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )


def _should_flush_predictions(index: int, *, total: int, interval: int) -> bool:
    interval = max(1, int(interval))
    return index == total or index % interval == 0


def _merge_prediction_records(
    existing: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    manifest_ids: Sequence[str],
) -> list[dict[str, Any]]:
    order: dict[str, int] = {}
    for raw_sample_id in manifest_ids:
        if type(raw_sample_id) is not str or not raw_sample_id.strip():
            raise ExperimentContractError("invalid manifest prediction sample_id")
        if raw_sample_id in order:
            safe_id = sanitize_text(raw_sample_id, limit=200)
            raise ExperimentContractError(
                f"duplicate manifest prediction sample_id: {safe_id}"
            )
        order[raw_sample_id] = len(order)

    merged: dict[str, dict[str, Any]] = {}
    for source_name, rows in (("existing", existing), ("updates", updates)):
        seen: set[str] = set()
        for row in rows:
            if type(row) is not dict:
                raise ExperimentContractError(
                    f"invalid {source_name} prediction row: expected mapping"
                )
            try:
                raw_sample_id = row.get("sample_id")
                if raw_sample_id in (None, ""):
                    raw_sample_id = row.get("id")
                sample_id = str(raw_sample_id or "").strip()
            except Exception:
                sample_id = "<unprintable>"
            safe_id = sanitize_text(sample_id, limit=200)
            if not sample_id or sample_id not in order or sample_id in seen:
                raise ExperimentContractError(
                    f"invalid {source_name} prediction sample_id: {safe_id}"
                )
            seen.add(sample_id)
            try:
                merged[sample_id] = dict(row)
            except Exception as exc:
                error_type = sanitize_text(type(exc).__name__, limit=100)
                raise ExperimentContractError(
                    f"invalid {source_name} prediction row: {error_type}"
                ) from None
    return [merged[sample_id] for sample_id in manifest_ids if sample_id in merged]


def _resolve_env_reference(value: Any, *, description: str) -> str:
    resolved = str(value or "").strip()
    if not resolved.startswith("env:"):
        return resolved
    variable = resolved.removeprefix("env:").strip()
    if not variable or not variable.replace("_", "").isalnum():
        raise ValueError(f"Invalid {description} environment reference: {resolved}")
    resolved = str(os.getenv(variable) or "").strip()
    if not resolved:
        raise ValueError(f"Required environment variable {variable} is not set")
    return resolved


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {
        "experiment",
        "models",
        "endpoints",
        "input_tokens",
        "output_tokens",
        "images",
        "request_limits",
        "data",
        "paths",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Experiment config is missing sections: {', '.join(missing)}")
    if config["models"].get("vision") != "qwen3-vl-flash":
        raise ValueError("Formal pilot vision model must be qwen3-vl-flash")
    if config["models"].get("text") != "gpt-4o-mini":
        raise ValueError("Formal pilot text model must be gpt-4o-mini")

    _load_local_env()
    endpoints = config["endpoints"]
    for key in ("text_base_url", "vision_base_url"):
        endpoints[key] = _resolve_env_reference(
            endpoints.get(key), description="endpoint"
        )

    data = config["data"]
    if not data.get("source_dataset_root") or not data.get("runtime_dataset_root"):
        raise ValueError(
            "Experiment config requires source_dataset_root and runtime_dataset_root"
        )
    data["runtime_dataset_root"] = _resolve_env_reference(
        data["runtime_dataset_root"], description="dataset root"
    )

    text_provider = str(endpoints.get("text_provider") or "Crazyrouter")
    vision_provider = str(endpoints.get("vision_provider") or "Crazyrouter")
    text_url = urlsplit(str(endpoints.get("text_base_url") or ""))
    vision_url = urlsplit(str(endpoints.get("vision_base_url") or ""))
    if text_provider != "Crazyrouter" or text_url.netloc.lower() != "cn.crazyrouter.com":
        raise ValueError("Formal pilot text endpoint must be Crazyrouter at cn.crazyrouter.com")
    if vision_provider == "Crazyrouter":
        if vision_url.netloc.lower() != "cn.crazyrouter.com":
            raise ValueError("Crazyrouter vision endpoint must be cn.crazyrouter.com")
    elif vision_provider == "Alibaba Cloud Model Studio":
        host = vision_url.netloc.lower()
        if not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("Model Studio vision endpoint must use a Beijing workspace domain")
        if vision_url.path.rstrip("/") != "/compatible-mode/v1":
            raise ValueError("Model Studio vision endpoint must use /compatible-mode/v1")
    else:
        raise ValueError(f"Unsupported formal vision provider: {vision_provider}")
    return config


def sanitized_contract_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove workspace identifiers while preserving endpoint compatibility hashes."""

    sanitized = copy.deepcopy(config)
    endpoints = sanitized.get("endpoints") or {}
    for prefix in ("text", "vision"):
        key = f"{prefix}_base_url"
        value = str(endpoints.get(key) or "")
        identity = endpoint_identity(value)
        parsed = urlsplit(value)
        endpoints[key] = f"{parsed.scheme}://{identity['host']}{parsed.path}"
        endpoints[f"{prefix}_endpoint_sha256"] = identity["sha256"]
    return sanitized


def _select_pilot_inputs(
    manifest_path: Path,
    evidence_path: Path,
    output_dir: Path,
    size: int,
    seed: int,
    *,
    path_mapper: DatasetPathMapper | None = None,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    manifest = load_jsonl(manifest_path)
    evidence = load_jsonl(evidence_path)
    if path_mapper is not None:
        manifest = path_mapper.relocate_records(manifest)
        evidence = path_mapper.relocate_records(evidence)
    selected = select_pilot_records(manifest, size=size, seed=seed)
    selected_ids = [str(record.get("sample_id") or record.get("id")) for record in selected]
    selected_set = set(selected_ids)
    manifest_by_id = {str(record.get("sample_id") or record.get("id")): record for record in manifest}
    evidence_by_id = {str(record.get("sample_id") or record.get("id")): record for record in evidence}
    if any(sample_id not in evidence_by_id for sample_id in selected_ids):
        raise ValueError("Pilot selection is missing frozen evidence records")
    manifest_records = [manifest_by_id[sample_id] for sample_id in selected_ids]
    evidence_records = [evidence_by_id[sample_id] for sample_id in selected_ids]
    if len(selected_set) != size:
        raise ValueError("Pilot selection contains duplicate sample IDs")
    pilot_manifest = output_dir / "manifest.jsonl"
    pilot_evidence = output_dir / "web_evidence.jsonl"
    _atomic_write_jsonl(pilot_manifest, manifest_records)
    _atomic_write_jsonl(pilot_evidence, evidence_records)
    return pilot_manifest, pilot_evidence, manifest_records


def _select_full_inputs(
    manifest_path: Path,
    evidence_path: Path,
    output_dir: Path,
    *,
    expected_size: int,
    path_mapper: DatasetPathMapper | None = None,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Freeze the complete manifest/evidence pair in manifest order."""

    manifest = load_jsonl(manifest_path)
    evidence = load_jsonl(evidence_path)
    if path_mapper is not None:
        manifest = path_mapper.relocate_records(manifest)
        evidence = path_mapper.relocate_records(evidence)
    if len(manifest) != expected_size:
        raise ValueError(
            f"Full manifest contains {len(manifest)} records; expected {expected_size}"
        )

    manifest_ids = [str(record.get("sample_id") or record.get("id") or "") for record in manifest]
    evidence_ids = [str(record.get("sample_id") or record.get("id") or "") for record in evidence]
    if any(not sample_id for sample_id in manifest_ids + evidence_ids):
        raise ValueError("Full manifest and evidence records require non-empty sample IDs")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("Full manifest contains duplicate sample IDs")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("Full evidence contains duplicate sample IDs")
    if set(manifest_ids) != set(evidence_ids):
        raise ValueError("Full evidence sample IDs must exactly match the manifest")

    evidence_by_id = dict(zip(evidence_ids, evidence, strict=True))
    ordered_evidence = [evidence_by_id[sample_id] for sample_id in manifest_ids]
    frozen_manifest = output_dir / "manifest.jsonl"
    frozen_evidence = output_dir / "web_evidence.jsonl"
    _atomic_write_jsonl(frozen_manifest, manifest)
    _atomic_write_jsonl(frozen_evidence, ordered_evidence)
    return frozen_manifest, frozen_evidence, manifest


def _safe_exception_payload(exc: Exception) -> dict[str, str]:
    try:
        error_type = type(exc).__name__
        if type(error_type) is not str or not error_type:
            error_type = "Exception"
    except BaseException:
        error_type = "Exception"
    fallback = {
        "type": sanitize_text(error_type, limit=100) or "Exception",
        "message": "<redacted-field>",
    }
    try:
        projected = project_request_metadata(
            {
                "error_type": fallback["type"],
                "message": fallback["message"],
            }
        )
        safe = sanitize_payload(
            {
                "type": projected.get("error_type") or fallback["type"],
                "message": projected.get("message") or fallback["message"],
            }
        )
        if not isinstance(safe, Mapping):
            return fallback
        return {
            "type": sanitize_text(safe.get("type"), limit=100)
            or fallback["type"],
            "message": sanitize_text(safe.get("message"), limit=500)
            or fallback["message"],
        }
    except BaseException:
        return fallback


def _control_error_for_inference_failure(
    exc: Exception,
    *,
    allow_unexpected_local: bool = False,
) -> RunControlError | None:
    if isinstance(exc, RunControlError):
        return exc
    classification = classify_failure(exc)
    if classification.systemic:
        control_type: type[RunControlError]
        if classification.category == "auth_billing":
            control_type = AuthBillingStop
        elif classification.category == "api_budget":
            control_type = ApiBudgetStop
        elif classification.category == "contract_drift":
            control_type = ContractDriftStop
        else:
            raise exc
        return control_type(classification.message)
    if classification.category in {"sample_permanent", "transient"}:
        return None
    request_metadata = getattr(exc, "request_metadata", None)
    if (
        isinstance(request_metadata, Mapping)
        and (
            request_metadata.get("repair_attempted") is True
            or request_metadata.get("schema_repair_attempted") is True
        )
    ):
        return None
    if allow_unexpected_local:
        return None
    raise exc


def _control_stop_fields(exc: RunControlError) -> dict[str, Any]:
    if isinstance(exc, GracefulStopRequested):
        return {
            "reason": "graceful_pause",
            "category": "graceful_stop",
            "exit_code": exc.exit_code,
            "boundary": exc.boundary,
            "forced": False,
            "message": "graceful stop requested",
        }
    if isinstance(exc, BreakerAcknowledgementRequired):
        reason = "breaker_ack_required"
        message = "prior circuit breaker acknowledgement required"
    elif isinstance(exc, CircuitBreakerStop):
        reason = "circuit_breaker"
        message = "circuit breaker opened"
    elif isinstance(exc, ApiBudgetStop):
        reason = "api_budget"
        message = "physical API call budget exhausted"
    elif isinstance(exc, ContractDriftStop):
        reason = "contract_drift"
        message = "experiment contract drift detected"
    elif isinstance(exc, AuthBillingStop):
        reason = "auth_billing"
        message = "authentication or billing failure"
    else:
        reason = "control_stop"
        message = "run control stop requested"
    return {
        "reason": reason,
        "category": reason,
        "exit_code": exc.exit_code,
        "boundary": "sample_loop",
        "forced": False,
        "message": message,
    }


def _partial_prediction(sample: dict[str, Any], state: ExperimentState, exc: Exception) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or sample.get("id") or "")
    failure = _safe_exception_payload(exc)
    checkpoint = state.load_stages(sample_id)
    outputs = {
        stage: value.get("output") or {}
        for stage, value in checkpoint.items()
        if value.get("status") != "error" and isinstance(value.get("output"), dict)
    }
    metadata = {
        stage: value.get("request_metadata") or {}
        for stage, value in checkpoint.items()
        if isinstance(value, dict)
    }
    partial = {
        "binary": INVALID_BINARY,
        "scam_type": "none",
        "description": "",
        "reasoning": f"backend_error:{failure['type']}: {failure['message']}",
        "boxes": [],
        "evidence_used": {"prompt_stages": outputs, "request_metadata": metadata},
        "sample_id": sample_id,
        "backend_error": True,
    }
    sanitized = sanitize_payload(partial)
    return dict(sanitized) if isinstance(sanitized, Mapping) else partial


def _run_primary_and_oracle_diagnostic(
    pipeline: ScamMarkPipeline,
    sample: dict[str, Any],
    *,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the primary cascade, then isolate optional Gold-gated diagnostics."""

    primary = pipeline.predict(sample)
    diagnostic: dict[str, Any] = {"status": "not_requested"}
    if enabled and sample.get("binary") == "scam" and primary.get("binary") != "scam":
        try:
            diagnostic_prediction = pipeline.predict(sample, force_downstream=True)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never replace a valid primary result.
            control = _control_error_for_inference_failure(
                exc, allow_unexpected_local=True
            )
            if control is not None:
                if control is exc:
                    raise
                raise control from exc
            diagnostic = {
                "status": "error",
                "oracle_gated": True,
                "error": _safe_exception_payload(exc),
            }
        else:
            diagnostic = {
                "status": "complete",
                "oracle_gated": True,
                "prediction": diagnostic_prediction,
            }
    return primary, diagnostic


def _prediction_record(
    sample: dict[str, Any],
    prediction: dict[str, Any],
    oracle_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truth = {
        "binary": sample.get("binary"),
        "scam_type": sample.get("scam_type") or "none",
        "boxes": sample.get("boxes") or [],
        "description": sample.get("description") or "",
        "label_source": sample.get("label_source", "manifest"),
    }
    if sample.get("image_size"):
        truth["image_size"] = sample["image_size"]
    return {
        "sample_id": sample.get("sample_id") or sample.get("id"),
        "input": {
            "url": sample.get("url"),
            "title": sample.get("title"),
            "image_path": sample.get("image_path"),
            "mhtml_audit_status": sample.get("mhtml_audit_status"),
            "mhtml_used": sample.get("mhtml_used"),
            "gold_input_conflict": sample.get("gold_input_conflict", False),
        },
        "truth": truth,
        "prediction": prediction,
        "diagnostics": {
            "oracle_forced_downstream": oracle_diagnostic or {"status": "not_requested"}
        },
    }


def _stage_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        prediction = record.get("prediction") or {}
        evidence = prediction.get("evidence_used") or {}
        for stage, metadata in (evidence.get("request_metadata") or {}).items():
            if isinstance(metadata, dict):
                rows.append({"sample_id": record.get("sample_id"), "stage": stage, "diagnostic": False, **metadata})
        diagnostic = ((record.get("diagnostics") or {}).get("oracle_forced_downstream") or {})
        diagnostic_prediction = diagnostic.get("prediction") or {}
        diagnostic_evidence = diagnostic_prediction.get("evidence_used") or {}
        for stage, metadata in (diagnostic_evidence.get("request_metadata") or {}).items():
            if isinstance(metadata, dict) and not any(
                row.get("sample_id") == record.get("sample_id")
                and row.get("stage") == stage
                and row.get("request_fingerprint") == metadata.get("request_fingerprint")
                for row in rows
            ):
                rows.append({"sample_id": record.get("sample_id"), "stage": stage, "diagnostic": True, **metadata})
    return rows


def _image_payload_within_limits(image: dict[str, Any], policy: dict[str, Any]) -> bool:
    if int(image.get("total_pixels") or 0) > int(policy["total_pixels"]):
        return False
    for asset in image.get("assets") or []:
        size = int(asset.get("data_url_size") or 0)
        limit = int(policy["max_base64_bytes"])
        if asset.get("kind") == "contact_sheet":
            limit = min(limit, int(policy["contact_sheet_base64_bytes"]))
        if size > limit:
            return False
    return True


def _estimate_api_cost(
    pricing: dict[str, Any],
    *,
    text_usage: dict[str, int],
    vision_usage: dict[str, int],
    text_model: str,
    vision_model: str,
) -> dict[str, Any]:
    if isinstance(pricing.get("text"), dict) and isinstance(pricing.get("vision"), dict):
        by_role: dict[str, dict[str, Any]] = {}
        totals_by_currency: dict[str, float] = {}
        for role, role_pricing, role_usage, model in (
            ("text", pricing["text"], text_usage, text_model),
            ("vision", pricing["vision"], vision_usage, vision_model),
        ):
            currency = str(role_pricing.get("currency") or "UNKNOWN")
            input_rate = float(role_pricing.get("input_per_million") or 0)
            output_rate = float(role_pricing.get("output_per_million") or 0)
            estimated = (
                int(role_usage.get("prompt_tokens") or 0) * input_rate
                + int(role_usage.get("completion_tokens") or 0) * output_rate
            ) / 1_000_000
            by_role[role] = {
                "model": model,
                "currency": currency,
                "estimated_total": estimated,
                "input_per_million": input_rate,
                "output_per_million": output_rate,
                "source": role_pricing.get("source"),
            }
            totals_by_currency[currency] = totals_by_currency.get(currency, 0.0) + estimated
        return {
            "status": str(pricing.get("status") or "not_computed"),
            "by_role": by_role,
            "totals_by_currency": totals_by_currency,
            "access_date": pricing.get("access_date"),
            "note": "Estimates retain provider billing currencies; no exchange rate is assumed.",
        }

    estimated = (
        int(text_usage.get("prompt_tokens") or 0) * float(pricing.get("text_input_per_million", 0))
        + int(text_usage.get("completion_tokens") or 0) * float(pricing.get("text_output_per_million", 0))
        + int(vision_usage.get("prompt_tokens") or 0) * float(pricing.get("vision_input_per_million", 0))
        + int(vision_usage.get("completion_tokens") or 0) * float(pricing.get("vision_output_per_million", 0))
    ) / 1_000_000
    return {
        "status": str(pricing.get("status") or "not_computed"),
        "currency": str(pricing.get("currency") or "USD"),
        "estimated_total": estimated,
        "rates_per_million_tokens": {
            "text_input": pricing.get("text_input_per_million"),
            "text_output": pricing.get("text_output_per_million"),
            "vision_input": pricing.get("vision_input_per_million"),
            "vision_output": pricing.get("vision_output_per_million"),
        },
        "sources": [pricing.get("text_source"), pricing.get("vision_source")],
        "access_date": pricing.get("access_date"),
        "note": "Estimate only; the provider invoice remains authoritative.",
    }


def summarize_engineering(
    records: list[dict[str, Any]],
    *,
    client: OpenAIAPIModelClient,
    state: ExperimentState,
    config: dict[str, Any],
    elapsed_seconds: float,
    pilot_size: int | None,
) -> dict[str, Any]:
    rows = _stage_metadata(records)
    attempts = [int(row.get("cumulative_attempt_count") or row.get("attempt_count") or 0) for row in rows]
    usage: Counter[str] = Counter()
    usage_by_model: dict[str, Counter[str]] = {}
    for row in rows:
        row_usage = {key: int(value or 0) for key, value in (row.get("usage") or {}).items()}
        usage.update(row_usage)
        model = str(row.get("model") or "unknown")
        usage_by_model.setdefault(model, Counter()).update(row_usage)
    finish_reasons = Counter(str(row.get("finish_reason") or "unknown") for row in rows)
    first_attempt_failures = sum(attempt > 1 for attempt in attempts)
    expected_stages_ok = True
    oracle_diagnostics_complete = True
    usage_complete = True
    image_limits_ok = True
    request_limits_ok = True
    routing_ok = True
    truncated_completion_count = 0
    image_policy = config["images"]
    request_limits = config["request_limits"]
    expected_models = {
        "url": config["models"]["text"],
        "text": config["models"]["text"],
        "visual": config["models"]["vision"],
        "cross": config["models"]["vision"],
        "binary": config["models"]["vision"],
        "type": config["models"]["vision"],
        "grounding": config["models"]["vision"],
    }
    expected_hosts = {
        "url": endpoint_identity(config["endpoints"]["text_base_url"])["host"],
        "text": endpoint_identity(config["endpoints"]["text_base_url"])["host"],
        "visual": endpoint_identity(config["endpoints"]["vision_base_url"])["host"],
        "cross": endpoint_identity(config["endpoints"]["vision_base_url"])["host"],
        "binary": endpoint_identity(config["endpoints"]["vision_base_url"])["host"],
        "type": endpoint_identity(config["endpoints"]["vision_base_url"])["host"],
        "grounding": endpoint_identity(config["endpoints"]["vision_base_url"])["host"],
    }
    for record in records:
        prediction = record.get("prediction") or {}
        evidence = prediction.get("evidence_used") or {}
        stages = set((evidence.get("prompt_stages") or {}).keys())
        if not set(("url", "text", "visual", "cross", "binary")).issubset(stages):
            expected_stages_ok = False
        if prediction.get("binary") == "scam" and not {"type", "grounding"}.issubset(stages):
            expected_stages_ok = False
        diagnostic = ((record.get("diagnostics") or {}).get("oracle_forced_downstream") or {})
        if (
            bool(config["experiment"].get("forced_diagnostics"))
            and (record.get("truth") or {}).get("binary") == "scam"
            and prediction.get("binary") != "scam"
        ):
            diagnostic_stages = set(
                (((diagnostic.get("prediction") or {}).get("evidence_used") or {}).get("prompt_stages") or {})
            )
            if diagnostic.get("status") != "complete" or not {"type", "grounding"}.issubset(diagnostic_stages):
                oracle_diagnostics_complete = False
    for row in rows:
        if not (row.get("usage") or {}):
            usage_complete = False
        stage = str(row.get("stage") or "")
        if str(row.get("finish_reason") or "").lower() == "length":
            truncated_completion_count += 1
        if str(row.get("model") or "") != str(expected_models.get(stage) or ""):
            routing_ok = False
        if row.get("response_model_matches_request") is not True:
            routing_ok = False
        if str(row.get("endpoint_host") or "").lower() != str(expected_hosts.get(stage) or "").lower():
            routing_ok = False
        image = row.get("image") or {}
        total_pixels = int(image.get("total_pixels") or 0)
        total_base64_bytes = int(image.get("total_base64_bytes") or 0)
        if not _image_payload_within_limits(image, image_policy):
            image_limits_ok = False
        estimated_visual_tokens = math.ceil(total_pixels / 1024) if total_pixels else 0
        if estimated_visual_tokens > int(request_limits["max_visual_tokens"]):
            request_limits_ok = False
        prompt_tokens = int((row.get("usage") or {}).get("prompt_tokens") or 0)
        if total_pixels and prompt_tokens > int(request_limits["max_multimodal_prompt_tokens"]):
            request_limits_ok = False
        estimated_request_bytes = total_base64_bytes + int(
            ((row.get("prompt") or {}).get("final_tokens") or 0) * 4
        )
        if estimated_request_bytes > int(request_limits["max_total_request_bytes"]):
            request_limits_ok = False
    model_ids = selected_model_ids(client)
    cumulative_api_calls = state.total_api_calls()
    accounted_api_calls = sum(
        int(row.get("cumulative_api_call_count") or row.get("api_call_count") or 0)
        for row in rows
    )
    pricing = config.get("pricing") or {}
    text_usage = usage_by_model.get(model_ids["text_model"], Counter())
    vision_usage = usage_by_model.get(model_ids["visual_model"], Counter())
    cost = _estimate_api_cost(
        pricing,
        text_usage=text_usage,
        vision_usage=vision_usage,
        text_model=model_ids["text_model"],
        vision_model=model_ids["visual_model"],
    )
    return {
        "backend": "openai-compatible",
        "provider": config["experiment"]["provider"],
        "provider_access_date": config["experiment"]["provider_access_date"],
        **model_ids,
        "prompt_mode": config["experiment"]["prompt_mode"],
        "run_mode": "full" if pilot_size is None else "pilot",
        "pilot_size": pilot_size,
        "sample_count": len(records),
        "stage_count": len(rows),
        "backend_error_count": sum(bool((record.get("prediction") or {}).get("backend_error")) for record in records),
        "schema_invalid_count": sum(not (record.get("prediction") or {}).get("schema_valid", False) for record in records),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / len(records) if records else None,
        "api_calls_this_process": client.api_call_count,
        "api_calls_cumulative": cumulative_api_calls,
        "api_calls_accounted_by_stage": accounted_api_calls,
        "unattributed_api_call_count": max(0, cumulative_api_calls - accounted_api_calls),
        "first_attempt_failure_rate": first_attempt_failures / len(attempts) if attempts else 0.0,
        "average_attempt_count": sum(attempts) / len(attempts) if attempts else 0.0,
        "max_attempt_count": max(attempts, default=0),
        "usage": dict(usage),
        "usage_by_model": {model: dict(values) for model, values in usage_by_model.items()},
        "finish_reasons": dict(finish_reasons),
        "expected_stage_sets_complete": expected_stages_ok,
        "oracle_diagnostics_complete": oracle_diagnostics_complete,
        "usage_complete": usage_complete,
        "image_limits_ok": image_limits_ok,
        "request_limits_ok": request_limits_ok,
        "routing_ok": routing_ok,
        "truncated_completion_count": truncated_completion_count,
        "cost": cost,
    }


def _acceptance_gates(
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
    config: dict[str, Any],
    pilot_size: int | None,
) -> dict[str, Any]:
    engineering = metrics["engineering"]
    common = {
        "backend_error_count_zero": engineering.get("backend_error_count") == 0,
        "schema_error_count_zero": engineering.get("schema_invalid_count") == 0,
        "stage_sets_complete": bool(engineering.get("expected_stage_sets_complete")),
        "usage_complete": bool(engineering.get("usage_complete")),
        "image_limits_ok": bool(engineering.get("image_limits_ok")),
        "request_limits_ok": bool(engineering.get("request_limits_ok")),
        "routing_ok": bool(engineering.get("routing_ok")),
        "no_truncated_completions": engineering.get("truncated_completion_count") == 0,
        "metrics_inputs_clean": bool(engineering.get("clean_metrics")),
        "all_api_calls_attributed": engineering.get("unattributed_api_call_count") == 0,
        "oracle_diagnostics_complete": bool(engineering.get("oracle_diagnostics_complete")),
        "api_call_cap_respected": engineering.get("api_calls_cumulative", 0)
        <= config["experiment"]["api_call_limit"],
    }
    if pilot_size == 60 or pilot_size is None:
        support = {
            (record.get("truth") or {}).get("scam_type")
            for record in records
            if (record.get("truth") or {}).get("binary") == "scam"
        }
        common.update(
            {
                "first_attempt_failure_rate_at_most_0_05": engineering.get("first_attempt_failure_rate", 1) <= 0.05,
                "average_attempt_at_most_1_25": engineering.get("average_attempt_count", 99) <= 1.25,
                "all_25_subtypes_supported": support == set(CANONICAL_LABELS),
                "grounding_has_end_to_end_iou_0_5_match": metrics["grounding"]["end_to_end"]["counts_at_0_5"]["tp"] >= 1,
            }
        )
        bertscore_status = metrics["rationale"]["end_to_end"]["bertscore_status"]
        if bool(config["experiment"].get("with_bertscore")):
            common["bertscore_end_to_end_computed"] = bertscore_status == "computed"
        else:
            common["bertscore_deferred_as_configured"] = bertscore_status == "not_requested"
    if pilot_size is None:
        common["full_sample_count_exact"] = engineering.get("sample_count") == int(
            config["experiment"]["expected_full_sample_count"]
        )
    return {"passed": all(common.values()), "checks": common}


def _write_human_candidates(path: Path, records: list[dict[str, Any]], limit: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        prediction = record.get("prediction") or {}
        for box_index, box in enumerate(prediction.get("boxes") or []):
            if not isinstance(box, dict) or not box.get("xyxy"):
                continue
            rows.append(
                {
                    "candidate_id": "ScamMark",
                    "item_id": f"item-{len(rows) + 1:04d}",
                    "sample_id": record.get("sample_id"),
                    "image_path": (record.get("input") or {}).get("image_path"),
                    "xyxy": json.dumps(box.get("xyxy")),
                    "predicted_reason": box.get("reason", ""),
                    "rater_id": "",
                    "correctness": "",
                    "faithfulness": "",
                    "specificity": "",
                    "helpfulness": "",
                }
            )
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "candidate_id", "item_id", "sample_id", "image_path", "xyxy", "predicted_reason",
            "rater_id", "correctness", "faithfulness", "specificity", "helpfulness",
        ])
        writer.writeheader()
        writer.writerows(rows)


def _write_pilot_summary(path: Path, metrics: dict[str, Any], gates: dict[str, Any]) -> None:
    binary = metrics["binary"]
    subtype = metrics["subtype"]["end_to_end"]
    grounding = metrics["grounding"]["end_to_end"]
    rationale = metrics["rationale"]["end_to_end"]
    engineering = metrics["engineering"]
    cost = engineering["cost"]
    if "estimated_total" in cost:
        cost_text = f"{cost['estimated_total']:.6f} {cost['currency']}"
    else:
        cost_text = " + ".join(
            f"{value:.6f} {currency}"
            for currency, value in sorted((cost.get("totals_by_currency") or {}).items())
        ) or "not computed"
    run_mode = str(engineering.get("run_mode") or "pilot")
    title = "ScamMark Qwen3 Full Experiment Summary" if run_mode == "full" else "ScamMark Qwen3 Pilot Summary"
    lines = [
        f"# {title}",
        "",
        f"- Samples: {engineering['sample_count']}",
        f"- Binary accuracy/F1: {binary['accuracy']:.4f} / {binary['f1']:.4f}",
        f"- Subtype accuracy/fixed-25 macro-F1: {subtype['accuracy']:.4f} / {subtype['macro_f1']:.4f}",
        f"- Subtype observed-class macro-F1 (diagnostic only): {subtype['observed_class_macro_f1']:.4f}",
        f"- Grounding mIoU/F1@0.5: {grounding['miou']:.4f} / {grounding['f1_at_0_5']:.4f}",
        f"- Rationale ROUGE-L/BERTScore: {rationale['rouge_l_f1']:.4f} / {rationale['bertscore_f1']}",
        f"- Backend/schema errors: {engineering['backend_error_count']} / {engineering['schema_invalid_count']}",
        f"- API calls (process/cumulative): {engineering['api_calls_this_process']} / {engineering['api_calls_cumulative']}",
        f"- Estimated API cost: {cost_text}",
        f"- First-attempt failure rate: {engineering['first_attempt_failure_rate']:.4f}",
        f"- Average attempts: {engineering['average_attempt_count']:.4f}",
        f"- Acceptance gates: {'PASS' if gates['passed'] else 'FAIL'}",
        "",
        "## Gate Details",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in gates["checks"].items())
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _implementation_source_hashes() -> dict[str, str]:
    module_dir = Path(__file__).resolve().parent
    names = (
        "run_experiment.py",
        "pipeline.py",
        "model_client.py",
        "experiment_state.py",
        "input_budget.py",
        "image_preprocessing.py",
        "mhtml_policy.py",
        "prompts.py",
        "prediction_schema.py",
        "reliability.py",
        "metrics.py",
        "evaluate.py",
        "observability.py",
        "run_status.py",
        "result_contract.py",
    )
    return {name: _sha256(module_dir / name) for name in names}


def _selected_image_hashes(records: list[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for record in records:
        sample_id = str(record.get("sample_id") or record.get("id") or "")
        image_path = Path(str(record.get("image_path") or ""))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Experiment screenshot is missing for sample {sample_id}: {image_path}"
            )
        hashes[sample_id] = _sha256(image_path)
    return hashes


_V2_EXPERIMENT_NAME = "scammark-qwen3-dashscope-server-full-v2"
_V2_OUTPUT_NAMESPACE = "scammark_qwen3_dashscope_server_full_v2"
_FULL_EVENT_MAX_ROWS = 1_000_000
_FULL_EVENT_MAX_ROW_BYTES = 256 * 1024
_FULL_EVENT_MAX_FILE_BYTES = 512 * 1024 * 1024


class _StateConstructionContractError(ExperimentContractError):
    pass


class _RunnerContractDriftStop(ContractDriftStop, ExperimentContractError):
    """Task-10 control stop retaining the legacy contract-error identity."""

    pass


def _runtime_metadata(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 runtime path metadata could not be validated"
        ) from exc


def _reject_unsafe_runtime_metadata(
    path: Path,
    metadata: os.stat_result,
    *,
    output_root: Path,
) -> None:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or file_attributes & reparse_flag:
        raise ExperimentContractError(
            "full-v2 runtime namespace contains a link or reparse point"
        )
    if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
        raise ExperimentContractError(
            "full-v2 runtime namespace contains a hard-linked file"
        )
    if (
        path != output_root
        and stat.S_ISDIR(metadata.st_mode)
        and os.path.ismount(path)
    ):
        raise ExperimentContractError(
            "full-v2 runtime namespace contains a child mount point"
        )


def _validate_runtime_path(output_root: Path, path: Path) -> None:
    try:
        resolved_root = output_root.resolve(strict=False)
        absolute_root = output_root.absolute()
        absolute_path = path.absolute()
        relative = absolute_path.relative_to(absolute_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExperimentContractError(
            "full-v2 runtime path escapes its configured namespace"
        ) from exc
    current = absolute_root
    candidates = [current]
    for part in relative.parts:
        current = current / part
        candidates.append(current)
    for candidate in candidates:
        metadata = _runtime_metadata(candidate)
        if metadata is not None:
            _reject_unsafe_runtime_metadata(
                candidate,
                metadata,
                output_root=absolute_root,
            )
    try:
        resolved_path = path.resolve(strict=False)
        resolved_path.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExperimentContractError(
            "full-v2 runtime path resolves outside its configured namespace"
        ) from exc


def _reject_existing_runtime_symlinks(output_root: Path) -> None:
    _validate_runtime_path(output_root, output_root)
    if not output_root.exists():
        return
    pending = [output_root]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    metadata = _runtime_metadata(path)
                    if metadata is None:
                        continue
                    _reject_unsafe_runtime_metadata(
                        path,
                        metadata,
                        output_root=output_root,
                    )
                    if stat.S_ISDIR(metadata.st_mode):
                        pending.append(path)
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 runtime namespace could not be validated"
        ) from exc


def _validate_open_runtime_fd(fd: int) -> os.stat_result:
    try:
        metadata = os.fstat(fd)
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 runtime file descriptor could not be validated"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ExperimentContractError(
            "full-v2 runtime file descriptor is not a private regular file"
        )
    return metadata


def _runtime_fd_path(fd: int) -> Path:
    path = Path(f"/proc/self/fd/{fd}")
    if not path.exists():
        raise ExperimentContractError(
            "full-v2 runtime descriptor projection is unavailable"
        )
    return path


def _directory_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ExperimentContractError(
            "full-v2 runtime requires no-follow directory opens"
        )
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _validate_open_runtime_dir_fd(fd: int) -> os.stat_result:
    try:
        metadata = os.fstat(fd)
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 runtime directory descriptor could not be validated"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ExperimentContractError(
            "full-v2 runtime directory descriptor is not a directory"
        )
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(metadata, "st_file_attributes", 0) & reparse_flag:
        raise ExperimentContractError(
            "full-v2 runtime directory descriptor is a reparse point"
        )
    return metadata


def _runtime_dir_fd_path(fd: int) -> Path:
    _validate_open_runtime_dir_fd(fd)
    path = Path(f"/proc/self/fd/{fd}")
    if not path.exists():
        raise ExperimentContractError(
            "full-v2 runtime directory projection is unavailable"
        )
    return path


def _validate_pinned_child_dir(
    parent_fd: int,
    name: str,
    child_fd: int,
) -> None:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ExperimentContractError("invalid full-v2 runtime child directory")
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 runtime child directory entry could not be validated"
        ) from exc
    pinned = _validate_open_runtime_dir_fd(child_fd)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISDIR(entry.st_mode)
        or getattr(entry, "st_file_attributes", 0) & reparse_flag
        or (entry.st_dev, entry.st_ino) != (pinned.st_dev, pinned.st_ino)
    ):
        raise ExperimentContractError(
            "full-v2 runtime child directory identity changed"
        )


def _open_runtime_root_dir(output_root: Path) -> int:
    absolute_root = output_root.absolute()
    _validate_runtime_path(absolute_root, absolute_root)
    absolute_root.parent.mkdir(parents=True, exist_ok=True)
    parent_fd: int | None = None
    try:
        parent_fd = os.open(absolute_root.parent, _directory_open_flags())
        try:
            os.mkdir(absolute_root.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        root_fd = os.open(
            absolute_root.name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
        try:
            _validate_pinned_child_dir(parent_fd, absolute_root.name, root_fd)
            _runtime_dir_fd_path(root_fd)
        except BaseException:
            os.close(root_fd)
            raise
        return root_fd
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 runtime root directory could not be securely opened"
        ) from exc
    finally:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _open_runtime_child_dir(
    parent_fd: int,
    name: str,
    *,
    create: bool,
    missing_ok: bool = False,
) -> int | None:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ExperimentContractError("invalid full-v2 runtime child directory")
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ExperimentContractError(
                "full-v2 runtime child directory could not be created"
            ) from exc
    try:
        child_fd = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ExperimentContractError(
            "full-v2 runtime child directory is missing"
        ) from None
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 runtime child directory could not be securely opened"
        ) from exc
    try:
        parent_metadata = _validate_open_runtime_dir_fd(parent_fd)
        child_metadata = _validate_open_runtime_dir_fd(child_fd)
        _validate_pinned_child_dir(parent_fd, name, child_fd)
        if child_metadata.st_dev != parent_metadata.st_dev:
            raise ExperimentContractError(
                "full-v2 runtime child directory crosses a mount boundary"
            )
        _runtime_dir_fd_path(child_fd)
    except BaseException:
        try:
            os.close(child_fd)
        except OSError:
            pass
        raise
    return child_fd


def _open_runtime_file_at(
    directory_fd: int,
    name: str,
    *,
    flags: int,
    pinned_parent_fd: int,
    pinned_directory_name: str,
    logical_output_root: Path,
    logical_path: Path,
    missing_ok: bool = False,
) -> int | None:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ExperimentContractError("invalid full-v2 runtime file name")
    _validate_runtime_path(logical_output_root, logical_path)
    _validate_pinned_child_dir(
        pinned_parent_fd,
        pinned_directory_name,
        directory_fd,
    )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ExperimentContractError(
            "full-v2 runtime requires no-follow file opens"
        )
    secure_flags = flags | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, secure_flags, 0o600, dir_fd=directory_fd)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ExperimentContractError(
            "full-v2 runtime file is missing at secure open"
        ) from None
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 runtime file could not be securely opened"
        ) from exc
    try:
        _validate_open_runtime_dir_fd(directory_fd)
        _validate_open_runtime_fd(fd)
        _runtime_fd_path(fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return fd


class _PinnedRunEventLogger:
    def __init__(self, logger: Any, fd: int) -> None:
        self._logger = logger
        self._fd = fd

    def emit(self, event: str, *, critical: bool = False, **fields: Any) -> None:
        _validate_open_runtime_fd(self._fd)
        self._logger.emit(event, critical=critical, **fields)


def _validate_v2_runtime_namespace(
    config: Mapping[str, Any],
    output_root: Path,
    output_dir: Path,
    *,
    validate_only: bool,
) -> bool:
    experiment = config.get("experiment")
    experiment_name = (
        experiment.get("name") if isinstance(experiment, Mapping) else None
    )
    exact_identity = experiment_name == _V2_EXPERIMENT_NAME
    lexical_v2_root = output_root.name == _V2_OUTPUT_NAMESPACE
    if exact_identity != lexical_v2_root:
        raise ExperimentContractError(
            "full-v2 experiment identity and output namespace must match exactly"
        )
    if not exact_identity:
        if validate_only:
            return False
        raise ExperimentContractError(
            "non-validate runner integration requires the exact full-v2 identity"
        )
    try:
        if output_root.resolve(strict=False).name != _V2_OUTPUT_NAMESPACE:
            raise ExperimentContractError(
                "full-v2 output root resolves to a different namespace"
            )
    except OSError as exc:
        raise ExperimentContractError(
            "full-v2 output root could not be resolved"
        ) from exc
    _validate_runtime_path(output_root, output_dir)
    paths = config.get("paths")
    image_cache = paths.get("image_cache") if isinstance(paths, Mapping) else None
    if type(image_cache) is not str or not image_cache.strip():
        raise ExperimentContractError("full-v2 image cache path is missing")
    _validate_runtime_path(output_root, Path(image_cache))
    _reject_existing_runtime_symlinks(output_root)
    return True


def _count_full_retry_events(
    path: Path,
    *,
    expected_run_mode: str,
) -> int:
    if not path.is_file():
        return 0
    try:
        if path.stat().st_size > _FULL_EVENT_MAX_FILE_BYTES:
            return 0
        seen: set[tuple[str, int]] = set()
        with path.open("rb") as handle:
            for row_index, raw in enumerate(handle, start=1):
                if row_index > _FULL_EVENT_MAX_ROWS:
                    return 0
                if not raw.strip() or len(raw) > _FULL_EVENT_MAX_ROW_BYTES:
                    continue
                try:
                    payload = json.loads(
                        raw,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"invalid JSON constant: {value}")
                        ),
                    )
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValueError,
                    RecursionError,
                    MemoryError,
                ):
                    continue
                if type(payload) is not dict:
                    continue
                safe_rows = _project_events(
                    [payload], expected_run_mode=expected_run_mode
                )
                if not safe_rows:
                    continue
                safe = safe_rows[0]
                if safe.get("event") != "retry_scheduled":
                    continue
                key = (safe["attempt_id"], safe["sequence"])
                if key in seen:
                    continue
                seen.add(key)
        return len(seen)
    except (OSError, OverflowError, ValueError):
        return 0


def _prior_safe_position(
    events: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None, str | None]:
    sample_id: str | None = None
    stage: str | None = None
    stop_reason: str | None = None
    for row in events:
        event = row.get("event")
        if event == "resume_audit":
            prior_sample = row.get("last_safe_sample_id")
            prior_stage = row.get("last_safe_stage")
            if isinstance(prior_sample, str):
                sample_id = prior_sample
            if isinstance(prior_stage, str):
                stage = prior_stage
        elif event in {"stage_completed", "stage_reused"}:
            prior_sample = row.get("sample_id")
            prior_stage = row.get("stage")
            if isinstance(prior_sample, str) and isinstance(prior_stage, str):
                sample_id, stage = prior_sample, prior_stage
        elif event == "run_stopped" and isinstance(row.get("reason"), str):
            stop_reason = str(row["reason"])
    return sample_id, stage, stop_reason


def _breaker_history_success_ids(
    events: Sequence[Mapping[str, Any]],
) -> set[str]:
    recorded_successes: set[str] = set()
    seen_rows: set[tuple[str, int]] = set()
    for row in events:
        if not isinstance(row, Mapping):
            continue
        event = row.get("event")
        attempt_id = row.get("attempt_id")
        sequence = row.get("sequence")
        if (
            type(attempt_id) is str
            and attempt_id
            and type(sequence) is int
            and sequence > 0
        ):
            row_id = (attempt_id, sequence)
            if row_id in seen_rows:
                continue
            seen_rows.add(row_id)
        if event != "sample_complete":
            continue
        sample_id = row.get("sample_id")
        backend_error = row.get("backend_error")
        if (
            type(sample_id) is not str
            or not sample_id.strip()
            or type(backend_error) is not bool
        ):
            continue
        if not backend_error:
            recorded_successes.add(sample_id)
    return recorded_successes


def _checkpoint_binary_labels(
    state: ExperimentState,
    manifest_ids: Sequence[str],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for sample_id in manifest_ids:
        stages = state.load_stages(sample_id)
        binary = stages.get("binary")
        output = binary.get("output") if isinstance(binary, Mapping) else None
        label = output.get("binary") if isinstance(output, Mapping) else None
        if binary and binary.get("status") == "complete" and label in {
            "benign",
            "scam",
        }:
            labels[sample_id] = str(label)
    return labels


def _record_stage_observations(
    tracker: ProgressTracker,
    events: Sequence[Mapping[str, Any]],
    *,
    source: str,
    aggregates: dict[str, dict[str, deque[dict[str, Any]]]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    for row in events:
        if row.get("event") != "stage_completed":
            continue
        stage = row.get("stage")
        model = row.get("model")
        usage = row.get("usage")
        elapsed = row.get("elapsed_seconds")
        if (
            not isinstance(stage, str)
            or not isinstance(model, str)
            or not isinstance(usage, Mapping)
            or not isinstance(elapsed, (int, float))
        ):
            continue
        try:
            tracker.record_stage(
                stage,
                model=model,
                usage=usage,
                elapsed_seconds=float(elapsed),
                source=source,  # type: ignore[arg-type]
            )
        except BaseException:
            pass
        if aggregates is not None and config is not None:
            try:
                _record_safe_stage_aggregate(
                    aggregates,
                    source=source,
                    stage=stage,
                    model=model,
                    usage=usage,
                    elapsed_seconds=elapsed,
                    config=config,
                )
            except BaseException:
                pass


def _new_stage_aggregates(
    history_size: int,
) -> dict[str, dict[str, deque[dict[str, Any]]]]:
    size = max(1, int(history_size))
    return {
        "smoke": {},
        "full": {},
        "_history_size": {"value": deque((size,), maxlen=1)},
    }


def _safe_nonnegative_number(value: Any, *, integer: bool) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 10**12:
        return None
    if integer and not number.is_integer():
        return None
    return number


def _record_safe_stage_aggregate(
    aggregates: dict[str, dict[str, deque[dict[str, Any]]]],
    *,
    source: str,
    stage: str,
    model: str,
    usage: Mapping[str, Any],
    elapsed_seconds: float,
    config: Mapping[str, Any],
) -> None:
    try:
        if source not in {"smoke", "full"} or not stage.strip():
            return
        models = config.get("models")
        pricing = config.get("pricing")
        if not isinstance(models, Mapping) or not isinstance(pricing, Mapping):
            return
        if model == models.get("text"):
            role = "text"
        elif model == models.get("vision"):
            role = "vision"
        else:
            return
        role_pricing = pricing.get(role)
        if not isinstance(role_pricing, Mapping):
            return
        currency = role_pricing.get("currency")
        input_rate = _safe_nonnegative_number(
            role_pricing.get("input_per_million"), integer=False
        )
        output_rate = _safe_nonnegative_number(
            role_pricing.get("output_per_million"), integer=False
        )
        if currency not in {"USD", "CNY"} or input_rate is None or output_rate is None:
            return
        input_value = next(
            (usage[name] for name in ("prompt_tokens", "input_tokens") if name in usage),
            None,
        )
        output_value = next(
            (
                usage[name]
                for name in ("completion_tokens", "output_tokens")
                if name in usage
            ),
            None,
        )
        input_tokens = _safe_nonnegative_number(input_value, integer=True)
        output_tokens = _safe_nonnegative_number(output_value, integer=True)
        elapsed = _safe_nonnegative_number(elapsed_seconds, integer=False)
        if input_tokens is None or output_tokens is None or elapsed is None:
            return
        amount = (
            input_tokens * input_rate + output_tokens * output_rate
        ) / 1_000_000
        if not math.isfinite(amount) or amount < 0:
            return
        size_holder = aggregates.get("_history_size", {}).get("value")
        history_size = (
            int(size_holder[0])
            if isinstance(size_holder, deque) and size_holder
            else 1
        )
        stage_rows = aggregates[source].setdefault(
            stage, deque(maxlen=history_size)
        )
        stage_rows.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_seconds": elapsed,
                "currency": currency,
                "cost": amount,
            }
        )
    except BaseException:
        return


def _selected_stage_means(
    aggregates: Mapping[str, Mapping[str, deque[dict[str, Any]]]],
    *,
    stages: Sequence[str],
    prefer_full: bool,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for stage in stages:
        source = "full" if prefer_full else "smoke"
        rows = aggregates.get(source, {}).get(stage, ())
        if not rows and prefer_full:
            source = "smoke"
            rows = aggregates.get(source, {}).get(stage, ())
        if not rows:
            continue
        count = len(rows)
        currencies = {str(row["currency"]) for row in rows}
        if len(currencies) != 1:
            continue
        selected[stage] = {
            "source": source,
            "observation_count": count,
            "mean_input_tokens": sum(
                float(row["input_tokens"]) for row in rows
            )
            / count,
            "mean_output_tokens": sum(
                float(row["output_tokens"]) for row in rows
            )
            / count,
            "mean_elapsed_seconds": sum(
                float(row["elapsed_seconds"]) for row in rows
            )
            / count,
            "currency": currencies.pop(),
            "mean_cost": sum(float(row["cost"]) for row in rows) / count,
        }
    return selected


def _safe_pricing_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    pricing = config.get("pricing")
    if not isinstance(pricing, Mapping):
        raise ValueError("pricing must be a mapping")
    projected: dict[str, dict[str, Any]] = {}
    for role in ("text", "vision"):
        row = pricing.get(role)
        if not isinstance(row, Mapping):
            raise ValueError(f"pricing.{role} must be a mapping")
        currency = row.get("currency")
        input_rate = _safe_nonnegative_number(
            row.get("input_per_million"), integer=False
        )
        output_rate = _safe_nonnegative_number(
            row.get("output_per_million"), integer=False
        )
        if currency not in {"USD", "CNY"} or input_rate is None or output_rate is None:
            raise ValueError(f"pricing.{role} is invalid")
        projected[role] = {
            "currency": currency,
            "input_per_million": input_rate,
            "output_per_million": output_rate,
        }
    return projected


def _observed_benign_false_positive_rate(
    manifest: Sequence[Mapping[str, Any]],
    binary_outputs: Mapping[str, str],
) -> float | None:
    observed = false_positives = 0
    for row in manifest:
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if row.get("binary") != "benign":
            continue
        label = binary_outputs.get(sample_id)
        if label not in {"benign", "scam"}:
            continue
        observed += 1
        false_positives += label == "scam"
    return false_positives / observed if observed else None


def _safe_pilot_false_positive_rate(
    output_root: Path,
    *,
    metrics_path: Path | None = None,
) -> float | None:
    selected_path = metrics_path or output_root / "pilot_10" / "metrics.json"
    try:
        if not selected_path.is_file() or selected_path.stat().st_size > 16 * 1024 * 1024:
            return None
        raw = selected_path.read_bytes()
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
        binary = payload.get("binary") if type(payload) is dict else None
        confusion = binary.get("confusion") if type(binary) is dict else None
        if type(confusion) is not dict:
            return None
        false_positives = confusion.get("fp")
        true_negatives = confusion.get("tn")
        if (
            type(false_positives) is not int
            or type(true_negatives) is not int
            or not 0 <= false_positives <= 10**12
            or not 0 <= true_negatives <= 10**12
        ):
            return None
        benign = false_positives + true_negatives
        return false_positives / benign if benign else None
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ):
        return None


def _durably_complete_sample_ids(
    manifest: Sequence[Mapping[str, Any]],
    *,
    complete_stages: frozenset[tuple[str, str]],
    binary_outputs: Mapping[str, str],
) -> set[str]:
    completed: set[str] = set()
    base = {"url", "text", "visual", "cross", "binary"}
    downstream = {"type", "grounding"}
    for row in manifest:
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        required = base | (
            downstream
            if row.get("binary") == "scam"
            or binary_outputs.get(sample_id) == "scam"
            else set()
        )
        done = {
            stage
            for current_sample, stage in complete_stages
            if current_sample == sample_id
        }
        if required <= done:
            completed.add(sample_id)
    return completed


class _ProgressActivity:
    def __init__(self) -> None:
        try:
            now = max(0.0, time.monotonic())
        except BaseException:
            now = 0.0
        self._lock = threading.RLock()
        self._current_sample_id: str | None = None
        self._current_stage: str | None = None
        self._last_progress_at = now
        self._last_sample_completed_at = now

    def begin_sample(self, sample_id: str) -> None:
        checked = str(sample_id).strip()
        if not checked:
            return
        now = max(0.0, time.monotonic())
        with self._lock:
            self._current_sample_id = checked
            self._current_stage = None
            self._last_progress_at = now

    def begin_stage(self, sample_id: str, stage: str) -> None:
        checked_sample = str(sample_id).strip()
        checked_stage = str(stage).strip()
        if not checked_sample or not checked_stage:
            return
        now = max(0.0, time.monotonic())
        with self._lock:
            if self._current_sample_id != checked_sample:
                return
            self._current_stage = checked_stage
            self._last_progress_at = now

    def complete_sample(self, sample_id: str) -> None:
        checked = str(sample_id).strip()
        now = max(0.0, time.monotonic())
        with self._lock:
            if checked and self._current_sample_id != checked:
                return
            self._current_sample_id = None
            self._current_stage = None
            self._last_progress_at = now
            self._last_sample_completed_at = now

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_sample_id": self._current_sample_id,
                "current_stage": self._current_stage,
                "last_progress_at": self._last_progress_at,
                "last_sample_completed_at": self._last_sample_completed_at,
            }


def _combined_progress_snapshot(
    tracker: ProgressTracker,
    *,
    remaining_stage_counts: Mapping[str, int],
    durable_completed: int,
    durable_reserved: int,
    durable_outstanding: int,
    durable_retries: int,
    api_call_limit: int,
    activity_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    baselines = {
        "durable_completed": durable_completed,
        "durable_reserved": durable_reserved,
        "durable_outstanding": durable_outstanding,
        "durable_retries": durable_retries,
        "api_call_limit": api_call_limit,
    }
    if any(type(value) is not int or value < 0 for value in baselines.values()):
        raise ValueError("progress baselines must be nonnegative integers")
    current = tracker.snapshot(
        remaining_stage_counts=remaining_stage_counts
    )
    activity = dict(activity_snapshot or {})
    return {
        "completed_samples": durable_completed + current.completed_samples,
        "current_sample_id": activity.get(
            "current_sample_id", current.current_sample_id
        ),
        "current_stage": activity.get("current_stage", current.current_stage),
        "api_calls": durable_reserved + current.api_calls,
        "api_call_limit": api_call_limit,
        "outstanding_calls": (
            durable_outstanding + current.outstanding_calls
        ),
        "retries": durable_retries + current.retries,
        "backend_errors": current.backend_errors,
        "eta_seconds": current.eta_seconds,
        "preliminary_eta_seconds": current.preliminary_eta_seconds,
        "last_progress_at": activity.get(
            "last_progress_at", current.last_progress_at
        ),
        "last_sample_completed_at": activity.get(
            "last_sample_completed_at", current.last_sample_completed_at
        ),
    }


def _execute_experiment(
    *,
    config: dict[str, Any],
    config_path: Path,
    state_contract: dict[str, Any],
    output_root: Path,
    output_dir: Path,
    state_root: Path,
    runtime_root_fd: int,
    runtime_output_dir_fd: int,
    runtime_state_dir_fd: int,
    runtime_state_root: Path,
    runtime_pilot_dir_fd: int | None,
    runtime_image_cache_dir_fd: int,
    runtime_image_cache_name: str,
    runtime_image_cache: Path,
    run_manifest: Path,
    run_evidence: Path,
    pilot_size: int | None,
    full_run: bool,
    resume: bool,
    ack_breaker: bool,
    settings: ObservabilitySettings,
    sink: SafeEventSink,
    prior_events: Sequence[Mapping[str, Any]],
    prior_retry_count: int,
    controller: StopController,
) -> dict[str, Any]:
    client: OpenAIAPIModelClient | None = None
    heartbeat: HeartbeatWorker | None = None
    try:
        _validate_pinned_child_dir(
            runtime_root_fd,
            output_dir.name,
            runtime_output_dir_fd,
        )
        _validate_pinned_child_dir(
            runtime_root_fd,
            state_root.name,
            runtime_state_dir_fd,
        )
        _validate_pinned_child_dir(
            runtime_root_fd,
            runtime_image_cache_name,
            runtime_image_cache_dir_fd,
        )
        _reject_existing_runtime_symlinks(output_root)
        for runtime_path in (
            output_dir,
            output_dir / "manifest.jsonl",
            output_dir / "web_evidence.jsonl",
            state_root,
            state_root / "run_config.json",
            state_root / "checkpoints",
            state_root / "api_call_ledger.jsonl",
            Path(config["paths"]["image_cache"]),
        ):
            _validate_runtime_path(output_root, runtime_path)
        try:
            state = ExperimentState(runtime_state_root, state_contract)
        except _StateConstructionContractError:
            raise
        except ExperimentContractError as exc:
            raise _StateConstructionContractError(
                "state construction contract failed"
            ) from exc
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            raise _StateConstructionContractError(
                "state construction failed"
            ) from exc
        checkpoint_audit = scan_checkpoints(
            state.checkpoints, config_hash=state.config_hash
        )
        if checkpoint_audit.corrupt_files:
            names = sanitize_text(
                ",".join(checkpoint_audit.corrupt_files), limit=200
            )
            raise ExperimentContractError(
                f"corrupt durable checkpoint scan: {names}"
            )
        ledger_audit = scan_call_ledger(
            state.call_ledger,
            config_hash=state.config_hash,
            checkpoint_stages=checkpoint_audit.complete_stages,
        )
        if ledger_audit.corrupt:
            raise ExperimentContractError("corrupt durable API call ledger")

        contract_manifest = load_jsonl(run_manifest)
        manifest_ids = [
            str(row.get("sample_id") or row.get("id") or "")
            for row in contract_manifest
        ]
        binary_outputs = _checkpoint_binary_labels(state, manifest_ids)
        durable_progress = summarize_sample_progress(
            contract_manifest,
            complete_stages=checkpoint_audit.complete_stages,
            error_stages=checkpoint_audit.error_stages,
            binary_outputs=binary_outputs,
        )
        last_sample, last_stage, prior_stop = _prior_safe_position(prior_events)
        reserved_calls = state.total_api_calls()
        sink.emit(
            "resume_audit",
            complete_samples=durable_progress.complete,
            partial_samples=durable_progress.partial,
            error_samples=durable_progress.error,
            complete_stages=len(checkpoint_audit.complete_stages),
            error_stages=len(checkpoint_audit.error_stages),
            reserved=reserved_calls,
            finished=ledger_audit.finished,
            outstanding=ledger_audit.outstanding,
            outstanding_by_stage=dict(ledger_audit.outstanding_by_stage),
            replay_required_by_stage=dict(
                ledger_audit.replay_required_by_stage
            ),
            remaining_call_budget=max(
                0,
                int(config["experiment"]["api_call_limit"])
                - reserved_calls,
            ),
            last_safe_sample_id=last_sample,
            last_safe_stage=last_stage,
            previous_stop_reason=prior_stop,
            ledger_corrupt=False,
        )

        logical_predictions_path = output_dir / "predictions.jsonl"
        predictions_path = _runtime_dir_fd_path(
            runtime_output_dir_fd
        ) / "predictions.jsonl"
        loaded_predictions = []
        if resume:
            predictions_fd = _open_runtime_file_at(
                runtime_output_dir_fd,
                "predictions.jsonl",
                flags=os.O_RDONLY,
                pinned_parent_fd=runtime_root_fd,
                pinned_directory_name=output_dir.name,
                logical_output_root=output_root,
                logical_path=logical_predictions_path,
                missing_ok=True,
            )
            if predictions_fd is not None:
                try:
                    _validate_open_runtime_fd(predictions_fd)
                    try:
                        loaded_predictions = load_jsonl(
                            _runtime_fd_path(predictions_fd)
                        )
                    except (
                        UnicodeError,
                        json.JSONDecodeError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise ExperimentContractError(
                            "resume predictions could not be parsed"
                        ) from exc
                finally:
                    try:
                        os.close(predictions_fd)
                    except OSError:
                        pass
        existing_predictions = tuple(
            _merge_prediction_records(
                loaded_predictions,
                [],
                manifest_ids,
            )
        )

        tracker = ProgressTracker(
            settings,
            total_samples=len(contract_manifest),
            api_call_limit=int(config["experiment"]["api_call_limit"]),
            pricing=config["pricing"],
            text_model=str(config["models"]["text"]),
            vision_model=str(config["models"]["vision"]),
        )
        activity = _ProgressActivity()
        stage_aggregates = _new_stage_aggregates(settings.eta_history_size)
        smoke_events_path = output_root / "pilot_10" / "run_events.jsonl"
        smoke_metrics_path = output_root / "pilot_10" / "metrics.json"
        smoke_events_fd = (
            _open_runtime_file_at(
                runtime_pilot_dir_fd,
                "run_events.jsonl",
                flags=os.O_RDONLY,
                pinned_parent_fd=runtime_root_fd,
                pinned_directory_name="pilot_10",
                logical_output_root=output_root,
                logical_path=smoke_events_path,
                missing_ok=True,
            )
            if runtime_pilot_dir_fd is not None
            else None
        )
        if smoke_events_fd is None:
            smoke_rows = []
        else:
            try:
                _validate_open_runtime_fd(smoke_events_fd)
                smoke_rows, _ = read_jsonl_tail(
                    _runtime_fd_path(smoke_events_fd)
                )
            finally:
                try:
                    os.close(smoke_events_fd)
                except OSError:
                    pass
        smoke_events = _project_events(smoke_rows, expected_run_mode="pilot")
        _record_stage_observations(
            tracker,
            smoke_events,
            source="smoke",
            aggregates=stage_aggregates,
            config=config,
        )
        smoke_metrics_fd = (
            _open_runtime_file_at(
                runtime_pilot_dir_fd,
                "metrics.json",
                flags=os.O_RDONLY,
                pinned_parent_fd=runtime_root_fd,
                pinned_directory_name="pilot_10",
                logical_output_root=output_root,
                logical_path=smoke_metrics_path,
                missing_ok=True,
            )
            if runtime_pilot_dir_fd is not None
            else None
        )
        if smoke_metrics_fd is None:
            smoke_false_positive_rate = None
        else:
            try:
                _validate_open_runtime_fd(smoke_metrics_fd)
                smoke_false_positive_rate = _safe_pilot_false_positive_rate(
                    output_root,
                    metrics_path=_runtime_fd_path(smoke_metrics_fd),
                )
            finally:
                try:
                    os.close(smoke_metrics_fd)
                except OSError:
                    pass
        _record_stage_observations(
            tracker,
            prior_events,
            source="full",
            aggregates=stage_aggregates,
            config=config,
        )
        summary_scheduler = SummaryScheduler(
            settings, started_at=time.monotonic()
        )
        breaker = FailureCircuitBreaker.from_events(
            prior_events,
            threshold=settings.breaker_consecutive_backend_errors,
        )
        recorded_breaker_success_ids = _breaker_history_success_ids(
            prior_events
        )
        durable_complete_ids = _durably_complete_sample_ids(
            contract_manifest,
            complete_stages=checkpoint_audit.complete_stages,
            binary_outputs=binary_outputs,
        )
        telemetry_complete_stages = set(checkpoint_audit.complete_stages)
        telemetry_error_stages = set(checkpoint_audit.error_stages)
        telemetry_binary_outputs = dict(binary_outputs)
        initial_false_positive_rate = _observed_benign_false_positive_rate(
            contract_manifest, binary_outputs
        )
        remaining_stage_counts_holder = {
            "value": project_remaining_stage_counts(
                contract_manifest,
                complete_stages=checkpoint_audit.complete_stages,
                binary_outputs=binary_outputs,
                observed_benign_false_positive_rate=(
                    initial_false_positive_rate
                ),
                smoke_benign_false_positive_rate=(
                    smoke_false_positive_rate
                    if smoke_false_positive_rate is not None
                    else 1.0
                ),
            )
        }

        def forward_event(event: str, payload: dict[str, Any]) -> None:
            try:
                sample_id = str(payload.get("sample_id") or "")
                stage = str(payload.get("stage") or "")
            except BaseException:
                sample_id = ""
                stage = ""
            if event in {"stage_started", "stage_reused"}:
                try:
                    activity.begin_stage(sample_id, stage)
                except BaseException:
                    pass
            try:
                if event == "stage_started":
                    tracker.begin_stage(stage)
                elif event == "stage_completed":
                    tracker.record_stage(
                        stage,
                        model=str(payload.get("model") or ""),
                        usage=(
                            payload.get("usage")
                            if isinstance(payload.get("usage"), Mapping)
                            else {}
                        ),
                        elapsed_seconds=float(
                            payload.get("elapsed_seconds") or 0.0
                        ),
                    )
                elif event == "api_call_reserved":
                    tracker.reserve_call(stage)
                elif event == "api_call_finished":
                    tracker.finish_call(
                        stage,
                        transport_error=payload.get("status")
                        == "transport_error",
                    )
                elif event == "retry_scheduled":
                    tracker.record_retry()
            except BaseException:
                pass
            if event == "stage_completed":
                try:
                    _record_safe_stage_aggregate(
                        stage_aggregates,
                        source="full",
                        stage=stage,
                        model=str(payload.get("model") or ""),
                        usage=(
                            payload.get("usage")
                            if isinstance(payload.get("usage"), Mapping)
                            else {}
                        ),
                        elapsed_seconds=float(
                            payload.get("elapsed_seconds") or 0.0
                        ),
                        config=config,
                    )
                except BaseException:
                    pass
            try:
                key = (sample_id, stage)
                if sample_id and stage and event == "stage_completed":
                    telemetry_complete_stages.add(key)
                    telemetry_error_stages.discard(key)
                elif sample_id and stage and event == "stage_failed":
                    telemetry_error_stages.add(key)
                    telemetry_complete_stages.discard(key)
            except BaseException:
                pass
            try:
                sink.emit(event, **dict(payload))
            except BaseException:
                pass

        retry = config["retry"]
        timeouts = config["timeouts"]
        client = OpenAIAPIModelClient(
            model=str(config["models"]["text"]),
            text_model=str(config["models"]["text"]),
            vision_model=str(config["models"]["vision"]),
            text_base_url=str(config["endpoints"]["text_base_url"]),
            vision_base_url=str(config["endpoints"]["vision_base_url"]),
            strict_routing=True,
            allow_env_image_override=False,
            request_policy=RequestPolicy(
                max_attempts=int(retry["max_attempts"]),
                connect_timeout_seconds=float(timeouts["connect_seconds"]),
                llm_read_timeout_seconds=float(timeouts["llm_read_seconds"]),
                vlm_read_timeout_seconds=float(timeouts["vlm_read_seconds"]),
                backoff_seconds=tuple(
                    float(value) for value in retry["backoff_seconds"]
                ),
                full_jitter=bool(retry["full_jitter"]),
            ),
            max_api_calls=int(config["experiment"]["api_call_limit"]),
            call_ledger=state,
            event_callback=forward_event,
        )
        image = config["images"]
        pipeline = ScamMarkPipeline(
            model_client=client,
            prompt_mode=str(config["experiment"]["prompt_mode"]),
            experiment_state=state,
            input_token_limits={
                key: int(value) for key, value in config["input_tokens"].items()
            },
            output_token_limits={
                key: int(value) for key, value in config["output_tokens"].items()
            },
            image_cache_dir=runtime_image_cache,
            image_policy=ImagePolicy(
                overview_pixels=int(image["overview_pixels"]),
                total_pixels=int(image["total_pixels"]),
                max_long_side=int(image["max_long_side"]),
                max_tiles=int(image["max_tiles"]),
                overlap_ratio=float(image["overlap_ratio"]),
                max_base64_bytes=int(image["max_base64_bytes"]),
                contact_sheet_pixels=int(image["contact_sheet_pixels"]),
                contact_sheet_base64_bytes=int(
                    image["contact_sheet_base64_bytes"]
                ),
            ),
            max_stage_attempts_across_resume=int(
                config["experiment"]["max_stage_attempts_across_resume"]
            ),
            event_callback=forward_event,
            stop_controller=controller,
        )
        heartbeat = HeartbeatWorker(
            settings,
            snapshot=lambda: _combined_progress_snapshot(
                tracker,
                remaining_stage_counts=remaining_stage_counts_holder["value"],
                durable_completed=durable_progress.complete,
                durable_reserved=reserved_calls,
                durable_outstanding=ledger_audit.outstanding,
                durable_retries=prior_retry_count,
                api_call_limit=int(config["experiment"]["api_call_limit"]),
                activity_snapshot=activity.snapshot(),
            ),
            emit=sink.emit,
            api_call_limit=int(config["experiment"]["api_call_limit"]),
        )
        try:
            heartbeat.start()
        except BaseException:
            pass
        samples = load_samples(run_manifest, run_evidence)
        records: list[dict[str, Any]] = []
        prediction_flush_interval = int(
            config["experiment"].get("prediction_flush_interval", 1)
        )

        def flush_predictions(reason: str) -> list[dict[str, Any]]:
            _validate_runtime_path(output_root, logical_predictions_path)
            _validate_pinned_child_dir(
                runtime_root_fd,
                output_dir.name,
                runtime_output_dir_fd,
            )
            merged = _merge_prediction_records(
                existing_predictions,
                records,
                manifest_ids,
            )
            _atomic_write_jsonl(predictions_path, merged)
            sink.emit(
                "prediction_flushed",
                record_count=len(merged),
                path_sha256=_sha256(predictions_path),
                reason=reason,
            )
            return merged

        def flush_and_emit_control_stop(exc: RunControlError) -> None:
            if isinstance(exc, GracefulStopRequested):
                sink.emit(
                    "stop_requested",
                    critical=True,
                    signum=exc.signum,
                    boundary=exc.boundary,
                    message="graceful stop requested",
                )
            flush_predictions("stop")
            sink.emit(
                "run_stopped",
                critical=True,
                **_control_stop_fields(exc),
            )

        try:
            breaker.require_start_allowed(acknowledged=ack_breaker)
        except BreakerAcknowledgementRequired as exc:
            flush_and_emit_control_stop(exc)
            raise
        if breaker.opened:
            acknowledgement = breaker.acknowledge(
                reason="operator explicitly acknowledged prior breaker"
            )
            sink.emit(
                "breaker_reset_ack",
                critical=True,
                reason=acknowledgement["reason"],
                prior_consecutive=acknowledgement[
                    "prior_consecutive"
                ],
            )
        if (
            int(config["experiment"]["api_call_limit"])
            - state.total_api_calls()
            <= 0
        ):
            control = ApiBudgetStop(
                "physical API call budget exhausted"
            )
            flush_and_emit_control_stop(control)
            raise control

        def emit_post_prediction_telemetry(
            *,
            sample_index: int,
            sample_id: str,
            prediction: Mapping[str, Any],
            backend_error: bool,
            breaker_outcome: bool | None,
        ) -> None:
            try:
                current_stages = state.load_stages(sample_id)
                binary_stage = current_stages.get("binary")
                binary_output = (
                    binary_stage.get("output")
                    if isinstance(binary_stage, Mapping)
                    else None
                )
                binary_label = (
                    binary_output.get("binary")
                    if isinstance(binary_output, Mapping)
                    else None
                )
                if binary_label in {"benign", "scam"}:
                    telemetry_binary_outputs[sample_id] = str(binary_label)
                observed_false_positive_rate = (
                    _observed_benign_false_positive_rate(
                        contract_manifest,
                        telemetry_binary_outputs,
                    )
                )
                remaining_stage_counts = project_remaining_stage_counts(
                    contract_manifest,
                    complete_stages=frozenset(telemetry_complete_stages),
                    binary_outputs=telemetry_binary_outputs,
                    observed_benign_false_positive_rate=(
                        observed_false_positive_rate
                    ),
                    smoke_benign_false_positive_rate=(
                        smoke_false_positive_rate
                        if smoke_false_positive_rate is not None
                        else 1.0
                    ),
                )
                remaining_stage_counts_holder["value"] = (
                    remaining_stage_counts
                )
                tracker_snapshot = _combined_progress_snapshot(
                    tracker,
                    remaining_stage_counts=remaining_stage_counts,
                    durable_completed=durable_progress.complete,
                    durable_reserved=reserved_calls,
                    durable_outstanding=ledger_audit.outstanding,
                    durable_retries=prior_retry_count,
                    api_call_limit=int(
                        config["experiment"]["api_call_limit"]
                    ),
                    activity_snapshot=activity.snapshot(),
                )
                elapsed_for_progress = max(
                    0.0, time.monotonic() - started_monotonic
                )
                sample_fields = {
                    "sample_index": sample_index,
                    "sample_total": len(samples),
                    "sample_id": sample_id,
                    "result": str(
                        prediction.get("binary") or INVALID_BINARY
                    ),
                    "api_calls": tracker_snapshot["api_calls"],
                    "api_call_limit": int(
                        config["experiment"]["api_call_limit"]
                    ),
                    "elapsed_seconds": elapsed_for_progress,
                    "eta_seconds": tracker_snapshot["eta_seconds"],
                    "preliminary_eta_seconds": tracker_snapshot[
                        "preliminary_eta_seconds"
                    ],
                    "health": (
                        "warming_up"
                        if tracker_snapshot["eta_seconds"] is None
                        else "ok"
                    ),
                    "retry_count": tracker_snapshot["retries"],
                    "breaker_consecutive": breaker.consecutive,
                }
                if breaker_outcome is not None:
                    sample_fields["backend_error"] = breaker_outcome
                sink.emit(
                    "sample_complete",
                    **sample_fields,
                )
                if not summary_scheduler.should_emit(
                    completed_samples=sample_index,
                    now=time.monotonic(),
                ):
                    return
                current_ledger_audit = scan_call_ledger(
                    state.call_ledger,
                    config_hash=state.config_hash,
                    checkpoint_stages=frozenset(
                        telemetry_complete_stages
                    ),
                )
                if current_ledger_audit.corrupt:
                    return
                raw_remaining_cost = tracker.estimate_remaining_cost(
                    remaining_stage_counts,
                    outstanding_by_stage=(
                        current_ledger_audit.outstanding_by_stage
                    ),
                    replay_required_by_stage=(
                        current_ledger_audit.replay_required_by_stage
                    ),
                )
                remaining_cost = {
                    currency: float(raw_remaining_cost.get(currency, 0.0))
                    for currency in ("USD", "CNY")
                }
                completed_for_summary = (
                    durable_progress.complete + new_successful_samples
                )
                rate = (
                    new_successful_samples * 60.0 / elapsed_for_progress
                    if elapsed_for_progress > 0
                    else 0.0
                )
                requested_stages = sorted(
                    stage
                    for stage in (
                        set(remaining_stage_counts)
                        | set(current_ledger_audit.outstanding_by_stage)
                        | set(current_ledger_audit.replay_required_by_stage)
                    )
                    if (
                        remaining_stage_counts.get(stage, 0)
                        + current_ledger_audit.outstanding_by_stage.get(
                            stage, 0
                        )
                        + current_ledger_audit.replay_required_by_stage.get(
                            stage, 0
                        )
                    )
                    > 0
                )
                stage_means = _selected_stage_means(
                    stage_aggregates,
                    stages=requested_stages,
                    prefer_full=(
                        new_successful_samples
                        >= settings.cost_full_update_samples
                    ),
                )
                sink.emit(
                    "summary",
                    completed_samples=completed_for_summary,
                    sample_total=len(samples),
                    rate_per_minute=rate,
                    eta_seconds=tracker_snapshot["eta_seconds"],
                    preliminary_eta_seconds=tracker_snapshot[
                        "preliminary_eta_seconds"
                    ],
                    remaining_cost_est=remaining_cost,
                    cost_calculation={
                        "stage_means": stage_means,
                        "population": {
                            "manifest_samples": len(contract_manifest),
                            "remaining_stage_counts": dict(
                                remaining_stage_counts
                            ),
                            "known_binary_outputs": len(
                                telemetry_binary_outputs
                            ),
                            "observed_benign_false_positive_rate": (
                                observed_false_positive_rate
                            ),
                        },
                        "pricing": _safe_pricing_inputs(config),
                        "retry_assumptions": {
                            "current_retry_count": tracker_snapshot[
                                "retries"
                            ],
                            "usage_included_in_stage_means": True,
                        },
                        "outstanding_by_stage": dict(
                            current_ledger_audit.outstanding_by_stage
                        ),
                        "replay_required_by_stage": dict(
                            current_ledger_audit.replay_required_by_stage
                        ),
                        "calculated_at_process_seconds": (
                            elapsed_for_progress
                        ),
                    },
                    health=(
                        "warming_up"
                        if tracker_snapshot["eta_seconds"] is None
                        else "ok"
                    ),
                )
            except BaseException:
                return

        start = time.perf_counter()
        started_monotonic = time.monotonic()
        new_successful_samples = 0
        for sample_index, sample in enumerate(samples, start=1):
            sample_id = str(sample.get("sample_id") or sample.get("id") or "")
            track_new_work = sample_id not in durable_complete_ids
            try:
                activity.begin_sample(sample_id)
            except BaseException:
                pass
            if track_new_work:
                try:
                    tracker.begin_sample(sample_id, index=sample_index)
                except BaseException:
                    pass
            sink.emit(
                "sample_started",
                sample_index=sample_index,
                sample_total=len(samples),
                sample_id=sample_id,
            )
            try:
                prediction, oracle_diagnostic = _run_primary_and_oracle_diagnostic(
                    pipeline,
                    sample,
                    enabled=bool(config["experiment"]["forced_diagnostics"]),
                )
            except Exception as exc:  # noqa: BLE001 - preserve partial checkpoints for resume.
                control = _control_error_for_inference_failure(exc)
                if control is not None:
                    flush_and_emit_control_stop(control)
                    if control is exc:
                        raise
                    raise control from exc
                prediction = _partial_prediction(sample, state, exc)
                oracle_diagnostic = {"status": "not_run_primary_error"}
            prediction = normalize_final_prediction(prediction)
            records.append(
                _prediction_record(sample, prediction, oracle_diagnostic)
            )
            backend_error = bool(prediction.get("backend_error"))
            breaker_opened = False
            breaker_outcome: bool | None = None
            if track_new_work:
                try:
                    tracker.complete_sample(
                        sample_id,
                        index=sample_index,
                        backend_error=backend_error,
                    )
                    if not backend_error:
                        new_successful_samples += 1
                except BaseException:
                    pass
            if (
                track_new_work
                or not backend_error
                and sample_id not in recorded_breaker_success_ids
            ):
                breaker_outcome = backend_error
                if backend_error:
                    breaker_opened = breaker.record_sample(
                        sample_id=sample_id,
                        backend_error=True,
                    )
                else:
                    breaker.record_sample(
                        sample_id=sample_id,
                        backend_error=False,
                    )
                    recorded_breaker_success_ids.add(sample_id)

            try:
                activity.complete_sample(sample_id)
            except BaseException:
                pass
            emit_post_prediction_telemetry(
                sample_index=sample_index,
                sample_id=sample_id,
                prediction=prediction,
                backend_error=backend_error,
                breaker_outcome=breaker_outcome,
            )
            if breaker_opened:
                flush_predictions("breaker")
                sink.emit(
                    "breaker_opened",
                    critical=True,
                    consecutive=breaker.consecutive,
                    sample_id=sample_id,
                    message="consecutive backend error threshold reached",
                )
                control = CircuitBreakerStop(
                    "consecutive backend error threshold reached"
                )
                sink.emit(
                    "run_stopped",
                    critical=True,
                    **_control_stop_fields(control),
                )
                raise control
            if _should_flush_predictions(
                sample_index,
                total=len(samples),
                interval=prediction_flush_interval,
            ):
                flush_predictions(
                    "final" if sample_index == len(samples) else "periodic"
                )
        elapsed = time.perf_counter() - start

        engineering = summarize_engineering(
            records,
            client=client,
            state=state,
            config=config,
            elapsed_seconds=elapsed,
            pilot_size=pilot_size,
        )
        bertscore_fn = None
        if config["experiment"].get("with_bertscore"):
            bertscore = config["bertscore"]
            bertscore_fn = get_bertscore_evaluator(
                BERTScoreConfig(
                    model=str(default_model_dir()),
                    revision=str(bertscore["model_revision"]),
                    num_layers=int(bertscore["num_layers"]),
                    batch_size=int(bertscore["batch_size"]),
                    device=str(bertscore["device"]),
                    idf=bool(bertscore["idf"]),
                    rescale_with_baseline=bool(
                        bertscore["rescale_with_baseline"]
                    ),
                )
            )
        metrics = evaluate_prediction_records(
            records,
            engineering=engineering,
            bootstrap_samples=int(config["experiment"]["bootstrap_samples"]),
            bertscore_fn=bertscore_fn,
        )
        gates = _acceptance_gates(metrics, records, config, pilot_size)
        metrics["engineering"]["acceptance_gates"] = gates
        runtime_output_dir = _runtime_dir_fd_path(runtime_output_dir_fd)
        logical_metrics_path = output_dir / "metrics.json"
        logical_human_candidates_path = (
            output_dir / "human_evaluation_candidates.csv"
        )
        summary_name = "full_summary.md" if full_run else "pilot_summary.md"
        logical_summary_path = output_dir / summary_name
        metrics_path = runtime_output_dir / "metrics.json"
        human_candidates_path = (
            runtime_output_dir / "human_evaluation_candidates.csv"
        )
        summary_path = runtime_output_dir / summary_name
        for artifact_path in (
            logical_metrics_path,
            logical_human_candidates_path,
            logical_summary_path,
        ):
            _validate_runtime_path(output_root, artifact_path)
        _validate_pinned_child_dir(
            runtime_root_fd,
            output_dir.name,
            runtime_output_dir_fd,
        )
        _atomic_write_json(metrics_path, metrics)
        sink.emit(
            "metrics_written",
            sample_count=len(records),
            gates_passed=bool(gates.get("passed")),
            path_sha256=_sha256(metrics_path),
        )
        human_fd = _open_runtime_file_at(
            runtime_output_dir_fd,
            "human_evaluation_candidates.csv",
            flags=os.O_WRONLY | os.O_CREAT,
            pinned_parent_fd=runtime_root_fd,
            pinned_directory_name=output_dir.name,
            logical_output_root=output_root,
            logical_path=logical_human_candidates_path,
        )
        assert human_fd is not None
        try:
            _validate_open_runtime_fd(human_fd)
            _write_human_candidates(_runtime_fd_path(human_fd), records)
        finally:
            try:
                os.close(human_fd)
            except OSError:
                pass
        _write_pilot_summary(summary_path, metrics, gates)
        sink.emit(
            "run_completed",
            sample_count=len(records),
            api_calls=state.total_api_calls(),
            gates_passed=bool(gates.get("passed")),
            message="run artifacts are durable",
        )
        return {"output_dir": str(output_dir), "metrics": metrics, "gates": gates}
    finally:
        if heartbeat is not None:
            try:
                heartbeat.close()
            except BaseException:
                pass
        if client is not None:
            try:
                client.close()
            except BaseException:
                pass


def run(
    config_path: Path,
    pilot_size: int | None,
    resume: bool,
    *,
    validate_only: bool = False,
    ack_breaker: bool = False,
) -> dict[str, Any]:
    try:
        config = load_experiment_config(config_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _RunnerContractDriftStop(
            "experiment configuration validation failed"
        ) from exc
    settings: ObservabilitySettings | None = None
    try:
        full_run = pilot_size is None
        if full_run and not bool(
            config["experiment"].get("allow_full_run")
        ):
            raise _RunnerContractDriftStop(
                "configuration does not authorize a full experiment run"
            )
        output_root = Path(config["paths"]["output_root"])
        output_dir = output_root / (
            "full" if full_run else f"pilot_{pilot_size}"
        )
        state_root = output_root / (
            "state_full" if full_run else "state"
        )
        runtime_image_cache = Path(config["paths"]["image_cache"])
        runtime_image_cache_name = runtime_image_cache.name
        exact_v2 = _validate_v2_runtime_namespace(
            config,
            output_root,
            output_dir,
            validate_only=validate_only,
        )
        if not validate_only:
            if not exact_v2:
                raise ExperimentContractError(
                    "non-validate runs require the exact full-v2 identity"
                )
            settings = ObservabilitySettings.from_mapping(
                config.get("observability")
            )
    except _RunnerContractDriftStop:
        raise
    except (ExperimentContractError, OSError, KeyError, TypeError, ValueError) as exc:
        raise _RunnerContractDriftStop(
            "experiment bootstrap contract validation failed"
        ) from exc

    state_existed_before_pin = state_root.exists()
    sink: SafeEventSink | None = None
    prior_events: list[dict[str, Any]] = []
    prior_retry_count = 0
    controller: StopController | None = None
    event_fd: int | None = None
    runtime_root_fd: int | None = None
    runtime_output_dir_fd: int | None = None
    runtime_state_dir_fd: int | None = None
    runtime_pilot_dir_fd: int | None = None
    runtime_image_cache_dir_fd: int | None = None
    runtime_output_dir = output_dir
    runtime_state_root = state_root
    state_contract_built = False

    try:
        if exact_v2:
            runtime_root_fd = _open_runtime_root_dir(output_root)
            runtime_output_dir_fd = _open_runtime_child_dir(
                runtime_root_fd,
                output_dir.name,
                create=True,
            )
            assert runtime_output_dir_fd is not None
            runtime_output_dir = _runtime_dir_fd_path(
                runtime_output_dir_fd
            )
        if not validate_only:
            assert runtime_root_fd is not None
            runtime_pilot_dir_fd = _open_runtime_child_dir(
                runtime_root_fd,
                "pilot_10",
                create=False,
                missing_ok=True,
            )
            image_cache = Path(config["paths"]["image_cache"])
            try:
                runtime_image_cache_name = image_cache.absolute().relative_to(
                    output_root.absolute()
                ).as_posix()
            except ValueError as exc:
                raise ExperimentContractError(
                    "full-v2 image cache is outside the pinned runtime root"
                ) from exc
            runtime_image_cache_dir_fd = _open_runtime_child_dir(
                runtime_root_fd,
                runtime_image_cache_name,
                create=True,
            )
            assert runtime_image_cache_dir_fd is not None
            runtime_image_cache = _runtime_dir_fd_path(
                runtime_image_cache_dir_fd
            )
    except BaseException:
        for fd in (
            runtime_image_cache_dir_fd,
            runtime_pilot_dir_fd,
            runtime_state_dir_fd,
            runtime_output_dir_fd,
            runtime_root_fd,
        ):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise

    if not validate_only:
        events_path = output_dir / "run_events.jsonl"
        assert runtime_root_fd is not None
        assert runtime_output_dir_fd is not None
        try:
            event_fd = _open_runtime_file_at(
                runtime_output_dir_fd,
                "run_events.jsonl",
                flags=os.O_RDWR | os.O_APPEND | os.O_CREAT,
                pinned_parent_fd=runtime_root_fd,
                pinned_directory_name=output_dir.name,
                logical_output_root=output_root,
                logical_path=events_path,
            )
            assert event_fd is not None
            event_fd_path = _runtime_fd_path(event_fd)
            _validate_open_runtime_fd(event_fd)
            prior_rows, _ = read_jsonl_tail(event_fd_path)
            prior_events = _project_events(
                prior_rows,
                expected_run_mode="full" if full_run else "pilot",
            )
            prior_retry_count = _count_full_retry_events(
                event_fd_path,
                expected_run_mode="full" if full_run else "pilot",
            )
            logger = _PinnedRunEventLogger(
                RunEventLogger(
                    event_fd_path,
                    settings,
                    stream=sys.stdout,
                    experiment_version="full-v2",
                    run_mode="full" if full_run else "pilot",
                ),
                event_fd,
            )
            sink = SafeEventSink(logger, fallback_stream=None)
            sink.emit(
                "run_started",
                config_sha256=_sha256(Path(config_path)),
                output_dir=str(output_dir),
                resume=resume,
                ack_breaker=ack_breaker,
            )
            controller = StopController()
        except BaseException:
            if event_fd is not None:
                try:
                    os.close(event_fd)
                except OSError:
                    pass
            event_fd = None
            for fd in (
                runtime_image_cache_dir_fd,
                runtime_pilot_dir_fd,
                runtime_state_dir_fd,
                runtime_output_dir_fd,
                runtime_root_fd,
            ):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            raise

    try:
        if controller is not None:
            controller.install()

        manifest_path = Path(config["data"]["manifest"])
        evidence_path = Path(config["data"]["web_evidence"])
        version_path = Path(config["data"]["dataset_version"])
        for path in (manifest_path, evidence_path, version_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Frozen experiment asset is missing: {path}"
                )
        if len(
            {
                manifest_path.parent.resolve(),
                evidence_path.parent.resolve(),
                version_path.parent.resolve(),
            }
        ) != 1:
            raise ValueError(
                "Frozen manifest, evidence, and dataset version must share one directory"
            )
        frozen_asset_verification = verify_frozen_assets(
            manifest_path.parent
        )
        if state_existed_before_pin and not resume:
            raise FileExistsError(
                "Experiment state exists; pass --resume to reuse verified checkpoints"
            )
        path_mapper = DatasetPathMapper(
            source_root=str(config["data"]["source_dataset_root"]),
            runtime_root=str(config["data"]["runtime_dataset_root"]),
        )
        if exact_v2:
            assert runtime_root_fd is not None
            assert runtime_output_dir_fd is not None
            _validate_pinned_child_dir(
                runtime_root_fd,
                output_dir.name,
                runtime_output_dir_fd,
            )
        if full_run:
            run_manifest, run_evidence, contract_records = _select_full_inputs(
                manifest_path,
                evidence_path,
                runtime_output_dir,
                expected_size=int(
                    config["experiment"]["expected_full_sample_count"]
                ),
                path_mapper=path_mapper,
            )
        else:
            run_manifest, run_evidence, _ = _select_pilot_inputs(
                manifest_path,
                evidence_path,
                runtime_output_dir,
                int(pilot_size),
                int(config["experiment"]["seed"]),
                path_mapper=path_mapper,
            )
            contract_records = select_pilot_records(
                path_mapper.relocate_records(load_jsonl(manifest_path)),
                size=60,
                seed=int(config["experiment"]["seed"]),
            )
        state_contract = {
            "config": sanitized_contract_config(config),
            "run_mode": "full" if full_run else "pilot",
            "dataset_path_mapping": path_mapper.metadata,
            "frozen_asset_verification": frozen_asset_verification,
            "source_sha256": {
                "manifest": _sha256(manifest_path),
                "web_evidence": _sha256(evidence_path),
                "dataset_version": _sha256(version_path),
                "config": _sha256(config_path),
            },
            "implementation_sha256": _implementation_source_hashes(),
            "contract_sample_ids": [
                str(record.get("sample_id")) for record in contract_records
            ],
            "selected_image_sha256": _selected_image_hashes(
                contract_records
            ),
            "taxonomy": list(CANONICAL_LABELS),
        }
        state_contract_built = True
        if validate_only:
            preflight = {
                "status": "validated",
                "run_mode": "full" if full_run else "pilot",
                "selected_sample_count": len(load_jsonl(run_manifest)),
                "contract_image_count": len(
                    state_contract["selected_image_sha256"]
                ),
                "expected_full_sample_count": config["experiment"].get(
                    "expected_full_sample_count"
                ),
                "api_call_limit": int(
                    config["experiment"]["api_call_limit"]
                ),
                "text_model": str(config["models"]["text"]),
                "visual_model": str(config["models"]["vision"]),
                "text_endpoint_host": endpoint_identity(
                    str(config["endpoints"]["text_base_url"])
                )["host"],
                "text_endpoint_sha256": endpoint_identity(
                    str(config["endpoints"]["text_base_url"])
                )["sha256"],
                "vision_endpoint_host": endpoint_identity(
                    str(config["endpoints"]["vision_base_url"])
                )["host"],
                "vision_endpoint_sha256": endpoint_identity(
                    str(config["endpoints"]["vision_base_url"])
                )["sha256"],
                "source_sha256": state_contract["source_sha256"],
                "frozen_asset_verification": frozen_asset_verification,
                "implementation_sha256": state_contract[
                    "implementation_sha256"
                ],
            }
            preflight_path = runtime_output_dir / "preflight.json"
            if exact_v2:
                _reject_existing_runtime_symlinks(output_root)
                assert runtime_root_fd is not None
                assert runtime_output_dir_fd is not None
                _validate_pinned_child_dir(
                    runtime_root_fd,
                    output_dir.name,
                    runtime_output_dir_fd,
                )
                _validate_runtime_path(
                    output_root, output_dir / "preflight.json"
                )
            _atomic_write_json(preflight_path, preflight)
            return {"output_dir": str(output_dir), "preflight": preflight}

        assert settings is not None and sink is not None and controller is not None
        _reject_existing_runtime_symlinks(output_root)
        assert runtime_root_fd is not None
        runtime_state_dir_fd = _open_runtime_child_dir(
            runtime_root_fd,
            state_root.name,
            create=True,
        )
        assert runtime_state_dir_fd is not None
        assert runtime_image_cache_dir_fd is not None
        runtime_state_root = _runtime_dir_fd_path(runtime_state_dir_fd)
        return _execute_experiment(
            config=config,
            config_path=Path(config_path),
            state_contract=state_contract,
            output_root=output_root,
            output_dir=output_dir,
            state_root=state_root,
            runtime_root_fd=runtime_root_fd,
            runtime_output_dir_fd=runtime_output_dir_fd,
            runtime_state_dir_fd=runtime_state_dir_fd,
            runtime_state_root=runtime_state_root,
            runtime_pilot_dir_fd=runtime_pilot_dir_fd,
            runtime_image_cache_dir_fd=runtime_image_cache_dir_fd,
            runtime_image_cache_name=runtime_image_cache_name,
            runtime_image_cache=runtime_image_cache,
            run_manifest=run_manifest,
            run_evidence=run_evidence,
            pilot_size=pilot_size,
            full_run=full_run,
            resume=resume,
            ack_breaker=ack_breaker,
            settings=settings,
            sink=sink,
            prior_events=prior_events,
            prior_retry_count=prior_retry_count,
            controller=controller,
        )
    except ExperimentContractError as exc:
        control = _RunnerContractDriftStop(
            "experiment contract validation failed"
        )
        if sink is not None:
            fields = _control_stop_fields(control)
            fields["reason"] = (
                "state_construction_failed"
                if isinstance(exc, _StateConstructionContractError)
                else "contract_validation_failed"
            )
            fields["boundary"] = "bootstrap"
            sink.emit(
                "run_stopped",
                critical=True,
                **fields,
            )
        raise control from exc
    except (OSError, ValueError) as exc:
        if state_contract_built:
            raise
        control = _RunnerContractDriftStop(
            "experiment contract validation failed"
        )
        if sink is not None:
            fields = _control_stop_fields(control)
            fields["reason"] = "contract_validation_failed"
            fields["boundary"] = "bootstrap"
            sink.emit(
                "run_stopped",
                critical=True,
                **fields,
            )
        raise control from exc
    finally:
        if controller is not None:
            try:
                controller.close()
            except BaseException:
                pass
        if event_fd is not None:
            try:
                os.close(event_fd)
            except OSError:
                pass
        for fd in (
            runtime_image_cache_dir_fd,
            runtime_pilot_dir_fd,
            runtime_state_dir_fd,
            runtime_output_dir_fd,
            runtime_root_fd,
        ):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed ScamMark Qwen3 pilot or full experiment.")
    parser.add_argument("--config", default="configs/qwen3_dashscope.toml")
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--pilot", type=int, choices=[10, 60])
    run_group.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ack-breaker",
        action="store_true",
        help="Acknowledge a prior open circuit breaker before resuming.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen inputs and hashes without making API calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(
            Path(args.config),
            None if args.full else args.pilot,
            args.resume,
            validate_only=args.validate_only,
            ack_breaker=args.ack_breaker,
        )
    except GracefulStopRequested as exc:
        message = str(_control_stop_fields(exc)["message"])
        print(f"{type(exc).__name__}: {message}", file=sys.stderr)
        return exc.exit_code
    except RunControlError as exc:
        message = str(_control_stop_fields(exc)["message"])
        print(
            f"{sanitize_text(type(exc).__name__, limit=100)}: "
            f"{sanitize_text(message, limit=500)}",
            file=sys.stderr,
        )
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - CLI maps unexpected failures to exit 1.
        classification = classify_failure(exc)
        print(
            f"{sanitize_text(type(exc).__name__, limit=100)}: "
            f"{sanitize_text(classification.message, limit=500)}",
            file=sys.stderr,
        )
        if classification.systemic and classification.exit_code is not None:
            return classification.exit_code
        return 1
    if args.validate_only:
        print(json.dumps({"output_dir": result["output_dir"], "preflight": result["preflight"]}, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps({"output_dir": result["output_dir"], "gates": result["gates"]}, ensure_ascii=False, indent=2))
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
