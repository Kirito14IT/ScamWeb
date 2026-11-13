"""ScamMark pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable

from .consistency_analysis import analyze_consistency
from .experiment_state import (
    CheckpointContractError,
    ExperimentState,
    project_request_metadata,
)
from .image_preprocessing import (
    ImageBundle,
    ImagePolicy,
    assets_for_stage,
    normalize_grounding_boxes,
    prepare_image_bundle,
)
from .input_budget import build_budgeted_stage_prompt
from .mhtml_policy import apply_mhtml_policy
from .model_client import ModelClient, RuleBasedModelClient, selected_model_ids
from .observability import (
    EVENT_FIELDS,
    RunControlError,
    StopController,
    classify_failure,
    sanitize_text,
)
from .prediction_schema import normalize_binary_label
from .prompts import PROMPT_STAGES, build_stage_prompt, should_include_image
from .text_analysis import analyze_textual_persuasion
from .taxonomy import INVALID_LABEL, canonicalize_scam_type
from .url_analysis import analyze_url
from .visual_analysis import analyze_visual_layout


SYSTEM_PROMPT_TOKEN_RESERVE = 64


def _request_fingerprint(
    *,
    client: ModelClient,
    stage: str,
    prompt: str,
    max_output_tokens: int | None,
    image_assets: list[Any] | None,
    include_image: bool,
) -> tuple[str, str]:
    models = selected_model_ids(client)
    model = models["visual_model" if include_image else "text_model"]
    endpoint = getattr(client, "vision_base_url" if include_image else "text_base_url", None)
    image_hashes = []
    for asset in image_assets or []:
        path = Path(str(getattr(asset, "path", "") or ""))
        if path.is_file():
            image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    payload = {
        "stage": stage,
        "model": model,
        "endpoint": str(endpoint or ""),
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
        "image_sha256": image_hashes,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return fingerprint, model


_PIPELINE_STAGE_EVENTS = frozenset(
    {"stage_started", "stage_reused", "stage_completed", "stage_failed"}
)
_STAGE_EVENT_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    }
)
_MAX_STAGE_EVENT_ELAPSED_SECONDS = 366.0 * 24.0 * 60.0 * 60.0
_MAX_STAGE_EVENT_TOKEN_COUNT = 1_000_000_000_000
_MAX_STAGE_EVENT_ATTEMPT_COUNT = 1_000_000
_MAX_DURABLE_STAGE_COUNT = 1_000_000_000_000
_MAX_FAILURE_METADATA_DEPTH = 8
_MAX_FAILURE_METADATA_NODES = 100_000
_MAX_FAILURE_METADATA_CONTAINER_ITEMS = 1_000
_MAX_FAILURE_METADATA_STRING_LENGTH = 1_000
_STAGE_FAILURE_CATEGORIES = frozenset(
    {
        "auth_billing",
        "api_budget",
        "contract_drift",
        "transient",
        "sample_permanent",
        "unexpected",
    }
)
_GENERIC_STAGE_FAILURE_MESSAGES = frozenset(
    {
        "authentication or billing failure",
        "physical API call budget exhausted",
        "experiment contract drift",
        "transient provider or transport failure",
        "sample-specific permanent failure",
        "unexpected failure",
    }
)


@dataclass
class _StageCheckpointSnapshot:
    stages: dict[str, Any] | None = None


@dataclass
class _ImageBundleSnapshot:
    prepared: bool = False
    bundle: ImageBundle | None = None


class _UnsafeFailureMetadata(ValueError):
    """Raised internally when failure metadata cannot be projected safely."""


def _safe_stage_event_text(value: Any, *, limit: int = 200) -> str:
    try:
        text = sanitize_text(value, limit=limit)
    except BaseException:
        return ""
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    if "://" in text:
        return "<redacted-field>"
    return text[:limit]


def _safe_stage_event_host(value: Any) -> str:
    text = _safe_stage_event_text(value, limit=255).lower().rstrip(".")
    if re.fullmatch(r"(?:\*\.)?[a-z0-9.-]+(?::[0-9]{1,5})?", text):
        return text
    return ""


def _safe_stage_event_digest(value: Any) -> str:
    text = _safe_stage_event_text(value, limit=64)
    return text.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", text) else ""


def _safe_stage_event_number(value: Any, *, maximum: float) -> int | float | None:
    if type(value) is int:
        return value if 0 <= value <= maximum else None
    if type(value) is not float:
        return None
    if not math.isfinite(value) or not 0.0 <= value <= maximum:
        return None
    return value


def _safe_stage_event_usage(value: Any) -> dict[str, int]:
    if type(value) is not dict:
        return {}
    try:
        return {
            name: item
            for name, item in value.items()
            if type(name) is str
            and name in _STAGE_EVENT_USAGE_FIELDS
            and type(item) is int
            and 0 <= item <= _MAX_STAGE_EVENT_TOKEN_COUNT
        }
    except BaseException:
        return {}


def _safe_stage_failure_message(value: Any) -> str:
    text = _safe_stage_event_text(value, limit=200)
    if text in _GENERIC_STAGE_FAILURE_MESSAGES:
        return text
    matched = re.fullmatch(r"HTTP ([1-5][0-9]{2}): (.+)", text)
    if matched is not None and matched.group(2) in _GENERIC_STAGE_FAILURE_MESSAGES:
        return text
    return "<redacted-field>" if text else ""


def _project_pipeline_stage_event(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    if event not in _PIPELINE_STAGE_EVENTS or type(payload) is not dict:
        return {}
    allowed = EVENT_FIELDS[event]
    projected: dict[str, Any] = {}
    for name in ("sample_id", "stage", "model", "provider"):
        if name in allowed and name in payload:
            value = _safe_stage_event_text(payload[name])
            if value:
                projected[name] = value
    if "endpoint_host" in allowed and "endpoint_host" in payload:
        host = _safe_stage_event_host(payload["endpoint_host"])
        if host:
            projected["endpoint_host"] = host
    if "endpoint_sha256" in allowed and "endpoint_sha256" in payload:
        digest = _safe_stage_event_digest(payload["endpoint_sha256"])
        if digest:
            projected["endpoint_sha256"] = digest
    if "elapsed_seconds" in allowed and "elapsed_seconds" in payload:
        elapsed = _safe_stage_event_number(
            payload["elapsed_seconds"],
            maximum=_MAX_STAGE_EVENT_ELAPSED_SECONDS,
        )
        if elapsed is not None:
            projected["elapsed_seconds"] = elapsed
    if "usage" in allowed and "usage" in payload:
        projected["usage"] = _safe_stage_event_usage(payload["usage"])
    if "attempt_count" in allowed and "attempt_count" in payload:
        attempts = _safe_stage_event_number(
            payload["attempt_count"],
            maximum=_MAX_STAGE_EVENT_ATTEMPT_COUNT,
        )
        if type(attempts) is int:
            projected["attempt_count"] = attempts
    if (
        "error_category" in allowed
        and payload.get("error_category") in _STAGE_FAILURE_CATEGORIES
    ):
        projected["error_category"] = payload["error_category"]
    if "error_type" in allowed and "error_type" in payload:
        error_type = _safe_stage_event_text(payload["error_type"], limit=100)
        projected["error_type"] = (
            error_type
            if error_type in {"TransientRequestError", "PermanentRequestError", "Exception"}
            else "Exception"
        )
    if "message" in allowed and "message" in payload:
        message = _safe_stage_failure_message(payload["message"])
        if message:
            projected["message"] = message
    return projected


def _bounded_stage_elapsed(started_at: float | None) -> float:
    if started_at is None:
        return 0.0
    try:
        elapsed = time.perf_counter() - started_at
        if type(elapsed) not in {int, float} or not math.isfinite(elapsed):
            return 0.0
        return min(max(0.0, float(elapsed)), _MAX_STAGE_EVENT_ELAPSED_SECONDS)
    except BaseException:
        return 0.0


def _shallow_failure_counts(metadata: dict[str, Any]) -> tuple[int, int]:
    current_attempts = 1
    current_api_calls = 1
    attempt_seen = False
    api_call_seen = False
    try:
        oversized = len(metadata) > _MAX_FAILURE_METADATA_CONTAINER_ITEMS
        for key, value in islice(
            metadata.items(),
            _MAX_FAILURE_METADATA_CONTAINER_ITEMS,
        ):
            if type(key) is not str:
                continue
            if key == "attempt_count":
                attempt_seen = True
                if type(value) is int and value >= 0:
                    current_attempts = max(
                        1,
                        min(value, _MAX_DURABLE_STAGE_COUNT),
                    )
            elif key == "api_call_count":
                api_call_seen = True
                if type(value) is int and value >= 0:
                    current_api_calls = min(value, _MAX_DURABLE_STAGE_COUNT)
            if attempt_seen and api_call_seen:
                break
    except BaseException:
        return _MAX_DURABLE_STAGE_COUNT, _MAX_DURABLE_STAGE_COUNT
    if oversized:
        if not attempt_seen:
            current_attempts = _MAX_DURABLE_STAGE_COUNT
        if not api_call_seen:
            current_api_calls = _MAX_DURABLE_STAGE_COUNT
    return current_attempts, current_api_calls


def _saturating_failure_count(prior: int, current: int) -> int:
    if type(prior) is not int or prior < 0:
        return _MAX_DURABLE_STAGE_COUNT
    return min(
        min(prior, _MAX_DURABLE_STAGE_COUNT) + current,
        _MAX_DURABLE_STAGE_COUNT,
    )


def _safe_failure_fingerprint(value: Any) -> str:
    if type(value) is not str or len(value) != 64:
        return ""
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""


def _consume_failure_metadata_node(nodes: list[int]) -> None:
    nodes[0] += 1
    if nodes[0] > _MAX_FAILURE_METADATA_NODES:
        raise _UnsafeFailureMetadata("failure metadata exceeds node limit")


def _json_safe_failure_metadata_value(
    value: Any,
    *,
    depth: int,
    nodes: list[int],
) -> Any:
    if depth > _MAX_FAILURE_METADATA_DEPTH:
        raise _UnsafeFailureMetadata("failure metadata exceeds depth limit")
    _consume_failure_metadata_node(nodes)

    if value is None:
        return None
    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is str:
        if len(value) > _MAX_FAILURE_METADATA_STRING_LENGTH:
            raise _UnsafeFailureMetadata("failure metadata string exceeds limit")
        return value
    if value_type is int:
        if value < 0:
            raise _UnsafeFailureMetadata("failure metadata integer is negative")
        return min(value, _MAX_DURABLE_STAGE_COUNT)
    if value_type is float:
        if not math.isfinite(value) or value < 0.0:
            raise _UnsafeFailureMetadata("failure metadata float is unsafe")
        return min(value, float(_MAX_DURABLE_STAGE_COUNT))
    if value_type is dict:
        if len(value) > _MAX_FAILURE_METADATA_CONTAINER_ITEMS:
            raise _UnsafeFailureMetadata("failure metadata mapping exceeds limit")
        projected: dict[str, Any] = {}
        for key, item in value.items():
            _consume_failure_metadata_node(nodes)
            if type(key) is not str or len(key) > _MAX_FAILURE_METADATA_STRING_LENGTH:
                raise _UnsafeFailureMetadata("failure metadata key is unsafe")
            projected[key] = _json_safe_failure_metadata_value(
                item,
                depth=depth + 1,
                nodes=nodes,
            )
        return projected
    if value_type in {list, tuple}:
        if len(value) > _MAX_FAILURE_METADATA_CONTAINER_ITEMS:
            raise _UnsafeFailureMetadata("failure metadata sequence exceeds limit")
        return [
            _json_safe_failure_metadata_value(
                item,
                depth=depth + 1,
                nodes=nodes,
            )
            for item in value
        ]
    raise _UnsafeFailureMetadata("failure metadata contains an unsafe value")


def _minimal_failure_request_metadata(
    *,
    request_fingerprint: str,
    current_attempts: int,
    current_api_calls: int,
    prior_attempts: int,
    prior_api_calls: int,
) -> dict[str, Any]:
    return {
        "request_fingerprint": _safe_failure_fingerprint(request_fingerprint),
        "attempt_count": current_attempts,
        "api_call_count": current_api_calls,
        "cumulative_attempt_count": _saturating_failure_count(
            prior_attempts,
            current_attempts,
        ),
        "cumulative_api_call_count": _saturating_failure_count(
            prior_api_calls,
            current_api_calls,
        ),
    }


def _project_failure_request_metadata(
    error: Exception,
    *,
    request_fingerprint: str,
    prior_attempts: int,
    prior_api_calls: int,
) -> dict[str, Any]:
    fallback = _minimal_failure_request_metadata(
        request_fingerprint=request_fingerprint,
        current_attempts=1,
        current_api_calls=1,
        prior_attempts=prior_attempts,
        prior_api_calls=prior_api_calls,
    )
    try:
        raw_metadata = getattr(error, "request_metadata", None)
    except BaseException:
        return fallback
    if raw_metadata is None:
        raw_metadata = {}
    elif type(raw_metadata) is not dict:
        return fallback
    current_attempts, current_api_calls = _shallow_failure_counts(raw_metadata)
    fallback = _minimal_failure_request_metadata(
        request_fingerprint=request_fingerprint,
        current_attempts=current_attempts,
        current_api_calls=current_api_calls,
        prior_attempts=prior_attempts,
        prior_api_calls=prior_api_calls,
    )
    try:
        failure_metadata = _json_safe_failure_metadata_value(
            raw_metadata,
            depth=0,
            nodes=[0],
        )
        if type(failure_metadata) is not dict:
            raise _UnsafeFailureMetadata("raw failure metadata root is not a mapping")
    except BaseException:
        return fallback

    failure_metadata.update(fallback)
    try:
        candidate = _json_safe_failure_metadata_value(
            project_request_metadata(failure_metadata),
            depth=0,
            nodes=[0],
        )
        if type(candidate) is not dict:
            raise _UnsafeFailureMetadata("failure metadata root is not a mapping")
        candidate.update(fallback)
        json.dumps(
            candidate,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return candidate
    except BaseException:
        return fallback


def _evidence_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for item in value.values():
            text = _evidence_to_text(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            text = _evidence_to_text(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    return str(value)


class ScamMarkPipeline:
    def __init__(
        self,
        model_client: ModelClient | None = None,
        prompt_mode: str = "staged",
        force_downstream: bool = False,
        experiment_state: ExperimentState | None = None,
        input_token_limits: dict[str, int] | None = None,
        output_token_limits: dict[str, int] | None = None,
        image_cache_dir: str | Path | None = None,
        image_policy: ImagePolicy | None = None,
        max_stage_attempts_across_resume: int = 6,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        stop_controller: StopController | None = None,
    ):
        self.model_client = model_client or RuleBasedModelClient()
        self.prompt_mode = prompt_mode
        self.force_downstream = force_downstream
        self.experiment_state = experiment_state
        self.input_token_limits = dict(input_token_limits or {})
        self.output_token_limits = dict(output_token_limits or {})
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None
        self.image_policy = image_policy or ImagePolicy()
        self.max_stage_attempts_across_resume = max_stage_attempts_across_resume
        self.event_callback = event_callback
        self.stop_controller = stop_controller
        if self.prompt_mode not in {"heuristic", "staged"}:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    def _raise_if_requested(self, boundary: str) -> None:
        if self.stop_controller is not None:
            self.stop_controller.raise_if_requested(boundary)

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        callback = self.event_callback
        if callback is None:
            return
        try:
            callback(event, _project_pipeline_stage_event(event, payload))
        except BaseException:
            pass

    def _stage_timer_start(self) -> float | None:
        if self.event_callback is None:
            return None
        try:
            return time.perf_counter()
        except BaseException:
            return None

    def _emit_stage_completed(
        self,
        *,
        sample_id: str,
        stage: str,
        model: str,
        request_metadata: dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        if self.event_callback is None:
            return
        try:
            payload = {
                "sample_id": sample_id,
                "stage": stage,
                "model": request_metadata.get("model") or model,
                "provider": request_metadata.get("provider"),
                "endpoint_host": request_metadata.get("endpoint_host"),
                "endpoint_sha256": request_metadata.get("endpoint_sha256"),
                "elapsed_seconds": elapsed_seconds,
                "usage": request_metadata.get("usage", {}),
                "attempt_count": request_metadata.get("attempt_count"),
            }
        except BaseException:
            return
        self._emit("stage_completed", payload)

    def _emit_stage_failed(
        self,
        *,
        sample_id: str,
        stage: str,
        model: str,
        request_metadata: dict[str, Any],
        elapsed_seconds: float,
        error: Exception,
    ) -> None:
        if self.event_callback is None:
            return
        try:
            classification = classify_failure(error)
            payload = {
                "sample_id": sample_id,
                "stage": stage,
                "model": request_metadata.get("model") or model,
                "provider": request_metadata.get("provider"),
                "endpoint_host": request_metadata.get("endpoint_host"),
                "endpoint_sha256": request_metadata.get("endpoint_sha256"),
                "elapsed_seconds": elapsed_seconds,
                "error_category": classification.category,
                "error_type": type(error).__name__,
                "message": classification.message,
            }
        except BaseException:
            return
        self._emit("stage_failed", payload)

    def predict(self, sample: dict, *, force_downstream: bool | None = None) -> dict:
        self._raise_if_requested("before_sample")
        if self.prompt_mode == "staged":
            return self._predict_staged(sample, force_downstream=force_downstream)
        return self._predict_heuristic(sample)

    def _predict_heuristic(self, sample: dict) -> dict:
        model_sample = apply_mhtml_policy(sample)
        analyses = self._heuristic_analyses(model_sample)
        prediction = self.model_client.predict(model_sample, analyses)
        prediction["evidence_used"] = analyses
        prediction["sample_id"] = str(sample.get("sample_id") or sample.get("id") or "")
        return prediction

    @staticmethod
    def _bundle_prompt_context(bundle: ImageBundle, stage: str) -> list[dict[str, Any]]:
        if bundle.contact_sheet is not None and stage in {"visual", "grounding"}:
            panels = {
                int(panel["image_index"]): panel
                for panel in bundle.contact_sheet_panels
                if isinstance(panel, dict) and "image_index" in panel
            }
            return [
                {
                    "image_index": asset.index,
                    "kind": asset.kind,
                    "crop_xyxy_in_original": asset.crop_xyxy,
                    "original_size": asset.source_size,
                    "submitted_size": [asset.width, asset.height],
                    "submission_mode": "contact_sheet",
                    "contact_sheet_size": [bundle.contact_sheet.width, bundle.contact_sheet.height],
                    "contact_sheet_panel_xyxy": list(
                        (panels.get(asset.index) or {}).get("content_xyxy") or []
                    ),
                }
                for asset in bundle.assets
            ]
        return [
            {
                "image_index": asset.index,
                "kind": asset.kind,
                "crop_xyxy_in_original": asset.crop_xyxy,
                "original_size": asset.source_size,
                "submitted_size": [asset.width, asset.height],
                "data_url_size": asset.data_url_size,
            }
            for asset in assets_for_stage(bundle, stage)
        ]

    @staticmethod
    def _submitted_asset_context(assets: list[Any] | None) -> list[dict[str, Any]]:
        return [
            {
                "submitted_index": asset.index,
                "kind": asset.kind,
                "submitted_size": [asset.width, asset.height],
                "data_url_size": asset.data_url_size,
                "logical_indexes": list(asset.logical_indexes or [asset.index]),
            }
            for asset in (assets or [])
        ]

    def _prepare_bundle(self, sample: dict) -> ImageBundle | None:
        image_path = sample.get("image_path")
        if not self.image_cache_dir or not image_path:
            return None
        return prepare_image_bundle(image_path, self.image_cache_dir, self.image_policy)

    def _execute_stage(
        self,
        *,
        stage: str,
        sample: dict,
        previous: dict[str, Any],
        checkpoint_snapshot: _StageCheckpointSnapshot,
        bundle_snapshot: _ImageBundleSnapshot,
        sample_id: str,
    ) -> tuple[dict, dict[str, Any]]:
        self._raise_if_requested(f"before_stage:{stage}")
        if checkpoint_snapshot.stages is None:
            checkpoint_snapshot.stages = (
                self.experiment_state.load_stages(sample_id)
                if self.experiment_state
                else {}
            )
        loaded = checkpoint_snapshot.stages
        checkpoint = loaded.get(stage) if isinstance(loaded.get(stage), dict) else None
        if not bundle_snapshot.prepared:
            bundle_snapshot.bundle = self._prepare_bundle(sample)
            bundle_snapshot.prepared = True
        bundle = bundle_snapshot.bundle
        include_image = should_include_image(stage)
        if bundle and include_image:
            previous["image_inputs"] = self._bundle_prompt_context(bundle, stage)
        input_limit = self.input_token_limits.get(stage)
        if input_limit:
            user_prompt_limit = max(128, input_limit - SYSTEM_PROMPT_TOKEN_RESERVE)
            prompt, prompt_metadata = build_budgeted_stage_prompt(
                stage,
                sample,
                previous,
                max_tokens=user_prompt_limit,
            )
            prompt_metadata["configured_total_input_tokens"] = input_limit
            prompt_metadata["system_prompt_reserve"] = SYSTEM_PROMPT_TOKEN_RESERVE
        else:
            prompt = build_stage_prompt(stage, sample, previous)
            prompt_metadata = {"truncated": False}

        selected_assets = assets_for_stage(bundle, stage) if bundle and include_image else None
        request_fingerprint, model = _request_fingerprint(
            client=self.model_client,
            stage=stage,
            prompt=prompt,
            max_output_tokens=self.output_token_limits.get(stage),
            image_assets=selected_assets,
            include_image=include_image,
        )
        prior_metadata = dict(checkpoint.get("request_metadata") or {}) if checkpoint else {}
        if checkpoint and prior_metadata.get("request_fingerprint") != request_fingerprint:
            raise CheckpointContractError(
                f"Checkpoint request fingerprint mismatch for sample {sample_id}, stage {stage}"
            )
        if (
            checkpoint
            and checkpoint.get("status") != "error"
            and isinstance(checkpoint.get("output"), dict)
        ):
            self._emit(
                "stage_reused",
                {"sample_id": sample_id, "stage": stage, "model": model},
            )
            self._raise_if_requested(f"after_stage:{stage}")
            return dict(checkpoint["output"]), prior_metadata
        prior_attempts = int(
            prior_metadata.get("cumulative_attempt_count")
            or prior_metadata.get("attempt_count")
            or 0
        )
        prior_api_calls = int(
            prior_metadata.get("cumulative_api_call_count")
            or prior_metadata.get("api_call_count")
            or 0
        )
        remaining_attempts = self.max_stage_attempts_across_resume - prior_attempts
        if remaining_attempts <= 0:
            raise RuntimeError(
                f"Stage {stage} exhausted its cross-resume attempt budget "
                f"({self.max_stage_attempts_across_resume})"
            )
        self._raise_if_requested(f"before_request:{stage}")
        self._emit(
            "stage_started",
            {"sample_id": sample_id, "stage": stage, "model": model},
        )
        request_started_at = self._stage_timer_start()
        try:
            if hasattr(self.model_client, "complete_json_with_metadata"):
                output, request_metadata = self.model_client.complete_json_with_metadata(
                    stage=stage,
                    prompt=prompt,
                    sample=sample,
                    include_image=include_image,
                    max_output_tokens=self.output_token_limits.get(stage),
                    image_assets=selected_assets,
                    max_attempts=min(4, remaining_attempts),
                    max_total_calls=remaining_attempts,
                )
            else:
                output = self.model_client.complete_json(
                    stage=stage,
                    prompt=prompt,
                    sample=sample,
                    include_image=include_image,
                )
                request_metadata = {"attempt_count": 1, "api_call_count": 0}
        except RunControlError:
            raise
        except Exception as exc:
            request_elapsed = _bounded_stage_elapsed(request_started_at)
            failure_metadata = _project_failure_request_metadata(
                exc,
                request_fingerprint=request_fingerprint,
                prior_attempts=prior_attempts,
                prior_api_calls=prior_api_calls,
            )
            if self.experiment_state:
                self.experiment_state.save_stage_error(
                    sample_id,
                    stage,
                    exc,
                    failure_metadata,
                )
            self._emit_stage_failed(
                sample_id=sample_id,
                stage=stage,
                model=model,
                request_metadata=failure_metadata,
                elapsed_seconds=request_elapsed,
                error=exc,
            )
            raise

        request_elapsed = _bounded_stage_elapsed(request_started_at)
        if stage == "grounding" and bundle:
            output = normalize_grounding_boxes(output, bundle)
        request_metadata = project_request_metadata({
            **request_metadata,
            "request_fingerprint": request_fingerprint,
            "cumulative_attempt_count": prior_attempts + int(
                request_metadata.get("attempt_count") or 1
            ),
            "cumulative_api_call_count": prior_api_calls + int(
                request_metadata.get("api_call_count")
                or request_metadata.get("attempt_count")
                or 1
            ),
            "prompt": prompt_metadata,
            "image": {
                "asset_count": len(selected_assets or []),
                "total_pixels": sum(asset.width * asset.height for asset in (selected_assets or [])),
                "total_base64_bytes": sum(asset.data_url_size for asset in (selected_assets or [])),
                "assets": self._submitted_asset_context(selected_assets),
                "logical_inputs": (
                    self._bundle_prompt_context(bundle, stage) if bundle and include_image else []
                ),
            },
        })
        if self.experiment_state:
            self.experiment_state.save_stage(sample_id, stage, output, request_metadata)
            loaded[stage] = {"output": output, "request_metadata": request_metadata}
        self._emit_stage_completed(
            sample_id=sample_id,
            stage=stage,
            model=model,
            request_metadata=request_metadata,
            elapsed_seconds=request_elapsed,
        )
        self._raise_if_requested(f"after_stage:{stage}")
        return output, request_metadata

    def _predict_staged(self, sample: dict, *, force_downstream: bool | None = None) -> dict:
        if not any(
            hasattr(self.model_client, method)
            for method in ("complete_json", "complete_json_with_metadata")
        ):
            raise RuntimeError("staged prompt mode requires a staged JSON model client.")

        runtime_force_downstream = self.force_downstream if force_downstream is None else force_downstream
        model_sample = apply_mhtml_policy(sample)
        heuristic = self._heuristic_analyses(model_sample)
        sample_id = str(sample.get("sample_id") or sample.get("id") or "")
        checkpoint_snapshot = _StageCheckpointSnapshot()
        bundle_snapshot = _ImageBundleSnapshot()
        stages: dict[str, dict] = {}
        request_metadata: dict[str, dict] = {}
        for stage in PROMPT_STAGES[:5]:
            previous = {
                "heuristic": heuristic,
                "prompt_stages": stages,
            }
            stages[stage], request_metadata[stage] = self._execute_stage(
                stage=stage,
                sample=model_sample,
                previous=previous,
                checkpoint_snapshot=checkpoint_snapshot,
                bundle_snapshot=bundle_snapshot,
                sample_id=sample_id,
            )

        binary_result = stages.get("binary") or {}
        predicted_binary = normalize_binary_label(binary_result.get("binary"))

        if predicted_binary == "scam" or runtime_force_downstream:
            for stage in PROMPT_STAGES[5:]:
                previous = {
                    "heuristic": heuristic,
                    "prompt_stages": stages,
                    "force_downstream": runtime_force_downstream,
                }
                stages[stage], request_metadata[stage] = self._execute_stage(
                    stage=stage,
                    sample=model_sample,
                    previous=previous,
                    checkpoint_snapshot=checkpoint_snapshot,
                    bundle_snapshot=bundle_snapshot,
                    sample_id=sample_id,
                )

        type_result = stages.get("type") or {}
        grounding_result = stages.get("grounding") or {}

        predicted_type = canonicalize_scam_type(type_result.get("scam_type") or INVALID_LABEL)
        if predicted_binary != "scam":
            predicted_type = "none"

        boxes = grounding_result.get("boxes") if isinstance(grounding_result.get("boxes"), list) else []
        if predicted_binary != "scam":
            boxes = []

        reasoning_parts = [
            str(binary_result.get("reasoning") or "").strip(),
            str(type_result.get("reasoning") or "").strip(),
            str(grounding_result.get("reasoning") or "").strip(),
        ]
        prediction = {
            "binary": predicted_binary,
            "scam_type": predicted_type,
            "description": grounding_result.get("description") or model_sample.get("title") or model_sample.get("url") or "",
            "reasoning": " ".join(part for part in reasoning_parts if part),
            "boxes": boxes,
            "evidence_used": {
                "heuristic": heuristic,
                "prompt_stages": stages,
                "request_metadata": request_metadata,
            },
            "sample_id": sample_id,
        }
        return prediction

    def _heuristic_analyses(self, sample: dict) -> dict:
        evidence_text = "\n".join(
            str(part or "")
            for part in [
                sample.get("title"),
                sample.get("text"),
                sample.get("visible_text"),
                _evidence_to_text(sample.get("buttons")),
                _evidence_to_text(sample.get("forms")),
            ]
        ).strip()

        url_result = analyze_url(str(sample.get("url") or ""))
        text_result = analyze_textual_persuasion(title=str(sample.get("title") or ""), text=evidence_text)
        visual_result = analyze_visual_layout(sample.get("image_path"))
        consistency_result = analyze_consistency(url_result, text_result, visual_result)

        analyses = {
            "url": url_result,
            "text": text_result,
            "visual": visual_result,
            "consistency": consistency_result,
        }
        return analyses
