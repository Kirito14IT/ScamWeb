"""Prediction backends for ScamMark.

The default `RuleBasedModelClient` is intentionally small and CPU-only so it can
run on this laptop. API and local/server VLM clients share the same interface so
future full-dataset runs can switch backends without changing the pipeline.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
import urllib.error
import urllib.request
import base64
import mimetypes
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit

from .reliability import (
    PermanentRequestError,
    RequestPolicy,
    TransientRequestError,
    _safe_request_error_details,
    execute_with_retry,
    is_retryable_http_status,
    provider_429_failure_category,
)
from .observability import sanitize_text

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_QWEN_VL_MODEL = "qwen3-vl-flash"

_MODEL_EVENT_FIELDS = {
    "api_call_reserved": frozenset(
        {
            "sample_id",
            "stage",
            "call_id",
            "model",
            "provider",
            "endpoint_host",
            "endpoint_sha256",
            "attempt",
            "api_calls",
            "api_call_limit",
        }
    ),
    "api_call_finished": frozenset(
        {
            "sample_id",
            "stage",
            "call_id",
            "model",
            "provider",
            "endpoint_host",
            "endpoint_sha256",
            "attempt",
            "status",
            "elapsed_seconds",
            "usage",
            "error_category",
            "error_type",
            "message",
        }
    ),
    "retry_scheduled": frozenset(
        {
            "sample_id",
            "stage",
            "model",
            "provider",
            "endpoint_host",
            "endpoint_sha256",
            "attempt",
            "next_attempt",
            "delay_seconds",
            "error_type",
            "message",
        }
    ),
}
_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    }
)
_FAILURE_CATEGORIES = frozenset(
    {
        "auth_billing",
        "api_budget",
        "contract_drift",
        "transient",
        "sample_permanent",
        "unexpected",
    }
)
_GENERIC_FAILURE_MESSAGES = frozenset(
    {
        "authentication or billing failure",
        "physical API call budget exhausted",
        "experiment contract drift",
        "transient provider or transport failure",
        "sample-specific permanent failure",
        "unexpected failure",
    }
)
_MAX_EVENT_ELAPSED_SECONDS = 366.0 * 24.0 * 60.0 * 60.0
_MAX_EVENT_TOKEN_COUNT = 1_000_000_000_000
_MAX_EVENT_ATTEMPT = 1_000_000
_MAX_EVENT_API_CALLS = 1_000_000_000_000


def _safe_event_text(value: Any, *, limit: int = 200) -> str:
    try:
        text = sanitize_text(value, limit=limit)
    except BaseException:
        return ""
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    if "://" in text:
        return "<redacted-field>"
    return text[:limit]


def _safe_event_call_id(value: Any) -> str:
    text = _safe_event_text(value, limit=200)
    if re.fullmatch(r"(?:[0-9a-fA-F]{32}|call-[0-9]+|[0-9a-fA-F-]{36})", text):
        return text
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_event_host(value: Any) -> str:
    text = _safe_event_text(value, limit=255).lower().rstrip(".")
    if re.fullmatch(r"(?:\*\.)?[a-z0-9.-]+(?::[0-9]{1,5})?", text):
        return text
    return ""


def _safe_event_digest(value: Any) -> str:
    text = _safe_event_text(value, limit=64)
    return text.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", text) else ""


def _safe_event_int(value: Any, *, maximum: int) -> int | None:
    if type(value) is not int or not 0 <= value <= maximum:
        return None
    return value


def _safe_event_number(value: Any, *, maximum: float) -> int | float | None:
    if type(value) is int:
        return value if 0 <= value <= maximum else None
    if type(value) is not float:
        return None
    if not math.isfinite(value) or not 0.0 <= value <= maximum:
        return None
    return value


def _safe_event_usage(value: Any) -> dict[str, int]:
    if type(value) is not dict:
        return {}
    try:
        return {
            name: item
            for name, item in value.items()
            if type(name) is str
            and name in _USAGE_FIELDS
            and type(item) is int
            and 0 <= item <= _MAX_EVENT_TOKEN_COUNT
        }
    except BaseException:
        return {}


def _project_provider_usage(metadata: Any) -> dict[str, int]:
    try:
        if type(metadata) is not dict:
            return {}
        return _safe_event_usage(metadata.get("usage"))
    except BaseException:
        return {}


def _safe_event_message(value: Any) -> str:
    text = _safe_event_text(value, limit=200)
    if text in _GENERIC_FAILURE_MESSAGES:
        return text
    match = re.fullmatch(r"HTTP ([1-5][0-9]{2}): (.+)", text)
    if match is not None and match.group(2) in _GENERIC_FAILURE_MESSAGES:
        return text
    return "<redacted-field>" if text else ""


def _project_model_event(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = _MODEL_EVENT_FIELDS.get(event)
    if allowed is None or not isinstance(payload, dict):
        return {}
    projected: dict[str, Any] = {}
    for name in ("sample_id", "stage", "model", "provider"):
        if name in allowed and name in payload:
            value = _safe_event_text(payload[name])
            if value:
                projected[name] = value
    if "call_id" in allowed and "call_id" in payload:
        call_id = _safe_event_call_id(payload["call_id"])
        if call_id:
            projected["call_id"] = call_id
    if "endpoint_host" in allowed and "endpoint_host" in payload:
        host = _safe_event_host(payload["endpoint_host"])
        if host:
            projected["endpoint_host"] = host
    if "endpoint_sha256" in allowed and "endpoint_sha256" in payload:
        digest = _safe_event_digest(payload["endpoint_sha256"])
        if digest:
            projected["endpoint_sha256"] = digest
    for name, maximum in (
        ("api_calls", _MAX_EVENT_API_CALLS),
        ("api_call_limit", _MAX_EVENT_API_CALLS),
        ("attempt", _MAX_EVENT_ATTEMPT),
        ("next_attempt", _MAX_EVENT_ATTEMPT),
    ):
        if name in allowed and name in payload:
            number = _safe_event_int(payload[name], maximum=maximum)
            if number is not None:
                projected[name] = number
    if "delay_seconds" in allowed and "delay_seconds" in payload:
        delay = _safe_event_number(
            payload["delay_seconds"],
            maximum=_MAX_EVENT_ELAPSED_SECONDS,
        )
        if delay is not None:
            projected["delay_seconds"] = delay
    if "elapsed_seconds" in allowed and "elapsed_seconds" in payload:
        elapsed = _safe_event_number(
            payload["elapsed_seconds"],
            maximum=_MAX_EVENT_ELAPSED_SECONDS,
        )
        if elapsed is not None:
            projected["elapsed_seconds"] = elapsed
    if "usage" in allowed and "usage" in payload:
        projected["usage"] = _safe_event_usage(payload["usage"])
    if "status" in allowed and payload.get("status") in {"completed", "transport_error"}:
        projected["status"] = payload["status"]
    if "error_category" in allowed and payload.get("error_category") in _FAILURE_CATEGORIES:
        projected["error_category"] = payload["error_category"]
    if "error_type" in allowed and "error_type" in payload:
        error_type = _safe_event_text(payload["error_type"], limit=100)
        projected["error_type"] = (
            error_type
            if error_type in {"TransientRequestError", "PermanentRequestError", "Exception"}
            else "Exception"
        )
    if "message" in allowed and "message" in payload:
        message = _safe_event_message(payload["message"])
        if message:
            projected["message"] = message
    return projected


def _bounded_elapsed_since(started_at: float) -> float:
    try:
        elapsed = time.perf_counter() - started_at
    except BaseException:
        return 0.0
    if not math.isfinite(elapsed):
        return 0.0
    return min(max(0.0, elapsed), _MAX_EVENT_ELAPSED_SECONDS)


def endpoint_identity(base_url: str) -> dict[str, str]:
    """Return a stable endpoint fingerprint without exposing workspace IDs."""

    parsed = urlsplit(str(base_url or "").strip())
    host = parsed.netloc.lower()
    suffix = ".cn-beijing.maas.aliyuncs.com"
    public_host = f"*{suffix}" if host.endswith(suffix) and host != suffix.lstrip(".") else host
    return {
        "host": public_host,
        "sha256": hashlib.sha256(str(base_url or "").strip().encode("utf-8")).hexdigest(),
    }


class ModelClient(Protocol):
    def predict(self, sample: dict, analyses: dict) -> dict:
        """Return ScamMark output fields for one sample."""

    def complete_json(self, stage: str, prompt: str, sample: dict, include_image: bool = False) -> dict:
        """Return one strict JSON object for a staged ScamMark prompt."""


@dataclass
class RuleBasedModelClient:
    scam_threshold: int = 2

    def complete_json(self, stage: str, prompt: str, sample: dict, include_image: bool = False) -> dict:
        """Small deterministic fallback for staged smoke tests without API calls."""

        if stage == "binary":
            text = " ".join(str(sample.get(key) or "") for key in ["url", "title", "text", "visible_text"]).lower()
            is_scam = any(token in text for token in ["loan", "bonus", "wallet", "login", "profit", "投资", "贷款", "钱包"])
            return {"binary": "scam" if is_scam else "benign", "reasoning": "rule-based staged fallback"}
        if stage == "type":
            text = " ".join(str(sample.get(key) or "") for key in ["title", "text", "visible_text"]).lower()
            if any(token in text for token in ["loan", "贷款", "借款"]):
                scam_type = "Online Loan Scam"
            elif any(token in text for token in ["wallet", "crypto", "bitcoin", "虚拟币", "钱包"]):
                scam_type = "Cryptocurrency Investment Scam"
            elif any(token in text for token in ["profit", "investment", "投资", "理财"]):
                scam_type = "Online Investment Scam"
            else:
                scam_type = "none"
            return {"scam_type": scam_type, "reasoning": "rule-based staged fallback"}
        if stage == "grounding":
            return {"description": sample.get("title") or sample.get("url") or "", "reasoning": "rule-based staged fallback", "boxes": []}
        return {"risk_level": "low", "evidence_summary": "rule-based staged fallback"}

    def predict(self, sample: dict, analyses: dict) -> dict:
        url_result = analyses["url"]
        text_result = analyses["text"]
        consistency_result = analyses["consistency"]
        total_risk = (
            url_result.get("risk_score", 0)
            + text_result.get("risk_score", 0)
            + consistency_result.get("risk_score", 0)
        )
        predicted_binary = "scam" if total_risk >= self.scam_threshold else "benign"
        predicted_type = text_result.get("suggested_scam_type", "none") if predicted_binary == "scam" else "none"
        if predicted_binary == "scam" and predicted_type == "none":
            predicted_type = "Unknown Scam"

        signals = []
        for key in ["url", "text", "visual", "consistency"]:
            signals.extend(analyses[key].get("signals", []))
        reasoning = "; ".join(signals[:12]) or "No strong scam-specific signal was found by the rule baseline."
        description = text_result.get("text_excerpt") or sample.get("title") or sample.get("url") or "No textual evidence."

        return {
            "binary": predicted_binary,
            "scam_type": predicted_type,
            "description": description[:600],
            "reasoning": reasoning,
            "boxes": [],
        }


@dataclass
class OpenAIAPIModelClient:
    """OpenAI-compatible API backend for local low-VRAM demos.

    Formal experiments route text stages to GPT-4o-mini and multimodal stages
    to Qwen3-VL-Flash through the configured OpenAI-compatible provider.
    """

    model: str = DEFAULT_OPENAI_MODEL
    text_model: str | None = None
    vision_model: str | None = None
    text_base_url: str | None = None
    vision_base_url: str | None = None
    strict_routing: bool = False
    allow_env_image_override: bool = True
    api_key_env: str = "OPENAI_API_KEY"
    openrouter_key_env: str = "OPENROUTER_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    request_policy: RequestPolicy = field(default_factory=RequestPolicy)
    transport: Callable[..., tuple[str, dict[str, Any]]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    max_api_calls: int | None = None
    call_ledger: Any | None = field(default=None, repr=False, compare=False)
    event_callback: Callable[[str, dict[str, Any]], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    api_call_count: int = field(default=0, init=False)
    _http_clients: dict[tuple[str, float, float], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def predict(self, sample: dict, analyses: dict) -> dict:
        user_prompt = (
            "You are ScamMark, a training-free scam webpage analyst. "
            "Return only strict JSON with keys binary, scam_type, description, reasoning, boxes. "
            "Do not use labels as input. Evidence follows:\n"
            f"{json.dumps({'sample': _model_safe_sample(sample), 'analyses': analyses}, ensure_ascii=False)[:12000]}"
        )
        return _complete_chat_json(
            model=self.model,
            prompt=user_prompt,
            sample=sample,
            include_image=False,
            api_key_env=self.api_key_env,
            openrouter_key_env=self.openrouter_key_env,
            base_url_env=self.base_url_env,
        )

    def complete_json(self, stage: str, prompt: str, sample: dict, include_image: bool = False) -> dict:
        result, _ = self.complete_json_with_metadata(
            stage=stage,
            prompt=prompt,
            sample=sample,
            include_image=include_image,
        )
        return result

    def complete_json_with_metadata(
        self,
        stage: str,
        prompt: str,
        sample: dict,
        include_image: bool = False,
        max_output_tokens: int | None = None,
        image_assets: list[Any] | None = None,
        max_attempts: int | None = None,
        max_total_calls: int | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        model = self._stage_model(include_image)
        base_url = self._configured_base_url(model, include_image=include_image)
        messages = _build_api_messages(
            model,
            sample,
            prompt,
            include_image=include_image,
            image_assets=image_assets,
            allow_env_image_override=self.allow_env_image_override,
        )
        valid_image_indexes = _image_asset_indexes(image_assets) if stage == "grounding" else None
        policy = self.request_policy
        if max_attempts is not None:
            policy = replace(policy, max_attempts=max(1, min(policy.max_attempts, int(max_attempts))))
        read_timeout = (
            policy.vlm_read_timeout_seconds if _is_vlm_model(model) else policy.llm_read_timeout_seconds
        )
        transport = self.transport or self._provider_transport
        request_fingerprint = _request_payload_fingerprint(
            stage=stage,
            model=model,
            base_url=base_url,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            image_assets=image_assets if include_image else [],
        )
        endpoint = endpoint_identity(base_url)
        call_context = {
            "sample_id": str(sample.get("sample_id") or sample.get("id") or ""),
            "stage": stage,
            "model": model,
            "provider": _provider_name(base_url),
            "endpoint_host": endpoint["host"],
            "endpoint_sha256": endpoint["sha256"],
            "request_fingerprint": request_fingerprint,
        }

        def request_once(request_messages: list[dict]) -> tuple[str, dict[str, Any]]:
            return transport(
                model=model,
                base_url=base_url,
                messages=request_messages,
                max_output_tokens=max_output_tokens,
                connect_timeout_seconds=policy.connect_timeout_seconds,
                read_timeout_seconds=read_timeout,
                enable_thinking=False,
            )

        text, first_metadata = self._request_with_retry(
            lambda: request_once(messages),
            policy,
            call_context=call_context,
        )
        first_metadata["endpoint_host"] = endpoint["host"]
        first_metadata["endpoint_sha256"] = endpoint["sha256"]
        call_metadata = [first_metadata]
        repair_attempted = False
        parse_error = ""
        try:
            self._validate_provider_metadata(model, first_metadata)
        except PermanentRequestError as exc:
            setattr(
                exc,
                "request_metadata",
                _aggregate_stage_request_metadata(
                    stage=stage,
                    model=model,
                    provider=_provider_name(base_url),
                    call_metadata=call_metadata,
                    max_output_tokens=max_output_tokens,
                    repair_attempted=False,
                    repair_error="",
                ),
            )
            raise
        try:
            result = _parse_and_validate_stage_output(
                stage,
                text,
                valid_image_indexes=valid_image_indexes,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            calls_used = sum(int(item.get("api_call_count") or 0) for item in call_metadata)
            if max_total_calls is not None and calls_used >= max_total_calls:
                setattr(
                    exc,
                    "request_metadata",
                    _aggregate_stage_request_metadata(
                        stage=stage,
                        model=model,
                        provider=_provider_name(base_url),
                        call_metadata=call_metadata,
                        max_output_tokens=max_output_tokens,
                        repair_attempted=False,
                        repair_error=parse_error,
                    ),
                )
                raise
            repair_attempted = True
            repair_prompt = _build_schema_repair_prompt(stage, text, parse_error)
            repair_messages = _build_api_messages(
                model,
                {},
                repair_prompt,
                include_image=False,
                image_assets=[],
                allow_env_image_override=False,
            )
            repair_metadata = None
            try:
                repaired_text, repair_metadata = self._request_with_retry(
                    lambda: request_once(repair_messages),
                    replace(policy, max_attempts=1),
                    call_context={**call_context, "schema_repair": True},
                )
                repair_metadata["endpoint_host"] = endpoint["host"]
                repair_metadata["endpoint_sha256"] = endpoint["sha256"]
                call_metadata.append(repair_metadata)
                self._validate_provider_metadata(model, repair_metadata)
            except (PermanentRequestError, TransientRequestError) as repair_request_exc:
                if repair_metadata is None:
                    call_metadata.append(dict(getattr(repair_request_exc, "request_metadata", {}) or {}))
                setattr(
                    repair_request_exc,
                    "request_metadata",
                    _aggregate_stage_request_metadata(
                        stage=stage,
                        model=model,
                        provider=_provider_name(base_url),
                        call_metadata=call_metadata,
                        max_output_tokens=max_output_tokens,
                        repair_attempted=True,
                        repair_error=parse_error,
                    ),
                )
                raise
            try:
                result = _parse_and_validate_stage_output(
                    stage,
                    repaired_text,
                    valid_image_indexes=valid_image_indexes,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as repair_exc:
                setattr(
                    repair_exc,
                    "request_metadata",
                    _aggregate_stage_request_metadata(
                        stage=stage,
                        model=model,
                        provider=_provider_name(base_url),
                        call_metadata=call_metadata,
                        max_output_tokens=max_output_tokens,
                        repair_attempted=True,
                        repair_error=parse_error,
                    ),
                )
                raise

        metadata = _aggregate_stage_request_metadata(
            stage=stage,
            model=model,
            provider=_provider_name(base_url),
            call_metadata=call_metadata,
            max_output_tokens=max_output_tokens,
            repair_attempted=repair_attempted,
            repair_error=parse_error,
        )
        return result, metadata

    def _stage_model(self, include_image: bool) -> str:
        if include_image and self.vision_model:
            return str(self.vision_model)
        if not include_image and self.text_model:
            return str(self.text_model)
        return _select_stage_model(self.model, include_image)

    def _configured_base_url(self, model: str, *, include_image: bool | None = None) -> str:
        if include_image is True and self.vision_base_url:
            return str(self.vision_base_url)
        if include_image is False and self.text_base_url:
            return str(self.text_base_url)
        if include_image is None:
            if _is_vlm_model(model) and self.vision_base_url:
                return str(self.vision_base_url)
            if not _is_vlm_model(model) and self.text_base_url:
                return str(self.text_base_url)
        _, base_url = _select_api_config(
            model,
            self.api_key_env,
            self.openrouter_key_env,
            self.base_url_env,
        )
        return base_url

    def _validate_provider_metadata(self, requested_model: str, metadata: dict[str, Any]) -> None:
        if str(metadata.get("finish_reason") or "").lower() == "length":
            raise PermanentRequestError("Provider truncated the response with finish_reason=length")
        response_model = str(metadata.get("response_model") or "")
        matches = _response_model_matches(requested_model, response_model)
        metadata["response_model_matches_request"] = matches
        if self.strict_routing and not matches:
            raise PermanentRequestError(
                f"Provider response model {response_model or '<missing>'} does not match requested model {requested_model}"
            )

    def _provider_transport(self, **request: Any) -> tuple[str, dict[str, Any]]:
        model = str(request["model"])
        api_key, environment_base_url = _select_api_config(
            model,
            self.api_key_env,
            self.openrouter_key_env,
            self.base_url_env,
        )
        if not api_key:
            raise PermanentRequestError(
                f"{self.api_key_env}, SCAMMARK_LLM_API_KEY, SCAMMARK_VLM_API_KEY or "
                f"{self.openrouter_key_env} is not set"
            )
        base_url = str(request.pop("base_url", "") or environment_base_url)
        cache_key = (
            str(base_url),
            float(request["connect_timeout_seconds"]),
            float(request["read_timeout_seconds"]),
        )
        http_client = self._http_clients.get(cache_key)
        if http_client is None:
            try:
                import httpx
            except ImportError as exc:
                raise PermanentRequestError("httpx is required for resilient API calls") from exc
            http_client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=cache_key[1],
                    read=cache_key[2],
                    write=cache_key[2],
                    pool=cache_key[1],
                )
            )
            self._http_clients[cache_key] = http_client
        return _http_chat_completion_once(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            **request,
        )

    def close(self) -> None:
        for client in self._http_clients.values():
            client.close()
        self._http_clients.clear()

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        callback = self.event_callback
        if callback is None:
            return
        try:
            projected = _project_model_event(event, payload)
            callback(event, projected)
        except BaseException:
            pass

    def _request_with_retry(
        self,
        operation: Callable[[], tuple[str, dict[str, Any]]],
        policy: RequestPolicy,
        *,
        call_context: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        physical_calls = 0
        safe_retry_context: dict[str, Any] = {}
        if self.event_callback is not None:
            try:
                safe_retry_context = {
                    name: (call_context or {}).get(name)
                    for name in (
                        "sample_id",
                        "stage",
                        "model",
                        "provider",
                        "endpoint_host",
                        "endpoint_sha256",
                    )
                }
            except BaseException:
                safe_retry_context = {}

        def emit_retry(event: str, payload: dict[str, Any]) -> None:
            self._emit(event, {**payload, **safe_retry_context})

        def counted_operation() -> tuple[str, dict[str, Any]]:
            nonlocal physical_calls
            attempt_started = time.perf_counter()
            call_id = ""
            if self.call_ledger is not None:
                if self.max_api_calls is None:
                    raise PermanentRequestError("A call ledger requires a global max_api_calls value")
                try:
                    call_id = self.call_ledger.reserve_api_call(
                        dict(call_context or {}),
                        max_calls=self.max_api_calls,
                    )
                except Exception as exc:
                    raise PermanentRequestError(
                        "Durable API call reservation failed"
                    ) from exc
            elif self.max_api_calls is not None and self.api_call_count >= self.max_api_calls:
                raise PermanentRequestError(
                    f"Physical API call cap reached ({self.max_api_calls})"
                )
            self.api_call_count += 1
            physical_calls += 1
            if call_id:
                self._emit(
                    "api_call_reserved",
                    {
                        **dict(call_context or {}),
                        "call_id": call_id,
                        "attempt": physical_calls,
                        "api_calls": self.api_call_count,
                        "api_call_limit": self.max_api_calls,
                    },
                )
            try:
                result = operation()
            except Exception as exc:
                if call_id:
                    elapsed_seconds = _bounded_elapsed_since(attempt_started)
                    safe_error = _safe_request_error_details(exc)
                    try:
                        self.call_ledger.finish_api_call(
                            call_id,
                            status="transport_error",
                            metadata={
                                **safe_error,
                                "elapsed_seconds": elapsed_seconds,
                            },
                        )
                    except Exception as finish_exc:
                        raise PermanentRequestError(
                            "Durable API call completion recording failed"
                        ) from finish_exc
                    self._emit(
                        "api_call_finished",
                        {
                            **dict(call_context or {}),
                            "call_id": call_id,
                            "attempt": physical_calls,
                            "status": "transport_error",
                            "elapsed_seconds": elapsed_seconds,
                            **safe_error,
                        },
                    )
                raise
            if call_id:
                provider_metadata = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
                elapsed_seconds = _bounded_elapsed_since(attempt_started)
                usage = _project_provider_usage(provider_metadata)
                try:
                    self.call_ledger.finish_api_call(
                        call_id,
                        status="completed",
                        metadata={
                            "usage": usage,
                            "elapsed_seconds": elapsed_seconds,
                        },
                    )
                except Exception as finish_exc:
                    raise PermanentRequestError(
                        "Durable API call completion recording failed"
                    ) from finish_exc
                self._emit(
                    "api_call_finished",
                    {
                        **dict(call_context or {}),
                        "call_id": call_id,
                        "attempt": physical_calls,
                        "status": "completed",
                        "elapsed_seconds": elapsed_seconds,
                        "usage": usage,
                    },
                )
            return result

        try:
            result, retry_metadata = execute_with_retry(
                counted_operation,
                policy,
                event_callback=emit_retry if self.event_callback is not None else None,
            )
        except (PermanentRequestError, TransientRequestError) as exc:
            metadata = dict(getattr(exc, "request_metadata", {}) or {})
            metadata["api_call_count"] = physical_calls
            setattr(exc, "request_metadata", metadata)
            raise
        text, provider_metadata = result
        return text, {**retry_metadata, "api_call_count": physical_calls, **provider_metadata}


@dataclass
class LocalVLMModelClient:
    """HTTP client for a future server-hosted Qwen3-VL endpoint.

    On the laptop, use `OpenAIAPIModelClient` for Qwen API demos. On the server,
    deploy Qwen3-VL behind an HTTP endpoint that accepts this
    payload and returns the ScamMark JSON schema. This preserves the same
    pipeline for both API smoke tests and local full-dataset execution.
    """

    endpoint: str = ""
    model: str = DEFAULT_QWEN_VL_MODEL
    timeout_seconds: int = 300

    def complete_json(self, stage: str, prompt: str, sample: dict, include_image: bool = False) -> dict:
        _load_local_env()
        endpoint = self.endpoint or os.getenv("LOCAL_VLM_ENDPOINT") or "http://localhost:8000/scammark/predict"
        payload = {
            "stage": stage,
            "model": os.getenv("LOCAL_VLM_MODEL") or self.model,
            "prompt": prompt,
            "sample": _model_safe_sample(sample),
            "include_image": include_image,
            "image_url": _validated_image_data_url(sample.get("image_path")) if include_image else "",
            "schema": ["binary", "scam_type", "description", "reasoning", "boxes"],
        }
        return _post_local_json(endpoint, payload, self.timeout_seconds)

    def predict(self, sample: dict, analyses: dict) -> dict:
        _load_local_env()
        endpoint = self.endpoint or os.getenv("LOCAL_VLM_ENDPOINT") or "http://localhost:8000/scammark/predict"
        payload = {
            "model": os.getenv("LOCAL_VLM_MODEL") or self.model,
            "sample": _model_safe_sample(sample),
            "analyses": analyses,
            "image_url": _validated_image_data_url(sample.get("image_path")),
            "schema": ["binary", "scam_type", "description", "reasoning", "boxes"],
        }
        return _post_local_json(endpoint, payload, self.timeout_seconds)


def _model_safe_sample(sample: dict) -> dict:
    from .prompts import build_sample_context

    return build_sample_context(sample)


def make_model_client(backend: str, model: str | None = None) -> ModelClient:
    _load_local_env()
    backend = (backend or "rule").lower()
    if backend == "rule":
        return RuleBasedModelClient()
    if backend == "openai":
        return OpenAIAPIModelClient(model=model or _default_api_model())
    if backend in {"local", "local-vlm", "server"}:
        return LocalVLMModelClient(model=model or DEFAULT_QWEN_VL_MODEL)
    raise ValueError(f"Unknown backend: {backend}")


def selected_model_ids(client: ModelClient) -> dict[str, str]:
    """Return the concrete text and vision model identifiers used by a client."""

    if isinstance(client, RuleBasedModelClient):
        return {"text_model": "rule-based", "visual_model": "rule-based"}
    if isinstance(client, OpenAIAPIModelClient):
        return {
            "text_model": client._stage_model(include_image=False),
            "visual_model": client._stage_model(include_image=True),
        }
    if isinstance(client, LocalVLMModelClient):
        model = os.getenv("LOCAL_VLM_MODEL") or client.model
        return {"text_model": model, "visual_model": model}
    model = str(getattr(client, "model", "unknown") or "unknown")
    return {"text_model": model, "visual_model": model}


def _default_api_model() -> str:
    _load_local_env()
    configured = os.getenv("OPENAI_MODEL")
    if configured:
        return configured
    configured_llm = os.getenv("SCAMMARK_LLM_MODEL")
    if configured_llm:
        return configured_llm
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or ""
    if "openrouter.ai" in base_url or os.getenv("OPENROUTER_API_KEY"):
        return DEFAULT_OPENROUTER_MODEL
    return DEFAULT_OPENAI_MODEL


def _select_stage_model(configured_model: str, include_image: bool) -> str:
    _load_local_env()
    if include_image:
        configured_vlm = os.getenv("SCAMMARK_VLM_MODEL")
        if configured_vlm:
            return configured_vlm
        if configured_model and _is_vlm_model(configured_model):
            return configured_model
        return DEFAULT_QWEN_VL_MODEL
    configured_llm = os.getenv("SCAMMARK_LLM_MODEL")
    if configured_llm:
        return configured_llm
    if configured_model and not _is_vlm_model(configured_model):
        return configured_model
    return _default_api_model()


def _select_api_config(model: str, api_key_env: str, openrouter_key_env: str, base_url_env: str) -> tuple[str, str]:
    _load_local_env()
    is_vlm = _is_vlm_model(model)
    key_candidates = []
    if is_vlm:
        key_candidates.extend(["SCAMMARK_VLM_API_KEY", api_key_env, openrouter_key_env])
    else:
        key_candidates.extend(["SCAMMARK_LLM_API_KEY", api_key_env, openrouter_key_env])
    api_key = ""
    for key_name in key_candidates:
        api_key = _normalize_api_key(os.getenv(key_name) or "")
        if api_key:
            break
    base_url = (
        os.getenv("SCAMMARK_VLM_BASE_URL" if is_vlm else "SCAMMARK_LLM_BASE_URL")
        or os.getenv(base_url_env)
        or os.getenv("CRAZYROUTER_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or ""
    )
    return api_key, base_url


def _is_vlm_model(model: str) -> bool:
    model_lower = (model or "").lower()
    vlm_model = (os.getenv("SCAMMARK_VLM_MODEL") or DEFAULT_QWEN_VL_MODEL).lower()
    return "qwen" in model_lower or "vl" in model_lower or model_lower == vlm_model


def _build_api_messages(
    model: str,
    sample: dict,
    user_prompt: str,
    include_image: bool = False,
    image_assets: list[Any] | None = None,
    allow_env_image_override: bool = True,
) -> list[dict]:
    env_include_image = allow_env_image_override and str(
        os.getenv("SCAMMARK_INCLUDE_IMAGE") or ""
    ).strip().lower() in {"1", "true", "yes", "y"}
    attach_image = include_image or env_include_image
    image_urls: list[str] = []
    if attach_image and image_assets is not None:
        for asset in image_assets:
            if isinstance(asset, dict):
                data_url = str(asset.get("data_url") or "")
            else:
                data_url_method = getattr(asset, "data_url", None)
                data_url = str(data_url_method() if callable(data_url_method) else "")
            if data_url:
                image_urls.append(data_url)
    elif attach_image:
        image_urls.append(_validated_image_data_url(sample.get("image_path")))
    if image_urls:
        user_content: str | list[dict] = [{"type": "text", "text": user_prompt}]
        user_content.extend(
            {"type": "image_url", "image_url": {"url": image_url}}
            for image_url in image_urls
        )
    else:
        user_content = user_prompt
    return [
        {"role": "system", "content": "You are a careful multimodal scam webpage analyst."},
        {"role": "user", "content": user_content},
    ]


def _image_data_url(path_value: object) -> str:
    path = Path(str(path_value or ""))
    if not path.exists() or not path.is_file():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _validated_image_data_url(path_value: object) -> str:
    path = Path(str(path_value or ""))
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image-required stage cannot read screenshot: {path}")
    try:
        from .visual_analysis import get_image_size

        width, height = get_image_size(str(path))
    except Exception as exc:
        raise ValueError(f"Image-required stage cannot decode screenshot: {path}") from exc
    if not width or not height or width <= 0 or height <= 0:
        raise ValueError(f"Image-required stage received an invalid screenshot: {path}")
    image_url = _image_data_url(path)
    if not image_url:
        raise ValueError(f"Image-required stage cannot encode screenshot: {path}")
    return image_url


def _provider_name(base_url: str) -> str:
    lowered = str(base_url or "").lower()
    if "crazyrouter" in lowered:
        return "Crazyrouter"
    if "openrouter" in lowered:
        return "OpenRouter"
    if "dashscope.aliyuncs.com" in lowered or ".maas.aliyuncs.com" in lowered:
        return "Alibaba Cloud Model Studio"
    if "openai.com" in lowered or not lowered:
        return "OpenAI"
    return "OpenAI-compatible"


def _response_model_matches(requested: str, response: str) -> bool:
    requested_name = str(requested or "").strip().lower().split("/")[-1]
    response_name = str(response or "").strip().lower().split("/")[-1]
    if not requested_name or not response_name:
        return False
    return (
        response_name == requested_name
        or response_name.startswith(requested_name + "-")
        or requested_name in response_name
    )


def _request_payload_fingerprint(
    *,
    stage: str,
    model: str,
    base_url: str,
    prompt: str,
    max_output_tokens: int | None,
    image_assets: list[Any] | None,
) -> str:
    image_hashes = []
    for asset in image_assets or []:
        if isinstance(asset, dict):
            path_value = asset.get("path")
            data_url = str(asset.get("data_url") or "")
        else:
            path_value = getattr(asset, "path", None)
            data_url = ""
        path = Path(str(path_value or ""))
        if path.is_file():
            image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        elif data_url:
            image_hashes.append(hashlib.sha256(data_url.encode("utf-8")).hexdigest())
    payload = {
        "stage": stage,
        "model": model,
        "endpoint_host": urlsplit(base_url).netloc,
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
        "image_sha256": image_hashes,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sum_usage(values: list[dict[str, Any]]) -> dict[str, int]:
    keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    }
    return {
        key: sum(int(value.get(key) or 0) for value in values)
        for key in keys
        if any(key in value for value in values)
    }


def _image_asset_indexes(image_assets: list[Any] | None) -> set[int] | None:
    if not image_assets:
        return None
    indexes = set()
    for asset in image_assets:
        logical_values = (
            asset.get("logical_indexes")
            if isinstance(asset, dict)
            else getattr(asset, "logical_indexes", None)
        )
        values = logical_values or [
            asset.get("index") if isinstance(asset, dict) else getattr(asset, "index", None)
        ]
        for value in values:
            try:
                indexes.add(int(value))
            except (TypeError, ValueError):
                continue
    return indexes or None


def _aggregate_stage_request_metadata(
    *,
    stage: str,
    model: str,
    provider: str,
    call_metadata: list[dict[str, Any]],
    max_output_tokens: int | None,
    repair_attempted: bool,
    repair_error: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "model": model,
        "provider": provider,
        "attempt_count": sum(int(item.get("attempt_count") or 0) for item in call_metadata),
        "retry_count": sum(int(item.get("retry_count") or 0) for item in call_metadata),
        "api_call_count": sum(int(item.get("api_call_count") or 0) for item in call_metadata),
        "elapsed_seconds": sum(float(item.get("elapsed_seconds") or 0.0) for item in call_metadata),
        "attempt_errors": [
            error
            for item in call_metadata
            for error in (item.get("attempt_errors") or [])
        ],
        "usage": _sum_usage([item.get("usage") or {} for item in call_metadata]),
        "finish_reason": call_metadata[-1].get("finish_reason", ""),
        "response_model": call_metadata[-1].get("response_model", ""),
        "response_model_matches_request": all(
            bool(item.get("response_model_matches_request", True))
            for item in call_metadata
        ),
        "endpoint_host": call_metadata[-1].get("endpoint_host", ""),
        "endpoint_sha256": call_metadata[-1].get("endpoint_sha256", ""),
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "schema_repair_attempted": repair_attempted,
        "schema_repair_error": repair_error,
    }


def _stage_required_keys(stage: str) -> set[str]:
    if stage == "mhtml_identity_audit":
        return {"status", "confidence", "reasons"}
    if stage == "gold_conflict_audit":
        return {"gold_input_conflict", "confidence", "reasons"}
    try:
        from .prompts import STAGE_TASKS

        return set(STAGE_TASKS[stage]["schema"])
    except KeyError:
        return set()


def _parse_and_validate_stage_output(
    stage: str,
    text: str,
    *,
    valid_image_indexes: set[int] | None = None,
) -> dict[str, Any]:
    result = _parse_json_response(text)
    if not isinstance(result, dict):
        raise TypeError("Provider output must be one JSON object")
    missing = sorted(_stage_required_keys(stage) - set(result))
    if missing:
        raise ValueError(f"Stage {stage} output is missing keys: {', '.join(missing)}")
    if stage not in {"mhtml_identity_audit", "gold_conflict_audit"}:
        from .prompts import STAGE_TASKS

        for key, example in STAGE_TASKS.get(stage, {}).get("schema", {}).items():
            if isinstance(example, list) and not isinstance(result.get(key), list):
                raise ValueError(f"Stage {stage} field {key} must be a JSON list")
            if isinstance(example, str) and not isinstance(result.get(key), str):
                raise ValueError(f"Stage {stage} field {key} must be a JSON string")
    if stage in {"url", "cross"} and result.get("risk_level") not in {"low", "medium", "high"}:
        raise ValueError(f"Stage {stage} risk_level must be low, medium, or high")
    if stage == "binary" and result.get("binary") not in {"scam", "benign"}:
        raise ValueError("Binary stage must return binary=scam or binary=benign")
    if stage == "text":
        from .taxonomy import CANONICAL_LABELS

        allowed_types = set(CANONICAL_LABELS) | {"none"}
        values = result.get("candidate_scam_types") or []
        if any(value not in allowed_types for value in values):
            raise ValueError(f"Stage {stage} returned a non-canonical scam type")
    if stage == "type":
        from .taxonomy import CANONICAL_LABELS

        type_id = result.get("scam_type_id")
        if isinstance(type_id, bool) or not isinstance(type_id, int):
            raise ValueError("Stage type scam_type_id must be an integer")
        if not 0 <= type_id <= len(CANONICAL_LABELS):
            raise ValueError(f"Stage type scam_type_id must be within 0..{len(CANONICAL_LABELS)}")
        expected_label = "none" if type_id == 0 else CANONICAL_LABELS[type_id - 1]
        reported_label = str(result.get("scam_type") or "")
        if reported_label != expected_label:
            result["reported_scam_type"] = reported_label
        result["scam_type"] = expected_label
    if stage == "mhtml_identity_audit":
        if result.get("status") not in {"valid_redirect", "mhtml_mismatch", "uncertain"}:
            raise ValueError("MHTML audit returned an invalid status")
    if stage == "gold_conflict_audit":
        if not isinstance(result.get("gold_input_conflict"), bool):
            raise ValueError("MHTML audit gold_input_conflict must be boolean")
    if stage in {"mhtml_identity_audit", "gold_conflict_audit"}:
        if not isinstance(result.get("reasons"), list):
            raise ValueError("MHTML audit reasons must be a list")
        try:
            confidence = float(result.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("MHTML audit confidence must be numeric") from exc
        if not 0 <= confidence <= 1:
            raise ValueError("MHTML audit confidence must be within 0..1")
    if stage == "grounding" and not isinstance(result.get("boxes"), list):
        raise ValueError("Grounding boxes must be a JSON list")
    if stage == "grounding":
        if len(result.get("boxes") or []) > 4:
            raise ValueError("Grounding may return at most four boxes")
        for box in result.get("boxes") or []:
            if not isinstance(box, dict):
                raise ValueError("Each grounding box must be a JSON object")
            if not str(box.get("reason") or "").strip():
                raise ValueError("Each grounding box requires a nonempty reason")
            if not str(box.get("evidence_type") or "").strip():
                raise ValueError("Each grounding box requires an evidence_type")
            if valid_image_indexes is None:
                continue
            try:
                image_index = int(box.get("image_index"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Each grounding box requires an integer image_index") from exc
            if image_index not in valid_image_indexes:
                raise ValueError(
                    f"Grounding image_index {image_index} is not one of {sorted(valid_image_indexes)}"
                )
            coordinates = box.get("bbox_1000")
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                raise ValueError("Each tiled grounding box requires four bbox_1000 coordinates")
    return result


def _build_schema_repair_prompt(stage: str, raw_text: str, error: str) -> str:
    allowed_values = ""
    if stage == "mhtml_identity_audit":
        schema: Any = {
            "status": "valid_redirect|mhtml_mismatch|uncertain",
            "confidence": 0.0,
            "reasons": ["short reason"],
        }
    elif stage == "gold_conflict_audit":
        schema = {
            "gold_input_conflict": False,
            "confidence": 0.0,
            "reasons": ["short reason"],
        }
    else:
        from .prompts import STAGE_TASKS

        schema = STAGE_TASKS.get(stage, {}).get("schema", {})
        if stage in {"text", "type"}:
            from .taxonomy import CANONICAL_LABELS

            field = "candidate_scam_types" if stage == "text" else "scam_type"
            values: Any = [*CANONICAL_LABELS, "none"]
            if stage == "type":
                values = [
                    {"id": 0, "label": "none"},
                    *({"id": index, "label": label} for index, label in enumerate(CANONICAL_LABELS, start=1)),
                ]
            allowed_values = f"\nAllowed {field} values: {json.dumps(values, ensure_ascii=False)}"
    return (
        "Repair the following malformed model output. Preserve its supported meaning, fill required keys conservatively, "
        "and return exactly one strict JSON object with no Markdown.\n"
        f"Stage: {stage}\nValidation error: {error}\n"
        f"Required schema: {json.dumps(schema, ensure_ascii=False)}{allowed_values}\n"
        f"Malformed output: {raw_text[:12000]}"
    )


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None


def _is_retryable_provider_http_error(status_code: int, detail: str) -> bool:
    """Recognize a provider-specific temporary failure that is mislabeled HTTP 400."""

    if status_code != 400:
        return False
    try:
        payload = json.loads(detail)
        error = payload.get("error") if isinstance(payload, dict) else None
    except (TypeError, ValueError):
        error = None
    if not isinstance(error, dict) or str(error.get("code") or "") != "data_inspection_failed":
        return False
    message = str(error.get("message") or "").lower()
    return "temporarily unavailable" in message and "try again later" in message


def _is_explicit_exhausted_quota_429(status_code: int, detail: str) -> bool:
    return (
        status_code == 429
        and provider_429_failure_category(detail) == "auth_billing"
    )


def _http_chat_completion_once(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    max_output_tokens: int | None,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    enable_thinking: bool,
    http_client: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """Perform exactly one OpenAI-compatible request; retry lives above this layer."""
    try:
        import httpx
    except ImportError as exc:
        raise PermanentRequestError("httpx is required for resilient API calls") from exc

    endpoint_base = (base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(_openrouter_headers(endpoint_base))
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = int(max_output_tokens)
    if _is_vlm_model(model):
        payload["enable_thinking"] = bool(enable_thinking)
    try:
        if http_client is None:
            timeout = httpx.Timeout(
                connect=connect_timeout_seconds,
                read=read_timeout_seconds,
                write=read_timeout_seconds,
                pool=connect_timeout_seconds,
            )
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{endpoint_base}/chat/completions", headers=headers, json=payload)
        else:
            response = http_client.post(f"{endpoint_base}/chat/completions", headers=headers, json=payload)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise TransientRequestError(f"{type(exc).__name__}: {str(exc)[:500]}") from exc

    if response.status_code >= 400:
        detail = response.text[:500]
        message = f"HTTP {response.status_code}: {detail}"
        if _is_explicit_exhausted_quota_429(response.status_code, detail):
            raise PermanentRequestError(message)
        temporary_provider_400 = _is_retryable_provider_http_error(
            response.status_code,
            detail,
        )
        if is_retryable_http_status(response.status_code) or temporary_provider_400:
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            if temporary_provider_400:
                retry_after = max(30.0, retry_after or 0.0)
            raise TransientRequestError(
                message,
                retry_after_seconds=retry_after,
            )
        raise PermanentRequestError(message)
    try:
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise TransientRequestError(f"Invalid provider response: {str(exc)[:500]}") from exc
    if not str(content).strip():
        raise TransientRequestError(
            f"Provider returned empty content; finish_reason={choice.get('finish_reason', '')}"
        )
    return str(content), {
        "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
        "finish_reason": str(choice.get("finish_reason") or ""),
        "response_model": str(data.get("model") or model),
        "response_id": str(data.get("id") or ""),
        "http_status": response.status_code,
    }


def _complete_chat_json(
    model: str,
    prompt: str,
    sample: dict,
    include_image: bool,
    api_key_env: str,
    openrouter_key_env: str,
    base_url_env: str,
) -> dict:
    _load_local_env()
    api_key, base_url = _select_api_config(model, api_key_env, openrouter_key_env, base_url_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env}, SCAMMARK_LLM_API_KEY, SCAMMARK_VLM_API_KEY or {openrouter_key_env} is not set; "
            "use --backend rule or configure an API key."
        )
    messages = _build_api_messages(model, sample, prompt, include_image=include_image)
    if base_url and "openrouter.ai" in base_url:
        text = _openrouter_chat_completion(api_key, base_url, model, messages)
    else:
        try:
            from openai import OpenAI
        except Exception:
            text = _generic_chat_completion(api_key, base_url, model, messages)
        else:
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            text = response.choices[0].message.content or ""
    return _parse_json_response(text)


def _post_local_json(endpoint: str, payload: dict, timeout_seconds: int) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local VLM request failed with HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Local VLM endpoint is unavailable. Deploy a Qwen3-VL endpoint on the server "
            "and set LOCAL_VLM_ENDPOINT before using --backend local."
        ) from exc
    return _parse_json_response(text)


def _openrouter_headers(base_url: str) -> dict[str, str]:
    if "openrouter.ai" not in base_url:
        return {}
    headers = {}
    referer = os.getenv("OPENROUTER_SITE_URL") or os.getenv("HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_NAME") or "ScamWeb-V"
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def _openrouter_chat_completion(api_key: str, base_url: str, model: str, messages: list[dict]) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(_openrouter_headers(base_url))
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    last_empty_reason = ""
    for attempt in range(1, 4):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}: {detail[:500]}") from exc
        content = data["choices"][0]["message"].get("content") or ""
        if content.strip():
            return content
        last_empty_reason = str(data["choices"][0].get("finish_reason", ""))
        if attempt < 3:
            time.sleep(attempt)
    raise RuntimeError(f"OpenRouter returned empty content after retries; finish_reason={last_empty_reason}")


def _generic_chat_completion(api_key: str, base_url: str, model: str, messages: list[dict]) -> str:
    endpoint_base = (base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{endpoint_base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {detail[:500]}") from exc
    content = data["choices"][0]["message"].get("content") or ""
    if not content.strip():
        finish_reason = data["choices"][0].get("finish_reason", "")
        raise RuntimeError(f"OpenAI-compatible backend returned empty content; finish_reason={finish_reason}")
    return content


def _normalize_api_key(value: str) -> str:
    text = (value or "").strip().strip('"').strip("'")
    if text.lower().startswith("authorization:"):
        text = text.split(":", 1)[1].strip()
    if text.lower().startswith("bearer "):
        text = text.split(None, 1)[1].strip()
    return text


def _load_local_env() -> None:
    # The project-root file is canonical; keep src/.env as a legacy fallback.
    for path in (Path(".env"), Path("src/.env")):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _parse_json_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _first_balanced_json_object(text)
        if not candidate:
            raise
        return json.loads(candidate)


def _first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""
