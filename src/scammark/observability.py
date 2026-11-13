from __future__ import annotations

import argparse
from collections import defaultdict, deque
import dataclasses
import json
import math
import os
import re
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, TextIO
from zoneinfo import ZoneInfo

if os.name == "posix":
    import fcntl
else:  # pragma: no cover - exercised by Windows test environments
    fcntl = None  # type: ignore[assignment]


EVENT_SCHEMA = "scammark-run-event-v1"
HEALTH_VALUES = frozenset(
    {
        "ok",
        "warming_up",
        "waiting_api",
        "retrying",
        "degraded",
        "paused",
        "stopped",
    }
)
CRITICAL_EVENTS = frozenset(
    {
        "run_started",
        "resume_audit",
        "sample_complete",
        "prediction_flushed",
        "metrics_written",
        "warning",
        "error",
        "breaker_opened",
        "breaker_reset_ack",
        "stop_requested",
        "run_stopped",
        "run_completed",
        "shell_exited",
        "delivery_stage",
    }
)
SENSITIVE_FIELDS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "prompt",
        "prompts",
        "messages",
        "response",
        "raw_response",
        "image",
        "image_data",
        "image_payload",
        "base64",
        "mhtml",
        "html",
        "webpage",
        "page_content",
        "traceback",
    }
)
AUTH_RE = re.compile(
    r"(?i)(authorization\s*[:=]\s*bearer\s+|bearer\s+)[^\s,;]+"
)
KEY_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|api[_-]?key\s*[:=]\s*[^\s,;]+)"
)
DATA_URL_RE = re.compile(
    r"(?is)data:image/[^;,\s]+;base64,[A-Za-z0-9+/=]+"
)
WORKSPACE_RE = re.compile(
    r"(?i)\b[A-Za-z0-9-]+\.cn-beijing\.maas\.aliyuncs\.com\b"
)
QUERY_RE = re.compile(r"(?i)(https?://[^\s?]+)\?[^\s]+")
_HTTP_STATUS_RE = re.compile(
    r"(?i)\bHTTP(?:\s+status)?\s*(?:code\s*)?[:=]?\s*([1-5][0-9]{2})\b"
)
_ACK_HTTP_STATUS_RE = re.compile(
    r"(?i)\b(?:HTTP(?:[ _-]*error)?|provider[ _-]?error|status[ _-]?code)"
    r"\s*[:=]?\s*([1-5][0-9]{2})\b"
)
_ENDPOINT_RE = re.compile(r"(?i)\bhttps?://[^\s,;]+")
_FAILURE_SECRET_FIELD_RE = re.compile(
    r"(?i)\b(?:prompt|prompts|messages?|raw[_ -]?response|response[_ -]?body|"
    r"image[_ -]?(?:data|payload)|base64|html|mhtml|page[_ -]?content)"
    r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_FAILURE_SECRET_TAIL_RE = re.compile(
    r"(?is)\b(?:authorization|auth(?:[_ -]?token)?|api[_ -]?key|password|"
    r"client[_ -]?secret|"
    r"credentials?|refresh[_ -]?token|session[_ -]?token|private[_ -]?key|"
    r"token|key|secret|"
    r"access[_ -]?token|prompt|prompts|messages?|raw[_ -]?response|"
    r"response[_ -]?body|image[_ -]?(?:data|payload)|base64|html|mhtml|"
    r"page[_ -]?content)\b[\"']?\s*[:=].*$"
)
_TRACEBACK_RE = re.compile(
    r"(?is)\bTraceback\s*\(most recent call last\)\s*:.*"
)
_PROVIDER_BODY_REASON_RE = re.compile(
    r"(?is)(?:(?:provider|response|raw|upstream)[._ -]+body|"
    r"unknown[ _-]+raw[ _-]+body|raw[_ -]?provider[_ -]?body|"
    r"(?:^|\s)(?:body|response)\s*[:=]|"
    r"^\s*(?:\[\s*)?\{\s*[\"']?(?:detail|body|response)[\"']?\s*:|"
    r"^\s*[\[{].*(?:error|code|message))"
)
_EXPLICIT_AUTH_BILLING_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"\b(?:invalid|incorrect|missing|expired|revoked|disabled)\s+"
    r"(?:api[ _-]?key|credentials?|access[ _-]?token)\b"
    r"|\b(?:invalid|missing)[_\s-]+api[_\s-]+key\b"
    r"|\b(?:api[ _-]?key|credentials?|access[ _-]?token)\b.{0,40}"
    r"\b(?:invalid|incorrect|missing|expired|revoked|disabled)\b"
    r"|\b(?:unauthori[sz]ed|authentication\s+(?:failed|required))\b"
    r"|\baccount[ _-]+(?:is[ _-]+)?(?:disabled|suspended|deactivated|blocked|closed|inactive)\b"
    r"|\binsufficient\s+(?:account\s+)?(?:balance|funds?|credits?|quota)\b"
    r"|\b(?:balance|credits?|quota)\s+(?:is\s+)?"
    r"(?:exhausted|exceeded|depleted|empty)\b"
    r"|\bquota\s+limit\s+(?:reached|exceeded)\b"
    r"|\b(?:exceeded|exhausted)\s+(?:your\s+current\s+)?quota\b"
    r"|\binsufficient[_\s-]+quota\b"
    r"|\b(?:quota|credits?)[_\s-]+(?:exceeded|exhausted)\b"
    r"|\brun\s+out\s+of\s+credits?\b"
    r"|\bcredit\s+balance\s+is\s+too\s+low\b"
    r"|\bbilling\s+(?:account\s+)?(?:disabled|suspended|required|failed)\b"
    r"|\b[A-Za-z0-9]+_API_KEY\b.{0,60}\b"
    r"(?:is\s+not\s+set|not\s+configured|missing|required)\b"
    r")"
)
_AUTH_ACCOUNT_403_RE = re.compile(
    r"(?i)\b(?:auth(?:entication|orization)?|unauthori[sz]ed|credentials?|"
    r"api[ _-]?key|access[ _-]?token|account)\b"
)
_CONTENT_POLICY_RE = re.compile(
    r"(?i)\b(?:content[ _-]?(?:policy|filter(?:ed|ing)?)|moderation|"
    r"safety\s+(?:policy|refusal))\b"
)
_TRANSPORT_RE = re.compile(
    r"(?i)\b(?:timed?\s*out|timeout|transport|network|connection\s+"
    r"(?:reset|refused|aborted|closed)|dns|tls|socket|rate[ _-]?limit(?:ed|ing)?|"
    r"temporar(?:y|ily)\s+unavailable)\b"
)
_SAMPLE_PERMANENT_RE = re.compile(
    r"(?i)\b(?:content[ _-]?(?:policy|filter(?:ed|ing)?)|moderation|"
    r"safety\s+(?:policy|refusal)|"
    r"input|schema|invalid\s+(?:provider\s+)?(?:output|response|request)|"
    r"malformed\s+(?:output|response)|refus(?:al|ed)|"
    r"(?:for|on)\s+(?:this|the)\s+(?:input(?:\s+item)?|sample|item))\b"
)
_SAMPLE_ERROR_TYPE_RE = re.compile(
    r"(?i)(?:schema|moderation|repair|input.*validation)"
)
_CLASSIFICATION_MESSAGES = {
    "auth_billing": "authentication or billing failure",
    "api_budget": "physical API call budget exhausted",
    "contract_drift": "experiment contract drift",
    "transient": "transient provider or transport failure",
    "sample_permanent": "sample-specific permanent failure",
    "unexpected": "unexpected failure",
}
_ANALYSIS_PRIORITY_KEYS = (
    "code",
    "error_code",
    "status",
    "type",
    "message",
    "detail",
    "error",
)
_ANALYSIS_PRIORITY_KEY_SET = frozenset(_ANALYSIS_PRIORITY_KEYS)
EVENT_FIELDS = {
    "run_started": frozenset(
        {"config_sha256", "output_dir", "resume", "ack_breaker"}
    ),
    "resume_audit": frozenset(
        {
            "complete_samples",
            "partial_samples",
            "error_samples",
            "complete_stages",
            "error_stages",
            "reserved",
            "finished",
            "outstanding",
            "outstanding_by_stage",
            "replay_required_by_stage",
            "remaining_call_budget",
            "last_safe_sample_id",
            "last_safe_stage",
            "previous_stop_reason",
            "ledger_corrupt",
        }
    ),
    "sample_started": frozenset(
        {"sample_index", "sample_total", "sample_id"}
    ),
    "sample_complete": frozenset(
        {
            "sample_index",
            "sample_total",
            "sample_id",
            "result",
            "api_calls",
            "api_call_limit",
            "elapsed_seconds",
            "eta_seconds",
            "preliminary_eta_seconds",
            "health",
            "backend_error",
            "retry_count",
            "breaker_consecutive",
        }
    ),
    "stage_started": frozenset({"sample_id", "stage", "model"}),
    "stage_reused": frozenset({"sample_id", "stage", "model"}),
    "stage_completed": frozenset(
        {
            "sample_id",
            "stage",
            "model",
            "provider",
            "endpoint_host",
            "endpoint_sha256",
            "elapsed_seconds",
            "usage",
            "attempt_count",
        }
    ),
    "stage_failed": frozenset(
        {
            "sample_id",
            "stage",
            "model",
            "provider",
            "endpoint_host",
            "endpoint_sha256",
            "elapsed_seconds",
            "error_category",
            "error_type",
            "message",
        }
    ),
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
    "heartbeat": frozenset(
        {
            "sample_id",
            "stage",
            "wait_seconds",
            "api_calls",
            "api_call_limit",
            "outstanding",
            "health",
        }
    ),
    "summary": frozenset(
        {
            "completed_samples",
            "sample_total",
            "rate_per_minute",
            "eta_seconds",
            "preliminary_eta_seconds",
            "projected_completion",
            "remaining_cost_est",
            "cost_calculation",
            "health",
        }
    ),
    "prediction_flushed": frozenset(
        {"record_count", "path_sha256", "reason"}
    ),
    "stop_requested": frozenset({"signum", "boundary", "message"}),
    "breaker_opened": frozenset({"consecutive", "sample_id", "message"}),
    "breaker_reset_ack": frozenset({"reason", "prior_consecutive"}),
    "metrics_written": frozenset(
        {"sample_count", "gates_passed", "path_sha256"}
    ),
    "run_stopped": frozenset(
        {"reason", "category", "exit_code", "boundary", "forced", "message"}
    ),
    "run_completed": frozenset(
        {"sample_count", "api_calls", "gates_passed", "message"}
    ),
    "shell_started": frozenset(
        {"pid", "starttime", "runner_attempt_id", "message"}
    ),
    "shell_exited": frozenset(
        {"pid", "exit_code", "forced", "runner_attempt_id", "message"}
    ),
    "delivery_stage": frozenset(
        {
            "delivery_mode",
            "stage",
            "status",
            "exit_code",
            "archive_sha256",
            "runner_attempt_id",
            "message",
        }
    ),
    "warning": frozenset(
        {"category", "error_type", "http_status", "message"}
    ),
    "error": frozenset(
        {"category", "error_type", "http_status", "message"}
    ),
}

SHELL_EVENTS = frozenset({"shell_started", "shell_exited", "delivery_stage"})
_ATTEMPT_ID_LIMIT = 200


@dataclass(frozen=True)
class ObservabilitySettings:
    event_schema: str = EVENT_SCHEMA
    timezone: str = "Asia/Shanghai"
    heartbeat_poll_seconds: float = 30.0
    heartbeat_stall_seconds: float = 60.0
    summary_sample_interval: int = 25
    summary_seconds: float = 300.0
    eta_warmup_samples: int = 5
    eta_history_size: int = 50
    cost_full_update_samples: int = 25
    breaker_consecutive_backend_errors: int = 5

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "ObservabilitySettings":
        supplied = dict(values or {})
        allowed = {item.name for item in dataclasses.fields(cls)}
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(f"Unknown observability settings: {unknown}")
        return cls(**supplied)


@dataclass(frozen=True)
class StageObservation:
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class FailureClassification:
    category: str
    systemic: bool
    exit_code: int | None
    http_status: int | None
    message: str


class RunControlError(RuntimeError):
    exit_code = 1


class AuthBillingStop(RunControlError):
    exit_code = 20


class ApiBudgetStop(RunControlError):
    exit_code = 21


class CircuitBreakerStop(RunControlError):
    exit_code = 22


class ContractDriftStop(RunControlError):
    exit_code = 23


class BreakerAcknowledgementRequired(CircuitBreakerStop):
    pass


def _signal_exit_code(signum: int) -> int:
    if signum == signal.SIGINT:
        return 130
    if signum == signal.SIGTERM:
        return 143
    raise ValueError("signum must be SIGINT or SIGTERM")


class GracefulStopRequested(RunControlError):
    def __init__(self, signum: int, boundary: str):
        checked_boundary = _sanitize_control_text(boundary)
        if not checked_boundary:
            raise ValueError("boundary must be nonempty")
        self.signum = signum
        self.boundary = checked_boundary
        self.exit_code = _signal_exit_code(signum)
        super().__init__(
            f"graceful stop requested at {checked_boundary}"
        )


class StopController:
    _SIGNALS = (signal.SIGINT, signal.SIGTERM)

    def __init__(
        self,
        *,
        force_exit: Callable[[int], Any] = os._exit,
    ) -> None:
        if not callable(force_exit):
            raise ValueError("force_exit must be callable")
        self._force_exit = force_exit
        self._lock = threading.RLock()
        self._requested_signum: int | None = None
        self._forced = False
        self._previous_handlers: dict[int, Any] = {}
        self._installed = False

    @staticmethod
    def _require_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "signal handlers can only be changed on the main thread"
            )

    def handle_signal(self, signum: int, frame: object) -> None:
        _signal_exit_code(signum)
        force_code: int | None = None
        with self._lock:
            if self._requested_signum is None:
                self._requested_signum = signum
                return
            if self._forced:
                return
            self._forced = True
            force_code = _signal_exit_code(self._requested_signum)
        self._force_exit(force_code)

    def raise_if_requested(self, boundary: str) -> None:
        checked_boundary = _sanitize_control_text(boundary)
        if not checked_boundary:
            raise ValueError("boundary must be nonempty")
        with self._lock:
            signum = self._requested_signum
        if signum is not None:
            raise GracefulStopRequested(signum, checked_boundary)

    def install(self) -> None:
        self._require_main_thread()
        with self._lock:
            if self._installed:
                return
            if self._previous_handlers:
                raise RuntimeError(
                    "partial signal handler cleanup requires close()"
                )
        previous = {
            signum: signal.getsignal(signum) for signum in self._SIGNALS
        }
        installed: list[int] = []
        try:
            for signum in self._SIGNALS:
                signal.signal(signum, self.handle_signal)
                installed.append(signum)
        except Exception:
            failed_rollbacks: dict[int, Any] = {}
            for signum in reversed(installed):
                try:
                    signal.signal(signum, previous[signum])
                except Exception:
                    failed_rollbacks[signum] = previous[signum]
            with self._lock:
                self._previous_handlers = failed_rollbacks
                self._installed = False
            raise
        with self._lock:
            self._previous_handlers = previous
            self._installed = True

    def close(self) -> None:
        self._require_main_thread()
        with self._lock:
            if not self._previous_handlers:
                return
            previous = dict(self._previous_handlers)
        failed: dict[int, Any] = {}
        first_error: Exception | None = None
        for signum in reversed(self._SIGNALS):
            if signum not in previous:
                continue
            try:
                signal.signal(signum, previous[signum])
            except Exception as error:
                failed[signum] = previous[signum]
                if first_error is None:
                    first_error = error
        with self._lock:
            self._previous_handlers = failed
            self._installed = False
        if first_error is not None:
            raise first_error


def _positive_threshold(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("threshold must be a positive integer")
    return value


class FailureCircuitBreaker:
    def __init__(self, *, threshold: int = 5) -> None:
        self.threshold = _positive_threshold(threshold)
        self._lock = threading.RLock()
        self.consecutive = 0
        self.opened = False
        self._active_error_sample_ids: set[str] = set()

    @classmethod
    def from_events(
        cls,
        events: Sequence[Mapping[str, Any]],
        *,
        threshold: int = 5,
    ) -> "FailureCircuitBreaker":
        breaker = cls(threshold=threshold)
        seen_rows: set[tuple[str, int]] = set()
        for raw_event in events:
            if not isinstance(raw_event, Mapping):
                continue
            try:
                event = raw_event.get("event")
                attempt_id = raw_event.get("attempt_id")
                sequence = raw_event.get("sequence")
            except Exception:
                continue
            if type(event) is not str:
                continue
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

            if event == "breaker_reset_ack":
                try:
                    reason = raw_event.get("reason")
                    prior = raw_event.get("prior_consecutive")
                except Exception:
                    continue
                if (
                    type(reason) is not str
                    or not _sanitize_control_text(reason)
                    or type(prior) is not int
                    or prior < 0
                ):
                    continue
                with breaker._lock:
                    if (
                        not breaker.opened
                        or prior != breaker.consecutive
                    ):
                        continue
                    breaker.consecutive = 0
                    breaker.opened = False
                    breaker._active_error_sample_ids.clear()
                continue

            if event == "breaker_opened":
                try:
                    prior = raw_event.get("consecutive")
                except Exception:
                    prior = None
                restored_count = (
                    prior
                    if type(prior) is int and prior >= 1
                    else breaker.threshold
                )
                with breaker._lock:
                    breaker.consecutive = max(
                        breaker.consecutive, restored_count
                    )
                    breaker.opened = True
                continue

            if event != "sample_complete":
                continue
            try:
                sample_id = raw_event.get("sample_id")
                backend_error = raw_event.get("backend_error")
            except Exception:
                continue
            if (
                type(sample_id) is not str
                or not sample_id.strip()
                or type(backend_error) is not bool
            ):
                continue
            breaker.record_sample(
                sample_id=sample_id,
                backend_error=backend_error,
            )
        return breaker

    def record(self, *, backend_error: bool) -> bool:
        if type(backend_error) is not bool:
            raise ValueError("backend_error must be a boolean")
        with self._lock:
            if not backend_error:
                self.consecutive = 0
                self._active_error_sample_ids.clear()
                return False
            self.consecutive += 1
            if self.consecutive >= self.threshold:
                self.opened = True
            return self.opened

    def record_sample(self, *, sample_id: str, backend_error: bool) -> bool:
        if type(sample_id) is not str or not sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        if type(backend_error) is not bool:
            raise ValueError("backend_error must be a boolean")
        with self._lock:
            if not backend_error:
                self.consecutive = 0
                self._active_error_sample_ids.clear()
                return False
            if sample_id in self._active_error_sample_ids:
                return self.opened
            self._active_error_sample_ids.add(sample_id)
            self.consecutive += 1
            if self.consecutive >= self.threshold:
                self.opened = True
            return self.opened

    def require_start_allowed(self, *, acknowledged: bool) -> None:
        if type(acknowledged) is not bool:
            raise ValueError("acknowledged must be a boolean")
        with self._lock:
            if self.opened and not acknowledged:
                raise BreakerAcknowledgementRequired(
                    "prior circuit breaker opening requires explicit acknowledgement"
                )

    def acknowledge(self, *, reason: str) -> dict[str, Any]:
        checked_reason = _sanitize_breaker_ack_reason(reason)
        if not checked_reason:
            raise ValueError("reason must be nonempty")
        with self._lock:
            if not self.opened:
                raise ValueError("cannot acknowledge a closed breaker")
            event = {
                "event": "breaker_reset_ack",
                "reason": checked_reason,
                "prior_consecutive": self.consecutive,
            }
            self.consecutive = 0
            self.opened = False
            self._active_error_sample_ids.clear()
            return event


@dataclass(frozen=True)
class _CostObservation:
    currency: str
    amount: float


@dataclass(frozen=True)
class ProgressSnapshot:
    completed_samples: int
    current_sample_id: str | None
    current_stage: str | None
    api_calls: int
    outstanding_calls: int
    retries: int
    backend_errors: int
    eta_seconds: int | None
    preliminary_eta_seconds: int | None
    last_progress_at: float
    last_sample_completed_at: float


def _nonnegative_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a nonnegative finite number"
        ) from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return number


def _token_count(usage: Mapping[str, Any], *names: str) -> int:
    value: Any = 0
    for name in names:
        if name in usage:
            value = usage[name]
            break
    number = _nonnegative_finite(value, name=names[0])
    if not number.is_integer():
        raise ValueError(f"{names[0]} must be an integer")
    return int(number)


class ProgressTracker:
    def __init__(
        self,
        settings: ObservabilitySettings,
        *,
        total_samples: int,
        api_call_limit: int,
        pricing: Mapping[str, Mapping[str, Any]],
        text_model: str,
        vision_model: str,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(total_samples, bool) or total_samples < 0:
            raise ValueError("total_samples must be nonnegative")
        if isinstance(api_call_limit, bool) or api_call_limit < 0:
            raise ValueError("api_call_limit must be nonnegative")
        if settings.eta_history_size < 1:
            raise ValueError("eta_history_size must be positive")
        if settings.eta_warmup_samples < 0:
            raise ValueError("eta_warmup_samples must be nonnegative")
        if settings.cost_full_update_samples < 0:
            raise ValueError("cost_full_update_samples must be nonnegative")

        normalized_models = (str(text_model).strip(), str(vision_model).strip())
        if not all(normalized_models):
            raise ValueError("text_model and vision_model must be nonempty")
        if normalized_models[0] == normalized_models[1]:
            raise ValueError("text_model and vision_model must be distinct")

        model_pricing: dict[str, tuple[str, float, float]] = {}
        for role, model_name in zip(("text", "vision"), normalized_models):
            role_pricing = pricing.get(role)
            if not isinstance(role_pricing, Mapping):
                raise ValueError(f"pricing.{role} must be a mapping")
            currency = str(role_pricing.get("currency") or "").strip()
            if currency not in {"USD", "CNY"}:
                raise ValueError(
                    f"pricing.{role}.currency must be USD or CNY"
                )
            input_rate = _nonnegative_finite(
                role_pricing.get("input_per_million", 0),
                name=f"pricing.{role}.input_per_million",
            )
            output_rate = _nonnegative_finite(
                role_pricing.get("output_per_million", 0),
                name=f"pricing.{role}.output_per_million",
            )
            model_pricing[model_name] = (
                currency,
                input_rate,
                output_rate,
            )

        self.settings = settings
        self.total_samples = int(total_samples)
        self.api_call_limit = int(api_call_limit)
        self.pricing = pricing
        self.text_model, self.vision_model = normalized_models
        self._model_pricing = model_pricing
        self._currencies = frozenset(
            item[0] for item in model_pricing.values()
        )
        self.monotonic = monotonic

        self._lock = threading.RLock()
        history_size = settings.eta_history_size
        self._smoke_observations: dict[str, deque[StageObservation]] = (
            defaultdict(lambda: deque(maxlen=history_size))
        )
        self._full_observations: dict[str, deque[StageObservation]] = (
            defaultdict(lambda: deque(maxlen=history_size))
        )
        self._smoke_costs: dict[
            str, deque[_CostObservation | None]
        ] = defaultdict(lambda: deque(maxlen=history_size))
        self._full_costs: dict[
            str, deque[_CostObservation | None]
        ] = defaultdict(lambda: deque(maxlen=history_size))

        started_at = _nonnegative_finite(
            self.monotonic(), name="monotonic time"
        )
        self._last_progress_at = started_at
        self._last_sample_completed_at = started_at
        self._completed_samples = 0
        self._completed_sample_ids: set[str] = set()
        self._completed_sample_indexes: set[int] = set()
        self._successful_full_samples = 0
        self._current_sample_id: str | None = None
        self._current_sample_index: int | None = None
        self._current_stage: str | None = None
        self._api_calls = 0
        self._outstanding_calls = 0
        self._outstanding_by_stage: dict[str, int] = defaultdict(int)
        self._retries = 0
        self._backend_errors = 0

    def _checked_sample(self, sample_id: str, index: int) -> tuple[str, int]:
        checked_id = str(sample_id).strip()
        if not checked_id:
            raise ValueError("sample_id must be nonempty")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError("index must be a positive integer")
        if self.total_samples and index > self.total_samples:
            raise ValueError("index must not exceed total_samples")
        return checked_id, index

    @staticmethod
    def _checked_stage(stage: str) -> str:
        checked = str(stage).strip()
        if not checked:
            raise ValueError("stage must be nonempty")
        return checked

    def begin_sample(self, sample_id: str, *, index: int) -> None:
        checked_id, checked_index = self._checked_sample(sample_id, index)
        now = _nonnegative_finite(
            self.monotonic(), name="monotonic time"
        )
        with self._lock:
            if self._current_sample_id is not None:
                raise RuntimeError("a sample is already in progress")
            if (
                checked_id in self._completed_sample_ids
                or checked_index in self._completed_sample_indexes
            ):
                raise RuntimeError("sample ID or index already completed")
            self._current_sample_id = checked_id
            self._current_sample_index = checked_index
            self._current_stage = None
            self._last_progress_at = now

    def begin_stage(self, stage: str) -> None:
        checked_stage = self._checked_stage(stage)
        now = _nonnegative_finite(
            self.monotonic(), name="monotonic time"
        )
        with self._lock:
            if self._current_sample_id is None:
                raise RuntimeError("no sample is in progress")
            if self._outstanding_calls:
                raise RuntimeError("cannot begin a stage with an outstanding call")
            self._current_stage = checked_stage
            self._last_progress_at = now

    def reserve_call(self, stage: str) -> None:
        checked_stage = self._checked_stage(stage)
        now = _nonnegative_finite(
            self.monotonic(), name="monotonic time"
        )
        with self._lock:
            if self._current_sample_id is None:
                raise RuntimeError("no sample is in progress")
            if self._current_stage != checked_stage:
                raise RuntimeError("call stage does not match current stage")
            if self._api_calls >= self.api_call_limit:
                raise RuntimeError("api call limit exceeded")
            self._api_calls += 1
            self._outstanding_calls += 1
            self._outstanding_by_stage[checked_stage] += 1
            self._last_progress_at = now

    def finish_call(self, stage: str, *, transport_error: bool) -> None:
        checked_stage = self._checked_stage(stage)
        if not isinstance(transport_error, bool):
            raise ValueError("transport_error must be a boolean")
        now = _nonnegative_finite(
            self.monotonic(), name="monotonic time"
        )
        with self._lock:
            if self._outstanding_by_stage.get(checked_stage, 0) < 1:
                raise RuntimeError("no outstanding call for stage")
            self._outstanding_by_stage[checked_stage] -= 1
            if not self._outstanding_by_stage[checked_stage]:
                del self._outstanding_by_stage[checked_stage]
            self._outstanding_calls -= 1
            self._last_progress_at = now

    def record_retry(self) -> None:
        now = _nonnegative_finite(
            self.monotonic(), name="monotonic time"
        )
        with self._lock:
            if self._current_sample_id is None or self._current_stage is None:
                raise RuntimeError("no stage is in progress")
            self._retries += 1
            self._last_progress_at = now

    def record_stage(
        self,
        stage: str,
        *,
        model: str,
        usage: Mapping[str, Any],
        elapsed_seconds: float,
        reused: bool = False,
        source: Literal["smoke", "full"] = "full",
    ) -> None:
        stage_name = str(stage).strip()
        if not stage_name:
            raise ValueError("stage must be nonempty")
        if not isinstance(usage, Mapping):
            raise ValueError("usage must be a mapping")
        elapsed = _nonnegative_finite(
            elapsed_seconds, name="elapsed_seconds"
        )
        if source not in {"smoke", "full"}:
            raise ValueError("source must be 'smoke' or 'full'")
        observation = StageObservation(
            elapsed_seconds=elapsed,
            input_tokens=_token_count(
                usage, "prompt_tokens", "input_tokens"
            ),
            output_tokens=_token_count(
                usage, "completion_tokens", "output_tokens"
            ),
        )
        model_name = str(model).strip()
        try:
            currency, input_rate, output_rate = self._model_pricing[
                model_name
            ]
        except KeyError as error:
            raise ValueError(f"unknown configured model: {model_name!r}") from error
        has_input_usage = any(
            name in usage
            for name in ("prompt_tokens", "input_tokens")
        )
        has_output_usage = any(
            name in usage
            for name in ("completion_tokens", "output_tokens")
        )
        cost_observation = (
            _CostObservation(
                currency=currency,
                amount=(
                    observation.input_tokens * input_rate
                    + observation.output_tokens * output_rate
                )
                / 1_000_000,
            )
            if has_input_usage and has_output_usage
            else None
        )
        now = _nonnegative_finite(
            self.monotonic(), name="monotonic time"
        )
        with self._lock:
            self._last_progress_at = now
            if reused:
                return
            if source == "smoke":
                self._smoke_observations[stage_name].append(observation)
                self._smoke_costs[stage_name].append(cost_observation)
            else:
                self._full_observations[stage_name].append(observation)
                self._full_costs[stage_name].append(cost_observation)

    def complete_sample(
        self,
        sample_id: str,
        *,
        index: int,
        backend_error: bool,
    ) -> None:
        checked_id, checked_index = self._checked_sample(sample_id, index)
        if not isinstance(backend_error, bool):
            raise ValueError("backend_error must be a boolean")
        now = _nonnegative_finite(
            self.monotonic(), name="monotonic time"
        )
        with self._lock:
            if (
                self._current_sample_id is not None
                and (
                    checked_id != self._current_sample_id
                    or checked_index != self._current_sample_index
                )
            ):
                raise RuntimeError("completed sample does not match current sample")
            if self._outstanding_calls:
                raise RuntimeError(
                    "cannot complete a sample with outstanding calls"
                )
            if (
                checked_id in self._completed_sample_ids
                or checked_index in self._completed_sample_indexes
            ):
                raise RuntimeError("sample ID or index already completed")
            if self._completed_samples >= self.total_samples:
                raise RuntimeError("completed samples exceed total_samples")
            self._completed_sample_ids.add(checked_id)
            self._completed_sample_indexes.add(checked_index)
            self._completed_samples += 1
            if backend_error:
                self._backend_errors += 1
            else:
                self._successful_full_samples += 1
            self._current_sample_id = None
            self._current_sample_index = None
            self._current_stage = None
            self._last_progress_at = now
            self._last_sample_completed_at = now

    @staticmethod
    def _mean_elapsed(
        observations: Sequence[StageObservation],
    ) -> float | None:
        if not observations:
            return None
        return sum(item.elapsed_seconds for item in observations) / len(
            observations
        )

    def _eta_locked(
        self,
        remaining_stage_counts: Mapping[str, int],
        *,
        prefer_full: bool,
    ) -> int | None:
        estimate = 0.0
        for raw_stage, raw_count in remaining_stage_counts.items():
            stage = str(raw_stage).strip()
            if not stage:
                raise ValueError("remaining stage names must be nonempty")
            if isinstance(raw_count, bool):
                raise ValueError("remaining stage counts must be nonnegative")
            count = _nonnegative_finite(
                raw_count, name="remaining stage count"
            )
            if not count.is_integer():
                raise ValueError("remaining stage counts must be integers")
            if not count:
                continue
            primary = (
                self._full_observations.get(stage, ())
                if prefer_full
                else self._smoke_observations.get(stage, ())
            )
            fallback = (
                self._smoke_observations.get(stage, ())
                if prefer_full
                else ()
            )
            mean = self._mean_elapsed(primary)
            if mean is None:
                mean = self._mean_elapsed(fallback)
            if mean is None:
                return None
            estimate += count * mean
        return int(round(estimate))

    def snapshot(
        self,
        *,
        remaining_stage_counts: Mapping[str, int] | None = None,
    ) -> ProgressSnapshot:
        with self._lock:
            eta_seconds: int | None = None
            preliminary_eta_seconds: int | None = None
            if remaining_stage_counts is not None:
                preliminary_eta_seconds = self._eta_locked(
                    remaining_stage_counts, prefer_full=False
                )
                if (
                    self._successful_full_samples
                    >= self.settings.eta_warmup_samples
                ):
                    eta_seconds = self._eta_locked(
                        remaining_stage_counts, prefer_full=True
                    )
            return ProgressSnapshot(
                completed_samples=self._completed_samples,
                current_sample_id=self._current_sample_id,
                current_stage=self._current_stage,
                api_calls=self._api_calls,
                outstanding_calls=self._outstanding_calls,
                retries=self._retries,
                backend_errors=self._backend_errors,
                eta_seconds=eta_seconds,
                preliminary_eta_seconds=preliminary_eta_seconds,
                last_progress_at=self._last_progress_at,
                last_sample_completed_at=self._last_sample_completed_at,
            )

    @staticmethod
    def _validated_counts(
        counts: Mapping[str, int], *, label: str
    ) -> dict[str, int]:
        if not isinstance(counts, Mapping):
            raise ValueError(f"{label} must be a mapping")
        validated: dict[str, int] = {}
        for raw_stage, raw_count in counts.items():
            stage = str(raw_stage).strip()
            if not stage:
                raise ValueError(f"{label} stage names must be nonempty")
            if isinstance(raw_count, bool):
                raise ValueError(f"{label} counts must be nonnegative integers")
            count = _nonnegative_finite(raw_count, name=f"{label} count")
            if not count.is_integer():
                raise ValueError(f"{label} counts must be integers")
            validated[stage] = int(count)
        return validated

    @staticmethod
    def _mean_costs(
        observations: Sequence[_CostObservation | None],
    ) -> dict[str, float]:
        available = [item for item in observations if item is not None]
        if not available:
            return {}
        totals: dict[str, float] = defaultdict(float)
        for observation in available:
            totals[observation.currency] += observation.amount
        sample_count = len(available)
        return {
            currency: amount / sample_count
            for currency, amount in totals.items()
        }

    def estimate_remaining_cost(
        self,
        remaining_stage_counts: Mapping[str, int],
        *,
        outstanding_by_stage: Mapping[str, int],
        replay_required_by_stage: Mapping[str, int],
    ) -> dict[str, float]:
        remaining = self._validated_counts(
            remaining_stage_counts, label="remaining_stage_counts"
        )
        outstanding = self._validated_counts(
            outstanding_by_stage, label="outstanding_by_stage"
        )
        replay = self._validated_counts(
            replay_required_by_stage,
            label="replay_required_by_stage",
        )
        stages = set(remaining) | set(outstanding) | set(replay)
        with self._lock:
            use_full = (
                self._successful_full_samples
                >= self.settings.cost_full_update_samples
            )
            estimate = {currency: 0.0 for currency in self._currencies}
            for stage in stages:
                call_count = (
                    remaining.get(stage, 0)
                    + outstanding.get(stage, 0)
                    + replay.get(stage, 0)
                )
                if not call_count:
                    continue
                primary = (
                    self._full_costs.get(stage, ())
                    if use_full
                    else self._smoke_costs.get(stage, ())
                )
                fallback = (
                    self._smoke_costs.get(stage, ())
                    if use_full
                    else ()
                )
                mean_costs = self._mean_costs(primary)
                if not mean_costs:
                    mean_costs = self._mean_costs(fallback)
                if not mean_costs:
                    raise ValueError(
                        "no cost observation available for requested stage"
                    )
                for currency, mean_cost in mean_costs.items():
                    estimate[currency] = (
                        estimate.get(currency, 0.0)
                        + call_count * mean_cost
                    )
            return estimate


_BASE_STAGES = ("url", "text", "visual", "cross", "binary")
_DOWNSTREAM_STAGES = ("type", "grounding")


def project_remaining_stage_counts(
    manifest: Sequence[Mapping[str, Any]],
    *,
    complete_stages: frozenset[tuple[str, str]],
    binary_outputs: Mapping[str, str],
    observed_benign_false_positive_rate: float | None,
    smoke_benign_false_positive_rate: float,
) -> dict[str, int]:
    smoke_rate = _nonnegative_finite(
        smoke_benign_false_positive_rate,
        name="smoke_benign_false_positive_rate",
    )
    if smoke_rate > 1:
        raise ValueError("smoke_benign_false_positive_rate must not exceed 1")
    if observed_benign_false_positive_rate is None:
        rate = smoke_rate
    else:
        rate = _nonnegative_finite(
            observed_benign_false_positive_rate,
            name="observed_benign_false_positive_rate",
        )
        if rate > 1:
            raise ValueError(
                "observed_benign_false_positive_rate must not exceed 1"
            )

    valid_binary_labels = frozenset({"benign", "scam"})
    rows: list[tuple[str, str]] = []
    manifest_ids: set[str] = set()
    for record in manifest:
        if not isinstance(record, Mapping):
            raise ValueError("manifest records must be mappings")
        sample_id = str(
            record.get("sample_id") or record.get("id") or ""
        ).strip()
        if not sample_id:
            raise ValueError("manifest records must have a sample ID")
        if sample_id in manifest_ids:
            raise ValueError(f"duplicate manifest sample ID: {sample_id!r}")
        truth_binary = record.get("binary")
        if truth_binary not in valid_binary_labels:
            raise ValueError(
                f"invalid manifest binary for sample {sample_id!r}"
            )
        manifest_ids.add(sample_id)
        rows.append((sample_id, str(truth_binary)))

    if not isinstance(binary_outputs, Mapping):
        raise ValueError("binary_outputs must be a mapping")
    normalized_binary_outputs: dict[str, str] = {}
    for raw_sample_id, observed_binary in binary_outputs.items():
        sample_id = str(raw_sample_id).strip()
        if sample_id not in manifest_ids:
            raise ValueError(f"unknown binary output sample ID: {sample_id!r}")
        if observed_binary not in valid_binary_labels:
            raise ValueError(
                f"invalid binary output for sample {sample_id!r}"
            )
        if sample_id in normalized_binary_outputs:
            raise ValueError(f"duplicate binary output sample ID: {sample_id!r}")
        normalized_binary_outputs[sample_id] = str(observed_binary)

    counts = {
        stage: sum(
            (sample_id, stage) not in complete_stages
            for sample_id, _ in rows
        )
        for stage in _BASE_STAGES
    }
    known_downstream_ids = {
        sample_id
        for sample_id, truth_binary in rows
        if truth_binary == "scam"
        or (
            truth_binary == "benign"
            and normalized_binary_outputs.get(sample_id) == "scam"
        )
    }
    unresolved_benign_ids = {
        sample_id
        for sample_id, truth_binary in rows
        if (
            truth_binary == "benign"
            and sample_id not in normalized_binary_outputs
        )
    }
    for stage in _DOWNSTREAM_STAGES:
        known_remaining = sum(
            (sample_id, stage) not in complete_stages
            for sample_id in known_downstream_ids
        )
        unresolved_remaining = sum(
            (sample_id, stage) not in complete_stages
            for sample_id in unresolved_benign_ids
        )
        counts[stage] = known_remaining + math.ceil(
            unresolved_remaining * rate
        )
    return counts


class SummaryScheduler:
    def __init__(
        self,
        settings: ObservabilitySettings,
        *,
        started_at: float,
    ) -> None:
        if settings.summary_sample_interval < 1:
            raise ValueError("summary_sample_interval must be positive")
        if settings.summary_seconds <= 0 or not math.isfinite(
            settings.summary_seconds
        ):
            raise ValueError("summary_seconds must be positive and finite")
        self.settings = settings
        checked_started_at = _nonnegative_finite(started_at, name="started_at")
        self._last_emitted_at = checked_started_at
        self._last_completed_samples = 0
        self._last_seen_at = checked_started_at
        self._last_seen_completed_samples = 0
        self._lock = threading.Lock()

    def should_emit(self, *, completed_samples: int, now: float) -> bool:
        if isinstance(completed_samples, bool) or completed_samples < 0:
            raise ValueError("completed_samples must be nonnegative")
        checked_now = _nonnegative_finite(now, name="now")
        with self._lock:
            if completed_samples < self._last_seen_completed_samples:
                raise ValueError("completed_samples must not decrease")
            if checked_now < self._last_seen_at:
                raise ValueError("now must not move backwards")
            self._last_seen_completed_samples = int(completed_samples)
            self._last_seen_at = checked_now
            count_due = (
                completed_samples - self._last_completed_samples
                >= self.settings.summary_sample_interval
            )
            time_due = (
                checked_now - self._last_emitted_at
                >= self.settings.summary_seconds
            )
            if not (count_due or time_due):
                return False
            self._last_completed_samples = int(completed_samples)
            self._last_emitted_at = checked_now
            return True


def _snapshot_field(
    snapshot: Mapping[str, Any] | ProgressSnapshot,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if isinstance(snapshot, Mapping):
            if name in snapshot:
                return snapshot[name]
        elif hasattr(snapshot, name):
            return getattr(snapshot, name)
    return default


class HeartbeatWorker:
    def __init__(
        self,
        settings: ObservabilitySettings,
        *,
        snapshot: Callable[[], Mapping[str, Any] | ProgressSnapshot],
        emit: Callable[..., Any],
        monotonic: Callable[[], float] = time.monotonic,
        on_poll: Callable[[float], None] | None = None,
        api_call_limit: int | None = None,
    ) -> None:
        if settings.heartbeat_poll_seconds <= 0 or not math.isfinite(
            settings.heartbeat_poll_seconds
        ):
            raise ValueError(
                "heartbeat_poll_seconds must be positive and finite"
            )
        if settings.heartbeat_stall_seconds <= 0 or not math.isfinite(
            settings.heartbeat_stall_seconds
        ):
            raise ValueError(
                "heartbeat_stall_seconds must be positive and finite"
            )
        if not callable(snapshot) or not callable(emit):
            raise ValueError("snapshot and emit must be callable")
        if on_poll is not None and not callable(on_poll):
            raise ValueError("on_poll must be callable")
        snapshot_owner = getattr(snapshot, "__self__", None)
        if isinstance(snapshot_owner, ProgressTracker):
            api_call_limit = snapshot_owner.api_call_limit
        elif api_call_limit is None:
            api_call_limit = 50_000
        checked_limit = _nonnegative_finite(
            api_call_limit, name="api_call_limit"
        )
        if not checked_limit.is_integer():
            raise ValueError("api_call_limit must be an integer")
        api_call_limit = int(checked_limit)
        self.settings = settings
        self.snapshot = snapshot
        self.emit = emit
        self.monotonic = monotonic
        self.on_poll = on_poll
        self.api_call_limit = api_call_limit

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._join_started = False
        self._last_heartbeat_at: float | None = None
        self._last_heartbeat_completion_at: float | None = None

    def tick(self) -> None:
        now = _nonnegative_finite(self.monotonic(), name="monotonic time")
        if self.on_poll is not None:
            try:
                self.on_poll(now)
            except Exception:
                pass

        current = self.snapshot()
        if not isinstance(current, Mapping) and not isinstance(
            current, ProgressSnapshot
        ):
            raise ValueError("heartbeat snapshot must be a mapping or snapshot")
        completed_at = _nonnegative_finite(
            _snapshot_field(current, "last_sample_completed_at"),
            name="last_sample_completed_at",
        )
        wait_seconds = max(0.0, now - completed_at)
        if wait_seconds < self.settings.heartbeat_stall_seconds:
            return

        with self._lock:
            same_completion = (
                self._last_heartbeat_completion_at == completed_at
            )
            emitted_too_recently = (
                self._last_heartbeat_at is not None
                and now - self._last_heartbeat_at
                < self.settings.heartbeat_stall_seconds
            )
            if same_completion and emitted_too_recently:
                return
            self._last_heartbeat_at = now
            self._last_heartbeat_completion_at = completed_at

        outstanding = int(
            _nonnegative_finite(
                _snapshot_field(
                    current,
                    "outstanding",
                    "outstanding_calls",
                    default=0,
                ),
                name="outstanding",
            )
        )
        api_calls = int(
            _nonnegative_finite(
                _snapshot_field(current, "api_calls", default=0),
                name="api_calls",
            )
        )
        raw_api_call_limit = _snapshot_field(
            current,
            "api_call_limit",
            default=self.api_call_limit,
        )
        if raw_api_call_limit is None:
            raise ValueError("heartbeat snapshot has no api_call_limit context")
        api_call_limit = int(
            _nonnegative_finite(
                raw_api_call_limit,
                name="api_call_limit",
            )
        )
        self.emit(
            "heartbeat",
            sample_id=_snapshot_field(
                current, "sample_id", "current_sample_id"
            ),
            stage=_snapshot_field(current, "stage", "current_stage"),
            wait_seconds=int(wait_seconds),
            api_calls=api_calls,
            api_call_limit=api_call_limit,
            outstanding=outstanding,
            health="waiting_api" if outstanding else "degraded",
        )

    def _run(self) -> None:
        while not self._stop.wait(self.settings.heartbeat_poll_seconds):
            try:
                self.tick()
            except Exception:
                continue

    def start(self) -> None:
        with self._lock:
            if self._closed or self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="scammark-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop.set()
            thread = self._thread
            should_join = (
                thread is not None
                and thread is not threading.current_thread()
                and not self._join_started
            )
            if should_join:
                self._join_started = True
        if should_join:
            thread.join()


def sanitize_text(value: object, *, limit: int = 500) -> str:
    text = str(value or "")
    text = AUTH_RE.sub("<redacted-auth>", text)
    text = KEY_RE.sub("<redacted-key>", text)
    text = DATA_URL_RE.sub("<redacted-image>", text)
    text = WORKSPACE_RE.sub("*.cn-beijing.maas.aliyuncs.com", text)
    text = QUERY_RE.sub(lambda match: match.group(1), text)
    return text[:limit]


def sanitize_payload(value: Any, *, key: str | None = None) -> Any:
    normalized = str(key or "").lower().replace("-", "_")
    if normalized in SENSITIVE_FIELDS:
        return "<redacted-field>"
    if isinstance(value, Mapping):
        return {
            str(name): sanitize_payload(item, key=str(name))
            for name, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(value)


def _sanitize_control_text(value: object) -> str:
    try:
        raw = str(value or "")
    except Exception:
        return ""
    text = _TRACEBACK_RE.sub("<redacted-traceback>", raw)
    text = _FAILURE_SECRET_TAIL_RE.sub("<redacted-field>", text)
    text = _FAILURE_SECRET_FIELD_RE.sub("<redacted-field>", text)
    text = _ENDPOINT_RE.sub("<redacted-endpoint>", text)
    text = sanitize_text(text)
    text = "".join(
        character if character.isprintable() else " "
        for character in text
    )
    return " ".join(text.split())


def _sanitize_breaker_ack_reason(reason: object) -> str:
    if type(reason) is not str:
        return ""
    matched_status = (
        _ACK_HTTP_STATUS_RE.search(reason)
        or _HTTP_STATUS_RE.search(reason)
    )
    if matched_status is not None:
        return (
            "operator acknowledged breaker after "
            f"HTTP {matched_status.group(1)} failure"
        )
    if _PROVIDER_BODY_REASON_RE.search(reason):
        return "operator acknowledged breaker after provider failure"
    return _sanitize_control_text(reason)


def _project_analysis_structure(
    value: object,
    *,
    depth: int,
    ancestors: set[int],
    budget: list[int],
) -> Any:
    if budget[0] <= 0:
        return "<truncated>"
    budget[0] -= 1
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value if value.bit_length() <= 256 else "<integer-too-large>"
    if type(value) is float:
        return value if math.isfinite(value) else None
    if type(value) is str:
        return value[:600]
    if type(value) is bytes:
        return value[:600].decode("utf-8", errors="replace")
    if depth >= 6 or type(value) not in {dict, list, tuple}:
        return "<unsupported>"

    identity = id(value)
    if identity in ancestors:
        return "<cycle>"
    ancestors.add(identity)
    try:
        if type(value) is dict:
            projected: dict[str, Any] = {}
            complex_priorities: list[tuple[str, object]] = []
            complex_results: list[tuple[int, int, str, Any]] = []
            for key in _ANALYSIS_PRIORITY_KEYS:
                try:
                    item = value[key]
                except KeyError:
                    continue
                except Exception:
                    projected[key] = "<unavailable>"
                    continue
                if item is None or type(item) in {
                    bool,
                    int,
                    float,
                    str,
                    bytes,
                }:
                    projected[key] = _project_analysis_structure(
                        item,
                        depth=depth + 1,
                        ancestors=ancestors,
                        budget=budget,
                    )
                else:
                    complex_priorities.append((key, item))
            for index, (key, item) in enumerate(complex_priorities):
                remaining_priorities = len(complex_priorities) - index
                local_limit = budget[0] // remaining_priorities
                local_budget = [local_limit]
                projected_item = _project_analysis_structure(
                    item,
                    depth=depth + 1,
                    ancestors=ancestors,
                    budget=local_budget,
                )
                used = local_limit - local_budget[0]
                budget[0] -= used
                complex_results.append((used, index, key, projected_item))
            for _, _, key, projected_item in sorted(complex_results):
                projected[key] = projected_item
            scanned = 0
            remaining = 24
            for key, item in value.items():
                if scanned >= 24 + len(_ANALYSIS_PRIORITY_KEYS):
                    break
                scanned += 1
                if (
                    type(key) is not str
                    or key in _ANALYSIS_PRIORITY_KEY_SET
                ):
                    continue
                projected.setdefault(
                    key[:120],
                    _project_analysis_structure(
                        item,
                        depth=depth + 1,
                        ancestors=ancestors,
                        budget=budget,
                    ),
                )
                remaining -= 1
                if remaining <= 0:
                    break
            if len(value) > scanned:
                projected.setdefault("<truncated>", True)
            return projected
        projected_items = [
            _project_analysis_structure(
                item,
                depth=depth + 1,
                ancestors=ancestors,
                budget=budget,
            )
            for item in value[:24]
        ]
        if len(value) > 24:
            projected_items.append("<truncated>")
        return projected_items
    finally:
        ancestors.remove(identity)


def _bounded_analysis_text(value: object, *, limit: int) -> str:
    if type(value) is bytes:
        value = value[:limit].decode("utf-8", errors="replace")
    elif type(value) is str:
        value = value[:limit]
    else:
        if type(value) not in {dict, list, tuple, bool, int, float}:
            return ""
        try:
            projected = _project_analysis_structure(
                value,
                depth=0,
                ancestors=set(),
                budget=[128],
            )
            value = json.dumps(
                projected,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except Exception:
            value = "<unavailable structured provider detail>"
        value = value[:limit]
    text = "".join(
        character if character.isprintable() else " "
        for character in value
    )
    return " ".join(text.split())[:limit]


def _exception_analysis_text(error: BaseException) -> str:
    try:
        raw = str(error)
    except Exception:
        raw = "<unprintable exception>"
    primary = (
        _bounded_analysis_text(raw, limit=1000)
        or "<empty exception message>"
    )
    parts = [primary]
    seen = {primary}
    try:
        response = getattr(error, "response", None)
    except Exception:
        response = None
    for source, attributes in (
        (error, ("body", "response_body", "text")),
        (response, ("body", "text")),
    ):
        if source is None:
            continue
        for attribute in attributes:
            try:
                value = getattr(source, attribute, None)
            except Exception:
                continue
            bounded = _bounded_analysis_text(value, limit=600)
            if bounded and bounded not in seen:
                parts.append(bounded)
                seen.add(bounded)
    return " ".join(parts)[:4000]


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    pending = deque([error])
    seen: set[int] = set()
    ordered: list[BaseException] = []
    while pending:
        current = pending.popleft()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append(current)
        for attribute in ("__cause__", "__context__"):
            try:
                linked = getattr(current, attribute, None)
            except Exception:
                linked = None
            if isinstance(linked, BaseException) and id(linked) not in seen:
                pending.append(linked)
    return tuple(ordered)


def _valid_http_status(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 100 <= value <= 599 else None


def _http_status(error: BaseException, message: str) -> int | None:
    for attribute in ("status_code", "http_status"):
        try:
            status = _valid_http_status(getattr(error, attribute, None))
        except Exception:
            status = None
        if status is not None:
            return status
    try:
        response = getattr(error, "response", None)
    except Exception:
        response = None
    if response is not None:
        for attribute in ("status_code", "http_status"):
            try:
                status = _valid_http_status(
                    getattr(response, attribute, None)
                )
            except Exception:
                status = None
            if status is not None:
                return status
    matched = _HTTP_STATUS_RE.search(message)
    return int(matched.group(1)) if matched is not None else None


def _classification(
    category: str,
    systemic: bool,
    exit_code: int | None,
    error: BaseException,
    analysis_message: str,
) -> FailureClassification:
    status = _http_status(error, analysis_message)
    return FailureClassification(
        category=category,
        systemic=systemic,
        exit_code=exit_code,
        http_status=status,
        message=(
            f"HTTP {status}: {_CLASSIFICATION_MESSAGES[category]}"
            if status is not None
            else _CLASSIFICATION_MESSAGES[category]
        ),
    )


def classify_failure(error: BaseException) -> FailureClassification:
    from .experiment_state import ApiCallLimitError, ExperimentContractError
    from .reliability import (
        PermanentRequestError,
        TransientRequestError,
        provider_429_failure_category,
    )

    chain = _exception_chain(error)
    details = tuple(
        (item, _exception_analysis_text(item))
        for item in chain
    )
    provider_429_categories = {
        id(item): provider_429_failure_category(analysis_message)
        for item, analysis_message in details
        if _http_status(item, analysis_message) == 429
    }

    for item, analysis_message in details:
        if isinstance(item, ApiCallLimitError):
            return _classification(
                "api_budget",
                True,
                21,
                item,
                analysis_message,
            )
    for item, analysis_message in details:
        if isinstance(item, ExperimentContractError):
            return _classification(
                "contract_drift",
                True,
                23,
                item,
                analysis_message,
            )
    for item, analysis_message in details:
        if _http_status(item, analysis_message) in {401, 402}:
            return _classification(
                "auth_billing",
                True,
                20,
                item,
                analysis_message,
            )
    for item, analysis_message in details:
        if provider_429_categories.get(id(item)) == "auth_billing":
            return _classification(
                "auth_billing",
                True,
                20,
                item,
                analysis_message,
            )
    for item, analysis_message in details:
        status = _http_status(item, analysis_message)
        if (
            _EXPLICIT_AUTH_BILLING_RE.search(analysis_message)
            and not (
                status == 429
                and provider_429_categories.get(id(item)) == "transient"
            )
            and not (
                status == 403
                and _CONTENT_POLICY_RE.search(analysis_message)
            )
        ):
            return _classification(
                "auth_billing",
                True,
                20,
                item,
                analysis_message,
            )
    for item, analysis_message in details:
        if (
            _http_status(item, analysis_message) == 403
            and _AUTH_ACCOUNT_403_RE.search(analysis_message)
            and not _CONTENT_POLICY_RE.search(analysis_message)
        ):
            return _classification(
                "auth_billing",
                True,
                20,
                item,
                analysis_message,
            )
    for item, analysis_message in details:
        if _http_status(item, analysis_message) == 403:
            return _classification(
                "sample_permanent",
                False,
                None,
                item,
                analysis_message,
            )
    for item, analysis_message in details:
        status = _http_status(item, analysis_message)
        type_name = type(item).__name__.lower()
        if (
            status in {408, 429}
            or status is not None
            and 500 <= status <= 599
            or isinstance(item, (TransientRequestError, TimeoutError, ConnectionError))
            or "timeout" in type_name
            or "transport" in type_name
            or _TRANSPORT_RE.search(analysis_message)
        ):
            return _classification(
                "transient",
                False,
                None,
                item,
                analysis_message,
            )
    for item, analysis_message in details:
        status = _http_status(item, analysis_message)
        if (
            _SAMPLE_PERMANENT_RE.search(analysis_message)
            or _SAMPLE_ERROR_TYPE_RE.search(type(item).__name__)
            or isinstance(item, PermanentRequestError)
            and status == 403
        ):
            return _classification(
                "sample_permanent",
                False,
                None,
                item,
                analysis_message,
            )

    item, analysis_message = details[0]
    return _classification(
        "unexpected",
        False,
        None,
        item,
        analysis_message,
    )


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_human_event(event: Mapping[str, Any]) -> str | None:
    name = str(event.get("event") or "")
    stamp = str(event.get("timestamp_local") or "")
    if name == "sample_complete":
        index, total = int(event["sample_index"]), int(event["sample_total"])
        filled = min(20, int(20 * index / max(total, 1)))
        bar = "#" * filled + "-" * (20 - filled)
        eta = event.get("eta_seconds")
        eta_text = "warming_up" if eta is None else format_duration(float(eta))
        suffix = ""
        if int(event.get("retry_count") or 0):
            suffix += f" retries={int(event['retry_count'])}"
        if bool(event.get("backend_error")):
            suffix += " errors=1"
        if int(event.get("breaker_consecutive") or 0):
            suffix += f" breaker={int(event['breaker_consecutive'])}"
        return (
            f"[{stamp}] [{bar}] {index}/{total} {100 * index / total:.2f}% "
            f"id={event['sample_id']} result={event['result']} "
            f"calls={event['api_calls']}/{event['api_call_limit']} "
            f"elapsed={format_duration(float(event.get('elapsed_seconds') or 0))} "
            f"eta~{eta_text} health={event['health']}{suffix}"
        )
    if name == "heartbeat":
        return (
            f"HEARTBEAT id={event.get('sample_id')} stage={event.get('stage')} "
            f"waiting={int(event['wait_seconds'])}s "
            f"calls={event.get('api_calls')}/{event.get('api_call_limit')} "
            f"outstanding={event.get('outstanding')} health={event['health']}"
        )
    if name == "summary":
        raw_costs = event.get("remaining_cost_est")
        costs = raw_costs if isinstance(raw_costs, Mapping) else {}
        cost_parts: list[str] = []
        for currency in ("USD", "CNY"):
            if currency not in costs:
                continue
            try:
                amount = _nonnegative_finite(
                    costs[currency], name=f"remaining {currency} cost"
                )
            except ValueError:
                continue
            cost_parts.append(f"{currency} {amount:.2f}")
        cost_text = " + ".join(cost_parts)
        eta = event.get("eta_seconds")
        eta_text = "warming_up" if eta is None else format_duration(float(eta))
        return (
            f"SUMMARY complete={event['completed_samples']}/{event['sample_total']} "
            f"rate={float(event.get('rate_per_minute') or 0):.2f}/min "
            f"eta~{eta_text} remaining_cost_est~{cost_text}"
        )
    if name in {
        "warning",
        "error",
        "breaker_opened",
        "run_stopped",
        "run_completed",
        "shell_exited",
        "delivery_stage",
    }:
        details = [
            str(value)
            for value in (
                event.get("category"),
                event.get("http_status"),
                event.get("error_type"),
                event.get("stage"),
                event.get("status"),
            )
            if value not in (None, "")
        ]
        message = sanitize_text(event.get("message") or event.get("reason") or "")
        return f"[{stamp}] {name.upper()} {' '.join(details)} {message}".rstrip()
    return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunEventLogger:
    def __init__(
        self,
        path: Path,
        settings: ObservabilitySettings,
        *,
        stream: TextIO | None = sys.stdout,
        utc_now: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        attempt_id: str | None = None,
        experiment_version: str = "full-v2",
        run_mode: str = "full",
    ) -> None:
        self.path = Path(path)
        self.settings = settings
        self.stream = stream
        self.utc_now = utc_now
        self.monotonic = monotonic
        self.experiment_version = str(experiment_version)
        self.run_mode = str(run_mode)

        selected_attempt_id = attempt_id
        if selected_attempt_id is None:
            shell_attempt_id = os.environ.get("SCAMMARK_ATTEMPT_ID", "")
            selected_attempt_id = (
                shell_attempt_id.strip()
                if shell_attempt_id.strip()
                else uuid.uuid4().hex
            )
        if not str(selected_attempt_id).strip():
            raise ValueError("attempt_id must be nonempty")
        self.attempt_id = str(selected_attempt_id).strip()

        self._local_timezone = ZoneInfo(settings.timezone)
        self._started_at = float(self.monotonic())
        self._sequence = 0
        self._lock = threading.Lock()

    def emit(self, event: str, *, critical: bool = False, **fields: Any) -> None:
        if event not in EVENT_FIELDS:
            raise ValueError(f"Unknown event: {event!r}")
        if "health" in fields and fields["health"] not in HEALTH_VALUES:
            raise ValueError(f"Unknown health value: {fields['health']!r}")

        allowed_fields = EVENT_FIELDS[event]
        sanitized_fields = {
            name: sanitize_payload(value, key=name)
            for name, value in fields.items()
            if name in allowed_fields
        }

        with self._lock:
            self._sequence += 1
            now = self.utc_now()
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            now_utc = now.astimezone(timezone.utc)
            elapsed = max(0.0, float(self.monotonic()) - self._started_at)
            payload: dict[str, Any] = {
                "schema": self.settings.event_schema,
                "event": event,
                "experiment_version": sanitize_text(
                    self.experiment_version, limit=100
                ),
                "run_mode": sanitize_text(self.run_mode, limit=100),
                "attempt_id": sanitize_text(
                    self.attempt_id, limit=_ATTEMPT_ID_LIMIT
                ),
                "sequence": self._sequence,
                "timestamp_utc": now_utc.isoformat(timespec="seconds"),
                "timestamp_local": now_utc.astimezone(
                    self._local_timezone
                ).isoformat(timespec="seconds"),
                "process_elapsed_seconds": round(elapsed, 6),
            }
            payload.update(sanitized_fields)
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                file_locked = False
                try:
                    if os.name == "posix" and fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                        file_locked = True
                    handle.write(serialized)
                    handle.write("\n")
                    handle.flush()
                    if critical or event in CRITICAL_EVENTS:
                        os.fsync(handle.fileno())
                finally:
                    if file_locked:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            human = render_human_event(payload)
            if human is not None and self.stream is not None:
                print(human, file=self.stream, flush=True)


class SafeEventSink:
    def __init__(
        self,
        logger: Any,
        *,
        fallback_stream: TextIO | None = sys.stderr,
    ) -> None:
        self.logger = logger
        self.fallback_stream = fallback_stream
        self.degraded = False
        self._warning_written = False
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> Any:
        try:
            return self.logger.emit(event, **fields)
        except Exception as error:
            with self._lock:
                self.degraded = True
                if not self._warning_written:
                    self._warning_written = True
                    try:
                        warning = sanitize_text(
                            "WARNING observability degraded: "
                            f"{type(error).__name__}: {error}",
                            limit=500,
                        )
                    except Exception:
                        warning = (
                            "WARNING observability degraded: <unprintable>"
                        )
                    try:
                        if self.fallback_stream is not None:
                            print(
                                warning,
                                file=self.fallback_stream,
                                flush=True,
                            )
                    except Exception:
                        pass
            return None


def append_external_event(
    path: Path,
    event: str,
    *,
    attempt_id: str,
    experiment_version: str = "full-v2",
    run_mode: str = "full",
    fields: Mapping[str, Any] | None = None,
) -> None:
    logger = RunEventLogger(
        path,
        ObservabilitySettings(),
        attempt_id=attempt_id,
        experiment_version=experiment_version,
        run_mode=run_mode,
    )
    logger.emit(event, critical=True, **dict(fields or {}))


def _parse_cli_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _new_shell_attempt_id(runner_attempt_id: str) -> str:
    unique_component = uuid.uuid4().hex
    suffix = f"-shell-{os.getpid()}-{unique_component}"
    prefix_limit = _ATTEMPT_ID_LIMIT - len(suffix)
    if prefix_limit < 1:
        raise ValueError("Shell attempt ID suffix exceeds the event ID limit")
    sanitized_prefix = sanitize_text(
        runner_attempt_id, limit=_ATTEMPT_ID_LIMIT
    )
    safe_prefix = re.sub(
        r"[^A-Za-z0-9_.-]+", "-", sanitized_prefix
    ).strip("-")
    return f"{(safe_prefix or 'runner')[:prefix_limit]}{suffix}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append a sanitized run event")
    subparsers = parser.add_subparsers(dest="command", required=True)
    append_parser = subparsers.add_parser(
        "append", description="Append an allowlisted shell event"
    )
    append_parser.add_argument("--path", type=Path, required=True)
    append_parser.add_argument(
        "--event", choices=sorted(SHELL_EVENTS), required=True
    )
    append_parser.add_argument("--attempt-id", required=True)
    append_parser.add_argument("--field", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command != "append":  # pragma: no cover - argparse enforces this
        parser.error(f"Unknown command: {arguments.command}")

    runner_attempt_id = str(arguments.attempt_id).strip()
    if not runner_attempt_id:
        parser.error("--attempt-id must be nonempty")

    allowed_fields = EVENT_FIELDS[arguments.event]
    parsed_fields: dict[str, Any] = {}
    for item in arguments.field:
        if "=" not in item:
            parser.error(f"Invalid --field value (expected key=value): {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in allowed_fields:
            parser.error(
                f"Unknown field for {arguments.event!r}: {key!r}"
            )
        parsed_fields[key] = _parse_cli_value(raw_value)

    parsed_fields["runner_attempt_id"] = runner_attempt_id
    shell_attempt_id = _new_shell_attempt_id(runner_attempt_id)
    append_external_event(
        arguments.path,
        arguments.event,
        attempt_id=shell_attempt_id,
        fields=parsed_fields,
    )
    return 0


__all__ = [
    "AUTH_RE",
    "ApiBudgetStop",
    "AuthBillingStop",
    "BreakerAcknowledgementRequired",
    "CRITICAL_EVENTS",
    "CircuitBreakerStop",
    "ContractDriftStop",
    "DATA_URL_RE",
    "EVENT_FIELDS",
    "EVENT_SCHEMA",
    "FailureCircuitBreaker",
    "FailureClassification",
    "GracefulStopRequested",
    "HEALTH_VALUES",
    "HeartbeatWorker",
    "KEY_RE",
    "ObservabilitySettings",
    "ProgressSnapshot",
    "ProgressTracker",
    "QUERY_RE",
    "RunEventLogger",
    "RunControlError",
    "SENSITIVE_FIELDS",
    "SafeEventSink",
    "StageObservation",
    "SummaryScheduler",
    "StopController",
    "WORKSPACE_RE",
    "append_external_event",
    "classify_failure",
    "format_duration",
    "project_remaining_stage_counts",
    "render_human_event",
    "sanitize_payload",
    "sanitize_text",
]


if __name__ == "__main__":
    raise SystemExit(main())
