"""Deterministic retry policy and request-error classification."""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar


T = TypeVar("T")

_PROVIDER_DETAIL_LIMIT = 2000
_TRANSIENT_429_CODES = frozenset(
    {
        "rate_limit_exceeded",
        "rate_limit_error",
        "request_rate_exceeded",
        "requests_per_minute_exceeded",
        "quota_exceeded",
        "tokens_per_minute_exceeded",
        "throttled",
        "throttling",
        "too_many_requests",
    }
)
_AUTH_BILLING_429_CODES = frozenset(
    {
        "account_balance_not_enough",
        "account_deactivated",
        "account_disabled",
        "billing_hard_limit_reached",
        "credit_limit_reached",
        "credit_exhausted",
        "hard_limit_reached",
        "insufficient_balance",
        "insufficient_credit",
        "insufficient_credits",
        "insufficient_funds",
        "insufficient_quota",
        "quota_exhausted",
        "quota_limit_reached",
    }
)
_TIME_WINDOW_429_TEXT_RE = re.compile(
    r"(?ix)(?:"
    r"\brate[ _-]?limit(?:ed|ing|[ _-]?error|[ _-]?exceeded)?\b"
    r"|\btoo\s+many\s+requests\b"
    r"|\b(?:requests?|tokens?)\s+per\s+(?:second|minute|hour)\b"
    r"|\b(?:rpm|tpm)\b"
    r"|\b(?:for|within)\s+(?:this|the|a)\s+(?:second|minute|hour|window)\b"
    r"|\bthrottl(?:ed|ing)\b"
    r")"
)
_GENERIC_RETRY_429_TEXT_RE = re.compile(r"(?i)\bretry\s+(?:after|later)\b")
_AUTH_BILLING_429_TEXT_RE = re.compile(
    r"(?ix)(?:"
    r"\binsufficient\s+(?:account\s+)?(?:quota|balance|funds?|credits?)\b"
    r"|\b(?:quota|balance|credits?|funds?)\s+(?:is\s+)?"
    r"(?:exhausted|depleted|empty|insufficient)\b"
    r"|\b(?:account|user)\s+(?:quota|balance|credits?)\s+(?:is\s+)?"
    r"(?:exhausted|depleted|empty|insufficient|not\s+enough)\b"
    r"|\b(?:account|user)\s+(?:is\s+)?"
    r"(?:exhausted|depleted|out\s+of\s+(?:funds|credits?)|in\s+arrears)\b"
    r"|\baccount\s+(?:is\s+)?"
    r"(?:disabled|suspended|deactivated|blocked|closed|inactive)\b"
    r"|\b(?:credit\s+)?balance\s+(?:is\s+)?"
    r"(?:too\s+low|not\s+enough|insufficient|exhausted)\b"
    r"|\b(?:no|zero)\s+(?:remaining\s+)?(?:quota|balance|credits?)\b"
    r"|\b(?:exceeded|exhausted)\s+your\s+current\s+quota\b"
    r"|\brun\s+out\s+of\s+credits?\b"
    r"|\b(?:quota|credit)\s+limit\s+(?:reached|exceeded)\b"
    r"|\b(?:billing\s+)?hard\s+limit\s+(?:reached|exceeded)\b"
    r"|\barrearage\b"
    r"|余额不足|额度不足|额度已用尽|账户欠费"
    r")"
)
_STRUCTURED_CODE_RE = re.compile(
    r"(?i)[\"']?(?:code|error_code|type)[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9 _-]{1,100})"
)


def _normalized_provider_code(value: object) -> str:
    if type(value) is bytes:
        value = value[:100].decode("utf-8", errors="replace")
    if type(value) is not str:
        return ""
    return re.sub(r"[\s-]+", "_", value.strip().lower())[:100]


def _structured_provider_codes(value: object) -> tuple[str, ...]:
    codes: list[str] = []
    pending = [(value, 0)]
    budget = 64
    while pending and budget > 0:
        current, depth = pending.pop()
        budget -= 1
        if depth > 5:
            continue
        if type(current) is dict:
            for key, item in list(current.items())[:24]:
                if type(key) is str and key.lower() in {"code", "error_code", "type"}:
                    code = _normalized_provider_code(item)
                    if code:
                        codes.append(code)
                elif type(item) in {dict, list, tuple}:
                    pending.append((item, depth + 1))
        elif type(current) in {list, tuple}:
            pending.extend((item, depth + 1) for item in current[:24])
    return tuple(codes[:24])


def provider_429_failure_category(detail: object) -> str | None:
    """Classify bounded provider 429 detail without stringifying opaque values."""

    if type(detail) is bytes:
        text = detail[:_PROVIDER_DETAIL_LIMIT].decode("utf-8", errors="replace")
    elif type(detail) is str:
        text = detail[:_PROVIDER_DETAIL_LIMIT]
    else:
        text = ""
    codes: tuple[str, ...] = ()
    if text:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if type(parsed) in {dict, list, tuple}:
            codes = _structured_provider_codes(parsed)
        if not codes:
            codes = tuple(
                code
                for match in _STRUCTURED_CODE_RE.finditer(text)
                if (code := _normalized_provider_code(match.group(1)))
            )[:24]
    if any(code in _TRANSIENT_429_CODES for code in codes):
        return "transient"
    if any(code in _AUTH_BILLING_429_CODES for code in codes):
        return "auth_billing"
    normalized = re.sub(r"[_-]+", " ", text.lower())
    if _TIME_WINDOW_429_TEXT_RE.search(normalized):
        return "transient"
    if _AUTH_BILLING_429_TEXT_RE.search(normalized):
        return "auth_billing"
    if _GENERIC_RETRY_429_TEXT_RE.search(normalized):
        return "transient"
    return None


class TransientRequestError(RuntimeError):
    """A network or provider failure that may succeed when retried."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class PermanentRequestError(RuntimeError):
    """A request failure that must not be retried without changing input/config."""


_SAFE_FAILURE_MESSAGES = {
    "auth_billing": "authentication or billing failure",
    "api_budget": "physical API call budget exhausted",
    "contract_drift": "experiment contract drift",
    "transient": "transient provider or transport failure",
    "sample_permanent": "sample-specific permanent failure",
    "unexpected": "unexpected failure",
}


def _safe_request_error_details(error: BaseException) -> dict[str, str]:
    """Project an exception to generic fields that cannot contain provider data."""

    fallback_category = (
        "transient" if isinstance(error, TransientRequestError) else "unexpected"
    )
    category = fallback_category
    http_status = None
    try:
        from .observability import classify_failure

        classification = classify_failure(error)
        if classification.category in _SAFE_FAILURE_MESSAGES:
            category = classification.category
        if (
            isinstance(classification.http_status, int)
            and not isinstance(classification.http_status, bool)
            and 100 <= classification.http_status <= 599
        ):
            http_status = classification.http_status
    except BaseException:
        pass
    if isinstance(error, TransientRequestError):
        error_type = "TransientRequestError"
    elif isinstance(error, PermanentRequestError):
        error_type = "PermanentRequestError"
    else:
        error_type = "Exception"
    message = _SAFE_FAILURE_MESSAGES[category]
    if http_status is not None:
        message = f"HTTP {http_status}: {message}"
    return {
        "error_category": category,
        "error_type": error_type,
        "message": message,
    }


@dataclass(frozen=True)
class RequestPolicy:
    max_attempts: int = 4
    connect_timeout_seconds: float = 20.0
    llm_read_timeout_seconds: float = 120.0
    vlm_read_timeout_seconds: float = 300.0
    backoff_seconds: tuple[float, ...] = (2.0, 4.0, 8.0)
    full_jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")


def is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504}


def execute_with_retry(
    operation: Callable[[], T],
    policy: RequestPolicy,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[float, float], float] = random.uniform,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[T, dict[str, Any]]:
    """Execute one provider operation and return reproducible attempt metadata."""

    start = time.perf_counter()
    errors: list[dict[str, str | int]] = []
    for attempt in range(1, policy.max_attempts + 1):
        try:
            result = operation()
            return result, {
                "attempt_count": attempt,
                "retry_count": attempt - 1,
                "elapsed_seconds": time.perf_counter() - start,
                "attempt_errors": errors,
            }
        except PermanentRequestError as exc:
            safe_error = _safe_request_error_details(exc)
            errors.append(
                {
                    "attempt": attempt,
                    "error_type": safe_error["error_type"],
                    "message": safe_error["message"],
                }
            )
            setattr(
                exc,
                "request_metadata",
                {
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "elapsed_seconds": time.perf_counter() - start,
                    "attempt_errors": errors,
                },
            )
            raise
        except TransientRequestError as exc:
            safe_error = _safe_request_error_details(exc)
            errors.append(
                {
                    "attempt": attempt,
                    "error_type": safe_error["error_type"],
                    "message": safe_error["message"],
                }
            )
            if attempt >= policy.max_attempts:
                setattr(
                    exc,
                    "request_metadata",
                    {
                        "attempt_count": attempt,
                        "retry_count": attempt - 1,
                        "elapsed_seconds": time.perf_counter() - start,
                        "attempt_errors": errors,
                    },
                )
                raise
            delay_index = min(attempt - 1, max(0, len(policy.backoff_seconds) - 1))
            delay = policy.backoff_seconds[delay_index] if policy.backoff_seconds else 0.0
            if policy.full_jitter and delay > 0:
                delay = random_fn(0.0, delay)
            if exc.retry_after_seconds is not None:
                delay = max(delay, max(0.0, float(exc.retry_after_seconds)))
            if event_callback is not None:
                try:
                    event_callback(
                        "retry_scheduled",
                        {
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "delay_seconds": delay,
                            "error_type": safe_error["error_type"],
                            "message": safe_error["message"],
                        },
                    )
                except BaseException:
                    pass
            sleep_fn(delay)

    raise AssertionError("unreachable retry state")
