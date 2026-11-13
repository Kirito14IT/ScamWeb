"""Atomic per-stage checkpoints for resumable ScamMark experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

from .observability import sanitize_text

if os.name == "posix":
    import fcntl
else:  # pragma: no cover - exercised by Windows test environments
    fcntl = None  # type: ignore[assignment]


class ExperimentContractError(ValueError):
    pass


class CheckpointContractError(ExperimentContractError):
    pass


class ApiCallLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class CallLedgerAudit:
    reserved: int
    finished: int
    completed: int
    transport_error: int
    outstanding: int
    outstanding_by_stage: dict[str, int]
    outstanding_contexts: Sequence[dict[str, str]]
    finished_without_checkpoint: frozenset[tuple[str, str]]
    replay_required_by_stage: dict[str, int]
    corrupt: bool


@dataclass(frozen=True)
class CheckpointAudit:
    complete_stages: frozenset[tuple[str, str]]
    error_stages: frozenset[tuple[str, str]]
    sample_stage_counts: dict[str, int]
    corrupt_files: Sequence[str]


@dataclass(frozen=True)
class SampleProgress:
    complete: int
    partial: int
    error: int
    required_stage_counts: dict[str, int]


def scan_checkpoints(
    checkpoint_dir: Path,
    *,
    config_hash: str | None = None,
) -> CheckpointAudit:
    complete: set[tuple[str, str]] = set()
    errors: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    corrupt: list[str] = []
    if not checkpoint_dir.is_dir():
        return CheckpointAudit(frozenset(), frozenset(), {}, ())
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            sample_id = str(payload["sample_id"])
            if config_hash is not None and payload.get("config_hash") != config_hash:
                raise ValueError("checkpoint config hash mismatch")
            stages = dict(payload.get("stages") or {})
            file_complete: set[tuple[str, str]] = set()
            file_errors: set[tuple[str, str]] = set()
            for stage, value in stages.items():
                if not isinstance(value, Mapping):
                    raise TypeError("checkpoint stage must be a mapping")
                key = (sample_id, str(stage))
                if value.get("status") == "error":
                    file_errors.add(key)
                elif value.get("status") == "complete" and isinstance(
                    value.get("output"), dict
                ):
                    file_complete.add(key)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            corrupt.append(path.name)
            continue
        complete.update(file_complete)
        errors.update(file_errors)
        counts[sample_id] = sum(item[0] == sample_id for item in complete)
    return CheckpointAudit(frozenset(complete), frozenset(errors), counts, tuple(corrupt))


def scan_call_ledger(
    path: Path,
    *,
    config_hash: str | None,
    checkpoint_stages: frozenset[tuple[str, str]],
) -> CallLedgerAudit:
    reservations: list[dict[str, Any]] = []
    reservation_indexes: dict[str, deque[int]] = {}
    seen_reservation_ids: set[str] = set()
    finishes: list[tuple[dict[str, Any], dict[str, str]]] = []
    legacy_api_calls_before_ledger = 0
    first_reservation_seen = False
    corrupt = False

    def add_reservation(
        *,
        call_id: str,
        context: dict[str, str],
        matched: bool = False,
    ) -> int:
        reservations.append(
            {
                "call_id": call_id,
                "context": context,
                "matched": matched,
            }
        )
        return len(reservations) - 1

    def add_unknown_reservation() -> None:
        add_reservation(
            call_id="",
            context={"sample_id": "unknown", "stage": "unknown"},
        )

    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                corrupt = True
                add_unknown_reservation()
                continue
            if not isinstance(row, Mapping):
                corrupt = True
                add_unknown_reservation()
                continue
            if config_hash is not None:
                row_config_hash = str(row.get("config_hash") or "")
                if not row_config_hash:
                    corrupt = True
                    add_unknown_reservation()
                    continue
                if row_config_hash != config_hash:
                    continue
            call_id = str(row.get("call_id") or "")
            if not call_id:
                corrupt = True
                add_unknown_reservation()
                continue
            if row.get("event") == "reserved":
                if not first_reservation_seen:
                    first_reservation_seen = True
                    baseline = row.get("legacy_api_calls_before_ledger", 0)
                    if (
                        isinstance(baseline, int)
                        and not isinstance(baseline, bool)
                        and baseline >= 0
                    ):
                        legacy_api_calls_before_ledger = baseline
                    else:
                        corrupt = True
                raw_context = row.get("context")
                if isinstance(raw_context, Mapping):
                    context = {
                        "sample_id": str(raw_context.get("sample_id") or "unknown"),
                        "stage": str(raw_context.get("stage") or "unknown"),
                    }
                else:
                    context = {"sample_id": "unknown", "stage": "unknown"}
                    corrupt = True
                if "unknown" in context.values():
                    corrupt = True
                if call_id in seen_reservation_ids:
                    corrupt = True
                seen_reservation_ids.add(call_id)
                reservation_index = add_reservation(
                    call_id=call_id,
                    context=context,
                )
                reservation_indexes.setdefault(call_id, deque()).append(reservation_index)
            elif row.get("event") == "finished":
                pending = reservation_indexes.get(call_id)
                if pending:
                    reservation_index = pending.popleft()
                    reservations[reservation_index]["matched"] = True
                    context = dict(reservations[reservation_index]["context"])
                else:
                    corrupt = True
                    context = {"sample_id": "unknown", "stage": "unknown"}
                    add_reservation(
                        call_id=call_id,
                        context=context,
                        matched=True,
                    )
                finish = dict(row)
                if finish.get("status") not in {"completed", "transport_error"}:
                    corrupt = True
                finishes.append((finish, context))
            else:
                corrupt = True
                add_unknown_reservation()

    outstanding_contexts = [
        dict(reservation["context"])
        for reservation in reservations
        if not reservation["matched"]
    ]
    outstanding_by_stage = dict(Counter(item["stage"] for item in outstanding_contexts))
    completed_finishes = [
        (row, context) for row, context in finishes if row.get("status") == "completed"
    ]
    transport_finishes = [
        (row, context)
        for row, context in finishes
        if row.get("status") == "transport_error"
    ]
    finished_without = frozenset(
        (context["sample_id"], context["stage"])
        for _, context in completed_finishes
        if (context["sample_id"], context["stage"]) not in checkpoint_stages
    )
    replay_keys = {
        (context["sample_id"], context["stage"])
        for _, context in finishes
        if (context["sample_id"], context["stage"]) not in checkpoint_stages
    }
    replay_by_stage = dict(Counter(stage for _, stage in replay_keys))
    return CallLedgerAudit(
        reserved=legacy_api_calls_before_ledger + len(reservations),
        finished=len(finishes),
        completed=len(completed_finishes),
        transport_error=len(transport_finishes),
        outstanding=len(outstanding_contexts),
        outstanding_by_stage=outstanding_by_stage,
        outstanding_contexts=tuple(outstanding_contexts),
        finished_without_checkpoint=finished_without,
        replay_required_by_stage=replay_by_stage,
        corrupt=corrupt,
    )


def summarize_sample_progress(
    manifest: Sequence[Mapping[str, Any]],
    *,
    complete_stages: frozenset[tuple[str, str]],
    error_stages: frozenset[tuple[str, str]],
    binary_outputs: Mapping[str, str],
) -> SampleProgress:
    base = frozenset({"url", "text", "visual", "cross", "binary"})
    downstream = frozenset({"type", "grounding"})
    complete = partial = error = 0
    required_counts: dict[str, int] = {}
    for row in manifest:
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        needs_downstream = row.get("binary") == "scam" or binary_outputs.get(sample_id) == "scam"
        required = base | (downstream if needs_downstream else frozenset())
        required_counts[sample_id] = len(required)
        done = {stage for current_id, stage in complete_stages if current_id == sample_id}
        failed = {
            stage
            for current_id, stage in error_stages
            if current_id == sample_id and stage in required
        }
        if required <= done:
            complete += 1
        elif failed:
            error += 1
        elif done:
            partial += 1
    return SampleProgress(complete, partial, error, required_counts)


_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)
_METADATA_INTEGER_FIELDS = (
    "attempt_count",
    "retry_count",
    "api_call_count",
    "cumulative_attempt_count",
    "cumulative_api_call_count",
    "max_output_tokens",
    "http_status",
)
_METADATA_FLOAT_FIELDS = ("elapsed_seconds", "temperature")
_METADATA_BOOLEAN_FIELDS = (
    "response_model_matches_request",
    "enable_thinking",
    "schema_repair_attempted",
)
_METADATA_TEXT_FIELDS = (
    "stage",
    "model",
    "provider",
    "finish_reason",
    "response_model",
    "status",
    "error_type",
    "error_category",
)
_ERROR_MESSAGE_FIELDS = ("message", "error", "schema_repair_error")
_PROMPT_INTEGER_FIELDS = (
    "max_tokens",
    "original_tokens",
    "final_tokens",
    "configured_total_input_tokens",
    "system_prompt_reserve",
)
_IMAGE_INTEGER_FIELDS = ("asset_count", "total_pixels", "total_base64_bytes")
_FINISH_METADATA_FIELDS = frozenset(
    {
        "model",
        "provider",
        "response_model",
        "finish_reason",
        "usage",
        "endpoint_host",
        "endpoint_sha256",
        "request_fingerprint",
        "error_type",
        "error_category",
        "message",
        "error",
        "http_status",
        "elapsed_seconds",
        "attempt_count",
        "retry_count",
        "api_call_count",
    }
)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _safe_digest(value: Any) -> str | None:
    text = str(value or "").strip()
    if re.fullmatch(r"[0-9A-Fa-f]{64}", text):
        return text
    return None


def _safe_endpoint_host(value: Any) -> str | None:
    text = sanitize_text(value, limit=500).strip()
    if not text:
        return None
    parsed = urlsplit(text if "://" in text else f"//{text}")
    host = str(parsed.hostname or "").strip().rstrip(".")
    return sanitize_text(host, limit=255) or None


def _safe_int_list(value: Any, *, limit: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    projected: list[int] = []
    for item in value[:limit]:
        number = _nonnegative_int(item)
        if number is None:
            return None
        projected.append(number)
    return projected


def _project_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, int] = {}
    for name in _USAGE_FIELDS:
        if name in value:
            number = _nonnegative_int(value[name])
            if number is not None:
                projected[name] = number
    return projected


def _project_attempt_errors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    projected: list[dict[str, Any]] = []
    for item in value[:20]:
        if not isinstance(item, Mapping):
            continue
        error: dict[str, Any] = {}
        attempt = _nonnegative_int(item.get("attempt"))
        if attempt is not None:
            error["attempt"] = attempt
        if item.get("error_type") not in (None, ""):
            error["error_type"] = sanitize_text(item["error_type"], limit=200)
        if item.get("message") not in (None, ""):
            error["message"] = "<redacted-field>"
        if error:
            projected.append(error)
    return projected


def _project_prompt_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for name in _PROMPT_INTEGER_FIELDS:
        if name in value:
            number = _nonnegative_int(value[name])
            if number is not None:
                projected[name] = number
    if isinstance(value.get("truncated"), bool):
        projected["truncated"] = value["truncated"]
    if value.get("tokenizer") not in (None, ""):
        projected["tokenizer"] = sanitize_text(value["tokenizer"], limit=100)
    return projected


def _project_image_asset(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for name in ("submitted_index", "data_url_size"):
        if name in value:
            number = _nonnegative_int(value[name])
            if number is not None:
                projected[name] = number
    kind = str(value.get("kind") or "")
    if kind in {"overview", "tile", "contact_sheet"}:
        projected["kind"] = kind
    submitted_size = _safe_int_list(value.get("submitted_size"), limit=2)
    if submitted_size is not None:
        projected["submitted_size"] = submitted_size
    logical_indexes = _safe_int_list(value.get("logical_indexes"), limit=1000)
    if logical_indexes is not None:
        projected["logical_indexes"] = logical_indexes
    return projected


def _project_logical_image(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for name in ("image_index", "data_url_size"):
        if name in value:
            number = _nonnegative_int(value[name])
            if number is not None:
                projected[name] = number
    kind = str(value.get("kind") or "")
    if kind in {"overview", "tile", "contact_sheet"}:
        projected["kind"] = kind
    submission_mode = str(value.get("submission_mode") or "")
    if submission_mode in {"direct", "contact_sheet"}:
        projected["submission_mode"] = submission_mode
    for name, limit in (
        ("crop_xyxy_in_original", 4),
        ("original_size", 2),
        ("submitted_size", 2),
        ("contact_sheet_size", 2),
        ("contact_sheet_panel_xyxy", 4),
    ):
        numbers = _safe_int_list(value.get(name), limit=limit)
        if numbers is not None:
            projected[name] = numbers
    return projected


def _project_image_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for name in _IMAGE_INTEGER_FIELDS:
        if name in value:
            number = _nonnegative_int(value[name])
            if number is not None:
                projected[name] = number
    if isinstance(value.get("assets"), (list, tuple)):
        projected["assets"] = [
            asset
            for item in value["assets"][:1000]
            if (asset := _project_image_asset(item))
        ]
    if isinstance(value.get("logical_inputs"), (list, tuple)):
        projected["logical_inputs"] = [
            image
            for item in value["logical_inputs"][:1000]
            if (image := _project_logical_image(item))
        ]
    return projected


def project_request_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for name in _METADATA_INTEGER_FIELDS:
        if name in value:
            number = _nonnegative_int(value[name])
            if number is not None:
                projected[name] = number
    for name in _METADATA_FLOAT_FIELDS:
        if name in value:
            number = _nonnegative_float(value[name])
            if number is not None:
                projected[name] = number
    for name in _METADATA_BOOLEAN_FIELDS:
        if isinstance(value.get(name), bool):
            projected[name] = value[name]
    for name in _METADATA_TEXT_FIELDS:
        if value.get(name) not in (None, ""):
            projected[name] = sanitize_text(value[name], limit=200)
    for name in _ERROR_MESSAGE_FIELDS:
        if name in value:
            projected[name] = (
                "<redacted-field>" if value[name] not in (None, "") else ""
            )
    endpoint_host = _safe_endpoint_host(value.get("endpoint_host"))
    if endpoint_host is not None:
        projected["endpoint_host"] = endpoint_host
    endpoint_sha256 = _safe_digest(value.get("endpoint_sha256"))
    if endpoint_sha256 is not None:
        projected["endpoint_sha256"] = endpoint_sha256
    request_fingerprint = _safe_digest(value.get("request_fingerprint"))
    if request_fingerprint is not None:
        projected["request_fingerprint"] = request_fingerprint
    if "usage" in value:
        projected["usage"] = _project_usage(value["usage"])
    if "attempt_errors" in value:
        projected["attempt_errors"] = _project_attempt_errors(value["attempt_errors"])
    prompt = _project_prompt_metrics(value.get("prompt"))
    if prompt:
        projected["prompt"] = prompt
    image = _project_image_metrics(value.get("image"))
    if image:
        projected["image"] = image
    return projected


def _project_finish_metadata(value: Any) -> dict[str, Any]:
    projected = project_request_metadata(value)
    return {name: item for name, item in projected.items() if name in _FINISH_METADATA_FIELDS}


def _project_reservation_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for name in ("sample_id", "stage", "model", "provider"):
        if value.get(name) not in (None, ""):
            projected[name] = sanitize_text(value[name], limit=200)
    endpoint_host = _safe_endpoint_host(value.get("endpoint_host"))
    if endpoint_host is not None:
        projected["endpoint_host"] = endpoint_host
    endpoint_sha256 = _safe_digest(value.get("endpoint_sha256"))
    if endpoint_sha256 is not None:
        projected["endpoint_sha256"] = endpoint_sha256
    request_fingerprint = _safe_digest(value.get("request_fingerprint"))
    if request_fingerprint is not None:
        projected["request_fingerprint"] = request_fingerprint
    if isinstance(value.get("schema_repair"), bool):
        projected["schema_repair"] = value["schema_repair"]
    return projected


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExperimentState:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.call_ledger = self.root / "api_call_ledger.jsonl"
        self.config = config
        self.config_hash = stable_config_hash(config)
        self._ledger_lock = threading.Lock()
        config_path = self.root / "run_config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != self.config_hash:
                raise ExperimentContractError(
                    "Experiment configuration changed; refusing to resume incompatible checkpoints"
                )
        else:
            self._atomic_write(
                config_path,
                {
                    "config_hash": self.config_hash,
                    "config": config,
                },
            )

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _sample_path(self, sample_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._")[:80] or "sample"
        digest = hashlib.sha1(str(sample_id).encode("utf-8")).hexdigest()[:10]
        return self.checkpoints / f"{safe}.{digest}.json"

    def load_stages(self, sample_id: str) -> dict[str, Any]:
        path = self._sample_path(sample_id)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config_hash") != self.config_hash:
            raise CheckpointContractError(
                f"Checkpoint configuration mismatch for sample {sample_id}"
            )
        projected: dict[str, Any] = {}
        for stage, value in dict(payload.get("stages") or {}).items():
            if not isinstance(value, Mapping):
                continue
            record = dict(value)
            record["request_metadata"] = project_request_metadata(
                value.get("request_metadata")
            )
            if isinstance(value.get("error"), Mapping):
                error = value["error"]
                record["error"] = {
                    "type": sanitize_text(error.get("type"), limit=200),
                    "message": (
                        "<redacted-field>"
                        if error.get("message") not in (None, "")
                        else ""
                    ),
                }
            projected[str(stage)] = record
        return projected

    def api_call_audit(self) -> CallLedgerAudit:
        checkpoint_audit = scan_checkpoints(
            self.checkpoints,
            config_hash=self.config_hash,
        )
        return scan_call_ledger(
            self.call_ledger,
            config_hash=self.config_hash,
            checkpoint_stages=checkpoint_audit.complete_stages,
        )

    def _checkpoint_api_calls(self) -> int:
        total = 0
        for path in self.checkpoints.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("config_hash") != self.config_hash:
                continue
            for stage in (payload.get("stages") or {}).values():
                metadata = stage.get("request_metadata") or {}
                total += int(
                    metadata.get("cumulative_api_call_count")
                    or metadata.get("api_call_count")
                    or metadata.get("cumulative_attempt_count")
                    or metadata.get("attempt_count")
                    or 0
                )
        return total

    def _ledger_has_reservation(self) -> bool:
        if not self.call_ledger.is_file():
            return False
        for raw in self.call_ledger.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, Mapping)
                and row.get("event") == "reserved"
                and row.get("config_hash") == self.config_hash
            ):
                return True
        return False

    def total_api_calls(self) -> int:
        if self.call_ledger.is_file():
            ledger_api_calls = self.api_call_audit().reserved
            if not self._ledger_has_reservation():
                return self._checkpoint_api_calls() + ledger_api_calls
            return ledger_api_calls
        return self._checkpoint_api_calls()

    @contextmanager
    def _locked_ledger(self) -> Iterator[BinaryIO]:
        self.call_ledger.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_lock:
            with self.call_ledger.open("a+b") as handle:
                file_locked = False
                try:
                    if os.name == "posix" and fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                        file_locked = True
                    yield handle
                finally:
                    if file_locked:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _append_ledger_locked(handle: BinaryIO, payload: dict[str, Any]) -> None:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) not in {b"\n", b"\r"}:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        handle.seek(0, os.SEEK_END)
        handle.write(serialized + b"\n")
        handle.flush()
        os.fsync(handle.fileno())

    def _append_ledger(self, payload: dict[str, Any]) -> None:
        with self._locked_ledger() as handle:
            self._append_ledger_locked(handle, payload)

    def reserve_api_call(self, context: dict[str, Any], *, max_calls: int) -> str:
        """Durably reserve one provider call before transport begins.

        A reservation is counted even if the process dies before receiving a response.
        This conservative accounting prevents a resumed run from exceeding its cap.
        """

        with self._locked_ledger() as handle:
            first_reservation = not self._ledger_has_reservation()
            legacy_api_calls = self._checkpoint_api_calls() if first_reservation else 0
            ledger_api_calls = self.api_call_audit().reserved
            total_api_calls = (
                legacy_api_calls + ledger_api_calls
                if first_reservation
                else ledger_api_calls
            )
            if total_api_calls >= int(max_calls):
                raise ApiCallLimitError(f"Physical API call cap reached ({max_calls})")
            call_id = uuid.uuid4().hex
            reservation = {
                "event": "reserved",
                "call_id": call_id,
                "unix_time": time.time(),
                "config_hash": self.config_hash,
                "context": _project_reservation_context(context),
            }
            if first_reservation:
                reservation["legacy_api_calls_before_ledger"] = legacy_api_calls
            self._append_ledger_locked(handle, reservation)
        return call_id

    def finish_api_call(
        self,
        call_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        safe_status = str(status) if status in {"completed", "transport_error"} else "unknown"
        safe_metadata = _project_finish_metadata(metadata)
        self._append_ledger(
            {
                "event": "finished",
                "call_id": str(call_id),
                "unix_time": time.time(),
                "config_hash": self.config_hash,
                "status": safe_status,
                "metadata": safe_metadata,
            }
        )

    def save_stage(
        self,
        sample_id: str,
        stage: str,
        output: dict[str, Any],
        request_metadata: dict[str, Any],
    ) -> None:
        stages = self.load_stages(sample_id)
        stages[stage] = {
            "status": "complete",
            "output": output,
            "request_metadata": project_request_metadata(request_metadata),
        }
        self._save_stages(sample_id, stages)

    def save_stage_error(
        self,
        sample_id: str,
        stage: str,
        error: Exception,
        request_metadata: dict[str, Any],
    ) -> None:
        stages = self.load_stages(sample_id)
        stages[stage] = {
            "status": "error",
            "output": {},
            "request_metadata": project_request_metadata(request_metadata),
            "error": {
                "type": sanitize_text(type(error).__name__, limit=200),
                "message": "<redacted-field>",
            },
        }
        self._save_stages(sample_id, stages)

    def _save_stages(self, sample_id: str, stages: dict[str, Any]) -> None:
        self._atomic_write(
            self._sample_path(sample_id),
            {
                "sample_id": str(sample_id),
                "config_hash": self.config_hash,
                "stages": stages,
            },
        )
