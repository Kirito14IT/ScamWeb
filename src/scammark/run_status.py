"""Pure read-only status inspection for the ScamMark full-v2 run."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .experiment_state import (
    CallLedgerAudit,
    CheckpointAudit,
    scan_call_ledger,
    scan_checkpoints,
    summarize_sample_progress,
)
from .observability import (
    EVENT_FIELDS,
    EVENT_SCHEMA,
    FailureCircuitBreaker,
    ObservabilitySettings,
    ProgressTracker,
    _sanitize_control_text,
    format_duration,
    project_remaining_stage_counts,
)
from .result_contract import ResultContractError, load_delivery_states


_V1_OUTPUT_NAMESPACE = "scammark_qwen3_dashscope_server_full"
_V2_OUTPUT_NAMESPACE = "scammark_qwen3_dashscope_server_full_v2"
_TMUX_INTEGER_RE = re.compile(r"[0-9]{1,20}")
_EVENT_TAIL_BYTES = 8 * 1024 * 1024
_EVENT_ROW_BYTES = 256 * 1024
_EVENT_MAX_ROWS = 10_000
_EVENT_MAX_DEPTH = 8
_EVENT_MAX_NODES = 2_000
_EVENT_MAX_DICT_ITEMS = 128
_EVENT_MAX_LIST_ITEMS = 256
_EVENT_MAX_STRING_CHARS = 16_384
_MANIFEST_FILE_BYTES = 32 * 1024 * 1024
_MANIFEST_ROW_BYTES = 256 * 1024
_MANIFEST_MAX_ROWS = 10_000
_PREDICTIONS_FILE_BYTES = 128 * 1024 * 1024
_PREDICTION_ROW_BYTES = 2 * 1024 * 1024
_PREDICTION_MAX_ROWS = 10_000
_CHECKPOINT_FILE_BYTES = 4 * 1024 * 1024
_CHECKPOINT_MAX_FILES = 10_000
_RUN_CONFIG_FILE_BYTES = 4 * 1024 * 1024
_LEDGER_FILE_BYTES = 128 * 1024 * 1024
_ARTIFACT_FILE_BYTES = 16 * 1024 * 1024
_STATUS_INTEGER_LIMIT = 10**12
_SAMPLE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CHECKPOINT_STAGES = frozenset(
    {"url", "text", "visual", "cross", "binary", "type", "grounding"}
)
_DELIVERY_STAGES = frozenset({"package", "upload", "publish"})


@dataclass(frozen=True)
class TmuxState:
    state: Literal["absent", "running", "dead"]
    pid: int | None = None
    exit_code: int | None = None


@dataclass(frozen=True)
class PredictionAudit:
    row_count: int
    sample_ids: frozenset[str]
    corrupt: bool


def _tmux_integer(value: object, *, positive: bool = False) -> int | None:
    if type(value) is not str or _TMUX_INTEGER_RE.fullmatch(value) is None:
        return None
    number = int(value)
    if positive and number < 1:
        return None
    return number


def query_tmux(
    session: str,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> TmuxState:
    if type(session) is not str or not session or len(session) > 200:
        return TmuxState("absent")
    command = [
        "tmux",
        "display-message",
        "-p",
        "-t",
        f"={session}:0.0",
        "#{pane_dead}|#{pane_pid}|#{pane_dead_status}",
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if type(result.returncode) is not int or result.returncode != 0:
            return TmuxState("absent")
        if type(result.stdout) is not str:
            return TmuxState("absent")
        fields = result.stdout.strip().split("|")
    except Exception:
        return TmuxState("absent")
    if len(fields) != 3 or fields[0] not in {"0", "1"}:
        return TmuxState("absent")
    pid = _tmux_integer(fields[1], positive=True)
    if pid is None:
        return TmuxState("absent")
    if fields[0] == "0":
        if fields[2] != "":
            return TmuxState("absent")
        return TmuxState("running", pid=pid)
    exit_code = _tmux_integer(fields[2]) if fields[2] else None
    if fields[2] and exit_code is None:
        return TmuxState("absent")
    return TmuxState("dead", pid=pid, exit_code=exit_code)


def _configured_output_root(config: dict[str, Any]) -> Path:
    paths = config.get("paths")
    if type(paths) is not dict:
        raise ValueError("Status config requires a [paths] table")
    raw_output_root = paths.get("output_root")
    if type(raw_output_root) is not str or not raw_output_root.strip():
        raise ValueError("Status config requires paths.output_root")
    output_root = Path(raw_output_root)
    resolved = output_root.resolve(strict=False)
    if (
        output_root.name == _V1_OUTPUT_NAMESPACE
        or resolved.name == _V1_OUTPUT_NAMESPACE
        or output_root.name != _V2_OUTPUT_NAMESPACE
        or resolved.name != _V2_OUTPUT_NAMESPACE
    ):
        raise ValueError("Status inspection requires the full-v2 output namespace")
    return output_root


def _validate_output_candidate(output_root: Path, candidate: Path) -> None:
    try:
        resolved_root = output_root.resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "Status v2 output path escapes the configured namespace"
        ) from error
    if _V1_OUTPUT_NAMESPACE in resolved_candidate.parts:
        raise ValueError("Status inspection cannot read the v1 runtime namespace")


def _validate_output_reads(output_root: Path) -> None:
    relative_paths = (
        ("full",),
        ("state_full",),
        ("pilot_10",),
        ("pilot_60",),
        ("full", "run_events.jsonl"),
        ("full", "manifest.jsonl"),
        ("full", "predictions.jsonl"),
        ("full", "metrics.json"),
        ("full", "metrics_final.json"),
        ("full", "preflight.json"),
        ("full", "delivery_state.json"),
        ("state_full", "run_config.json"),
        ("state_full", "checkpoints"),
        ("state_full", "api_call_ledger.jsonl"),
        ("pilot_10", "run_events.jsonl"),
        ("pilot_10", "metrics.json"),
        ("pilot_60", "run_events.jsonl"),
        ("pilot_60", "metrics.json"),
    )
    for parts in relative_paths:
        _validate_output_candidate(output_root, output_root.joinpath(*parts))
    checkpoint_dir = output_root / "state_full" / "checkpoints"
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("*.json"):
            _validate_output_candidate(output_root, path)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _bounded_event_object(payload: dict[str, Any]) -> bool:
    remaining = _EVENT_MAX_NODES
    pending: list[tuple[object, int]] = [(payload, 0)]
    while pending:
        value, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > _EVENT_MAX_DEPTH:
            return False
        if type(value) is dict:
            if len(value) > _EVENT_MAX_DICT_ITEMS:
                return False
            for key, item in value.items():
                if type(key) is not str or len(key) > 200:
                    return False
                pending.append((item, depth + 1))
        elif type(value) is list:
            if len(value) > _EVENT_MAX_LIST_ITEMS:
                return False
            pending.extend((item, depth + 1) for item in value)
        elif type(value) is str:
            if len(value) > _EVENT_MAX_STRING_CHARS:
                return False
        elif type(value) is int:
            if value.bit_length() > 256:
                return False
        elif type(value) is float:
            if not math.isfinite(value):
                return False
        elif value is not None and type(value) is not bool:
            return False
    return True


def read_jsonl_tail(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    event_path = Path(path)
    if not event_path.is_file():
        return [], []

    warnings: list[str] = []
    try:
        size = event_path.stat().st_size
        with event_path.open("rb") as handle:
            if size > _EVENT_TAIL_BYTES:
                handle.seek(max(0, size - _EVENT_TAIL_BYTES))
                chunk = handle.read(_EVENT_TAIL_BYTES)
                boundary = chunk.find(b"\n")
                chunk = chunk[boundary + 1 :] if boundary >= 0 else b""
                warnings.append("event history truncated to bounded tail")
            else:
                chunk = handle.read(_EVENT_TAIL_BYTES + 1)
                if len(chunk) > _EVENT_TAIL_BYTES:
                    boundary = chunk.find(b"\n", 1)
                    chunk = chunk[boundary + 1 :] if boundary >= 0 else b""
                    warnings.append("event history truncated to bounded tail")
    except (OSError, OverflowError, ValueError):
        return [], ["event file could not be read"]

    rows = chunk.splitlines()
    if len(rows) > _EVENT_MAX_ROWS:
        rows = rows[-_EVENT_MAX_ROWS:]
        warnings.append("event row count truncated to bounded tail")
    nonblank = [index for index, raw in enumerate(rows) if raw.strip()]
    last_nonblank = nonblank[-1] if nonblank else None
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, start=1):
        if not raw.strip():
            continue
        if len(raw) > _EVENT_ROW_BYTES:
            warnings.append(f"ignored oversized event row {index}")
            continue
        try:
            payload = json.loads(raw, parse_constant=_reject_json_constant)
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
            RecursionError,
            MemoryError,
        ):
            if last_nonblank == index - 1:
                warnings.append("ignored truncated final event row")
            else:
                warnings.append(f"malformed event row {index}")
            continue
        if type(payload) is not dict:
            warnings.append(f"ignored non-object event row {index}")
            continue
        if not _bounded_event_object(payload):
            warnings.append(f"ignored unsafe event row {index}")
            continue
        events.append(payload)
    return events, warnings


def _read_bounded_json_object(path: Path, *, byte_limit: int) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > byte_limit:
            return None
        with path.open("rb") as handle:
            raw = handle.read(byte_limit + 1)
        if len(raw) > byte_limit:
            return None
        payload = json.loads(raw, parse_constant=_reject_json_constant)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ):
        return None
    return payload if type(payload) is dict else None


def _read_manifest(path: Path) -> tuple[list[dict[str, Any]], bool]:
    try:
        if not path.is_file():
            return [], False
        if path.stat().st_size > _MANIFEST_FILE_BYTES:
            return [], True
        with path.open("rb") as handle:
            raw = handle.read(_MANIFEST_FILE_BYTES + 1)
        if len(raw) > _MANIFEST_FILE_BYTES:
            return [], True
    except OSError:
        return [], True

    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        if len(records) >= _MANIFEST_MAX_ROWS or len(line) > _MANIFEST_ROW_BYTES:
            return [], True
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            RecursionError,
            MemoryError,
        ):
            return [], True
        if type(payload) is not dict or not _bounded_event_object(payload):
            return [], True
        sample_id = _safe_sample_id(
            payload.get("sample_id") or payload.get("id")
        )
        if (
            sample_id is None
            or sample_id in sample_ids
            or payload.get("binary") not in {"benign", "scam"}
        ):
            return [], True
        sample_ids.add(sample_id)
        projected = dict(payload)
        projected["sample_id"] = sample_id
        records.append(projected)
    return records, False


def _audit_prediction_rows(path: Path) -> PredictionAudit:
    try:
        if not path.is_file():
            return PredictionAudit(0, frozenset(), False)
        if path.stat().st_size > _PREDICTIONS_FILE_BYTES:
            return PredictionAudit(0, frozenset(), True)
        count = 0
        consumed = 0
        sample_ids: set[str] = set()
        corrupt = False
        with path.open("rb") as handle:
            for line in handle:
                consumed += len(line)
                if consumed > _PREDICTIONS_FILE_BYTES:
                    return PredictionAudit(0, frozenset(), True)
                if not line.strip():
                    continue
                if count >= _PREDICTION_MAX_ROWS or len(line) > _PREDICTION_ROW_BYTES:
                    return PredictionAudit(0, frozenset(), True)
                payload = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
                if type(payload) is not dict or not _bounded_event_object(payload):
                    return PredictionAudit(0, frozenset(), True)
                sample_id = payload.get("sample_id")
                if _safe_sample_id(sample_id) is None:
                    return PredictionAudit(0, frozenset(), True)
                count += 1
                if sample_id in sample_ids:
                    corrupt = True
                sample_ids.add(sample_id)
        return PredictionAudit(count, frozenset(sample_ids), corrupt)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ):
        return PredictionAudit(0, frozenset(), True)


def _path_uses_v1_namespace(path: Path) -> bool:
    if _V1_OUTPUT_NAMESPACE in path.parts:
        return True
    try:
        return _V1_OUTPUT_NAMESPACE in path.resolve(strict=False).parts
    except (OSError, RuntimeError):
        return True


def _manifest_path(config: dict[str, Any], output_root: Path) -> Path | None:
    local = output_root / "full" / "manifest.jsonl"
    if local.is_file():
        return local
    data = config.get("data")
    raw_manifest = data.get("manifest") if type(data) is dict else None
    if type(raw_manifest) is not str or not raw_manifest.strip():
        return None
    configured = Path(raw_manifest)
    if _path_uses_v1_namespace(configured):
        raise ValueError("Status inspection cannot read the v1 runtime namespace")
    return configured


def _read_config_hash(state_root: Path) -> tuple[str | None, bool]:
    path = state_root / "run_config.json"
    if not path.is_file():
        return None, False
    payload = _read_bounded_json_object(
        path,
        byte_limit=_RUN_CONFIG_FILE_BYTES,
    )
    if payload is None:
        return None, True
    config_hash = payload.get("config_hash")
    if type(config_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", config_hash) is None:
        return None, True
    return config_hash, False


def _durable_state_present(checkpoint_dir: Path, ledger_path: Path) -> bool:
    try:
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            return True
        if ledger_path.is_file() and ledger_path.stat().st_size > 0:
            return True
    except OSError:
        return True
    return False


def _scan_checkpoint_status(
    checkpoint_dir: Path,
    *,
    config_hash: str | None,
) -> CheckpointAudit:
    try:
        return scan_checkpoints(checkpoint_dir, config_hash=config_hash)
    except Exception:
        corrupt = (
            tuple(sorted(path.name for path in checkpoint_dir.glob("*.json")))
            if checkpoint_dir.is_dir()
            else ("checkpoint scan failed",)
        )
        return CheckpointAudit(frozenset(), frozenset(), {}, corrupt)


def _checkpoint_payload_shape_is_valid(payload: dict[str, Any]) -> bool:
    sample_id = payload.get("sample_id")
    config_hash = payload.get("config_hash")
    stages = payload.get("stages")
    if (
        _safe_sample_id(sample_id) is None
        or type(config_hash) is not str
        or re.fullmatch(r"[0-9a-f]{64}", config_hash) is None
        or type(stages) is not dict
        or len(stages) > len(_CHECKPOINT_STAGES)
        or not _bounded_event_object(payload)
    ):
        return False
    for stage, record in stages.items():
        if (
            stage not in _CHECKPOINT_STAGES
            or type(record) is not dict
            or record.get("status") not in {"complete", "error"}
        ):
            return False
        if record.get("status") == "complete" and type(record.get("output")) is not dict:
            return False
        if "output" in record and type(record.get("output")) is not dict:
            return False
        if stage == "binary" and record.get("status") == "complete":
            if record["output"].get("binary") not in {"benign", "scam"}:
                return False
    return True


def _preflight_checkpoint_payloads(
    checkpoint_dir: Path,
) -> tuple[dict[Path, dict[str, Any]], tuple[str, ...]]:
    if not checkpoint_dir.is_dir():
        return {}, ()
    try:
        paths: list[Path] = []
        for path in sorted(checkpoint_dir.glob("*.json")):
            paths.append(path)
            if len(paths) > _CHECKPOINT_MAX_FILES:
                names = [candidate.name for candidate in paths]
                names.append("checkpoint file count exceeds limit")
                return {}, tuple(sorted(set(names)))
    except (OSError, RuntimeError):
        return {}, ("checkpoint preflight failed",)

    payloads: dict[Path, dict[str, Any]] = {}
    sample_paths: dict[str, Path] = {}
    corrupt: set[str] = set()
    for path in paths:
        try:
            if path.stat().st_size > _CHECKPOINT_FILE_BYTES:
                corrupt.add(path.name)
                continue
        except OSError:
            corrupt.add(path.name)
            continue
        payload = _read_bounded_json_object(
            path,
            byte_limit=_CHECKPOINT_FILE_BYTES,
        )
        if payload is None or not _checkpoint_payload_shape_is_valid(payload):
            corrupt.add(path.name)
            continue
        sample_id = payload["sample_id"]
        previous = sample_paths.get(sample_id)
        if previous is not None:
            corrupt.update({previous.name, path.name})
        else:
            sample_paths[sample_id] = path
        payloads[path] = payload
    if corrupt:
        return {}, tuple(sorted(corrupt))
    return payloads, ()


def _checkpoint_binary_outputs(
    payloads: Mapping[Path, dict[str, Any]],
    *,
    config_hash: str | None,
    complete_stages: frozenset[tuple[str, str]],
) -> dict[str, str]:
    binary_outputs: dict[str, str] = {}
    valid_binary = {
        sample_id
        for sample_id, stage in complete_stages
        if stage == "binary"
    }
    if not valid_binary:
        return binary_outputs
    for payload in payloads.values():
        sample_id = payload.get("sample_id")
        if type(sample_id) is not str or sample_id not in valid_binary:
            continue
        if config_hash is not None and payload.get("config_hash") != config_hash:
            continue
        stages = payload.get("stages")
        binary = stages.get("binary") if type(stages) is dict else None
        output = binary.get("output") if type(binary) is dict else None
        label = output.get("binary") if type(output) is dict else None
        if (
            type(binary) is dict
            and binary.get("status") == "complete"
            and label in {"benign", "scam"}
        ):
            binary_outputs[sample_id] = label
    return binary_outputs


def _conservative_ledger_audit() -> CallLedgerAudit:
    unknown = {"sample_id": "unknown", "stage": "unknown"}
    return CallLedgerAudit(
        reserved=1,
        finished=0,
        completed=0,
        transport_error=0,
        outstanding=1,
        outstanding_by_stage={"unknown": 1},
        outstanding_contexts=(unknown,),
        finished_without_checkpoint=frozenset(),
        replay_required_by_stage={},
        corrupt=True,
    )


def _project_ledger_stage_counts(
    counts: Mapping[object, object],
) -> tuple[dict[str, int], bool]:
    projected: dict[str, int] = {}
    corrupt = False
    for raw_stage, raw_count in counts.items():
        count = _bounded_nonnegative_integer(raw_count)
        if count is None:
            count = 1
            corrupt = True
        stage = (
            raw_stage
            if type(raw_stage) is str
            and raw_stage in _CHECKPOINT_STAGES | {"unknown"}
            else "unknown"
        )
        if stage == "unknown" and raw_stage != "unknown":
            corrupt = True
        combined = projected.get(stage, 0) + count
        if combined > _STATUS_INTEGER_LIMIT:
            combined = _STATUS_INTEGER_LIMIT
            corrupt = True
        projected[stage] = combined
    return projected, corrupt


def _project_ledger_scalar(
    value: object,
    *,
    saturate_unknown: bool,
) -> tuple[int, bool]:
    if type(value) is int:
        if value < 0:
            return 0, True
        if value > _STATUS_INTEGER_LIMIT:
            return _STATUS_INTEGER_LIMIT, True
        return value, False
    return (_STATUS_INTEGER_LIMIT if saturate_unknown else 0), True


def _project_ledger_audit(audit: CallLedgerAudit) -> CallLedgerAudit:
    outstanding_by_stage, outstanding_stage_corrupt = _project_ledger_stage_counts(
        audit.outstanding_by_stage
    )
    replay_by_stage, replay_corrupt = _project_ledger_stage_counts(
        audit.replay_required_by_stage
    )
    reserved, reserved_corrupt = _project_ledger_scalar(
        audit.reserved,
        saturate_unknown=True,
    )
    finished, finished_corrupt = _project_ledger_scalar(
        audit.finished,
        saturate_unknown=False,
    )
    completed, completed_corrupt = _project_ledger_scalar(
        audit.completed,
        saturate_unknown=False,
    )
    transport_error, transport_corrupt = _project_ledger_scalar(
        audit.transport_error,
        saturate_unknown=False,
    )
    outstanding, outstanding_scalar_corrupt = _project_ledger_scalar(
        audit.outstanding,
        saturate_unknown=True,
    )
    scalar_corrupt = any(
        (
            reserved_corrupt,
            finished_corrupt,
            completed_corrupt,
            transport_corrupt,
            outstanding_scalar_corrupt,
        )
    )

    if completed + transport_error > _STATUS_INTEGER_LIMIT:
        transport_error = _STATUS_INTEGER_LIMIT - completed
        scalar_corrupt = True
    classified_finished = completed + transport_error
    if finished < classified_finished:
        finished = classified_finished
        scalar_corrupt = True
    if finished + outstanding > _STATUS_INTEGER_LIMIT:
        outstanding = _STATUS_INTEGER_LIMIT - finished
        scalar_corrupt = True
    minimum_reserved = finished + outstanding
    if reserved < minimum_reserved:
        reserved = minimum_reserved
        scalar_corrupt = True
    return CallLedgerAudit(
        reserved=reserved,
        finished=finished,
        completed=completed,
        transport_error=transport_error,
        outstanding=outstanding,
        outstanding_by_stage=outstanding_by_stage,
        outstanding_contexts=(),
        finished_without_checkpoint=frozenset(),
        replay_required_by_stage=replay_by_stage,
        corrupt=(
            audit.corrupt
            or scalar_corrupt
            or outstanding_stage_corrupt
            or replay_corrupt
        ),
    )


def _scan_ledger_status(
    path: Path,
    *,
    config_hash: str | None,
    checkpoint_stages: frozenset[tuple[str, str]],
) -> CallLedgerAudit:
    try:
        if path.is_file() and path.stat().st_size > _LEDGER_FILE_BYTES:
            return _conservative_ledger_audit()
        return _project_ledger_audit(
            scan_call_ledger(
                path,
                config_hash=config_hash,
                checkpoint_stages=checkpoint_stages,
            )
        )
    except Exception:
        return _conservative_ledger_audit()


def _bounded_nonnegative_number(value: object) -> int | float | None:
    if type(value) is int:
        return value if 0 <= value <= _STATUS_INTEGER_LIMIT else None
    if type(value) is float:
        if math.isfinite(value) and 0 <= value <= _STATUS_INTEGER_LIMIT:
            return value
    return None


def _bounded_nonnegative_integer(value: object) -> int | None:
    if type(value) is int and 0 <= value <= _STATUS_INTEGER_LIMIT:
        return value
    return None


def _safe_identifier(value: object, *, limit: int = 200) -> str | None:
    if type(value) is not str or not value or len(value) > limit:
        return None
    try:
        safe = _sanitize_control_text(value)
    except Exception:
        return None
    return safe if safe and len(safe) <= limit else None


def _safe_sample_id(value: object) -> str | None:
    if type(value) is not str or _SAMPLE_ID_RE.fullmatch(value) is None:
        return None
    try:
        sanitized = _sanitize_control_text(value)
    except Exception:
        return None
    return value if sanitized == value else None


def _safe_stage(value: object) -> str | None:
    if type(value) is str and value in _CHECKPOINT_STAGES:
        return value
    return None


def _safe_delivery_stage(value: object) -> str | None:
    if type(value) is str and value in _DELIVERY_STAGES:
        return value
    return None


def _safe_delivery_mode(value: object) -> str | None:
    if type(value) is str and value in {"inference", "final"}:
        return value
    return None


def _safe_timestamp(value: object) -> tuple[str, datetime] | None:
    if type(value) is not str or not value or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        normalized = parsed.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError):
        return None
    if not 2000 <= normalized.year <= 2100:
        return None
    return normalized.isoformat(), normalized


def _safe_local_timestamp(
    value: object,
    *,
    expected_utc: datetime,
) -> str | None:
    if (
        type(value) is not str
        or not value
        or len(value) > 64
        or not value.endswith("+08:00")
    ):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if (
            parsed.tzinfo is None
            or parsed.utcoffset() != timedelta(hours=8)
            or parsed.astimezone(timezone.utc) != expected_utc
        ):
            return None
    except (OverflowError, TypeError, ValueError):
        return None
    if not 2000 <= parsed.year <= 2100:
        return None
    return parsed.isoformat()


def _safe_usage(value: object) -> dict[str, int]:
    if type(value) is not dict:
        return {}
    usage: dict[str, int] = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        number = _bounded_nonnegative_integer(value.get(name))
        if number is not None:
            usage[name] = number
    return usage


def _safe_remaining_cost(value: object) -> dict[str, float]:
    if type(value) is not dict:
        return {}
    costs: dict[str, float] = {}
    for currency in ("USD", "CNY"):
        number = _bounded_nonnegative_number(value.get(currency))
        if number is not None:
            costs[currency] = float(number)
    return costs


_EVENT_TEXT_FIELDS = frozenset(
    {
        "sample_id",
        "stage",
        "model",
        "provider",
        "result",
        "health",
        "category",
        "error_category",
        "error_type",
        "status",
        "reason",
        "boundary",
        "previous_stop_reason",
        "last_safe_sample_id",
        "last_safe_stage",
        "runner_attempt_id",
    }
)
_EVENT_SAMPLE_ID_FIELDS = frozenset({"sample_id", "last_safe_sample_id"})
_EVENT_STAGE_FIELDS = frozenset({"stage", "last_safe_stage"})
_EVENT_INTEGER_FIELDS = frozenset(
    {
        "sample_index",
        "sample_total",
        "completed_samples",
        "api_calls",
        "api_call_limit",
        "retry_count",
        "breaker_consecutive",
        "consecutive",
        "prior_consecutive",
        "http_status",
        "attempt",
        "next_attempt",
        "pid",
        "exit_code",
        "signum",
        "record_count",
        "sample_count",
        "complete_samples",
        "partial_samples",
        "error_samples",
        "complete_stages",
        "error_stages",
        "reserved",
        "finished",
        "outstanding",
        "remaining_call_budget",
    }
)
_EVENT_NUMBER_FIELDS = frozenset(
    {
        "elapsed_seconds",
        "eta_seconds",
        "preliminary_eta_seconds",
        "rate_per_minute",
        "delay_seconds",
        "wait_seconds",
    }
)
_EVENT_BOOLEAN_FIELDS = frozenset(
    {
        "backend_error",
        "forced",
        "gates_passed",
        "resume",
        "ack_breaker",
        "ledger_corrupt",
    }
)


def _valid_summary_row(row: dict[str, Any]) -> bool:
    present = False
    for field in ("completed_samples", "sample_total"):
        if field not in row:
            continue
        present = True
        if _bounded_nonnegative_integer(row.get(field)) is None:
            return False
    for field in ("eta_seconds", "preliminary_eta_seconds"):
        if field not in row:
            continue
        present = True
        value = row.get(field)
        if value is not None and _bounded_nonnegative_number(value) is None:
            return False
    if "rate_per_minute" in row:
        present = True
        if _bounded_nonnegative_number(row.get("rate_per_minute")) is None:
            return False
    if "projected_completion" in row:
        present = True
        if _safe_timestamp(row.get("projected_completion")) is None:
            return False
    if "remaining_cost_est" in row:
        present = True
        costs = row.get("remaining_cost_est")
        if type(costs) is not dict:
            return False
        for currency in ("USD", "CNY"):
            if (
                currency in costs
                and _bounded_nonnegative_number(costs.get(currency)) is None
            ):
                return False
    return present


def _project_event(
    row: dict[str, Any],
    *,
    expected_run_mode: Literal["full", "pilot"],
) -> dict[str, Any] | None:
    if (
        row.get("schema") != EVENT_SCHEMA
        or row.get("experiment_version") != "full-v2"
        or row.get("run_mode") != expected_run_mode
    ):
        return None
    name = row.get("event")
    if type(name) is not str or name not in EVENT_FIELDS:
        return None
    if (
        name == "delivery_stage"
        and "delivery_mode" in row
        and _safe_delivery_mode(row.get("delivery_mode")) is None
    ):
        return None
    if name == "summary" and not _valid_summary_row(row):
        return None
    attempt_id = _safe_identifier(row.get("attempt_id"))
    sequence = _bounded_nonnegative_integer(row.get("sequence"))
    timestamp = _safe_timestamp(row.get("timestamp_utc"))
    timestamp_local = (
        _safe_local_timestamp(
            row.get("timestamp_local"),
            expected_utc=timestamp[1],
        )
        if timestamp is not None
        else None
    )
    process_elapsed = _bounded_nonnegative_number(
        row.get("process_elapsed_seconds")
    )
    if (
        attempt_id is None
        or attempt_id != row.get("attempt_id")
        or sequence is None
        or sequence <= 0
        or timestamp is None
        or timestamp_local is None
        or process_elapsed is None
    ):
        return None
    projected: dict[str, Any] = {
        "event": name,
        "attempt_id": attempt_id,
        "sequence": sequence,
        "timestamp_utc": timestamp[0],
        "timestamp_local": timestamp_local,
        "process_elapsed_seconds": process_elapsed,
    }

    for field in EVENT_FIELDS[name]:
        value = row.get(field)
        if field in _EVENT_SAMPLE_ID_FIELDS:
            safe = _safe_sample_id(value)
            if safe is not None:
                projected[field] = safe
        elif field in _EVENT_STAGE_FIELDS:
            safe = (
                _safe_delivery_stage(value)
                if name == "delivery_stage" and field == "stage"
                else _safe_stage(value)
            )
            if safe is not None:
                projected[field] = safe
        elif field == "delivery_mode":
            safe = _safe_delivery_mode(value)
            if safe is not None:
                projected[field] = safe
        elif field in _EVENT_TEXT_FIELDS:
            safe = _safe_identifier(value, limit=300)
            if safe is not None:
                projected[field] = safe
        elif field == "message":
            if type(value) is str:
                try:
                    message = _sanitize_control_text(value)
                except Exception:
                    message = ""
                if message:
                    projected[field] = message[:500]
        elif field in _EVENT_INTEGER_FIELDS:
            number = _bounded_nonnegative_integer(value)
            if number is not None:
                projected[field] = number
        elif field in _EVENT_NUMBER_FIELDS:
            number = _bounded_nonnegative_number(value)
            if number is not None:
                projected[field] = number
        elif field in _EVENT_BOOLEAN_FIELDS and type(value) is bool:
            projected[field] = value
        elif field == "usage":
            usage = _safe_usage(value)
            if usage:
                projected[field] = usage
        elif field == "remaining_cost_est":
            costs = _safe_remaining_cost(value)
            if costs:
                projected[field] = costs
        elif field == "projected_completion":
            projected_timestamp = _safe_timestamp(value)
            if projected_timestamp is not None:
                projected[field] = projected_timestamp[0]
    return projected


def _project_events(
    rows: list[dict[str, Any]],
    *,
    expected_run_mode: Literal["full", "pilot"],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        try:
            event = _project_event(
                row,
                expected_run_mode=expected_run_mode,
            )
        except Exception:
            event = None
        if event is not None:
            projected.append(event)
    return projected


def _latest_summary(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None
    fields = (
        "completed_samples",
        "sample_total",
        "rate_per_minute",
        "eta_seconds",
        "preliminary_eta_seconds",
        "projected_completion",
        "remaining_cost_est",
    )
    for event in events:
        if event.get("event") != "summary":
            continue
        candidate = {
            field: event[field]
            for field in fields
            if field in event
        }
        if candidate:
            selected = candidate
    return selected


def _event_positions(
    events: list[dict[str, Any]],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    current: dict[str, str | None] = {"sample_id": None, "stage": None}
    last_safe: dict[str, str | None] = {"sample_id": None, "stage": None}
    for event in events:
        name = event.get("event")
        sample_id = event.get("sample_id")
        stage = event.get("stage")
        if name == "resume_audit":
            if event.get("last_safe_sample_id") is not None:
                last_safe = {
                    "sample_id": event.get("last_safe_sample_id"),
                    "stage": event.get("last_safe_stage"),
                }
        elif name == "sample_started" and type(sample_id) is str:
            current = {"sample_id": sample_id, "stage": None}
        elif name == "stage_started" and type(sample_id) is str:
            current = {
                "sample_id": sample_id,
                "stage": stage if type(stage) is str else None,
            }
        elif name in {"stage_completed", "stage_reused"}:
            if type(sample_id) is str and type(stage) is str:
                last_safe = {"sample_id": sample_id, "stage": stage}
            if current == {"sample_id": sample_id, "stage": stage}:
                current["stage"] = None
        elif name == "stage_failed":
            if current == {"sample_id": sample_id, "stage": stage}:
                current["stage"] = None
        elif name == "sample_complete" and current.get("sample_id") == sample_id:
            current = {"sample_id": None, "stage": None}
        elif name in {"run_stopped", "run_completed"}:
            current = {"sample_id": None, "stage": None}
    return current, last_safe


def _latest_errors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for event in events:
        name = event.get("event")
        if name not in {"error", "warning", "stage_failed"} and not (
            name == "api_call_finished"
            and event.get("status") == "transport_error"
        ):
            continue
        error = {
            field: event[field]
            for field in (
                "category",
                "error_category",
                "error_type",
                "http_status",
                "message",
                "sample_id",
                "stage",
            )
            if field in event
        }
        if "category" not in error and "error_category" in error:
            error["category"] = error.pop("error_category")
        if error:
            errors.append(error)
    return errors[-5:]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _lifecycle_fields(events: list[dict[str, Any]]) -> dict[str, Any]:
    started_at: str | None = None
    last_timestamp: datetime | None = None
    elapsed: int | float | None = None
    for event in events:
        timestamp = _safe_timestamp(event.get("timestamp_utc"))
        if event.get("event") == "run_started" and timestamp is not None:
            started_at = timestamp[0]
        if timestamp is not None:
            last_timestamp = timestamp[1]
        event_elapsed = _bounded_nonnegative_number(
            event.get("process_elapsed_seconds")
        )
        if event_elapsed is not None:
            elapsed = event_elapsed
    event_age: int | None = None
    if last_timestamp is not None:
        try:
            now = _utc_now()
            if now.tzinfo is not None and now.utcoffset() is not None:
                difference = (
                    now.astimezone(timezone.utc) - last_timestamp
                ).total_seconds()
                if math.isfinite(difference) and 0 <= difference <= _STATUS_INTEGER_LIMIT:
                    event_age = int(round(difference))
        except Exception:
            event_age = None
    return {
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "event_age_seconds": event_age,
    }


def _observed_benign_false_positive_rate(
    manifest: list[dict[str, Any]],
    binary_outputs: Mapping[str, str],
) -> float | None:
    observed = 0
    false_positives = 0
    for row in manifest:
        sample_id = row.get("sample_id") or row.get("id")
        if type(sample_id) is not str or row.get("binary") != "benign":
            continue
        label = binary_outputs.get(sample_id)
        if label not in {"benign", "scam"}:
            continue
        observed += 1
        false_positives += label == "scam"
    return false_positives / observed if observed else None


def _pilot_metrics(output_root: Path) -> dict[str, Any] | None:
    for directory in ("pilot_60", "pilot_10"):
        metrics = _read_bounded_json_object(
            output_root / directory / "metrics.json",
            byte_limit=_ARTIFACT_FILE_BYTES,
        )
        if metrics is not None:
            return metrics
    return None


def _smoke_false_positive_rate(output_root: Path) -> float | None:
    metrics = _pilot_metrics(output_root)
    binary = metrics.get("binary") if type(metrics) is dict else None
    confusion = binary.get("confusion") if type(binary) is dict else None
    if type(confusion) is not dict:
        return None
    false_positives = _bounded_nonnegative_integer(confusion.get("fp"))
    true_negatives = _bounded_nonnegative_integer(confusion.get("tn"))
    if false_positives is None or true_negatives is None:
        return None
    benign = false_positives + true_negatives
    return false_positives / benign if benign else None


def _record_tracker_stage(
    tracker: ProgressTracker,
    event: dict[str, Any],
    *,
    source: Literal["smoke", "full"],
) -> None:
    if event.get("event") != "stage_completed":
        return
    stage = event.get("stage")
    model = event.get("model")
    usage = event.get("usage")
    elapsed = event.get("elapsed_seconds")
    if (
        type(stage) is not str
        or type(model) is not str
        or type(usage) is not dict
        or _bounded_nonnegative_number(elapsed) is None
    ):
        return
    try:
        tracker.record_stage(
            stage,
            model=model,
            usage=usage,
            elapsed_seconds=elapsed,
            source=source,
        )
    except Exception:
        return


def _record_tracker_samples(
    tracker: ProgressTracker,
    events: list[dict[str, Any]],
) -> None:
    for event in events:
        if event.get("event") != "sample_complete":
            continue
        sample_id = event.get("sample_id")
        index = event.get("sample_index")
        backend_error = event.get("backend_error")
        if (
            type(sample_id) is not str
            or type(index) is not int
            or index < 1
            or type(backend_error) is not bool
        ):
            continue
        try:
            tracker.complete_sample(
                sample_id,
                index=index,
                backend_error=backend_error,
            )
        except Exception:
            continue


def _tracker_estimates(
    config: dict[str, Any],
    output_root: Path,
    *,
    events: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    complete_stages: frozenset[tuple[str, str]],
    binary_outputs: Mapping[str, str],
    calls: Mapping[str, Any],
) -> tuple[dict[str, int], int | None, int | None, dict[str, float], list[str]]:
    observed_rate = _observed_benign_false_positive_rate(
        manifest,
        binary_outputs,
    )
    smoke_rate = _smoke_false_positive_rate(output_root)
    projection_rate = smoke_rate
    if projection_rate is None:
        projection_rate = 0.0 if observed_rate is not None else 1.0
    try:
        remaining_counts = project_remaining_stage_counts(
            manifest,
            complete_stages=complete_stages,
            binary_outputs=binary_outputs,
            observed_benign_false_positive_rate=observed_rate,
            smoke_benign_false_positive_rate=projection_rate,
        )
    except Exception:
        return {}, None, None, {}, []

    observability = config.get("observability")
    models = config.get("models")
    pricing = config.get("pricing")
    experiment = config.get("experiment")
    if (
        type(models) is not dict
        or type(pricing) is not dict
        or type(experiment) is not dict
    ):
        return remaining_counts, None, None, {}, []
    try:
        settings = ObservabilitySettings.from_mapping(
            observability if type(observability) is dict else None
        )
        tracker = ProgressTracker(
            settings,
            total_samples=len(manifest),
            api_call_limit=_nonnegative_config_integer(
                experiment.get("api_call_limit")
            ),
            pricing=pricing,
            text_model=models.get("text"),
            vision_model=models.get("vision"),
        )
    except Exception:
        return remaining_counts, None, None, {}, []

    pilot_warnings: list[str] = []
    for directory in ("pilot_10", "pilot_60"):
        pilot_rows, warnings = read_jsonl_tail(
            output_root / directory / "run_events.jsonl"
        )
        pilot_events = _project_events(
            pilot_rows,
            expected_run_mode="pilot",
        )
        pilot_warnings.extend(
            f"{directory}: {warning}" for warning in warnings
        )
        for event in pilot_events:
            _record_tracker_stage(tracker, event, source="smoke")
    for event in events:
        _record_tracker_stage(tracker, event, source="full")
    _record_tracker_samples(tracker, events)

    try:
        snapshot = tracker.snapshot(
            remaining_stage_counts=remaining_counts,
        )
        eta = snapshot.eta_seconds
        preliminary_eta = snapshot.preliminary_eta_seconds
    except Exception:
        eta = preliminary_eta = None
    try:
        costs = tracker.estimate_remaining_cost(
            remaining_counts,
            outstanding_by_stage=(
                calls.get("outstanding_by_stage")
                if type(calls.get("outstanding_by_stage")) is dict
                else {}
            ),
            replay_required_by_stage=(
                calls.get("replay_required_by_stage")
                if type(calls.get("replay_required_by_stage")) is dict
                else {}
            ),
        )
        costs = _safe_remaining_cost(costs)
    except Exception:
        costs = {}
    return remaining_counts, eta, preliminary_eta, costs, pilot_warnings


def _nested_dict(value: object, *keys: str) -> dict[str, Any] | None:
    current = value
    for key in keys:
        if type(current) is not dict:
            return None
        current = current.get(key)
    return current if type(current) is dict else None


def _acceptance_passed(metrics: object) -> bool:
    gates = _nested_dict(metrics, "engineering", "acceptance_gates")
    return type(gates) is dict and gates.get("passed") is True


def _smoke_complete(output_root: Path) -> bool:
    metrics = _read_bounded_json_object(
        output_root / "pilot_10" / "metrics.json",
        byte_limit=_ARTIFACT_FILE_BYTES,
    )
    engineering = metrics.get("engineering") if type(metrics) is dict else None
    return (
        type(engineering) is dict
        and engineering.get("sample_count") == 10
        and _acceptance_passed(metrics)
    )


def _preflight_complete(output_root: Path, *, expected_total: int) -> bool:
    preflight = _read_bounded_json_object(
        output_root / "full" / "preflight.json",
        byte_limit=_ARTIFACT_FILE_BYTES,
    )
    return (
        type(preflight) is dict
        and preflight.get("status") == "validated"
        and preflight.get("selected_sample_count") == expected_total
    )


def _inference_complete(
    output_root: Path,
    *,
    events: list[dict[str, Any]],
    expected_total: int,
    manifest_total: int,
    manifest_ids: frozenset[str],
    prediction_audit: PredictionAudit,
    sample_complete: int,
    sample_partial: int,
    sample_error: int,
    contract_corrupt: bool,
) -> bool:
    durable_complete = (
        expected_total > 0
        and manifest_total == expected_total
        and len(manifest_ids) == expected_total
        and sample_complete == manifest_total
        and sample_partial == 0
        and sample_error == 0
        and not contract_corrupt
        and not prediction_audit.corrupt
        and prediction_audit.row_count == expected_total
        and prediction_audit.sample_ids == manifest_ids
    )
    if not durable_complete:
        return False
    completed_event = any(
        (
            event.get("event") == "run_completed"
            and event.get("sample_count") == expected_total
            and event.get("gates_passed") is True
        )
        for event in events
    )
    metrics = _read_bounded_json_object(
        output_root / "full" / "metrics.json",
        byte_limit=_ARTIFACT_FILE_BYTES,
    )
    return completed_event or _acceptance_passed(metrics)


def _bertscore_complete(output_root: Path, *, expected_total: int) -> bool:
    metrics = _read_bounded_json_object(
        output_root / "full" / "metrics_final.json",
        byte_limit=_ARTIFACT_FILE_BYTES,
    )
    rationale = _nested_dict(metrics, "rationale", "end_to_end")
    engineering = metrics.get("engineering") if type(metrics) is dict else None
    sample_count = (
        engineering.get("prediction_sample_count")
        if type(engineering) is dict
        else None
    )
    if type(sample_count) is not int and type(engineering) is dict:
        sample_count = engineering.get("sample_count")
    return (
        type(rationale) is dict
        and rationale.get("bertscore_status") == "computed"
        and type(engineering) is dict
        and sample_count == expected_total
    )


def _safe_delivery(value: object) -> dict[str, Any] | None:
    if type(value) is not dict:
        return None
    stage = value.get("stage")
    status = value.get("status")
    exit_code = value.get("exit_code")
    if stage not in {"package", "upload", "publish"}:
        return None
    if status not in {"pending", "running", "complete", "failed"}:
        return None
    projected: dict[str, Any] = {"stage": stage, "status": status}
    if type(exit_code) is int and exit_code in {30, 31, 32}:
        projected["exit_code"] = exit_code
    return projected


def _safe_durable_delivery(value: object) -> dict[str, Any] | None:
    if type(value) is not dict:
        return None
    packaged = value.get("packaged")
    uploaded = value.get("uploaded")
    published = value.get("published")
    if not all(type(item) is bool for item in (packaged, uploaded, published)):
        return None
    if (uploaded and not packaged) or (published and not uploaded):
        return None
    last_exit_code = value.get("last_exit_code")
    if last_exit_code is not None and last_exit_code not in {30, 31, 32}:
        return None
    if type(last_exit_code) is int:
        stage = {30: "package", 31: "upload", 32: "publish"}[
            last_exit_code
        ]
        return {
            "stage": stage,
            "status": "failed",
            "exit_code": last_exit_code,
        }
    if not packaged:
        return {"stage": "package", "status": "pending"}
    if not uploaded:
        return {"stage": "upload", "status": "pending"}
    if not published:
        return {"stage": "publish", "status": "pending"}
    return {"stage": "publish", "status": "complete"}


_DELIVERY_ENTRY_FIELDS = {
    "mode",
    "packaged",
    "archive_sha256",
    "uploaded",
    "published",
    "last_exit_code",
}


def _safe_mode_delivery(
    value: object,
    *,
    expected_mode: str,
) -> dict[str, Any] | None:
    if type(value) is not dict or set(value) != _DELIVERY_ENTRY_FIELDS:
        return None
    if value.get("mode") != expected_mode:
        return None
    packaged = value.get("packaged")
    uploaded = value.get("uploaded")
    published = value.get("published")
    if any(type(item) is not bool for item in (packaged, uploaded, published)):
        return None
    archive_sha256 = value.get("archive_sha256")
    if packaged:
        if (
            type(archive_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
        ):
            return None
    elif archive_sha256 is not None:
        return None
    if (uploaded and not packaged) or (published and not uploaded):
        return None
    last_exit_code = value.get("last_exit_code")
    if last_exit_code is not None and (
        type(last_exit_code) is not int
        or last_exit_code not in {30, 31, 32}
    ):
        return None
    if last_exit_code == 30 and packaged:
        return None
    if last_exit_code == 31 and (not packaged or uploaded):
        return None
    if last_exit_code == 32 and (not uploaded or published):
        return None
    if type(last_exit_code) is int:
        return {
            "stage": {30: "package", 31: "upload", 32: "publish"}[
                last_exit_code
            ],
            "status": "failed",
            "exit_code": last_exit_code,
        }
    if not packaged:
        return {"stage": "package", "status": "pending"}
    if not uploaded:
        return {"stage": "upload", "status": "pending"}
    if not published:
        return {"stage": "publish", "status": "pending"}
    return {"stage": "publish", "status": "complete"}


def _default_delivery_modes() -> dict[str, dict[str, Any]]:
    return {
        "inference": {"stage": "package", "status": "pending"},
        "final": {"stage": "package", "status": "pending"},
    }


def _delivery_status(
    output_root: Path,
    *,
    events: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    int | None,
    bool,
]:
    event_modes: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "delivery_stage":
            mode = _safe_delivery_mode(event.get("delivery_mode"))
            if mode is None and "delivery_mode" not in event:
                mode = "inference"
            delivery = _safe_delivery(event)
            if mode is not None and delivery is not None:
                event_modes[mode] = delivery
    state_path = output_root / "full" / "delivery_state.json"
    modes = _default_delivery_modes()
    corrupt = False
    try:
        loaded = load_delivery_states(state_path)
        if loaded is None:
            modes.update(event_modes)
        else:
            projected = {
                mode: _safe_mode_delivery(
                    asdict(loaded[mode]),
                    expected_mode=mode,
                )
                for mode in ("inference", "final")
            }
            if any(value is None for value in projected.values()):
                raise ResultContractError("Delivery state projection is invalid")
            modes = {
                "inference": projected["inference"],
                "final": projected["final"],
            }
    except ResultContractError:
        corrupt = True

    selected = modes["inference"]
    failure: int | None = None
    if selected.get("status") == "failed":
        failure = selected.get("exit_code")
        if type(failure) is not int:
            failure = {"package": 30, "upload": 31, "publish": 32}[
                selected["stage"]
            ]
        selected["exit_code"] = failure
    return modes, selected, failure, corrupt


def _stop_facts(
    events: list[dict[str, Any]],
    *,
    tmux: TmuxState,
) -> dict[str, Any]:
    active_attempt: str | None = None
    for event in events:
        if event.get("event") == "run_started":
            attempt_id = event.get("attempt_id")
            if type(attempt_id) is str:
                active_attempt = attempt_id

    def stop_attempt(event: Mapping[str, Any]) -> str | None:
        if event.get("event") == "shell_exited":
            runner_attempt_id = event.get("runner_attempt_id")
            if type(runner_attempt_id) is str:
                return runner_attempt_id
        attempt_id = event.get("attempt_id")
        return attempt_id if type(attempt_id) is str else None

    indexed_facts = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") in {"run_stopped", "shell_exited"}
    ]
    if active_attempt is None and indexed_facts:
        active_attempt = stop_attempt(indexed_facts[-1][1])
    if active_attempt is not None:
        indexed_facts = [
            (index, event)
            for index, event in indexed_facts
            if stop_attempt(event) == active_attempt
        ]

    run_stop: dict[str, Any] | None = None
    run_stop_index = -1
    for index, event in indexed_facts:
        if event.get("event") == "run_stopped":
            run_stop = event
            run_stop_index = index
    selected_index, selected = (
        indexed_facts[-1] if indexed_facts else (-1, None)
    )
    exit_code: int | None = None
    if selected is not None and type(selected.get("exit_code")) is int:
        exit_code = selected["exit_code"]
    elif tmux.state == "dead":
        exit_code = tmux.exit_code
    selected_forced = bool((selected or {}).get("forced") is True)
    graceful = bool(
        exit_code in {130, 143}
        and not selected_forced
        and (
            (selected or {}).get("event") == "run_stopped"
            or (
                run_stop is not None
                and run_stop_index <= selected_index
                and run_stop.get("exit_code") == exit_code
                and run_stop.get("forced") is not True
            )
        )
    )
    return {
        "exit_code": exit_code,
        "graceful_stop_event": graceful,
        "forced_stop": selected_forced,
        "stop_reason": (run_stop or {}).get("reason"),
        "stop_category": (run_stop or {}).get("category"),
    }


def _nonnegative_config_integer(value: object, *, default: int = 0) -> int:
    if type(value) is int and value >= 0:
        return value
    return default


def collect_status(
    config_path: Path,
    *,
    tmux_session: str,
    tmux_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    output_root = _configured_output_root(config)
    _validate_output_reads(output_root)
    state_root = output_root / "state_full"
    tmux = query_tmux(tmux_session, runner=tmux_runner)
    raw_events, event_warnings = read_jsonl_tail(
        output_root / "full" / "run_events.jsonl"
    )
    events = _project_events(raw_events, expected_run_mode="full")
    config_hash, run_config_corrupt = _read_config_hash(state_root)
    checkpoint_dir = state_root / "checkpoints"
    ledger_path = state_root / "api_call_ledger.jsonl"
    unbound_durable_state = config_hash is None and _durable_state_present(
        checkpoint_dir,
        ledger_path,
    )
    if unbound_durable_state:
        run_config_corrupt = True
        checkpoints = CheckpointAudit(frozenset(), frozenset(), {}, ())
        binary_outputs: dict[str, str] = {}
        ledger = _conservative_ledger_audit()
    else:
        checkpoint_payloads, preflight_corrupt = _preflight_checkpoint_payloads(
            checkpoint_dir
        )
        if preflight_corrupt:
            checkpoints = CheckpointAudit(
                frozenset(), frozenset(), {}, preflight_corrupt
            )
            binary_outputs = {}
        else:
            checkpoints = _scan_checkpoint_status(
                checkpoint_dir,
                config_hash=config_hash,
            )
            binary_outputs = _checkpoint_binary_outputs(
                checkpoint_payloads,
                config_hash=config_hash,
                complete_stages=checkpoints.complete_stages,
            )
        ledger = _scan_ledger_status(
            ledger_path,
            config_hash=config_hash,
            checkpoint_stages=checkpoints.complete_stages,
        )
    selected_manifest_path = _manifest_path(config, output_root)
    if selected_manifest_path is None:
        manifest, manifest_corrupt = [], False
    else:
        manifest, manifest_corrupt = _read_manifest(selected_manifest_path)
    try:
        progress = summarize_sample_progress(
            manifest,
            complete_stages=checkpoints.complete_stages,
            error_stages=checkpoints.error_stages,
            binary_outputs=binary_outputs,
        )
    except Exception:
        manifest_corrupt = True
        progress = summarize_sample_progress(
            [],
            complete_stages=frozenset(),
            error_stages=frozenset(),
            binary_outputs={},
        )
    experiment = config.get("experiment")
    experiment_values = experiment if type(experiment) is dict else {}
    call_limit = _nonnegative_config_integer(
        experiment_values.get("api_call_limit")
    )
    calls = {
        "reserved": ledger.reserved,
        "finished": ledger.finished,
        "completed": ledger.completed,
        "transport_error": ledger.transport_error,
        "outstanding": ledger.outstanding,
        "remaining": max(0, call_limit - ledger.reserved),
        "corrupt": ledger.corrupt,
        "retries": sum(
            type(event.get("event")) is str
            and event.get("event") == "retry_scheduled"
            for event in events
        ),
        "outstanding_by_stage": dict(ledger.outstanding_by_stage),
        "replay_required_by_stage": dict(ledger.replay_required_by_stage),
    }
    expected_total = _nonnegative_config_integer(
        experiment_values.get("expected_full_sample_count")
    )
    total = len(manifest) if manifest else expected_total
    manifest_ids = frozenset(
        sample_id
        for row in manifest
        for sample_id in [row.get("sample_id") or row.get("id")]
        if type(sample_id) is str and sample_id
    )
    prediction_audit = _audit_prediction_rows(
        output_root / "full" / "predictions.jsonl"
    )
    prediction_rows = prediction_audit.row_count
    predictions_corrupt = prediction_audit.corrupt or bool(
        prediction_audit.sample_ids - manifest_ids
    )
    checkpoint_corrupt = list(checkpoints.corrupt_files)
    if run_config_corrupt:
        checkpoint_corrupt.append("run_config.json")
    if manifest_corrupt:
        checkpoint_corrupt.append("manifest.jsonl")
    if predictions_corrupt:
        checkpoint_corrupt.append("predictions.jsonl")
    checkpoint_corrupt = [
        safe_name
        for raw_name in checkpoint_corrupt
        for safe_name in [_safe_display_text(raw_name, limit=200)]
        if safe_name
    ]
    summary = _latest_summary(events)
    remaining_stage_counts: dict[str, int] = {}
    eta_seconds: int | float | None = None
    preliminary_eta_seconds: int | float | None = None
    remaining_cost_est: dict[str, float] = {}
    projected_completion: str | None = None
    throughput: int | float | None = None
    if summary is not None:
        eta_seconds = summary.get("eta_seconds")
        preliminary_eta_seconds = summary.get("preliminary_eta_seconds")
        projected_completion = summary.get("projected_completion")
        throughput = summary.get("rate_per_minute")
        remaining_cost_est = dict(
            summary.get("remaining_cost_est") or {}
        )
    else:
        (
            remaining_stage_counts,
            eta_seconds,
            preliminary_eta_seconds,
            remaining_cost_est,
            pilot_warnings,
        ) = _tracker_estimates(
            config,
            output_root,
            events=events,
            manifest=manifest,
            complete_stages=checkpoints.complete_stages,
            binary_outputs=binary_outputs,
            calls=calls,
        )
        event_warnings.extend(pilot_warnings)
        if eta_seconds is not None:
            try:
                now = _utc_now()
                if now.tzinfo is not None and now.utcoffset() is not None:
                    projected_completion = (
                        now.astimezone(timezone.utc)
                        + timedelta(seconds=eta_seconds)
                    ).isoformat()
            except (OverflowError, TypeError, ValueError):
                projected_completion = None
    current, last_safe = _event_positions(events)
    lifecycle = _lifecycle_fields(events)
    elapsed_for_rate = _bounded_nonnegative_number(
        lifecycle.get("elapsed_seconds")
    )
    if (
        throughput is None
        and elapsed_for_rate is not None
        and elapsed_for_rate > 0
        and progress.complete > 0
    ):
        throughput = progress.complete * 60 / elapsed_for_rate
    threshold = 5
    observability = config.get("observability")
    if type(observability) is dict:
        configured_threshold = _bounded_nonnegative_integer(
            observability.get("breaker_consecutive_backend_errors")
        )
        if configured_threshold is not None and configured_threshold > 0:
            threshold = configured_threshold
    try:
        breaker_state = FailureCircuitBreaker.from_events(
            events,
            threshold=threshold,
        )
        breaker = {
            "consecutive": breaker_state.consecutive,
            "opened": breaker_state.opened,
        }
    except Exception:
        breaker = {"consecutive": threshold, "opened": True}
    stop_facts = _stop_facts(events, tmux=tmux)
    preflight_complete = _preflight_complete(
        output_root,
        expected_total=expected_total,
    )
    smoke_complete = _smoke_complete(output_root)
    inference_complete = _inference_complete(
        output_root,
        events=events,
        expected_total=expected_total,
        manifest_total=len(manifest),
        manifest_ids=manifest_ids,
        prediction_audit=prediction_audit,
        sample_complete=progress.complete,
        sample_partial=progress.partial,
        sample_error=progress.error,
        contract_corrupt=bool(checkpoint_corrupt) or ledger.corrupt,
    )
    bertscore_complete = _bertscore_complete(
        output_root,
        expected_total=expected_total,
    )
    delivery_modes, delivery, delivery_failure, delivery_corrupt = _delivery_status(
        output_root,
        events=events,
    )
    status: dict[str, Any] = {
        "tmux": asdict(tmux),
        "config_hash": config_hash,
        **lifecycle,
        **stop_facts,
        "samples": {
            "complete": progress.complete,
            "partial": progress.partial,
            "error": progress.error,
            "total": total,
        },
        "required_stage_counts": dict(progress.required_stage_counts),
        "stages": {
            "complete": len(checkpoints.complete_stages),
            "reused": sum(
                event.get("event") == "stage_reused" for event in events
            ),
            "failed": len(checkpoints.error_stages),
        },
        "current": current,
        "last_safe": last_safe,
        "remaining_stage_counts": remaining_stage_counts,
        "prediction_rows": prediction_rows,
        "calls": calls,
        "eta_seconds": eta_seconds,
        "preliminary_eta_seconds": preliminary_eta_seconds,
        "projected_completion": projected_completion,
        "throughput": throughput,
        "remaining_cost_est": remaining_cost_est,
        "errors": _latest_errors(events),
        "breaker": breaker,
        "preflight_complete": preflight_complete,
        "smoke_complete": smoke_complete,
        "inference_complete": inference_complete,
        "bertscore_complete": bertscore_complete,
        "delivery": delivery,
        "delivery_modes": delivery_modes,
        "delivery_failure": delivery_failure,
        "delivery_corrupt": delivery_corrupt,
        "artifact_state": {
            "preflight": "complete" if preflight_complete else "pending",
            "smoke": "complete" if smoke_complete else "pending",
            "inference": "complete" if inference_complete else "pending",
            "bertscore": "complete" if bertscore_complete else "pending",
            "delivery": delivery.get("status"),
        },
        "event_warnings": event_warnings,
        "checkpoint_corrupt": sorted(set(checkpoint_corrupt)),
    }
    status["next_action"] = choose_next_action(status)
    return status


def _mapping_get(value: object, key: str, default: Any = None) -> Any:
    if not isinstance(value, Mapping):
        return default
    try:
        return value.get(key, default)
    except Exception:
        return default


def _delivery_recovery_action(
    delivery: object,
    *,
    mode: str,
) -> str | None:
    stage = _mapping_get(delivery, "stage")
    status = _mapping_get(delivery, "status")
    if status == "complete":
        return None
    if stage not in {"package", "upload", "publish"}:
        stage = "package"
    if mode == "final":
        return f"bash scripts/server/recover_delivery.sh --mode final {stage}"
    return f"bash scripts/server/recover_delivery.sh {stage}"


def choose_next_action(status: Mapping[str, Any]) -> str:
    tmux = _mapping_get(status, "tmux", {})
    if _mapping_get(tmux, "state") == "running":
        return "monitor"

    calls = _mapping_get(status, "calls", {})
    if _mapping_get(calls, "corrupt") is True:
        return "inspect contract drift; do not force resume"
    corrupt_checkpoints = _mapping_get(status, "checkpoint_corrupt", ())
    if isinstance(corrupt_checkpoints, (list, tuple)) and corrupt_checkpoints:
        return "inspect contract drift; do not force resume"
    if _mapping_get(status, "delivery_corrupt") is True:
        return "inspect contract drift; do not force resume"

    delivery_actions = {
        30: "bash scripts/server/recover_delivery.sh package",
        31: "bash scripts/server/recover_delivery.sh upload",
        32: "bash scripts/server/recover_delivery.sh publish",
    }
    delivery_modes = _mapping_get(status, "delivery_modes")
    has_delivery_modes = isinstance(delivery_modes, Mapping)
    delivery_failure = _mapping_get(status, "delivery_failure")
    if (
        not has_delivery_modes
        and type(delivery_failure) is int
        and delivery_failure in delivery_actions
    ):
        return delivery_actions[delivery_failure]
    if _mapping_get(status, "inference_complete") is True:
        inference_delivery = (
            _mapping_get(delivery_modes, "inference", {})
            if has_delivery_modes
            else {"stage": "package", "status": "pending"}
        )
        inference_action = _delivery_recovery_action(
            inference_delivery,
            mode="inference",
        )
        if inference_action is not None:
            return inference_action
        if _mapping_get(status, "bertscore_complete") is not True:
            return "finalize BERTScore"
        final_delivery = (
            _mapping_get(delivery_modes, "final", {})
            if has_delivery_modes
            else {"stage": "package", "status": "pending"}
        )
        final_action = _delivery_recovery_action(
            final_delivery,
            mode="final",
        )
        if final_action is not None:
            return final_action

    remaining_calls = _mapping_get(calls, "remaining")
    if type(remaining_calls) is int and remaining_calls == 0:
        return "physical call cap exhausted; do not resume this version"
    breaker = _mapping_get(status, "breaker", {})
    if _mapping_get(breaker, "opened") is True:
        return "inspect errors, then start_full.sh --ack-breaker"

    exit_actions = {
        20: "replenish credentials/balance, then resume",
        21: "physical call cap exhausted; do not resume this version",
        22: "inspect errors, then start_full.sh --ack-breaker",
        23: "inspect contract drift; do not force resume",
        2: "inspect failed acceptance gates",
    }
    exit_code = _mapping_get(status, "exit_code")
    if type(exit_code) is int and exit_code in exit_actions:
        return exit_actions[exit_code]
    if type(exit_code) is int and exit_code in {130, 143}:
        if _mapping_get(status, "graceful_stop_event") is True:
            return "resume inference"
        return "inspect forced interruption and orphan call"

    if (
        _mapping_get(status, "preflight_complete") is True
        and _mapping_get(status, "smoke_complete") is not True
    ):
        return "run paid 10-page smoke"
    if (
        _mapping_get(status, "smoke_complete") is True
        and _mapping_get(status, "inference_complete") is not True
    ):
        return "resume inference"
    return "inspect status and resume inference"


def _nonnegative_finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if (
        number < 0
        or number > _STATUS_INTEGER_LIMIT
        or not math.isfinite(number)
    ):
        return None
    return number


def _safe_display_text(value: object, *, limit: int = 200) -> str:
    if type(value) is not str:
        return ""
    try:
        text = _sanitize_control_text(value)[:limit]
    except Exception:
        return ""
    printable = "".join(
        character if character.isprintable() else " "
        for character in text
    )
    return " ".join(printable.split())


def render_status(status: Mapping[str, Any]) -> str:
    lines: list[str] = []
    tmux = _mapping_get(status, "tmux", {})
    tmux_state = _mapping_get(tmux, "state")
    safe_tmux_state = (
        tmux_state
        if type(tmux_state) is str
        and tmux_state in {"absent", "running", "dead"}
        else "absent"
    )
    tmux_parts = [f"tmux: {safe_tmux_state}"]
    pid = _bounded_nonnegative_integer(_mapping_get(tmux, "pid"))
    if pid is not None and pid > 0:
        tmux_parts.append(f"pid={pid}")
    pane_exit = _bounded_nonnegative_integer(
        _mapping_get(tmux, "exit_code")
    )
    if pane_exit is not None:
        tmux_parts.append(f"exit={pane_exit}")
    lines.append(" ".join(tmux_parts))

    started_at = _safe_display_text(
        _mapping_get(status, "started_at"),
        limit=80,
    )
    if started_at:
        lines.append(f"started: {started_at}")
    elapsed = _nonnegative_finite_number(
        _mapping_get(status, "elapsed_seconds")
    )
    if elapsed is not None:
        lines.append(f"elapsed: {format_duration(elapsed)}")
    event_age = _nonnegative_finite_number(
        _mapping_get(status, "event_age_seconds")
    )
    if event_age is not None:
        lines.append(f"event age: {format_duration(event_age)}")

    samples = _mapping_get(status, "samples")
    if isinstance(samples, Mapping):
        complete, partial, error, total = (
            _bounded_nonnegative_integer(_mapping_get(samples, name, 0))
            for name in ("complete", "partial", "error", "total")
        )
        if all(value is not None for value in (complete, partial, error, total)):
            lines.append(
                f"complete: {complete} / {total} "
                f"(partial: {partial}, error: {error})"
            )

    stages = _mapping_get(status, "stages")
    if isinstance(stages, Mapping):
        stage_values = tuple(
            _bounded_nonnegative_integer(_mapping_get(stages, name, 0))
            for name in ("complete", "reused", "failed")
        )
        if all(value is not None for value in stage_values):
            lines.append(
                "stages: complete={} reused={} failed={}".format(
                    *stage_values
                )
            )

    for label, key in (("current", "current"), ("last safe", "last_safe")):
        position = _mapping_get(status, key, {})
        sample_id = _safe_display_text(_mapping_get(position, "sample_id"), limit=100)
        stage = _safe_display_text(_mapping_get(position, "stage"), limit=100)
        if sample_id or stage:
            lines.append(f"{label}: sample={sample_id or '-'} stage={stage or '-'}")

    calls = _mapping_get(status, "calls", {})
    call_names = ("reserved", "finished", "outstanding", "remaining")
    call_values = tuple(_mapping_get(calls, name) for name in call_names)
    safe_call_parts = [
        f"{name}={safe_value}"
        for name, value in zip(call_names, call_values)
        for safe_value in [_bounded_nonnegative_integer(value)]
        if safe_value is not None
    ]
    if safe_call_parts:
        lines.append("calls: " + " ".join(safe_call_parts))

    eta = _nonnegative_finite_number(_mapping_get(status, "eta_seconds"))
    if eta is not None:
        lines.append(f"approximate ETA: {format_duration(eta)}")
    preliminary_eta = _nonnegative_finite_number(
        _mapping_get(status, "preliminary_eta_seconds")
    )
    if eta is None and preliminary_eta is not None:
        lines.append(
            f"preliminary ETA: {format_duration(preliminary_eta)}"
        )
    throughput = _nonnegative_finite_number(
        _mapping_get(status, "throughput")
    )
    if throughput is not None:
        lines.append(f"throughput: {throughput:.2f}/min")
    projected_completion = _safe_display_text(
        _mapping_get(status, "projected_completion"),
        limit=80,
    )
    if projected_completion:
        lines.append(f"projected completion: {projected_completion}")

    costs = _mapping_get(status, "remaining_cost_est", {})
    cost_parts: list[str] = []
    for currency in ("CNY", "USD"):
        amount = _nonnegative_finite_number(_mapping_get(costs, currency))
        if amount is not None:
            cost_parts.append(f"{currency} {amount:.2f}")
    if cost_parts:
        lines.append("remaining cost estimate: " + " + ".join(cost_parts))

    errors = _mapping_get(status, "errors", ())
    if type(errors) in {list, tuple}:
        for item in errors[-5:]:
            category = _safe_display_text(_mapping_get(item, "category"), limit=100)
            message = _safe_display_text(_mapping_get(item, "message"), limit=200)
            if category or message:
                lines.append(f"error: {category} {message}".rstrip())

    breaker = _mapping_get(status, "breaker", {})
    consecutive = _bounded_nonnegative_integer(
        _mapping_get(breaker, "consecutive")
    )
    opened = _mapping_get(breaker, "opened")
    if consecutive is not None and type(opened) is bool:
        lines.append(f"breaker: consecutive={consecutive} opened={str(opened).lower()}")

    action = _safe_display_text(_mapping_get(status, "next_action"), limit=300)
    if action:
        lines.append(f"next action: {action}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the ScamMark full-v2 run without changing it"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tmux-session", required=True)
    arguments = parser.parse_args(argv)
    try:
        status = collect_status(
            arguments.config,
            tmux_session=arguments.tmux_session,
            tmux_runner=subprocess.run,
        )
    except (OSError, ValueError):
        parser.error("status inspection failed; inspect the v2 configuration")
    print(
        render_status(status)
    )
    return 0


__all__ = [
    "TmuxState",
    "choose_next_action",
    "collect_status",
    "main",
    "query_tmux",
    "read_jsonl_tail",
    "render_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
