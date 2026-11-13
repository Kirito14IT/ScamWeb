"""Fail-closed validation for complete ScamMark result directories."""

from __future__ import annotations

import argparse
import json
import gzip
import hashlib
import math
import os
import re
import stat
import sys
import tarfile
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
from zoneinfo import ZoneInfo

try:
    import fcntl
except ImportError:  # pragma: no cover - delivery runs on the POSIX server
    fcntl = None  # type: ignore[assignment]


DEFAULT_EXPECTED_SAMPLE_COUNT = 5658
FINAL_EXPECTED_PREDICTION_COUNT = 5658
ResultMode = Literal["inference", "final"]

RUN_CONFIG_AUDIT_FIELDS = (
    "schema",
    "run_mode",
    "source_sha256",
    "implementation_sha256",
    "contract_sample_ids_sha256",
    "selected_image_sha256_count",
    "taxonomy",
)
LEDGER_AUDIT_FIELDS = (
    "timestamp",
    "event",
    "call_id",
    "sample_id",
    "stage",
    "status",
    "error_category",
    "sanitized_error_message",
)
INFERENCE_ARCHIVE_PAYLOAD = (
    "full/predictions.jsonl",
    "full/metrics.json",
    "full/manifest.jsonl",
    "full/preflight.json",
    "full/full_summary.md",
    "full/human_evaluation_candidates.csv",
    "audit/run_config.sanitized.json",
    "audit/api_call_ledger.sanitized.jsonl",
    "logs/run_events.jsonl",
    "logs/full_run.log",
    "config/qwen3_dashscope_server_full_v2.toml",
    "frozen/dataset_version.json",
    "frozen/verification_report.json",
    "provenance.json",
)
FINAL_ARCHIVE_PAYLOAD = (
    "full/metrics.json",
    "full/metrics_final.json",
    "full/final_evaluation_summary.md",
    "environment/bertscore_environment.txt",
    "environment/bertscore_asset_status.json",
    "logs/bertscore_finalize.log",
    "provenance.json",
)
INFERENCE_ARCHIVE_ROOT = "scammark-qwen3-dashscope-full-v2-inference"
FINAL_ARCHIVE_ROOT = "scammark-qwen3-dashscope-full-v2-final-evaluation"
DELIVERY_STATE_SCHEMA = "scammark-delivery-state-v1"
SANITIZED_RUN_CONFIG_SCHEMA = "scammark-run-config-sanitized-v1"
SANITIZED_ERROR_MESSAGE = "<redacted>"
_PRODUCTION_STAGES = frozenset(
    {"url", "text", "visual", "cross", "binary", "type", "grounding"}
)
_SAFE_ERROR_CATEGORIES = frozenset(
    {
        "auth_billing",
        "api_budget",
        "contract_drift",
        "transient",
        "sample_permanent",
        "unexpected",
    }
)

_LEAK_PATTERNS = (
    (
        "authorization",
        re.compile(
            r"(?<![A-Za-z0-9_])[\"']?authorization[\"']?\s*[:=]",
            re.IGNORECASE,
        ),
    ),
    ("bearer", re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)),
    (
        "key-marker",
        re.compile(
            r"(?<![A-Za-z0-9_])[\"']?(?:key|api[_-]?key|access[_-]?key|"
            r"secret[_-]?key|private[_-]?key)[\"']?\s*[:=]",
            re.IGNORECASE,
        ),
    ),
    (
        "standalone-key",
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{4,}", re.IGNORECASE),
    ),
    ("data-url", re.compile(r"\bdata:[^,\s]+,", re.IGNORECASE)),
    (
        "prompt-or-raw-marker",
        re.compile(
            r"(?<![A-Za-z0-9_])[\"']?(?:system[_ -]?prompt|user[_ -]?prompt|"
            r"prompt(?:_payload)?|raw[_ -]?(?:request|response|content|payload))"
            r"[\"']?\s*[:=]",
            re.IGNORECASE,
        ),
    ),
    (
        "workspace-label",
        re.compile(
            r"(?:/workspace(?:/|\b)|/home/[^/\s]+/|"
            r"[A-Za-z]:[\\/](?:Users|workspace)[\\/]|"
            r"\b[A-Za-z0-9-]+\.cn-[a-z0-9-]+\.maas\.aliyuncs\.com\b|"
            r"(?<![A-Za-z0-9_])[\"']?(?:workspace|runtime_root|"
            r"dataset_path_mapping)[\"']?\s*[:=])",
            re.IGNORECASE,
        ),
    ),
    (
        "url-query",
        re.compile(r"https?://[^\s\"'<>?]+\?[^\s\"'<>]*", re.IGNORECASE),
    ),
)


class ResultContractError(ValueError):
    """Raised when a result directory does not satisfy the release contract."""


class ArchiveRecoveryRequired(ResultContractError):
    """Raised when a stable old archive sibling must be restored manually."""

    def __init__(self, backup_name: str, archive_name: str) -> None:
        safe_backup = re.fullmatch(
            r"\.[A-Za-z0-9._-]+\.recovery-[0-9a-f]{32}\.bak",
            backup_name,
        )
        safe_archive = re.fullmatch(r"[A-Za-z0-9._-]+", archive_name)
        if safe_backup is None or safe_archive is None:
            super().__init__("result archive recovery required")
            return
        super().__init__(
            "result archive recovery required: verify and rename "
            f"{backup_name} to {archive_name}, then verify or regenerate "
            f"{archive_name}.sha256"
        )


class _StrictJSONError(ValueError):
    pass


_GENERIC_LEAK_LOCATION_LABELS = {
    "staged": "staged file",
    "audit": "audit projection",
}


def _reject_generic_leak_markers(
    text: str,
    *,
    scope: Literal["staged", "audit"],
    ordinal: int,
) -> None:
    """Reject a generic leak using only a fixed category and ordinal."""

    location = _GENERIC_LEAK_LOCATION_LABELS[scope]
    for category, pattern in _LEAK_PATTERNS:
        if pattern.search(text):
            raise ResultContractError(
                f"Leak category {category} at {location} #{ordinal}"
            )


_STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)

_CONTROLLED_OUTPUT_ROOTS = frozenset(
    {
        "scammark_qwen3_dashscope_server_full_v2",
        "dataset_frozen_v1",
    }
)
_LEGACY_OUTPUT_NAMESPACE = "scammark_qwen3_dashscope_server_full"


def _may_follow_controlled_outputs_component(
    pending: Sequence[str],
    index: int,
) -> bool:
    """Allow only the Task11-approved logical outputs-parent indirection."""

    return (
        pending[index] == "outputs"
        and index + 1 < len(pending)
        and pending[index + 1] in _CONTROLLED_OUTPUT_ROOTS
    )


def _may_follow_pinned_proc_directory_component(
    pending: Sequence[str],
    index: int,
) -> bool:
    """Allow only an explicit in-process pinned directory descriptor path."""

    if len(pending) <= 4 or tuple(pending[:3]) != ("proc", "self", "fd"):
        return False
    if index == 1:
        return True
    return index == 3 and pending[3].isdigit()


def _verify_controlled_outputs_directory_fd(descriptor: int) -> None:
    """Reject a controlled outputs link whose pinned target enters v1 storage."""

    try:
        target = os.readlink(f"/proc/self/fd/{descriptor}")
    except OSError as exc:
        raise ResultContractError(
            "Controlled outputs target cannot be verified"
        ) from exc
    if _LEGACY_OUTPUT_NAMESPACE in Path(target).parts:
        raise ResultContractError("Controlled outputs target enters v1 storage")


def _verified_directory_target(
    descriptor: int,
    *,
    label: str,
) -> Path:
    try:
        raw_target = os.readlink(f"/proc/self/fd/{descriptor}")
    except OSError as exc:
        raise ResultContractError(f"{label} target cannot be verified") from exc
    if not raw_target or raw_target.endswith(" (deleted)"):
        raise ResultContractError(f"{label} target cannot be verified")
    target = Path(raw_target)
    if not target.is_absolute():
        raise ResultContractError(f"{label} target cannot be verified")
    if _LEGACY_OUTPUT_NAMESPACE in target.parts:
        raise ResultContractError(f"{label} target enters v1 storage")
    return target


def _validate_archive_directory_fd(
    descriptor: int,
    *,
    parent_metadata: os.stat_result | None,
    check_mount: bool,
) -> os.stat_result:
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise ResultContractError("Archive destination parent is unavailable") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (reparse_flag and file_attributes & reparse_flag)
    ):
        raise ResultContractError("Archive destination parent is unsafe")
    target = _verified_directory_target(
        descriptor,
        label="Archive destination parent",
    )
    if parent_metadata is not None and metadata.st_dev != parent_metadata.st_dev:
        raise ResultContractError("Archive destination crosses a device boundary")
    if check_mount and os.path.ismount(target):
        raise ResultContractError("Archive destination contains a mount point")
    return metadata


def _open_archive_parent(path: Path, *, create_missing: bool) -> int:
    """Create and pin an archive parent without following any component link."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ResultContractError("Secure archive destination access is unavailable")
    if "\x00" in os.fspath(path):
        raise ResultContractError("Archive destination parent is unsafe")
    components = path.parts
    if any(
        component in ("..", _LEGACY_OUTPUT_NAMESPACE)
        or "\\" in component
        for component in components
    ):
        raise ResultContractError("Archive destination parent is unsafe")
    descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | directory_flag
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        if path.is_absolute():
            descriptor = os.open(path.anchor, flags)
            pending = components[1:]
        else:
            descriptor = os.open(".", flags)
            pending = tuple(
                component for component in components if component not in ("", ".")
            )
        parent_metadata = _validate_archive_directory_fd(
            descriptor,
            parent_metadata=None,
            check_mount=False,
        )
        for component in pending:
            if component == _LEGACY_OUTPUT_NAMESPACE:
                raise ResultContractError("Archive destination parent enters v1 storage")
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create_missing:
                    raise ResultContractError(
                        "Archive destination parent binding is missing"
                    )
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                next_metadata = _validate_archive_directory_fd(
                    next_descriptor,
                    parent_metadata=parent_metadata,
                    check_mount=True,
                )
            except Exception:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
            parent_metadata = next_metadata
        result = descriptor
        descriptor = None
        return result
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Secure archive destination open failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_or_create_archive_parent(path: Path) -> int:
    return _open_archive_parent(path, create_missing=True)


def _open_existing_archive_parent(path: Path) -> int:
    return _open_archive_parent(path, create_missing=False)


def _open_pinned_regular_fd(
    path: Path,
    *,
    require_single_link: bool = True,
) -> tuple[int, os.stat_result]:
    """Open a regular file without following any path-component link."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ResultContractError("Secure local file opening is unavailable")
    components = path.parts
    if not components or any(part == ".." for part in components):
        raise ResultContractError("Local file path is unsafe")
    directory_descriptor: int | None = None
    file_descriptor: int | None = None
    try:
        directory_flags = (
            os.O_RDONLY
            | directory_flag
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        if path.is_absolute():
            directory_descriptor = os.open(path.anchor, directory_flags)
            pending = components[1:]
        else:
            directory_descriptor = os.open(".", directory_flags)
            pending = tuple(part for part in components if part != ".")
        if not pending:
            raise ResultContractError("Local file path does not name a file")
        for index, component in enumerate(pending[:-1]):
            component_flags = directory_flags
            controlled_outputs = _may_follow_controlled_outputs_component(
                pending,
                index,
            )
            pinned_proc_directory = _may_follow_pinned_proc_directory_component(
                pending,
                index,
            )
            if controlled_outputs or pinned_proc_directory:
                component_flags &= ~nofollow
            next_descriptor = os.open(
                component,
                component_flags,
                dir_fd=directory_descriptor,
            )
            try:
                metadata = os.fstat(next_descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ResultContractError(
                        "Local file parent is not a directory"
                    )
                if controlled_outputs:
                    _verify_controlled_outputs_directory_fd(next_descriptor)
                if pinned_proc_directory:
                    _verified_directory_target(
                        next_descriptor,
                        label="Pinned local directory",
                    )
            except Exception:
                os.close(next_descriptor)
                raise
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            pending[-1],
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ResultContractError("Local source is not a regular file")
        if require_single_link and metadata.st_nlink != 1:
            raise ResultContractError("Local source is not singly linked")
        result = file_descriptor, metadata
        file_descriptor = None
        return result
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Secure local file open failed") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _require_stable_fd(fd: int, before: os.stat_result) -> None:
    try:
        after = os.fstat(fd)
    except OSError as exc:
        raise ResultContractError("Local source stability check failed") from exc
    if any(
        getattr(before, field) != getattr(after, field)
        for field in _STABLE_STAT_FIELDS
    ):
        raise ResultContractError("Local source changed while it was read")


def _read_pinned_regular_bytes(path: Path) -> bytes:
    fd, before = _open_pinned_regular_fd(path)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        _require_stable_fd(fd, before)
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Secure local file read failed") from exc
    finally:
        os.close(fd)
    return b"".join(chunks)


def _read_pinned_regular_text(path: Path, *, label: str) -> str:
    try:
        return _read_pinned_regular_bytes(path).decode("utf-8")
    except ResultContractError:
        raise
    except UnicodeError as exc:
        raise ResultContractError(f"Required {label} file is unreadable") from exc


@dataclass(frozen=True)
class FullContractValidation:
    """Summary returned after a full result directory has been validated."""

    mode: ResultMode
    sample_count: int
    sample_ids: frozenset[str]


def _reject_nonstandard_constant(value: str) -> None:
    raise _StrictJSONError("non-standard numeric constant")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError("duplicate object key")
        result[key] = value
    return result


def _loads_strict_json(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (RecursionError, ValueError) as exc:
        raise ResultContractError(f"{label} is not strict JSON") from exc


def _load_strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    text = _read_pinned_regular_text(path, label=label)
    payload = _loads_strict_json(text, label=label)
    if type(payload) is not dict:
        raise ResultContractError(f"{label} must contain a JSON object")
    return payload


def _load_jsonl_sample_ids(path: Path, *, label: str) -> frozenset[str]:
    sample_ids: set[str] = set()
    try:
        text = _read_pinned_regular_text(path, label=label)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ResultContractError(
                    f"{label} contains a blank JSONL row at line {line_number}"
                )
            row = _loads_strict_json(
                line,
                label=f"{label} line {line_number}",
            )
            if type(row) is not dict:
                raise ResultContractError(
                    f"{label} line {line_number} must contain a JSON object"
                )
            sample_id = row.get("sample_id")
            if type(sample_id) is not str or not sample_id.strip():
                raise ResultContractError(
                    f"{label} line {line_number} has an invalid sample_id"
                )
            if sample_id in sample_ids:
                raise ResultContractError(
                    f"{label} contains a duplicate sample_id"
                )
            sample_ids.add(sample_id)
    except ResultContractError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ResultContractError(f"Required {label} file is unreadable") from exc

    return frozenset(sample_ids)


def _require_object(
    container: dict[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    value = container.get(key)
    if type(value) is not dict:
        raise ResultContractError(f"{label}.{key} must be a JSON object")
    return value


def _require_exact_integer(
    container: dict[str, Any],
    key: str,
    expected: int,
    *,
    label: str,
) -> None:
    value = container.get(key)
    if type(value) is not int or value != expected:
        raise ResultContractError(
            f"{label}.{key} must be the exact expected integer"
        )


def validate_full_contract(
    full_dir: str | Path,
    expected_count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
    mode: ResultMode = "inference",
) -> FullContractValidation:
    """Validate a full inference or final result directory.

    All files are parsed locally. Any missing, malformed, ambiguous, or
    type-inexact field raises :class:`ResultContractError`.
    """

    if type(mode) is not str or mode not in ("inference", "final"):
        raise ResultContractError("mode must be 'inference' or 'final'")
    if type(expected_count) is not int or expected_count < 0:
        raise ResultContractError("expected_count must be a nonnegative integer")

    directory = Path(full_dir)

    manifest_ids = _load_jsonl_sample_ids(
        directory / "manifest.jsonl",
        label="manifest.jsonl",
    )
    prediction_ids = _load_jsonl_sample_ids(
        directory / "predictions.jsonl",
        label="predictions.jsonl",
    )
    if len(manifest_ids) != expected_count:
        raise ResultContractError("Manifest sample count is not exact")
    if len(prediction_ids) != expected_count:
        raise ResultContractError("Prediction sample count is not exact")
    if manifest_ids != prediction_ids:
        raise ResultContractError("Manifest and prediction sample IDs do not match")

    metrics = _load_strict_json_object(
        directory / "metrics.json",
        label="metrics.json",
    )
    engineering = _require_object(metrics, "engineering", label="metrics.json")
    _require_exact_integer(
        engineering,
        "sample_count",
        expected_count,
        label="metrics.json.engineering",
    )
    _require_exact_integer(
        engineering,
        "prediction_sample_count",
        expected_count,
        label="metrics.json.engineering",
    )

    gates = _require_object(
        engineering,
        "acceptance_gates",
        label="metrics.json.engineering",
    )
    if gates.get("passed") is not True:
        raise ResultContractError("metrics.json acceptance_gates.passed must be true")
    checks = _require_object(
        gates,
        "checks",
        label="metrics.json.engineering.acceptance_gates",
    )
    if checks.get("full_sample_count_exact") is not True:
        raise ResultContractError(
            "metrics.json full_sample_count_exact check must be true"
        )

    if mode == "final":
        final_metrics = _load_strict_json_object(
            directory / "metrics_final.json",
            label="metrics_final.json",
        )
        final_engineering = _require_object(
            final_metrics,
            "engineering",
            label="metrics_final.json",
        )
        _require_exact_integer(
            final_engineering,
            "prediction_sample_count",
            FINAL_EXPECTED_PREDICTION_COUNT,
            label="metrics_final.json.engineering",
        )
        rationale = _require_object(
            final_metrics,
            "rationale",
            label="metrics_final.json",
        )
        end_to_end = _require_object(
            rationale,
            "end_to_end",
            label="metrics_final.json.rationale",
        )
        if end_to_end.get("bertscore_status") != "computed":
            raise ResultContractError(
                "metrics_final.json BERTScore status must be computed"
            )

    return FullContractValidation(
        mode=mode,
        sample_count=expected_count,
        sample_ids=manifest_ids,
    )


def _validate_mode(mode: object) -> ResultMode:
    if type(mode) is not str or mode not in ("inference", "final"):
        raise ResultContractError("mode must be 'inference' or 'final'")
    return mode


def _load_strict_jsonl_objects(
    path: Path,
    *,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = _read_pinned_regular_text(path, label=label)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ResultContractError(
                    f"{label} contains a blank JSONL row at line {line_number}"
                )
            row = _loads_strict_json(
                line,
                label=f"{label} line {line_number}",
            )
            if type(row) is not dict:
                raise ResultContractError(
                    f"{label} line {line_number} must contain a JSON object"
                )
            rows.append(row)
    except ResultContractError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ResultContractError(f"Required {label} file is unreadable") from exc
    return rows


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise ResultContractError("Unable to synchronize result directory") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ResultContractError("Unable to synchronize result directory") from exc
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    except (OSError, UnicodeError) as exc:
        raise ResultContractError("Unable to atomically write result metadata") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _strict_json_text(payload: object) -> str:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ResultContractError("Sanitized audit data is not valid JSON") from exc


def _persist_audit_projection(
    audit_dir: str | Path,
    projected_run: dict[str, Any],
    projected_ledger: list[dict[str, Any]],
) -> Path:
    run_text = _strict_json_text(projected_run)
    ledger_text = "".join(_strict_json_text(row) for row in projected_ledger)
    for ordinal, projection_text in enumerate((run_text, ledger_text), start=1):
        _reject_generic_leak_markers(
            projection_text,
            scope="audit",
            ordinal=ordinal,
        )
    destination = Path(audit_dir)
    if destination.is_symlink():
        raise ResultContractError("Audit destination must not be a symbolic link")
    try:
        destination.mkdir(parents=True, exist_ok=True)
        permitted_names = {
            "run_config.sanitized.json",
            "api_call_ledger.sanitized.jsonl",
        }
        if any(path.name not in permitted_names for path in destination.iterdir()):
            raise ResultContractError("Audit destination contains an unexpected entry")
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Audit destination is unavailable") from exc
    _atomic_write_text(destination / "run_config.sanitized.json", run_text)
    _atomic_write_text(
        destination / "api_call_ledger.sanitized.jsonl",
        ledger_text,
    )
    return destination


def _flat_digest_mapping(value: object, *, required: bool) -> dict[str, str] | None:
    if value is None and not required:
        return None
    if type(value) is not dict or not value:
        raise ResultContractError("Flat audit digest mapping is invalid")
    projected: dict[str, str] = {}
    for key, digest in value.items():
        if (
            type(key) is not str
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", key) is None
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{3,64}", digest) is None
        ):
            raise ResultContractError("Flat audit digest mapping is invalid")
        projected[key] = digest
    return projected


def _project_flat_audit_state(
    persisted: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = persisted.get("schema")
    if (
        type(schema) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}", schema) is None
    ):
        raise ResultContractError("Flat run config schema is invalid")
    run_mode = persisted.get("run_mode")
    if run_mode not in ("pilot", "full"):
        raise ResultContractError("Flat run config mode is invalid")
    implementation = _flat_digest_mapping(
        persisted.get("implementation_sha256"),
        required=True,
    )
    source_sha256 = _flat_digest_mapping(
        persisted.get("source_sha256"),
        required=False,
    )
    sample_digest = persisted.get("contract_sample_ids_sha256")
    if sample_digest is not None and (
        type(sample_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", sample_digest) is None
    ):
        raise ResultContractError("Flat run config sample digest is invalid")
    image_count = persisted.get("selected_image_sha256_count")
    if image_count is not None and (
        type(image_count) is not int or image_count < 0
    ):
        raise ResultContractError("Flat run config image count is invalid")
    taxonomy = persisted.get("taxonomy")
    if taxonomy is not None:
        from .taxonomy import CANONICAL_LABELS

        if type(taxonomy) is not list or taxonomy != list(CANONICAL_LABELS):
            raise ResultContractError("Flat run config taxonomy is invalid")
    projected_run = {
        "schema": schema,
        "run_mode": run_mode,
        "source_sha256": source_sha256,
        "implementation_sha256": implementation,
        "contract_sample_ids_sha256": sample_digest,
        "selected_image_sha256_count": image_count,
        "taxonomy": taxonomy,
    }

    projected_ledger: list[dict[str, Any]] = []
    for row in ledger:
        event = row.get("event")
        if event not in ("reserved", "finished"):
            raise ResultContractError("Flat audit ledger event is invalid")
        call_id = row.get("call_id")
        sample_id = row.get("sample_id")
        stage = row.get("stage")
        if (
            type(call_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_.:]{1,128}", call_id) is None
            or type(sample_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", sample_id) is None
            or stage not in _PRODUCTION_STAGES
        ):
            raise ResultContractError("Flat audit ledger identity is invalid")
        timestamp = row.get("timestamp")
        if timestamp is not None and (
            type(timestamp) is not str
            or re.fullmatch(r"[0-9TZ:+.-]{1,64}", timestamp) is None
        ):
            raise ResultContractError("Flat audit ledger timestamp is invalid")
        status = "reserved" if event == "reserved" else row.get("status")
        if status not in ("reserved", "completed", "transport_error"):
            raise ResultContractError("Flat audit ledger status is invalid")
        category = row.get("error_category")
        if category is not None and (
            type(category) is not str or category not in _SAFE_ERROR_CATEGORIES
        ):
            raise ResultContractError("Flat audit ledger error category is invalid")
        projected_ledger.append(
            {
                "timestamp": timestamp,
                "event": event,
                "call_id": call_id,
                "sample_id": sample_id,
                "stage": stage,
                "status": status,
                "error_category": category,
                "sanitized_error_message": (
                    SANITIZED_ERROR_MESSAGE
                    if status == "transport_error"
                    else None
                ),
            }
        )
    return projected_run, projected_ledger


def write_audit_projection(
    state_dir: str | Path,
    audit_dir: str | Path,
    *,
    require_runtime_state: bool = False,
) -> Path:
    """Write a strict, allowlisted projection of raw runtime audit state."""

    if type(require_runtime_state) is not bool:
        raise ResultContractError("require_runtime_state must be a boolean")
    source = Path(state_dir)
    run_config_path = source / "run_config.json"
    persisted = _load_strict_json_object(
        run_config_path,
        label="run_config.json",
    )
    if "config_hash" not in persisted or "config" not in persisted:
        if require_runtime_state:
            raise ResultContractError("Packaging requires nested runtime state")
        flat_ledger = _load_strict_jsonl_objects(
            source / "api_call_ledger.jsonl",
            label="api_call_ledger.jsonl",
        )
        projected_run, projected_ledger = _project_flat_audit_state(
            persisted,
            flat_ledger,
        )
        return _persist_audit_projection(
            audit_dir,
            projected_run,
            projected_ledger,
        )
    config_hash = persisted.get("config_hash")
    if type(config_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", config_hash) is None:
        raise ResultContractError("run_config.json config_hash is invalid")
    run_config = persisted.get("config")
    if type(run_config) is not dict:
        raise ResultContractError("run_config.json config must be a JSON object")

    run_mode = run_config.get("run_mode")
    if type(run_mode) is not str or run_mode not in ("pilot", "full"):
        raise ResultContractError("run_config.json run_mode is invalid")

    def require_digest_mapping(field: str) -> dict[str, str]:
        value = run_config.get(field)
        if type(value) is not dict or not value:
            raise ResultContractError(f"run_config.json {field} must be an object")
        projected: dict[str, str] = {}
        for key, digest in value.items():
            if field == "source_sha256":
                safe_key = key in {
                    "manifest",
                    "web_evidence",
                    "dataset_version",
                    "config",
                }
            else:
                safe_key = (
                    type(key) is str
                    and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}\.py", key) is not None
                )
            if (
                not safe_key
                or type(digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ResultContractError(f"run_config.json {field} is invalid")
            projected[key] = digest
        return projected

    source_sha256 = require_digest_mapping("source_sha256")
    implementation_sha256 = require_digest_mapping("implementation_sha256")
    if set(source_sha256) != {
        "manifest",
        "web_evidence",
        "dataset_version",
        "config",
    }:
        raise ResultContractError("run_config.json source_sha256 fields are not exact")

    contract_sample_ids = run_config.get("contract_sample_ids")
    if type(contract_sample_ids) is not list or not contract_sample_ids or any(
        type(sample_id) is not str
        or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", sample_id)
        for sample_id in contract_sample_ids
    ):
        raise ResultContractError("run_config.json contract_sample_ids is invalid")
    if len(contract_sample_ids) != len(set(contract_sample_ids)):
        raise ResultContractError("run_config.json contract_sample_ids is not unique")
    contract_id_set = set(contract_sample_ids)
    canonical_ids = json.dumps(
        contract_sample_ids,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    contract_sample_ids_sha256 = hashlib.sha256(canonical_ids).hexdigest()

    selected_image_sha256 = run_config.get("selected_image_sha256")
    if type(selected_image_sha256) is not dict or not selected_image_sha256:
        raise ResultContractError(
            "run_config.json selected_image_sha256 must be an object"
        )
    for key, digest in selected_image_sha256.items():
        if (
            type(key) is not str
            or key not in contract_id_set
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ResultContractError("run_config.json selected_image_sha256 is invalid")
    if set(selected_image_sha256) != contract_id_set:
        raise ResultContractError(
            "run_config.json selected_image_sha256 keys do not match contract IDs"
        )

    taxonomy = run_config.get("taxonomy")
    try:
        from .taxonomy import CANONICAL_LABELS
    except (ImportError, AttributeError) as exc:
        raise ResultContractError("Canonical taxonomy is unavailable") from exc
    if type(taxonomy) is not list or taxonomy != list(CANONICAL_LABELS):
        raise ResultContractError("run_config.json taxonomy is not canonical")

    canonical_config = json.dumps(
        run_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical_config).hexdigest() != config_hash.lower():
        raise ResultContractError("run_config.json config_hash does not match config")

    ledger = _load_strict_jsonl_objects(
        source / "api_call_ledger.jsonl",
        label="api_call_ledger.jsonl",
    )

    projected_run: dict[str, Any] = {
        "schema": SANITIZED_RUN_CONFIG_SCHEMA,
        "run_mode": run_mode,
        "source_sha256": source_sha256,
        "implementation_sha256": implementation_sha256,
        "contract_sample_ids_sha256": contract_sample_ids_sha256,
        "selected_image_sha256_count": len(selected_image_sha256),
        "taxonomy": taxonomy,
    }
    projected_ledger: list[dict[str, Any]] = []
    reservations: dict[str, list[tuple[str, str]]] = {}
    for row in ledger:
        if row.get("config_hash") != config_hash:
            raise ResultContractError("api_call_ledger.jsonl config_hash mismatch")
        event = row.get("event")
        if event not in ("reserved", "finished"):
            raise ResultContractError("api_call_ledger.jsonl event is invalid")
        call_id = row.get("call_id")
        if type(call_id) is not str or re.fullmatch(r"[0-9a-f]{32}", call_id) is None:
            raise ResultContractError("api_call_ledger.jsonl call_id is invalid")
        unix_time = row.get("unix_time")
        if type(unix_time) not in (int, float):
            raise ResultContractError("api_call_ledger.jsonl unix_time is invalid")
        try:
            unix_time_is_finite = math.isfinite(unix_time)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ResultContractError(
                "api_call_ledger.jsonl unix_time is invalid"
            ) from exc
        if not unix_time_is_finite or unix_time < 0:
            raise ResultContractError("api_call_ledger.jsonl unix_time is invalid")
        try:
            timestamp = datetime.fromtimestamp(
                unix_time,
                tz=timezone.utc,
            ).isoformat().replace("+00:00", "Z")
        except (OSError, OverflowError, ValueError) as exc:
            raise ResultContractError(
                "api_call_ledger.jsonl unix_time is invalid"
            ) from exc

        if event == "reserved":
            context = row.get("context")
            if type(context) is not dict:
                raise ResultContractError(
                    "api_call_ledger.jsonl reservation context is invalid"
                )
            sample_id = context.get("sample_id")
            stage = context.get("stage")
            if (
                type(sample_id) is not str
                or re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", sample_id) is None
                or sample_id not in contract_id_set
                or type(stage) is not str
                or stage not in _PRODUCTION_STAGES
            ):
                raise ResultContractError(
                    "api_call_ledger.jsonl reservation identity is invalid"
                )
            if call_id in reservations:
                raise ResultContractError(
                    "api_call_ledger.jsonl reservation call_id is duplicated"
                )
            reservations[call_id] = [(sample_id, stage)]
            status = "reserved"
            error_category: str | None = None
            sanitized_error_message: str | None = None
        else:
            pending = reservations.get(call_id)
            if not pending:
                raise ResultContractError(
                    "api_call_ledger.jsonl finish has no matching reservation"
                )
            sample_id, stage = pending.pop(0)
            status = row.get("status")
            if status not in ("completed", "transport_error"):
                raise ResultContractError("api_call_ledger.jsonl status is invalid")
            metadata = row.get("metadata")
            if type(metadata) is not dict:
                raise ResultContractError("api_call_ledger.jsonl metadata is invalid")
            error_category_value = metadata.get("error_category")
            if error_category_value is None:
                error_category = None
            elif (
                type(error_category_value) is str
                and error_category_value in _SAFE_ERROR_CATEGORIES
            ):
                error_category = error_category_value
            else:
                raise ResultContractError(
                    "api_call_ledger.jsonl error_category is invalid"
                )
            sanitized_error_message = (
                SANITIZED_ERROR_MESSAGE if status == "transport_error" else None
            )
        projected_ledger.append(
            {
                "timestamp": timestamp,
                "event": event,
                "call_id": call_id,
                "sample_id": sample_id,
                "stage": stage,
                "status": status,
                "error_category": error_category,
                "sanitized_error_message": sanitized_error_message,
            }
        )
    if any(pending for pending in reservations.values()):
        raise ResultContractError(
            "api_call_ledger.jsonl contains an unmatched reservation"
        )
    return _persist_audit_projection(
        audit_dir,
        projected_run,
        projected_ledger,
    )


def _validated_secret_values(secret_values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(secret_values, (str, bytes)):
        raise ResultContractError("secret_values must be a sequence of strings")
    try:
        values = tuple(secret_values)
    except TypeError as exc:
        raise ResultContractError("secret_values must be a sequence of strings") from exc
    if any(type(value) is not str for value in values):
        raise ResultContractError("secret_values must contain only strings")
    return tuple(value for value in values if value)


def _scan_candidates(root: Path) -> list[Path]:
    if root.is_symlink():
        raise ResultContractError("Leak scan rejected a symbolic-link root")
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise ResultContractError("Leak scan root is missing")
    try:
        entries = sorted(root.rglob("*"), key=lambda path: path.as_posix())
    except OSError as exc:
        raise ResultContractError("Leak scan could not enumerate staged files") from exc
    candidates: list[Path] = []
    for entry in entries:
        if entry.is_symlink():
            raise ResultContractError("Leak scan rejected a symbolic-link entry")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ResultContractError("Leak scan rejected a non-regular entry")
        candidates.append(entry)
    return candidates


def scan_for_leaks(
    root: str | Path,
    *,
    secret_values: Iterable[str] = (),
    relative_paths: Iterable[str] | None = None,
) -> dict[str, str]:
    """Reject leaks and return SHA256 digests of the exact bytes scanned."""

    exact_values = _validated_secret_values(secret_values)
    scan_root = Path(root)
    candidates: list[tuple[str, Path]] = []
    if relative_paths is None:
        candidates = [
            (
                path.name if scan_root.is_file() else path.relative_to(scan_root).as_posix(),
                path,
            )
            for path in _scan_candidates(scan_root)
        ]
    else:
        if isinstance(relative_paths, (str, bytes)):
            raise ResultContractError("Leak scan paths must be a sequence")
        try:
            supplied_paths = tuple(relative_paths)
        except TypeError as exc:
            raise ResultContractError("Leak scan paths must be a sequence") from exc
        for relative in supplied_paths:
            if type(relative) is not str or not relative:
                raise ResultContractError("Leak scan relative path is invalid")
            if "\\" in relative or relative.startswith("/") or "\x00" in relative:
                raise ResultContractError("Leak scan relative path is unsafe")
            parts = relative.split("/")
            if any(part in ("", ".", "..") for part in parts):
                raise ResultContractError("Leak scan relative path is unsafe")
            candidates.append((relative, scan_root.joinpath(*parts)))
    digests: dict[str, str] = {}
    for index, (relative, path) in enumerate(candidates, start=1):
        location = f"staged file #{index}"
        try:
            content = _read_pinned_regular_bytes(path)
            text = content.decode("utf-8")
        except ResultContractError:
            raise
        except UnicodeError as exc:
            raise ResultContractError(
                f"Leak scan could not read {location}"
            ) from exc
        if any(value in text for value in exact_values):
            raise ResultContractError(f"Leak category exact-value at {location}")
        _reject_generic_leak_markers(
            text,
            scope="staged",
            ordinal=index,
        )
        if relative in digests:
            raise ResultContractError("Leak scan contains a duplicate path")
        digests[relative] = hashlib.sha256(content).hexdigest()
    return digests


def _archive_payload(mode: object) -> tuple[str, ...]:
    validated_mode = _validate_mode(mode)
    if validated_mode == "inference":
        return INFERENCE_ARCHIVE_PAYLOAD
    return FINAL_ARCHIVE_PAYLOAD


def _canonical_member_parts(name: object) -> tuple[str, ...]:
    if type(name) is not str or not name:
        raise ResultContractError("Archive member name is invalid")
    if "\\" in name or "\x00" in name or name.startswith("/"):
        raise ResultContractError("Archive member path is unsafe")
    if re.match(r"^[A-Za-z]:/", name):
        raise ResultContractError("Archive member path is unsafe")
    parts = tuple(name.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise ResultContractError("Archive member path is not canonical")
    if "/".join(parts) != name:
        raise ResultContractError("Archive member path is not canonical")
    return parts


def _resolved_link_name(
    *,
    member_name: str,
    link_name: object,
    release_root: str,
) -> str:
    if type(link_name) is not str or not link_name:
        raise ResultContractError("Archive link target is invalid")
    if "\\" in link_name or "\x00" in link_name or link_name.startswith("/"):
        raise ResultContractError("Archive link target is unsafe")
    if re.match(r"^[A-Za-z]:/", link_name):
        raise ResultContractError("Archive link target is unsafe")

    target_parts = link_name.split("/")
    if any(part in ("", ".", "..") for part in target_parts):
        raise ResultContractError("Archive link target is not canonical")
    if target_parts[0] == release_root:
        resolved: list[str] = []
    else:
        resolved = member_name.split("/")[:-1]
    for part in target_parts:
        resolved.append(part)
    if not resolved or resolved[0] != release_root:
        raise ResultContractError("Archive link target escapes release root")
    return "/".join(resolved)


def validate_tar_members(
    members: Sequence[tarfile.TarInfo | str] | Iterable[tarfile.TarInfo | str],
    *,
    mode: ResultMode,
) -> str:
    """Validate exact member names, one root, file types, and link targets."""

    payload = _archive_payload(mode)
    if isinstance(members, (str, bytes)):
        raise ResultContractError("Archive members must be a sequence")
    try:
        materialized = list(members)
    except TypeError as exc:
        raise ResultContractError("Archive members must be a sequence") from exc
    if not materialized:
        raise ResultContractError("Archive contains no members")

    names: list[str] = []
    tar_members: list[tarfile.TarInfo | None] = []
    roots: set[str] = set()
    for member in materialized:
        if isinstance(member, tarfile.TarInfo):
            name = member.name
            tar_member: tarfile.TarInfo | None = member
        elif type(member) is str:
            name = member
            tar_member = None
        else:
            raise ResultContractError("Archive member metadata is invalid")
        parts = _canonical_member_parts(name)
        if len(parts) < 2:
            raise ResultContractError("Archive member must be below a release root")
        roots.add(parts[0])
        names.append(name)
        tar_members.append(tar_member)

    if len(names) != len(set(names)):
        raise ResultContractError("Archive contains duplicate member names")
    if len(roots) != 1:
        raise ResultContractError("Archive must contain exactly one release root")
    release_root = next(iter(roots))
    expected_root = (
        INFERENCE_ARCHIVE_ROOT if mode == "inference" else FINAL_ARCHIVE_ROOT
    )
    if release_root != expected_root:
        raise ResultContractError("Archive release root is not mode-exact")
    expected_names = {f"{release_root}/{relative}" for relative in payload}
    if set(names) != expected_names:
        raise ResultContractError("Archive members do not match the exact allowlist")

    known_names = set(names)
    for name, member in zip(names, tar_members):
        if member is None or member.isreg():
            continue
        if member.issym() or member.islnk():
            resolved = _resolved_link_name(
                member_name=name,
                link_name=member.linkname,
                release_root=release_root,
            )
            if resolved not in known_names:
                raise ResultContractError(
                    "Archive link target is outside the exact member allowlist"
                )
            continue
        raise ResultContractError("Archive contains a prohibited special member")
    return release_root


def _copy_regular_source(source: Path, destination: Path) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ResultContractError("Secure source-file opening is unavailable")
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    destination_created = False
    copy_succeeded = False
    try:
        source_descriptor, before = _open_pinned_regular_fd(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | nofollow
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        destination_created = True
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
        os.fsync(destination_descriptor)
        _require_stable_fd(source_descriptor, before)
        copy_succeeded = True
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Unable to stage a required archive file") from exc
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_created and not copy_succeeded:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor: int | None = None
    try:
        descriptor, before = _open_pinned_regular_fd(path)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        _require_stable_fd(descriptor, before)
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Unable to hash result archive") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return digest.hexdigest()


def _open_pinned_regular_at(
    directory_descriptor: int,
    filename: str,
) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if (
        nofollow is None
        or not filename
        or filename in (".", "..")
        or "/" in filename
        or "\\" in filename
    ):
        raise ResultContractError("Archive filename is unsafe")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ResultContractError("Archive is not a singly linked regular file")
        result = descriptor, metadata
        descriptor = None
        return result
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Secure archive open failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sha256_file_at(directory_descriptor: int, filename: str) -> str:
    descriptor, before = _open_pinned_regular_at(directory_descriptor, filename)
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        _require_stable_fd(descriptor, before)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _require_exact_regular_bytes_at(
    directory_descriptor: int,
    filename: str,
    expected: bytes,
) -> None:
    """Require one small regular file to match exact bytes via a pinned fd."""

    descriptor, before = _open_pinned_regular_at(
        directory_descriptor,
        filename,
    )
    try:
        if before.st_size != len(expected):
            raise ResultContractError("Archive checksum content is not exact")
        chunks: list[bytes] = []
        remaining = len(expected) + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if b"".join(chunks) != expected:
            raise ResultContractError("Archive checksum content is not exact")
        _require_stable_fd(descriptor, before)
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Archive checksum is unreadable") from exc
    finally:
        os.close(descriptor)


def verify_result_archive_checksum(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> str:
    """Reopen one exact v2 archive safely and bind it to its exact checksum."""

    if (
        type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ResultContractError("Expected archive checksum is invalid")
    try:
        archive = Path(path)
    except (TypeError, ValueError) as exc:
        raise ResultContractError("Archive verification path is invalid") from exc
    permitted_names = {
        f"{INFERENCE_ARCHIVE_ROOT}.tar.gz",
        f"{FINAL_ARCHIVE_ROOT}.tar.gz",
    }
    if archive.name not in permitted_names:
        raise ResultContractError("Archive verification filename is invalid")

    parent_descriptor = _open_existing_archive_parent(archive.parent)
    try:
        if _sha256_file_at(parent_descriptor, archive.name) != expected_sha256:
            raise ResultContractError("Archive checksum does not match expected bytes")
        checksum_name = f"{archive.name}.sha256"
        _require_exact_regular_bytes_at(
            parent_descriptor,
            checksum_name,
            f"{expected_sha256}  {archive.name}\n".encode("ascii"),
        )
        if _sha256_file_at(parent_descriptor, archive.name) != expected_sha256:
            raise ResultContractError("Archive changed during checksum verification")
    finally:
        os.close(parent_descriptor)
    return expected_sha256


def _sha256_archive_destination(path: Path) -> str:
    parent_descriptor = _open_existing_archive_parent(path.parent)
    try:
        return _sha256_file_at(parent_descriptor, path.name)
    finally:
        os.close(parent_descriptor)


def _atomic_write_text_at(
    directory_descriptor: int,
    filename: str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    try:
        content = text.encode(encoding)
    except UnicodeError as exc:
        raise ResultContractError("Unable to encode result metadata") from exc
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ResultContractError("Secure result metadata write is unavailable")
    temporary_name = f".{filename}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    created = False
    replaced = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | nofollow
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        created = True
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short metadata write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        replaced = True
        os.fsync(directory_descriptor)
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Unable to atomically write result metadata") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created and not replaced:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except OSError:
                pass


def _read_pinned_regular_bytes_at(
    directory_descriptor: int,
    filename: str,
) -> bytes:
    descriptor, before = _open_pinned_regular_at(directory_descriptor, filename)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        _require_stable_fd(descriptor, before)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _move_existing_archive_to_recovery_backup(
    directory_descriptor: int,
    archive_name: str,
) -> str:
    """Move one verified archive inode to a collision-safe stable sibling."""

    source_descriptor, source_metadata = _open_pinned_regular_at(
        directory_descriptor,
        archive_name,
    )
    source_digest = _sha256_pinned_descriptor(source_descriptor)
    backup_name: str | None = None
    linked = False
    source_removed = False
    try:
        for _ in range(16):
            candidate = (
                f".{archive_name}.recovery-{uuid.uuid4().hex}.bak"
            )
            try:
                os.link(
                    archive_name,
                    candidate,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                continue
            backup_name = candidate
            linked = True
            break
        if backup_name is None:
            raise ResultContractError("Archive recovery sibling is unavailable")

        linked_source = os.fstat(source_descriptor)
        linked_backup = os.stat(
            backup_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        expected_identity = (
            source_metadata.st_dev,
            source_metadata.st_ino,
            source_metadata.st_mode,
            source_metadata.st_size,
        )
        if (
            (
                linked_source.st_dev,
                linked_source.st_ino,
                linked_source.st_mode,
                linked_source.st_size,
            )
            != expected_identity
            or (
                linked_backup.st_dev,
                linked_backup.st_ino,
                linked_backup.st_mode,
                linked_backup.st_size,
            )
            != expected_identity
            or linked_source.st_nlink != 2
            or linked_backup.st_nlink != 2
            or _sha256_pinned_descriptor(source_descriptor) != source_digest
        ):
            raise ResultContractError("Archive recovery sibling binding failed")

        os.unlink(archive_name, dir_fd=directory_descriptor)
        source_removed = True
        backup_descriptor, backup_metadata = _open_pinned_regular_at(
            directory_descriptor,
            backup_name,
        )
        try:
            if (
                (
                    backup_metadata.st_dev,
                    backup_metadata.st_ino,
                    backup_metadata.st_mode,
                    backup_metadata.st_size,
                )
                != expected_identity
                or _sha256_pinned_descriptor(backup_descriptor) != source_digest
            ):
                raise ResultContractError(
                    "Archive recovery sibling changed during move"
                )
            os.fchmod(backup_descriptor, 0o600)
            os.fsync(backup_descriptor)
            private_metadata = os.fstat(backup_descriptor)
            if (
                not stat.S_ISREG(private_metadata.st_mode)
                or private_metadata.st_nlink != 1
                or stat.S_IMODE(private_metadata.st_mode) != 0o600
                or (
                    private_metadata.st_dev,
                    private_metadata.st_ino,
                    private_metadata.st_size,
                )
                != (
                    source_metadata.st_dev,
                    source_metadata.st_ino,
                    source_metadata.st_size,
                )
                or _sha256_pinned_descriptor(backup_descriptor) != source_digest
            ):
                raise ResultContractError(
                    "Archive recovery sibling privacy binding failed"
                )
        finally:
            os.close(backup_descriptor)
        os.fsync(directory_descriptor)
        return backup_name
    except ArchiveRecoveryRequired:
        raise
    except Exception as exc:
        if linked and not source_removed and backup_name is not None:
            try:
                os.unlink(backup_name, dir_fd=directory_descriptor)
            except OSError:
                pass
        if source_removed and backup_name is not None:
            raise ArchiveRecoveryRequired(backup_name, archive_name) from exc
        if isinstance(exc, ResultContractError):
            raise
        raise ResultContractError("Unable to preserve previous result archive") from exc
    finally:
        os.close(source_descriptor)


def _sha256_pinned_descriptor(descriptor: int) -> str:
    """Hash a pinned regular file without reopening its pathname."""

    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ResultContractError("Pinned archive temporary is not regular")
        original_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        _require_stable_fd(descriptor, before)
        os.lseek(descriptor, original_offset, os.SEEK_SET)
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Unable to hash pinned result archive") from exc
    return digest.hexdigest()


def _require_temporary_archive_binding(
    path: Path,
    descriptor: int,
    *,
    expected_digest: str | None = None,
) -> os.stat_result:
    """Bind an archive pathname to its pinned fd before an atomic install."""

    try:
        path_metadata = os.lstat(path)
        pinned_metadata = os.fstat(descriptor)
    except OSError as exc:
        raise ResultContractError("Archive temporary binding is unavailable") from exc
    if (
        not stat.S_ISREG(path_metadata.st_mode)
        or not stat.S_ISREG(pinned_metadata.st_mode)
        or path_metadata.st_nlink != 1
        or pinned_metadata.st_nlink != 1
        or (
            path_metadata.st_dev,
            path_metadata.st_ino,
            path_metadata.st_mode,
            path_metadata.st_nlink,
            path_metadata.st_size,
        )
        != (
            pinned_metadata.st_dev,
            pinned_metadata.st_ino,
            pinned_metadata.st_mode,
            pinned_metadata.st_nlink,
            pinned_metadata.st_size,
        )
    ):
        raise ResultContractError("Archive temporary binding changed")
    if (
        expected_digest is not None
        and _sha256_pinned_descriptor(descriptor) != expected_digest
    ):
        raise ResultContractError("Pinned archive bytes changed before install")
    return pinned_metadata


def _require_installed_archive_binding(
    directory_descriptor: int,
    filename: str,
    descriptor: int,
    *,
    pinned_metadata: os.stat_result,
    expected_digest: str,
) -> None:
    """Verify the installed name still identifies the validated pinned bytes."""

    installed_descriptor: int | None = None
    try:
        current_pinned = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current_pinned.st_mode)
            or current_pinned.st_nlink != 1
            or (
                current_pinned.st_dev,
                current_pinned.st_ino,
                current_pinned.st_mode,
                current_pinned.st_size,
            )
            != (
                pinned_metadata.st_dev,
                pinned_metadata.st_ino,
                pinned_metadata.st_mode,
                pinned_metadata.st_size,
            )
            or _sha256_pinned_descriptor(descriptor) != expected_digest
        ):
            raise ResultContractError("Pinned archive changed during install")
        installed_descriptor, installed_metadata = _open_pinned_regular_at(
            directory_descriptor,
            filename,
        )
        if (
            installed_metadata.st_dev,
            installed_metadata.st_ino,
            installed_metadata.st_mode,
            installed_metadata.st_nlink,
            installed_metadata.st_size,
        ) != (
            current_pinned.st_dev,
            current_pinned.st_ino,
            current_pinned.st_mode,
            current_pinned.st_nlink,
            current_pinned.st_size,
        ):
            raise ResultContractError("Installed archive binding changed")
        if _sha256_pinned_descriptor(installed_descriptor) != expected_digest:
            raise ResultContractError("Installed archive bytes changed")
    finally:
        if installed_descriptor is not None:
            os.close(installed_descriptor)


def _validate_release_metadata(
    release_tag: object,
    repository: object,
    git_commit: object,
) -> tuple[str, str, str]:
    if (
        type(release_tag) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", release_tag) is None
    ):
        raise ResultContractError("Release tag metadata is invalid")
    if (
        type(repository) is not str
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/"
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}",
            repository,
        )
        is None
    ):
        raise ResultContractError("Repository metadata is invalid")
    if type(git_commit) is not str or re.fullmatch(
        r"(?:[0-9a-f]{6,40}|[0-9a-f]{64})",
        git_commit,
    ) is None:
        raise ResultContractError("Git commit metadata is invalid")
    return release_tag, repository, git_commit


def _archive_root_for_destination(mode: ResultMode, destination: Path) -> str:
    expected_root = (
        INFERENCE_ARCHIVE_ROOT if mode == "inference" else FINAL_ARCHIVE_ROOT
    )
    if destination.name != f"{expected_root}.tar.gz":
        raise ResultContractError("Archive destination filename is not mode-exact")
    return expected_root


def _normalized_tar_info(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mode = 0o644
    member.mtime = 0
    member.pax_headers = {}
    return member


def _ordered_manifest_ids(path: Path) -> tuple[str, ...]:
    rows = _load_strict_jsonl_objects(path, label="staged manifest.jsonl")
    sample_ids: list[str] = []
    for row in rows:
        sample_id = row.get("sample_id")
        if type(sample_id) is not str or not sample_id.strip():
            raise ResultContractError("Staged manifest contains an invalid sample_id")
        sample_ids.append(sample_id)
    if len(sample_ids) != len(set(sample_ids)):
        raise ResultContractError("Staged manifest contains duplicate sample IDs")
    return tuple(sample_ids)


def _validate_staged_full_against_state(
    *,
    staged_full: Path,
    audit_dir: Path,
    mode: ResultMode,
) -> tuple[tuple[str, str], ...]:
    validate_full_contract(staged_full, mode=mode)
    projected = _load_strict_json_object(
        audit_dir / "run_config.sanitized.json",
        label="sanitized run config",
    )
    if set(projected) != set(RUN_CONFIG_AUDIT_FIELDS):
        raise ResultContractError("Sanitized run config fields are not exact")
    if projected.get("schema") != SANITIZED_RUN_CONFIG_SCHEMA:
        raise ResultContractError("Sanitized run config schema is invalid")
    if projected.get("run_mode") != "full":
        raise ResultContractError("Result packaging requires full runtime state")
    selected_count = projected.get("selected_image_sha256_count")
    if type(selected_count) is not int or selected_count != DEFAULT_EXPECTED_SAMPLE_COUNT:
        raise ResultContractError("Runtime state selected image count is not exact")
    expected_digest = projected.get("contract_sample_ids_sha256")
    if (
        type(expected_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
    ):
        raise ResultContractError("Runtime state sample ID digest is invalid")
    ordered_ids = _ordered_manifest_ids(staged_full / "manifest.jsonl")
    if len(ordered_ids) != DEFAULT_EXPECTED_SAMPLE_COUNT:
        raise ResultContractError("Staged manifest count is not exact")
    actual_digest = hashlib.sha256(
        json.dumps(
            ordered_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if actual_digest != expected_digest:
        raise ResultContractError(
            "Runtime state sample ID order does not match staged manifest"
        )
    core_names = ["manifest.jsonl", "predictions.jsonl", "metrics.json"]
    if mode == "final":
        core_names.append("metrics_final.json")
    return tuple(
        (name, _sha256_file(staged_full / name)) for name in core_names
    )


def build_result_archive(
    *,
    mode: ResultMode,
    full_dir: str | Path,
    state_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    config_path: str | Path | None = None,
    frozen_dir: str | Path | None = None,
    destination: str | Path,
    release_tag: str,
    repository: str,
    git_commit: str,
    secret_values: Iterable[str] = (),
    bertscore_asset_status_path: str | Path | None = None,
    finalize_log_path: str | Path | None = None,
) -> Path:
    """Validate a pinned destination, then build and install one archive."""

    validated_mode = _validate_mode(mode)
    _validate_release_metadata(release_tag, repository, git_commit)
    try:
        archive = Path(destination)
    except (TypeError, ValueError) as exc:
        raise ResultContractError("Archive destination is invalid") from exc
    _archive_root_for_destination(validated_mode, archive)
    parent_descriptor = _open_or_create_archive_parent(archive.parent)
    try:
        result = _build_result_archive_with_parent(
            mode=mode,
            full_dir=full_dir,
            state_dir=state_dir,
            log_path=log_path,
            config_path=config_path,
            frozen_dir=frozen_dir,
            destination=archive,
            release_tag=release_tag,
            repository=repository,
            git_commit=git_commit,
            secret_values=secret_values,
            bertscore_asset_status_path=bertscore_asset_status_path,
            finalize_log_path=finalize_log_path,
            archive_parent_descriptor=parent_descriptor,
        )
        binding_descriptor = _open_existing_archive_parent(archive.parent)
        try:
            expected = os.fstat(parent_descriptor)
            actual = os.fstat(binding_descriptor)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise ResultContractError(
                    "Archive destination parent binding changed"
                )
        finally:
            os.close(binding_descriptor)
        return result
    finally:
        os.close(parent_descriptor)


def _build_result_archive_with_parent(
    *,
    mode: ResultMode,
    full_dir: str | Path,
    state_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    config_path: str | Path | None = None,
    frozen_dir: str | Path | None = None,
    destination: str | Path,
    release_tag: str,
    repository: str,
    git_commit: str,
    secret_values: Iterable[str] = (),
    bertscore_asset_status_path: str | Path | None = None,
    finalize_log_path: str | Path | None = None,
    archive_parent_descriptor: int,
) -> Path:
    """Build through an already validated and pinned destination parent."""

    validated_mode = _validate_mode(mode)
    release_tag_value, repository_value, commit_value = _validate_release_metadata(
        release_tag,
        repository,
        git_commit,
    )
    full = Path(full_dir)
    validate_full_contract(full, mode=validated_mode)
    exact_values = _validated_secret_values(secret_values)

    archive = Path(destination)
    release_root = _archive_root_for_destination(validated_mode, archive)
    archive_parent = Path(f"/proc/self/fd/{archive_parent_descriptor}")

    temporary_archive: Path | None = None
    temporary_descriptor: int | None = None
    validated_archive_digest: str | None = None
    with tempfile.TemporaryDirectory(
        prefix=".result-stage-",
        dir=archive_parent,
    ) as staging_name:
        os.chmod(staging_name, 0o700)
        staged_root = Path(staging_name) / release_root
        staged_root.mkdir()
        validation_root = Path(staging_name)
        if state_dir is None:
            raise ResultContractError("Archive runtime state input is missing")
        state = Path(state_dir)
        audit_dir = (
            staged_root / "audit"
            if validated_mode == "inference"
            else validation_root / ".validation-audit"
        )

        if validated_mode == "inference":
            if log_path is None or config_path is None or frozen_dir is None:
                raise ResultContractError("Inference archive inputs are incomplete")
            frozen = Path(frozen_dir)
            sources = {
                "full/predictions.jsonl": full / "predictions.jsonl",
                "full/metrics.json": full / "metrics.json",
                "full/manifest.jsonl": full / "manifest.jsonl",
                "full/preflight.json": full / "preflight.json",
                "full/full_summary.md": full / "full_summary.md",
                "full/human_evaluation_candidates.csv": (
                    full / "human_evaluation_candidates.csv"
                ),
                "logs/run_events.jsonl": full / "run_events.jsonl",
                "logs/full_run.log": Path(log_path),
                "config/qwen3_dashscope_server_full_v2.toml": Path(config_path),
                "frozen/dataset_version.json": frozen / "dataset_version.json",
                "frozen/verification_report.json": (
                    frozen / "verification_report.json"
                ),
            }
            for relative, source in sources.items():
                _copy_regular_source(source, staged_root / relative)
            write_audit_projection(
                state,
                audit_dir,
                require_runtime_state=True,
            )
            staged_full_for_validation = staged_root / "full"
        else:
            if bertscore_asset_status_path is None or finalize_log_path is None:
                raise ResultContractError("Final archive inputs are incomplete")
            staged_full_for_validation = validation_root / ".validation-full"
            for filename in (
                "manifest.jsonl",
                "predictions.jsonl",
                "metrics.json",
                "metrics_final.json",
            ):
                _copy_regular_source(
                    full / filename,
                    staged_full_for_validation / filename,
                )
            sources = {
                "full/metrics.json": staged_full_for_validation / "metrics.json",
                "full/metrics_final.json": (
                    staged_full_for_validation / "metrics_final.json"
                ),
                "full/final_evaluation_summary.md": (
                    full / "final_evaluation_summary.md"
                ),
                "environment/bertscore_environment.txt": (
                    full / "bertscore_environment.txt"
                ),
                "environment/bertscore_asset_status.json": Path(
                    bertscore_asset_status_path
                ),
                "logs/bertscore_finalize.log": Path(finalize_log_path),
            }
            for relative, source in sources.items():
                _copy_regular_source(source, staged_root / relative)
            write_audit_projection(
                state,
                audit_dir,
                require_runtime_state=True,
            )

        validated_core_hashes = _validate_staged_full_against_state(
            staged_full=staged_full_for_validation,
            audit_dir=audit_dir,
            mode=validated_mode,
        )
        if validated_mode == "final":
            for filename in ("metrics.json", "metrics_final.json"):
                if _sha256_file(staged_root / "full" / filename) != _sha256_file(
                    staged_full_for_validation / filename
                ):
                    raise ResultContractError(
                        "Final archive core bytes differ from staged validation bytes"
                    )

        provenance = {
            "schema": "scammark-result-provenance-v2",
            "mode": validated_mode,
            "repository": repository_value,
            "release_tag": release_tag_value,
            "git_commit": commit_value,
            "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
            "secrets_included": False,
        }
        _atomic_write_text(
            staged_root / "provenance.json",
            _strict_json_text(provenance),
        )

        payload = _archive_payload(validated_mode)
        scanned_payload_digests = scan_for_leaks(
            staged_root,
            secret_values=exact_values,
            relative_paths=payload,
        )
        if set(scanned_payload_digests) != set(payload):
            raise ResultContractError("Leak scan payload digest set is not exact")
        if _validate_staged_full_against_state(
            staged_full=staged_full_for_validation,
            audit_dir=audit_dir,
            mode=validated_mode,
        ) != validated_core_hashes:
            raise ResultContractError("Staged core result changed after validation")
        if validated_mode == "final":
            for filename in ("metrics.json", "metrics_final.json"):
                if _sha256_file(staged_root / "full" / filename) != _sha256_file(
                    staged_full_for_validation / filename
                ):
                    raise ResultContractError(
                        "Final archive core bytes changed after validation"
                    )
        archive_build_dir = validation_root / ".archive-build"
        archive_build_dir.mkdir(mode=0o700)
        os.chmod(archive_build_dir, 0o700)
        previous_archive_name: str | None = None
        archive_installed = False
        try:
            temporary_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{archive.name}.",
                suffix=".tmp",
                dir=archive_build_dir,
            )
            temporary_archive = Path(temporary_name)
            _require_temporary_archive_binding(
                temporary_archive,
                temporary_descriptor,
            )
            os.ftruncate(temporary_descriptor, 0)
            os.lseek(temporary_descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(temporary_descriptor), "wb") as raw_archive:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_archive,
                    mtime=0,
                ) as compressed_archive:
                    with tarfile.open(
                        fileobj=compressed_archive,
                        mode="w",
                        format=tarfile.PAX_FORMAT,
                    ) as handle:
                        for relative in sorted(payload):
                            handle.add(
                                staged_root / relative,
                                arcname=f"{release_root}/{relative}",
                                recursive=False,
                                filter=_normalized_tar_info,
                            )
                raw_archive.flush()
                os.fsync(raw_archive.fileno())
            os.lseek(temporary_descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(temporary_descriptor), "rb") as archive_file:
                with tarfile.open(fileobj=archive_file, mode="r:gz") as handle:
                    members = handle.getmembers()
                    validate_tar_members(members, mode=validated_mode)
                    if not all(member.isreg() for member in members):
                        raise ResultContractError(
                            "Built archive contains a non-regular member"
                        )
                    archived_payload_digests: dict[str, str] = {}
                    for relative in payload:
                        extracted_payload = handle.extractfile(
                            f"{release_root}/{relative}"
                        )
                        if extracted_payload is None:
                            raise ResultContractError(
                                "Validated archive payload member is unreadable"
                            )
                        payload_digest = hashlib.sha256()
                        with extracted_payload:
                            for chunk in iter(
                                lambda: extracted_payload.read(1024 * 1024),
                                b"",
                            ):
                                payload_digest.update(chunk)
                        archived_payload_digests[relative] = (
                            payload_digest.hexdigest()
                        )
                    if archived_payload_digests != scanned_payload_digests:
                        raise ResultContractError(
                            "Archived payload bytes differ from leak-scanned bytes"
                        )
                    archived_core_names = (
                        ("manifest.jsonl", "predictions.jsonl", "metrics.json")
                        if validated_mode == "inference"
                        else ("metrics.json", "metrics_final.json")
                    )
                    validated_digests = dict(validated_core_hashes)
                    for filename in archived_core_names:
                        if (
                            archived_payload_digests[f"full/{filename}"]
                            != validated_digests[filename]
                        ):
                            raise ResultContractError(
                                "Archived core bytes differ from validated staged bytes"
                            )
            validated_archive_digest = _sha256_pinned_descriptor(
                temporary_descriptor
            )
            pinned_metadata = _require_temporary_archive_binding(
                temporary_archive,
                temporary_descriptor,
                expected_digest=validated_archive_digest,
            )

            try:
                existing_metadata = os.stat(
                    archive.name,
                    dir_fd=archive_parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing_metadata = None
            if existing_metadata is not None:
                if (
                    not stat.S_ISREG(existing_metadata.st_mode)
                    or existing_metadata.st_nlink != 1
                ):
                    raise ResultContractError(
                        "Existing archive destination is unsafe"
                    )
                previous_archive_name = (
                    _move_existing_archive_to_recovery_backup(
                        archive_parent_descriptor,
                        archive.name,
                    )
                )

            try:
                os.replace(
                    temporary_archive,
                    archive.name,
                    dst_dir_fd=archive_parent_descriptor,
                )
                archive_installed = True
                _require_installed_archive_binding(
                    archive_parent_descriptor,
                    archive.name,
                    temporary_descriptor,
                    pinned_metadata=pinned_metadata,
                    expected_digest=validated_archive_digest,
                )
                os.fsync(archive_parent_descriptor)
            except Exception:
                if archive_installed:
                    try:
                        os.unlink(
                            archive.name,
                            dir_fd=archive_parent_descriptor,
                        )
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
                if previous_archive_name is not None:
                    try:
                        os.replace(
                            previous_archive_name,
                            archive.name,
                            src_dir_fd=archive_parent_descriptor,
                            dst_dir_fd=archive_parent_descriptor,
                        )
                        previous_archive_name = None
                        os.fsync(archive_parent_descriptor)
                    except OSError as exc:
                        raise ArchiveRecoveryRequired(
                            previous_archive_name,
                            archive.name,
                        ) from exc
                raise
            temporary_archive = None
        except ResultContractError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise ResultContractError("Unable to build validated result archive") from exc
        finally:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
                temporary_descriptor = None
            if temporary_archive is not None:
                try:
                    temporary_archive.unlink(missing_ok=True)
                except OSError:
                    pass

    try:
        if validated_archive_digest is None:
            raise ResultContractError("Validated archive digest is unavailable")
        digest = validated_archive_digest
        if _sha256_file_at(archive_parent_descriptor, archive.name) != digest:
            raise ResultContractError(
                "Installed archive checksum verification failed"
            )
        checksum_name = f"{archive.name}.sha256"
        _atomic_write_text_at(
            archive_parent_descriptor,
            checksum_name,
            f"{digest}  {archive.name}\n",
            encoding="ascii",
        )
        if _sha256_file_at(archive_parent_descriptor, archive.name) != digest:
            raise ResultContractError("Archive checksum verification failed")
        expected_checksum = f"{digest}  {archive.name}\n".encode("ascii")
        if (
            _read_pinned_regular_bytes_at(
                archive_parent_descriptor,
                checksum_name,
            )
            != expected_checksum
        ):
            raise ResultContractError("Archive checksum binding changed")
        if previous_archive_name is not None:
            os.unlink(
                previous_archive_name,
                dir_fd=archive_parent_descriptor,
            )
            previous_archive_name = None
            os.fsync(archive_parent_descriptor)
    except ArchiveRecoveryRequired:
        raise
    except Exception as exc:
        if previous_archive_name is not None:
            raise ArchiveRecoveryRequired(
                previous_archive_name,
                archive.name,
            ) from exc
        if isinstance(exc, ResultContractError):
            raise
        raise ResultContractError("Archive transaction commit failed") from exc
    return archive


@dataclass(frozen=True)
class DeliveryState:
    """Monotonic delivery progress for one independently recoverable mode."""

    mode: ResultMode = "inference"
    packaged: bool = False
    archive_sha256: str | None = None
    uploaded: bool = False
    published: bool = False
    last_exit_code: int | None = None


@dataclass(frozen=True)
class DeliveryAssetSnapshot:
    """Content digests for one privately staged, verified delivery asset."""

    mode: ResultMode
    archive_sha256: str
    checksum_sha256: str
    archive_name: str
    checksum_name: str


@dataclass(frozen=True)
class DeliveryInputSnapshot:
    """Mode selection returned after archive/runtime input binding."""

    mode: ResultMode
    selected_archive_sha256: str
    selected_checksum_sha256: str
    selected_archive_name: str
    selected_checksum_name: str
    included_modes: tuple[ResultMode, ...]


@dataclass(frozen=True)
class FinalCandidateSnapshot:
    """Digest and names for a privately verified, cross-bound final candidate."""

    archive_sha256: str
    checksum_sha256: str
    archive_name: str
    checksum_name: str


_DELIVERY_ENTRY_FIELDS = (
    "mode",
    "packaged",
    "archive_sha256",
    "uploaded",
    "published",
    "last_exit_code",
)
_DELIVERY_MODES: tuple[ResultMode, ...] = ("inference", "final")
_DELIVERY_STAGES = ("package", "upload", "publish")
_DELIVERY_EXIT_CODES = {"package": 30, "upload": 31, "publish": 32}
_DELIVERY_STATE_MAX_BYTES = 64 * 1024


def next_delivery_stages(state: DeliveryState) -> tuple[str, ...]:
    """Return only the stages that remain, without validating ad-hoc objects."""

    if not isinstance(state, DeliveryState):
        raise ResultContractError("delivery state is invalid")
    if not state.packaged:
        return ("package", "upload", "publish")
    if not state.uploaded:
        return ("upload", "publish")
    if not state.published:
        return ("publish",)
    return ()


def _valid_archive_sha256(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_delivery_entry(
    payload: object,
    *,
    expected_mode: ResultMode | None = None,
) -> DeliveryState:
    if type(payload) is not dict or set(payload) != set(_DELIVERY_ENTRY_FIELDS):
        raise ResultContractError("Delivery state fields are not exact")
    mode = payload.get("mode")
    if type(mode) is not str or mode not in _DELIVERY_MODES:
        raise ResultContractError("Delivery state mode is invalid")
    if expected_mode is not None and mode != expected_mode:
        raise ResultContractError("Delivery state mode entry does not match its key")
    packaged = payload.get("packaged")
    uploaded = payload.get("uploaded")
    published = payload.get("published")
    if any(type(value) is not bool for value in (packaged, uploaded, published)):
        raise ResultContractError("Delivery completion flags must be booleans")
    archive_sha256 = payload.get("archive_sha256")
    if packaged:
        if not _valid_archive_sha256(archive_sha256):
            raise ResultContractError("Packaged delivery state requires a valid digest")
    elif archive_sha256 is not None:
        raise ResultContractError("Unpackaged delivery state cannot retain a digest")
    if uploaded and not packaged:
        raise ResultContractError("Uploaded delivery state requires packaging")
    if published and not uploaded:
        raise ResultContractError("Published delivery state requires upload")
    last_exit_code = payload.get("last_exit_code")
    if last_exit_code is not None and (
        type(last_exit_code) is not int
        or last_exit_code not in _DELIVERY_EXIT_CODES.values()
    ):
        raise ResultContractError("Delivery exit code is invalid")
    if last_exit_code == 30 and packaged:
        raise ResultContractError("Package failure cannot mark packaging complete")
    if last_exit_code == 31 and (not packaged or uploaded):
        raise ResultContractError("Upload failure state is inconsistent")
    if last_exit_code == 32 and (not uploaded or published):
        raise ResultContractError("Publication failure state is inconsistent")
    return DeliveryState(
        mode=mode,
        packaged=packaged,
        archive_sha256=archive_sha256,
        uploaded=uploaded,
        published=published,
        last_exit_code=last_exit_code,
    )


def _default_delivery_states() -> dict[ResultMode, DeliveryState]:
    return {mode: DeliveryState(mode=mode) for mode in _DELIVERY_MODES}


def _decode_delivery_document(payload: object) -> dict[ResultMode, DeliveryState]:
    if type(payload) is not dict:
        raise ResultContractError("Delivery state document must be an object")
    if set(payload) == {"schema", "modes"}:
        if payload.get("schema") != DELIVERY_STATE_SCHEMA:
            raise ResultContractError("Delivery state schema is invalid")
        modes = payload.get("modes")
        if type(modes) is not dict or set(modes) != set(_DELIVERY_MODES):
            raise ResultContractError("Delivery state modes are not exact")
        return {
            mode: _validate_delivery_entry(modes[mode], expected_mode=mode)
            for mode in _DELIVERY_MODES
        }

    legacy = _validate_delivery_entry(payload)
    states = _default_delivery_states()
    states[legacy.mode] = legacy
    return states


def _delivery_document(states: dict[ResultMode, DeliveryState]) -> dict[str, Any]:
    if set(states) != set(_DELIVERY_MODES):
        raise ResultContractError("Delivery state modes are not exact")
    entries: dict[str, dict[str, Any]] = {}
    for mode in _DELIVERY_MODES:
        state = states[mode]
        validated = _validate_delivery_entry(asdict(state), expected_mode=mode)
        entries[mode] = asdict(validated)
    return {"schema": DELIVERY_STATE_SCHEMA, "modes": entries}


def _delivery_json_line(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ResultContractError("Delivery state cannot be serialized") from exc


def _open_secure_parent_directory(
    path: Path,
    *,
    missing_ok: bool,
) -> tuple[int, str] | None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ResultContractError("Secure delivery-state access is unavailable")
    if "\x00" in os.fspath(path):
        raise ResultContractError("Delivery state path is unsafe")
    filename = path.name
    if not filename or filename in (".", "..") or "/" in filename or "\\" in filename:
        raise ResultContractError("Delivery state path is unsafe")
    parent = path.parent
    components = parent.parts
    if any(component == ".." for component in components):
        raise ResultContractError("Delivery state path is unsafe")
    descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | directory_flag
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        if parent.is_absolute():
            descriptor = os.open(parent.anchor, flags)
            pending = components[1:]
        else:
            descriptor = os.open(".", flags)
            pending = tuple(
                component for component in components if component not in ("", ".")
            )
        for index, component in enumerate(pending):
            component_flags = flags
            controlled_outputs = _may_follow_controlled_outputs_component(
                pending,
                index,
            )
            if controlled_outputs:
                component_flags &= ~nofollow
            next_descriptor = os.open(
                component,
                component_flags,
                dir_fd=descriptor,
            )
            try:
                metadata = os.fstat(next_descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ResultContractError(
                        "Delivery state parent is not a directory"
                    )
                if controlled_outputs:
                    _verify_controlled_outputs_directory_fd(next_descriptor)
            except Exception:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        result = descriptor, filename
        descriptor = None
        return result
    except FileNotFoundError as exc:
        if missing_ok:
            return None
        raise ResultContractError("Delivery state parent is missing") from exc
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Secure delivery-state parent open failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_delivery_document_at(
    directory_descriptor: int,
    filename: str,
    *,
    missing_ok: bool,
) -> dict[ResultMode, DeliveryState] | None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ResultContractError("Secure delivery-state access is unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ResultContractError("Delivery state must be a singly linked regular file")
        if before.st_size > _DELIVERY_STATE_MAX_BYTES:
            raise ResultContractError("Delivery state file is too large")
        chunks: list[bytes] = []
        remaining = _DELIVERY_STATE_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0:
            raise ResultContractError("Delivery state file is too large")
        _require_stable_fd(descriptor, before)
        try:
            text = b"".join(chunks).decode("utf-8")
        except UnicodeError as exc:
            raise ResultContractError("Delivery state is unreadable") from exc
        payload = _loads_strict_json(text, label="delivery state")
        return _decode_delivery_document(payload)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ResultContractError("Delivery state is missing")
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Secure delivery-state read failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_delivery_document_at(
    directory_descriptor: int,
    filename: str,
    states: dict[ResultMode, DeliveryState],
) -> None:
    content = _delivery_json_line(_delivery_document(states)).encode("utf-8")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ResultContractError("Secure delivery-state access is unavailable")
    temporary_name = f".{filename}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    created = False
    replaced = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | nofollow
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        created = True
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short delivery-state write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        replaced = True
        os.fsync(directory_descriptor)
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Atomic delivery-state write failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created and not replaced:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except OSError:
                pass


def load_delivery_states(
    path: str | os.PathLike[str],
) -> dict[ResultMode, DeliveryState] | None:
    """Read both modes from one pinned descriptor, or return None if absent."""

    try:
        state_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ResultContractError("Delivery state path is invalid") from exc
    opened = _open_secure_parent_directory(state_path, missing_ok=True)
    if opened is None:
        return None
    directory_descriptor, filename = opened
    try:
        states = _read_delivery_document_at(
            directory_descriptor,
            filename,
            missing_ok=True,
        )
    finally:
        os.close(directory_descriptor)
    return states


def load_delivery_state(
    path: str | os.PathLike[str],
    mode: ResultMode = "inference",
) -> DeliveryState:
    """Read one mode without creating or modifying any delivery-state path."""

    validated_mode = _validate_mode(mode)
    states = load_delivery_states(path)
    if states is None:
        return DeliveryState(mode=validated_mode)
    return states[validated_mode]


def _create_private_snapshot_directory(path: Path) -> None:
    try:
        parent = path.parent.lstat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or stat.S_ISLNK(parent.st_mode)
            or path.exists()
            or path.is_symlink()
        ):
            raise ResultContractError("Delivery snapshot destination is unsafe")
        os.mkdir(path, 0o700)
        created = path.lstat()
        if not stat.S_ISDIR(created.st_mode) or stat.S_ISLNK(created.st_mode):
            raise ResultContractError("Delivery snapshot destination is unsafe")
        os.chmod(path, 0o700)
    except ResultContractError:
        raise
    except OSError as exc:
        raise ResultContractError("Delivery snapshot destination is unavailable") from exc


def _extract_verified_archive_members(
    *,
    archive: Path,
    mode: ResultMode,
    relative_paths: Sequence[str],
    destination: Path,
) -> None:
    release_root = (
        INFERENCE_ARCHIVE_ROOT if mode == "inference" else FINAL_ARCHIVE_ROOT
    )
    descriptor: int | None = None
    try:
        descriptor, before = _open_pinned_regular_fd(archive)
        with os.fdopen(os.dup(descriptor), "rb") as archive_file:
            with tarfile.open(fileobj=archive_file, mode="r:gz") as handle:
                members = handle.getmembers()
                validate_tar_members(members, mode=mode)
                if not all(member.isreg() for member in members):
                    raise ResultContractError(
                        "Delivery archive contains a non-regular member"
                    )
                for relative in relative_paths:
                    if relative not in _archive_payload(mode):
                        raise ResultContractError(
                            "Delivery archive extraction request is invalid"
                        )
                    source = handle.extractfile(f"{release_root}/{relative}")
                    if source is None:
                        raise ResultContractError(
                            "Delivery archive member is unreadable"
                        )
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target_descriptor = os.open(
                        target,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                    )
                    try:
                        with source:
                            while True:
                                chunk = source.read(1024 * 1024)
                                if not chunk:
                                    break
                                view = memoryview(chunk)
                                while view:
                                    written = os.write(target_descriptor, view)
                                    if written <= 0:
                                        raise OSError("short delivery snapshot write")
                                    view = view[written:]
                        os.fsync(target_descriptor)
                    finally:
                        os.close(target_descriptor)
        _require_stable_fd(descriptor, before)
    except ResultContractError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ResultContractError("Unable to inspect delivery archive") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stage_delivery_asset(
    *,
    mode: ResultMode,
    state: DeliveryState,
    archive_path: Path,
    snapshot_root: Path,
) -> DeliveryAssetSnapshot:
    if (
        state.mode != mode
        or not state.packaged
        or not _valid_archive_sha256(state.archive_sha256)
    ):
        raise ResultContractError("Verified delivery package state is required")
    expected_root = (
        INFERENCE_ARCHIVE_ROOT if mode == "inference" else FINAL_ARCHIVE_ROOT
    )
    if archive_path.name != f"{expected_root}.tar.gz":
        raise ResultContractError("Delivery archive filename is not mode-exact")
    checksum_path = Path(f"{archive_path}.sha256")
    asset_dir = snapshot_root / ".verified-assets" / mode
    asset_dir.mkdir(parents=True, mode=0o700)
    staged_archive = asset_dir / archive_path.name
    staged_checksum = asset_dir / checksum_path.name
    _copy_regular_source(archive_path, staged_archive)
    _copy_regular_source(checksum_path, staged_checksum)
    if _sha256_file(staged_archive) != state.archive_sha256:
        raise ResultContractError("Delivery archive digest does not match state")
    try:
        checksum_text = _read_pinned_regular_bytes(staged_checksum).decode("ascii")
    except UnicodeError as exc:
        raise ResultContractError("Delivery checksum is invalid") from exc
    expected_checksum = f"{state.archive_sha256}  {archive_path.name}\n"
    if checksum_text != expected_checksum:
        raise ResultContractError("Delivery checksum does not bind archive")
    _extract_verified_archive_members(
        archive=staged_archive,
        mode=mode,
        relative_paths=(),
        destination=snapshot_root,
    )
    if _sha256_file(staged_archive) != state.archive_sha256:
        raise ResultContractError("Delivery archive changed during verification")
    checksums = snapshot_root / "checksums"
    checksums.mkdir(parents=True, mode=0o700, exist_ok=True)
    _copy_regular_source(staged_checksum, checksums / checksum_path.name)
    return DeliveryAssetSnapshot(
        mode=mode,
        archive_sha256=state.archive_sha256,
        checksum_sha256=_sha256_file(staged_checksum),
        archive_name=archive_path.name,
        checksum_name=checksum_path.name,
    )


def stage_verified_delivery_assets(
    *,
    mode: ResultMode,
    delivery_state_path: str | os.PathLike[str],
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> DeliveryAssetSnapshot:
    """Privately snapshot one packaged archive/checksum with exact tar checks."""

    validated_mode = _validate_mode(mode)
    snapshot_root = Path(destination)
    _create_private_snapshot_directory(snapshot_root)
    state = load_delivery_state(delivery_state_path, mode=validated_mode)
    return _stage_delivery_asset(
        mode=validated_mode,
        state=state,
        archive_path=Path(archive_path),
        snapshot_root=snapshot_root,
    )


def _copy_and_bind_runtime_file(
    *,
    runtime_source: Path,
    archive_source: Path,
    snapshot_root: Path,
    relative: str,
) -> None:
    runtime_copy = snapshot_root / ".runtime" / relative
    _copy_regular_source(runtime_source, runtime_copy)
    if _sha256_file(runtime_copy) != _sha256_file(archive_source):
        raise ResultContractError("Runtime result differs from delivery archive")


def _leaf_present(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ResultContractError("Delivery asset presence is unreadable") from exc


def _bind_staged_inference_inputs(
    *,
    snapshot_root: Path,
    runtime: Path,
    inference_archive: Path,
    expected_count: int,
) -> None:
    inference_relatives = (
        "full/manifest.jsonl",
        "full/predictions.jsonl",
        "full/metrics.json",
        "full/full_summary.md",
    )
    _extract_verified_archive_members(
        archive=(
            snapshot_root
            / ".verified-assets"
            / "inference"
            / inference_archive.name
        ),
        mode="inference",
        relative_paths=inference_relatives,
        destination=snapshot_root / "inference",
    )
    for filename in (
        "manifest.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "full_summary.md",
    ):
        _copy_and_bind_runtime_file(
            runtime_source=runtime / filename,
            archive_source=snapshot_root / "inference" / "full" / filename,
            snapshot_root=snapshot_root,
            relative=f"inference/{filename}",
        )
    validate_full_contract(
        snapshot_root / "inference" / "full",
        expected_count=expected_count,
        mode="inference",
    )


def _bind_staged_final_inputs(
    *,
    snapshot_root: Path,
    runtime: Path,
    final_archive: Path,
    expected_count: int,
) -> None:
    final_relatives = (
        "full/metrics.json",
        "full/metrics_final.json",
        "full/final_evaluation_summary.md",
    )
    _extract_verified_archive_members(
        archive=(
            snapshot_root / ".verified-assets" / "final" / final_archive.name
        ),
        mode="final",
        relative_paths=final_relatives,
        destination=snapshot_root / "final",
    )
    for filename in (
        "metrics.json",
        "metrics_final.json",
        "final_evaluation_summary.md",
    ):
        _copy_and_bind_runtime_file(
            runtime_source=runtime / filename,
            archive_source=snapshot_root / "final" / "full" / filename,
            snapshot_root=snapshot_root,
            relative=f"final/{filename}",
        )
    if _sha256_file(
        snapshot_root / "inference" / "full" / "metrics.json"
    ) != _sha256_file(snapshot_root / "final" / "full" / "metrics.json"):
        raise ResultContractError("Inference and final archive metrics differ")
    validation_full = snapshot_root / ".validation-final"
    validation_full.mkdir(mode=0o700)
    for filename, source in (
        ("manifest.jsonl", snapshot_root / "inference" / "full" / "manifest.jsonl"),
        (
            "predictions.jsonl",
            snapshot_root / "inference" / "full" / "predictions.jsonl",
        ),
        ("metrics.json", snapshot_root / "final" / "full" / "metrics.json"),
        (
            "metrics_final.json",
            snapshot_root / "final" / "full" / "metrics_final.json",
        ),
    ):
        _copy_regular_source(source, validation_full / filename)
    validate_full_contract(
        validation_full,
        expected_count=expected_count,
        mode="final",
    )


def stage_verified_delivery_inputs(
    *,
    mode: ResultMode,
    delivery_state_path: str | os.PathLike[str],
    runtime_full_dir: str | os.PathLike[str],
    inference_archive_path: str | os.PathLike[str],
    final_archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    expected_count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
    include_optional_final: bool = True,
    require_selected_uploaded: bool = True,
    require_inference_published: bool = False,
    require_inference_uploaded: bool = True,
) -> DeliveryInputSnapshot:
    """Stage and bind every results-branch/finalizer input before mutation."""

    validated_mode = _validate_mode(mode)
    if type(expected_count) is not int or expected_count < 0:
        raise ResultContractError("Delivery snapshot count is invalid")
    if type(include_optional_final) is not bool:
        raise ResultContractError("Optional final snapshot setting is invalid")
    if type(require_selected_uploaded) is not bool:
        raise ResultContractError("Selected upload requirement is invalid")
    if type(require_inference_published) is not bool:
        raise ResultContractError("Inference publication requirement is invalid")
    if type(require_inference_uploaded) is not bool:
        raise ResultContractError("Inference upload requirement is invalid")
    states = load_delivery_states(delivery_state_path)
    if states is None:
        raise ResultContractError("Delivery state is missing")
    inference_state = states["inference"]
    final_state = states["final"]
    if not inference_state.packaged or (
        require_inference_uploaded and not inference_state.uploaded
    ):
        raise ResultContractError("Verified inference delivery is required")
    if require_inference_published and not inference_state.published:
        raise ResultContractError("Published inference delivery is required")
    if validated_mode == "final" and (
        not inference_state.published
        or not final_state.packaged
        or (require_selected_uploaded and not final_state.uploaded)
    ):
        raise ResultContractError("Published inference and verified final are required")

    snapshot_root = Path(destination)
    _create_private_snapshot_directory(snapshot_root)
    runtime = Path(runtime_full_dir)
    inference_archive = Path(inference_archive_path)
    final_archive = Path(final_archive_path)
    inference_asset = _stage_delivery_asset(
        mode="inference",
        state=inference_state,
        archive_path=inference_archive,
        snapshot_root=snapshot_root,
    )
    _bind_staged_inference_inputs(
        snapshot_root=snapshot_root,
        runtime=runtime,
        inference_archive=inference_archive,
        expected_count=expected_count,
    )

    included_modes: list[ResultMode] = ["inference"]
    selected_asset = inference_asset
    final_declared = include_optional_final and (
        final_state.packaged
        or _leaf_present(final_archive)
        or _leaf_present(Path(f"{final_archive}.sha256"))
    )
    if (
        validated_mode == "inference"
        and final_declared
        and not final_state.published
    ):
        raise ResultContractError(
            "Optional final branch inputs require published final delivery"
        )
    if validated_mode == "final" or final_declared:
        if not final_state.packaged or (
            not final_state.uploaded
            and (validated_mode != "final" or require_selected_uploaded)
        ):
            raise ResultContractError("Final delivery asset is not verified as uploaded")
        final_asset = _stage_delivery_asset(
            mode="final",
            state=final_state,
            archive_path=final_archive,
            snapshot_root=snapshot_root,
        )
        included_modes.append("final")
        if validated_mode == "final":
            selected_asset = final_asset

    if "final" in included_modes:
        _bind_staged_final_inputs(
            snapshot_root=snapshot_root,
            runtime=runtime,
            final_archive=final_archive,
            expected_count=expected_count,
        )

    branch_paths = [
        "inference/full/metrics.json",
        "inference/full/full_summary.md",
        f"checksums/{inference_archive.name}.sha256",
    ]
    if "final" in included_modes:
        branch_paths.append(f"checksums/{final_archive.name}.sha256")
        branch_paths.extend(
            (
                "final/full/metrics_final.json",
                "final/full/final_evaluation_summary.md",
            )
        )
    scan_for_leaks(snapshot_root, relative_paths=branch_paths)
    selected_state = states[validated_mode]
    if not _valid_archive_sha256(selected_state.archive_sha256):
        raise ResultContractError("Selected delivery state has no archive digest")
    return DeliveryInputSnapshot(
        mode=validated_mode,
        selected_archive_sha256=selected_state.archive_sha256,
        selected_checksum_sha256=selected_asset.checksum_sha256,
        selected_archive_name=selected_asset.archive_name,
        selected_checksum_name=selected_asset.checksum_name,
        included_modes=tuple(included_modes),
    )


def stage_verified_final_candidate(
    *,
    delivery_state_path: str | os.PathLike[str],
    runtime_full_dir: str | os.PathLike[str],
    inference_archive_path: str | os.PathLike[str],
    final_archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    expected_count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
) -> FinalCandidateSnapshot:
    """Privately bind an unrecorded final candidate to published inference."""

    if type(expected_count) is not int or expected_count < 0:
        raise ResultContractError("Delivery snapshot count is invalid")
    states = load_delivery_states(delivery_state_path)
    if states is None:
        raise ResultContractError("Delivery state is missing")
    inference_state = states["inference"]
    final_state = states["final"]
    if not inference_state.published:
        raise ResultContractError("Published inference delivery is required")
    if final_state.packaged:
        raise ResultContractError("Final delivery candidate is already recorded")

    snapshot_root = Path(destination)
    _create_private_snapshot_directory(snapshot_root)
    runtime = Path(runtime_full_dir)
    inference_archive = Path(inference_archive_path)
    final_archive = Path(final_archive_path)
    _stage_delivery_asset(
        mode="inference",
        state=inference_state,
        archive_path=inference_archive,
        snapshot_root=snapshot_root,
    )

    candidate_source = snapshot_root / ".candidate-source"
    candidate_source.mkdir(mode=0o700)
    private_archive = candidate_source / final_archive.name
    private_checksum = Path(f"{private_archive}.sha256")
    _copy_regular_source(final_archive, private_archive)
    _copy_regular_source(Path(f"{final_archive}.sha256"), private_checksum)
    candidate_sha = _sha256_file(private_archive)
    candidate_state = DeliveryState(
        mode="final",
        packaged=True,
        archive_sha256=candidate_sha,
    )
    candidate_asset = _stage_delivery_asset(
        mode="final",
        state=candidate_state,
        archive_path=private_archive,
        snapshot_root=snapshot_root,
    )
    _bind_staged_inference_inputs(
        snapshot_root=snapshot_root,
        runtime=runtime,
        inference_archive=inference_archive,
        expected_count=expected_count,
    )
    _bind_staged_final_inputs(
        snapshot_root=snapshot_root,
        runtime=runtime,
        final_archive=final_archive,
        expected_count=expected_count,
    )
    scan_for_leaks(
        snapshot_root,
        relative_paths=(
            "inference/full/metrics.json",
            "inference/full/full_summary.md",
            "final/full/metrics_final.json",
            "final/full/final_evaluation_summary.md",
            f"checksums/{inference_archive.name}.sha256",
            f"checksums/{final_archive.name}.sha256",
        ),
    )
    return FinalCandidateSnapshot(
        archive_sha256=candidate_asset.archive_sha256,
        checksum_sha256=candidate_asset.checksum_sha256,
        archive_name=candidate_asset.archive_name,
        checksum_name=candidate_asset.checksum_name,
    )


def _validated_record_arguments(
    *,
    stage: object,
    status: object,
    archive_sha256: object,
    exit_code: object,
) -> tuple[str, str, str | None, int | None]:
    if type(stage) is not str or stage not in _DELIVERY_STAGES:
        raise ResultContractError("Delivery stage is invalid")
    if type(status) is not str or status not in ("complete", "failed"):
        raise ResultContractError("Delivery status is invalid")
    if archive_sha256 is not None and not _valid_archive_sha256(archive_sha256):
        raise ResultContractError("Delivery archive digest is invalid")
    if exit_code is not None and type(exit_code) is not int:
        raise ResultContractError("Delivery exit code is invalid")
    return stage, status, archive_sha256, exit_code


def _transition_delivery_state(
    state: DeliveryState,
    *,
    stage: str,
    status: str,
    archive_sha256: str | None,
    exit_code: int | None,
) -> DeliveryState:
    expected_exit_code = _DELIVERY_EXIT_CODES[stage]
    completed = {
        "package": state.packaged,
        "upload": state.uploaded,
        "publish": state.published,
    }[stage]

    if archive_sha256 is not None and state.archive_sha256 is not None:
        if archive_sha256 != state.archive_sha256:
            raise ResultContractError("Packaged archive digest is immutable")

    if status == "failed":
        if exit_code != expected_exit_code:
            raise ResultContractError("Delivery stage and exit code do not match")
        if completed:
            raise ResultContractError("A completed delivery stage cannot regress")
        if stage == "package":
            if archive_sha256 is not None or state.packaged:
                raise ResultContractError("Package failure cannot retain a digest")
        elif stage == "upload":
            if not state.packaged or state.uploaded:
                raise ResultContractError("Upload failure prerequisites are not met")
        elif not state.uploaded or state.published:
            raise ResultContractError("Publication failure prerequisites are not met")
        if state.last_exit_code not in (None, expected_exit_code):
            raise ResultContractError("Delivery failure cannot skip a pending stage")
        return replace(state, last_exit_code=expected_exit_code)

    if exit_code is not None:
        raise ResultContractError("Completed delivery stage cannot retain an exit code")
    if stage == "package":
        if archive_sha256 is None:
            raise ResultContractError("Package completion requires an archive digest")
        if state.packaged:
            if state.archive_sha256 != archive_sha256:
                raise ResultContractError("Packaged archive digest is immutable")
            if state.last_exit_code is not None:
                raise ResultContractError("Completed earlier stage cannot clear later failure")
            return state
        if state.last_exit_code not in (None, 30):
            raise ResultContractError("Package completion cannot skip a pending stage")
        return replace(
            state,
            packaged=True,
            archive_sha256=archive_sha256,
            last_exit_code=None,
        )
    if not state.packaged:
        raise ResultContractError("Delivery upload requires a packaged archive")
    if archive_sha256 is not None and archive_sha256 != state.archive_sha256:
        raise ResultContractError("Packaged archive digest is immutable")
    if stage == "upload":
        if state.uploaded:
            if state.last_exit_code is not None:
                raise ResultContractError("Completed earlier stage cannot clear later failure")
            return state
        if state.last_exit_code not in (None, 31):
            raise ResultContractError("Upload completion cannot skip a pending stage")
        return replace(state, uploaded=True, last_exit_code=None)
    if not state.uploaded:
        raise ResultContractError("Delivery publication requires upload")
    if state.published:
        if state.last_exit_code is not None:
            raise ResultContractError("Completed publication state is inconsistent")
        return state
    if state.last_exit_code not in (None, 32):
        raise ResultContractError("Publication completion cannot skip a pending stage")
    return replace(state, published=True, last_exit_code=None)


def _validated_runner_attempt_id(value: object) -> str:
    if (
        type(value) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None
    ):
        raise ResultContractError("Delivery runner attempt ID is invalid")
    return value


def _append_delivery_event_best_effort(
    path: Path,
    *,
    state: DeliveryState,
    stage: str,
    status: str,
    runner_attempt_id: str,
) -> None:
    """Append an allowlisted event; all failures leave durable state authoritative."""

    directory_descriptor: int | None = None
    event_descriptor: int | None = None
    locked = False
    try:
        from .observability import (
            EVENT_FIELDS,
            EVENT_SCHEMA,
            ObservabilitySettings,
            sanitize_payload,
            sanitize_text,
        )

        opened = _open_secure_parent_directory(path, missing_ok=True)
        if opened is None:
            return
        directory_descriptor, filename = opened
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None or fcntl is None:
            return
        event_descriptor = os.open(
            filename,
            os.O_RDWR
            | os.O_APPEND
            | os.O_CREAT
            | nofollow
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(event_descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            return

        attempt_id = f"delivery-{state.mode}-{uuid.uuid4().hex}"
        settings = ObservabilitySettings()
        now_utc = datetime.now(timezone.utc)
        fields = {
            "delivery_mode": state.mode,
            "stage": stage,
            "status": status,
            "exit_code": state.last_exit_code if status == "failed" else None,
            "archive_sha256": state.archive_sha256,
            "runner_attempt_id": runner_attempt_id,
            "message": "delivery stage state recorded",
        }
        if not set(fields).issubset(EVENT_FIELDS["delivery_stage"]):
            return
        payload: dict[str, Any] = {
            "schema": EVENT_SCHEMA,
            "event": "delivery_stage",
            "experiment_version": "full-v2",
            "run_mode": "full",
            "attempt_id": sanitize_text(attempt_id, limit=200),
            "sequence": 1,
            "timestamp_utc": now_utc.isoformat(timespec="seconds"),
            "timestamp_local": now_utc.astimezone(
                ZoneInfo(settings.timezone)
            ).isoformat(timespec="seconds"),
            "process_elapsed_seconds": 0.0,
        }
        payload.update(
            {
                name: sanitize_payload(value, key=name)
                for name, value in fields.items()
            }
        )
        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        fcntl.flock(event_descriptor, fcntl.LOCK_EX)
        locked = True
        locked_metadata = os.fstat(event_descriptor)
        if (
            not stat.S_ISREG(locked_metadata.st_mode)
            or locked_metadata.st_nlink != 1
            or locked_metadata.st_dev != metadata.st_dev
            or locked_metadata.st_ino != metadata.st_ino
        ):
            return
        separator = b""
        if locked_metadata.st_size > 0:
            if os.pread(
                event_descriptor,
                1,
                locked_metadata.st_size - 1,
            ) != b"\n":
                separator = b"\n"
        view = memoryview(separator + serialized)
        while view:
            written = os.write(event_descriptor, view)
            if written <= 0:
                return
            view = view[written:]
        os.fsync(event_descriptor)
        after = os.fstat(event_descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
        ):
            return
        os.fsync(directory_descriptor)
    except Exception:
        return
    finally:
        if locked and event_descriptor is not None and fcntl is not None:
            try:
                fcntl.flock(event_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        if event_descriptor is not None:
            try:
                os.close(event_descriptor)
            except OSError:
                pass
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass


def record_delivery_stage(
    path: str | os.PathLike[str],
    *,
    mode: ResultMode = "inference",
    stage: Literal["package", "upload", "publish"],
    status: Literal["complete", "failed"],
    archive_sha256: str | None = None,
    exit_code: int | None = None,
    event_path: str | os.PathLike[str] | None = None,
    runner_attempt_id: str | None = None,
) -> DeliveryState:
    """Atomically record one monotonic stage transition under a directory lock."""

    validated_mode = _validate_mode(mode)
    stage_value, status_value, digest_value, exit_value = _validated_record_arguments(
        stage=stage,
        status=status,
        archive_sha256=archive_sha256,
        exit_code=exit_code,
    )
    try:
        state_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ResultContractError("Delivery state path is invalid") from exc
    selected_event_path: Path | None = None
    selected_runner_attempt_id: str | None = None
    if event_path is not None:
        try:
            selected_event_path = Path(event_path)
        except (TypeError, ValueError) as exc:
            raise ResultContractError("Delivery event path is invalid") from exc
        selected_runner_attempt_id = _validated_runner_attempt_id(runner_attempt_id)
        if os.path.abspath(selected_event_path) == os.path.abspath(state_path):
            raise ResultContractError("Delivery event path must differ from state path")
    elif runner_attempt_id is not None:
        raise ResultContractError("Delivery runner attempt ID requires an event path")

    opened = _open_secure_parent_directory(state_path, missing_ok=False)
    if opened is None:  # pragma: no cover - missing_ok=False guarantees an exception
        raise ResultContractError("Delivery state parent is missing")
    directory_descriptor, filename = opened
    locked = False
    try:
        if fcntl is None:
            raise ResultContractError("Delivery-state locking is unavailable")
        try:
            fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise ResultContractError("Delivery-state locking failed") from exc
        states = _read_delivery_document_at(
            directory_descriptor,
            filename,
            missing_ok=True,
        )
        if states is None:
            states = _default_delivery_states()
        updated = _transition_delivery_state(
            states[validated_mode],
            stage=stage_value,
            status=status_value,
            archive_sha256=digest_value,
            exit_code=exit_value,
        )
        states[validated_mode] = _validate_delivery_entry(
            asdict(updated),
            expected_mode=validated_mode,
        )
        _write_delivery_document_at(directory_descriptor, filename, states)
    finally:
        if locked:
            try:
                fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(directory_descriptor)

    if selected_event_path is not None and selected_runner_attempt_id is not None:
        _append_delivery_event_best_effort(
            selected_event_path,
            state=updated,
            stage=stage_value,
            status=status_value,
            runner_attempt_id=selected_runner_attempt_id,
        )
    return updated


class _SecretSafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "error: invalid command arguments\n")


def _build_result_contract_parser() -> argparse.ArgumentParser:
    parser = _SecretSafeArgumentParser(
        description="Validate, package, and recover ScamMark result delivery"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    load = commands.add_parser("load-delivery")
    load.add_argument("--state-path", type=Path, required=True)
    load.add_argument("--mode", choices=_DELIVERY_MODES, default="inference")

    record = commands.add_parser("record-delivery")
    record.add_argument("--state-path", type=Path, required=True)
    record.add_argument("--mode", choices=_DELIVERY_MODES, default="inference")
    record.add_argument("--stage", choices=_DELIVERY_STAGES, required=True)
    record.add_argument("--status", choices=("complete", "failed"), required=True)
    record.add_argument("--archive-sha256")
    record.add_argument("--exit-code", type=int)
    record.add_argument("--event-path", type=Path)
    record.add_argument("--runner-attempt-id")

    validate = commands.add_parser("validate-full")
    validate.add_argument("--full-dir", type=Path, required=True)
    validate.add_argument("--mode", choices=_DELIVERY_MODES, default="inference")
    validate.add_argument(
        "--expected-count",
        type=int,
        default=DEFAULT_EXPECTED_SAMPLE_COUNT,
    )

    build = commands.add_parser("build-archive")
    build.add_argument("--mode", choices=_DELIVERY_MODES, required=True)
    build.add_argument("--full-dir", type=Path, required=True)
    build.add_argument("--state-dir", type=Path, required=True)
    build.add_argument("--log-path", type=Path)
    build.add_argument("--config-path", type=Path)
    build.add_argument("--frozen-dir", type=Path)
    build.add_argument("--destination", type=Path, required=True)
    build.add_argument("--release-tag", required=True)
    build.add_argument("--repository", required=True)
    build.add_argument("--git-commit", required=True)
    build.add_argument("--bertscore-asset-status-path", type=Path)
    build.add_argument("--finalize-log-path", type=Path)
    build.add_argument("--secret-values-stdin-json", action="store_true")
    return parser


def _secret_values_from_stdin(enabled: bool) -> tuple[str, ...]:
    if not enabled:
        return ()
    try:
        buffered_stdin = getattr(sys.stdin, "buffer", None)
        if buffered_stdin is not None:
            raw = buffered_stdin.read(_DELIVERY_STATE_MAX_BYTES + 1)
            if type(raw) is not bytes:
                raise ResultContractError("Secret input is unreadable")
        else:
            text_fallback = sys.stdin.read(_DELIVERY_STATE_MAX_BYTES + 1)
            if type(text_fallback) is not str:
                raise ResultContractError("Secret input is unreadable")
            raw = text_fallback.encode("utf-8")
    except ResultContractError:
        raise
    except (AttributeError, OSError, UnicodeError) as exc:
        raise ResultContractError("Secret input is unreadable") from exc
    if len(raw) > _DELIVERY_STATE_MAX_BYTES:
        raise ResultContractError("Secret input is too large")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ResultContractError("Secret input is unreadable") from exc
    payload = _loads_strict_json(text, label="secret input")
    if type(payload) is not list:
        raise ResultContractError("Secret input must be a JSON array")
    return _validated_secret_values(payload)


def _write_cli_json(payload: object) -> None:
    sys.stdout.write(_delivery_json_line(payload))
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = _build_result_contract_parser()
    try:
        arguments = parser.parse_args(argv)
        if arguments.command == "load-delivery":
            state = load_delivery_state(arguments.state_path, mode=arguments.mode)
            _write_cli_json(asdict(state))
        elif arguments.command == "record-delivery":
            state = record_delivery_stage(
                arguments.state_path,
                mode=arguments.mode,
                stage=arguments.stage,
                status=arguments.status,
                archive_sha256=arguments.archive_sha256,
                exit_code=arguments.exit_code,
                event_path=arguments.event_path,
                runner_attempt_id=arguments.runner_attempt_id,
            )
            _write_cli_json(asdict(state))
        elif arguments.command == "validate-full":
            result = validate_full_contract(
                arguments.full_dir,
                expected_count=arguments.expected_count,
                mode=arguments.mode,
            )
            _write_cli_json(
                {"mode": result.mode, "sample_count": result.sample_count}
            )
        elif arguments.command == "build-archive":
            secret_values = _secret_values_from_stdin(
                arguments.secret_values_stdin_json
            )
            archive = build_result_archive(
                mode=arguments.mode,
                full_dir=arguments.full_dir,
                state_dir=arguments.state_dir,
                log_path=arguments.log_path,
                config_path=arguments.config_path,
                frozen_dir=arguments.frozen_dir,
                destination=arguments.destination,
                release_tag=arguments.release_tag,
                repository=arguments.repository,
                git_commit=arguments.git_commit,
                secret_values=secret_values,
                bertscore_asset_status_path=(
                    arguments.bertscore_asset_status_path
                ),
                finalize_log_path=arguments.finalize_log_path,
            )
            _write_cli_json(
                {
                    "archive": archive.name,
                    "sha256": _sha256_archive_destination(archive),
                }
            )
        else:  # pragma: no cover - argparse requires one known subcommand
            raise ResultContractError("Result contract command is invalid")
        return 0
    except ArchiveRecoveryRequired as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    except ResultContractError:
        sys.stderr.write("result contract error\n")
        return 1
    except Exception:
        sys.stderr.write("result contract error\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
