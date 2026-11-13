"""Portable relocation for machine-specific paths in frozen dataset records."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")


def _normalize(value: str) -> str:
    return value.strip().replace("\\", "/").rstrip("/")


@dataclass(frozen=True)
class DatasetPathMapper:
    source_root: str
    runtime_root: str

    def __post_init__(self) -> None:
        if not _normalize(self.source_root):
            raise ValueError("Frozen source dataset root must not be empty")
        if not _normalize(self.runtime_root):
            raise ValueError("Runtime dataset root must not be empty")

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "source_root": _normalize(self.source_root),
            "runtime_root": _normalize(self.runtime_root),
            "policy": "dataset-root-relocation-v1",
        }

    def relocate_path(self, value: str) -> str:
        if not value:
            return value
        normalized = _normalize(value)
        source = _normalize(self.source_root)
        source_folded = source.casefold()
        value_folded = normalized.casefold()
        if value_folded == source_folded:
            relative = ""
        elif value_folded.startswith(source_folded + "/"):
            relative = normalized[len(source) + 1 :]
        else:
            if _WINDOWS_ABSOLUTE.match(normalized):
                raise ValueError(
                    f"Frozen path is outside the frozen source root: {value}"
                )
            return value

        runtime = _normalize(self.runtime_root)
        return runtime if not relative else f"{runtime}/{relative}"

    def relocate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        relocated = copy.deepcopy(record)
        for key, value in tuple(relocated.items()):
            if key.endswith("_path") and isinstance(value, str):
                relocated[key] = self.relocate_path(value)
        return relocated

    def relocate_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.relocate_record(record) for record in records]
