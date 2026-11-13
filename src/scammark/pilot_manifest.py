"""Deterministic 10/60-sample ScamWeb-V engineering pilots."""

from __future__ import annotations

import hashlib
from typing import Any

from .taxonomy import CANONICAL_LABELS


EXTRA_SUBTYPES = (
    "Online Shopping Scam",
    "Cryptocurrency Investment Scam",
    "Online Investment Scam",
    "Technical Support Scam",
    "Online Gambling Scam",
)

PILOT10_SUBTYPES = (
    "Online Shopping Scam",
    "Cryptocurrency Investment Scam",
    "Phishing",
    "Technical Support Scam",
    "Sextortion",
)


def _rank(record: dict[str, Any], seed: int) -> str:
    sample_id = str(record.get("sample_id") or record.get("id") or "")
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _status(record: dict[str, Any]) -> str:
    explicit = str(record.get("mhtml_audit_status") or "").strip().lower()
    if explicit:
        return explicit
    if str(record.get("parse_status") or "") != "ok":
        return "missing"
    return "valid" if record.get("url_domain_observed") is not False else "uncertain"


def _height(record: dict[str, Any]) -> int:
    size = record.get("image_size") or [0, 0]
    try:
        return int(size[1])
    except (IndexError, TypeError, ValueError):
        return 0


def _is_valid_gold_scam(record: dict[str, Any]) -> bool:
    boxes = record.get("boxes")
    return (
        record.get("binary") == "scam"
        and record.get("is_gold") is True
        and record.get("label_source") == "gold_manual_json"
        and not bool(record.get("gold_input_conflict"))
        and record.get("gold_conflict_check_status") != "error"
        and isinstance(boxes, list)
        and bool(boxes)
        and all(
            isinstance(box, dict)
            and isinstance(box.get("xyxy"), list)
            and len(box["xyxy"]) == 4
            and bool(str(box.get("reason") or "").strip())
            for box in boxes
        )
    )


def _pick_for_label(records: list[dict[str, Any]], label: str, count: int, seed: int) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if _is_valid_gold_scam(record) and record.get("scam_type") == label
    ]
    candidates.sort(key=lambda record: (_status(record) != "valid", _rank(record, seed)))
    if len(candidates) < count:
        raise ValueError(f"Not enough records for subtype {label}: need {count}, found {len(candidates)}")
    return candidates[:count]


def _height_spread(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    candidates = sorted(records, key=lambda record: (_height(record), _rank(record, seed)))
    if len(candidates) < count:
        raise ValueError(f"Not enough benign records: need {count}, found {len(candidates)}")
    if count == 1:
        return [candidates[len(candidates) // 2]]
    indices = [round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)]
    return [candidates[index] for index in indices]


def _select_sixty(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected_scam: list[dict[str, Any]] = []
    for label in CANONICAL_LABELS:
        selected_scam.extend(_pick_for_label(records, label, 2 if label in EXTRA_SUBTYPES else 1, seed))

    benign = [
        record
        for record in records
        if record.get("binary") == "benign"
        and not bool(record.get("gold_input_conflict"))
        and record.get("gold_conflict_check_status") != "error"
    ]
    selected_benign: list[dict[str, Any]] = []
    quotas = {"valid_redirect": 2, "mhtml_mismatch": 2, "uncertain": 1, "missing": 1}
    for status, quota in quotas.items():
        matches = sorted((record for record in benign if _status(record) == status), key=lambda record: _rank(record, seed))
        selected_benign.extend(matches[:quota])
    selected_ids = {str(record.get("sample_id") or record.get("id")) for record in selected_benign}
    clean = [
        record
        for record in benign
        if str(record.get("sample_id") or record.get("id")) not in selected_ids and _status(record) == "valid"
    ]
    selected_benign.extend(_height_spread(clean, 30 - len(selected_benign), seed))
    return sorted(selected_scam + selected_benign, key=lambda record: _rank(record, seed + 60))


def select_pilot_records(records: list[dict[str, Any]], size: int, seed: int = 2027) -> list[dict[str, Any]]:
    sixty = _select_sixty(records, seed)
    if size == 60:
        return sixty
    if size != 10:
        raise ValueError("Only fixed 10- and 60-sample pilots are supported")
    scam = []
    for label in PILOT10_SUBTYPES:
        scam.append(next(record for record in sixty if record.get("binary") == "scam" and record.get("scam_type") == label))
    clean_benign = [record for record in sixty if record.get("binary") == "benign" and _status(record) == "valid"]
    benign = _height_spread(clean_benign, 5, seed)
    return sorted(scam + benign, key=lambda record: _rank(record, seed + 10))
