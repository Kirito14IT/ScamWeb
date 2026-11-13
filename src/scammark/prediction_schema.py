"""Strict final-output normalization shared by inference and evaluation."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .taxonomy import INVALID_LABEL, canonicalize_scam_type


INVALID_BINARY = "__invalid__"


def normalize_binary_label(value: Any) -> str:
    if not isinstance(value, str):
        return INVALID_BINARY
    normalized = value.strip().lower()
    return normalized if normalized in {"scam", "benign"} else INVALID_BINARY


def normalize_final_prediction(prediction: Any) -> dict[str, Any]:
    """Validate required fields and enforce the binary cascade gate."""

    if not isinstance(prediction, dict):
        prediction = {}
    normalized = deepcopy(prediction)
    existing_errors = normalized.get("schema_errors")
    errors: list[str] = [
        str(error) for error in existing_errors
    ] if isinstance(existing_errors, list) else []

    binary = normalize_binary_label(normalized.get("binary"))
    if binary == INVALID_BINARY:
        errors.append("binary_must_be_scam_or_benign")
    normalized["binary"] = binary

    raw_boxes = normalized.get("boxes")
    if not isinstance(raw_boxes, list):
        errors.append("boxes_must_be_list")
        raw_boxes = [{"xyxy": None, "reason": "", "schema_error": "boxes_must_be_list"}]
    else:
        validated_boxes = []
        for box in raw_boxes:
            coordinates = box.get("xyxy") if isinstance(box, dict) else None
            bbox_1000 = box.get("bbox_1000") if isinstance(box, dict) else None
            valid = isinstance(coordinates, (list, tuple)) and len(coordinates) == 4
            if valid:
                try:
                    numeric = [float(value) for value in coordinates]
                    valid = (
                        all(math.isfinite(value) for value in numeric)
                        and numeric[2] > numeric[0]
                        and numeric[3] > numeric[1]
                    )
                except (TypeError, ValueError):
                    valid = False
            if not valid:
                has_geometry_shape = (
                    isinstance(coordinates, (list, tuple)) and len(coordinates) == 4
                ) or (
                    isinstance(bbox_1000, (list, tuple)) and len(bbox_1000) == 4
                )
                if isinstance(box, dict) and has_geometry_shape:
                    validated = deepcopy(box)
                    validated["xyxy"] = None
                    validated["schema_error"] = "invalid_geometry"
                    validated_boxes.append(validated)
                    continue
                errors.append("each_box_requires_four_finite_numeric_xyxy_values")
                validated_boxes.append(
                    {"xyxy": None, "reason": "", "schema_error": "invalid_box"}
                )
                continue
            validated = deepcopy(box)
            validated["xyxy"] = numeric
            validated_boxes.append(validated)
        raw_boxes = validated_boxes

    if binary == "scam":
        mapped_type = canonicalize_scam_type(normalized.get("scam_type"))
        if mapped_type in {"none", INVALID_LABEL}:
            errors.append("scam_type_must_be_one_canonical_label")
            mapped_type = INVALID_LABEL
        normalized["scam_type"] = mapped_type
        normalized["boxes"] = raw_boxes
    else:
        if isinstance(raw_boxes, list) and raw_boxes and binary == "benign":
            errors.append("benign_output_must_not_contain_boxes")
        normalized["scam_type"] = "none"
        normalized["boxes"] = []

    for key in ["description", "reasoning"]:
        value = normalized.get(key)
        if not isinstance(value, str):
            errors.append(f"{key}_must_be_string")
            normalized[key] = "" if value is None else str(value)

    if not isinstance(normalized.get("evidence_used"), dict):
        errors.append("evidence_used_must_be_object")
        normalized["evidence_used"] = {}

    if normalized.get("backend_error"):
        errors.append("backend_error")
        normalized["binary"] = INVALID_BINARY
        normalized["scam_type"] = "none"
        normalized["boxes"] = []

    normalized["schema_errors"] = list(dict.fromkeys(errors))
    normalized["schema_valid"] = not normalized["schema_errors"]
    return normalized
