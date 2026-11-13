"""Paper-aligned metrics for ScamWeb-V classification and grounding."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ci95(values: list[float]) -> list[float]:
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _binary_core(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    tp = sum(true == "scam" and pred == "scam" for true, pred in zip(y_true, y_pred, strict=True))
    fp = sum(true == "benign" and pred != "benign" for true, pred in zip(y_true, y_pred, strict=True))
    tn = sum(true == "benign" and pred == "benign" for true, pred in zip(y_true, y_pred, strict=True))
    fn = sum(true == "scam" and pred != "scam" for true, pred in zip(y_true, y_pred, strict=True))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "support": {"scam": tp + fn, "benign": tn + fp},
        "invalid_prediction_count": sum(pred not in {"scam", "benign"} for pred in y_pred),
    }


def compute_binary_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Compute hard-label binary metrics with scam fixed as the positive class."""

    true_list = [str(value).strip().lower() for value in y_true]
    if any(value not in {"scam", "benign"} for value in true_list):
        raise ValueError("y_true must contain only scam or benign")
    pred_list = [
        normalized if normalized in {"scam", "benign"} else "__invalid__"
        for value in y_pred
        for normalized in [str(value).strip().lower()]
    ]
    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    metrics = _binary_core(true_list, pred_list)
    if not true_list or bootstrap_samples <= 0:
        metrics["ci95"] = {}
        return metrics

    rng = random.Random(seed)
    samples = {name: [] for name in ["accuracy", "precision", "recall", "f1"]}
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(true_list)) for _ in true_list]
        result = _binary_core([true_list[i] for i in indices], [pred_list[i] for i in indices])
        for name in samples:
            samples[name].append(result[name])
    metrics["ci95"] = {name: _ci95(values) for name, values in samples.items()}
    return metrics


def _class_stats(y_true: Sequence[str], y_pred: Sequence[str], label: str) -> dict[str, Any]:
    tp = sum(true == label and pred == label for true, pred in zip(y_true, y_pred, strict=True))
    fp = sum(true != label and pred == label for true, pred in zip(y_true, y_pred, strict=True))
    fn = sum(true == label and pred != label for true, pred in zip(y_true, y_pred, strict=True))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "support": tp + fn,
    }


def _subtype_core(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    per_class = {label: _class_stats(y_true, y_pred, label) for label in labels}
    columns = list(labels)
    for predicted in y_pred:
        if predicted not in columns:
            columns.append(predicted)
    confusion = {
        label: {column: 0 for column in columns}
        for label in labels
    }
    for true, pred in zip(y_true, y_pred, strict=True):
        if true in confusion:
            confusion[true][pred] = confusion[true].get(pred, 0) + 1
    count = len(labels)
    return {
        "accuracy": _safe_div(sum(true == pred for true, pred in zip(y_true, y_pred, strict=True)), len(y_true)),
        "macro_precision": _safe_div(sum(item["precision"] for item in per_class.values()), count),
        "macro_recall": _safe_div(sum(item["recall"] for item in per_class.values()), count),
        "macro_f1": _safe_div(sum(item["f1"] for item in per_class.values()), count),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "support": dict(Counter(y_true)),
        "invalid_prediction_count": sum(pred not in labels for pred in y_pred),
    }


def compute_subtype_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    labels: Sequence[str] | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Compute fixed-taxonomy metrics for single-label scam subtype classification."""

    if labels is None:
        from .taxonomy import CANONICAL_LABELS

        labels = CANONICAL_LABELS
    labels = tuple(labels)
    true_list = [str(value or "none") for value in y_true]
    pred_list = [str(value or "none") for value in y_pred]
    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    metrics = _subtype_core(true_list, pred_list, labels)
    observed_labels = tuple(label for label in labels if label in set(true_list))
    observed = _subtype_core(true_list, pred_list, observed_labels) if observed_labels else {
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
    }
    metrics.update(
        {
            "observed_labels": list(observed_labels),
            "observed_class_macro_precision": observed["macro_precision"],
            "observed_class_macro_recall": observed["macro_recall"],
            "observed_class_macro_f1": observed["macro_f1"],
        }
    )
    if not true_list or bootstrap_samples <= 0:
        metrics["ci95"] = {}
        return metrics

    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(true_list):
        by_label.setdefault(label, []).append(index)
    rng = random.Random(seed)
    samples = {name: [] for name in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}
    for _ in range(bootstrap_samples):
        indices: list[int] = []
        for group in by_label.values():
            indices.extend(rng.choice(group) for _ in group)
        result = _subtype_core([true_list[i] for i in indices], [pred_list[i] for i in indices], labels)
        for name in samples:
            samples[name].append(result[name])
    metrics["ci95"] = {name: _ci95(values) for name, values in samples.items()}
    return metrics


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return _safe_div(intersection, first_area + second_area - intersection)


def _record_image_size(record: dict[str, Any]) -> tuple[float, float] | None:
    candidates = [
        (record.get("truth") or {}).get("image_size"),
        (record.get("input") or {}).get("image_size"),
    ]
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            try:
                width, height = float(candidate[0]), float(candidate[1])
            except (TypeError, ValueError):
                continue
            if width > 0 and height > 0:
                return width, height
    path_value = (record.get("input") or {}).get("image_path")
    path = Path(str(path_value or ""))
    if path.is_file():
        try:
            from .visual_analysis import get_image_size

            width, height = get_image_size(path)
        except Exception:
            return None
        if width and height:
            return float(width), float(height)
    return None


def _normalize_box(box: Any, image_size: tuple[float, float] | None) -> tuple[float, float, float, float] | None:
    coordinates = box.get("xyxy") if isinstance(box, dict) else box
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
        return None
    try:
        values = [float(value) for value in coordinates]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    if image_size:
        width, height = image_size
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _prepare_boxes(
    boxes: Any,
    image_size: tuple[float, float] | None,
    *,
    invalid_if_non_list: bool = False,
) -> tuple[list[tuple[tuple[float, float, float, float], dict[str, Any]]], int]:
    prepared = []
    invalid = 0
    if not isinstance(boxes, list):
        return prepared, int(invalid_if_non_list)
    for item in boxes:
        coordinates = _normalize_box(item, image_size)
        if coordinates is None:
            invalid += 1
            continue
        payload = item if isinstance(item, dict) else {"xyxy": list(item)}
        prepared.append((coordinates, payload))
    return prepared, invalid


def _maximum_iou_matching(
    truth: Sequence[tuple[Sequence[float], dict[str, Any]]],
    predictions: Sequence[tuple[Sequence[float], dict[str, Any]]],
) -> list[tuple[int, int, float]]:
    """Return the exact maximum-total-IoU one-to-one matching.

    Gold pages contain at most four boxes, so dynamic programming over the gold
    mask remains exact even if a model emits many predictions.
    """

    truth_count = len(truth)
    if not truth_count or not predictions:
        return []
    dp: dict[int, tuple[float, tuple[tuple[int, int, float], ...]]] = {0: (0.0, ())}
    for pred_index, (pred_box, _) in enumerate(predictions):
        updated = dict(dp)
        for mask, (score, assignments) in dp.items():
            for truth_index, (truth_box, _) in enumerate(truth):
                if mask & (1 << truth_index):
                    continue
                overlap = box_iou(truth_box, pred_box)
                if overlap <= 0:
                    continue
                next_mask = mask | (1 << truth_index)
                candidate = (score + overlap, assignments + ((truth_index, pred_index, overlap),))
                current = updated.get(next_mask)
                if current is None or candidate[0] > current[0]:
                    updated[next_mask] = candidate
        dp = updated
    return list(max(dp.values(), key=lambda item: item[0])[1])


def _threshold_matrix_matching(
    iou_matrix: Sequence[Sequence[float]],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Maximize threshold hits first and total IoU second for one-to-one matches."""

    truth_count = len(iou_matrix)
    prediction_count = max((len(row) for row in iou_matrix), default=0)
    if not truth_count or not prediction_count:
        return []
    dp: dict[int, tuple[int, float, tuple[tuple[int, int, float], ...]]] = {
        0: (0, 0.0, ())
    }
    for pred_index in range(prediction_count):
        updated = dict(dp)
        for mask, (count, score, assignments) in dp.items():
            for truth_index, row in enumerate(iou_matrix):
                if mask & (1 << truth_index) or pred_index >= len(row):
                    continue
                overlap = float(row[pred_index])
                if overlap < threshold:
                    continue
                next_mask = mask | (1 << truth_index)
                candidate = (
                    count + 1,
                    score + overlap,
                    assignments + ((truth_index, pred_index, overlap),),
                )
                current = updated.get(next_mask)
                if current is None or candidate[:2] > current[:2]:
                    updated[next_mask] = candidate
        dp = updated
    return list(max(dp.values(), key=lambda item: (item[0], item[1]))[2])


def _threshold_iou_matching(
    truth: Sequence[tuple[Sequence[float], dict[str, Any]]],
    predictions: Sequence[tuple[Sequence[float], dict[str, Any]]],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    matrix = [
        [box_iou(truth_box, pred_box) for pred_box, _ in predictions]
        for truth_box, _ in truth
    ]
    return _threshold_matrix_matching(matrix, threshold=threshold)


def compute_grounding_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    page_mious: list[float] = []
    page_recalls_03: list[float] = []
    page_recalls_05: list[float] = []
    tp = fp = fn = 0
    invalid_predictions = invalid_truth = 0
    gold_box_count = predicted_box_count = positive_pages = page_count = 0

    for record in records:
        page_count += 1
        image_size = _record_image_size(record)
        truth, invalid_gt = _prepare_boxes((record.get("truth") or {}).get("boxes"), image_size)
        predictions, invalid_pred = _prepare_boxes(
            (record.get("prediction") or {}).get("boxes"),
            image_size,
            invalid_if_non_list=True,
        )
        invalid_truth += invalid_gt
        invalid_predictions += invalid_pred
        gold_box_count += len(truth)
        predicted_box_count += len(predictions) + invalid_pred
        miou_matches = _maximum_iou_matching(truth, predictions)
        matches_03 = _threshold_iou_matching(truth, predictions, threshold=0.3)
        matches_05 = _threshold_iou_matching(truth, predictions, threshold=0.5)
        hits_05 = len(matches_05)
        tp += hits_05
        fp += len(predictions) - hits_05 + invalid_pred
        fn += len(truth) - hits_05
        if truth:
            positive_pages += 1
            iou_by_truth = [0.0] * len(truth)
            for truth_index, _, overlap in miou_matches:
                iou_by_truth[truth_index] = overlap
            page_mious.append(sum(iou_by_truth) / len(truth))
            page_recalls_03.append(len(matches_03) / len(truth))
            page_recalls_05.append(len(matches_05) / len(truth))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "miou": _safe_div(sum(page_mious), len(page_mious)),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": _f1(precision, recall),
        "page_recall_at_0_3": _safe_div(sum(page_recalls_03), len(page_recalls_03)),
        "page_recall_at_0_5": _safe_div(sum(page_recalls_05), len(page_recalls_05)),
        "counts_at_0_5": {"tp": tp, "fp": fp, "fn": fn},
        "page_count": page_count,
        "positive_page_count": positive_pages,
        "gold_box_count": gold_box_count,
        "predicted_box_count": predicted_box_count,
        "invalid_prediction_count": invalid_predictions,
        "invalid_ground_truth_count": invalid_truth,
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", str(text or "").lower())


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return 0.0

    @lru_cache(maxsize=None)
    def lcs(i: int, j: int) -> int:
        if i == len(predicted) or j == len(expected):
            return 0
        if predicted[i] == expected[j]:
            return 1 + lcs(i + 1, j + 1)
        return max(lcs(i + 1, j), lcs(i, j + 1))

    length = lcs(0, 0)
    return _f1(length / len(predicted), length / len(expected))


def compute_rationale_metrics(
    records: Iterable[dict[str, Any]],
    *,
    bertscore_fn: Callable[[list[str], list[str]], Sequence[float]] | None = None,
) -> dict[str, Any]:
    predictions_text: list[str] = []
    references_text: list[str] = []
    gold_box_count = 0

    for record in records:
        image_size = _record_image_size(record)
        truth, _ = _prepare_boxes((record.get("truth") or {}).get("boxes"), image_size)
        predictions, _ = _prepare_boxes(
            (record.get("prediction") or {}).get("boxes"),
            image_size,
            invalid_if_non_list=True,
        )
        gold_box_count += len(truth)
        for truth_index, pred_index, overlap in _threshold_iou_matching(
            truth,
            predictions,
            threshold=0.5,
        ):
            predicted_reason = str(predictions[pred_index][1].get("reason") or "").strip()
            reference_reason = str(truth[truth_index][1].get("reason") or "").strip()
            if predicted_reason and reference_reason:
                predictions_text.append(predicted_reason)
                references_text.append(reference_reason)

    rouge_scores = [
        rouge_l_f1(prediction, reference)
        for prediction, reference in zip(predictions_text, references_text, strict=True)
    ]
    bertscore = None
    bertscore_error = ""
    if bertscore_fn is None:
        bertscore_status = "not_requested"
    elif not predictions_text:
        bertscore_status = "no_matched_pairs"
    else:
        try:
            scores = [float(value) for value in bertscore_fn(predictions_text, references_text)]
            bertscore = _safe_div(sum(scores), len(scores))
            bertscore_status = "computed"
        except Exception as exc:  # noqa: BLE001 - evaluation artifact must record metric failures.
            bertscore_status = "error"
            bertscore_error = f"{type(exc).__name__}: {str(exc)[:500]}"
    metadata_value = getattr(bertscore_fn, "metadata", None) if bertscore_fn is not None else None
    if callable(metadata_value):
        metadata_value = metadata_value()
    return {
        "gold_box_count": gold_box_count,
        "scored_pair_count": len(predictions_text),
        "grounded_rationale_coverage_at_0_5": _safe_div(len(predictions_text), gold_box_count),
        "rouge_l_f1": _safe_div(sum(rouge_scores), len(rouge_scores)),
        "bertscore_f1": bertscore,
        "bertscore_status": bertscore_status,
        "bertscore_metadata": metadata_value if isinstance(metadata_value, dict) else {},
        "bertscore_error": bertscore_error,
    }
