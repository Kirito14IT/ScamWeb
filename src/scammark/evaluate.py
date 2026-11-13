"""Standalone paper-aligned evaluator for ScamMark prediction files."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

from .metrics import (
    compute_binary_metrics,
    compute_grounding_metrics,
    compute_rationale_metrics,
    compute_subtype_metrics,
)
from .prediction_schema import normalize_final_prediction
from .taxonomy import INVALID_LABEL, canonicalize_scam_type


HUMAN_DIMENSIONS = ["correctness", "faithfulness", "specificity", "helpfulness"]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_engineering_metrics(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    engineering = payload.get("engineering")
    if not isinstance(engineering, dict):
        raise ValueError("Server metrics file does not contain an engineering object")
    return copy.deepcopy(engineering)


def enrich_truth_from_manifest(
    prediction_records: list[dict[str, Any]],
    manifest_records: list[dict[str, Any]],
    *,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    """Return prediction records with authoritative manifest truth attached."""

    def unique_index(records: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
        indexed = {}
        for record in records:
            sample_id = str(record.get("sample_id") or record.get("id") or record.get("raw_id") or "")
            if not sample_id:
                raise ValueError(f"{source} contains an empty sample_id")
            if sample_id in indexed:
                raise ValueError(f"{source} contains duplicate sample_id: {sample_id}")
            indexed[sample_id] = record
        return indexed

    manifest_by_id = unique_index(manifest_records, "manifest")
    prediction_by_id = unique_index(prediction_records, "predictions")
    unknown = sorted(set(prediction_by_id) - set(manifest_by_id))
    if unknown:
        raise ValueError(f"Predictions contain sample_ids absent from manifest: {unknown[:5]}")
    missing = sorted(set(manifest_by_id) - set(prediction_by_id))
    if missing and not allow_partial:
        raise ValueError(
            f"Predictions cover only {len(prediction_by_id)}/{len(manifest_by_id)} manifest samples; "
            "use --allow-partial only for diagnostics."
        )
    enriched = []
    for original in prediction_records:
        record = copy.deepcopy(original)
        sample_id = str(record.get("sample_id") or "")
        manifest = manifest_by_id.get(sample_id)
        if not manifest:
            raise KeyError(f"Prediction sample_id is absent from manifest: {sample_id}")
        binary = "scam" if str(manifest.get("binary") or "").lower() == "scam" else "benign"
        raw_type = manifest.get("scam_type") if binary == "scam" else "none"
        scam_type = canonicalize_scam_type(raw_type)
        record["truth"] = {
            "binary": binary,
            "scam_type": scam_type,
            "boxes": copy.deepcopy(manifest.get("boxes") or []),
            "description": str(manifest.get("description") or ""),
            "label_source": manifest.get("label_source", "manifest"),
        }
        if manifest.get("image_size"):
            record["truth"]["image_size"] = copy.deepcopy(manifest["image_size"])
        record.setdefault("input", {})
        for key in ["url", "title", "image_path", "parse_status"]:
            if manifest.get(key) not in (None, ""):
                record["input"][key] = manifest[key]
        enriched.append(record)
    return enriched


def _stage_output(record: dict[str, Any], stage: str) -> dict[str, Any]:
    prediction = record.get("prediction") or {}
    evidence = prediction.get("evidence_used") or {}
    stages = evidence.get("prompt_stages") or {}
    value = stages.get(stage)
    return value if isinstance(value, dict) else {}


def _canonical_prediction(value: Any) -> str:
    mapped = canonicalize_scam_type(str(value or ""))
    return mapped if mapped != INVALID_LABEL else INVALID_LABEL


def _forced_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    forced = []
    type_available = grounding_available = 0
    scam_count = 0
    for original in records:
        if (original.get("truth") or {}).get("binary") != "scam":
            continue
        scam_count += 1
        record = copy.deepcopy(original)
        primary_prediction = record.get("prediction") or {}
        diagnostic = ((record.get("diagnostics") or {}).get("oracle_forced_downstream") or {})
        stage_record = record
        if primary_prediction.get("binary") != "scam" and diagnostic.get("status") == "complete":
            stage_record = {"prediction": diagnostic.get("prediction") or {}}
        type_output = _stage_output(stage_record, "type")
        grounding_output = _stage_output(stage_record, "grounding")
        if type_output:
            type_available += 1
        if grounding_output:
            grounding_available += 1
        prediction = record.setdefault("prediction", {})
        prediction["scam_type"] = _canonical_prediction(type_output.get("scam_type") or "none")
        prediction["boxes"] = copy.deepcopy(grounding_output.get("boxes") or [])
        prediction["description"] = grounding_output.get("description") or prediction.get("description") or ""
        prediction["reasoning"] = grounding_output.get("reasoning") or prediction.get("reasoning") or ""
        forced.append(record)
    availability = {
        "oracle_gated": True,
        "scam_sample_count": scam_count,
        "type_stage_available": type_available,
        "grounding_stage_available": grounding_available,
        "complete": type_available == scam_count and grounding_available == scam_count,
    }
    return forced, availability


def evaluate_prediction_records(
    records: list[dict[str, Any]],
    *,
    engineering: dict[str, Any] | None = None,
    bootstrap_samples: int = 1000,
    bertscore_fn: Callable[[list[str], list[str]], Sequence[float]] | None = None,
) -> dict[str, Any]:
    records = [copy.deepcopy(record) for record in records]
    for record in records:
        record["prediction"] = normalize_final_prediction(record.get("prediction"))

    y_true_binary = [(record.get("truth") or {}).get("binary") for record in records]
    y_pred_binary = [(record.get("prediction") or {}).get("binary") for record in records]
    binary = compute_binary_metrics(
        y_true_binary,
        y_pred_binary,
        bootstrap_samples=bootstrap_samples,
    )

    scam_records = [record for record in records if (record.get("truth") or {}).get("binary") == "scam"]
    subtype_true = [_canonical_prediction((record.get("truth") or {}).get("scam_type")) for record in scam_records]
    subtype_end_pred = [
        "none"
        if (record.get("prediction") or {}).get("binary") != "scam"
        else _canonical_prediction((record.get("prediction") or {}).get("scam_type"))
        for record in scam_records
    ]
    subtype_end = compute_subtype_metrics(
        subtype_true,
        subtype_end_pred,
        bootstrap_samples=bootstrap_samples,
    )

    forced_records, forced_availability = _forced_records(records)
    forced_true = [_canonical_prediction((record.get("truth") or {}).get("scam_type")) for record in forced_records]
    forced_pred = [_canonical_prediction((record.get("prediction") or {}).get("scam_type")) for record in forced_records]
    subtype_forced = compute_subtype_metrics(
        forced_true,
        forced_pred,
        bootstrap_samples=bootstrap_samples,
    )
    subtype_forced["availability"] = forced_availability

    rationale_end = compute_rationale_metrics(records, bertscore_fn=bertscore_fn)
    rationale_forced = compute_rationale_metrics(forced_records, bertscore_fn=bertscore_fn)
    rationale_forced["availability"] = forced_availability
    grounding_end = compute_grounding_metrics(records)
    grounding_forced = compute_grounding_metrics(forced_records)

    engineering_metrics = dict(engineering or {})
    schema_invalid_count = sum(
        not (record.get("prediction") or {}).get("schema_valid", False) for record in records
    )
    backend_error_count = sum(
        bool((record.get("prediction") or {}).get("backend_error")) for record in records
    )
    engineering_metrics.update(
        {
            "sample_count": len(records),
            "schema_invalid_count": schema_invalid_count,
            "json_valid_rate": (len(records) - schema_invalid_count) / len(records) if records else 0.0,
            "backend_error_count": backend_error_count,
            "clean_metrics": (
                bool(records)
                and schema_invalid_count == 0
                and backend_error_count == 0
                and grounding_end["invalid_ground_truth_count"] == 0
            ),
        }
    )

    return {
        "binary": binary,
        "subtype": {
            "end_to_end": subtype_end,
            "forced": subtype_forced,
        },
        "grounding": {
            "end_to_end": grounding_end,
            "forced": {
                **grounding_forced,
                "availability": forced_availability,
            },
        },
        "rationale": {
            "end_to_end": rationale_end,
            "forced": rationale_forced,
            "human": {"status": "not_provided"},
        },
        "engineering": engineering_metrics,
    }


def multilingual_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    from .bertscore_eval import get_bertscore_evaluator

    return get_bertscore_evaluator()(predictions, references)


def _bootstrap_mean_ci(values: list[float], samples: int, seed: int = 2027) -> list[float]:
    if not values or samples <= 0:
        return []
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        resample = [rng.choice(values) for _ in values]
        means.append(sum(resample) / len(resample))
    means.sort()
    lower = means[round((len(means) - 1) * 0.025)]
    upper = means[round((len(means) - 1) * 0.975)]
    return [lower, upper]


def _default_alpha(matrix: list[list[float]]) -> float:
    try:
        import krippendorff
    except ImportError as exc:
        raise RuntimeError("Install requirements-eval.txt before aggregating human ratings.") from exc
    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"))


def aggregate_human_ratings(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    alpha_fn: Callable[[list[list[float]]], float] | None = None,
) -> dict[str, Any]:
    """Aggregate blinded 1-5 ratings by candidate and dimension."""

    alpha_fn = alpha_fn or _default_alpha
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        rater_id = str(row.get("rater_id") or "").strip()
        item_id = str(row.get("item_id") or "").strip()
        if not candidate_id or not rater_id or not item_id:
            raise ValueError("candidate_id, rater_id, and item_id must be nonempty")
        key = (candidate_id, rater_id, item_id)
        if key in seen:
            raise ValueError(f"Duplicate human rating row: {key}")
        seen.add(key)
    candidates: dict[str, dict[str, Any]] = {}
    candidate_ids = sorted({str(row.get("candidate_id") or "") for row in rows if row.get("candidate_id")})
    for candidate_id in candidate_ids:
        candidate_rows = [row for row in rows if str(row.get("candidate_id") or "") == candidate_id]
        candidate_metrics: dict[str, Any] = {}
        raters = sorted({str(row.get("rater_id") or "") for row in candidate_rows})
        items = sorted({str(row.get("item_id") or "") for row in candidate_rows})
        for dimension in HUMAN_DIMENSIONS:
            values = []
            by_pair: dict[tuple[str, str], float] = {}
            for row in candidate_rows:
                try:
                    value = float(row[dimension])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid {dimension} rating for candidate {candidate_id}") from exc
                if value < 1 or value > 5 or not value.is_integer():
                    raise ValueError(f"{dimension} rating must be an integer within 1..5")
                values.append(value)
                by_pair[(str(row.get("rater_id") or ""), str(row.get("item_id") or ""))] = value
            matrix = [
                [by_pair.get((rater, item), math.nan) for item in items]
                for rater in raters
            ]
            candidate_metrics[dimension] = {
                "mean": sum(values) / len(values) if values else 0.0,
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci95": _bootstrap_mean_ci(values, bootstrap_samples),
                "alpha_ordinal": alpha_fn(matrix) if matrix and items else None,
                "rating_count": len(values),
            }
        candidates[candidate_id] = candidate_metrics
    return {
        "status": "computed",
        "dimensions": list(HUMAN_DIMENSIONS),
        "rating_count": len(rows),
        "candidates": candidates,
    }


def load_human_ratings(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ScamMark predictions against a frozen manifest.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--human-ratings", default=None)
    parser.add_argument(
        "--engineering-metrics",
        default=None,
        help="Preserve the engineering section from a server metrics.json file.",
    )
    parser.add_argument("--with-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default=None)
    parser.add_argument("--bertscore-device", default="cuda")
    parser.add_argument("--bertscore-batch-size", type=int, default=4)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_jsonl(args.predictions)
    manifest = load_jsonl(args.manifest)
    records = enrich_truth_from_manifest(predictions, manifest, allow_partial=args.allow_partial)
    bertscore_fn = None
    if args.with_bertscore:
        from .bertscore_eval import BERTScoreConfig, default_model_dir, get_bertscore_evaluator

        bertscore_fn = get_bertscore_evaluator(
            BERTScoreConfig(
                model=args.bertscore_model or str(default_model_dir()),
                device=args.bertscore_device,
                batch_size=args.bertscore_batch_size,
            )
        )
    engineering = (
        load_engineering_metrics(args.engineering_metrics)
        if args.engineering_metrics
        else {}
    )
    engineering.update(
        {
            "manifest_sample_count": len(manifest),
            "prediction_sample_count": len(predictions),
            "coverage": len(predictions) / len(manifest) if manifest else 0.0,
            "partial_evaluation": args.allow_partial,
            "predictions": str(args.predictions),
            "manifest": str(args.manifest),
            "posthoc_bertscore_requested": bool(args.with_bertscore),
        }
    )
    metrics = evaluate_prediction_records(
        records,
        engineering=engineering,
        bootstrap_samples=args.bootstrap_samples,
        bertscore_fn=bertscore_fn,
    )
    if args.human_ratings:
        metrics["rationale"]["human"] = aggregate_human_ratings(load_human_ratings(args.human_ratings))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
