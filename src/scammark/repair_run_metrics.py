from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .evaluate import evaluate_prediction_records


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_retry_predictions(base_dir: Path, retry_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    base_metrics = load_json(base_dir / "metrics.json")
    records = load_jsonl(base_dir / "predictions.jsonl")
    by_id: dict[str, int] = {}
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id") or "")
        if not sample_id or sample_id in by_id:
            raise ValueError(f"Base predictions contain an empty or duplicate sample_id: {sample_id}")
        by_id[sample_id] = index
    replaced: list[str] = []
    seen_retry_ids: set[str] = set()

    for retry_dir in retry_dirs:
        for retry_record in load_jsonl(retry_dir / "predictions.jsonl"):
            sample_id = str(retry_record.get("sample_id") or "")
            if not sample_id or sample_id in seen_retry_ids:
                raise ValueError(f"Retry predictions contain an empty or duplicate sample_id: {sample_id}")
            seen_retry_ids.add(sample_id)
            if sample_id not in by_id:
                continue
            if retry_record.get("prediction", {}).get("backend_error"):
                continue
            old_record = records[by_id[sample_id]]
            if not old_record.get("prediction", {}).get("backend_error"):
                continue
            if retry_record.get("truth") and retry_record["truth"] != old_record.get("truth"):
                raise ValueError(f"Retry truth differs from the authoritative base record: {sample_id}")
            old_input = old_record.get("input") or {}
            retry_input = retry_record.get("input") or {}
            for key in ["url", "image_path"]:
                if old_input.get(key) and retry_input.get(key) and old_input[key] != retry_input[key]:
                    raise ValueError(f"Retry input provenance differs for {sample_id}: {key}")
            repaired = copy.deepcopy(old_record)
            repaired["prediction"] = copy.deepcopy(retry_record.get("prediction") or {})
            records[by_id[sample_id]] = repaired
            replaced.append(sample_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, records)
    metrics = compute_metrics(records, base_metrics, predictions_path, replaced)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def compute_metrics(
    prediction_records: list[dict[str, Any]],
    base_metrics: dict[str, Any],
    predictions_path: Path,
    replaced: list[str],
) -> dict[str, Any]:
    engineering = dict(base_metrics.get("engineering") or {})
    legacy_paths = base_metrics.get("paths") or {}
    base_predictions = str((engineering.get("paths") or legacy_paths).get("predictions", ""))
    engineering["backend_error_count"] = sum(
        1 for record in prediction_records if record["prediction"].get("backend_error")
    )
    engineering["sample_count"] = len(prediction_records)
    engineering["paths"] = dict(engineering.get("paths") or legacy_paths)
    engineering["paths"]["predictions"] = str(predictions_path)
    engineering["repair"] = {
        "base_predictions": base_predictions,
        "replaced_sample_ids": replaced,
    }
    return evaluate_prediction_records(prediction_records, engineering=engineering)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge successful retry predictions into a failed ScamMark run.")
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--retry-output-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = merge_retry_predictions(
        Path(args.base_output_dir),
        [Path(path) for path in args.retry_output_dir],
        Path(args.output_dir),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
