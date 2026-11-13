"""Run URL-XGBoost or CLIP-linear baselines on the frozen ScamWeb-V split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import (
    URL_FEATURE_NAMES,
    binary_targets,
    extract_clip_embeddings,
    fit_clip_linear,
    fit_url_xgboost,
    label_predictions,
    load_jsonl,
    load_split,
    records_for_split,
    url_features,
    verify_split_manifest,
)
from .metrics import compute_binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("url-xgboost", "clip-lr"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/baselines"))
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = load_split(args.split)
    verify_split_manifest(split, args.manifest)
    records = load_jsonl(args.manifest)
    train = records_for_split(records, split, "train")
    validation = records_for_split(records, split, "validation")
    test = records_for_split(records, split, "test")

    if args.baseline == "url-xgboost":
        import numpy as np

        model = fit_url_xgboost(train, validation, seed=args.seed, workers=args.workers)
        test_x = np.asarray([url_features(record.get("url", "")) for record in test])
        predicted = label_predictions(model.predict(test_x))
        protocol = {
            "features": list(URL_FEATURE_NAMES),
            "estimator": "XGBClassifier",
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
        }
    else:
        ordered = train + validation + test
        matrix = extract_clip_embeddings(
            ordered,
            dataset_root=args.dataset_root,
            model_name=args.clip_model,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        train_end = len(train)
        validation_end = train_end + len(validation)
        model = fit_clip_linear(
            matrix[:train_end],
            binary_targets(train),
            matrix[train_end:validation_end],
            binary_targets(validation),
            seed=args.seed,
        )
        predicted = label_predictions(model.predict(matrix[validation_end:]))
        protocol = {
            "encoder": args.clip_model,
            "encoder_frozen": True,
            "classifier": "LogisticRegression",
            "validation_selected_C": float(model.C),
        }

    truth = [str(record["binary"]) for record in test]
    metrics = compute_binary_metrics(truth, predicted, bootstrap_samples=1000, seed=args.seed)
    metrics["protocol"] = {
        "baseline": args.baseline,
        "seed": args.seed,
        "split_schema": split["schema_version"],
        "split_counts": split["split_counts"],
        **protocol,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{args.baseline}_test_predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(
                {"sample_id": record["sample_id"], "truth": true, "prediction": pred},
                ensure_ascii=True,
            )
            + "\n"
            for record, true, pred in zip(test, truth, predicted, strict=True)
        ),
        encoding="utf-8",
    )
    (args.output_dir / f"{args.baseline}_test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
