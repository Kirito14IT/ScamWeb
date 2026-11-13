"""Command-line entry point for the ScamMark local demo."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .evaluate import evaluate_prediction_records, multilingual_bertscore
from .evidence_loader import load_samples, read_dataset_root, write_jsonl
from .model_client import make_model_client, selected_model_ids
from .pipeline import ScamMarkPipeline
from .prediction_schema import INVALID_BINARY, normalize_final_prediction
from .prompts import PROMPT_STAGES


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root or read_dataset_root())
    manifest_path = Path(args.manifest or "outputs/dataset/demo_manifest.jsonl")
    evidence_path = Path(args.web_evidence or "outputs/dataset/web_evidence.jsonl")

    manifest_source = "provided_manifest" if args.manifest else "frozen_gold_benign_demo_manifest"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen manifest not found: {manifest_path}. Build it first with "
            "`python -m src.data.scamweb_dataset --dataset-root <dataset-root>`."
        )

    samples = load_samples(manifest_path, evidence_path if evidence_path.exists() else None, limit=args.limit)
    model_client = make_model_client(args.backend, args.model)
    pipeline = ScamMarkPipeline(
        model_client=model_client,
        prompt_mode=args.prompt_mode,
        force_downstream=args.force_downstream,
    )

    start = time.perf_counter()
    prediction_records = []
    for sample in samples:
        try:
            prediction = pipeline.predict(sample)
        except Exception as exc:  # noqa: BLE001 - keep overnight/API demos inspectable.
            prediction = {
                "binary": INVALID_BINARY,
                "scam_type": "none",
                "description": "",
                "reasoning": f"backend_error:{type(exc).__name__}: {str(exc)[:500]}",
                "boxes": [],
                "evidence_used": {},
                "sample_id": str(sample.get("sample_id") or sample.get("id") or ""),
                "backend_error": True,
            }
        prediction = normalize_prediction_schema(prediction)
        prediction_records.append(build_prediction_record(sample, prediction))
    elapsed = time.perf_counter() - start

    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, prediction_records)

    backend_error_count = sum(
        1 for record in prediction_records if record["prediction"].get("backend_error")
    )
    model_ids = selected_model_ids(model_client)
    engineering = {
        "backend": args.backend,
        **model_ids,
        "prompt_mode": args.prompt_mode,
        "force_downstream": args.force_downstream,
        "manifest_source": manifest_source,
        "json_valid_rate": (
            sum(record["prediction"].get("schema_valid", False) for record in prediction_records)
            / len(prediction_records)
            if prediction_records
            else 0.0
        ),
        "schema_invalid_count": sum(
            not record["prediction"].get("schema_valid", False) for record in prediction_records
        ),
        "backend_error_count": backend_error_count,
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / len(prediction_records) if prediction_records else None,
        "sample_count": len(prediction_records),
        "paths": {
            "manifest": str(manifest_path),
            "web_evidence": str(evidence_path),
            "predictions": str(predictions_path),
        },
    }
    metrics = evaluate_prediction_records(
        prediction_records,
        engineering=engineering,
        bootstrap_samples=args.bootstrap_samples,
        bertscore_fn=multilingual_bertscore if args.with_bertscore else None,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    write_reports(
        reports_dir=reports_dir,
        output_dir=output_dir,
        metrics=metrics,
        predictions=prediction_records,
        dataset_root=dataset_root,
    )
    return 0


def normalize_prediction_schema(prediction: dict) -> dict:
    """Validate API JSON and enforce the final cascade schema."""

    return normalize_final_prediction(prediction)


def build_prediction_record(sample: dict, prediction: dict) -> dict:
    """Build the stable prediction/truth record shared by the demo and evaluator."""

    truth = {
        "binary": sample.get("binary"),
        "scam_type": sample.get("scam_type") or "none",
        "boxes": sample.get("boxes") or [],
        "description": sample.get("description") or sample.get("truth_description") or "",
        "label_source": sample.get("label_source", "manifest"),
    }
    if sample.get("image_size") is not None:
        truth["image_size"] = sample.get("image_size")
    return {
        "sample_id": sample.get("sample_id") or sample.get("id"),
        "input": {
            "url": sample.get("url"),
            "title": sample.get("title"),
            "parse_status": sample.get("parse_status"),
            "image_path": sample.get("image_path"),
            "image_exists": bool(sample.get("image_path") and Path(sample["image_path"]).exists()),
        },
        "truth": truth,
        "prediction": prediction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local ScamMark demo.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--web-evidence", default=None)
    parser.add_argument("--output-dir", default="outputs/scammark_demo")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--backend", choices=["rule", "openai", "local"], default="rule")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-mode", choices=["heuristic", "staged"], default="staged")
    parser.add_argument("--force-downstream", action="store_true")
    parser.add_argument("--with-bertscore", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def write_reports(
    reports_dir: Path,
    output_dir: Path,
    metrics: dict,
    predictions: list[dict],
    dataset_root: Path,
) -> None:
    engineering = metrics["engineering"]
    binary = metrics["binary"]
    subtype = metrics["subtype"]["end_to_end"]
    grounding = metrics["grounding"]["end_to_end"]
    rationale = metrics["rationale"]["end_to_end"]
    speed = engineering.get("seconds_per_sample") or 0.0
    backend_error_count = engineering.get("backend_error_count", 0)
    schema_invalid_count = engineering.get("schema_invalid_count", 0)
    prompt_mode = engineering.get("prompt_mode", "staged")
    full_gold_hours = speed * 3416 / 3600 if speed else 0.0
    full_dataset_hours = speed * 5658 / 3600 if speed else 0.0

    example_lines = []
    for record in predictions[:5]:
        pred = record["prediction"]
        truth = record["truth"]
        example_lines.append(
            f"- `{record['sample_id']}` truth={truth['binary']}/{truth.get('scam_type')} "
            f"pred={pred['binary']}/{pred.get('scam_type')} "
            f"reasoning={pred.get('reasoning', '')[:160]}"
        )
    if not example_lines:
        example_lines.append("- No predictions were produced.")

    parse_counts = Counter(str(record.get("input", {}).get("parse_status") or "unknown") for record in predictions)
    parse_lines = [f"- `{key}`: `{value}`" for key, value in sorted(parse_counts.items())]
    if not parse_lines:
        parse_lines.append("- No input records were loaded.")

    paths = engineering.get("paths") or {}
    demo_report = f"""# ScamMark Demo Report

## Run Summary
- Backend: `{engineering.get('backend', '')}`
- Text model: `{engineering.get('text_model', '')}`
- Visual model: `{engineering.get('visual_model', '')}`
- Prompt mode: `{prompt_mode}`
- Force downstream: `{engineering.get('force_downstream', False)}`
- Dataset root: `{dataset_root}`
- Manifest source: `{engineering.get('manifest_source', '')}`
- Samples: `{engineering.get('sample_count', 0)}`
- JSON valid rate: `{engineering.get('json_valid_rate', 0.0):.4f}`
- Schema-invalid outputs: `{schema_invalid_count}`
- Backend errors: `{backend_error_count}`
- Elapsed seconds: `{engineering.get('elapsed_seconds', 0.0):.2f}`
- Seconds per sample: `{speed:.4f}`

## Outputs
- Manifest: `{paths.get('manifest', '')}`
- Web evidence: `{paths.get('web_evidence', '')}`
- Predictions: `{paths.get('predictions', '')}`
- Metrics: `{output_dir / 'metrics.json'}`

## End-to-End Metrics
- Binary Accuracy / Precision / Recall / F1: `{binary['accuracy']:.4f}` / `{binary['precision']:.4f}` / `{binary['recall']:.4f}` / `{binary['f1']:.4f}`
- Subtype Accuracy / Macro Precision / Macro Recall / Macro F1: `{subtype['accuracy']:.4f}` / `{subtype['macro_precision']:.4f}` / `{subtype['macro_recall']:.4f}` / `{subtype['macro_f1']:.4f}`
- Grounding mIoU / F1@0.5 / page Recall@0.3 / page Recall@0.5: `{grounding['miou']:.4f}` / `{grounding['f1_at_0_5']:.4f}` / `{grounding['page_recall_at_0_3']:.4f}` / `{grounding['page_recall_at_0_5']:.4f}`
- Rationale ROUGE-L F1 / BERTScore F1 / grounded coverage@0.5: `{rationale['rouge_l_f1']:.4f}` / `{rationale['bertscore_f1']}` / `{rationale['grounded_rationale_coverage_at_0_5']:.4f}`

## Example Predictions
{chr(10).join(example_lines)}

## Prompt Stage Summaries
{_prompt_stage_lines(predictions)}

## MHTML Parse Status
{chr(10).join(parse_lines)}

## Interpretation
- This run uses the same evaluator as the standalone `scammark.evaluate` command.
- End-to-end subtype, grounding, and rationale scores include upstream binary errors.
- Forced downstream results are diagnostic only and belong in the appendix.
- BERTScore remains `not_run` unless `--with-bertscore` is supplied.
- Any backend or schema error makes the run an engineering diagnostic rather than a clean model result.

## Runtime Projection
- At the measured speed, 3,416 Gold scam pages would take about `{full_gold_hours:.2f}` hours.
- All 5,658 pages would take about `{full_dataset_hours:.2f}` hours, excluding API rate limits, network variance, image upload overhead, and retries.
"""
    (reports_dir / "demo_report.md").write_text(demo_report, encoding="utf-8")

    status = "red" if not predictions or backend_error_count or schema_invalid_count else "yellow"
    morning_summary = f"""# Morning Summary

## Status
{status}

## Plain-English Conclusion
- The ScamMark demo reads samples, parses MHTML evidence, predicts the cascade, and evaluates all paper tasks through one metrics implementation.
- Current prompt mode: `{prompt_mode}`.
- Local Windows API experiments do not require a local GPU; a server is preferable for stable full-dataset runs.
- This report is a smoke-test artifact, not a paper result.

## Next Step
1. Confirm the 3,416 scam + 2,242 benign frozen manifest audit.
2. Run a 2 scam + 2 benign staged API smoke test.
3. Only after that check, schedule the full 5,658-page experiment.
"""
    (reports_dir / "morning_summary.md").write_text(morning_summary, encoding="utf-8")


def _prompt_stage_lines(predictions: list[dict]) -> str:
    lines: list[str] = []
    for record in predictions[:5]:
        prediction = record.get("prediction") or {}
        evidence = prediction.get("evidence_used") or {}
        stages = evidence.get("prompt_stages") or {}
        if not stages:
            continue
        lines.append(f"- `{record.get('sample_id')}` staged analyses:")
        for stage in PROMPT_STAGES:
            data = stages.get(stage) or {}
            summary = (
                data.get("evidence_summary")
                or data.get("reasoning")
                or data.get("risk_level")
                or data.get("binary")
                or data.get("scam_type")
                or ""
            )
            if isinstance(summary, list):
                summary = "; ".join(str(item) for item in summary[:3])
            lines.append(f"  - `{stage}`: {str(summary)[:180]}")
    return "\n".join(lines) if lines else "- Prompt stage summaries are unavailable because no staged outputs were recorded."


if __name__ == "__main__":
    raise SystemExit(main())
