"""Generate paper-facing ScamWeb-V statistics from raw and frozen evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scammark.taxonomy import (  # noqa: E402
    CANONICAL_LABELS,
    INVALID_LABEL,
    canonicalize_scam_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy-summary",
        type=Path,
        default=REPO_ROOT / "src" / "benchmark" / "summary.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def numeric_box(annotation: Any) -> bool:
    if not isinstance(annotation, dict):
        return False
    center = annotation.get("center")
    size = annotation.get("size")
    if not isinstance(center, dict) or not isinstance(size, dict):
        return False
    try:
        values = [
            float(center["x"]),
            float(center["y"]),
            float(size["width"]),
            float(size["height"]),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return all(value == value for value in values) and values[2] > 0 and values[3] > 0


def audit_gold(dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_root = dataset_root / "dataset_scam_gold"
    json_dir = gold_root / "人工标注结果_scam"
    image_dir = gold_root / "截图_scam"
    mhtml_dir = gold_root / "mhtml_scam"
    json_paths = sorted(json_dir.glob("*.json"), key=lambda path: path.name.lower())
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    mhtml_stems = {path.stem for path in mhtml_dir.glob("*.mhtml")}

    raw_labels: Counter[str] = Counter()
    canonical_labels: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    boxes_per_page: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    unknown_labels: list[dict[str, str]] = []
    missing_image: list[str] = []
    missing_mhtml: list[str] = []
    empty_page_descriptions = 0
    empty_region_reasons = 0
    invalid_boxes = 0
    multi_label_records = 0
    annotation_label_mismatches = 0
    cjk_page_descriptions = 0
    cjk_region_reasons = 0
    total_boxes = 0

    for path in json_paths:
        payload = load_json(path)
        raw_label = str(payload.get("label") or "").strip()
        canonical = canonicalize_scam_type(raw_label)
        raw_labels[raw_label] += 1
        if canonical == INVALID_LABEL:
            unknown_labels.append({"file": path.name, "label": raw_label})
        else:
            canonical_labels[canonical] += 1
        language = str(payload.get("language") or "").strip()
        languages[language] += 1
        description = str(payload.get("description") or "").strip()
        empty_page_descriptions += not bool(description)
        cjk_page_descriptions += bool(re.search(r"[\u4e00-\u9fff]", description))
        multi_label_records += ";" in raw_label

        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            annotations = []
        boxes_per_page[str(len(annotations))] += 1
        total_boxes += len(annotations)
        for annotation in annotations:
            reason = str(annotation.get("description") or "").strip() if isinstance(annotation, dict) else ""
            empty_region_reasons += not bool(reason)
            cjk_region_reasons += bool(re.search(r"[\u4e00-\u9fff]", reason))
            invalid_boxes += not numeric_box(annotation)
            if isinstance(annotation, dict):
                box_label = canonicalize_scam_type(str(annotation.get("label") or ""))
                annotation_label_mismatches += box_label != canonical

        if path.stem not in image_stems:
            missing_image.append(path.name)
        if path.stem not in mhtml_stems:
            missing_mhtml.append(path.name)
        records.append(
            {
                "sample_id": path.stem,
                "raw_label": raw_label,
                "canonical_label": canonical,
                "language": language,
                "box_count": len(annotations),
            }
        )

    counts = list(map(int, boxes_per_page.keys())) if boxes_per_page else []
    return (
        {
            "json_file_count": len(json_paths),
            "screenshot_file_count": len(image_stems),
            "mhtml_file_count": len(mhtml_stems),
            "same_stem_complete_count": sum(
                path.stem in image_stems and path.stem in mhtml_stems for path in json_paths
            ),
            "raw_label_string_count": len(raw_labels),
            "canonical_subtype_count": len(canonical_labels),
            "canonical_subtype_support": sorted_counter(canonical_labels),
            "raw_label_support": sorted_counter(raw_labels),
            "language_support": sorted_counter(languages),
            "gold_box_count": total_boxes,
            "boxes_per_page_distribution": sorted_counter(boxes_per_page),
            "minimum_boxes_per_page": min(counts) if counts else 0,
            "maximum_boxes_per_page": max(counts) if counts else 0,
            "empty_label_count": raw_labels.get("", 0),
            "unknown_label_count": len(unknown_labels),
            "unknown_labels": unknown_labels,
            "multi_label_record_count": multi_label_records,
            "empty_page_description_count": empty_page_descriptions,
            "empty_region_reason_count": empty_region_reasons,
            "invalid_box_count": invalid_boxes,
            "annotation_label_mismatch_count": annotation_label_mismatches,
            "page_description_with_cjk_count": cjk_page_descriptions,
            "region_reason_with_cjk_count": cjk_region_reasons,
            "missing_screenshot_count": len(missing_image),
            "missing_mhtml_count": len(missing_mhtml),
        },
        records,
    )


def find_header(headers: tuple[Any, ...], prefix: str) -> int:
    for index, value in enumerate(headers):
        if str(value or "").strip().startswith(prefix):
            return index
    raise KeyError(f"Workbook header not found: {prefix}")


def audit_workbook(path: Path) -> dict[str, Any]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("Install requirements-paper.txt to audit the workbook") from exc

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    original = workbook["scam原始总表"]
    rows = original.iter_rows(values_only=True)
    headers = next(rows)
    index = {
        "id": find_header(headers, "ID"),
        "url": find_header(headers, "URL"),
        "source": find_header(headers, "来源（名字）"),
        "download_date": find_header(headers, "DOWNLOAD_DATE"),
        "valid_pic": find_header(headers, "valid_pic"),
        "is_unique": find_header(headers, "IS_UNIQUE"),
        "ai_annotated": find_header(headers, "AI是否标注"),
        "manual_file": find_header(headers, "人工标注文件"),
    }
    counters: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    unique_urls: set[str] = set()
    download_dates: list[str] = []
    for row in rows:
        counters["worksheet_records"] += 1
        url = str(row[index["url"]] or "").strip()
        source = str(row[index["source"]] or "").strip() or "(empty)"
        sources[source] += 1
        if url:
            counters["candidate_url_records"] += 1
            unique_urls.add(url.lower())
        if row[index["valid_pic"]] == 1:
            counters["complete_visual_capture_records"] += 1
        if row[index["is_unique"]] == 1:
            counters["unique_page_records"] += 1
        if row[index["ai_annotated"]] == 1:
            counters["ai_annotated_records"] += 1
        if row[index["manual_file"]] not in (None, ""):
            counters["manual_file_records"] += 1
        if row[index["is_unique"]] == 1 and row[index["ai_annotated"]] == 1:
            counters["unique_ai_annotated_records"] += 1
            if row[index["manual_file"]] in (None, ""):
                counters["unique_ai_without_manual_file_records"] += 1
        date = row[index["download_date"]]
        if date not in (None, ""):
            download_dates.append(str(date).split()[0])

    gold_sheet = workbook["scam人工Gold_3416"]
    benign_sheet = workbook["benign正例总表"]
    counters["unique_url_strings_casefolded"] = len(unique_urls)
    counters["gold_sheet_records"] = gold_sheet.max_row - 1
    counters["benign_sheet_records"] = benign_sheet.max_row - 1
    return {
        **dict(counters),
        "scam_download_date_min": min(download_dates) if download_dates else "",
        "scam_download_date_max": max(download_dates) if download_dates else "",
        "source_distribution": sorted_counter(sources),
        "definitions": {
            "candidate_url_records": "rows in scam original sheet with a nonempty URL",
            "complete_visual_capture_records": "rows whose valid_pic field equals 1",
            "unique_page_records": "rows whose IS_UNIQUE field equals 1",
            "unique_url_strings_casefolded": "distinct case-folded URL cell strings; no URL canonicalization",
            "unique_ai_without_manual_file_records": (
                "IS_UNIQUE=1 and AI annotation=1 with no manual annotation file; "
                "not asserted as a final weak-training pool"
            ),
        },
    }


def audit_benign(dataset_root: Path) -> dict[str, Any]:
    benign_root = dataset_root / "dataset_benign"
    images = {path.stem for path in (benign_root / "截图_benign").glob("*.png")}
    mhtml = {path.stem for path in (benign_root / "mhtml_benign").glob("*.mhtml")}
    return {
        "screenshot_file_count": len(images),
        "mhtml_file_count": len(mhtml),
        "same_stem_complete_count": len(images & mhtml),
        "screenshot_without_mhtml_count": len(images - mhtml),
        "mhtml_without_screenshot_count": len(mhtml - images),
    }


def audit_frozen(frozen_dir: Path, raw_gold: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = frozen_dir / "scamweb_v_manifest.jsonl"
    evidence_path = frozen_dir / "web_evidence.jsonl"
    audit_path = frozen_dir / "mhtml_audit.jsonl"
    boxes_path = frozen_dir / "boxes.jsonl"
    version_path = frozen_dir / "dataset_version.json"
    verification_path = frozen_dir / "verification_report.json"
    manifest = load_jsonl(manifest_path)
    audits = load_jsonl(audit_path)
    boxes = load_jsonl(boxes_path)
    version = load_json(version_path)
    verification = load_json(verification_path)

    binary = Counter(str(record.get("binary") or "") for record in manifest)
    labels = Counter(
        str(record.get("scam_type") or "")
        for record in manifest
        if record.get("binary") == "scam"
    )
    statuses = Counter(str(record.get("mhtml_audit_status") or "") for record in manifest)
    conflict_checks = Counter(str(record.get("gold_conflict_check_status") or "") for record in manifest)
    sample_ids = [str(record.get("sample_id") or "") for record in manifest]
    raw_ids = {record["sample_id"] for record in raw_gold}
    frozen_scam_ids = {
        str(record.get("sample_id") or "") for record in manifest if record.get("binary") == "scam"
    }
    hash_files = {
        "frozen_manifest": manifest_path,
        "frozen_evidence": evidence_path,
        "audit": audit_path,
        "recapture": frozen_dir / "mhtml_recapture.jsonl",
        "sample_ids": frozen_dir / "sample_ids.txt",
        "boxes": boxes_path,
    }
    observed_hashes = {name: sha256(path) for name, path in hash_files.items()}
    expected_hashes = version.get("sha256") or {}
    return {
        "manifest_record_count": len(manifest),
        "unique_sample_id_count": len(set(sample_ids)),
        "empty_sample_id_count": sum(not sample_id for sample_id in sample_ids),
        "binary_support": sorted_counter(binary),
        "canonical_subtype_count": len(labels),
        "canonical_subtype_support": sorted_counter(labels),
        "box_row_count": len(boxes),
        "mhtml_status_support": sorted_counter(statuses),
        "mhtml_used_count": sum(bool(record.get("mhtml_used")) for record in manifest),
        "gold_input_conflict_count": sum(bool(record.get("gold_input_conflict")) for record in manifest),
        "gold_conflict_check_support": sorted_counter(conflict_checks),
        "audit_record_count": len(audits),
        "raw_gold_ids_missing_from_frozen": sorted(raw_ids - frozen_scam_ids),
        "frozen_scam_ids_missing_from_raw": sorted(frozen_scam_ids - raw_ids),
        "dataset_version": version,
        "verification_report": verification,
        "observed_sha256": observed_hashes,
        "hash_match": {
            name: observed == expected_hashes.get(name)
            for name, observed in observed_hashes.items()
        },
    }


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold, gold_records = audit_gold(args.dataset_root)
    benign = audit_benign(args.dataset_root)
    workbook = audit_workbook(args.workbook)
    frozen = audit_frozen(args.frozen_dir, gold_records)
    legacy = load_json(args.legacy_summary) if args.legacy_summary.is_file() else {}

    statistics = {
        "paper_contract": {
            "scam_pages": gold["json_file_count"],
            "benign_pages": benign["screenshot_file_count"],
            "total_pages": gold["json_file_count"] + benign["screenshot_file_count"],
            "canonical_subtypes": gold["canonical_subtype_count"],
            "gold_boxes": gold["gold_box_count"],
            "collection_candidate_url_records": workbook["candidate_url_records"],
            "collection_complete_visual_capture_records": workbook[
                "complete_visual_capture_records"
            ],
            "collection_unique_page_records": workbook["unique_page_records"],
            "weak_training_pool": None,
        },
        "gold": gold,
        "benign": benign,
        "workbook": workbook,
        "frozen": frozen,
        "legacy_summary": {
            "status": "legacy_incomplete_not_authoritative_for_final_benchmark",
            "values": legacy,
            "comparison": {
                "candidate_scam_urls": {
                    "legacy": legacy.get("candidate_scam_urls"),
                    "current_defined_count": workbook["candidate_url_records"],
                },
                "complete_visual_captures": {
                    "legacy": legacy.get("complete_visual_captures"),
                    "current_defined_count": workbook["complete_visual_capture_records"],
                },
                "unique_pages": {
                    "legacy": legacy.get("unique_pages"),
                    "current_defined_count": workbook["unique_page_records"],
                },
                "weak_training_pool": {
                    "legacy": legacy.get("weak_training_pool"),
                    "current_defined_count": None,
                    "reason": "No current field or frozen artifact reproduces the legacy definition.",
                },
            },
        },
    }
    (args.output_dir / "dataset_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_csv(
        args.output_dir / "taxonomy_support.csv",
        ["class_id", "scam_type", "support"],
        (
            {
                "class_id": index,
                "scam_type": label,
                "support": gold["canonical_subtype_support"].get(label, 0),
            }
            for index, label in enumerate(CANONICAL_LABELS, start=1)
        ),
    )
    write_csv(
        args.output_dir / "language_support.csv",
        ["language", "support"],
        (
            {"language": language, "support": support}
            for language, support in gold["language_support"].items()
        ),
    )
    write_csv(
        args.output_dir / "field_mapping.csv",
        ["paper_field", "authoritative_source", "source_field", "notes"],
        [
            {
                "paper_field": "binary truth",
                "authoritative_source": "derived frozen manifest",
                "source_field": "binary",
                "notes": "Gold JSON presence defines scam; benign workbook/assets define benign.",
            },
            {
                "paper_field": "scam subtype",
                "authoritative_source": "Gold JSON",
                "source_field": "label",
                "notes": "Deterministic English alias normalization only.",
            },
            {
                "paper_field": "page rationale",
                "authoritative_source": "Gold JSON",
                "source_field": "description",
                "notes": "Workbook descriptions do not override JSON.",
            },
            {
                "paper_field": "page language",
                "authoritative_source": "Gold JSON",
                "source_field": "language",
                "notes": "Reported as annotation metadata, including multilingual combinations.",
            },
            {
                "paper_field": "region boxes and reasons",
                "authoritative_source": "Gold JSON",
                "source_field": "annotations[].center/size/description",
                "notes": "Converted to pixel-space xyxy in the derived manifest.",
            },
            {
                "paper_field": "URL and title",
                "authoritative_source": "Excel gold/benign sheets",
                "source_field": "URL, TITLE",
                "notes": "Used only for indexing and model input, not annotation truth.",
            },
            {
                "paper_field": "MHTML availability",
                "authoritative_source": "frozen MHTML audit",
                "source_field": "mhtml_audit_status, mhtml_used",
                "notes": "Only valid and valid_redirect evidence is exposed to ScamMark.",
            },
        ],
    )

    print(json.dumps(statistics["paper_contract"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
