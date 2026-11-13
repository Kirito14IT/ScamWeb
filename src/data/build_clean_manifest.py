from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def clean_text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).strip().split())


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def image_size(path: str) -> tuple[int, int] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width, height = struct.unpack(">II", header[16:24])
                return int(width), int(height)
            if header.startswith(b"\xff\xd8"):
                handle.seek(2)
                while True:
                    marker_start = handle.read(1)
                    if not marker_start:
                        return None
                    if marker_start != b"\xff":
                        continue
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                        length = struct.unpack(">H", handle.read(2))[0]
                        payload = handle.read(length - 2)
                        if len(payload) >= 5:
                            height, width = struct.unpack(">HH", payload[1:5])
                            return int(width), int(height)
                        return None
                    length_bytes = handle.read(2)
                    if len(length_bytes) != 2:
                        return None
                    length = struct.unpack(">H", length_bytes)[0]
                    handle.seek(max(0, length - 2), 1)
    except Exception:
        return None
    return None


def record_clean_status(record: dict[str, Any], evidence: dict[str, Any] | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    image_path = clean_text(record.get("image_path"))
    mhtml_path = clean_text(record.get("mhtml_path"))
    if not image_path or not Path(image_path).exists():
        reasons.append("image missing")
    if not mhtml_path or not Path(mhtml_path).exists():
        reasons.append("mhtml missing")
    if not evidence:
        reasons.append("web evidence missing")
    else:
        if evidence.get("parse_status") != "ok":
            reasons.append("mhtml parse not ok")
        if not evidence.get("url_domain_observed"):
            reasons.append("mhtml domain not observed")
    if record.get("binary") == "scam" and not is_true(record.get("is_gold")):
        reasons.append("scam not gold")
    return not reasons, reasons


def build_clean_outputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    evidence_path = Path(args.web_evidence)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    records = load_jsonl(manifest_path)
    evidence_records = load_jsonl(evidence_path)
    evidence_by_id = {clean_text(e.get("sample_id")): e for e in evidence_records}

    clean_records: list[dict[str, Any]] = []
    clean_evidence: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    reject_by_binary: Counter[str] = Counter()

    for record in records:
        evidence = evidence_by_id.get(clean_text(record.get("sample_id")))
        ok, reasons = record_clean_status(record, evidence)
        if ok:
            clean_records.append(record)
            if evidence:
                clean_evidence.append(evidence)
        else:
            binary = clean_text(record.get("binary")) or "unknown"
            for reason in reasons:
                reject_reasons[f"{binary} / {reason}"] += 1
                reject_by_binary[binary] += 1

    clean_scam = [r for r in clean_records if r.get("binary") == "scam"]
    clean_benign = [r for r in clean_records if r.get("binary") == "benign"]
    demo_scam = clean_scam[: args.demo_per_class]
    demo_benign = clean_benign[: args.demo_per_class]
    demo_records = demo_scam + demo_benign
    demo_ids = {clean_text(r.get("sample_id")) for r in demo_records}
    demo_evidence = [e for e in clean_evidence if clean_text(e.get("sample_id")) in demo_ids]

    clean_manifest = output_dir / "full_clean_manifest.jsonl"
    clean_web_evidence = output_dir / "full_clean_web_evidence.jsonl"
    demo_manifest = output_dir / "demo_manifest.clean_balanced.jsonl"
    demo_web_evidence = output_dir / "web_evidence.clean_balanced.jsonl"
    write_jsonl(clean_manifest, clean_records)
    write_jsonl(clean_web_evidence, clean_evidence)
    write_jsonl(demo_manifest, demo_records)
    write_jsonl(demo_web_evidence, demo_evidence)

    summary = {
        "input_records": len(records),
        "input_evidence_records": len(evidence_records),
        "clean_total": len(clean_records),
        "clean_scam_gold": len(clean_scam),
        "clean_benign": len(clean_benign),
        "demo_total": len(demo_records),
        "demo_scam": len(demo_scam),
        "demo_benign": len(demo_benign),
        "reject_reasons": dict(reject_reasons.most_common(20)),
        "reject_by_binary": dict(reject_by_binary),
        "outputs": {
            "full_clean_manifest": str(clean_manifest),
            "full_clean_web_evidence": str(clean_web_evidence),
            "demo_manifest_clean_balanced": str(demo_manifest),
            "web_evidence_clean_balanced": str(demo_web_evidence),
        },
    }
    write_text(reports_dir / "full_clean_manifest_report.md", render_clean_report(summary, demo_records))
    return summary


def inspect_boxes(args: argparse.Namespace) -> dict[str, Any]:
    manifest_records = load_jsonl(Path(args.manifest))
    image_by_id = {clean_text(r.get("sample_id")): clean_text(r.get("image_path")) for r in manifest_records}
    binary_by_id = {clean_text(r.get("sample_id")): clean_text(r.get("binary")) for r in manifest_records}
    dim_cache: dict[str, tuple[int, int] | None] = {}
    source_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    sample_issues: list[dict[str, Any]] = []
    total = 0

    with Path(args.boxes_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            sid = clean_text(row.get("sample_id"))
            source_counter[clean_text(row.get("source")) or "unknown"] += 1
            try:
                x1 = float(row.get("x1") or "")
                y1 = float(row.get("y1") or "")
                x2 = float(row.get("x2") or "")
                y2 = float(row.get("y2") or "")
            except ValueError:
                issue_counter["non_numeric_coordinate"] += 1
                add_issue(sample_issues, row, "non_numeric_coordinate")
                continue
            if x2 <= x1 or y2 <= y1:
                issue_counter["invalid_order_or_size"] += 1
                add_issue(sample_issues, row, "invalid_order_or_size")
            if min(x1, y1, x2, y2) < 0:
                issue_counter["negative_coordinate"] += 1
                add_issue(sample_issues, row, "negative_coordinate")

            image_path = image_by_id.get(sid, "")
            if not image_path or not Path(image_path).exists():
                issue_counter["image_missing_for_box"] += 1
                add_issue(sample_issues, row, "image_missing_for_box")
                continue
            if image_path not in dim_cache:
                dim_cache[image_path] = image_size(image_path)
            dims = dim_cache[image_path]
            if not dims:
                issue_counter["image_size_unreadable"] += 1
                add_issue(sample_issues, row, "image_size_unreadable")
                continue
            width, height = dims
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                issue_counter["out_of_image_bounds"] += 1
                add_issue(sample_issues, row, f"out_of_image_bounds:{width}x{height}")

    benign_with_boxes = sum(1 for sid, binary in binary_by_id.items() if binary == "benign" and sid in image_by_id)
    summary = {
        "coordinate_schema": "xyxy pixel columns in boxes.csv (x1,y1,x2,y2)",
        "box_rows": total,
        "box_source_counts": dict(source_counter),
        "issue_counts": dict(issue_counter),
        "sample_issues": sample_issues[:30],
        "image_dimensions_checked": sum(1 for value in dim_cache.values() if value),
        "benign_manifest_records": sum(1 for b in binary_by_id.values() if b == "benign"),
        "benign_box_rows": count_benign_box_rows(Path(args.boxes_csv)),
    }
    write_text(Path(args.reports_dir) / "boxes_quality_report.md", render_boxes_report(summary))
    return summary


def add_issue(sample_issues: list[dict[str, Any]], row: dict[str, Any], issue: str) -> None:
    if len(sample_issues) >= 30:
        return
    sample_issues.append(
        {
            "sample_id": clean_text(row.get("sample_id")),
            "box_id": clean_text(row.get("box_id")),
            "binary": clean_text(row.get("binary")),
            "source": clean_text(row.get("source")),
            "issue": issue,
            "xyxy": [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")],
        }
    )


def count_benign_box_rows(boxes_csv: Path) -> int:
    with boxes_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for row in csv.DictReader(handle) if clean_text(row.get("binary")) == "benign")


def render_clean_report(summary: dict[str, Any], demo_records: list[dict[str, Any]]) -> str:
    lines = [
        "# Full Clean Manifest Report",
        "",
        "## Filter Rule",
        "- image exists",
        "- MHTML exists",
        "- MHTML parse_status is `ok`",
        "- URL domain is observed in parsed MHTML",
        "- scam records must be gold; benign records are kept only for binary auxiliary evaluation",
        "",
        "## Summary",
    ]
    for key in [
        "input_records",
        "input_evidence_records",
        "clean_total",
        "clean_scam_gold",
        "clean_benign",
        "demo_total",
        "demo_scam",
        "demo_benign",
    ]:
        lines.append(f"- {key}: `{summary[key]}`")
    lines.extend(["", "## Outputs"])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Top Reject Reasons"])
    for reason, count in summary["reject_reasons"].items():
        lines.append(f"- {reason}: `{count}`")
    lines.extend(["", "## Demo Samples"])
    lines.append("| sample_id | binary | scam_type | url |")
    lines.append("|---|---|---|---|")
    for record in demo_records:
        lines.append(
            f"| {record.get('sample_id')} | {record.get('binary')} | {clean_text(record.get('scam_type')) or 'none'} | {clean_text(record.get('url'))} |"
        )
    return "\n".join(lines) + "\n"


def render_boxes_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Boxes Quality Report",
        "",
        "## Summary",
        f"- coordinate_schema: `{summary['coordinate_schema']}`",
        f"- box_rows: `{summary['box_rows']}`",
        f"- image_dimensions_checked: `{summary['image_dimensions_checked']}`",
        f"- benign_manifest_records: `{summary['benign_manifest_records']}`",
        f"- benign_box_rows: `{summary['benign_box_rows']}`",
        "",
        "## Box Source Counts",
    ]
    for source, count in summary["box_source_counts"].items():
        lines.append(f"- {source}: `{count}`")
    lines.extend(["", "## Issue Counts"])
    if summary["issue_counts"]:
        for issue, count in summary["issue_counts"].items():
            lines.append(f"- {issue}: `{count}`")
    else:
        lines.append("- none: `0`")
    lines.extend(["", "## Interpretation"])
    lines.append("- `boxes.csv` uses unified `xyxy` pixel columns.")
    if summary["benign_box_rows"] == 0:
        lines.append("- Benign records have no grounding boxes, matching the binary-only auxiliary policy.")
    else:
        lines.append("- Benign records unexpectedly have box rows and must be inspected before binary evaluation.")
    if summary["issue_counts"]:
        lines.append("- Records with negative or out-of-bounds boxes should be clamped or excluded before official grounding metrics.")
    else:
        lines.append("- No coordinate issues were detected in the checked box rows.")
    lines.extend(["", "## Sample Issues"])
    if summary["sample_issues"]:
        lines.append("| sample_id | box_id | binary | source | issue | xyxy |")
        lines.append("|---|---|---|---|---|---|")
        for issue in summary["sample_issues"]:
            lines.append(
                f"| {issue['sample_id']} | {issue['box_id']} | {issue['binary']} | {issue['source']} | {issue['issue']} | {issue['xyxy']} |"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean ScamWeb-V manifest and quality reports.")
    parser.add_argument("--manifest", default="outputs/dataset/scamweb_v_manifest.jsonl")
    parser.add_argument("--web-evidence", default="outputs/dataset/web_evidence.jsonl")
    parser.add_argument("--boxes-csv", default="outputs/dataset/boxes.csv")
    parser.add_argument("--output-dir", default="outputs/dataset")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--demo-per-class", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_summary = build_clean_outputs(args)
    boxes_summary = inspect_boxes(args)
    print(json.dumps({"clean": clean_summary, "boxes": boxes_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
