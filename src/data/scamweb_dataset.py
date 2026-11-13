from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import quopri
import re
import struct
import sys
from collections import Counter
from email import policy
from email.parser import BytesHeaderParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openpyxl import load_workbook

try:
    from scammark.taxonomy import canonicalize_scam_type
except ModuleNotFoundError:  # Supports `python -m src.data.scamweb_dataset` from the repository root.
    from src.scammark.taxonomy import canonicalize_scam_type


SCAM_SHEET = "scam人工Gold_3416"
BENIGN_SHEET = "benign正例总表"
EXPECTED_GOLD_RECORDS = 3416
EXPECTED_BENIGN_RECORDS = 2242
EXPECTED_TOTAL_RECORDS = 5658
EXPECTED_CANONICAL_SUBTYPES = 25
EXPECTED_GOLD_BOXES = 5489


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def domain_from_url(url: str) -> str:
    url = clean_text(url)
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split(":")[0]


def path_to_text(path: Path | None) -> str:
    return str(path) if path else ""


def read_png_size(path: Path | None) -> list[int] | None:
    """Read PNG width and height without adding an imaging dependency."""

    if not path or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            return None
        width, height = struct.unpack(">II", header[16:24])
        return [width, height]
    except OSError:
        return None


def valid_image_size(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    )


def parse_semicolon_numbers(value: Any) -> list[float]:
    text = clean_text(value)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[;,，；]", text) if p.strip()]
    numbers: list[float] = []
    for part in parts:
        try:
            numbers.append(float(part))
        except ValueError:
            continue
    return numbers


def xyxy_from_center(cx: float, cy: float, width: float, height: float) -> list[float]:
    return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]


def parse_semicolon_boxes(cx: Any, cy: Any, width: Any, height: Any) -> list[dict[str, Any]]:
    xs = parse_semicolon_numbers(cx)
    ys = parse_semicolon_numbers(cy)
    ws = parse_semicolon_numbers(width)
    hs = parse_semicolon_numbers(height)
    boxes: list[dict[str, Any]] = []
    for i, (x, y, w, h) in enumerate(zip(xs, ys, ws, hs), start=1):
        if w <= 0 or h <= 0:
            continue
        boxes.append(
            {
                "box_id": i,
                "center": {"x": x, "y": y},
                "size": {"width": w, "height": h},
                "xyxy": xyxy_from_center(x, y, w, h),
            }
        )
    return boxes


class EvidenceHTMLParser(HTMLParser):
    def __init__(self, list_limit: int = 200):
        super().__init__(convert_charrefs=True)
        self.list_limit = list_limit
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self.forms: list[dict[str, Any]] = []
        self.buttons: list[str] = []
        self.meta: dict[str, str] = {}
        self._tag_stack: list[str] = []
        self._skip_depth = 0
        self._current_form: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.lower(): clean_text(v) for k, v in attrs if k}
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "meta":
            key = attr.get("name") or attr.get("property")
            content = attr.get("content")
            if key and content and len(self.meta) < self.list_limit:
                self.meta[key] = content
        elif tag == "a" and len(self.links) < self.list_limit:
            self.links.append({"href": attr.get("href", ""), "title": attr.get("title", "")})
        elif tag == "form":
            self._current_form = {
                "action": attr.get("action", ""),
                "method": attr.get("method", ""),
                "inputs": [],
            }
            if len(self.forms) < self.list_limit:
                self.forms.append(self._current_form)
        elif tag in {"input", "textarea", "select"}:
            control = {
                "tag": tag,
                "name": attr.get("name", ""),
                "type": attr.get("type", ""),
                "placeholder": attr.get("placeholder", ""),
                "value": attr.get("value", ""),
            }
            if self._current_form is not None:
                self._current_form["inputs"].append(control)
            label = control["placeholder"] or control["value"] or control["name"]
            if label and len(self.buttons) < self.list_limit:
                self.buttons.append(label)
        elif tag == "button":
            label = attr.get("aria-label") or attr.get("title") or attr.get("value")
            if label and len(self.buttons) < self.list_limit:
                self.buttons.append(label)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "form":
            self._current_form = None
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = clean_text(data)
        if not text or self._skip_depth:
            return
        current = self._tag_stack[-1] if self._tag_stack else ""
        if current == "title":
            self.title_parts.append(text)
        elif current == "button" and len(self.buttons) < self.list_limit:
            self.buttons.append(text)
        self.text_parts.append(text)


def _read_mime_headers(handle: Any) -> Any:
    lines: list[bytes] = []
    while True:
        line = handle.readline()
        if not line or line in {b"\n", b"\r\n"}:
            break
        lines.append(line)
    return BytesHeaderParser(policy=policy.default).parsebytes(b"".join(lines))


def _decode_mime_payload(payload: bytes, transfer_encoding: str, charset: str) -> str:
    encoding = transfer_encoding.strip().lower()
    if encoding == "quoted-printable":
        decoded = quopri.decodestring(payload)
    elif encoding == "base64":
        decoded = base64.b64decode(payload, validate=False)
    else:
        decoded = payload
    try:
        return decoded.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return decoded.decode("utf-8", errors="replace")


def _stream_first_html_part(path: Path, byte_limit: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Read only the first HTML MIME part and skip embedded binary resources."""

    with path.open("rb") as handle:
        root_headers = _read_mime_headers(handle)
        boundary = root_headers.get_boundary()
        root_type = root_headers.get_content_type()
        if not boundary:
            if root_type not in {"text/html", "application/xhtml+xml"}:
                return {"html": "", "content_type": root_type, "archive_url": "", "truncated": False}
            payload = handle.read(byte_limit + 1)
            return {
                "html": _decode_mime_payload(
                    payload[:byte_limit],
                    root_headers.get("Content-Transfer-Encoding", ""),
                    root_headers.get_content_charset() or "utf-8",
                ),
                "content_type": root_type,
                "archive_url": clean_text(root_headers.get("Content-Location") or root_headers.get("Snapshot-Content-Location")),
                "truncated": len(payload) > byte_limit,
            }

        marker = b"--" + boundary.encode("utf-8", errors="replace")
        closing_marker = marker + b"--"
        for line in handle:
            if line.rstrip(b"\r\n") != marker:
                continue
            part_headers = _read_mime_headers(handle)
            content_type = part_headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                continue
            chunks: list[bytes] = []
            size = 0
            truncated = False
            while True:
                body_line = handle.readline()
                if not body_line:
                    break
                stripped = body_line.rstrip(b"\r\n")
                if stripped in {marker, closing_marker}:
                    break
                remaining = byte_limit - size
                if remaining <= 0:
                    truncated = True
                    break
                chunks.append(body_line[:remaining])
                size += min(len(body_line), remaining)
                if len(body_line) > remaining:
                    truncated = True
                    break
            return {
                "html": _decode_mime_payload(
                    b"".join(chunks),
                    part_headers.get("Content-Transfer-Encoding", ""),
                    part_headers.get_content_charset() or "utf-8",
                ),
                "content_type": content_type,
                "archive_url": clean_text(part_headers.get("Content-Location") or root_headers.get("Snapshot-Content-Location")),
                "truncated": truncated,
            }
    return {"html": "", "content_type": "", "archive_url": "", "truncated": False}


def parse_mhtml_file(
    path: Path,
    page_url: str = "",
    text_limit: int = 12000,
    list_limit: int = 200,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mhtml_path": str(path),
        "page_url": page_url,
        "parse_status": "missing",
        "file_size": 0,
        "title": "",
        "meta": {},
        "visible_text": "",
        "visible_text_length": 0,
        "links": [],
        "link_count": 0,
        "forms": [],
        "form_count": 0,
        "buttons": [],
        "button_count": 0,
        "external_domains": [],
        "content_types": {},
        "resource_count": 0,
        "html_length": 0,
        "html_truncated": False,
        "archive_url": "",
        "url_domain_observed": False,
        "error": "",
    }
    if not path.exists():
        return result

    result["file_size"] = path.stat().st_size
    try:
        streamed = _stream_first_html_part(path)
        html = streamed["html"]

        if not html:
            result["parse_status"] = "no_html"
            return result

        parser = EvidenceHTMLParser(list_limit=list_limit)
        parser.feed(html)
        visible_text = " ".join(parser.text_parts)
        link_domains = [domain_from_url(link.get("href", "")) for link in parser.links]
        page_domain = domain_from_url(page_url)
        observed_blob = " ".join(
            [
                visible_text,
                " ".join(parser.title_parts),
                " ".join(link.get("href", "") for link in parser.links),
                streamed["archive_url"],
            ]
        ).lower()

        result.update(
            {
                "parse_status": "ok",
                "title": " ".join(parser.title_parts)[:500],
                "meta": parser.meta,
                "visible_text": visible_text[:text_limit],
                "visible_text_length": len(visible_text),
                "links": parser.links[:list_limit],
                "link_count": len(parser.links),
                "forms": parser.forms[:list_limit],
                "form_count": len(parser.forms),
                "buttons": parser.buttons[:list_limit],
                "button_count": len(parser.buttons),
                "external_domains": sorted({d for d in link_domains if d and d != page_domain})[:list_limit],
                "html_length": len(html),
                "html_truncated": streamed["truncated"],
                "archive_url": streamed["archive_url"],
                "content_types": {streamed["content_type"]: 1} if streamed["content_type"] else {},
                "resource_count": None,
                "url_domain_observed": bool(page_domain and page_domain in observed_blob),
            }
        )
    except Exception as exc:  # noqa: BLE001 - evidence extraction should not abort a full dataset run.
        result["parse_status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def boxes_from_manual_json(obj: dict[str, Any]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    annotations = obj.get("annotations")
    if not isinstance(annotations, list):
        return boxes
    for i, ann in enumerate(annotations, start=1):
        center = ann.get("center") if isinstance(ann, dict) else None
        size = ann.get("size") if isinstance(ann, dict) else None
        if not isinstance(center, dict) or not isinstance(size, dict):
            continue
        try:
            cx = float(center["x"])
            cy = float(center["y"])
            width = float(size["width"])
            height = float(size["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        boxes.append(
            {
                "box_id": i,
                "xyxy": xyxy_from_center(cx, cy, width, height),
                "reason": clean_text(ann.get("description")),
                "box_label": clean_text(ann.get("label")),
                "source": "human_json",
            }
        )
    return boxes


def rows_from_sheet(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet_name]
        rows = worksheet.iter_rows(values_only=True)
        headers = [clean_text(cell) for cell in next(rows)]
        parsed_rows: list[dict[str, Any]] = []
        for values in rows:
            row = {headers[i]: values[i] if i < len(values) else None for i in range(len(headers))}
            identity = row.get("unit_id") if sheet_name == SCAM_SHEET else row.get("ID")
            if not clean_text(identity) and not clean_text(row.get("URL")):
                continue
            parsed_rows.append(row)
        return parsed_rows
    finally:
        workbook.close()


def index_files(folder: Path, suffix: str) -> dict[str, Path]:
    if not folder.exists():
        return {}
    return {path.stem: path for path in folder.glob(f"*{suffix}") if path.is_file()}


def build_benign_record(
    row: dict[str, Any],
    image_path: Path | None,
    mhtml_path: Path | None,
    ai_path: Path | None,
) -> dict[str, Any]:
    sample_id = str(as_int(row.get("ID")) or clean_text(row.get("ID")))
    _ = ai_path
    return {
        "sample_id": f"benign_{sample_id}",
        "raw_id": sample_id,
        "binary": "benign",
        "scam_type": "none",
        "label_source": "benign_sheet",
        "url": clean_text(row.get("URL")),
        "title": clean_text(row.get("TITLE")),
        "language": "",
        "description": "",
        "reasoning": "",
        "source": "benign",
        "tld": "",
        "image_path": path_to_text(image_path),
        "image_size": read_png_size(image_path),
        "mhtml_path": path_to_text(mhtml_path),
        "manual_json_path": "",
        "ai_json_path": "",
        "ai_model": "",
        "is_gold": False,
        "is_weak": False,
        "is_ai_assisted": False,
        "is_benign": True,
        "split": "all",
        "cluster_id": "",
        "boxes": [],
    }


def build_gold_record(
    row: dict[str, Any] | None,
    manual_path: Path,
    image_path: Path | None,
    mhtml_path: Path | None,
) -> dict[str, Any]:
    """Build one authoritative scam record from its manual Gold JSON."""

    row = row or {}
    manual_obj = load_json(manual_path)
    unit_id = manual_path.stem
    raw_label = clean_text(manual_obj.get("label"))
    return {
        "sample_id": unit_id,
        "raw_id": clean_text(row.get("base_id")) or unit_id.split("_", 1)[0],
        "gold_index": as_int(row.get("gold_index")),
        "workbook_id": as_int(row.get("workbook_id")),
        "workbook_row": as_int(row.get("workbook_row")),
        "binary": "scam",
        "scam_type": canonicalize_scam_type(raw_label),
        "scam_type_raw": raw_label,
        "label_source": "gold_manual_json",
        "url": clean_text(row.get("URL")),
        "title": clean_text(row.get("TITLE")),
        "language": clean_text(manual_obj.get("language")),
        "description": clean_text(manual_obj.get("description")),
        "reasoning": "",
        "source": "gold_manual_annotation",
        "tld": "",
        "image_path": path_to_text(image_path),
        "image_size": read_png_size(image_path),
        "mhtml_path": path_to_text(mhtml_path),
        "manual_json_path": str(manual_path),
        "ai_json_path": "",
        "ai_model": "",
        "is_gold": True,
        "is_weak": False,
        "is_ai_assisted": False,
        "is_benign": False,
        "split": "all",
        "cluster_id": "",
        "boxes": boxes_from_manual_json(manual_obj),
    }


def flatten_record(record: dict[str, Any], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    flat = {k: v for k, v in record.items() if k != "boxes"}
    flat["box_count"] = len(record.get("boxes") or [])
    flat["mhtml_parse_status"] = evidence.get("parse_status", "") if evidence else ""
    flat["mhtml_title"] = evidence.get("title", "") if evidence else ""
    flat["mhtml_url_domain_observed"] = evidence.get("url_domain_observed", "") if evidence else ""
    return flat


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _unit_sort_key(value: str) -> tuple[int, int, str]:
    base, _, suffix = value.partition("_")
    return (int(base) if base.isdigit() else sys.maxsize, int(suffix) if suffix.isdigit() else 0, value)


def build_records(dataset_root: Path, old_manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the frozen 3,416 Gold + 2,242 benign manifest from real files."""

    workbook_path = dataset_root / "总表_0325.xlsx"
    _ = old_manifest_path  # Kept for CLI compatibility; prior synthetic splits are intentionally ignored.

    scam_root = dataset_root / "dataset_scam_gold"
    scam_image = index_files(scam_root / "截图_scam", ".png")
    scam_mhtml = index_files(scam_root / "mhtml_scam", ".mhtml")
    scam_manual = index_files(scam_root / "人工标注结果_scam", ".json")
    benign_image = index_files(dataset_root / "dataset_benign" / "截图_benign", ".png")
    benign_mhtml = index_files(dataset_root / "dataset_benign" / "mhtml_benign", ".mhtml")

    records: list[dict[str, Any]] = []
    scam_rows = rows_from_sheet(workbook_path, SCAM_SHEET)
    scam_row_by_unit = {clean_text(row.get("unit_id")): row for row in scam_rows}
    for unit_id in sorted(scam_manual, key=_unit_sort_key):
        records.append(
            build_gold_record(
                scam_row_by_unit.get(unit_id),
                scam_manual[unit_id],
                scam_image.get(unit_id),
                scam_mhtml.get(unit_id),
            )
        )

    benign_rows = rows_from_sheet(workbook_path, BENIGN_SHEET)
    benign_row_by_id = {
        str(as_int(row.get("ID")) or clean_text(row.get("ID"))): row
        for row in benign_rows
    }
    benign_units = sorted(set(benign_image) | set(benign_mhtml), key=_unit_sort_key)
    for sample_id in benign_units:
        row = benign_row_by_id.get(sample_id) or {"ID": sample_id}
        records.append(
            build_benign_record(
                row,
                benign_image.get(sample_id),
                benign_mhtml.get(sample_id),
                None,
            )
        )

    inventory = {
        "gold_workbook_rows": len(scam_rows),
        "benign_workbook_rows": len(benign_rows),
        "gold_png_files": len(scam_image),
        "gold_mhtml_files": len(scam_mhtml),
        "gold_manual_json_files": len(scam_manual),
        "benign_png_files": len(benign_image),
        "benign_mhtml_files": len(benign_mhtml),
        "gold_json_without_workbook_row": sum(unit not in scam_row_by_unit for unit in scam_manual),
        "gold_json_without_png": sum(unit not in scam_image for unit in scam_manual),
        "gold_json_without_mhtml": sum(unit not in scam_mhtml for unit in scam_manual),
        "benign_files_without_workbook_row": sum(unit not in benign_row_by_id for unit in benign_units),
        "benign_without_png": sum(unit not in benign_image for unit in benign_units),
        "benign_without_mhtml": sum(unit not in benign_mhtml for unit in benign_units),
    }
    return records, inventory


def make_boxes_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for i, box in enumerate(record.get("boxes") or [], start=1):
            xyxy = box.get("xyxy") or ["", "", "", ""]
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "binary": record["binary"],
                    "box_id": box.get("box_id") or i,
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                    "reason": clean_text(box.get("reason")),
                    "box_label": clean_text(box.get("box_label")),
                    "source": clean_text(box.get("source")),
                }
            )
    return rows


def extract_evidence(records: list[dict[str, Any]], max_records: int = 0) -> dict[str, dict[str, Any]]:
    evidence_by_sample: dict[str, dict[str, Any]] = {}
    parsed = 0
    for record in records:
        mhtml_path = clean_text(record.get("mhtml_path"))
        if not mhtml_path:
            continue
        evidence = parse_mhtml_file(Path(mhtml_path), page_url=clean_text(record.get("url")))
        evidence["sample_id"] = record["sample_id"]
        evidence["binary"] = record["binary"]
        evidence["raw_id"] = record["raw_id"]
        evidence_by_sample[record["sample_id"]] = evidence
        parsed += 1
        if parsed % 1000 == 0:
            print(f"parsed {parsed} mhtml files", file=sys.stderr)
        if max_records and parsed >= max_records:
            break
    return evidence_by_sample


def make_demo_manifest(records: list[dict[str, Any]], per_class: int = 20) -> list[dict[str, Any]]:
    scam = [
        r
        for r in records
        if r["binary"] == "scam" and r["is_gold"] and r.get("image_path") and r.get("mhtml_path")
    ][:per_class]
    benign = [
        r
        for r in records
        if r["binary"] == "benign" and r.get("image_path") and r.get("mhtml_path")
    ][:per_class]
    return scam + benign


def summarize_records(records: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence_status = Counter(e.get("parse_status", "") for e in evidence.values())
    gold = [record for record in records if record.get("is_gold")]
    benign = [record for record in records if record["binary"] == "benign"]
    label_distribution = Counter(record.get("scam_type") or "" for record in gold)
    return {
        "total_records": len(records),
        "scam_records": len(gold),
        "benign_records": len(benign),
        "canonical_subtype_count": len(label_distribution),
        "gold_box_count": sum(len(record.get("boxes") or []) for record in gold),
        "empty_gold_label_count": sum(not record.get("scam_type") for record in gold),
        "invalid_gold_label_count": sum(record.get("scam_type") == "__invalid__" for record in gold),
        "empty_gold_description_count": sum(not record.get("description") for record in gold),
        "empty_gold_region_reason_count": sum(
            not box.get("reason") for record in gold for box in record.get("boxes") or []
        ),
        "multiple_label_gold_record_count": sum(
            ";" in clean_text(record.get("scam_type_raw")) for record in gold
        ),
        "duplicate_sample_id_count": len(records) - len({record.get("sample_id") for record in records}),
        "missing_image_size_count": sum(not valid_image_size(record.get("image_size")) for record in records),
        "empty_url_count": sum(not record.get("url") for record in records),
        "empty_title_count": sum(not record.get("title") for record in records),
        "records_with_image": sum(1 for r in records if r.get("image_path")),
        "records_with_mhtml": sum(1 for r in records if r.get("mhtml_path")),
        "scam_records_with_image": sum(1 for r in records if r["binary"] == "scam" and r.get("image_path")),
        "scam_records_with_mhtml": sum(1 for r in records if r["binary"] == "scam" and r.get("mhtml_path")),
        "benign_records_with_image": sum(1 for r in records if r["binary"] == "benign" and r.get("image_path")),
        "benign_records_with_mhtml": sum(1 for r in records if r["binary"] == "benign" and r.get("mhtml_path")),
        "gold_records": sum(1 for r in records if r.get("is_gold")),
        "weak_records": sum(1 for r in records if r.get("is_weak")),
        "ai_assisted_records": sum(1 for r in records if r.get("is_ai_assisted")),
        "box_rows": sum(len(r.get("boxes") or []) for r in records),
        "gold_label_distribution": dict(label_distribution.most_common()),
        "split_distribution": Counter(r.get("split") for r in records),
        "mhtml_parse_status": dict(evidence_status),
        "mhtml_url_not_observed": sum(
            1 for e in evidence.values() if e.get("parse_status") == "ok" and not e.get("url_domain_observed")
        ),
    }


def validate_frozen_dataset(summary: dict[str, Any], inventory: dict[str, Any]) -> None:
    """Fail fast if the official source files drift from the paper's frozen contract."""

    expected = {
        "total_records": EXPECTED_TOTAL_RECORDS,
        "scam_records": EXPECTED_GOLD_RECORDS,
        "benign_records": EXPECTED_BENIGN_RECORDS,
        "canonical_subtype_count": EXPECTED_CANONICAL_SUBTYPES,
        "gold_box_count": EXPECTED_GOLD_BOXES,
        "empty_gold_label_count": 0,
        "invalid_gold_label_count": 0,
        "empty_gold_description_count": 0,
        "empty_gold_region_reason_count": 0,
        "multiple_label_gold_record_count": 0,
        "duplicate_sample_id_count": 0,
        "missing_image_size_count": 0,
        "empty_url_count": 0,
        "empty_title_count": 0,
    }
    errors = [
        f"{key}: expected {value}, observed {summary.get(key)}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    for key in [
        "gold_json_without_workbook_row",
        "gold_json_without_png",
        "gold_json_without_mhtml",
        "benign_files_without_workbook_row",
        "benign_without_png",
        "benign_without_mhtml",
    ]:
        if inventory.get(key) != 0:
            errors.append(f"{key}: expected 0, observed {inventory.get(key)}")
    if errors:
        raise ValueError("Frozen ScamWeb-V dataset contract failed:\n- " + "\n- ".join(errors))


def write_reports(
    reports_dir: Path,
    inventory: dict[str, Any],
    summary: dict[str, Any],
    outputs: dict[str, Path],
    records: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    no_mhtml = [r for r in records if not r.get("mhtml_path")][:10]
    mhtml_not_observed = [
        e
        for e in evidence.values()
        if e.get("parse_status") == "ok" and not e.get("url_domain_observed")
    ][:10]
    dataset_report = reports_dir / "dataset_audit.md"
    mhtml_report = reports_dir / "mhtml_audit.md"

    dataset_report.write_text(
        "\n".join(
            [
                "# ScamWeb-V Dataset Audit",
                "",
                "## Inventory",
                *[f"- {key}: {value}" for key, value in inventory.items()],
                "",
                "## Record Summary",
                *[
                    f"- {key}: {value}"
                    for key, value in summary.items()
                    if key
                    not in {
                        "gold_label_distribution",
                        "split_distribution",
                    }
                ],
                "",
                "## Split Distribution",
                *[f"- {key}: {value}" for key, value in summary["split_distribution"].items()],
                "",
                "## Gold Scam Labels",
                *[f"- {label or '(empty)'}: {count}" for label, count in summary["gold_label_distribution"].items()],
                "",
                "## Outputs",
                *[f"- {name}: `{path}`" for name, path in outputs.items()],
                "",
                "## Benign Policy",
                "- Benign records are encoded as `binary=benign`, `scam_type=none`, `boxes=[]`, `split=all`.",
                "- Manual Gold JSON is authoritative for subtype, page description, language, and regions; Excel only supplies URL, title, and indices.",
                "",
                "## Sample Records Missing MHTML",
                *[f"- {r['sample_id']} {r['url']}" for r in no_mhtml],
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    mhtml_report.write_text(
        "\n".join(
            [
                "# ScamWeb-V MHTML Evidence Audit",
                "",
                "## Parse Status",
                *[f"- {key}: {value}" for key, value in summary["mhtml_parse_status"].items()],
                "",
                "## Evidence Fields",
                "- `title`, `meta`, `visible_text`, `links`, `forms`, `buttons`, `external_domains`, `content_types`, `resource_count`, `url_domain_observed`.",
                "",
                "## URL Domain Not Observed In Parsed MHTML",
                "This is a quality warning, not automatic exclusion. Some legitimate pages do not print their domain in visible text.",
                *[
                    f"- {e['sample_id']} url={e.get('page_url','')} title={clean_text(e.get('title'))[:120]}"
                    for e in mhtml_not_observed
                ],
                "",
                "## Known Interpretation",
                "- MHTML is treated as persistent webpage evidence and transformed into compressed structured fields for ScamMark.",
                "- The raw MHTML file is never modified by this pipeline.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    old_manifest = Path(args.old_manifest)

    records, inventory = build_records(dataset_root, old_manifest)
    evidence = extract_evidence(records, max_records=args.max_evidence)
    boxes_rows = make_boxes_rows(records)
    demo_manifest = make_demo_manifest(records, per_class=args.demo_per_class)
    summary = summarize_records(records, evidence)
    validate_frozen_dataset(summary, inventory)

    data_csv = output_dir / "data.csv"
    boxes_csv = output_dir / "boxes.csv"
    boxes_jsonl = output_dir / "boxes.jsonl"
    manifest_jsonl = output_dir / "scamweb_v_manifest.jsonl"
    evidence_jsonl = output_dir / "web_evidence.jsonl"
    demo_jsonl = output_dir / "demo_manifest.jsonl"
    summary_json = output_dir / "summary.json"

    evidence_rows = list(evidence.values())
    flat_rows = [flatten_record(record, evidence.get(record["sample_id"])) for record in records]
    data_fields = list(flat_rows[0].keys()) if flat_rows else []
    write_csv(data_csv, flat_rows, data_fields)
    write_csv(
        boxes_csv,
        boxes_rows,
        ["sample_id", "binary", "box_id", "x1", "y1", "x2", "y2", "reason", "box_label", "source"],
    )
    write_jsonl(boxes_jsonl, boxes_rows)
    write_jsonl(manifest_jsonl, records)
    write_jsonl(evidence_jsonl, evidence_rows)
    write_jsonl(demo_jsonl, demo_manifest)
    summary_json.write_text(json.dumps({"inventory": inventory, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = {
        "data_csv": data_csv,
        "boxes_csv": boxes_csv,
        "boxes_jsonl": boxes_jsonl,
        "manifest_jsonl": manifest_jsonl,
        "web_evidence_jsonl": evidence_jsonl,
        "demo_manifest_jsonl": demo_jsonl,
        "summary_json": summary_json,
    }
    write_reports(reports_dir, inventory, summary, outputs, records, evidence)

    print(json.dumps({"inventory": inventory, "summary": summary, "outputs": {k: str(v) for k, v in outputs.items()}}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ScamWeb-V dataset indices and MHTML evidence.")
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("SCAMWEB_DATASET_ROOT", "data/scamweb_dataset"),
        help="Raw ScamWeb-V root (or set SCAMWEB_DATASET_ROOT).",
    )
    parser.add_argument("--output-dir", default="outputs/dataset")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--old-manifest", default="src/benchmark/scamweb_v_manifest.jsonl")
    parser.add_argument("--max-evidence", type=int, default=0, help="Limit parsed MHTML files; 0 means all.")
    parser.add_argument("--demo-per-class", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
