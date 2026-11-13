"""Data and evidence loading helpers for the ScamMark demo."""

from __future__ import annotations

import json
import os
import re
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class EvidenceHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self.buttons: list[str] = []
        self.forms: list[str] = []
        self.meta: dict[str, str] = {}
        self._in_title = False
        self._in_button = False
        self._button_parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a" and attrs_dict.get("href"):
            self.links.append(attrs_dict["href"])
        elif tag == "button":
            self._in_button = True
            self._button_parts = []
        elif tag in {"input", "textarea", "select"}:
            field = " ".join(
                part
                for part in [
                    attrs_dict.get("type"),
                    attrs_dict.get("name"),
                    attrs_dict.get("placeholder"),
                    attrs_dict.get("value"),
                ]
                if part
            )
            if field:
                self.forms.append(field)
        elif tag == "meta":
            name = attrs_dict.get("name") or attrs_dict.get("property")
            content = attrs_dict.get("content")
            if name and content:
                self.meta[name[:80]] = content[:500]

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == "button":
            value = _clean_text(" ".join(self._button_parts))
            if value:
                self.buttons.append(value)
            self._in_button = False
            self._button_parts = []

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = _clean_text(data)
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        if self._in_button:
            self._button_parts.append(text)
        self.text_parts.append(text)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_html_evidence(html: str) -> dict:
    parser = EvidenceHTMLParser()
    parser.feed(html or "")
    visible_text = _clean_text(" ".join(parser.text_parts))
    title = _clean_text(" ".join(parser.title_parts))
    return {
        "title": title,
        "visible_text": visible_text[:8000],
        "buttons": _dedupe(parser.buttons)[:40],
        "forms": _dedupe(parser.forms)[:40],
        "links": _dedupe(parser.links)[:100],
        "meta": parser.meta,
        "external_domains": _extract_domains(parser.links)[:80],
    }


def parse_mhtml_evidence(path: str | Path | None) -> dict:
    if not path:
        return _empty_evidence("missing_path")
    mhtml_path = Path(path)
    if not mhtml_path.exists():
        return _empty_evidence("missing_file")

    try:
        raw = mhtml_path.read_bytes()
        message = BytesParser(policy=policy.default).parsebytes(raw)
        html_candidates: list[tuple[int, str, str]] = []
        source_url = ""
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == "text/html":
                    payload = part.get_payload(decode=True) or b""
                    charset = part.get_content_charset() or "utf-8"
                    html = payload.decode(charset, errors="replace")
                    content_location = part.get("Content-Location", "")
                    html_candidates.append((len(html), html, content_location))
        else:
            payload = message.get_payload(decode=True) or raw
            charset = message.get_content_charset() or "utf-8"
            html_candidates.append((len(payload), payload.decode(charset, errors="replace"), message.get("Content-Location", "")))

        if not html_candidates:
            text = raw.decode("utf-8", errors="replace")
            evidence = extract_html_evidence(text)
            evidence["parse_status"] = "fallback_text"
            evidence["source_url"] = ""
            return evidence

        _, html, source_url = max(html_candidates, key=lambda item: item[0])
        evidence = extract_html_evidence(html)
        evidence["parse_status"] = "ok"
        evidence["source_url"] = source_url or _first_http_link(evidence.get("links", []))
        evidence["mhtml_path"] = str(mhtml_path)
        return evidence
    except Exception as exc:  # robust demo loader; report parse status instead of crashing
        evidence = _empty_evidence("parse_error")
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["mhtml_path"] = str(mhtml_path)
        return evidence


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_samples(manifest_path: str | Path, web_evidence_path: str | Path | None = None, limit: int | None = None) -> list[dict]:
    samples = [_normalize_manifest_record(record) for record in load_jsonl(manifest_path)]
    evidence_by_id = {}
    if web_evidence_path and Path(web_evidence_path).exists():
        for evidence in load_jsonl(web_evidence_path):
            key = str(evidence.get("sample_id") or evidence.get("id") or "")
            if key:
                evidence_by_id[key] = evidence

    merged = []
    for sample in samples:
        key = str(sample.get("sample_id") or sample.get("id") or "")
        evidence = evidence_by_id.get(key)
        if evidence:
            _merge_evidence(sample, evidence)
        merged.append(sample)
        if limit and len(merged) >= limit:
            break
    return merged


def read_dataset_root(default: str = "data/scamweb_dataset") -> str:
    environment_value = os.getenv("SCAMWEB_DATASET_ROOT", "").strip()
    if environment_value:
        return environment_value
    path_file = Path("src/data/datasets_path.txt")
    if path_file.exists():
        value = path_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return default


def _normalize_ai_boxes(boxes: list[Any], image_path: Path | None = None) -> list[dict]:
    """Convert raw VLM `bbox` outputs into the shared xyxy pixel schema.

    Human/workbook annotations are already pixel coordinates. Raw model outputs
    in this dataset commonly use a 0..1000 coordinate frame, so this conversion
    is only used for the raw-AI demo builder and is marked as AI-derived.
    """

    image_size = _image_size(image_path)
    normalized = []
    for i, box in enumerate(boxes, start=1):
        if not isinstance(box, dict):
            continue
        coords = box.get("xyxy") or box.get("bbox")
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in coords]
        except (TypeError, ValueError):
            continue
        coordinate_frame = "pixel"
        if image_size and max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000:
            width, height = image_size
            x1, x2 = x1 / 1000 * width, x2 / 1000 * width
            y1, y2 = y1 / 1000 * height, y2 / 1000 * height
            coordinate_frame = "normalized_1000"
        if image_size:
            width, height = image_size
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized.append(
            {
                "box_id": i,
                "xyxy": [x1, y1, x2, y2],
                "reason": box.get("reason", ""),
                "box_label": box.get("label") or box.get("box_label", ""),
                "source": "ai_json",
                "coordinate_frame": coordinate_frame,
            }
        )
    return normalized


def _image_size(path: Path | None) -> tuple[int, int] | None:
    if not path or not path.exists():
        return None
    try:
        from scammark.visual_analysis import get_image_size

        width, height = get_image_size(path)
    except Exception:
        return None
    if width and height:
        return int(width), int(height)
    return None


def _normalize_manifest_record(record: dict) -> dict:
    sample = dict(record)
    sample["sample_id"] = str(sample.get("sample_id") or sample.get("id") or sample.get("raw_id") or "")
    binary = str(sample.get("binary") or "").lower()
    if not binary:
        binary = "benign" if str(sample.get("scam_type") or sample.get("label") or "").lower() == "none" else "scam"
    sample["binary"] = "scam" if binary == "scam" else "benign"
    if not sample.get("scam_type"):
        sample["scam_type"] = sample.get("label") or ("none" if sample["binary"] == "benign" else "Unknown Scam")
    if sample["binary"] == "benign":
        sample["scam_type"] = "none"
    sample.setdefault("boxes", [])
    sample.setdefault("text", "")
    return sample


def _merge_evidence(sample: dict, evidence: dict) -> None:
    if evidence.get("title") and not sample.get("title"):
        sample["title"] = evidence["title"]
    for key in [
        "visible_text",
        "buttons",
        "forms",
        "links",
        "meta",
        "external_domains",
        "parse_status",
        "archive_url",
        "source_url",
        "url_domain_observed",
        "html_truncated",
    ]:
        if key in evidence:
            sample[key] = evidence[key]
    if not sample.get("text"):
        sample["text"] = evidence.get("visible_text", "")


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        clean = _clean_text(value)
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def _extract_domains(links: list[str]) -> list[str]:
    domains = []
    for link in links:
        host = urlparse(link).hostname
        if host:
            domains.append(host.lower())
    return _dedupe(domains)


def _first_http_link(links: list[str]) -> str:
    for link in links:
        if str(link).startswith(("http://", "https://")):
            return str(link)
    return ""


def _empty_evidence(status: str) -> dict:
    return {
        "title": "",
        "visible_text": "",
        "buttons": [],
        "forms": [],
        "links": [],
        "meta": {},
        "external_domains": [],
        "parse_status": status,
        "source_url": "",
    }
