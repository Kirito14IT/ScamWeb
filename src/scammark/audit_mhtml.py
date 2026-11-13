"""Audit, conservatively gate, and freeze ScamWeb-V MHTML evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .evidence_loader import write_jsonl


AUDIT_VERSION = "mhtml-policy-v2"
FINAL_STATUSES = {"valid", "valid_redirect", "mhtml_mismatch", "uncertain", "missing"}
ERROR_PAGE_PATTERNS = (
    "domain for sale",
    "domain is parked",
    "this domain has expired",
    "404 not found",
    "server unavailable",
    "bad gateway",
    "access denied",
    "wayback machine",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_box_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for box_id, box in enumerate(record.get("boxes") or [], start=1):
            coordinates = box.get("xyxy") if isinstance(box, dict) else None
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                raise ValueError(f"Invalid frozen box for sample {record.get('sample_id')}")
            rows.append(
                {
                    "sample_id": str(record.get("sample_id") or record.get("id") or ""),
                    "binary": str(record.get("binary") or ""),
                    "box_id": box_id,
                    "xyxy": [float(value) for value in coordinates],
                    "reason": str(box.get("reason") or ""),
                    "box_label": str(box.get("box_label") or record.get("scam_type") or ""),
                    "source": str(box.get("source") or "human_json"),
                }
            )
    return rows


def finalize_box_asset(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "scamweb_v_manifest.jsonl"
    version_path = output_dir / "dataset_version.json"
    records = load_jsonl(manifest_path)
    boxes_path = output_dir / "boxes.jsonl"
    rows = build_box_rows(records)
    write_jsonl(boxes_path, rows)
    result = {"box_count": len(rows), "boxes_sha256": _file_sha256(boxes_path)}
    if version_path.exists():
        version = json.loads(version_path.read_text(encoding="utf-8"))
        version["gold_box_count"] = len(rows)
        version.setdefault("sha256", {})["boxes"] = result["boxes_sha256"]
        version_path.write_text(json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").lower()).strip()


def title_similarity(left: Any, right: Any) -> float:
    left_text = _normalized_text(left)
    right_text = _normalized_text(right)
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _host(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    return (urlparse(text).hostname or "").lower().strip(".")


def _registrable_domain(value: Any) -> str:
    host = _host(value)
    if not host:
        return ""
    try:
        import tldextract

        extracted = tldextract.TLDExtract(suffix_list_urls=())(host)
        return ".".join(part for part in [extracted.domain, extracted.suffix] if part)
    except Exception:
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host


def classify_mhtml_record(manifest: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    parse_status = str(evidence.get("parse_status") or "")
    if parse_status != "ok":
        return {
            "status": "missing",
            "confidence": 1.0,
            "audit_source": "deterministic_rule",
            "reasons": [f"parse_status={parse_status or 'missing'}"],
        }
    if evidence.get("url_domain_observed") is True:
        return {
            "status": "valid",
            "confidence": 1.0,
            "audit_source": "deterministic_rule",
            "reasons": ["workbook URL domain is observed in the archive evidence"],
        }

    original_domain = _registrable_domain(manifest.get("url"))
    archive_domain = _registrable_domain(evidence.get("archive_url"))
    similarity = title_similarity(manifest.get("title"), evidence.get("title"))
    evidence_blob = _normalized_text(
        " ".join([str(evidence.get("title") or ""), str(evidence.get("visible_text") or "")[:3000]])
    )
    errors = [pattern for pattern in ERROR_PAGE_PATTERNS if pattern in evidence_blob]
    if errors or (similarity < 0.30 and original_domain and archive_domain and original_domain != archive_domain):
        return {
            "status": "mhtml_mismatch",
            "confidence": 0.95 if errors else 0.85,
            "audit_source": "deterministic_rule",
            "reasons": ([f"error-page signature: {errors[0]}"] if errors else [])
            + [f"title_similarity={similarity:.3f}", f"domain={original_domain}->{archive_domain}"],
        }
    if similarity >= 0.70 and archive_domain and archive_domain != original_domain:
        return {
            "status": "valid_redirect",
            "confidence": min(0.99, similarity),
            "audit_source": "deterministic_rule",
            "reasons": [f"high title agreement after domain change ({similarity:.3f})", f"domain={original_domain}->{archive_domain}"],
        }
    return {
        "status": "uncertain",
        "confidence": max(0.0, min(0.69, similarity)),
        "audit_source": "deterministic_rule",
        "reasons": [f"insufficient redirect/content evidence; title_similarity={similarity:.3f}"],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recapture_one(page, cdp, manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    sample_id = str(manifest.get("sample_id") or manifest.get("id"))
    url = str(manifest.get("url") or "")
    if "://" not in url:
        url = "https://" + url
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "attempted": True,
        "accepted": False,
        "final_url": "",
        "title_similarity": 0.0,
        "phash_distance": None,
        "mhtml_path": "",
        "screenshot_path": "",
        "error": "",
    }
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
        result["final_url"] = page.url
        current_title = page.title()
        result["title_similarity"] = title_similarity(manifest.get("title"), current_title)
        screenshot_path = sample_dir / "recapture.png"
        page.screenshot(path=str(screenshot_path), full_page=True, timeout=30_000)
        snapshot = cdp.send("Page.captureSnapshot", {"format": "mhtml"})
        mhtml_path = sample_dir / "recapture.mhtml"
        mhtml_path.write_text(snapshot["data"], encoding="utf-8")
        result["mhtml_path"] = str(mhtml_path)
        result["screenshot_path"] = str(screenshot_path)
        try:
            import imagehash
            from PIL import Image

            with Image.open(manifest["image_path"]) as historical, Image.open(screenshot_path) as current:
                result["phash_distance"] = int(imagehash.phash(historical) - imagehash.phash(current))
        except Exception as exc:
            result["error"] = f"phash:{type(exc).__name__}:{str(exc)[:200]}"
        original_domain = _registrable_domain(url)
        final_domain = _registrable_domain(result["final_url"])
        redirected = bool(final_domain and final_domain != original_domain)
        domain_related = bool(original_domain and final_domain and (original_domain == final_domain or redirected))
        result["accepted"] = bool(
            response
            and response.ok
            and domain_related
            and result["title_similarity"] >= 0.70
            and result["phash_distance"] is not None
            and result["phash_distance"] <= 12
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}:{str(exc)[:500]}"
    return result


def recapture_missing(records: list[dict[str, Any]], output_dir: Path) -> dict[str, dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is required for MHTML recapture") from exc
    executable_candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    executable = next((path for path in executable_candidates if path.exists()), None)
    if executable is None:
        raise FileNotFoundError("Chrome or Edge executable was not found")
    results: dict[str, dict[str, Any]] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(executable),
            headless=True,
            args=["--disable-extensions", "--disable-background-networking", "--no-first-run"],
        )
        context = browser.new_context(ignore_https_errors=True, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        cdp = context.new_cdp_session(page)
        for record in records:
            sample_id = str(record.get("sample_id") or record.get("id"))
            results[sample_id] = _recapture_one(page, cdp, record, output_dir)
        browser.close()
    return results


def load_recaptured_evidence(
    recapture_result: dict[str, Any],
    manifest: dict[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    """Parse an accepted recapture instead of relabeling stale evidence as valid."""
    from data.scamweb_dataset import parse_mhtml_file

    evidence = parse_mhtml_file(
        Path(str(recapture_result["mhtml_path"])),
        page_url=str(manifest.get("url") or recapture_result.get("final_url") or ""),
    )
    if evidence.get("parse_status") != "ok":
        raise ValueError(f"Accepted recapture could not be parsed: {evidence.get('parse_status')}")
    evidence.update(
        {
            "sample_id": sample_id,
            "binary": manifest.get("binary"),
            "raw_id": manifest.get("raw_id"),
            "archive_url": evidence.get("archive_url") or recapture_result.get("final_url") or "",
        }
    )
    return evidence


def _qwen_audit(
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    client,
    image_cache_dir: Path,
) -> dict[str, Any]:
    from .image_preprocessing import ImagePolicy, prepare_image_bundle

    identity_payload = {
        "original_url": manifest.get("url"),
        "workbook_title": manifest.get("title"),
        "archive_url": evidence.get("archive_url"),
        "mhtml_title": evidence.get("title"),
        "mhtml_text_excerpt": str(evidence.get("visible_text") or "")[:4000],
    }
    identity_prompt = (
        "Audit only whether the archived MHTML belongs to the same webpage as the submitted screenshot and original "
        "URL/title. Return strict JSON with status=valid_redirect|mhtml_mismatch|uncertain, confidence from 0 to 1, "
        "and reasons as a list. A legitimate redirect may change the domain while preserving page identity. Do not "
        "classify scams and do not infer or request annotation labels. Evidence:\n"
        + json.dumps(identity_payload, ensure_ascii=False)
    )
    bundle = prepare_image_bundle(
        manifest.get("image_path"),
        image_cache_dir,
        ImagePolicy(total_pixels=1_048_576, overview_pixels=1_048_576, max_tiles=2),
    )
    gold_payload = {
        "original_url": manifest.get("url"),
        "workbook_title": manifest.get("title"),
        "annotation_label": manifest.get("scam_type") if manifest.get("binary") == "scam" else "benign",
        "annotation_regions": [
            {"reason": box.get("reason", ""), "xyxy": box.get("xyxy")}
            for box in (manifest.get("boxes") or [])[:4]
        ],
    }
    conflict_prompt = (
        "Independently check whether the submitted screenshot clearly contradicts its annotation. Ignore archived "
        "MHTML identity and do not change any archive-validity status. Return strict JSON with "
        "gold_input_conflict=true|false, confidence from 0 to 1, and reasons as a list. Mark true only for a clear "
        "screenshot-versus-annotation contradiction, not for weak evidence or an unrelated archive. Evidence:\n"
        + json.dumps(gold_payload, ensure_ascii=False)
    )
    if hasattr(client, "complete_json_with_metadata"):
        identity_response, identity_metadata = client.complete_json_with_metadata(
            stage="mhtml_identity_audit",
            prompt=identity_prompt,
            sample={"image_path": manifest.get("image_path")},
            include_image=True,
            max_output_tokens=512,
            image_assets=[bundle.assets[0]],
            max_attempts=4,
        )
        try:
            conflict_response, conflict_metadata = client.complete_json_with_metadata(
                stage="gold_conflict_audit",
                prompt=conflict_prompt,
                sample={"image_path": manifest.get("image_path")},
                include_image=True,
                max_output_tokens=512,
                image_assets=[bundle.assets[0]],
                max_attempts=4,
            )
            conflict_check_status = "checked"
        except Exception as exc:  # noqa: BLE001 - identity status remains usable without this audit.
            conflict_response = {
                "gold_input_conflict": False,
                "confidence": 0.0,
                "reasons": [f"audit_failed:{type(exc).__name__}:{str(exc)[:300]}"],
            }
            conflict_metadata = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                **dict(getattr(exc, "request_metadata", {}) or {}),
            }
            conflict_check_status = "error"
    else:
        identity_response = client.complete_json(
            stage="mhtml_identity_audit",
            prompt=identity_prompt,
            sample={"image_path": manifest.get("image_path")},
            include_image=True,
        )
        try:
            conflict_response = client.complete_json(
                stage="gold_conflict_audit",
                prompt=conflict_prompt,
                sample={"image_path": manifest.get("image_path")},
                include_image=True,
            )
            conflict_metadata = {"attempt_count": 1}
            conflict_check_status = "checked"
        except Exception as exc:  # noqa: BLE001
            conflict_response = {
                "gold_input_conflict": False,
                "confidence": 0.0,
                "reasons": [f"audit_failed:{type(exc).__name__}:{str(exc)[:300]}"],
            }
            conflict_metadata = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
            conflict_check_status = "error"
        identity_metadata = {"attempt_count": 1}
    status = str(identity_response.get("status") or "uncertain")
    if status not in {"valid_redirect", "mhtml_mismatch", "uncertain"}:
        status = "uncertain"
    return {
        "status": status,
        "confidence": float(identity_response.get("confidence") or 0.0),
        "gold_input_conflict": bool(conflict_response.get("gold_input_conflict")),
        "gold_conflict_check_status": conflict_check_status,
        "gold_conflict_confidence": float(conflict_response.get("confidence") or 0.0),
        "audit_source": "qwen3-vl-flash",
        "reasons": [str(value) for value in (identity_response.get("reasons") or [])][:10],
        "gold_conflict_reasons": [str(value) for value in (conflict_response.get("reasons") or [])][:10],
        "request_metadata": {
            "identity": identity_metadata,
            "gold_conflict": conflict_metadata,
        },
    }


def audit_and_freeze(
    manifest_path: Path,
    evidence_path: Path,
    output_dir: Path,
    *,
    recapture: bool = True,
    use_api_auditor: bool = True,
    audit_call_limit: int = 450,
) -> dict[str, Any]:
    manifest_records = load_jsonl(manifest_path)
    evidence_records = load_jsonl(evidence_path)
    evidence_by_id = {str(record.get("sample_id")): record for record in evidence_records}
    if len(manifest_records) != len(evidence_by_id):
        raise ValueError("Manifest and evidence coverage differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = {
        str(record.get("sample_id")): classify_mhtml_record(record, evidence_by_id[str(record.get("sample_id"))])
        for record in manifest_records
    }

    missing_records = [record for record in manifest_records if initial[str(record.get("sample_id"))]["status"] == "missing"]
    recaptures: dict[str, dict[str, Any]] = {}
    recaptured_evidence: dict[str, dict[str, Any]] = {}
    if recapture and missing_records:
        recaptures = recapture_missing(missing_records, output_dir / "recaptured_mhtml")
        for record in missing_records:
            sample_id = str(record.get("sample_id"))
            result = recaptures.get(sample_id) or {}
            if result.get("accepted"):
                try:
                    recaptured_evidence[sample_id] = load_recaptured_evidence(result, record, sample_id)
                except Exception as exc:
                    result["accepted"] = False
                    result["error"] = f"parse:{type(exc).__name__}:{str(exc)[:300]}"
                    continue
                initial[sample_id] = {
                    "status": "valid_redirect" if _registrable_domain(record.get("url")) != _registrable_domain(result.get("final_url")) else "valid",
                    "confidence": 1.0,
                    "audit_source": "accepted_recapture",
                    "reasons": [
                        f"title_similarity={result.get('title_similarity', 0.0):.3f}",
                        f"phash_distance={result.get('phash_distance')}",
                    ],
                }

    auditor = None
    audit_calls = 0
    if use_api_auditor and any(result["status"] == "uncertain" for result in initial.values()):
        from .model_client import make_model_client

        auditor = make_model_client("openai")
        if hasattr(auditor, "max_api_calls"):
            auditor.max_api_calls = audit_call_limit
    for record in manifest_records:
        sample_id = str(record.get("sample_id"))
        if initial[sample_id]["status"] != "uncertain" or auditor is None or audit_calls >= audit_call_limit:
            continue
        audit_calls += 1
        try:
            initial[sample_id] = _qwen_audit(
                record,
                evidence_by_id[sample_id],
                auditor,
                output_dir / "audit_image_cache",
            )
        except Exception as exc:
            initial[sample_id] = {
                "status": "uncertain",
                "confidence": 0.0,
                "gold_input_conflict": False,
                "gold_conflict_check_status": "not_run_identity_error",
                "audit_source": "qwen3_audit_failed",
                "reasons": [f"{type(exc).__name__}: {str(exc)[:300]}"],
                "request_metadata": dict(getattr(exc, "request_metadata", {}) or {}),
            }
        print(
            f"MHTML API audit {audit_calls}: {sample_id} -> {initial[sample_id]['status']}",
            flush=True,
        )

    frozen_manifest = []
    frozen_evidence = []
    audit_rows = []
    for record in manifest_records:
        sample_id = str(record.get("sample_id"))
        evidence = dict(evidence_by_id[sample_id])
        audit = dict(initial[sample_id])
        status = audit["status"] if audit["status"] in FINAL_STATUSES else "uncertain"
        recapture_result = recaptures.get(sample_id) or {}
        manifest_record = dict(record)
        manifest_record.update(
            {
                "mhtml_audit_status": status,
                "mhtml_valid": status in {"valid", "valid_redirect"},
                "mhtml_used": status in {"valid", "valid_redirect"},
                "gold_input_conflict": bool(audit.get("gold_input_conflict", False)),
                "gold_conflict_check_status": str(
                    audit.get("gold_conflict_check_status") or "not_requested"
                ),
                "input_policy_version": AUDIT_VERSION,
            }
        )
        if recapture_result.get("accepted"):
            manifest_record["mhtml_path"] = recapture_result["mhtml_path"]
            evidence = recaptured_evidence[sample_id]
        evidence.update(
            {
                "mhtml_audit_status": status,
                "mhtml_valid": status in {"valid", "valid_redirect"},
                "mhtml_used": status in {"valid", "valid_redirect"},
                "input_policy_version": AUDIT_VERSION,
                "gold_conflict_check_status": manifest_record["gold_conflict_check_status"],
            }
        )
        frozen_manifest.append(manifest_record)
        frozen_evidence.append(evidence)
        audit_rows.append(
            {
                "sample_id": sample_id,
                "status": status,
                "mhtml_valid": manifest_record["mhtml_valid"],
                "mhtml_used": manifest_record["mhtml_used"],
                "original_url": record.get("url"),
                "archive_url": evidence.get("archive_url"),
                "final_url": recapture_result.get("final_url", ""),
                "gold_input_conflict": manifest_record["gold_input_conflict"],
                "gold_conflict_check_status": manifest_record["gold_conflict_check_status"],
                "audit_source": audit.get("audit_source"),
                "confidence": audit.get("confidence", 0.0),
                "reasons": audit.get("reasons", []),
                "gold_conflict_confidence": audit.get("gold_conflict_confidence", 0.0),
                "gold_conflict_reasons": audit.get("gold_conflict_reasons", []),
                "request_metadata": audit.get("request_metadata", {}),
            }
        )

    frozen_manifest_path = output_dir / "scamweb_v_manifest.jsonl"
    frozen_evidence_path = output_dir / "web_evidence.jsonl"
    audit_path = output_dir / "mhtml_audit.jsonl"
    recapture_path = output_dir / "mhtml_recapture.jsonl"
    write_jsonl(frozen_manifest_path, frozen_manifest)
    write_jsonl(frozen_evidence_path, frozen_evidence)
    write_jsonl(audit_path, audit_rows)
    write_jsonl(
        recapture_path,
        [
            {"sample_id": sample_id, **result}
            for sample_id, result in sorted(recaptures.items())
        ],
    )
    sample_ids_path = output_dir / "sample_ids.txt"
    sample_ids_path.write_text("\n".join(str(record.get("sample_id")) for record in frozen_manifest) + "\n", encoding="utf-8")
    status_counts = Counter(row["status"] for row in audit_rows)
    version = {
        "dataset_version": "scamweb-v1.0-qwen3-pilot",
        "input_policy_version": AUDIT_VERSION,
        "created_unix": time.time(),
        "sample_count": len(frozen_manifest),
        "status_counts": dict(status_counts),
        "gold_input_conflict_count": sum(row["gold_input_conflict"] for row in audit_rows),
        "gold_conflict_check_counts": dict(
            Counter(row["gold_conflict_check_status"] for row in audit_rows)
        ),
        "api_audit_record_count": audit_calls,
        "api_audit_physical_call_count": int(getattr(auditor, "api_call_count", 0) or 0),
        "sha256": {
            "source_manifest": _file_sha256(manifest_path),
            "source_evidence": _file_sha256(evidence_path),
            "frozen_manifest": _file_sha256(frozen_manifest_path),
            "frozen_evidence": _file_sha256(frozen_evidence_path),
            "audit": _file_sha256(audit_path),
            "recapture": _file_sha256(recapture_path),
            "sample_ids": _file_sha256(sample_ids_path),
        },
    }
    version_path = output_dir / "dataset_version.json"
    version_path.write_text(json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8")
    finalize_box_asset(output_dir)
    version = json.loads(version_path.read_text(encoding="utf-8"))
    if auditor is not None and hasattr(auditor, "close"):
        auditor.close()
    return version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and freeze ScamWeb-V MHTML inputs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--web-evidence", default="outputs/dataset/web_evidence.jsonl")
    parser.add_argument("--output-dir", default="outputs/dataset_frozen_v1")
    parser.add_argument("--skip-recapture", action="store_true")
    parser.add_argument("--no-api-auditor", action="store_true")
    parser.add_argument("--audit-call-limit", type=int, default=450)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = audit_and_freeze(
        Path(args.manifest),
        Path(args.web_evidence),
        Path(args.output_dir),
        recapture=not args.skip_recapture,
        use_api_auditor=not args.no_api_auditor,
        audit_call_limit=args.audit_call_limit,
    )
    print(json.dumps(version, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
