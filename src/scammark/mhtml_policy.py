"""Frozen MHTML inclusion policy applied before prompt construction."""

from __future__ import annotations

from copy import deepcopy


TRUSTED_MHTML_STATUSES = {"valid", "valid_redirect"}
MHTML_EVIDENCE_KEYS = {
    "text",
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
}


def apply_mhtml_policy(sample: dict) -> dict:
    filtered = deepcopy(sample)
    status = str(filtered.get("mhtml_audit_status") or "").strip().lower()
    if not status:
        if str(filtered.get("parse_status") or "") != "ok":
            status = "missing"
        elif filtered.get("url_domain_observed") is False:
            status = "uncertain"
        else:
            status = "valid"
    trusted = status in TRUSTED_MHTML_STATUSES
    if not trusted:
        for key in MHTML_EVIDENCE_KEYS:
            filtered.pop(key, None)
    filtered["mhtml_audit_status"] = status
    filtered["mhtml_used"] = trusted
    return filtered
