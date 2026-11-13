"""URL evidence analysis for the training-free ScamMark demo."""

from __future__ import annotations

import re
from urllib.parse import urlparse


SUSPICIOUS_TLDS = {
    "cfd",
    "click",
    "club",
    "cyou",
    "icu",
    "loan",
    "monster",
    "online",
    "shop",
    "site",
    "top",
    "vip",
    "win",
    "xyz",
}

SUSPICIOUS_TOKENS = {
    "account",
    "airdrop",
    "bonus",
    "claim",
    "crypto",
    "gift",
    "login",
    "loan",
    "pay",
    "profit",
    "reward",
    "secure",
    "signin",
    "verify",
    "wallet",
}


def _ensure_parseable_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        return "https://" + url
    return url


def _hostname_parts(url: str) -> tuple[str, str]:
    parsed = urlparse(_ensure_parseable_url(url))
    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        return "", ""
    labels = [part for part in host.split(".") if part]
    tld = labels[-1] if labels else ""
    return host, tld


def analyze_url(url: str) -> dict:
    """Return lightweight URL risk signals without network access."""

    normalized = _ensure_parseable_url(url)
    parsed = urlparse(normalized)
    host, tld = _hostname_parts(url)
    haystack = " ".join([host, parsed.path.lower(), parsed.query.lower()])
    tokens = set(re.findall(r"[a-z0-9]+", haystack))

    signals: list[str] = []
    risk_score = 0

    if tld in SUSPICIOUS_TLDS:
        signals.append(f"suspicious_tld:{tld}")
        risk_score += 1

    for token in sorted(tokens & SUSPICIOUS_TOKENS):
        signals.append(f"suspicious_token:{token}")
        risk_score += 1

    if host and any(ch.isdigit() for ch in host):
        signals.append("digit_in_domain")
        risk_score += 1

    if parsed.path.count("/") >= 4:
        signals.append("deep_url_path")
        risk_score += 1

    return {
        "url": url,
        "host": host,
        "tld": tld,
        "risk_score": risk_score,
        "signals": signals,
    }
