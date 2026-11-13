"""ScamWeb-V taxonomy normalization for demo metrics.

The gold manual annotations use English scam-type labels, so this module keeps
normalization inside the English ScamWeb-V taxonomy.
"""

from __future__ import annotations

import re


CANONICAL_LABELS = (
    "Task-Based Commission Scam",
    "E-commerce Customer Service Impersonation Scam",
    "Spoofed Login Page",
    "Online Loan Scam",
    "Online Investment Scam",
    "Cryptocurrency Investment Scam",
    "Government / Law Enforcement Impersonation Scam",
    "Romance Scam",
    "Fake Streaming Service Scam",
    "Online Shopping Scam",
    "Technical Support Scam",
    "Gaming Item Trading Scam",
    "Online Gambling Scam",
    "Charity Scam",
    "Business Impersonation Scam",
    "Job / Recruitment Scam",
    "Lottery / Prize Scam",
    "Website Hijacking Scam",
    "Education / Training Scam",
    "Fraudulent Personal Website Scam",
    "Bank Impersonation Scam",
    "Fake News Website Scam",
    "Phishing",
    "Social Media Scam",
    "Sextortion",
)


INVALID_LABEL = "__invalid__"


LABEL_ALIASES = {
    "job/recruitment scam": "Job / Recruitment Scam",
    "lottery/prize scam": "Lottery / Prize Scam",
    "government/law enforcement impersonation scam": "Government / Law Enforcement Impersonation Scam",
    "education/training scam": "Education / Training Scam",
    "phisonline shopping scamhing": "Phishing",
}


NONE_LABELS = {"none", "benign", "safe", "legitimate", "false", "no scam", "not scam", "no fraud"}


def _normalize_key(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


CANONICAL_BY_KEY = {_normalize_key(label): label for label in CANONICAL_LABELS}
ALIAS_BY_KEY = {_normalize_key(key): value for key, value in LABEL_ALIASES.items()}
ALIAS_BY_KEY.update({_normalize_key(label): label for label in CANONICAL_LABELS})


def canonicalize_scam_type(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "none"
    key = _normalize_key(text)
    if key in NONE_LABELS:
        return "none"

    return ALIAS_BY_KEY.get(key, INVALID_LABEL)
