"""Cross-modal consistency heuristics for ScamMark."""

from __future__ import annotations


def analyze_consistency(url_result: dict, text_result: dict, visual_result: dict) -> dict:
    signals: list[str] = []
    risk_score = 0

    if url_result.get("risk_score", 0) > 0 and text_result.get("risk_score", 0) > 0:
        signals.append("url_and_text_both_risky")
        risk_score += 1

    if text_result.get("suggested_scam_type") != "none" and not visual_result.get("image_exists"):
        signals.append("text_risky_but_image_missing")

    if url_result.get("tld") in {"cfd", "xyz", "top"} and text_result.get("suggested_scam_type") in {
        "Online Investment Scam",
        "Cryptocurrency Investment Scam",
        "Online Loan Scam",
    }:
        signals.append("risky_tld_matches_financial_persuasion")
        risk_score += 1

    return {
        "risk_score": risk_score,
        "signals": signals,
    }
