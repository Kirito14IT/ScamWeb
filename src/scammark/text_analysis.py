"""Textual persuasion analysis for ScamMark."""

from __future__ import annotations

import re
from collections.abc import Iterable


TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Online Loan Scam",
        (
            "loan",
            "bad credit",
            "guaranteed approval",
            "instant approval",
            "payday",
            "credit accepted",
            "贷款",
            "借款",
            "放款",
            "信用",
            "免审核",
        ),
    ),
    (
        "Cryptocurrency Investment Scam",
        (
            "crypto",
            "cryptocurrency",
            "bitcoin",
            "wallet",
            "airdrop",
            "nft",
            "token",
            "区块链",
            "虚拟币",
            "钱包",
        ),
    ),
    (
        "Online Investment Scam",
        (
            "investment",
            "invest",
            "profit",
            "return",
            "guaranteed income",
            "high yield",
            "收益",
            "投资",
            "理财",
            "稳赚",
            "回报",
        ),
    ),
    (
        "Phishing",
        (
            "password",
            "verify account",
            "sign in",
            "login",
            "credential",
            "security check",
            "登录",
            "密码",
            "验证账号",
            "钓鱼",
        ),
    ),
    (
        "Online Shopping Scam",
        (
            "discount",
            "order now",
            "limited sale",
            "shopping",
            "checkout",
            "购物",
            "商城",
            "下单",
            "优惠",
        ),
    ),
    (
        "Online Gambling Scam",
        (
            "casino",
            "betting",
            "jackpot",
            "slot",
            "博彩",
            "赌博",
            "投注",
        ),
    ),
    (
        "Lottery / Prize Scam",
        (
            "reward",
            "bonus",
            "giveaway",
            "lottery",
            "prize",
            "claim now",
            "中奖",
            "红包",
            "抽奖",
        ),
    ),
    (
        "Job / Recruitment Scam",
        (
            "work from home",
            "part-time",
            "easy money",
            "兼职",
            "刷单",
            "返利",
        ),
    ),
]

URGENCY_KEYWORDS = (
    "limited",
    "urgent",
    "now",
    "today",
    "exclusive",
    "guaranteed",
    "no risk",
    "限时",
    "立即",
    "马上",
    "保证",
    "零风险",
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase in text


def _score_keywords(text: str, keywords: Iterable[str], prefix: str) -> tuple[int, list[str]]:
    signals = []
    score = 0
    for keyword in keywords:
        if _contains_phrase(text, keyword.lower()):
            signals.append(f"{prefix}:{keyword}")
            score += 1
    return score, signals


def analyze_textual_persuasion(title: str = "", text: str = "") -> dict:
    """Find persuasion and scam-type signals in visible text-like evidence."""

    combined = f"{title or ''}\n{text or ''}".lower()
    combined = re.sub(r"\s+", " ", combined)

    signals: list[str] = []
    risk_score = 0
    best_type = "none"
    best_score = 0

    for scam_type, keywords in TYPE_KEYWORDS:
        score, type_signals = _score_keywords(combined, keywords, "text_signal")
        if score > 0:
            risk_score += score
            signals.extend(type_signals)
        if score > best_score:
            best_type = scam_type
            best_score = score

    urgency_score, urgency_signals = _score_keywords(combined, URGENCY_KEYWORDS, "urgency")
    risk_score += urgency_score
    signals.extend(urgency_signals)

    return {
        "risk_score": risk_score,
        "signals": signals[:20],
        "suggested_scam_type": best_type,
        "text_excerpt": (text or title or "")[:500],
    }
