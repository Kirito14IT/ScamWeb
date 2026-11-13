"""Token-aware prompt compaction for reproducible ScamMark API inputs."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from .mhtml_policy import apply_mhtml_policy


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    tokenizer = _tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    ascii_count = sum(ord(character) < 128 for character in text)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4) + non_ascii_count)


def _truncate_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + " [truncated]"


def _compact_value(value: Any, *, text_tokens: int, list_limit: int, depth: int = 0) -> Any:
    if depth > 5:
        return "[nested value omitted]"
    if isinstance(value, str):
        return _truncate_text(value, text_tokens)
    if isinstance(value, dict):
        compacted = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= list_limit:
                break
            compacted[str(key)[:100]] = _compact_value(
                item,
                text_tokens=max(16, text_tokens // 2),
                list_limit=max(2, list_limit // 2),
                depth=depth + 1,
            )
        return compacted
    if isinstance(value, (list, tuple)):
        return [
            _compact_value(
                item,
                text_tokens=max(16, text_tokens // 2),
                list_limit=max(2, list_limit // 2),
                depth=depth + 1,
            )
            for item in value[:list_limit]
        ]
    return value


def _compact_sample(sample: dict[str, Any], text_tokens: int, list_limit: int) -> dict[str, Any]:
    sample = apply_mhtml_policy(sample)
    priority = (
        "url",
        "title",
        "tld",
        "parse_status",
        "archive_url",
        "source_url",
        "url_domain_observed",
        "html_truncated",
        "image_size",
        "forms",
        "buttons",
        "meta",
        "external_domains",
        "links",
        "visible_text",
        "text",
    )
    result = {}
    for key in priority:
        value = sample.get(key)
        if value in (None, "", [], {}):
            continue
        per_field_tokens = text_tokens
        if key in {"visible_text", "text"}:
            per_field_tokens = text_tokens * 2
        result[key] = _compact_value(value, text_tokens=per_field_tokens, list_limit=list_limit)
    return result


def _compact_analyses(analyses: dict[str, Any], *, text_tokens: int, list_limit: int) -> dict[str, Any]:
    """Compact evidence while preserving stage names and image geometry exactly."""

    result: dict[str, Any] = {}
    for key, value in analyses.items():
        if key == "image_inputs":
            # These records define the coordinate system of every submitted image.
            # Dropping or shortening one would make model-returned image indexes ambiguous.
            result[key] = value
        elif key in {"prompt_stages", "heuristic"} and isinstance(value, dict):
            result[key] = {
                str(stage): _compact_value(
                    stage_value,
                    text_tokens=max(16, text_tokens // 2),
                    list_limit=max(2, list_limit // 2),
                )
                for stage, stage_value in value.items()
            }
        else:
            result[key] = _compact_value(
                value,
                text_tokens=max(16, text_tokens // 2),
                list_limit=max(2, list_limit // 2),
            )
    return result


def build_budgeted_stage_prompt(
    stage: str,
    sample: dict[str, Any],
    analyses: dict[str, Any] | None,
    *,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Build a normal stage prompt while shrinking only model-visible evidence."""

    from .prompts import build_stage_prompt

    attempts = (
        (1024, 40),
        (512, 24),
        (256, 12),
        (128, 8),
        (64, 4),
        (16, 2),
    )
    original_sample = apply_mhtml_policy(sample)
    original_prompt = build_stage_prompt(stage, original_sample, analyses)
    original_tokens = count_tokens(original_prompt)
    if original_tokens <= max_tokens:
        return original_prompt, {
            "max_tokens": max_tokens,
            "original_tokens": original_tokens,
            "final_tokens": original_tokens,
            "truncated": False,
            "tokenizer": "o200k_base" if _tokenizer() is not None else "conservative_fallback",
        }

    final_prompt = original_prompt
    final_tokens = original_tokens
    for text_tokens, list_limit in attempts:
        compact_sample = _compact_sample(original_sample, text_tokens, list_limit)
        compact_analyses = _compact_analyses(
            analyses or {},
            text_tokens=max(32, text_tokens // 2),
            list_limit=max(2, list_limit // 2),
        )
        final_prompt = build_stage_prompt(stage, compact_sample, compact_analyses)
        final_tokens = count_tokens(final_prompt)
        if final_tokens <= max_tokens:
            break
    if final_tokens > max_tokens:
        raise ValueError(
            f"Fixed {stage} prompt requires {final_tokens} tokens, exceeding configured limit {max_tokens}"
        )
    return final_prompt, {
        "max_tokens": max_tokens,
        "original_tokens": original_tokens,
        "final_tokens": final_tokens,
        "truncated": True,
        "tokenizer": "o200k_base" if _tokenizer() is not None else "conservative_fallback",
    }
