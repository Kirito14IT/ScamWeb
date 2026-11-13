"""Prompt templates for the staged ScamMark pipeline."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .taxonomy import CANONICAL_LABELS


HIDDEN_SAMPLE_KEYS = {
    "sample_id",
    "id",
    "raw_id",
    "gold_index",
    "workbook_id",
    "workbook_row",
    "binary",
    "label",
    "scam_type",
    "scam_type_raw",
    "boxes",
    "description",
    "reasoning",
    "language",
    "rationale",
    "truth_description",
    "truth_reasoning",
    "label_source",
    "source",
    "split",
    "cluster_id",
    "is_gold",
    "is_benign",
    "is_ai_assisted",
    "is_weak",
    "mhtml_audit_status",
    "mhtml_valid",
    "mhtml_used",
    "gold_input_conflict",
    "gold_conflict_check_status",
    "input_policy_version",
}

MODEL_VISIBLE_SAMPLE_KEYS = {
    "url",
    "title",
    "tld",
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
    "image_size",
}

STAGE_SAMPLE_KEYS = {
    "url": {
        "url",
        "tld",
        "archive_url",
        "source_url",
        "url_domain_observed",
        "parse_status",
    },
    "text": {
        "title",
        "text",
        "visible_text",
        "buttons",
        "forms",
        "links",
        "meta",
        "external_domains",
        "parse_status",
        "html_truncated",
    },
    "visual": {"image_size"},
    "cross": {"image_size"},
    "binary": {"image_size"},
    "type": {"image_size"},
    "grounding": {"image_size"},
}

NESTED_SUPERVISION_KEYS = {
    "sample_id",
    "id",
    "raw_id",
    "gold_index",
    "workbook_id",
    "workbook_row",
    "label",
    "scam_type_raw",
    "rationale",
    "truth_description",
    "truth_reasoning",
    "label_source",
    "source",
    "split",
    "cluster_id",
    "is_gold",
    "is_benign",
    "is_ai_assisted",
    "is_weak",
    "mhtml_audit_status",
    "mhtml_valid",
    "mhtml_used",
    "gold_input_conflict",
    "gold_conflict_check_status",
    "input_policy_version",
    "truth",
    "gold",
    "annotation",
    "annotations",
    "metadata",
}

PROMPT_STAGES = ("url", "text", "visual", "cross", "binary", "type", "grounding")
IMAGE_STAGES = {"visual", "cross", "binary", "type", "grounding"}

STAGE_TASKS = {
    "url": {
        "title": "URL Evidence Analysis",
        "role": "LLM",
        "instruction": (
            "Analyze the webpage URL and structured webpage evidence. Focus on domain, TLD, path, redirects, "
            "brand-domain mismatch, off-platform contact cues, and scam-related tokens such as crypto, wallet, "
            "investment, loan, casino, lottery, prize, adult, dating, escort, job, task, streaming, gift card, "
            "support, bank, government, charity, education, login, verify, download, app, or QR."
        ),
        "schema": {
            "risk_level": "low|medium|high",
            "domain_observations": ["short observations about host, TLD, path, or redirect clues"],
            "brand_mismatch": ["possible brand/domain inconsistencies"],
            "suspicious_tokens": ["tokens that indicate login, reward, loan, wallet, pay, adult, casino, job, app, etc."],
            "evidence_summary": "one concise summary grounded in URL evidence",
        },
    },
    "text": {
        "title": "Textual Persuasion Analysis",
        "role": "LLM",
        "instruction": (
            "Analyze title, OCR or visible text, MHTML/DOM summaries, buttons, forms, and links. "
            "Extract persuasive claims, urgency, promised benefits, requested actions, sensitive fields, and candidate scam types. "
            "Look for dataset-relevant tactics: unrealistic discounts or product prices, fake investment or cryptocurrency returns, "
            "account verification/login capture, adult or romance lures, gambling or lottery promises, technical-support fear, "
            "social-platform redirection, education/job deposits, charity donation appeals, fake news clickbait, streaming or gift-card offers."
        ),
        "schema": {
            "persuasion_signals": ["high return, guaranteed approval, limited time, customer-service pressure, etc."],
            "claims": ["atomic claims or promises made by the page"],
            "requested_actions": ["login, deposit, pay, recharge, contact service, download, submit credentials, etc."],
            "candidate_scam_types": ["English ScamWeb-V taxonomy labels or none"],
            "evidence_summary": "one concise summary grounded in text evidence",
        },
    },
    "visual": {
        "title": "Visual Layout Analysis",
        "role": "VLM",
        "instruction": (
            "Inspect the webpage screenshot. Identify visible risk components across ScamWeb-V scam types: product grids, unrealistic discounts, "
            "luxury goods, checkout or payment panels, crypto dashboards, trading charts, wallet/deposit/withdraw widgets, login forms, "
            "QR codes, app/download prompts, adult/escort/dating profiles, provocative imagery, casino or lottery banners, Telegram/social handles, "
            "fake news cards, certification or accreditation logos, charity/donation blocks, bank/government seals, trust badges, testimonials, "
            "customer-service widgets, and mismatched or suspicious brand logos. Use image_inputs to distinguish the full-page overview from "
            "numbered vertical tiles. If submission_mode is contact_sheet, the single uploaded image contains labeled image_index panels; inspect "
            "each panel inside its contact_sheet_panel_xyxy content bounds and ignore panel headers or padding. Return at most five concise items "
            "per list and paraphrase visible text instead of copying long quotations."
        ),
        "schema": {
            "page_genre": "shopping|investment|crypto|phishing|gambling|adult_or_romance|social|streaming|education|job|charity|news|bank|government|support|other",
            "visual_scam_cues": ["visible risky components such as low prices, profit charts, casino banners, adult lures, or fake news cards"],
            "sensitive_or_monetary_inputs": ["password, phone, wallet, bank card, payment, deposit, identity, donation, or account fields"],
            "impersonated_or_trust_elements": ["logos, seals, certificates, accreditation marks, authority badges, testimonials, or fake trust signals"],
            "social_contact_or_redirects": ["Telegram, WhatsApp, social handles, QR codes, app downloads, or off-platform contact prompts"],
            "adult_or_romance_lures": ["adult, escort, dating, romance, provocative profile, or intimate-chat visual cues"],
            "candidate_regions": [{"name": "region name", "reason": "why this region matters"}],
            "evidence_summary": "one concise summary grounded in visual evidence",
        },
    },
    "cross": {
        "title": "Cross-modal Consistency Analysis",
        "role": "VLM",
        "instruction": (
            "Compare URL, title, structured text evidence, screenshot evidence, and previous analyses. "
            "Find whether the modalities support or contradict each other, especially brand, page genre, call-to-action, payment request, "
            "credential request, social-contact redirection, language, trust signal, and the overall risk story. Return at most four concise "
            "items per list and keep the evidence summary under 60 words."
        ),
        "schema": {
            "consistency_findings": ["aligned or contradictory multimodal findings"],
            "conflicts": ["specific URL-title-text-image conflicts"],
            "supporting_matches": ["specific signals that reinforce the same scam hypothesis"],
            "risk_level": "low|medium|high",
            "evidence_summary": "one concise cross-modal summary",
        },
    },
    "binary": {
        "title": "Binary Scam Classifier",
        "role": "VLM",
        "instruction": (
            "Judge whether the webpage is a scam or benign using the sanitized webpage input and staged analyses only. "
            "Treat money requests, credential capture, off-platform contact, adult or romance lures, fake authority, unrealistic benefits, "
            "suspicious QR/app downloads, and brand impersonation as high-value evidence. Do not infer from file names, folder names, "
            "evaluation truth, or benchmark metadata."
        ),
        "schema": {
            "binary": "scam|benign",
            "reasoning": "short evidence-based explanation",
        },
    },
    "type": {
        "title": "Single-Label Scam Subtype Classifier",
        "role": "VLM",
        "instruction": (
            "Choose exactly one closed-set ScamWeb-V subtype ID. Use ID 0 only for benign; in diagnostic force_downstream mode, "
            "choose the best ID from 1 through 25 even when the upstream binary stage predicted benign. Copy the label paired with that ID "
            "exactly into scam_type. Do not invent or paraphrase labels. This stage receives concise prior-stage evidence summaries rather "
            "than the raw screenshot or raw webpage text."
        ),
        "schema": {
            "scam_type_id": 0,
            "scam_type": "exact label paired with scam_type_id, or none for ID 0",
            "reasoning": "short evidence-based explanation",
        },
    },
    "grounding": {
        "title": "Region Grounding and Explanation",
        "role": "VLM",
        "instruction": (
            "Localize suspicious screenshot regions that justify the prediction. When image_inputs are supplied, return each region as "
            "image_index plus bbox_1000=[x1,y1,x2,y2], normalized to 0--1000 within that logical overview or tile; it will be mapped back "
            "to original screenshot pixel-space xyxy by the code. If submission_mode is contact_sheet, select the labeled image_index panel, "
            "ignore its header and padding, and normalize bbox_1000 within the selected panel's image content rather than across the full "
            "contact sheet. If image_inputs are absent, return original pixel-space xyxy. "
            "For benign pages, return an empty boxes list. In diagnostic force_downstream mode, still localize the strongest candidate "
            "evidence even when the upstream binary stage predicted benign. Return at most four strongest, non-duplicate regions. "
            "Each box must correspond to visible evidence, not a whole page unless necessary. Keep every reason under 40 words."
        ),
        "schema": {
            "description": "brief page description",
            "reasoning": "why the selected regions support the prediction",
            "boxes": [
                {
                    "image_index": "integer index from image_inputs",
                    "bbox_1000": [0, 0, 1000, 1000],
                    "reason": "region-level explanation",
                    "evidence_type": (
                        "text|form|button|brand|payment|layout|adult_lure|social_contact|qr_code|pricing|chart|"
                        "login_form|brand_logo|testimonial|fake_news|education_claim|charity_claim|download|other"
                    ),
                }
            ],
        },
    },
}


def _looks_like_local_path(value: str) -> bool:
    return bool(re.match(r"^(?:[a-zA-Z]:[\\/]|\\\\|file://|/(?:home|Users|mnt|tmp)/)", value.strip()))


def _sanitize_value(value: Any, *, filter_metadata_keys: bool) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if filter_metadata_keys and (
                key in NESTED_SUPERVISION_KEYS
                or key.startswith(("truth_", "gold_", "source_"))
                or key.endswith("_path")
            ):
                continue
            sanitized[key] = _sanitize_value(item, filter_metadata_keys=filter_metadata_keys)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, filter_metadata_keys=filter_metadata_keys) for item in value]
    if isinstance(value, str) and _looks_like_local_path(value):
        return "[redacted_local_path]"
    return value


def build_sample_context(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return model-visible sample fields without evaluation supervision."""

    context: dict[str, Any] = {}
    for key, value in sample.items():
        if key not in MODEL_VISIBLE_SAMPLE_KEYS:
            continue
        if value in (None, "", [], {}):
            continue
        context[key] = _sanitize_value(value, filter_metadata_keys=True)
    return context


def _stage_sample_context(stage: str, sample: Mapping[str, Any]) -> dict[str, Any]:
    allowed = STAGE_SAMPLE_KEYS[stage]
    return {
        key: value
        for key, value in build_sample_context(sample).items()
        if key in allowed
    }


def _present(value: Mapping[str, Any], key: str) -> bool:
    return value.get(key) not in (None, "", [], {})


def _controlled_previous(analyses: Mapping[str, Any]) -> dict[str, Any]:
    """Convert prior outputs into enums and presence flags without free-text quotes."""

    compact_stages: dict[str, dict[str, Any]] = {}
    prompt_stages = analyses.get("prompt_stages")
    if isinstance(prompt_stages, Mapping):
        url = prompt_stages.get("url")
        if isinstance(url, Mapping):
            risk_level = str(url.get("risk_level") or "").lower()
            compact_stages["url"] = {
                "risk_level": risk_level if risk_level in {"low", "medium", "high"} else "unknown",
                "domain_observations_present": _present(url, "domain_observations"),
                "brand_mismatch_present": _present(url, "brand_mismatch"),
                "suspicious_tokens_present": _present(url, "suspicious_tokens"),
            }

        text = prompt_stages.get("text")
        if isinstance(text, Mapping):
            candidates = [
                str(label)
                for label in text.get("candidate_scam_types") or []
                if str(label) in CANONICAL_LABELS
            ]
            compact_stages["text"] = {
                "persuasion_signals_present": _present(text, "persuasion_signals"),
                "claims_present": _present(text, "claims"),
                "requested_actions_present": _present(text, "requested_actions"),
                "candidate_scam_types": candidates,
            }

        visual = prompt_stages.get("visual")
        if isinstance(visual, Mapping):
            page_genre = str(visual.get("page_genre") or "other")
            allowed_genres = {
                "shopping",
                "investment",
                "crypto",
                "phishing",
                "gambling",
                "adult_or_romance",
                "social",
                "streaming",
                "education",
                "job",
                "charity",
                "news",
                "bank",
                "government",
                "support",
                "other",
            }
            compact_stages["visual"] = {
                "page_genre": page_genre if page_genre in allowed_genres else "other",
                "visual_scam_cues_present": _present(visual, "visual_scam_cues"),
                "sensitive_or_monetary_inputs_present": _present(
                    visual, "sensitive_or_monetary_inputs"
                ),
                "impersonated_or_trust_elements_present": _present(
                    visual, "impersonated_or_trust_elements"
                ),
                "social_contact_or_redirects_present": _present(
                    visual, "social_contact_or_redirects"
                ),
                "adult_or_romance_lures_present": _present(visual, "adult_or_romance_lures"),
                "candidate_regions_present": _present(visual, "candidate_regions"),
            }

        cross = prompt_stages.get("cross")
        if isinstance(cross, Mapping):
            risk_level = str(cross.get("risk_level") or "").lower()
            compact_stages["cross"] = {
                "risk_level": risk_level if risk_level in {"low", "medium", "high"} else "unknown",
                "consistency_findings_present": _present(cross, "consistency_findings"),
                "conflicts_present": _present(cross, "conflicts"),
                "supporting_matches_present": _present(cross, "supporting_matches"),
            }

        binary = prompt_stages.get("binary")
        if isinstance(binary, Mapping):
            label = str(binary.get("binary") or "").lower()
            compact_stages["binary"] = {
                "binary": label if label in {"scam", "benign"} else "unknown"
            }

        subtype = prompt_stages.get("type")
        if isinstance(subtype, Mapping):
            label = str(subtype.get("scam_type") or "")
            type_id = subtype.get("scam_type_id")
            compact_stages["type"] = {
                "scam_type_id": type_id if isinstance(type_id, int) and 0 <= type_id <= 25 else None,
                "scam_type": label if label in {*CANONICAL_LABELS, "none"} else "none",
            }

    result: dict[str, Any] = {"prompt_stages": compact_stages}
    image_inputs = analyses.get("image_inputs")
    if isinstance(image_inputs, (list, tuple)):
        result["image_inputs"] = _sanitize_value(image_inputs, filter_metadata_keys=True)
    if analyses.get("force_downstream") is not None:
        result["force_downstream"] = bool(analyses.get("force_downstream"))
    return result


def _stage_analysis_context(stage: str, analyses: Mapping[str, Any]) -> dict[str, Any]:
    if stage in {"url", "text"}:
        return {}
    if stage == "visual":
        image_inputs = analyses.get("image_inputs")
        return {
            "image_inputs": _sanitize_value(image_inputs, filter_metadata_keys=True)
        } if isinstance(image_inputs, (list, tuple)) else {}
    return _controlled_previous(analyses)


def build_stage_prompt(stage: str, sample: Mapping[str, Any], analyses: Mapping[str, Any] | None = None) -> str:
    """Build one strict-JSON prompt for a ScamMark stage."""

    if stage not in STAGE_TASKS:
        raise ValueError(f"Unknown prompt stage: {stage}")
    task = STAGE_TASKS[stage]
    sample_context = _stage_sample_context(stage, sample)
    analysis_context: Any = _stage_analysis_context(stage, analyses or {})
    allowed_labels: Any = [label for label in CANONICAL_LABELS if label.lower() != "none"]
    if stage == "type":
        allowed_labels = [
            {"id": 0, "label": "none"},
            *({"id": index, "label": label} for index, label in enumerate(CANONICAL_LABELS, start=1)),
        ]
    payload = {
        "sample": sample_context,
        "previous_analyses": analysis_context,
        "allowed_taxonomy_labels": allowed_labels,
    }
    context_text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    schema_text = json.dumps(task["schema"], ensure_ascii=False, indent=2)
    return (
        f"You are ScamMark's {task['role']} stage: {task['title']}.\n"
        f"Task: {task['instruction']}\n"
        "Use only the sanitized input below. Never use evaluation truth, annotation metadata, or hidden supervision fields.\n"
        "Return only strict JSON. Do not wrap the answer in Markdown. Keep every string concise, paraphrase long quotations, "
        "escape internal double quotes, and never put literal line breaks inside a JSON string.\n"
        f"Required JSON schema:\n{schema_text}\n\n"
        f"Sanitized input:\n{context_text}"
    )


def should_include_image(stage: str) -> bool:
    """Whether a stage should attach the screenshot when the backend supports images."""

    if stage not in STAGE_TASKS:
        raise ValueError(f"Unknown prompt stage: {stage}")
    return stage in IMAGE_STAGES
