"""Minimal reproducible supervised baselines for ScamWeb-V binary detection."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from urllib.parse import unquote, urlsplit


SUSPICIOUS_TOKENS = (
    "account",
    "bank",
    "bonus",
    "casino",
    "claim",
    "crypto",
    "download",
    "gift",
    "login",
    "pay",
    "prize",
    "secure",
    "support",
    "verify",
    "wallet",
)

URL_FEATURE_NAMES = (
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "fragment_length",
    "digit_count",
    "alpha_count",
    "non_alnum_count",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "at_count",
    "equals_count",
    "ampersand_count",
    "percent_count",
    "subdomain_count",
    "is_https",
    "host_is_ip",
    "has_punycode",
    "has_explicit_port",
    "suspicious_token_count",
    "character_entropy",
    "host_digit_ratio",
    "url_digit_ratio",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_split(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "scamweb-v-fixed-split-v1":
        raise ValueError("Unsupported split schema")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_split_manifest(split: dict[str, Any], manifest_path: str | Path) -> None:
    expected = str(split.get("manifest_sha256") or "")
    observed = sha256_file(manifest_path)
    if not expected or expected != observed:
        raise ValueError(f"Manifest hash mismatch: expected={expected}, observed={observed}")


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum((count / len(text)) * math.log2(count / len(text)) for count in counts.values())


def url_features(raw_url: str) -> list[float]:
    text = unquote(str(raw_url or "").strip()).lower()
    parse_target = text if "://" in text else f"http://{text}"
    parsed = urlsplit(parse_target)
    host = parsed.hostname or ""
    try:
        host_is_ip = int(bool(host) and ipaddress.ip_address(host) is not None)
    except ValueError:
        host_is_ip = 0
    try:
        has_explicit_port = int(parsed.port is not None)
    except ValueError:
        has_explicit_port = 1
    labels = [part for part in host.split(".") if part]
    suspicious_count = sum(text.count(token) for token in SUSPICIOUS_TOKENS)
    digits = sum(char.isdigit() for char in text)
    host_digits = sum(char.isdigit() for char in host)
    values = (
        len(text),
        len(host),
        len(parsed.path),
        len(parsed.query),
        len(parsed.fragment),
        digits,
        sum(char.isalpha() for char in text),
        sum(not char.isalnum() for char in text),
        text.count("."),
        text.count("-"),
        text.count("_"),
        text.count("/"),
        text.count("@"),
        text.count("="),
        text.count("&"),
        text.count("%"),
        max(0, len(labels) - 2),
        int(parsed.scheme == "https"),
        host_is_ip,
        int("xn--" in host),
        has_explicit_port,
        suspicious_count,
        _entropy(text),
        host_digits / len(host) if host else 0.0,
        digits / len(text) if text else 0.0,
    )
    return [float(value) for value in values]


def records_for_split(
    records: Iterable[dict[str, Any]],
    split: dict[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    mapping = split["sample_to_split"]
    selected = [record for record in records if mapping.get(str(record["sample_id"])) == name]
    expected = int(split["split_counts"][name])
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} {name} samples, found {len(selected)}")
    return selected


def binary_targets(records: Sequence[dict[str, Any]]) -> list[int]:
    return [1 if str(record.get("binary")) == "scam" else 0 for record in records]


def fit_url_xgboost(
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    *,
    seed: int = 2027,
    workers: int = 1,
):
    import numpy as np
    from xgboost import XGBClassifier

    train_x = np.asarray([url_features(record.get("url", "")) for record in train_records])
    train_y = np.asarray(binary_targets(train_records))
    validation_x = np.asarray([url_features(record.get("url", "")) for record in validation_records])
    validation_y = np.asarray(binary_targets(validation_records))
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=workers,
    )
    model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)
    return model


def resolve_dataset_asset(path_value: str, dataset_root: str | Path) -> Path:
    raw = str(path_value or "")
    direct = Path(raw)
    if direct.is_file():
        return direct
    normalized = raw.replace("\\", "/")
    marker = "/scamweb_dataset/"
    lowered = normalized.lower()
    position = lowered.find(marker)
    if position < 0:
        raise FileNotFoundError(f"Cannot map dataset asset: {raw}")
    relative = normalized[position + len(marker) :].lstrip("/")
    mapped = Path(dataset_root).joinpath(*PurePosixPath(relative).parts)
    if not mapped.is_file():
        raise FileNotFoundError(mapped)
    return mapped


def extract_clip_embeddings(
    records: Sequence[dict[str, Any]],
    *,
    dataset_root: str | Path,
    model_name: str,
    cache_dir: str | Path,
    batch_size: int,
    device: str,
):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    ids_hash = hashlib.sha256("\n".join(str(row["sample_id"]) for row in records).encode()).hexdigest()[:16]
    feature_path = cache_path / f"{model_slug}_{ids_hash}.npz"
    if feature_path.is_file():
        payload = np.load(feature_path, allow_pickle=False)
        return payload["features"]

    Image.MAX_IMAGE_PIXELS = None
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_path)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_path).to(device).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            rows = records[start : start + batch_size]
            images = [Image.open(resolve_dataset_asset(row["image_path"], dataset_root)).convert("RGB") for row in rows]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            batches.append(features.float().cpu().numpy())
            for image in images:
                image.close()
    matrix = np.concatenate(batches, axis=0)
    np.savez_compressed(feature_path, features=matrix)
    return matrix


def fit_clip_linear(
    train_features,
    train_targets: Sequence[int],
    validation_features,
    validation_targets: Sequence[int],
    *,
    seed: int = 2027,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    best_model = None
    best_score = -1.0
    for regularization in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        model.fit(train_features, train_targets)
        score = f1_score(validation_targets, model.predict(validation_features), pos_label=1)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def label_predictions(values: Iterable[int]) -> list[str]:
    return ["scam" if int(value) == 1 else "benign" for value in values]
