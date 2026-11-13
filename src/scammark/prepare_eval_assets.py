"""Download and verify the pinned ScamMark evaluation model."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from .bertscore_eval import (
    BERTSCORE_NUM_LAYERS,
    BERTSCORE_HASH_VERSION,
    BERTSCORE_PACKAGE_VERSION,
    BERTSCORE_REPO_ID,
    BERTSCORE_REVISION,
    TRANSFORMERS_PACKAGE_VERSION,
    BERTScoreConfig,
    _descriptor_model_path,
    _descriptor_number,
    _secure_open_model_directory,
    default_model_dir,
    get_bertscore_evaluator,
)


_REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "tokenizer.json",
)
_BASE_EVALUATOR_METADATA_FIELDS = frozenset(
    {
        "repo_id",
        "revision",
        "model",
        "num_layers",
        "batch_size",
        "device_requested",
        "idf",
        "rescale_with_baseline",
    }
)
_LOADED_EVALUATOR_METADATA_FIELDS = frozenset(
    {
        "device_actual",
        "bert_score_version",
        "transformers_version",
        "torch_version",
        "hash",
    }
)


def _absolute_path_without_resolving(path: Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("BERTScore cache path is invalid") from exc


def _directory_component_bindings(
    path: Path,
    *,
    require_leaf: bool,
) -> tuple[tuple[int, int, int, int], ...]:
    if not path.is_absolute() or not path.anchor:
        raise RuntimeError("BERTScore cache path is not absolute")
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    current = Path(path.anchor)
    components = (current,)
    for part in path.parts[1:]:
        current = current / part
        components += (current,)

    bindings: list[tuple[int, int, int, int]] = []
    missing_component = False
    for component in components:
        if missing_component:
            continue
        try:
            metadata = component.lstat()
        except FileNotFoundError as exc:
            if require_leaf:
                raise RuntimeError(
                    "BERTScore model directory is missing"
                ) from exc
            missing_component = True
            continue
        except OSError as exc:
            raise RuntimeError(
                "BERTScore model path metadata could not be validated"
            ) from exc
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or (
            reparse_flag and attributes & reparse_flag
        ):
            raise RuntimeError(
                "BERTScore model path contains an unsafe link"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(
                "BERTScore model path contains a non-directory component"
            )
        bindings.append(
            (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                attributes,
            )
        )
    if require_leaf and len(bindings) != len(components):
        raise RuntimeError("BERTScore model directory is missing")
    return tuple(bindings)


def _validated_model_binding(
    model_dir: Path,
) -> tuple[Path, tuple[tuple[int, int, int, int], ...]]:
    bindings = _directory_component_bindings(model_dir, require_leaf=True)
    try:
        resolved = model_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            "BERTScore model directory could not be resolved"
        ) from exc
    if resolved != model_dir:
        raise RuntimeError("BERTScore model directory binding is unsafe")
    return resolved, bindings


def _open_required_model_files(
    model_dir_fd: int,
) -> tuple[tuple[str, int, tuple[int, int, int, int]], ...]:
    if (
        os.open not in getattr(os, "supports_dir_fd", ())
        or os.stat not in getattr(os, "supports_dir_fd", ())
        or os.stat not in getattr(os, "supports_follow_symlinks", ())
        or not getattr(os, "O_NOFOLLOW", 0)
    ):
        raise RuntimeError(
            "Secure BERTScore model file validation is unavailable"
        )
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    opened: list[tuple[str, int, tuple[int, int, int, int]]] = []
    try:
        for filename in _REQUIRED_MODEL_FILES:
            try:
                descriptor = os.open(
                    filename,
                    flags,
                    dir_fd=model_dir_fd,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Pinned BERTScore snapshot is incomplete: {filename}"
                ) from exc
            try:
                metadata = os.fstat(descriptor)
                entry_metadata = os.stat(
                    filename,
                    dir_fd=model_dir_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or not stat.S_ISREG(entry_metadata.st_mode)
                    or entry_metadata.st_nlink != 1
                    or (metadata.st_dev, metadata.st_ino)
                    != (entry_metadata.st_dev, entry_metadata.st_ino)
                ):
                    raise RuntimeError(
                        "Pinned BERTScore model file binding is unsafe"
                    )
                identity = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_size,
                )
            except BaseException:
                os.close(descriptor)
                raise
            opened.append((filename, descriptor, identity))
        return tuple(opened)
    except BaseException:
        for _filename, descriptor, _identity in opened:
            os.close(descriptor)
        raise


def _revalidate_required_model_files(
    model_dir_fd: int,
    bindings: tuple[tuple[str, int, tuple[int, int, int, int]], ...],
) -> None:
    for filename, descriptor, identity in bindings:
        try:
            descriptor_metadata = os.fstat(descriptor)
            entry_metadata = os.stat(
                filename,
                dir_fd=model_dir_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeError(
                "Pinned BERTScore model file binding changed"
            ) from exc
        rebound_identity = (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
            descriptor_metadata.st_mode,
            descriptor_metadata.st_size,
        )
        if (
            rebound_identity != identity
            or not stat.S_ISREG(entry_metadata.st_mode)
            or entry_metadata.st_nlink != 1
            or (entry_metadata.st_dev, entry_metadata.st_ino)
            != identity[:2]
        ):
            raise RuntimeError(
                "Pinned BERTScore model file binding changed"
            )


def _close_required_model_files(
    bindings: tuple[tuple[str, int, tuple[int, int, int, int]], ...],
) -> None:
    for _filename, descriptor, _identity in bindings:
        os.close(descriptor)


def _descriptor_matches_directory(
    descriptor_model: object,
    model_dir_fd: int,
) -> bool:
    if type(descriptor_model) is not str:
        return False
    source_descriptor = _descriptor_number(descriptor_model)
    if source_descriptor is None:
        return False
    try:
        candidate_fd = _secure_open_model_directory(descriptor_model)
    except RuntimeError:
        return False
    try:
        candidate = os.fstat(candidate_fd)
        expected = os.fstat(model_dir_fd)
        return (
            stat.S_ISDIR(candidate.st_mode)
            and (candidate.st_dev, candidate.st_ino)
            == (expected.st_dev, expected.st_ino)
        )
    finally:
        os.close(candidate_fd)


def _require_exact_metadata_value(
    metadata: dict,
    field: str,
    expected: object,
) -> None:
    value = metadata.get(field)
    if type(value) is not type(expected) or value != expected:
        raise RuntimeError(
            f"BERTScore evaluator {field} identity drifted"
        )


def _validate_device_name(device: object, *, field: str) -> str:
    if type(device) is not str:
        raise RuntimeError(f"BERTScore evaluator {field} is invalid")
    valid = device in {"cpu", "mps", "cuda"}
    if device.startswith("cuda:"):
        index = device.removeprefix("cuda:")
        valid = bool(index) and index.isascii() and index.isdecimal()
    if not valid:
        raise RuntimeError(f"BERTScore evaluator {field} is invalid")
    return device


def _validate_public_label(
    value: object,
    *,
    field: str,
    maximum_length: int,
) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum_length
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(f"BERTScore evaluator {field} is invalid")
    return value


def _project_public_metadata(
    metadata: object,
    *,
    model_dir_fd: int,
    require_scorer_hash: bool,
    expected_device: str,
) -> dict:
    if type(metadata) is not dict:
        raise RuntimeError("BERTScore evaluator metadata is invalid")
    expected_device = _validate_device_name(
        expected_device,
        field="requested device",
    )
    required_fields = set(_BASE_EVALUATOR_METADATA_FIELDS)
    allowed_fields = set(required_fields)
    if require_scorer_hash:
        required_fields.update(_LOADED_EVALUATOR_METADATA_FIELDS)
        allowed_fields.update(_LOADED_EVALUATOR_METADATA_FIELDS)
        if expected_device.startswith("cuda"):
            allowed_fields.add("cuda_device_name")
    if not required_fields.issubset(metadata) or not set(metadata).issubset(
        allowed_fields
    ):
        raise RuntimeError("BERTScore evaluator metadata fields are invalid")

    expected_model = metadata.get("model")
    _require_exact_metadata_value(
        metadata,
        "repo_id",
        BERTSCORE_REPO_ID,
    )
    _require_exact_metadata_value(
        metadata,
        "revision",
        BERTSCORE_REVISION,
    )
    if not _descriptor_matches_directory(expected_model, model_dir_fd):
        raise RuntimeError("BERTScore evaluator model identity drifted")
    _require_exact_metadata_value(
        metadata,
        "num_layers",
        BERTSCORE_NUM_LAYERS,
    )
    _require_exact_metadata_value(metadata, "batch_size", 4)
    _require_exact_metadata_value(
        metadata,
        "device_requested",
        expected_device,
    )
    _require_exact_metadata_value(metadata, "idf", False)
    _require_exact_metadata_value(
        metadata,
        "rescale_with_baseline",
        False,
    )

    public_metadata = {
        "repo_id": BERTSCORE_REPO_ID,
        "revision": BERTSCORE_REVISION,
        "model": BERTSCORE_REPO_ID,
        "num_layers": BERTSCORE_NUM_LAYERS,
        "batch_size": 4,
        "device_requested": expected_device,
        "idf": False,
        "rescale_with_baseline": False,
    }
    if require_scorer_hash:
        _require_exact_metadata_value(
            metadata,
            "device_actual",
            expected_device,
        )
        _require_exact_metadata_value(
            metadata,
            "bert_score_version",
            BERTSCORE_PACKAGE_VERSION,
        )
        _require_exact_metadata_value(
            metadata,
            "transformers_version",
            TRANSFORMERS_PACKAGE_VERSION,
        )
        torch_version = _validate_public_label(
            metadata.get("torch_version"),
            field="torch_version",
            maximum_length=128,
        )
        expected_hash_suffix = (
            f"_L{BERTSCORE_NUM_LAYERS}_no-idf_version="
            f"{BERTSCORE_HASH_VERSION}"
            f"(hug_trans={TRANSFORMERS_PACKAGE_VERSION})"
        )
        expected_raw_hash = f"{expected_model}{expected_hash_suffix}"
        _require_exact_metadata_value(
            metadata,
            "hash",
            expected_raw_hash,
        )
        public_metadata.update(
            {
                "device_actual": expected_device,
                "bert_score_version": BERTSCORE_PACKAGE_VERSION,
                "transformers_version": TRANSFORMERS_PACKAGE_VERSION,
                "torch_version": torch_version,
                "hash": f"{BERTSCORE_REPO_ID}{expected_hash_suffix}",
            }
        )
        if "cuda_device_name" in metadata:
            public_metadata["cuda_device_name"] = _validate_public_label(
                metadata["cuda_device_name"],
                field="cuda_device_name",
                maximum_length=256,
            )
    return public_metadata


def prepare_assets(cache_dir: Path, device: str, *, run_smoke: bool = True) -> dict:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to prepare BERTScore assets") from exc
    cache_dir = _absolute_path_without_resolving(cache_dir)
    model_cache_key = f"FacebookAI_xlm-roberta-large_{BERTSCORE_REVISION}"
    model_dir = cache_dir / model_cache_key
    _directory_component_bindings(model_dir, require_leaf=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _directory_component_bindings(cache_dir, require_leaf=True)
    os.environ["HF_HOME"] = str(cache_dir)
    snapshot_download(
        repo_id=BERTSCORE_REPO_ID,
        revision=BERTSCORE_REVISION,
        local_dir=model_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "sentencepiece.bpe.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    resolved_model_dir, model_binding = _validated_model_binding(model_dir)
    model_dir_fd: int | None = None
    required_bindings: tuple[
        tuple[str, int, tuple[int, int, int, int]], ...
    ] = ()
    evaluator = None
    try:
        model_dir_fd = _secure_open_model_directory(str(resolved_model_dir))
        opened_model = os.fstat(model_dir_fd)
        if (opened_model.st_dev, opened_model.st_ino) != model_binding[-1][:2]:
            raise RuntimeError("BERTScore model directory binding changed")
        required_bindings = _open_required_model_files(model_dir_fd)
        evaluator = get_bertscore_evaluator(
            BERTScoreConfig(
                model=_descriptor_model_path(model_dir_fd),
                device=device,
                batch_size=4,
            )
        )
        scores = []
        if run_smoke:
            scores = evaluator(
                [
                    "The page asks for an urgent cryptocurrency payment.",
                    "该网页要求用户立即转账。",
                ],
                [
                    "The webpage requests an urgent crypto transfer.",
                    "页面诱导用户马上付款。",
                ],
            )
        _revalidate_required_model_files(model_dir_fd, required_bindings)
        rebound_model_dir, rebound_binding = _validated_model_binding(model_dir)
        if (
            rebound_model_dir != resolved_model_dir
            or rebound_binding != model_binding
        ):
            raise RuntimeError("BERTScore model directory binding changed")
        evaluator_metadata = getattr(evaluator, "raw_metadata", None)
        if evaluator_metadata is None:
            evaluator_metadata = evaluator.metadata
        public_metadata = _project_public_metadata(
            evaluator_metadata,
            model_dir_fd=model_dir_fd,
            require_scorer_hash=run_smoke,
            expected_device=device,
        )
        result = {
            "status": "ready",
            "model_dir": model_cache_key,
            "repo_id": BERTSCORE_REPO_ID,
            "revision": BERTSCORE_REVISION,
            "smoke_scores": scores,
            "metadata": public_metadata,
        }
        (cache_dir / "bertscore_asset_status.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
    finally:
        if evaluator is not None:
            close = getattr(evaluator, "close", None)
            if close is not None:
                close()
        _close_required_model_files(required_bindings)
        if model_dir_fd is not None:
            os.close(model_dir_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pinned multilingual BERTScore assets.")
    parser.add_argument("--cache-dir", default=str(default_model_dir().parent))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_assets(Path(args.cache_dir), args.device, run_smoke=not args.download_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
