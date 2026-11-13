"""Pinned, cached multilingual BERTScore evaluator."""

from __future__ import annotations

import importlib.metadata
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BERTSCORE_REPO_ID = "FacebookAI/xlm-roberta-large"
BERTSCORE_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
BERTSCORE_NUM_LAYERS = 17
BERTSCORE_PACKAGE_VERSION = "0.3.13"
BERTSCORE_HASH_VERSION = "0.3.12"
TRANSFORMERS_PACKAGE_VERSION = "4.57.6"
_PROC_FD_PREFIX = "/proc/self/fd/"


def _require_secure_local_model_support() -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if (
        not sys.platform.startswith("linux")
        or not directory_flag
        or not nofollow_flag
        or os.open not in getattr(os, "supports_dir_fd", ())
    ):
        raise RuntimeError(
            "Secure local BERTScore model binding is unavailable"
        )
    try:
        proc_metadata = os.stat("/proc/self/fd", follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(
            "Secure local BERTScore model binding is unavailable"
        ) from exc
    if not stat.S_ISDIR(proc_metadata.st_mode):
        raise RuntimeError(
            "Secure local BERTScore model binding is unavailable"
        )


def _descriptor_number(model: str) -> int | None:
    if not model.startswith(_PROC_FD_PREFIX):
        return None
    suffix = model[len(_PROC_FD_PREFIX) :]
    if not suffix or not suffix.isascii() or not suffix.isdecimal():
        return None
    return int(suffix, 10)


def _descriptor_model_path(descriptor: int) -> str:
    return f"{_PROC_FD_PREFIX}{descriptor}"


def _secure_open_model_directory(model: str | os.PathLike[str]) -> int:
    """Open a local model directory without following pathname links."""

    _require_secure_local_model_support()
    model_text = os.fspath(model)
    if type(model_text) is not str or not model_text or "\x00" in model_text:
        raise RuntimeError("BERTScore model path is invalid")

    source_descriptor = _descriptor_number(model_text)
    if source_descriptor is not None:
        try:
            descriptor = os.dup(source_descriptor)
            os.set_inheritable(descriptor, False)
        except OSError as exc:
            raise RuntimeError(
                "BERTScore model descriptor could not be duplicated"
            ) from exc
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise RuntimeError(
                    "BERTScore model descriptor is not a directory"
                )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    try:
        absolute_model = os.path.abspath(model_text)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("BERTScore model path is invalid") from exc
    parts = Path(absolute_model).parts
    if not parts or parts[0] != os.path.sep:
        raise RuntimeError("BERTScore model path is not absolute")

    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(os.path.sep, flags)
        for component in parts[1:]:
            next_descriptor = os.open(
                component,
                flags,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RuntimeError("BERTScore model path is not a directory")
        return descriptor
    except BaseException as exc:
        if descriptor is not None:
            os.close(descriptor)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            "BERTScore model directory could not be bound safely"
        ) from exc


def _is_local_model(model: str) -> bool:
    model_path = Path(model)
    return model_path.is_absolute() or model_path.parts[:1] == ("models",)


def _validate_runtime_device(device: object) -> str:
    if type(device) is not str:
        raise RuntimeError("BERTScore device is invalid")
    valid = device in {"cpu", "mps", "cuda"}
    if device.startswith("cuda:"):
        index = device.removeprefix("cuda:")
        valid = bool(index) and index.isascii() and index.isdecimal()
    if not valid:
        raise RuntimeError("BERTScore device is invalid")
    return device


def _safe_public_runtime_label(
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
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in value
        )
    ):
        raise RuntimeError(f"BERTScore {field} metadata is invalid")
    return value


def default_model_dir() -> Path:
    return Path("models/huggingface") / f"FacebookAI_xlm-roberta-large_{BERTSCORE_REVISION}"


@dataclass(frozen=True)
class BERTScoreConfig:
    model: str = str(default_model_dir())
    revision: str = BERTSCORE_REVISION
    num_layers: int = BERTSCORE_NUM_LAYERS
    batch_size: int = 4
    device: str = "cuda"
    idf: bool = False
    rescale_with_baseline: bool = False


class BERTScoreEvaluator:
    def __init__(self, config: BERTScoreConfig):
        _validate_runtime_device(config.device)
        self.config = config
        self._scorer = None
        self._model_dir_fd: int | None = None
        self._closed = False
        runtime_model = config.model
        self._local_model = _is_local_model(runtime_model)
        if self._local_model:
            self._model_dir_fd = _secure_open_model_directory(runtime_model)
            runtime_model = _descriptor_model_path(self._model_dir_fd)
        self._runtime_model = runtime_model
        self._metadata: dict[str, Any] = {
            "repo_id": BERTSCORE_REPO_ID,
            "revision": config.revision,
            "model": runtime_model,
            "num_layers": config.num_layers,
            "batch_size": config.batch_size,
            "device_requested": config.device,
            "idf": config.idf,
            "rescale_with_baseline": config.rescale_with_baseline,
        }

    def _load(self):
        if self._closed:
            raise RuntimeError("BERTScore evaluator is closed")
        if self._scorer is not None:
            return self._scorer
        try:
            try:
                import torch
                from bert_score import BERTScorer
            except ImportError as exc:
                raise RuntimeError(
                    "Install the pinned evaluation environment before enabling BERTScore"
                ) from exc
            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA BERTScore was requested but torch.cuda.is_available() is false"
                )
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            self._scorer = BERTScorer(
                model_type=self._runtime_model,
                num_layers=self.config.num_layers,
                batch_size=self.config.batch_size,
                nthreads=max(1, min(4, os.cpu_count() or 1)),
                all_layers=False,
                idf=self.config.idf,
                device=device,
                rescale_with_baseline=self.config.rescale_with_baseline,
                use_fast_tokenizer=True,
            )
            self._metadata.update(
                {
                    "device_actual": device,
                    "bert_score_version": importlib.metadata.version(
                        "bert-score"
                    ),
                    "transformers_version": importlib.metadata.version(
                        "transformers"
                    ),
                    "torch_version": str(torch.__version__),
                    "hash": str(
                        getattr(self._scorer, "hash", "")
                        or getattr(self._scorer, "_hash", "")
                    ),
                }
            )
            if device.startswith("cuda"):
                self._metadata["cuda_device_name"] = (
                    torch.cuda.get_device_name(torch.cuda.current_device())
                )
            return self._scorer
        except BaseException:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        descriptor = self._model_dir_fd
        self._model_dir_fd = None
        self._closed = True
        if descriptor is not None:
            os.close(descriptor)

    def __del__(self) -> None:
        try:
            if not self._closed:
                self.close()
        except BaseException:
            pass

    @property
    def raw_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def metadata(self) -> dict[str, Any]:
        raw = self.raw_metadata
        loaded_fields = {
            "device_actual",
            "bert_score_version",
            "transformers_version",
            "torch_version",
            "hash",
        }
        base_fields = {
            "repo_id",
            "revision",
            "model",
            "num_layers",
            "batch_size",
            "device_requested",
            "idf",
            "rescale_with_baseline",
        }
        loaded = bool(loaded_fields.intersection(raw))
        required_fields = base_fields | (loaded_fields if loaded else set())
        allowed_fields = set(required_fields)
        if loaded and self.config.device.startswith("cuda"):
            allowed_fields.add("cuda_device_name")
        if set(raw) != required_fields and not (
            set(raw) == required_fields | {"cuda_device_name"}
            and "cuda_device_name" in allowed_fields
        ):
            raise RuntimeError("BERTScore evaluator metadata fields are invalid")

        expected_values = {
            "repo_id": BERTSCORE_REPO_ID,
            "revision": BERTSCORE_REVISION,
            "model": self._runtime_model,
            "num_layers": BERTSCORE_NUM_LAYERS,
            "batch_size": 4,
            "device_requested": self.config.device,
            "idf": False,
            "rescale_with_baseline": False,
        }
        for field, expected in expected_values.items():
            value = raw.get(field)
            if type(value) is not type(expected) or value != expected:
                raise RuntimeError(
                    f"BERTScore evaluator {field} metadata drifted"
                )
        if not self._local_model and self._runtime_model != BERTSCORE_REPO_ID:
            raise RuntimeError("BERTScore evaluator model metadata drifted")

        public: dict[str, Any] = {
            "repo_id": BERTSCORE_REPO_ID,
            "revision": BERTSCORE_REVISION,
            "model": BERTSCORE_REPO_ID,
            "num_layers": BERTSCORE_NUM_LAYERS,
            "batch_size": 4,
            "device_requested": self.config.device,
            "idf": False,
            "rescale_with_baseline": False,
        }
        if not loaded:
            return public

        loaded_expected = {
            "device_actual": self.config.device,
            "bert_score_version": BERTSCORE_PACKAGE_VERSION,
            "transformers_version": TRANSFORMERS_PACKAGE_VERSION,
        }
        for field, expected in loaded_expected.items():
            value = raw.get(field)
            if type(value) is not type(expected) or value != expected:
                raise RuntimeError(
                    f"BERTScore evaluator {field} metadata drifted"
                )
        torch_version = _safe_public_runtime_label(
            raw.get("torch_version"),
            field="torch_version",
            maximum_length=128,
        )
        hash_suffix = (
            f"_L{BERTSCORE_NUM_LAYERS}_no-idf_version="
            f"{BERTSCORE_HASH_VERSION}"
            f"(hug_trans={TRANSFORMERS_PACKAGE_VERSION})"
        )
        expected_hash = f"{self._runtime_model}{hash_suffix}"
        if type(raw.get("hash")) is not str or raw["hash"] != expected_hash:
            raise RuntimeError("BERTScore evaluator hash metadata drifted")
        public.update(
            {
                **loaded_expected,
                "torch_version": torch_version,
                "hash": f"{BERTSCORE_REPO_ID}{hash_suffix}",
            }
        )
        if "cuda_device_name" in raw:
            public["cuda_device_name"] = _safe_public_runtime_label(
                raw["cuda_device_name"],
                field="cuda_device_name",
                maximum_length=256,
            )
        return public

    def __call__(self, predictions: list[str], references: list[str]) -> list[float]:
        if len(predictions) != len(references):
            raise ValueError("BERTScore predictions and references must have equal length")
        if not predictions:
            return []
        scorer = self._load()
        try:
            _, _, f1 = scorer.score(
                predictions,
                references,
                batch_size=self.config.batch_size,
            )
            return [
                float(value)
                for value in f1.detach().cpu().tolist()
            ]
        except BaseException:
            self.close()
            raise


_EVALUATOR_CACHE: dict[BERTScoreConfig, BERTScoreEvaluator] = {}


def get_bertscore_evaluator(config: BERTScoreConfig | None = None) -> BERTScoreEvaluator:
    config = config or BERTScoreConfig()
    if _descriptor_number(config.model) is not None:
        return BERTScoreEvaluator(config)
    evaluator = _EVALUATOR_CACHE.get(config)
    if evaluator is None or evaluator.closed:
        evaluator = BERTScoreEvaluator(config)
        _EVALUATOR_CACHE[config] = evaluator
    return evaluator
