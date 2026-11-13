"""Real-filesystem result package fixture used by offline contract tests."""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Iterable


INFERENCE_ASSET_NAME = "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
FINAL_ASSET_NAME = "scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz"
INFERENCE_RELEASE_ROOT = "scammark-qwen3-dashscope-full-v2-inference"
FINAL_RELEASE_ROOT = "scammark-qwen3-dashscope-full-v2-final-evaluation"
EXPERIMENT_RELEASE_TAG = "experiment-qwen3-dashscope-full-v2"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class CompleteResultFixture:
    """Create a complete mode-aware v2 result tree without mocks."""

    def __init__(self, sample_count: int = 5658) -> None:
        if type(sample_count) is not int or sample_count < 0:
            raise ValueError("sample_count must be a nonnegative integer")
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._root = Path(self._temporary_directory.name)
        self._sample_count = sample_count
        self._full = self._root / "full"
        self._state = self._root / "state_full"
        self._logs = self._root / "logs"
        self._config = (
            self._root / "config" / "qwen3_dashscope_server_full_v2.toml"
        )
        self._frozen = self._root / "frozen"
        self._dist = self._root / "dist"
        self._destination = self._dist / INFERENCE_ASSET_NAME
        self._final_destination = self._dist / FINAL_ASSET_NAME
        self._log = self._logs / "full_run.log"
        self._finalize_log = self._logs / "bertscore_finalize.log"
        self._bertscore_asset_status = (
            self._frozen / "bertscore_asset_status.json"
        )
        self._write_tree()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def full(self) -> Path:
        return self._full

    @property
    def state(self) -> Path:
        return self._state

    @property
    def log(self) -> Path:
        return self._log

    @property
    def finalize_log(self) -> Path:
        return self._finalize_log

    @property
    def config(self) -> Path:
        return self._config

    @property
    def frozen(self) -> Path:
        return self._frozen

    @property
    def bertscore_asset_status(self) -> Path:
        return self._bertscore_asset_status

    @property
    def destination(self) -> Path:
        return self._destination

    @property
    def final_destination(self) -> Path:
        return self._final_destination

    def __enter__(self) -> "CompleteResultFixture":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._temporary_directory.cleanup()

    def _write_tree(self) -> None:
        from scammark.taxonomy import CANONICAL_LABELS

        sample_ids = tuple(
            f"sample-{index}" for index in range(self._sample_count)
        )
        _write_jsonl(
            self._full / "manifest.jsonl",
            ({"sample_id": sample_id} for sample_id in sample_ids),
        )
        _write_jsonl(
            self._full / "predictions.jsonl",
            ({"sample_id": sample_id, "label": "benign"} for sample_id in sample_ids),
        )
        _write_json(
            self._full / "metrics.json",
            {
                "engineering": {
                    "sample_count": self._sample_count,
                    "prediction_sample_count": self._sample_count,
                    "acceptance_gates": {
                        "passed": True,
                        "checks": {"full_sample_count_exact": True},
                    },
                }
            },
        )
        _write_json(
            self._full / "metrics_final.json",
            {
                "engineering": {"prediction_sample_count": 5658},
                "rationale": {
                    "end_to_end": {"bertscore_status": "computed"}
                },
            },
        )
        _write_json(self._full / "preflight.json", {"passed": True})
        (self._full / "full_summary.md").write_text(
            "# Complete offline result\n",
            encoding="utf-8",
        )
        (self._full / "final_evaluation_summary.md").write_text(
            "# Complete final evaluation\n",
            encoding="utf-8",
        )
        (self._full / "human_evaluation_candidates.csv").write_text(
            "sample_id,decision\nsample-0,benign\n",
            encoding="utf-8",
        )
        (self._full / "bertscore_environment.txt").write_text(
            "bert-score==0.3.13\n",
            encoding="utf-8",
        )
        _write_jsonl(
            self._full / "run_events.jsonl",
            ({"event": "completed", "status": "ok"},),
        )

        state_contract = {
            "config": {"experiment": {"name": "offline-fixture"}},
            "run_mode": "full",
            "dataset_path_mapping": {"runtime_root": "/private/machine/path"},
            "frozen_asset_verification": {"verified": True},
            "source_sha256": {
                "manifest": "1" * 64,
                "web_evidence": "2" * 64,
                "dataset_version": "5" * 64,
                "config": "6" * 64,
            },
            "implementation_sha256": {"run_experiment.py": "3" * 64},
            "contract_sample_ids": list(sample_ids),
            "selected_image_sha256": {
                sample_id: "4" * 64 for sample_id in sample_ids
            },
            "taxonomy": list(CANONICAL_LABELS),
            "provider_response_id": "private-provider-id",
        }
        config_hash = hashlib.sha256(
            json.dumps(
                state_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        _write_json(
            self._state / "run_config.json",
            {"config_hash": config_hash, "config": state_contract},
        )
        _write_jsonl(
            self._state / "api_call_ledger.jsonl",
            (
                {
                    "event": "reserved",
                    "call_id": "a" * 32,
                    "unix_time": 1783900800.0,
                    "config_hash": config_hash,
                    "context": {
                        "sample_id": "sample-0",
                        "stage": "url",
                        "provider": "private-provider",
                    },
                    "machine_path": "/private/machine/path",
                },
                {
                    "event": "finished",
                    "call_id": "a" * 32,
                    "unix_time": 1783900801.0,
                    "config_hash": config_hash,
                    "status": "transport_error",
                    "metadata": {
                        "error_category": "transient",
                        "message": "unit-private-error-detail",
                    },
                    "provider_response_id": "private-provider-id",
                    "machine_path": "/private/machine/path",
                },
            ),
        )
        self._log.parent.mkdir(parents=True, exist_ok=True)
        self._log.write_text("full run completed\n", encoding="utf-8")
        self._finalize_log.write_text(
            "final evaluation completed\n",
            encoding="utf-8",
        )
        self._config.parent.mkdir(parents=True, exist_ok=True)
        real_config = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "qwen3_dashscope_server_full_v2.toml"
        )
        self._config.write_bytes(real_config.read_bytes())
        _write_json(self._frozen / "dataset_version.json", {"version": "v2"})
        _write_json(
            self._frozen / "verification_report.json",
            {"verified": True},
        )
        _write_json(
            self._bertscore_asset_status,
            {"status": "ready", "pinned": True},
        )
        self._dist.mkdir(parents=True, exist_ok=True)

    def build_archives(self) -> Path:
        """Build both v2 archives and return their containing directory."""

        from scammark.result_contract import build_result_archive

        common: dict[str, Any] = {
            "full_dir": self._full,
            "state_dir": self._state,
            "log_path": self._log,
            "config_path": self._config,
            "frozen_dir": self._frozen,
            "repository": "nlixsys/ScamWeb-V",
            "git_commit": "a" * 40,
            "secret_values": (),
        }
        build_result_archive(
            mode="inference",
            destination=self._destination,
            release_tag=EXPERIMENT_RELEASE_TAG,
            **common,
        )
        build_result_archive(
            mode="final",
            destination=self._final_destination,
            release_tag=EXPERIMENT_RELEASE_TAG,
            bertscore_asset_status_path=self._bertscore_asset_status,
            finalize_log_path=self._finalize_log,
            **common,
        )
        return self._dist
