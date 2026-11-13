from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import signal
import socket
import tempfile
from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Wl2nWQAAAAASUVORK5CYII="
)


class FullRunFixture:
    def __init__(
        self,
        sample_count: int,
        output_name: str = "scammark_qwen3_dashscope_server_full_v2",
    ) -> None:
        if type(sample_count) is not int or sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        self.sample_count = sample_count
        self.output_name = str(output_name)
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._stack: contextlib.ExitStack | None = None
        self.root: Path | None = None
        self.config_path: Path | None = None
        self.output_root: Path | None = None
        self.v1_root: Path | None = None
        self._runtime_root: Path | None = None
        self._frozen_root: Path | None = None
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._requests: list[tuple[str, str]] = []
        self._socket_attempts: list[str] = []
        self._opened_paths: list[Path] = []
        self._outcomes: dict[str, dict[str, Any]] = {}
        self._active_sample_id = ""
        self._active_stage = ""
        self._state_cache: Any | None = None
        self._run_manifest: Path | None = None
        self._reject_v1_opens = False
        self._v1_open_baseline = 0
        self._stop_before_sample: str | None = None
        self._stop_signum = signal.SIGINT
        self._clients: list[Any] = []
        self._critical_events: list[str] = []
        self._drop_sample_complete_ids: set[str] = set()
        self._prediction_reads = 0
        self._prediction_snapshots: list[list[dict[str, Any]]] = []
        self._artifact_checks: list[tuple[str, bool]] = []

    def __enter__(self) -> "FullRunFixture":
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self._stack = contextlib.ExitStack()
        self._stack.callback(self._temporary.cleanup)
        self._prepare_files()

        original_cwd = Path.cwd()
        os.chdir(self.root)
        self._stack.callback(os.chdir, original_cwd)
        self._stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "SCAMWEB_DATASET_ROOT": str(self._runtime_root),
                    "SCAMMARK_VLM_BASE_URL": (
                        "https://fixture.cn-beijing.maas.aliyuncs.com/"
                        "compatible-mode/v1"
                    ),
                },
                clear=False,
            )
        )
        self._stack.enter_context(
            patch("socket.create_connection", new=self._forbidden_socket)
        )

        original_open = Path.open

        def guarded_open(path: Path, *args: Any, **kwargs: Any):
            resolved = path.resolve(strict=False)
            self._opened_paths.append(resolved)
            if self._reject_v1_opens and self._is_within(resolved, self.v1_root):
                raise AssertionError(f"v1 fixture path opened: {resolved}")
            return original_open(path, *args, **kwargs)

        self._stack.enter_context(patch.object(Path, "open", new=guarded_open))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()

    @staticmethod
    def _is_within(path: Path, root: Path | None) -> bool:
        if root is None:
            return False
        try:
            path.relative_to(root.resolve(strict=False))
        except ValueError:
            return False
        return True

    def _require_entered(self) -> None:
        if self.root is None or self.config_path is None or self.output_root is None:
            raise RuntimeError("FullRunFixture must be entered before use")

    def _prepare_files(self) -> None:
        assert self.root is not None
        source_config = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "qwen3_dashscope_server_full_v2.toml"
        )
        self._runtime_root = self.root / "dataset_runtime"
        self._frozen_root = self.root / "dataset_frozen_v1"
        self.output_root = self.root / "outputs" / self.output_name
        self.v1_root = (
            self.root
            / "outputs"
            / "scammark_qwen3_dashscope_server_full"
        )
        image_dir = self._runtime_root / "images"
        image_dir.mkdir(parents=True)
        self._frozen_root.mkdir(parents=True)

        manifest: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        for index in range(1, self.sample_count + 1):
            sample_id = str(index)
            (image_dir / f"{sample_id}.png").write_bytes(_PNG_1X1)
            manifest.append(
                {
                    "sample_id": sample_id,
                    "binary": "benign",
                    "scam_type": "none",
                    "description": "fixture benign page",
                    "boxes": [],
                    "label_source": "fixture",
                    "url": f"https://example.test/{sample_id}",
                    "title": f"Fixture page {sample_id}",
                    "image_path": (
                        f"D:/datasets/scamweb_dataset/images/{sample_id}.png"
                    ),
                }
            )
            evidence.append(
                {
                    "sample_id": sample_id,
                    "title": f"Fixture page {sample_id}",
                    "visible_text": "ordinary fixture content",
                    "buttons": [],
                    "forms": [],
                    "links": [],
                    "meta": {},
                    "parse_status": "ok",
                }
            )

        manifest_path = self._frozen_root / "scamweb_v_manifest.jsonl"
        evidence_path = self._frozen_root / "web_evidence.jsonl"
        self._write_jsonl(manifest_path, manifest)
        self._write_jsonl(evidence_path, evidence)
        version = {
            "dataset_version": "fixture-v1",
            "sample_count": self.sample_count,
            "gold_box_count": 0,
            "sha256": {
                "frozen_manifest": self._sha256(manifest_path),
                "frozen_evidence": self._sha256(evidence_path),
            },
        }
        (self._frozen_root / "dataset_version.json").write_text(
            json.dumps(version, ensure_ascii=False), encoding="utf-8"
        )

        config_text = source_config.read_text(encoding="utf-8")
        replacements = {
            "expected_full_sample_count = 5658": (
                f"expected_full_sample_count = {self.sample_count}"
            ),
            'manifest = "outputs/dataset_frozen_v1/scamweb_v_manifest.jsonl"': (
                f'manifest = "{manifest_path.as_posix()}"'
            ),
            'web_evidence = "outputs/dataset_frozen_v1/web_evidence.jsonl"': (
                f'web_evidence = "{evidence_path.as_posix()}"'
            ),
            'dataset_version = "outputs/dataset_frozen_v1/dataset_version.json"': (
                f'dataset_version = "{(self._frozen_root / "dataset_version.json").as_posix()}"'
            ),
            'output_root = "outputs/scammark_qwen3_dashscope_server_full_v2"': (
                f'output_root = "{self.output_root.as_posix()}"'
            ),
            'image_cache = "outputs/scammark_qwen3_dashscope_server_full_v2/image_cache"': (
                f'image_cache = "{(self.output_root / "image_cache").as_posix()}"'
            ),
        }
        for before, after in replacements.items():
            if before not in config_text:
                raise AssertionError(f"fixture config replacement missing: {before}")
            config_text = config_text.replace(before, after, 1)
        self.config_path = self.root / "fixture_full_v2.toml"
        self.config_path.write_text(config_text, encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _forbidden_socket(self, address: Any, *args: Any, **kwargs: Any):
        self._socket_attempts.append(str(address))
        raise AssertionError(f"fixture attempted socket connection: {address}")

    def _stage_payload(self, stage: str, sample_id: str) -> dict[str, Any]:
        outcome = self._outcomes.get(sample_id, {})
        binary = str(outcome.get("binary") or "benign")
        payloads: dict[str, dict[str, Any]] = {
            "url": {
                "risk_level": "low",
                "domain_observations": [],
                "brand_mismatch": [],
                "suspicious_tokens": [],
                "evidence_summary": "fixture url evidence",
            },
            "text": {
                "persuasion_signals": [],
                "claims": [],
                "requested_actions": [],
                "candidate_scam_types": ["none"],
                "evidence_summary": "fixture text evidence",
            },
            "visual": {
                "page_genre": "other",
                "visual_scam_cues": [],
                "sensitive_or_monetary_inputs": [],
                "impersonated_or_trust_elements": [],
                "social_contact_or_redirects": [],
                "adult_or_romance_lures": [],
                "candidate_regions": [],
                "evidence_summary": "fixture visual evidence",
            },
            "cross": {
                "consistency_findings": [],
                "conflicts": [],
                "supporting_matches": [],
                "risk_level": "low",
                "evidence_summary": "fixture cross evidence",
            },
            "binary": {
                "binary": binary,
                "reasoning": "fixture binary evidence",
            },
            "type": {
                "scam_type_id": 0,
                "scam_type": "none",
                "reasoning": "fixture type evidence",
            },
            "grounding": {
                "description": "fixture page",
                "reasoning": "fixture grounding evidence",
                "boxes": [],
            },
        }
        return payloads[stage]

    def _transport(self, **request: Any) -> tuple[str, dict[str, Any]]:
        from scammark.reliability import (
            PermanentRequestError,
            TransientRequestError,
            provider_429_failure_category,
        )

        sample_id = self._active_sample_id
        stage = self._active_stage
        self._requests.append((sample_id, stage))
        outcome = self._outcomes.get(sample_id, {})
        kind = str(outcome.get("kind") or "success")
        if kind == "http_error":
            status = int(outcome.get("status") or 500)
            message = str(outcome.get("message") or "fixture provider error")
            error = f"HTTP {status}: {message}"
            if (
                status in {408, 429}
                and not (
                    status == 429
                    and provider_429_failure_category(error)
                    == "auth_billing"
                )
            ) or status >= 500:
                raise TransientRequestError(error)
            raise PermanentRequestError(error)
        if kind == "schema_error":
            return str(outcome.get("message") or "{}"), {
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "finish_reason": "stop",
                "response_model": request["model"],
            }
        if kind == "unprintable_transient":
            hostile = str(
                outcome.get("message") or "UNPRINTABLE_TRANSIENT_SENTINEL"
            )

            class UnprintableTransientError(TransientRequestError):
                def __str__(self) -> str:
                    raise RuntimeError(hostile)

            raise UnprintableTransientError("fixture transient failure")
        payload = self._stage_payload(stage, sample_id)
        return json.dumps(payload), {
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "finish_reason": "stop",
            "response_model": request["model"],
            "response_id": f"fixture-{sample_id}-{stage}",
            "http_status": 200,
        }

    def _client_class(self):
        from scammark.model_client import OpenAIAPIModelClient

        fixture = self

        class ScriptedOpenAIAPIModelClient(OpenAIAPIModelClient):
            def complete_json_with_metadata(
                self, stage: str, prompt: str, sample: dict[str, Any], **kwargs: Any
            ):
                fixture._active_sample_id = str(
                    sample.get("sample_id") or sample.get("id") or ""
                )
                fixture._active_stage = stage
                try:
                    return super().complete_json_with_metadata(
                        stage=stage,
                        prompt=prompt,
                        sample=sample,
                        **kwargs,
                    )
                finally:
                    fixture._active_sample_id = ""
                    fixture._active_stage = ""

        return ScriptedOpenAIAPIModelClient

    def _make_client(self, *args: Any, **kwargs: Any):
        client_class = self._client_class()
        client = client_class(*args, transport=self._transport, **kwargs)
        self._clients.append(client)
        return client

    def _recording_logger_class(self):
        from scammark.observability import RunEventLogger

        fixture = self

        class RecordingRunEventLogger(RunEventLogger):
            def emit(self, event: str, *, critical: bool = False, **fields: Any):
                if (
                    event == "sample_complete"
                    and str(fields.get("sample_id") or "")
                    in fixture._drop_sample_complete_ids
                ):
                    fixture._drop_sample_complete_ids.discard(
                        str(fields.get("sample_id") or "")
                    )
                    return None
                if critical:
                    fixture._critical_events.append(event)
                if event in {
                    "prediction_flushed",
                    "metrics_written",
                    "run_completed",
                }:
                    assert fixture.output_root is not None
                    full = fixture.output_root / "full"
                    if event == "prediction_flushed":
                        present = (full / "predictions.jsonl").is_file()
                    elif event == "metrics_written":
                        present = (full / "metrics.json").is_file()
                    else:
                        present = all(
                            (full / name).is_file()
                            for name in (
                                "predictions.jsonl",
                                "metrics.json",
                                "human_evaluation_candidates.csv",
                                "full_summary.md",
                            )
                        )
                    fixture._artifact_checks.append((event, present))
                return super().emit(event, critical=critical, **fields)

        return RecordingRunEventLogger

    def _recording_atomic_jsonl(self, writer: Any):
        def recording(path: Path, records: list[dict[str, Any]]) -> None:
            writer(path, records)
            assert self.output_root is not None
            prediction_path = self.output_root / "full" / "predictions.jsonl"
            if Path(path).resolve() != prediction_path.resolve():
                return
            with open(prediction_path, encoding="utf-8") as handle:
                snapshot = [
                    json.loads(line) for line in handle if line.strip()
                ]
            self._prediction_snapshots.append(snapshot)

        return recording

    def _recording_jsonl_loader(self, loader: Any):
        def recording(path: Path):
            assert self.output_root is not None
            prediction_path = self.output_root / "full" / "predictions.jsonl"
            if Path(path).resolve() == prediction_path.resolve():
                self._prediction_reads += 1
            return loader(path)

        return recording

    def _stoppable_pipeline_predict(self, predict: Any):
        def stoppable(pipeline: Any, sample: dict[str, Any], *args: Any, **kwargs: Any):
            sample_id = str(sample.get("sample_id") or sample.get("id") or "")
            if self._stop_before_sample == sample_id:
                self._stop_before_sample = None
                pipeline.stop_controller.handle_signal(
                    self._stop_signum, None
                )
            return predict(pipeline, sample, *args, **kwargs)

        return stoppable

    def _state(self):
        self._require_entered()
        if self._state_cache is not None:
            return self._state_cache

        from scammark.dataset_paths import DatasetPathMapper
        from scammark.experiment_state import ExperimentState
        from scammark.frozen_assets import verify_frozen_assets
        from scammark.run_experiment import (
            _implementation_source_hashes,
            _select_full_inputs,
            _selected_image_hashes,
            _sha256,
            load_experiment_config,
            sanitized_contract_config,
        )
        from scammark.taxonomy import CANONICAL_LABELS

        assert self.config_path is not None
        assert self.output_root is not None
        config = load_experiment_config(self.config_path)
        manifest_path = Path(config["data"]["manifest"])
        evidence_path = Path(config["data"]["web_evidence"])
        version_path = Path(config["data"]["dataset_version"])
        mapper = DatasetPathMapper(
            source_root=str(config["data"]["source_dataset_root"]),
            runtime_root=str(config["data"]["runtime_dataset_root"]),
        )
        run_manifest, _, contract_records = _select_full_inputs(
            manifest_path,
            evidence_path,
            self.output_root / "full",
            expected_size=self.sample_count,
            path_mapper=mapper,
        )
        self._run_manifest = run_manifest
        state_contract = {
            "config": sanitized_contract_config(config),
            "run_mode": "full",
            "dataset_path_mapping": mapper.metadata,
            "frozen_asset_verification": verify_frozen_assets(
                manifest_path.parent
            ),
            "source_sha256": {
                "manifest": _sha256(manifest_path),
                "web_evidence": _sha256(evidence_path),
                "dataset_version": _sha256(version_path),
                "config": _sha256(self.config_path),
            },
            "implementation_sha256": _implementation_source_hashes(),
            "contract_sample_ids": [
                str(record.get("sample_id")) for record in contract_records
            ],
            "selected_image_sha256": _selected_image_hashes(contract_records),
            "taxonomy": list(CANONICAL_LABELS),
        }
        self._state_cache = ExperimentState(
            self.output_root / "state_full", state_contract
        )
        return self._state_cache

    def _seed_stages(
        self, sample_id: str, complete_stages: Sequence[str]
    ) -> None:
        from scammark.evidence_loader import load_samples
        from scammark.image_preprocessing import ImagePolicy
        from scammark.pipeline import ScamMarkPipeline
        from scammark.run_experiment import load_experiment_config

        self._require_entered()
        state = self._state()
        assert self.config_path is not None
        assert self._run_manifest is not None
        assert self.output_root is not None
        config = load_experiment_config(self.config_path)
        samples = load_samples(
            self._run_manifest, self.output_root / "full" / "web_evidence.jsonl"
        )
        sample = next(
            row for row in samples if str(row.get("sample_id")) == str(sample_id)
        )
        self._outcomes[str(sample_id)] = {"kind": "success", "binary": "scam"}
        image = config["images"]
        client = self._make_client(
            model=str(config["models"]["text"]),
            text_model=str(config["models"]["text"]),
            vision_model=str(config["models"]["vision"]),
            text_base_url=str(config["endpoints"]["text_base_url"]),
            vision_base_url=str(config["endpoints"]["vision_base_url"]),
            strict_routing=True,
            allow_env_image_override=False,
        )
        request_start = len(self._requests)
        ScamMarkPipeline(
            model_client=client,
            prompt_mode="staged",
            experiment_state=state,
            input_token_limits={
                key: int(value) for key, value in config["input_tokens"].items()
            },
            output_token_limits={
                key: int(value) for key, value in config["output_tokens"].items()
            },
            image_cache_dir=Path(config["paths"]["image_cache"]),
            image_policy=ImagePolicy(
                overview_pixels=int(image["overview_pixels"]),
                total_pixels=int(image["total_pixels"]),
                max_long_side=int(image["max_long_side"]),
                max_tiles=int(image["max_tiles"]),
                overlap_ratio=float(image["overlap_ratio"]),
                max_base64_bytes=int(image["max_base64_bytes"]),
                contact_sheet_pixels=int(image["contact_sheet_pixels"]),
                contact_sheet_base64_bytes=int(
                    image["contact_sheet_base64_bytes"]
                ),
            ),
            max_stage_attempts_across_resume=int(
                config["experiment"]["max_stage_attempts_across_resume"]
            ),
        ).predict(sample)
        all_stages = state.load_stages(str(sample_id))
        selected = {
            stage: all_stages[stage]
            for stage in complete_stages
            if stage in all_stages
        }
        if len(selected) != len(tuple(complete_stages)):
            raise AssertionError("fixture could not seed every requested stage")
        state._save_stages(str(sample_id), selected)
        del self._requests[request_start:]
        client.close()

    def run(self, resume: bool) -> dict[str, Any]:
        from scammark import run_experiment

        self._require_entered()
        assert self.config_path is not None
        atomic_writer = run_experiment._atomic_write_jsonl
        jsonl_loader = run_experiment.load_jsonl
        pipeline_predict = run_experiment.ScamMarkPipeline.predict
        with patch.object(
            run_experiment, "OpenAIAPIModelClient", new=self._make_client
        ), patch.object(
            run_experiment.ScamMarkPipeline,
            "predict",
            new=self._stoppable_pipeline_predict(pipeline_predict),
        ), patch.object(
            run_experiment,
            "RunEventLogger",
            new=self._recording_logger_class(),
        ), patch.object(
            run_experiment,
            "_atomic_write_jsonl",
            new=self._recording_atomic_jsonl(atomic_writer),
        ), patch.object(
            run_experiment,
            "load_jsonl",
            new=self._recording_jsonl_loader(jsonl_loader),
        ), patch.object(
            run_experiment,
            "_acceptance_gates",
            return_value={"passed": True, "checks": {"fixture": True}},
        ), contextlib.redirect_stdout(self._stdout), contextlib.redirect_stderr(
            self._stderr
        ):
            return run_experiment.run(
                self.config_path, None, resume, validate_only=False
            )

    def run_main(self, resume: bool, ack_breaker: bool = False) -> int:
        from scammark import run_experiment

        self._require_entered()
        assert self.config_path is not None
        arguments = Namespace(
            config=str(self.config_path),
            pilot=None,
            full=True,
            resume=resume,
            validate_only=False,
            ack_breaker=ack_breaker,
        )
        atomic_writer = run_experiment._atomic_write_jsonl
        jsonl_loader = run_experiment.load_jsonl
        pipeline_predict = run_experiment.ScamMarkPipeline.predict
        with patch.object(run_experiment, "parse_args", return_value=arguments), patch.object(
            run_experiment, "OpenAIAPIModelClient", new=self._make_client
        ), patch.object(
            run_experiment.ScamMarkPipeline,
            "predict",
            new=self._stoppable_pipeline_predict(pipeline_predict),
        ), patch.object(
            run_experiment,
            "RunEventLogger",
            new=self._recording_logger_class(),
        ), patch.object(
            run_experiment,
            "_atomic_write_jsonl",
            new=self._recording_atomic_jsonl(atomic_writer),
        ), patch.object(
            run_experiment,
            "load_jsonl",
            new=self._recording_jsonl_loader(jsonl_loader),
        ), patch.object(
            run_experiment,
            "_acceptance_gates",
            return_value={"passed": True, "checks": {"fixture": True}},
        ), contextlib.redirect_stdout(self._stdout), contextlib.redirect_stderr(
            self._stderr
        ):
            return run_experiment.main()

    def respond(self, *, sample_id: str, outcome: Mapping[str, Any]) -> None:
        self._outcomes[str(sample_id)] = dict(outcome)

    def seed_complete_checkpoint(self, sample_id: str) -> None:
        self._seed_stages(
            str(sample_id),
            ("url", "text", "visual", "cross", "binary", "type", "grounding"),
        )

    def seed_partial_checkpoint(
        self, sample_id: str, complete_stages: Sequence[str]
    ) -> None:
        self._seed_stages(str(sample_id), tuple(complete_stages))

    def seed_predictions(self, count: int) -> None:
        self._require_entered()
        if type(count) is not int or not 0 <= count <= self.sample_count:
            raise ValueError("prediction count is outside the fixture manifest")
        assert self.output_root is not None
        rows = [
            {
                "sample_id": str(index),
                "prediction": {"binary": "benign"},
                "fixture_seed": True,
            }
            for index in range(1, count + 1)
        ]
        self._write_jsonl(self.output_root / "full" / "predictions.jsonl", rows)

    def seed_sample_events(self, backend_errors: int) -> None:
        from scammark.observability import (
            ObservabilitySettings,
            RunEventLogger,
        )

        self._require_entered()
        assert self.output_root is not None
        if type(backend_errors) is not int or not 0 <= backend_errors <= self.sample_count:
            raise ValueError("backend error count is outside the fixture manifest")
        for index in range(1, backend_errors + 1):
            self._seed_stages(
                str(index),
                ("url", "text", "visual", "cross", "binary", "type", "grounding"),
            )
        logger = RunEventLogger(
            self.output_root / "full" / "run_events.jsonl",
            ObservabilitySettings(),
            stream=self._stdout,
            attempt_id="fixture-prior",
        )
        for index in range(1, backend_errors + 1):
            logger.emit(
                "sample_complete",
                sample_index=index,
                sample_total=self.sample_count,
                sample_id=str(index),
                result="__invalid__",
                api_calls=index,
                api_call_limit=50_000,
                elapsed_seconds=index,
                health="degraded",
                backend_error=True,
            )

    def seed_breaker_opened(self) -> None:
        from scammark.observability import ObservabilitySettings, RunEventLogger

        self._require_entered()
        assert self.output_root is not None
        logger = RunEventLogger(
            self.output_root / "full" / "run_events.jsonl",
            ObservabilitySettings(),
            stream=self._stdout,
            attempt_id="fixture-breaker",
        )
        logger.emit("breaker_opened", consecutive=5, sample_id="5")

    def request_stop_before_sample(self, sample_id: str) -> None:
        self._stop_before_sample = str(sample_id)

    def create_v1_sentinel_that_raises_on_open(self) -> None:
        self._require_entered()
        assert self.v1_root is not None
        self.v1_root.mkdir(parents=True, exist_ok=True)
        (self.v1_root / "SENTINEL_DO_NOT_OPEN").write_text(
            "v1 sentinel", encoding="utf-8"
        )
        self._v1_open_baseline = sum(
            self._is_within(path, self.v1_root) for path in self._opened_paths
        )
        self._reject_v1_opens = True

    def load_events(self) -> list[dict[str, Any]]:
        self._require_entered()
        assert self.output_root is not None
        path = self.output_root / "full" / "run_events.jsonl"
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def load_predictions(self) -> list[dict[str, Any]]:
        self._require_entered()
        assert self.output_root is not None
        path = self.output_root / "full" / "predictions.jsonl"
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def calls_for(self, sample_id: str) -> list[str]:
        return [stage for current, stage in self._requests if current == str(sample_id)]

    def requested_sample_ids(self) -> list[str]:
        return list(dict.fromkeys(sample_id for sample_id, _ in self._requests))

    def all_durable_and_human_text(self) -> str:
        self._require_entered()
        assert self.output_root is not None
        parts = [self._stdout.getvalue(), self._stderr.getvalue()]
        if self.output_root.exists():
            for path in sorted(self.output_root.rglob("*")):
                if path.is_file() and self._is_within(path.resolve(), self.output_root):
                    try:
                        parts.append(path.read_text(encoding="utf-8"))
                    except UnicodeDecodeError:
                        continue
        return "\n".join(parts)

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def stdout(self) -> io.StringIO:
        return self._stdout

    @property
    def socket_attempt_count(self) -> int:
        return len(self._socket_attempts)

    @property
    def prediction_read_count(self) -> int:
        return self._prediction_reads

    @property
    def prediction_snapshots(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        return tuple(tuple(dict(row) for row in rows) for rows in self._prediction_snapshots)

    @property
    def artifact_checks(self) -> tuple[tuple[str, bool], ...]:
        return tuple(self._artifact_checks)

    @property
    def v1_open_count(self) -> int:
        return sum(
            self._is_within(path, self.v1_root) for path in self._opened_paths
        ) - self._v1_open_baseline
