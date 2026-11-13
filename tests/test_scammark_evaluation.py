import io
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
SECURE_MODEL_BINDING_SUPPORTED = (
    sys.platform.startswith("linux") and Path("/proc/self/fd").is_dir()
)


class _OfflineBERTScoreEvaluator:
    def __init__(
        self,
        config,
        *,
        metadata_overrides=None,
        hash_model=None,
        hash_suffix=None,
    ):
        from scammark.bertscore_eval import BERTSCORE_REPO_ID

        self._config = config
        self._hash_model = hash_model
        self._hash_suffix = hash_suffix
        self.metadata = {
            "repo_id": BERTSCORE_REPO_ID,
            "revision": config.revision,
            "model": config.model,
            "num_layers": config.num_layers,
            "batch_size": config.batch_size,
            "device_requested": config.device,
            "idf": config.idf,
            "rescale_with_baseline": config.rescale_with_baseline,
        }
        self.metadata.update(metadata_overrides or {})

    def __call__(self, predictions, references):
        if len(predictions) != len(references):
            raise AssertionError("offline evaluator inputs differ in length")
        hash_model = str(self._hash_model or self._config.model)
        hash_suffix = self._hash_suffix or (
            "_L17_no-idf_version=0.3.12(hug_trans=4.57.6)"
        )
        self.metadata.update(
            {
                "device_actual": self._config.device,
                "bert_score_version": "0.3.13",
                "transformers_version": "4.57.6",
                "torch_version": "2.7.0",
                "hash": f"{hash_model}{hash_suffix}",
            }
        )
        return [0.75, 0.80]


def _offline_snapshot_download(*, local_dir, **_kwargs):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "tokenizer.json",
    ):
        (local_dir / filename).write_text("offline fixture\n", encoding="utf-8")


@contextmanager
def _offline_eval_asset_dependencies(
    prepare_eval_assets,
    *,
    evaluator_factory=None,
):
    captured_configs = []

    def fake_evaluator(config):
        captured_configs.append(config)
        if evaluator_factory is None:
            return _OfflineBERTScoreEvaluator(config)
        return evaluator_factory(config)

    fake_huggingface = SimpleNamespace(
        snapshot_download=_offline_snapshot_download
    )
    with mock.patch.dict(
        sys.modules,
        {"huggingface_hub": fake_huggingface},
    ), mock.patch.object(
        prepare_eval_assets,
        "get_bertscore_evaluator",
        side_effect=fake_evaluator,
    ):
        yield captured_configs


class ScamMarkEvaluationTests(unittest.TestCase):
    def test_bertscore_distribution_and_hash_protocol_versions_are_distinct(self):
        from scammark import bertscore_eval

        self.assertEqual(bertscore_eval.BERTSCORE_PACKAGE_VERSION, "0.3.13")
        hash_version = getattr(
            bertscore_eval,
            "BERTSCORE_HASH_VERSION",
            bertscore_eval.BERTSCORE_PACKAGE_VERSION,
        )
        self.assertEqual(hash_version, "0.3.12")
        self.assertNotEqual(
            hash_version,
            bertscore_eval.BERTSCORE_PACKAGE_VERSION,
        )

    def test_bertscore_evaluator_rejects_unsafe_public_device_value(self):
        from scammark.bertscore_eval import (
            BERTScoreConfig,
            BERTScoreEvaluator,
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "trusted-model"
            model_dir.mkdir()
            with self.assertRaises(RuntimeError):
                BERTScoreEvaluator(
                    BERTScoreConfig(
                        model=str(model_dir),
                        device="/home/private/cuda",
                    )
                )

    def test_bertscore_evaluator_closes_model_descriptor_when_score_fails(self):
        from scammark.bertscore_eval import (
            BERTScoreConfig,
            BERTScoreEvaluator,
        )

        if not sys.platform.startswith("linux") or not Path(
            "/proc/self/fd"
        ).is_dir():
            self.skipTest("secure descriptor-backed model paths need Linux procfs")

        class FailingBERTScorer:
            def __init__(self, *, model_type, **_kwargs):
                self.hash = (
                    f"{model_type}_L17_no-idf_version=0.3.12"
                    "(hug_trans=4.57.6)"
                )

            def score(self, _predictions, _references, *, batch_size):
                raise RuntimeError(f"offline score failure at batch {batch_size}")

        fake_torch = SimpleNamespace(
            __version__="2.7.0",
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        fake_bert_score = SimpleNamespace(BERTScorer=FailingBERTScorer)

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "trusted-model"
            model_dir.mkdir()
            evaluator = BERTScoreEvaluator(
                BERTScoreConfig(model=str(model_dir), device="cpu")
            )
            runtime_model = evaluator.raw_metadata["model"]
            self.assertTrue(Path(runtime_model).is_dir())

            with mock.patch.dict(
                sys.modules,
                {"torch": fake_torch, "bert_score": fake_bert_score},
            ), mock.patch(
                "scammark.bertscore_eval.importlib.metadata.version",
                side_effect=lambda package: {
                    "bert-score": "0.3.13",
                    "transformers": "4.57.6",
                }[package],
            ):
                with self.assertRaisesRegex(RuntimeError, "offline score failure"):
                    evaluator(["one"], ["one"])

            self.assertTrue(evaluator.closed)
            self.assertFalse(Path(runtime_model).exists())

    def test_bertscore_evaluator_keeps_descriptor_model_bound_through_score(self):
        from scammark.bertscore_eval import (
            BERTSCORE_REPO_ID,
            BERTScoreConfig,
            BERTScoreEvaluator,
        )

        if not sys.platform.startswith("linux") or not Path(
            "/proc/self/fd"
        ).is_dir():
            self.skipTest("secure descriptor-backed model paths need Linux procfs")

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [0.75]

        observed_markers = []
        observed_model_types = []

        class FakeBERTScorer:
            def __init__(self, *, model_type, **_kwargs):
                self.hash = (
                    f"{model_type}_L17_no-idf_version=0.3.12"
                    "(hug_trans=4.57.6)"
                )
                observed_model_types.append(model_type)
                observed_markers.append(
                    (Path(model_type) / "config.json")
                    .read_text(encoding="utf-8")
                    .strip()
                )

            def score(self, _predictions, _references, *, batch_size):
                self.batch_size = batch_size
                observed_markers.append(
                    (Path(observed_model_types[-1]) / "config.json")
                    .read_text(encoding="utf-8")
                    .strip()
                )
                return None, None, FakeTensor()

        fake_torch = SimpleNamespace(
            __version__="2.7.0",
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        fake_bert_score = SimpleNamespace(BERTScorer=FakeBERTScorer)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "trusted-model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                "trusted\n",
                encoding="utf-8",
            )
            attacker_dir = root / "attacker-model"
            attacker_dir.mkdir()
            (attacker_dir / "config.json").write_text(
                "attacker\n",
                encoding="utf-8",
            )

            evaluator = BERTScoreEvaluator(
                BERTScoreConfig(model=str(model_dir), device="cpu")
            )
            held_original = root / "held-original-model"
            model_dir.rename(held_original)
            attacker_dir.rename(model_dir)
            try:
                with mock.patch.dict(
                    sys.modules,
                    {"torch": fake_torch, "bert_score": fake_bert_score},
                ), mock.patch(
                    "scammark.bertscore_eval.importlib.metadata.version",
                    side_effect=lambda package: {
                        "bert-score": "0.3.13",
                        "transformers": "4.57.6",
                    }[package],
                ):
                    self.assertEqual(evaluator(["one"], ["one"]), [0.75])
                    raw_metadata = evaluator.raw_metadata
                    published_metadata = evaluator.metadata
            finally:
                model_dir.rename(attacker_dir)
                held_original.rename(model_dir)
                close = getattr(evaluator, "close", None)
                if close is not None:
                    close()

        self.assertEqual(observed_markers, ["trusted", "trusted"])
        self.assertEqual(len(observed_model_types), 1)
        self.assertTrue(observed_model_types[0].startswith("/proc/self/fd/"))
        self.assertEqual(raw_metadata["model"], observed_model_types[0])
        self.assertEqual(
            raw_metadata["hash"],
            (
                f"{observed_model_types[0]}_L17_no-idf_version=0.3.12"
                "(hug_trans=4.57.6)"
            ),
        )
        self.assertEqual(published_metadata["model"], BERTSCORE_REPO_ID)
        self.assertEqual(
            published_metadata["hash"],
            (
                f"{BERTSCORE_REPO_ID}_L17_no-idf_version=0.3.12"
                "(hug_trans=4.57.6)"
            ),
        )
        self.assertNotIn(
            "/proc/self/fd/",
            json.dumps(published_metadata, ensure_ascii=False),
        )

    def test_prepare_eval_assets_keeps_original_model_bound_during_smoke(self):
        from scammark import prepare_eval_assets
        from scammark.bertscore_eval import (
            BERTSCORE_REPO_ID,
            BERTSCORE_REVISION,
        )

        if not sys.platform.startswith("linux") or not Path(
            "/proc/self/fd"
        ).is_dir():
            self.skipTest("secure descriptor-backed model paths need Linux procfs")

        observed_config_markers = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "safe-cache"
            model_cache_key = (
                f"FacebookAI_xlm-roberta-large_{BERTSCORE_REVISION}"
            )
            model_dir = cache_dir / model_cache_key
            attacker_dir = root / "attacker-model"
            attacker_dir.mkdir()
            (attacker_dir / "config.json").write_text(
                "attacker\n",
                encoding="utf-8",
            )

            class ReplaceThenRestoreEvaluator(_OfflineBERTScoreEvaluator):
                def __call__(self, predictions, references):
                    held_original = root / "held-original-model"
                    model_dir.rename(held_original)
                    attacker_dir.rename(model_dir)
                    try:
                        observed_config_markers.append(
                            (Path(self._config.model) / "config.json")
                            .read_text(encoding="utf-8")
                            .strip()
                        )
                    finally:
                        model_dir.rename(attacker_dir)
                        held_original.rename(model_dir)
                    self.metadata.update(
                        {
                            "device_actual": self._config.device,
                            "bert_score_version": "0.3.13",
                            "transformers_version": "4.57.6",
                            "torch_version": "2.7.0",
                            "hash": (
                                f"{self._config.model}_L17_no-idf_version="
                                "0.3.12(hug_trans=4.57.6)"
                            ),
                        }
                    )
                    return [0.75, 0.80]

            with _offline_eval_asset_dependencies(
                prepare_eval_assets,
                evaluator_factory=ReplaceThenRestoreEvaluator,
            ):
                result = prepare_eval_assets.prepare_assets(
                    cache_dir,
                    "cuda",
                    run_smoke=True,
                )

            self.assertEqual(observed_config_markers, ["offline fixture"])
            self.assertEqual(result["metadata"]["model"], BERTSCORE_REPO_ID)
            self.assertEqual(
                (model_dir / "config.json").read_text(encoding="utf-8"),
                "offline fixture\n",
            )

    def test_prepare_eval_assets_public_outputs_hide_absolute_model_path(self):
        if not SECURE_MODEL_BINDING_SUPPORTED:
            self.skipTest("secure descriptor-backed model paths need Linux procfs")
        from scammark import prepare_eval_assets
        from scammark.bertscore_eval import (
            BERTSCORE_REPO_ID,
            BERTSCORE_REVISION,
        )
        from scammark.result_contract import scan_for_leaks

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = (
                Path(tmp) / "workspace" / "models" / "huggingface"
            ).resolve()
            with _offline_eval_asset_dependencies(
                prepare_eval_assets
            ) as captured_configs:
                returned = prepare_eval_assets.prepare_assets(
                    cache_dir,
                    "cuda",
                    run_smoke=True,
                )
                stdout = io.StringIO()
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "scammark.prepare_eval_assets",
                        "--cache-dir",
                        str(cache_dir),
                        "--device",
                        "cuda",
                    ],
                ), redirect_stdout(stdout):
                    self.assertEqual(prepare_eval_assets.main(), 0)

            status_path = cache_dir / "bertscore_asset_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())
            public_model_key = (
                f"FacebookAI_xlm-roberta-large_{BERTSCORE_REVISION}"
            )

            self.assertEqual(len(captured_configs), 2)
            self.assertEqual(returned["smoke_scores"], [0.75, 0.80])
            for config in captured_configs:
                self.assertTrue(Path(config.model).is_absolute())
                self.assertTrue(config.model.startswith("/proc/self/fd/"))

            for payload in (returned, status, printed):
                self.assertEqual(payload["status"], "ready")
                self.assertEqual(payload["model_dir"], public_model_key)
                self.assertEqual(payload["repo_id"], BERTSCORE_REPO_ID)
                self.assertEqual(payload["revision"], BERTSCORE_REVISION)
                self.assertEqual(
                    set(payload["metadata"]),
                    {
                        "repo_id",
                        "revision",
                        "model",
                        "num_layers",
                        "batch_size",
                        "device_requested",
                        "idf",
                        "rescale_with_baseline",
                        "device_actual",
                        "bert_score_version",
                        "transformers_version",
                        "torch_version",
                        "hash",
                    },
                )
                self.assertEqual(
                    payload["metadata"]["model"],
                    BERTSCORE_REPO_ID,
                )
                self.assertEqual(
                    payload["metadata"]["repo_id"],
                    BERTSCORE_REPO_ID,
                )
                self.assertEqual(
                    payload["metadata"]["revision"],
                    BERTSCORE_REVISION,
                )
                self.assertEqual(
                    payload["metadata"]["hash"],
                    (
                        f"{BERTSCORE_REPO_ID}_L17_no-idf_version="
                        "0.3.12(hug_trans=4.57.6)"
                    ),
                )
                rendered = json.dumps(payload, ensure_ascii=False)
                self.assertNotIn(str(cache_dir), rendered)
                self.assertNotIn("/workspace/", rendered)
                self.assertNotIn("/home/", rendered)

            scan_for_leaks(status_path)

    def test_prepare_eval_assets_rejects_symlinked_model_path_components(self):
        if not SECURE_MODEL_BINDING_SUPPORTED:
            self.skipTest("secure descriptor-backed model paths need Linux procfs")
        from scammark import prepare_eval_assets
        from scammark.bertscore_eval import BERTSCORE_REVISION

        model_cache_key = (
            f"FacebookAI_xlm-roberta-large_{BERTSCORE_REVISION}"
        )
        for case in ("cache-parent", "model-directory"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                private_root = root / "workspace" / "private-model-root"
                private_root.mkdir(parents=True)
                if case == "cache-parent":
                    cache_dir = root / "cache-link"
                    cache_dir.symlink_to(private_root, target_is_directory=True)
                else:
                    cache_dir = root / "safe-cache"
                    cache_dir.mkdir()
                    private_model = private_root / "resolved-model"
                    private_model.mkdir()
                    (cache_dir / model_cache_key).symlink_to(
                        private_model,
                        target_is_directory=True,
                    )

                with _offline_eval_asset_dependencies(
                    prepare_eval_assets
                ) as captured_configs:
                    with self.assertRaises(RuntimeError):
                        prepare_eval_assets.prepare_assets(
                            cache_dir,
                            "cuda",
                            run_smoke=True,
                        )

                self.assertEqual(captured_configs, [])
                self.assertFalse(
                    (cache_dir / "bertscore_asset_status.json").exists()
                )

    def test_prepare_eval_assets_rejects_linked_required_model_files(self):
        if not SECURE_MODEL_BINDING_SUPPORTED:
            self.skipTest("secure descriptor-backed model paths need Linux procfs")
        from scammark import prepare_eval_assets

        for link_kind in ("symlink", "hardlink"):
            with self.subTest(link_kind=link_kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cache_dir = root / "safe-cache"
                external = root / "external-config.json"
                external.write_text("external\n", encoding="utf-8")

                def unsafe_snapshot_download(*, local_dir, **_kwargs):
                    local_dir = Path(local_dir)
                    local_dir.mkdir(parents=True, exist_ok=True)
                    for filename in (
                        "model.safetensors",
                        "sentencepiece.bpe.model",
                        "tokenizer.json",
                    ):
                        (local_dir / filename).write_text(
                            "offline fixture\n",
                            encoding="utf-8",
                        )
                    config_path = local_dir / "config.json"
                    if link_kind == "symlink":
                        config_path.symlink_to(external)
                    else:
                        config_path.hardlink_to(external)

                fake_huggingface = SimpleNamespace(
                    snapshot_download=unsafe_snapshot_download
                )
                with mock.patch.dict(
                    sys.modules,
                    {"huggingface_hub": fake_huggingface},
                ), mock.patch.object(
                    prepare_eval_assets,
                    "get_bertscore_evaluator",
                ) as evaluator_factory:
                    with self.assertRaises(RuntimeError):
                        prepare_eval_assets.prepare_assets(
                            cache_dir,
                            "cuda",
                            run_smoke=True,
                        )

                evaluator_factory.assert_not_called()
                self.assertFalse(
                    (cache_dir / "bertscore_asset_status.json").exists()
                )

    def test_prepare_eval_assets_rejects_evaluator_identity_or_hash_drift(self):
        if not SECURE_MODEL_BINDING_SUPPORTED:
            self.skipTest("secure descriptor-backed model paths need Linux procfs")
        from scammark import prepare_eval_assets

        cases = {
            "repo-id": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                metadata_overrides={"repo_id": "OtherOrg/other-model"},
            ),
            "revision": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                metadata_overrides={"revision": "0" * 40},
            ),
            "model": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                metadata_overrides={
                    "model": str(root / "workspace" / "other-model")
                },
            ),
            "hash-prefix": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                hash_model=root / "workspace" / "other-model",
            ),
            "num-layers": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                metadata_overrides={"num_layers": 999},
            ),
            "batch-size": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                metadata_overrides={"batch_size": 8},
            ),
            "idf": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                metadata_overrides={"idf": True},
            ),
            "rescale": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                metadata_overrides={"rescale_with_baseline": True},
            ),
            "hash-layers": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                hash_suffix=(
                    "_L999_no-idf_version=0.3.12(hug_trans=4.57.6)"
                ),
            ),
            "hash-idf": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                hash_suffix=(
                    "_L17_idf_version=0.3.12(hug_trans=4.57.6)"
                ),
            ),
            "hash-rescaled": lambda config, root: _OfflineBERTScoreEvaluator(
                config,
                hash_suffix=(
                    "_L17_no-idf_version=0.3.12(hug_trans=4.57.6)"
                    "-rescaled"
                ),
            ),
            "extra-cache-path": (
                lambda config, root: _OfflineBERTScoreEvaluator(
                    config,
                    metadata_overrides={
                        "cache_path": str(root / "workspace" / "cache")
                    },
                )
            ),
        }
        for case, factory in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cache_dir = root / "safe-cache"
                status_path = cache_dir / "bertscore_asset_status.json"
                with _offline_eval_asset_dependencies(
                    prepare_eval_assets,
                    evaluator_factory=lambda config: factory(config, root),
                ):
                    with self.assertRaises(RuntimeError):
                        prepare_eval_assets.prepare_assets(
                            cache_dir,
                            "cuda",
                            run_smoke=True,
                        )

                self.assertFalse(status_path.exists())
    def test_prepared_eval_asset_status_can_build_real_final_archive(self):
        if not SECURE_MODEL_BINDING_SUPPORTED:
            self.skipTest("secure descriptor-backed model paths need Linux procfs")
        from scammark import prepare_eval_assets
        from scammark.result_contract import build_result_archive
        from tests.result_fixture import (
            CompleteResultFixture,
            EXPERIMENT_RELEASE_TAG,
        )

        with CompleteResultFixture() as fixture:
            cache_dir = fixture.root / "workspace" / "models" / "huggingface"
            with _offline_eval_asset_dependencies(prepare_eval_assets):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "scammark.prepare_eval_assets",
                        "--cache-dir",
                        str(cache_dir),
                        "--device",
                        "cuda",
                    ],
                ), redirect_stdout(stdout), redirect_stderr(stderr):
                    self.assertEqual(prepare_eval_assets.main(), 0)

            self.assertEqual(
                json.loads(stdout.getvalue())["smoke_scores"],
                [0.75, 0.80],
            )
            self.assertEqual(stderr.getvalue(), "")
            fixture.finalize_log.write_text(
                stdout.getvalue() + stderr.getvalue(),
                encoding="utf-8",
            )

            built = build_result_archive(
                mode="final",
                full_dir=fixture.full,
                state_dir=fixture.state,
                destination=fixture.final_destination,
                release_tag=EXPERIMENT_RELEASE_TAG,
                repository="nlixsys/ScamWeb-V",
                git_commit="a" * 40,
                bertscore_asset_status_path=(
                    cache_dir / "bertscore_asset_status.json"
                ),
                finalize_log_path=fixture.finalize_log,
            )

            self.assertEqual(built, fixture.final_destination)

    def test_load_engineering_metrics_preserves_server_run_metadata(self):
        from scammark.evaluate import load_engineering_metrics

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metrics_server.json"
            path.write_text(
                json.dumps(
                    {
                        "engineering": {
                            "provider": "Crazyrouter + Alibaba Cloud Model Studio",
                            "api_calls_cumulative": 38852,
                            "elapsed_seconds": 255000,
                        }
                    }
                ),
                encoding="utf-8",
            )

            engineering = load_engineering_metrics(path)

        self.assertEqual(engineering["api_calls_cumulative"], 38852)
        self.assertEqual(
            engineering["provider"],
            "Crazyrouter + Alibaba Cloud Model Studio",
        )
    def test_binary_metrics_report_positive_class_and_confusion_counts(self):
        from scammark.metrics import compute_binary_metrics

        metrics = compute_binary_metrics(
            ["scam", "scam", "benign", "benign"],
            ["scam", "benign", "scam", "benign"],
            bootstrap_samples=0,
        )

        self.assertEqual(metrics["confusion"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertEqual(metrics["support"], {"scam": 2, "benign": 2})
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_binary_metrics_count_invalid_output_as_wrong(self):
        from scammark.metrics import compute_binary_metrics

        metrics = compute_binary_metrics(["benign"], ["garbage"], bootstrap_samples=0)

        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["confusion"], {"tp": 0, "fp": 1, "tn": 0, "fn": 0})
        self.assertEqual(metrics["invalid_prediction_count"], 1)

    def test_subtype_metrics_use_single_label_macro_average(self):
        from scammark.metrics import compute_subtype_metrics

        metrics = compute_subtype_metrics(
            ["A", "B", "C", "C"],
            ["A", "A", "C", "none"],
            labels=["A", "B", "C"],
            bootstrap_samples=0,
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_precision"], 0.5)
        self.assertAlmostEqual(metrics["macro_recall"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 4 / 9)
        self.assertEqual(metrics["per_class"]["B"]["support"], 1)
        self.assertEqual(metrics["confusion_matrix"]["C"]["none"], 1)

    def test_subtype_metrics_report_fixed_and_observed_class_macro(self):
        from scammark.metrics import compute_subtype_metrics

        metrics = compute_subtype_metrics(
            ["A", "B"],
            ["A", "B"],
            labels=["A", "B", "C"],
            bootstrap_samples=0,
        )

        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["observed_class_macro_f1"], 1.0)
        self.assertEqual(metrics["observed_labels"], ["A", "B"])

    def test_threshold_matching_maximizes_true_positive_count(self):
        from scammark.metrics import _threshold_matrix_matching

        matches = _threshold_matrix_matching(
            [[1 / 3, 0.5], [2 / 3, 1.0]],
            threshold=0.5,
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual({(truth, pred) for truth, pred, _ in matches}, {(0, 1), (1, 0)})

    def test_grounding_metrics_penalize_duplicates_misses_and_benign_boxes(self):
        from scammark.metrics import compute_grounding_metrics

        records = [
            self._record(
                "scam-1",
                "scam",
                truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "target"}],
                pred_boxes=[
                    {"xyxy": [0, 0, 10, 10], "reason": "target"},
                    {"xyxy": [20, 20, 30, 30], "reason": "duplicate"},
                ],
            ),
            self._record(
                "scam-2",
                "scam",
                truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "missed"}],
                pred_boxes=[],
            ),
            self._record(
                "benign-1",
                "benign",
                truth_boxes=[],
                pred_boxes=[{"xyxy": [1, 1, 5, 5], "reason": "false alarm"}],
            ),
        ]

        metrics = compute_grounding_metrics(records)

        self.assertEqual(metrics["counts_at_0_5"], {"tp": 1, "fp": 2, "fn": 1})
        self.assertAlmostEqual(metrics["precision_at_0_5"], 1 / 3)
        self.assertAlmostEqual(metrics["recall_at_0_5"], 0.5)
        self.assertAlmostEqual(metrics["f1_at_0_5"], 0.4)
        self.assertAlmostEqual(metrics["miou"], 0.5)
        self.assertAlmostEqual(metrics["page_recall_at_0_3"], 0.5)
        self.assertAlmostEqual(metrics["page_recall_at_0_5"], 0.5)

    def test_grounding_clamps_boxes_and_counts_degenerate_outputs_invalid(self):
        from scammark.metrics import compute_grounding_metrics

        records = [
            self._record(
                "scam-1",
                "scam",
                truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "target"}],
                pred_boxes=[
                    {"xyxy": [-5, -5, 10, 10], "reason": "clamped"},
                    {"xyxy": [5, 5, 5, 8], "reason": "degenerate"},
                    {"xyxy": ["bad", 0, 4, 4], "reason": "non numeric"},
                ],
                image_size=[20, 20],
            )
        ]

        metrics = compute_grounding_metrics(records)

        self.assertEqual(metrics["counts_at_0_5"], {"tp": 1, "fp": 2, "fn": 0})
        self.assertEqual(metrics["invalid_prediction_count"], 2)
        self.assertAlmostEqual(metrics["miou"], 1.0)

    def test_grounding_counts_non_list_prediction_as_invalid_false_positive(self):
        from scammark.metrics import compute_grounding_metrics

        record = self._record("scam-1", "scam", [], [])
        record["prediction"]["boxes"] = "not-a-list"
        metrics = compute_grounding_metrics([record])

        self.assertEqual(metrics["invalid_prediction_count"], 1)
        self.assertEqual(metrics["counts_at_0_5"], {"tp": 0, "fp": 1, "fn": 0})

    def test_strict_schema_replaces_malformed_boxes_and_marks_metrics_unclean(self):
        from scammark.evaluate import evaluate_prediction_records
        from scammark.prediction_schema import normalize_final_prediction

        short = normalize_final_prediction(
            {
                "binary": "scam",
                "scam_type": "Online Shopping Scam",
                "description": "",
                "reasoning": "",
                "boxes": [{"xyxy": [0, 0]}],
                "evidence_used": {},
            }
        )
        self.assertFalse(short["schema_valid"])
        self.assertIsNone(short["boxes"][0]["xyxy"])

        record = self._record(
            "scam-1",
            "scam",
            truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "truth"}],
            pred_boxes=[[0, 0, 10, 10]],
        )
        metrics = evaluate_prediction_records([record], bootstrap_samples=0)
        self.assertEqual(metrics["grounding"]["end_to_end"]["counts_at_0_5"], {"tp": 0, "fp": 1, "fn": 1})
        self.assertFalse(metrics["engineering"]["clean_metrics"])
        self.assertFalse(evaluate_prediction_records([], bootstrap_samples=0)["engineering"]["clean_metrics"])

    def test_invalid_bbox_geometry_counts_as_fp_without_invalidating_record_schema(self):
        from scammark.evaluate import evaluate_prediction_records
        from scammark.prediction_schema import normalize_final_prediction

        prediction = normalize_final_prediction(
            {
                "binary": "scam",
                "scam_type": "Online Shopping Scam",
                "description": "page",
                "reasoning": "risk",
                "boxes": [
                    {
                        "image_index": 0,
                        "bbox_1000": ["bad", 0, 500, 500],
                        "xyxy": None,
                        "reason": "visible but invalid geometry",
                        "evidence_type": "text",
                    }
                ],
                "evidence_used": {},
            }
        )
        record = self._record(
            "scam-1",
            "scam",
            truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "truth"}],
            pred_boxes=prediction["boxes"],
        )
        record["prediction"] = prediction
        metrics = evaluate_prediction_records([record], bootstrap_samples=0)

        self.assertTrue(prediction["schema_valid"])
        self.assertEqual(prediction["boxes"][0]["reason"], "visible but invalid geometry")
        self.assertEqual(prediction["boxes"][0]["schema_error"], "invalid_geometry")
        self.assertEqual(metrics["grounding"]["end_to_end"]["invalid_prediction_count"], 1)
        self.assertEqual(metrics["grounding"]["end_to_end"]["counts_at_0_5"]["fp"], 1)
        self.assertTrue(metrics["engineering"]["clean_metrics"])

    def test_rationale_metrics_score_only_grounded_nonempty_explanations(self):
        from scammark.metrics import compute_rationale_metrics

        records = [
            self._record(
                "scam-1",
                "scam",
                truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "urgent payment request"}],
                pred_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "urgent payment request"}],
            ),
            self._record(
                "scam-2",
                "scam",
                truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "fake certificate"}],
                pred_boxes=[],
            ),
        ]

        metrics = compute_rationale_metrics(
            records,
            bertscore_fn=lambda predictions, references: [0.8 for _ in predictions],
        )

        self.assertEqual(metrics["gold_box_count"], 2)
        self.assertEqual(metrics["scored_pair_count"], 1)
        self.assertAlmostEqual(metrics["grounded_rationale_coverage_at_0_5"], 0.5)
        self.assertAlmostEqual(metrics["rouge_l_f1"], 1.0)
        self.assertAlmostEqual(metrics["bertscore_f1"], 0.8)
        self.assertEqual(metrics["bertscore_status"], "computed")

    def test_rationale_bertscore_status_distinguishes_request_and_coverage(self):
        from scammark.metrics import compute_rationale_metrics

        record = self._record(
            "scam-1",
            "scam",
            truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "truth"}],
            pred_boxes=[],
        )
        not_requested = compute_rationale_metrics([record])
        no_pairs = compute_rationale_metrics(
            [record],
            bertscore_fn=lambda predictions, references: [1.0 for _ in predictions],
        )

        self.assertEqual(not_requested["bertscore_status"], "not_requested")
        self.assertEqual(no_pairs["bertscore_status"], "no_matched_pairs")

    def test_rationale_bertscore_error_is_explicit(self):
        from scammark.metrics import compute_rationale_metrics

        record = self._record(
            "scam-1",
            "scam",
            truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "truth"}],
            pred_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "prediction"}],
        )

        def fail(predictions, references):
            raise RuntimeError("model unavailable")

        metrics = compute_rationale_metrics([record], bertscore_fn=fail)
        self.assertEqual(metrics["bertscore_status"], "error")
        self.assertIn("model unavailable", metrics["bertscore_error"])

    def test_evaluator_reports_end_to_end_and_forced_sections(self):
        from scammark.evaluate import evaluate_prediction_records

        record = self._record(
            "scam-1",
            "scam",
            truth_boxes=[{"xyxy": [0, 0, 10, 10], "reason": "fake price"}],
            pred_boxes=[],
        )
        record["prediction"]["binary"] = "benign"
        record["prediction"]["scam_type"] = "none"
        record["prediction"]["boxes"] = [{"xyxy": [0, 0, 10, 10], "reason": "must be gated"}]
        record["prediction"]["evidence_used"]["prompt_stages"] = {
            "type": {"scam_type": "Online Shopping Scam"},
            "grounding": {"boxes": [{"xyxy": [0, 0, 10, 10], "reason": "fake price"}]},
        }

        metrics = evaluate_prediction_records(
            [record],
            engineering={"backend": "rule", "sample_count": 1},
            bootstrap_samples=0,
            bertscore_fn=lambda predictions, references: [1.0 for _ in predictions],
        )

        self.assertEqual(metrics["subtype"]["end_to_end"]["accuracy"], 0.0)
        self.assertEqual(metrics["subtype"]["forced"]["accuracy"], 1.0)
        self.assertEqual(metrics["grounding"]["end_to_end"]["recall_at_0_5"], 0.0)
        self.assertEqual(metrics["grounding"]["forced"]["recall_at_0_5"], 1.0)
        self.assertEqual(metrics["rationale"]["forced"]["grounded_rationale_coverage_at_0_5"], 1.0)
        self.assertEqual(metrics["engineering"]["backend"], "rule")

    def test_manifest_enrichment_restores_gold_boxes_and_description(self):
        from scammark.evaluate import enrich_truth_from_manifest

        records = [{"sample_id": "one", "truth": {"binary": "scam"}, "prediction": {}}]
        manifest = [
            {
                "sample_id": "one",
                "binary": "scam",
                "scam_type": "Online Shopping Scam",
                "description": "gold page",
                "boxes": [{"xyxy": [1, 2, 3, 4], "reason": "gold region"}],
                "image_path": "page.png",
            }
        ]

        enriched = enrich_truth_from_manifest(records, manifest)

        self.assertEqual(enriched[0]["truth"]["scam_type"], "Online Shopping Scam")
        self.assertEqual(enriched[0]["truth"]["description"], "gold page")
        self.assertEqual(enriched[0]["truth"]["boxes"][0]["reason"], "gold region")
        self.assertEqual(enriched[0]["input"]["image_path"], "page.png")

    def test_manifest_enrichment_rejects_duplicate_and_partial_coverage(self):
        from scammark.evaluate import enrich_truth_from_manifest

        prediction = {"sample_id": "one", "prediction": {}}
        manifest = [
            {"sample_id": "one", "binary": "benign", "scam_type": "none"},
            {"sample_id": "two", "binary": "benign", "scam_type": "none"},
        ]
        with self.assertRaises(ValueError):
            enrich_truth_from_manifest([prediction], manifest)
        partial = enrich_truth_from_manifest([prediction], manifest, allow_partial=True)
        self.assertEqual(len(partial), 1)
        with self.assertRaises(ValueError):
            enrich_truth_from_manifest([prediction, dict(prediction)], manifest, allow_partial=True)

    def test_human_ratings_aggregate_four_dimensions(self):
        from scammark.evaluate import aggregate_human_ratings

        rows = []
        for rater, score in [("r1", 4), ("r2", 5), ("r3", 3)]:
            rows.append(
                {
                    "item_id": "item-1",
                    "candidate_id": "A",
                    "rater_id": rater,
                    "correctness": score,
                    "faithfulness": score,
                    "specificity": score,
                    "helpfulness": score,
                }
            )

        metrics = aggregate_human_ratings(rows, bootstrap_samples=0, alpha_fn=lambda matrix: 0.75)

        self.assertEqual(metrics["rating_count"], 3)
        self.assertAlmostEqual(metrics["candidates"]["A"]["correctness"]["mean"], 4.0)
        self.assertAlmostEqual(metrics["candidates"]["A"]["correctness"]["alpha_ordinal"], 0.75)
        self.assertEqual(metrics["dimensions"], ["correctness", "faithfulness", "specificity", "helpfulness"])

    def test_human_ratings_reject_duplicate_rows_and_fractional_scores(self):
        from scammark.evaluate import aggregate_human_ratings

        row = {
            "item_id": "item-1",
            "candidate_id": "A",
            "rater_id": "r1",
            "correctness": 4,
            "faithfulness": 4,
            "specificity": 4,
            "helpfulness": 4,
        }
        with self.assertRaises(ValueError):
            aggregate_human_ratings([row, dict(row)], bootstrap_samples=0, alpha_fn=lambda matrix: 1.0)
        fractional = dict(row)
        fractional["correctness"] = 4.5
        with self.assertRaises(ValueError):
            aggregate_human_ratings([fractional], bootstrap_samples=0, alpha_fn=lambda matrix: 1.0)

    @staticmethod
    def _record(sample_id, binary, truth_boxes, pred_boxes, image_size=None):
        truth = {
            "binary": binary,
            "scam_type": "Online Shopping Scam" if binary == "scam" else "none",
            "boxes": truth_boxes,
            "description": "",
        }
        if image_size:
            truth["image_size"] = image_size
        return {
            "sample_id": sample_id,
            "truth": truth,
            "prediction": {
                "binary": binary,
                "scam_type": truth["scam_type"],
                "boxes": pred_boxes,
                "description": "",
                "reasoning": "",
                "evidence_used": {"prompt_stages": {}},
            },
        }


if __name__ == "__main__":
    unittest.main()
