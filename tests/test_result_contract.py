from __future__ import annotations

import json
import hashlib
import io
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields
from unittest import mock
from pathlib import Path
from typing import Any, Iterable


INFERENCE_PAYLOAD = (
    "full/predictions.jsonl",
    "full/metrics.json",
    "full/manifest.jsonl",
    "full/preflight.json",
    "full/full_summary.md",
    "full/human_evaluation_candidates.csv",
    "audit/run_config.sanitized.json",
    "audit/api_call_ledger.sanitized.jsonl",
    "logs/run_events.jsonl",
    "logs/full_run.log",
    "config/qwen3_dashscope_server_full_v2.toml",
    "frozen/dataset_version.json",
    "frozen/verification_report.json",
    "provenance.json",
)

FINAL_PAYLOAD = (
    "full/metrics.json",
    "full/metrics_final.json",
    "full/final_evaluation_summary.md",
    "environment/bertscore_environment.txt",
    "environment/bertscore_asset_status.json",
    "logs/bertscore_finalize.log",
    "provenance.json",
)

RUN_CONFIG_AUDIT_FIELDS = (
    "schema",
    "run_mode",
    "source_sha256",
    "implementation_sha256",
    "contract_sample_ids_sha256",
    "selected_image_sha256_count",
    "taxonomy",
)

LEDGER_AUDIT_FIELDS = (
    "timestamp",
    "event",
    "call_id",
    "sample_id",
    "stage",
    "status",
    "error_category",
    "sanitized_error_message",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _exact_tree_inventory(root: Path) -> tuple[tuple[object, ...], ...]:
    rows = []
    for path in sorted((root, *root.rglob("*"))):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            content: bytes | str | None = os.readlink(path)
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
            content = path.read_bytes()
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            content = None
        else:
            kind = "special"
            content = None
        rows.append(
            (
                path.relative_to(root).as_posix(),
                kind,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_mtime_ns,
                content,
            )
        )
    return tuple(rows)


class _FullContractFixture:
    def __init__(self, sample_count: int = 2) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.full = Path(self._temporary_directory.name)
        self.sample_count = sample_count
        self.sample_ids = [f"sample-{index}" for index in range(sample_count)]
        self.reset()

    def __enter__(self) -> "_FullContractFixture":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._temporary_directory.cleanup()

    def reset(self) -> None:
        _write_jsonl(
            self.full / "manifest.jsonl",
            ({"sample_id": sample_id} for sample_id in self.sample_ids),
        )
        _write_jsonl(
            self.full / "predictions.jsonl",
            ({"sample_id": sample_id} for sample_id in self.sample_ids),
        )
        self.write_metrics(self.valid_metrics())

    def valid_metrics(self) -> dict[str, Any]:
        return {
            "engineering": {
                "sample_count": self.sample_count,
                "prediction_sample_count": self.sample_count,
                "acceptance_gates": {
                    "passed": True,
                    "checks": {"full_sample_count_exact": True},
                },
            }
        }

    def write_metrics(self, payload: object) -> None:
        _write_json(self.full / "metrics.json", payload)

    def write_valid_final_metrics(self) -> None:
        _write_json(
            self.full / "metrics_final.json",
            {
                "engineering": {"prediction_sample_count": 5658},
                "rationale": {
                    "end_to_end": {"bertscore_status": "computed"}
                },
            },
        )


class ResultContractTests(unittest.TestCase):
    @staticmethod
    def _api():
        from scammark.result_contract import (
            ResultContractError,
            validate_full_contract,
        )

        return ResultContractError, validate_full_contract

    def test_default_contract_accepts_exact_5658_matching_results(self):
        _, validate_full_contract = self._api()

        with _FullContractFixture(sample_count=5658) as fixture:
            result = validate_full_contract(fixture.full)

        self.assertEqual(result.mode, "inference")
        self.assertEqual(result.sample_count, 5658)

    def test_contract_rejects_any_count_other_than_expected(self):
        ResultContractError, validate_full_contract = self._api()

        for sample_count in (1, 3):
            with self.subTest(sample_count=sample_count):
                with _FullContractFixture(sample_count=sample_count) as fixture:
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(fixture.full, expected_count=2)

    def test_contract_rejects_missing_required_files(self):
        ResultContractError, validate_full_contract = self._api()

        for filename in ("manifest.jsonl", "predictions.jsonl", "metrics.json"):
            with self.subTest(filename=filename):
                with _FullContractFixture(sample_count=1) as fixture:
                    (fixture.full / filename).unlink()
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(fixture.full, expected_count=1)

    def test_contract_reader_rejects_fifo_and_parent_symlink_without_blocking(self):
        ResultContractError, validate_full_contract = self._api()

        if hasattr(os, "mkfifo"):
            with _FullContractFixture(sample_count=1) as fixture:
                manifest = fixture.full / "manifest.jsonl"
                manifest.unlink()
                os.mkfifo(manifest)
                with self.assertRaises(ResultContractError):
                    validate_full_contract(fixture.full, expected_count=1)

        with _FullContractFixture(sample_count=1) as fixture:
            linked_parent = fixture.full / "linked-full"
            linked_parent.symlink_to(fixture.full, target_is_directory=True)
            with self.assertRaises(ResultContractError):
                validate_full_contract(linked_parent, expected_count=1)

    def test_controlled_outputs_link_rejects_physical_v1_parent_before_read(self):
        import scammark.result_contract as result_contract

        ResultContractError, validate_full_contract = self._api()
        with _FullContractFixture(sample_count=1) as fixture:
            physical = (
                fixture.full
                / "scammark_qwen3_dashscope_server_full"
            )
            physical.mkdir()
            v2 = physical / "scammark_qwen3_dashscope_server_full_v2"
            v2.mkdir()
            relocated_full = v2 / "full"
            relocated_full.mkdir()
            for name in ("manifest.jsonl", "predictions.jsonl", "metrics.json"):
                (fixture.full / name).rename(relocated_full / name)
            outputs = fixture.full / "outputs"
            outputs.symlink_to(physical, target_is_directory=True)
            before = {
                path.relative_to(physical).as_posix(): (
                    path.lstat().st_mode,
                    path.lstat().st_nlink,
                    path.lstat().st_mtime_ns,
                    path.read_bytes() if path.is_file() else None,
                )
                for path in (physical, *physical.rglob("*"))
            }
            original_read = result_contract.os.read
            read_reached = False

            def observe_read(*args):
                nonlocal read_reached
                read_reached = True
                return original_read(*args)

            with mock.patch.object(
                result_contract.os,
                "read",
                side_effect=observe_read,
            ):
                with self.assertRaises(ResultContractError):
                    validate_full_contract(
                        outputs
                        / "scammark_qwen3_dashscope_server_full_v2"
                        / "full",
                        expected_count=1,
                    )

            after = {
                path.relative_to(physical).as_posix(): (
                    path.lstat().st_mode,
                    path.lstat().st_nlink,
                    path.lstat().st_mtime_ns,
                    path.read_bytes() if path.is_file() else None,
                )
                for path in (physical, *physical.rglob("*"))
            }
            self.assertFalse(read_reached)
            self.assertEqual(after, before)

    def test_contract_reader_uses_bytes_from_pinned_descriptor_during_swap(self):
        import scammark.result_contract as result_contract

        with _FullContractFixture(sample_count=1) as fixture:
            metrics = fixture.full / "metrics.json"
            replacement = fixture.full / "replacement.json"
            replacement.write_text("{}\n", encoding="utf-8")
            original_open = result_contract._open_pinned_regular_fd
            swap_observed = False

            def open_then_swap(path, **kwargs):
                nonlocal swap_observed
                descriptor, metadata = original_open(path, **kwargs)
                if Path(path) == metrics and not swap_observed:
                    os.replace(replacement, metrics)
                    swap_observed = True
                return descriptor, metadata

            with mock.patch.object(
                result_contract,
                "_open_pinned_regular_fd",
                side_effect=open_then_swap,
            ):
                with self.assertRaises(result_contract.ResultContractError):
                    result_contract.validate_full_contract(
                        fixture.full,
                        expected_count=1,
                    )

        self.assertTrue(swap_observed)

    def test_jsonl_rejects_malformed_blank_non_object_and_duplicate_key_rows(self):
        ResultContractError, validate_full_contract = self._api()
        invalid_contents = (
            "{\n",
            "\n",
            "[]\n",
            '"scalar"\n',
            '{"sample_id":"sample-0","sample_id":"sample-0"}\n',
            '{"sample_id":NaN}\n',
        )

        for filename in ("manifest.jsonl", "predictions.jsonl"):
            for content in invalid_contents:
                with self.subTest(filename=filename, content=content):
                    with _FullContractFixture(sample_count=1) as fixture:
                        (fixture.full / filename).write_text(content, encoding="utf-8")
                        with self.assertRaises(ResultContractError):
                            validate_full_contract(fixture.full, expected_count=1)

    def test_jsonl_rejects_invalid_utf8(self):
        ResultContractError, validate_full_contract = self._api()

        for filename in ("manifest.jsonl", "predictions.jsonl"):
            with self.subTest(filename=filename):
                with _FullContractFixture(sample_count=1) as fixture:
                    (fixture.full / filename).write_bytes(b'{"sample_id":"sample-0"}\xff\n')
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(fixture.full, expected_count=1)

    def test_jsonl_rejects_blank_lines_between_records(self):
        ResultContractError, validate_full_contract = self._api()

        with _FullContractFixture(sample_count=2) as fixture:
            (fixture.full / "manifest.jsonl").write_text(
                '{"sample_id":"sample-0"}\n\n{"sample_id":"sample-1"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ResultContractError):
                validate_full_contract(fixture.full, expected_count=2)

    def test_overlong_json_integer_is_wrapped_as_result_contract_error(self):
        ResultContractError, validate_full_contract = self._api()

        with _FullContractFixture(sample_count=1) as fixture:
            overlong_integer = "9" * 5000
            (fixture.full / "manifest.jsonl").write_text(
                f'{{"sample_id":{overlong_integer}}}\n',
                encoding="utf-8",
            )
            try:
                validate_full_contract(fixture.full, expected_count=1)
            except ResultContractError:
                pass
            except ValueError:
                self.fail("validate_full_contract leaked a bare ValueError")
            else:
                self.fail("validate_full_contract accepted an overlong JSON integer")

    def test_sample_ids_must_be_nonempty_strings(self):
        ResultContractError, validate_full_contract = self._api()
        invalid_values = (None, "", "   ", 1, True, 1.0, [], {})

        for filename in ("manifest.jsonl", "predictions.jsonl"):
            for sample_id in invalid_values:
                with self.subTest(filename=filename, sample_id=sample_id):
                    with _FullContractFixture(sample_count=1) as fixture:
                        _write_jsonl(fixture.full / filename, ({"sample_id": sample_id},))
                        with self.assertRaises(ResultContractError):
                            validate_full_contract(fixture.full, expected_count=1)

            with self.subTest(filename=filename, sample_id="missing"):
                with _FullContractFixture(sample_count=1) as fixture:
                    _write_jsonl(fixture.full / filename, ({},))
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(fixture.full, expected_count=1)

    def test_sample_ids_must_be_unique_in_each_jsonl_file(self):
        ResultContractError, validate_full_contract = self._api()

        for filename in ("manifest.jsonl", "predictions.jsonl"):
            with self.subTest(filename=filename):
                with _FullContractFixture(sample_count=2) as fixture:
                    _write_jsonl(
                        fixture.full / filename,
                        ({"sample_id": "duplicate"}, {"sample_id": "duplicate"}),
                    )
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(fixture.full, expected_count=2)

    def test_manifest_and_prediction_id_sets_must_match(self):
        ResultContractError, validate_full_contract = self._api()

        with _FullContractFixture(sample_count=2) as fixture:
            _write_jsonl(
                fixture.full / "predictions.jsonl",
                ({"sample_id": "sample-0"}, {"sample_id": "other"}),
            )
            with self.assertRaises(ResultContractError):
                validate_full_contract(fixture.full, expected_count=2)

    def test_metrics_must_be_strict_json_with_required_object_shape(self):
        ResultContractError, validate_full_contract = self._api()
        invalid_payloads = (
            "{\n",
            "[]\n",
            "{}\n",
            '{"engineering":[]}\n',
            (
                '{"engineering":{"sample_count":1,"sample_count":1,'
                '"prediction_sample_count":1,"acceptance_gates":'
                '{"passed":true,"checks":{"full_sample_count_exact":true}}}}\n'
            ),
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with _FullContractFixture(sample_count=1) as fixture:
                    (fixture.full / "metrics.json").write_text(payload, encoding="utf-8")
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(fixture.full, expected_count=1)

    def test_engineering_counts_must_be_exact_integers(self):
        ResultContractError, validate_full_contract = self._api()
        invalid_values = (True, 1.0, float("nan"), float("inf"), float("-inf"), "1", None)

        for field in ("sample_count", "prediction_sample_count"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    with _FullContractFixture(sample_count=1) as fixture:
                        metrics = fixture.valid_metrics()
                        metrics["engineering"][field] = value
                        fixture.write_metrics(metrics)
                        with self.assertRaises(ResultContractError):
                            validate_full_contract(fixture.full, expected_count=1)

    def test_acceptance_gate_and_exact_count_check_must_be_true(self):
        ResultContractError, validate_full_contract = self._api()
        invalid_values = (False, 1, "true", None)

        for field in ("passed", "full_sample_count_exact"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    with _FullContractFixture(sample_count=1) as fixture:
                        metrics = fixture.valid_metrics()
                        gates = metrics["engineering"]["acceptance_gates"]
                        if field == "passed":
                            gates[field] = value
                        else:
                            gates["checks"][field] = value
                        fixture.write_metrics(metrics)
                        with self.assertRaises(ResultContractError):
                            validate_full_contract(fixture.full, expected_count=1)

    def test_acceptance_gate_objects_and_fields_are_required(self):
        ResultContractError, validate_full_contract = self._api()

        mutations = (
            "missing_gates",
            "gates_list",
            "missing_passed",
            "missing_checks",
            "checks_list",
            "missing_exact",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with _FullContractFixture(sample_count=1) as fixture:
                    metrics = fixture.valid_metrics()
                    engineering = metrics["engineering"]
                    if mutation == "missing_gates":
                        del engineering["acceptance_gates"]
                    elif mutation == "gates_list":
                        engineering["acceptance_gates"] = []
                    else:
                        gates = engineering["acceptance_gates"]
                        if mutation == "missing_passed":
                            del gates["passed"]
                        elif mutation == "missing_checks":
                            del gates["checks"]
                        elif mutation == "checks_list":
                            gates["checks"] = []
                        else:
                            del gates["checks"]["full_sample_count_exact"]
                    fixture.write_metrics(metrics)
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(fixture.full, expected_count=1)

    def test_mode_is_limited_to_inference_and_final(self):
        ResultContractError, validate_full_contract = self._api()

        for mode in ("full", "", None, True, 1):
            with self.subTest(mode=mode):
                with _FullContractFixture(sample_count=1) as fixture:
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(fixture.full, expected_count=1, mode=mode)

    def test_final_mode_accepts_exact_final_count_and_computed_bertscore(self):
        _, validate_full_contract = self._api()

        with _FullContractFixture(sample_count=2) as fixture:
            fixture.write_valid_final_metrics()
            result = validate_full_contract(
                fixture.full,
                expected_count=2,
                mode="final",
            )

        self.assertEqual(result.mode, "final")
        self.assertEqual(result.sample_count, 2)

    def test_final_mode_requires_metrics_final_file(self):
        ResultContractError, validate_full_contract = self._api()

        with _FullContractFixture(sample_count=1) as fixture:
            with self.assertRaises(ResultContractError):
                validate_full_contract(fixture.full, expected_count=1, mode="final")

    def test_final_prediction_count_must_be_exact_integer_5658(self):
        ResultContractError, validate_full_contract = self._api()
        invalid_values = (
            True,
            5658.0,
            float("nan"),
            float("inf"),
            5657,
            "5658",
            None,
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with _FullContractFixture(sample_count=1) as fixture:
                    fixture.write_valid_final_metrics()
                    final_metrics = json.loads(
                        (fixture.full / "metrics_final.json").read_text(encoding="utf-8")
                    )
                    final_metrics["engineering"]["prediction_sample_count"] = value
                    _write_json(fixture.full / "metrics_final.json", final_metrics)
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(
                            fixture.full,
                            expected_count=1,
                            mode="final",
                        )

    def test_final_bertscore_status_must_be_computed(self):
        ResultContractError, validate_full_contract = self._api()

        for status in (None, "", "not_requested", True):
            with self.subTest(status=status):
                with _FullContractFixture(sample_count=1) as fixture:
                    fixture.write_valid_final_metrics()
                    final_metrics = json.loads(
                        (fixture.full / "metrics_final.json").read_text(encoding="utf-8")
                    )
                    final_metrics["rationale"]["end_to_end"]["bertscore_status"] = status
                    _write_json(fixture.full / "metrics_final.json", final_metrics)
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(
                            fixture.full,
                            expected_count=1,
                            mode="final",
                        )

    def test_final_metrics_requires_strict_json_and_nested_object_shape(self):
        ResultContractError, validate_full_contract = self._api()
        invalid_payloads = (
            "{\n",
            "[]\n",
            '{"engineering":[],"rationale":{}}\n',
            '{"engineering":{"prediction_sample_count":5658},"rationale":[]}\n',
            '{"engineering":{"prediction_sample_count":5658},"rationale":{"end_to_end":[]}}\n',
            '{"engineering":{"prediction_sample_count":5658},"rationale":{"end_to_end":{}}}\n',
            (
                '{"engineering":{"prediction_sample_count":5658,'
                '"prediction_sample_count":5658},"rationale":{"end_to_end":'
                '{"bertscore_status":"computed"}}}\n'
            ),
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with _FullContractFixture(sample_count=1) as fixture:
                    (fixture.full / "metrics_final.json").write_text(
                        payload,
                        encoding="utf-8",
                    )
                    with self.assertRaises(ResultContractError):
                        validate_full_contract(
                            fixture.full,
                            expected_count=1,
                            mode="final",
                        )


class ResultPackagingContractTests(unittest.TestCase):
    @staticmethod
    def _api():
        from scammark.result_contract import (
            ResultContractError,
            build_result_archive,
            scan_for_leaks,
            validate_tar_members,
            write_audit_projection,
        )

        return (
            ResultContractError,
            build_result_archive,
            scan_for_leaks,
            validate_tar_members,
            write_audit_projection,
        )

    @staticmethod
    def _tar_members(root: str, payload: Iterable[str]) -> list[tarfile.TarInfo]:
        return [tarfile.TarInfo(f"{root}/{relative}") for relative in payload]

    def test_audit_projection_writes_only_exact_allowlisted_fields(self):
        _, _, _, _, write_audit_projection = self._api()
        from tests.result_fixture import CompleteResultFixture

        run_fields = {
            "schema",
            "run_mode",
            "source_sha256",
            "implementation_sha256",
            "contract_sample_ids_sha256",
            "selected_image_sha256_count",
            "taxonomy",
        }
        ledger_fields = {
            "timestamp",
            "event",
            "call_id",
            "sample_id",
            "stage",
            "status",
            "error_category",
            "sanitized_error_message",
        }
        with CompleteResultFixture(sample_count=1) as fixture:
            run_config_path = fixture.state / "run_config.json"
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            run_config["config"].update(
                {
                    "unknown_future_field": "unit-unknown-value",
                    "credential": "unit-credential-value",
                    "endpoint": "https://unit.invalid/v1",
                }
            )
            run_config["config_hash"] = hashlib.sha256(
                json.dumps(
                    run_config["config"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            _write_json(run_config_path, run_config)
            ledger_path = fixture.state / "api_call_ledger.jsonl"
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            for row in ledger_rows:
                row["config_hash"] = run_config["config_hash"]
            ledger_rows[0].update(
                {
                    "request_body": "unit-request-value",
                    "provider_response_id": "unit-provider-value",
                    "error": "unit-error-value",
                }
            )
            _write_jsonl(ledger_path, ledger_rows)

            audit = fixture.root / "audit"
            write_audit_projection(fixture.state, audit)
            self.assertEqual(
                {path.name for path in audit.iterdir()},
                {
                    "run_config.sanitized.json",
                    "api_call_ledger.sanitized.jsonl",
                },
            )
            persisted_run = json.loads(
                (audit / "run_config.sanitized.json").read_text(encoding="utf-8")
            )
            persisted_ledger = [
                json.loads(line)
                for line in (
                    audit / "api_call_ledger.sanitized.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            persisted_text = "\n".join(
                path.read_text(encoding="utf-8") for path in audit.iterdir()
            )

        self.assertEqual(set(persisted_run), run_fields)
        self.assertEqual(persisted_run["schema"], "scammark-run-config-sanitized-v1")
        expected_ids_digest = hashlib.sha256(b'["sample-0"]').hexdigest()
        self.assertEqual(
            persisted_run["contract_sample_ids_sha256"],
            expected_ids_digest,
        )
        self.assertEqual(persisted_run["selected_image_sha256_count"], 1)
        self.assertEqual(len(persisted_ledger), 2)
        self.assertTrue(all(set(row) == ledger_fields for row in persisted_ledger))
        self.assertEqual(
            persisted_ledger[0],
            {
                "timestamp": "2026-07-13T00:00:00Z",
                "event": "reserved",
                "call_id": "a" * 32,
                "sample_id": "sample-0",
                "stage": "url",
                "status": "reserved",
                "error_category": None,
                "sanitized_error_message": None,
            },
        )
        self.assertEqual(persisted_ledger[1]["sample_id"], "sample-0")
        self.assertEqual(persisted_ledger[1]["stage"], "url")
        self.assertEqual(persisted_ledger[1]["status"], "transport_error")
        self.assertEqual(persisted_ledger[1]["error_category"], "transient")
        self.assertEqual(
            persisted_ledger[1]["sanitized_error_message"],
            "<redacted>",
        )
        forbidden = (
            "/private/machine/path",
            "unit-unknown-value",
            "unit-credential-value",
            "https://unit.invalid/v1",
            "unit-request-value",
            "unit-provider-value",
            "unit-error-value",
            "unit-private-error-detail",
        )
        self.assertFalse(any(value in persisted_text for value in forbidden))

    def test_plan_flat_audit_projection_is_safe_and_compatible(self):
        _, _, _, _, write_audit_projection = self._api()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            state.mkdir()
            _write_json(
                state / "run_config.json",
                {
                    "schema": "contract-v1",
                    "run_mode": "full",
                    "dataset_path_mapping": {
                        "runtime_root": "/home/unit-private/dataset"
                    },
                    "implementation_sha256": {"run_experiment.py": "abc"},
                    "unknown_future_field": "PROMPT_SENTINEL",
                },
            )
            _write_jsonl(
                state / "api_call_ledger.jsonl",
                (
                    {
                        "event": "finished",
                        "call_id": "c1",
                        "sample_id": "1",
                        "stage": "url",
                        "status": "transport_error",
                        "error": "Bearer sk-unit-private",
                        "provider_response_id": "provider-private",
                        "unknown": "RAW_RESPONSE_SENTINEL",
                    },
                ),
            )
            audit = root / "audit"
            write_audit_projection(state, audit)
            run_projection = json.loads(
                (audit / "run_config.sanitized.json").read_text(encoding="utf-8")
            )
            ledger_projection = json.loads(
                (audit / "api_call_ledger.sanitized.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            persisted = json.dumps(
                {"run": run_projection, "ledger": ledger_projection},
                ensure_ascii=False,
            )

        self.assertEqual(set(run_projection), set(RUN_CONFIG_AUDIT_FIELDS))
        self.assertEqual(set(ledger_projection), set(LEDGER_AUDIT_FIELDS))
        self.assertEqual(
            run_projection["implementation_sha256"],
            {"run_experiment.py": "abc"},
        )
        self.assertEqual(ledger_projection["status"], "transport_error")
        self.assertEqual(
            ledger_projection["sanitized_error_message"],
            "<redacted>",
        )
        for forbidden in (
            "/home/unit-private",
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "provider-private",
            "sk-unit-private",
        ):
            self.assertNotIn(forbidden, persisted)

        ResultContractError, _, _, _, write_audit_projection = self._api()

        cases = (
            (
                "schema-key",
                "schema",
                "sk-UNITPROBE123",
                "Leak category standalone-key at audit projection #1",
            ),
            (
                "bare-digest-key",
                "digest-key",
                "key",
                "Leak category key-marker at audit projection #1",
            ),
            (
                "workspace-digest-key",
                "digest-key",
                "tenant.cn-shanghai.maas.aliyuncs.com",
                "Leak category workspace-label at audit projection #1",
            ),
            (
                "workspace-sample-id",
                "sample-id",
                "tenant.cn-shanghai.maas.aliyuncs.com",
                "Leak category workspace-label at audit projection #2",
            ),
        )
        for label, target, marker, expected_message in cases:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    state = root / "state"
                    state.mkdir()
                    run_config = {
                        "schema": "contract-v1",
                        "run_mode": "full",
                        "implementation_sha256": {"run_experiment.py": "abc"},
                    }
                    ledger_row = {
                        "event": "finished",
                        "call_id": "c1",
                        "sample_id": "1",
                        "stage": "url",
                        "status": "completed",
                    }
                    if target == "schema":
                        run_config["schema"] = marker
                    elif target == "digest-key":
                        run_config["implementation_sha256"] = {marker: "abc"}
                    else:
                        ledger_row["sample_id"] = marker
                    _write_json(state / "run_config.json", run_config)
                    _write_jsonl(state / "api_call_ledger.jsonl", (ledger_row,))
                    audit = root / "audit"

                    with self.assertRaises(ResultContractError) as caught:
                        write_audit_projection(state, audit)

                    self.assertEqual(str(caught.exception), expected_message)
                    self.assertNotIn(str(root), str(caught.exception))
                    self.assertFalse(
                        (audit / "run_config.sanitized.json").exists()
                    )
                    self.assertFalse(
                        (audit / "api_call_ledger.sanitized.jsonl").exists()
                    )

    def test_audit_projection_rejects_malformed_sources_and_bool_count(self):
        ResultContractError, _, _, _, write_audit_projection = self._api()
        from tests.result_fixture import CompleteResultFixture

        raw_mutations = {
            "run_nonobject": ("run_config.json", "[]\n"),
            "run_duplicate": ("run_config.json", '{"config":{},"config":{}}\n'),
            "run_nan": ("run_config.json", '{"config":{"value":NaN}}\n'),
            "ledger_blank": ("api_call_ledger.jsonl", "\n"),
            "ledger_nonobject": ("api_call_ledger.jsonl", "[]\n"),
            "ledger_duplicate": (
                "api_call_ledger.jsonl",
                '{"event":"a","event":"b"}\n',
            ),
        }
        for label, (filename, content) in raw_mutations.items():
            with self.subTest(case=label):
                with CompleteResultFixture(sample_count=1) as fixture:
                    (fixture.state / filename).write_text(content, encoding="utf-8")
                    with self.assertRaises(ResultContractError):
                        write_audit_projection(
                            fixture.state,
                            fixture.root / "audit",
                        )

        for label, field, value in (
            ("selected_bool", "selected_image_sha256", True),
            ("selected_list", "selected_image_sha256", []),
            ("ids_bool", "contract_sample_ids", True),
            ("ids_nonstring", "contract_sample_ids", [True]),
        ):
            with self.subTest(case=label):
                with CompleteResultFixture(sample_count=1) as fixture:
                    path = fixture.state / "run_config.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["config"][field] = value
                    _write_json(path, payload)
                    with self.assertRaises(ResultContractError):
                        write_audit_projection(fixture.state, fixture.root / "audit")

    def test_audit_projection_rejects_allowlisted_field_injection_safely(self):
        ResultContractError, _, _, _, write_audit_projection = self._api()
        from tests.result_fixture import CompleteResultFixture

        cases = (
            (
                "implementation-path",
                "implementation_sha256",
                {"/workspace/unit-private.py": "3" * 64},
                "unit-private",
            ),
            (
                "uppercase-digest",
                "source_sha256",
                {"manifest": "A" * 64},
                "A" * 64,
            ),
            (
                "taxonomy-marker",
                "taxonomy",
                ["Bearer unit-private"],
                "unit-private",
            ),
            (
                "taxonomy-arbitrary",
                "taxonomy",
                ["innocent-looking-unit-value"],
                "innocent-looking-unit-value",
            ),
            (
                "sample-marker",
                "contract_sample_ids",
                ["raw_response=unit-private"],
                "unit-private",
            ),
            (
                "image-digest",
                "selected_image_sha256",
                {"sample-0": "g" * 64},
                "g" * 64,
            ),
        )
        for label, field, value, forbidden in cases:
            with self.subTest(case=label):
                with CompleteResultFixture(sample_count=1) as fixture:
                    path = fixture.state / "run_config.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["config"][field] = value
                    payload["config_hash"] = hashlib.sha256(
                        json.dumps(
                            payload["config"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    _write_json(path, payload)
                    ledger_path = fixture.state / "api_call_ledger.jsonl"
                    rows = [
                        json.loads(line)
                        for line in ledger_path.read_text(encoding="utf-8").splitlines()
                    ]
                    for row in rows:
                        row["config_hash"] = payload["config_hash"]
                    _write_jsonl(ledger_path, rows)
                    try:
                        write_audit_projection(fixture.state, fixture.root / "audit")
                    except ResultContractError as exc:
                        if forbidden in str(exc):
                            self.fail("audit rejection exposed injected content")
                    else:
                        self.fail(f"allowlisted injection {label} was accepted")

    def test_audit_projection_rejects_inconsistent_or_unfinished_ledger(self):
        ResultContractError, _, _, _, write_audit_projection = self._api()
        from tests.result_fixture import CompleteResultFixture

        for label in (
            "unmatched_reservation",
            "duplicate_finish",
            "unknown_stage",
            "outside_sample",
            "overlong_call_id",
        ):
            with self.subTest(case=label):
                with CompleteResultFixture(sample_count=1) as fixture:
                    ledger_path = fixture.state / "api_call_ledger.jsonl"
                    rows = [
                        json.loads(line)
                        for line in ledger_path.read_text(encoding="utf-8").splitlines()
                    ]
                    if label == "unmatched_reservation":
                        rows = rows[:1]
                    elif label == "duplicate_finish":
                        rows.append(dict(rows[1]))
                    elif label == "unknown_stage":
                        rows[0]["context"]["stage"] = "unit-unknown-stage"
                    elif label == "outside_sample":
                        rows[0]["context"]["sample_id"] = "sample-999"
                    else:
                        rows[0]["call_id"] = "a" * 256
                    _write_jsonl(ledger_path, rows)
                    with self.assertRaises(ResultContractError):
                        write_audit_projection(fixture.state, fixture.root / "audit")

    def test_audit_invalid_shapes_are_contract_errors_without_content_leaks(self):
        ResultContractError, _, _, _, write_audit_projection = self._api()
        from tests.result_fixture import CompleteResultFixture

        marker = "unit-private-shape-value"
        mutations = (
            ("category-list", lambda rows: rows[1]["metadata"].update(
                {"error_category": [marker]}
            )),
            ("category-object", lambda rows: rows[1]["metadata"].update(
                {"error_category": {"value": marker}}
            )),
            ("huge-unix", lambda rows: rows[0].update({"unix_time": 10**1000})),
        )
        for label, mutate in mutations:
            with self.subTest(case=label):
                with CompleteResultFixture(sample_count=1) as fixture:
                    ledger_path = fixture.state / "api_call_ledger.jsonl"
                    rows = [
                        json.loads(line)
                        for line in ledger_path.read_text(encoding="utf-8").splitlines()
                    ]
                    mutate(rows)
                    _write_jsonl(ledger_path, rows)
                    try:
                        write_audit_projection(fixture.state, fixture.root / "audit")
                    except ResultContractError as exc:
                        if marker in str(exc):
                            self.fail("shape error exposed source content")
                    except Exception as exc:
                        self.fail(
                            "invalid shape leaked non-contract exception "
                            f"{type(exc).__name__}"
                        )
                    else:
                        self.fail(f"invalid audit shape {label} was accepted")

        for label, value in (
            ("unix_bool", True),
            ("unix_nan", float("nan")),
            ("unix_inf", float("inf")),
        ):
            with self.subTest(case=label):
                with CompleteResultFixture(sample_count=1) as fixture:
                    path = fixture.state / "api_call_ledger.jsonl"
                    rows = [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                    ]
                    rows[0]["unix_time"] = value
                    _write_jsonl(path, rows)
                    with self.assertRaises(ResultContractError):
                        write_audit_projection(fixture.state, fixture.root / "audit")

    def test_leak_scan_rejects_exact_values_and_markers_without_exposing_content(self):
        ResultContractError, _, scan_for_leaks, _, _ = self._api()
        probes = (
            ("exact-key", "unit-exact-value-7h3", ("unit-exact-value-7h3",)),
            ("exact-endpoint", "https://unit.invalid/v1", ("https://unit.invalid/v1",)),
            ("authorization", "Authorization: unit-value", ()),
            ("bearer", "Bearer unit-value", ()),
            ("key-marker", "api_key=unit-value", ()),
            ("bare-key", "key=unit-value", ()),
            ("standalone-key", "sk-unitStandalone123456", ()),
            ("short-standalone-key", "sk-secret", ()),
            (
                "workspace-host",
                "tenant.cn-beijing.maas.aliyuncs.com",
                (),
            ),
            (
                "workspace-region-host",
                "tenant.cn-shanghai.maas.aliyuncs.com",
                (),
            ),
            ("quoted-json-key", '{"api_key":"unit-value"}', ()),
            ("quoted-json-raw", '{"raw_response":"unit-value"}', ()),
            ("quoted-toml-raw", '"raw_response" = "unit-value"', ()),
            ("data-url", "data:image/png;base64,QUJD", ()),
            ("prompt-marker", "prompt_payload=unit-value", ()),
            ("raw-marker", "raw_response=unit-value", ()),
            ("workspace", "/workspace/private/file", ()),
            ("url-query", "https://unit.invalid/v1?token=value", ()),
        )
        for label, matched_text, supplied in probes:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root / "payload.txt").write_text(
                        matched_text + "\n",
                        encoding="utf-8",
                    )
                    try:
                        scan_for_leaks(root, secret_values=supplied)
                    except ResultContractError as exc:
                        message = str(exc)
                    else:
                        self.fail(f"leak category {label} was accepted")
                    if matched_text in message:
                        self.fail("leak exception exposed matched content")
                    self.assertIn("staged file #", message)
                    self.assertNotIn(str(root), message)

    def test_leak_scan_accepts_safe_files_and_ignores_empty_exact_values(self):
        _, _, scan_for_leaks, _, _ = self._api()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "safe.txt").write_text(
                "ordinary completed result\n",
                encoding="utf-8",
            )
            scan_for_leaks(root, secret_values=("",))

    def test_real_v2_config_is_not_a_leak_false_positive(self):
        _, _, scan_for_leaks, _, _ = self._api()
        config = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "qwen3_dashscope_server_full_v2.toml"
        )

        digests = scan_for_leaks(config, secret_values=())

        self.assertEqual(set(digests), {config.name})
        self.assertEqual(digests[config.name], hashlib.sha256(config.read_bytes()).hexdigest())

    def test_leak_scan_rejects_invalid_secret_value_shapes_safely(self):
        ResultContractError, _, scan_for_leaks, _, _ = self._api()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "safe.txt").write_text("ordinary\n", encoding="utf-8")
            for label, values in (
                ("scalar", "unit-value"),
                ("integer", (1,)),
                ("bool", (True,)),
            ):
                with self.subTest(case=label):
                    with self.assertRaises(ResultContractError):
                        scan_for_leaks(root, secret_values=values)

    def test_tar_validation_accepts_exact_inference_and_final_allowlists(self):
        _, _, _, validate_tar_members, _ = self._api()
        from tests.result_fixture import FINAL_RELEASE_ROOT, INFERENCE_RELEASE_ROOT

        validate_tar_members(
            self._tar_members(INFERENCE_RELEASE_ROOT, INFERENCE_PAYLOAD),
            mode="inference",
        )
        validate_tar_members(
            self._tar_members(FINAL_RELEASE_ROOT, FINAL_PAYLOAD),
            mode="final",
        )

    def test_tar_validation_rejects_unsafe_noncanonical_or_inexact_names(self):
        ResultContractError, _, _, validate_tar_members, _ = self._api()
        from tests.result_fixture import INFERENCE_RELEASE_ROOT

        root = INFERENCE_RELEASE_ROOT
        base = self._tar_members(root, INFERENCE_PAYLOAD)

        cases: dict[str, list[tarfile.TarInfo] | list[str]] = {
            "absolute": ["/absolute/file"],
            "parent": [f"{root}/../escape"],
            "noncanonical": [f"{root}//full/metrics.json"],
            "backslash": [f"{root}\\full\\metrics.json"],
            "multiple_roots": [base[0].name, "other-root/full/metrics.json"],
            "wrong_single_root": self._tar_members("wrong-root", INFERENCE_PAYLOAD),
            "duplicate": [*base, tarfile.TarInfo(base[0].name)],
            "extra": [*base, tarfile.TarInfo(f"{root}/extra.txt")],
            "missing": base[:-1],
        }
        for label, members in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ResultContractError):
                    validate_tar_members(members, mode="inference")

    def test_tar_validation_rejects_special_files_and_escaping_links(self):
        ResultContractError, _, _, validate_tar_members, _ = self._api()
        from tests.result_fixture import INFERENCE_RELEASE_ROOT

        root = INFERENCE_RELEASE_ROOT

        for label, member_type in (
            ("character", tarfile.CHRTYPE),
            ("block", tarfile.BLKTYPE),
            ("fifo", tarfile.FIFOTYPE),
            ("directory", tarfile.DIRTYPE),
        ):
            with self.subTest(case=label):
                members = self._tar_members(root, INFERENCE_PAYLOAD)
                members[0].type = member_type
                with self.assertRaises(ResultContractError):
                    validate_tar_members(members, mode="inference")

        for label, member_type, target in (
            ("absolute_symlink", tarfile.SYMTYPE, "/etc/passwd"),
            ("parent_symlink", tarfile.SYMTYPE, "../full/metrics.json"),
            ("escaping_symlink", tarfile.SYMTYPE, "../../../outside"),
            ("absolute_hardlink", tarfile.LNKTYPE, "/etc/passwd"),
            ("parent_hardlink", tarfile.LNKTYPE, "../full/metrics.json"),
            ("escaping_hardlink", tarfile.LNKTYPE, "../../../outside"),
        ):
            with self.subTest(case=label):
                members = self._tar_members(root, INFERENCE_PAYLOAD)
                members[0].type = member_type
                members[0].linkname = target
                with self.assertRaises(ResultContractError):
                    validate_tar_members(members, mode="inference")

    def test_archive_rejects_unsafe_destination_parents_before_read_or_touch(self):
        import scammark.result_contract as result_contract
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        asset = "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
        for label in ("ordinary-symlink", "physical-v1-symlink"):
            with self.subTest(case=label), CompleteResultFixture() as fixture:
                target_name = (
                    "ordinary-dist-target"
                    if label == "ordinary-symlink"
                    else "scammark_qwen3_dashscope_server_full"
                )
                physical = fixture.root / target_name
                physical.mkdir()
                (physical / "sentinel.bin").write_bytes(
                    b"untouched direct destination sentinel\n"
                )
                (physical / "sentinel-link").symlink_to("sentinel.bin")
                before = _exact_tree_inventory(physical)
                linked = fixture.root / f"linked-{label}"
                linked.symlink_to(physical, target_is_directory=True)
                destination = linked / "release-assets" / "full-v2" / asset
                original_read = result_contract.os.read
                read_reached = False

                def observe_read(*args):
                    nonlocal read_reached
                    read_reached = True
                    return original_read(*args)

                with mock.patch.object(
                    result_contract.os,
                    "read",
                    side_effect=observe_read,
                ):
                    with self.assertRaises(result_contract.ResultContractError):
                        result_contract.build_result_archive(
                            mode="inference",
                            full_dir=fixture.full,
                            state_dir=fixture.state,
                            log_path=fixture.log,
                            config_path=fixture.config,
                            frozen_dir=fixture.frozen,
                            destination=destination,
                            release_tag=EXPERIMENT_RELEASE_TAG,
                            repository="nlixsys/ScamWeb-V",
                            git_commit="a" * 40,
                        )

                self.assertFalse(read_reached)
                after = _exact_tree_inventory(physical)
                if after != before:
                    self.fail(f"archive build touched {label} destination target")
                self.assertFalse((physical / "release-assets").exists())
                self.assertFalse(Path(f"{destination}.sha256").exists())
                self.assertFalse((fixture.full / "delivery_state.json").exists())

    def test_archive_creates_and_uses_a_normal_safe_destination_parent(self):
        _, build_result_archive, _, _, _ = self._api()
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        with CompleteResultFixture() as fixture:
            destination = (
                fixture.root
                / "safe-dist"
                / "release-assets"
                / "full-v2"
                / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
            )

            archive = build_result_archive(
                mode="inference",
                full_dir=fixture.full,
                state_dir=fixture.state,
                log_path=fixture.log,
                config_path=fixture.config,
                frozen_dir=fixture.frozen,
                destination=destination,
                release_tag=EXPERIMENT_RELEASE_TAG,
                repository="nlixsys/ScamWeb-V",
                git_commit="a" * 40,
            )

            self.assertEqual(archive, destination)
            self.assertTrue(archive.is_file())
            self.assertTrue(Path(f"{archive}.sha256").is_file())

    def test_archive_checksum_verifier_is_exact_and_rejects_parent_links_before_read(self):
        import scammark.result_contract as result_contract
        from tests.result_fixture import CompleteResultFixture

        with CompleteResultFixture() as fixture:
            fixture.build_archives()
            archive = fixture.destination
            checksum = Path(f"{archive}.sha256")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            original_checksum = checksum.read_bytes()

            self.assertEqual(
                result_contract.verify_result_archive_checksum(
                    archive,
                    expected_sha256=digest,
                ),
                digest,
            )

            checksum.write_bytes(
                original_checksum
                + ("0" * 64 + "  unexpected-extra-file\n").encode("ascii")
            )
            with self.assertRaises(result_contract.ResultContractError):
                result_contract.verify_result_archive_checksum(
                    archive,
                    expected_sha256=digest,
                )
            checksum.write_bytes(original_checksum)

            physical = (
                fixture.root
                / "scammark_qwen3_dashscope_server_full"
            )
            physical.mkdir()
            shutil.copy2(archive, physical / archive.name)
            shutil.copy2(checksum, physical / checksum.name)
            linked = fixture.root / "linked-archive-verification"
            linked.symlink_to(physical, target_is_directory=True)
            before = _exact_tree_inventory(physical)
            original_read = result_contract.os.read
            read_reached = False

            def observe_read(*args):
                nonlocal read_reached
                read_reached = True
                return original_read(*args)

            with mock.patch.object(
                result_contract.os,
                "read",
                side_effect=observe_read,
            ):
                with self.assertRaises(result_contract.ResultContractError):
                    result_contract.verify_result_archive_checksum(
                        linked / archive.name,
                        expected_sha256=digest,
                    )

            self.assertFalse(read_reached)
            self.assertEqual(_exact_tree_inventory(physical), before)

    def test_archive_builds_exact_mode_payloads_checksums_and_sanitized_audit(self):
        ResultContractError, build_result_archive, _, validate_tar_members, _ = self._api()
        from tests.result_fixture import (
            CompleteResultFixture,
            EXPERIMENT_RELEASE_TAG,
            FINAL_RELEASE_ROOT,
            INFERENCE_RELEASE_ROOT,
        )

        with CompleteResultFixture() as fixture:
            archive_dir = fixture.build_archives()
            self.assertIsInstance(archive_dir, Path)
            self.assertEqual(archive_dir, fixture.destination.parent)
            inference = fixture.destination
            final = fixture.final_destination
            expected = (
                (inference, INFERENCE_RELEASE_ROOT, INFERENCE_PAYLOAD, "inference"),
                (final, FINAL_RELEASE_ROOT, FINAL_PAYLOAD, "final"),
            )
            for archive, release_root, payload, mode in expected:
                with self.subTest(mode=mode):
                    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                    checksum = Path(f"{archive}.sha256").read_text(encoding="ascii")
                    self.assertEqual(checksum, f"{digest}  {archive.name}\n")
                    with tarfile.open(archive, "r:gz") as handle:
                        members = handle.getmembers()
                        validate_tar_members(members, mode=mode)
                        self.assertEqual(
                            {member.name for member in members},
                            {f"{release_root}/{name}" for name in payload},
                        )
                        self.assertTrue(all(member.isreg() for member in members))
                        self.assertTrue(
                            all(
                                member.uid == 0
                                and member.gid == 0
                                and member.uname == ""
                                and member.gname == ""
                                and member.mode == 0o644
                                and member.mtime == 0
                                for member in members
                            )
                        )
                        archived_core = (
                            ("manifest.jsonl", "predictions.jsonl", "metrics.json")
                            if mode == "inference"
                            else ("metrics.json", "metrics_final.json")
                        )
                        for filename in archived_core:
                            archived_member = handle.extractfile(
                                f"{release_root}/full/{filename}"
                            )
                            self.assertIsNotNone(archived_member)
                            self.assertEqual(
                                hashlib.sha256(archived_member.read()).hexdigest(),
                                hashlib.sha256(
                                    (fixture.full / filename).read_bytes()
                                ).hexdigest(),
                            )
                        provenance_member = handle.extractfile(
                            f"{release_root}/provenance.json"
                        )
                        self.assertIsNotNone(provenance_member)
                        provenance = json.loads(provenance_member.read())
                        self.assertEqual(
                            provenance["release_tag"],
                            EXPERIMENT_RELEASE_TAG,
                        )
                        if mode == "inference":
                            run_member = handle.extractfile(
                                f"{release_root}/audit/run_config.sanitized.json"
                            )
                            ledger_member = handle.extractfile(
                                f"{release_root}/audit/api_call_ledger.sanitized.jsonl"
                            )
                            self.assertIsNotNone(run_member)
                            self.assertIsNotNone(ledger_member)
                            audit_text = (
                                run_member.read() + ledger_member.read()
                            ).decode("utf-8")
                            self.assertNotIn("/private/machine/path", audit_text)
                            self.assertNotIn("private-provider-id", audit_text)

            residues = [
                path.name
                for path in fixture.destination.parent.iterdir()
                if path.name.startswith(".") or path.name.endswith(".tmp")
            ]
            self.assertEqual(residues, [])

            original_archive = inference.read_bytes()
            original_checksum = Path(f"{inference}.sha256").read_bytes()
            exact_value = "unit-exact-value-archive-9k2"
            fixture.log.write_text(exact_value + "\n", encoding="utf-8")
            try:
                build_result_archive(
                    mode="inference",
                    full_dir=fixture.full,
                    state_dir=fixture.state,
                    log_path=fixture.log,
                    config_path=fixture.config,
                    frozen_dir=fixture.frozen,
                    destination=inference,
                    release_tag=EXPERIMENT_RELEASE_TAG,
                    repository="nlixsys/ScamWeb-V",
                    git_commit="a" * 40,
                    secret_values=(exact_value,),
                )
            except ResultContractError as exc:
                if exact_value in str(exc):
                    self.fail("archive failure exposed matched content")
            else:
                self.fail("archive accepted an exact supplied secret")
            self.assertEqual(inference.read_bytes(), original_archive)
            self.assertEqual(
                Path(f"{inference}.sha256").read_bytes(),
                original_checksum,
            )

    def test_archive_validates_core_contract_before_replacing_destination(self):
        ResultContractError, build_result_archive, _, _, _ = self._api()
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        with CompleteResultFixture(sample_count=1) as fixture:
            fixture.destination.write_bytes(b"existing-archive")
            checksum_path = Path(f"{fixture.destination}.sha256")
            checksum_path.write_bytes(b"existing-checksum")
            with self.assertRaises(ResultContractError):
                build_result_archive(
                    mode="inference",
                    full_dir=fixture.full,
                    state_dir=fixture.state,
                    log_path=fixture.log,
                    config_path=fixture.config,
                    frozen_dir=fixture.frozen,
                    destination=fixture.destination,
                    release_tag=EXPERIMENT_RELEASE_TAG,
                    repository="nlixsys/ScamWeb-V",
                    git_commit="a" * 40,
                    secret_values=(),
                )
            self.assertEqual(fixture.destination.read_bytes(), b"existing-archive")
            self.assertEqual(checksum_path.read_bytes(), b"existing-checksum")

    def test_archive_rejects_symlinked_or_multiply_linked_sources(self):
        ResultContractError, build_result_archive, _, _, _ = self._api()
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        with CompleteResultFixture() as fixture:
            linked_config = fixture.root / "linked-config.toml"
            os.link(fixture.config, linked_config)
            with self.assertRaises(ResultContractError):
                build_result_archive(
                    mode="inference",
                    full_dir=fixture.full,
                    state_dir=fixture.state,
                    log_path=fixture.log,
                    config_path=linked_config,
                    frozen_dir=fixture.frozen,
                    destination=fixture.destination,
                    release_tag=EXPERIMENT_RELEASE_TAG,
                    repository="nlixsys/ScamWeb-V",
                    git_commit="a" * 40,
                )

            linked_parent = fixture.root / "linked-config-parent"
            linked_parent.symlink_to(fixture.config.parent, target_is_directory=True)
            with self.assertRaises(ResultContractError):
                build_result_archive(
                    mode="inference",
                    full_dir=fixture.full,
                    state_dir=fixture.state,
                    log_path=fixture.log,
                    config_path=linked_parent / fixture.config.name,
                    frozen_dir=fixture.frozen,
                    destination=fixture.destination,
                    release_tag=EXPERIMENT_RELEASE_TAG,
                    repository="nlixsys/ScamWeb-V",
                    git_commit="a" * 40,
                )

            if hasattr(os, "mkfifo"):
                linked_config.unlink()
                os.mkfifo(linked_config)
                with self.assertRaises(ResultContractError):
                    build_result_archive(
                        mode="inference",
                        full_dir=fixture.full,
                        state_dir=fixture.state,
                        log_path=fixture.log,
                        config_path=linked_config,
                        frozen_dir=fixture.frozen,
                        destination=fixture.destination,
                        release_tag=EXPERIMENT_RELEASE_TAG,
                        repository="nlixsys/ScamWeb-V",
                        git_commit="a" * 40,
                    )
            linked_config.unlink()

            linked_config.symlink_to(fixture.config)
            with self.assertRaises(ResultContractError):
                build_result_archive(
                    mode="inference",
                    full_dir=fixture.full,
                    state_dir=fixture.state,
                    log_path=fixture.log,
                    config_path=linked_config,
                    frozen_dir=fixture.frozen,
                    destination=fixture.destination,
                    release_tag=EXPERIMENT_RELEASE_TAG,
                    repository="nlixsys/ScamWeb-V",
                    git_commit="a" * 40,
                )

    def test_archive_rejects_pilot_or_differently_ordered_runtime_state(self):
        ResultContractError, build_result_archive, _, _, _ = self._api()
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        with CompleteResultFixture() as fixture:
            run_config_path = fixture.state / "run_config.json"
            ledger_path = fixture.state / "api_call_ledger.jsonl"
            original_run_config = run_config_path.read_bytes()
            original_ledger = ledger_path.read_bytes()

            for label in ("pilot", "reordered_ids", "flat"):
                with self.subTest(case=label):
                    run_config_path.write_bytes(original_run_config)
                    ledger_path.write_bytes(original_ledger)
                    if label == "flat":
                        _write_json(
                            run_config_path,
                            {
                                "schema": "contract-v1",
                                "run_mode": "full",
                                "implementation_sha256": {
                                    "run_experiment.py": "abc"
                                },
                            },
                        )
                        _write_jsonl(
                            ledger_path,
                            (
                                {
                                    "event": "finished",
                                    "call_id": "c1",
                                    "sample_id": "sample-0",
                                    "stage": "url",
                                    "status": "completed",
                                },
                            ),
                        )
                    else:
                        persisted = json.loads(
                            run_config_path.read_text(encoding="utf-8")
                        )
                        if label == "pilot":
                            persisted["config"]["run_mode"] = "pilot"
                        else:
                            persisted["config"]["contract_sample_ids"].reverse()
                        persisted["config_hash"] = hashlib.sha256(
                            json.dumps(
                                persisted["config"],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        _write_json(run_config_path, persisted)
                        ledger_rows = [
                            json.loads(line)
                            for line in ledger_path.read_text(encoding="utf-8").splitlines()
                        ]
                        for row in ledger_rows:
                            row["config_hash"] = persisted["config_hash"]
                        _write_jsonl(ledger_path, ledger_rows)
                    with self.assertRaises(ResultContractError):
                        build_result_archive(
                            mode="inference",
                            full_dir=fixture.full,
                            state_dir=fixture.state,
                            log_path=fixture.log,
                            config_path=fixture.config,
                            frozen_dir=fixture.frozen,
                            destination=fixture.destination,
                            release_tag=EXPERIMENT_RELEASE_TAG,
                            repository="nlixsys/ScamWeb-V",
                            git_commit="a" * 40,
                        )
                    self.assertFalse(fixture.destination.exists())
                    self.assertFalse(Path(f"{fixture.destination}.sha256").exists())

    def test_archive_rejects_unsafe_provenance_domains(self):
        ResultContractError, build_result_archive, _, _, _ = self._api()
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        valid_repository = "nlixsys/ScamWeb-V"
        valid_commit = "a" * 40
        cases = (
            ("tag-slash", "unsafe/tag", valid_repository, valid_commit),
            ("tag-space", "unsafe tag", valid_repository, valid_commit),
            ("repository-shape", EXPERIMENT_RELEASE_TAG, "single-name", valid_commit),
            ("repository-query", EXPERIMENT_RELEASE_TAG, "owner/repo?x", valid_commit),
            ("commit-uppercase", EXPERIMENT_RELEASE_TAG, valid_repository, "A" * 40),
            ("commit-short", EXPERIMENT_RELEASE_TAG, valid_repository, "a" * 5),
            ("commit-gap", EXPERIMENT_RELEASE_TAG, valid_repository, "a" * 41),
            ("commit-near-full", EXPERIMENT_RELEASE_TAG, valid_repository, "a" * 63),
            ("commit-bool", EXPERIMENT_RELEASE_TAG, valid_repository, True),
        )
        with CompleteResultFixture() as fixture:
            for label, tag, repository, commit in cases:
                with self.subTest(case=label):
                    with self.assertRaises(ResultContractError):
                        build_result_archive(
                            mode="inference",
                            full_dir=fixture.full,
                            state_dir=fixture.state,
                            log_path=fixture.log,
                            config_path=fixture.config,
                            frozen_dir=fixture.frozen,
                            destination=fixture.destination,
                            release_tag=tag,
                            repository=repository,
                            git_commit=commit,
                        )

    def test_archive_accepts_plan_short_and_full_length_git_shas(self):
        _, build_result_archive, _, _, _ = self._api()
        from tests.result_fixture import (
            CompleteResultFixture,
            EXPERIMENT_RELEASE_TAG,
            INFERENCE_RELEASE_ROOT,
        )

        with CompleteResultFixture() as fixture:
            for commit in ("abc123", "a" * 64):
                with self.subTest(length=len(commit)):
                    archive = build_result_archive(
                        mode="inference",
                        full_dir=fixture.full,
                        state_dir=fixture.state,
                        log_path=fixture.log,
                        config_path=fixture.config,
                        frozen_dir=fixture.frozen,
                        destination=fixture.destination,
                        release_tag=EXPERIMENT_RELEASE_TAG,
                        repository="nlixsys/ScamWeb-V",
                        git_commit=commit,
                    )
                    with tarfile.open(archive, "r:gz") as handle:
                        provenance = handle.extractfile(
                            f"{INFERENCE_RELEASE_ROOT}/provenance.json"
                        )
                        self.assertIsNotNone(provenance)
                        self.assertEqual(
                            json.loads(provenance.read())["git_commit"],
                            commit,
                        )

    def test_archive_rejects_noncore_file_swapped_after_leak_scan(self):
        import scammark.result_contract as result_contract
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        with CompleteResultFixture() as fixture:
            fixture.build_archives()
            archive = fixture.destination
            checksum = Path(f"{archive}.sha256")
            original_archive = archive.read_bytes()
            original_checksum = checksum.read_bytes()
            exact_value = "unit-post-scan-value-4m8"
            original_scan = result_contract.scan_for_leaks

            def scan_then_swap(root, **kwargs):
                result = original_scan(root, **kwargs)
                (Path(root) / "logs" / "full_run.log").write_text(
                    exact_value + "\n",
                    encoding="utf-8",
                )
                return result

            caught = None
            with mock.patch.object(
                result_contract,
                "scan_for_leaks",
                side_effect=scan_then_swap,
            ):
                try:
                    result_contract.build_result_archive(
                        mode="inference",
                        full_dir=fixture.full,
                        state_dir=fixture.state,
                        log_path=fixture.log,
                        config_path=fixture.config,
                        frozen_dir=fixture.frozen,
                        destination=archive,
                        release_tag=EXPERIMENT_RELEASE_TAG,
                        repository="nlixsys/ScamWeb-V",
                        git_commit="a" * 40,
                        secret_values=(exact_value,),
                    )
                except result_contract.ResultContractError as exc:
                    caught = exc

            if caught is None:
                with tarfile.open(archive, "r:gz") as handle:
                    swapped = handle.extractfile(
                        "scammark-qwen3-dashscope-full-v2-inference/"
                        "logs/full_run.log"
                    )
                    self.assertIsNotNone(swapped)
                    self.assertIn(exact_value.encode("utf-8"), swapped.read())
                self.fail("archive installed a non-core file swapped after scan")
            if exact_value in str(caught):
                self.fail("post-scan rejection exposed matched content")
            self.assertEqual(archive.read_bytes(), original_archive)
            self.assertEqual(checksum.read_bytes(), original_checksum)


class DeliveryStateContractTests(unittest.TestCase):
    @staticmethod
    def _api():
        from scammark.result_contract import (
            DeliveryState,
            ResultContractError,
            load_delivery_state,
            next_delivery_stages,
            record_delivery_stage,
        )

        return (
            DeliveryState,
            ResultContractError,
            load_delivery_state,
            next_delivery_stages,
            record_delivery_stage,
        )

    @staticmethod
    def _subprocess_environment(**extra: str) -> dict[str, str]:
        environment = dict(os.environ)
        for name in (
            "SCAMMARK_LLM_API_KEY",
            "SCAMMARK_LLM_BASE_URL",
            "SCAMMARK_VLM_API_KEY",
            "SCAMMARK_VLM_BASE_URL",
        ):
            environment.pop(name, None)
        environment.update(extra)
        return environment

    def _run_module(
        self,
        *arguments: str,
        stdin: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "scammark.result_contract", *arguments],
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            env=environment or self._subprocess_environment(),
        )

    def test_delivery_state_is_frozen_exact_and_resume_examples_are_preserved(self):
        DeliveryState, _, _, next_delivery_stages, _ = self._api()

        self.assertEqual(
            tuple(field.name for field in fields(DeliveryState)),
            (
                "mode",
                "packaged",
                "archive_sha256",
                "uploaded",
                "published",
                "last_exit_code",
            ),
        )
        initial = DeliveryState()
        self.assertEqual(next_delivery_stages(initial), ("package", "upload", "publish"))
        upload_failure = DeliveryState(
            packaged=True,
            archive_sha256="abc",
            uploaded=False,
            published=False,
            last_exit_code=31,
        )
        self.assertEqual(next_delivery_stages(upload_failure), ("upload", "publish"))
        publication_failure = DeliveryState(
            packaged=True,
            archive_sha256="abc",
            uploaded=True,
            published=False,
            last_exit_code=32,
        )
        self.assertEqual(next_delivery_stages(publication_failure), ("publish",))
        self.assertEqual(
            next_delivery_stages(
                DeliveryState(
                    packaged=True,
                    archive_sha256="abc",
                    uploaded=True,
                    published=True,
                )
            ),
            (),
        )
        with self.assertRaises(FrozenInstanceError):
            initial.packaged = True

    def test_missing_delivery_load_is_read_only_and_mode_aware(self):
        DeliveryState, _, load_delivery_state, _, _ = self._api()

        with tempfile.TemporaryDirectory() as tmp:
            missing_parent = Path(tmp) / "not-created"
            state_path = missing_parent / "delivery_state.json"
            self.assertEqual(
                load_delivery_state(state_path, mode="inference"),
                DeliveryState(mode="inference"),
            )
            self.assertEqual(
                load_delivery_state(state_path, mode="final"),
                DeliveryState(mode="final"),
            )
            self.assertFalse(missing_parent.exists())

    def test_delivery_states_loads_both_modes_from_one_pinned_snapshot(self):
        import scammark.result_contract as result_contract

        sha = "a" * 64
        legacy_final = {
            "mode": "final",
            "packaged": True,
            "archive_sha256": sha,
            "uploaded": False,
            "published": False,
            "last_exit_code": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "delivery_state.json"
            self.assertIsNone(result_contract.load_delivery_states(state_path))
            self.assertFalse(state_path.exists())

            state_path.write_text(
                json.dumps(legacy_final) + "\n",
                encoding="utf-8",
            )
            original = state_path.read_bytes()
            with mock.patch.object(
                result_contract,
                "_read_delivery_document_at",
                wraps=result_contract._read_delivery_document_at,
            ) as read_document:
                states = result_contract.load_delivery_states(state_path)

            self.assertEqual(read_document.call_count, 1)
            self.assertIsNotNone(states)
            self.assertEqual(
                states,
                {
                    "inference": result_contract.DeliveryState(
                        mode="inference"
                    ),
                    "final": result_contract.DeliveryState(**legacy_final),
                },
            )
            self.assertEqual(state_path.read_bytes(), original)

    def test_delivery_records_independent_modes_and_migrates_legacy_state(self):
        (
            DeliveryState,
            _,
            load_delivery_state,
            _,
            record_delivery_stage,
        ) = self._api()
        inference_sha = "a" * 64
        final_sha = "b" * 64

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "delivery_state.json"
            legacy = {
                "mode": "inference",
                "packaged": True,
                "archive_sha256": inference_sha,
                "uploaded": False,
                "published": False,
                "last_exit_code": 31,
            }
            state_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
            original = state_path.read_bytes()

            self.assertEqual(
                load_delivery_state(state_path, mode="inference"),
                DeliveryState(**legacy),
            )
            self.assertEqual(
                load_delivery_state(state_path, mode="final"),
                DeliveryState(mode="final"),
            )
            self.assertEqual(state_path.read_bytes(), original)

            final = record_delivery_stage(
                state_path,
                mode="final",
                stage="package",
                status="complete",
                archive_sha256=final_sha,
            )
            self.assertEqual(final.archive_sha256, final_sha)
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(set(persisted), {"schema", "modes"})
            self.assertEqual(persisted["schema"], "scammark-delivery-state-v1")
            self.assertEqual(set(persisted["modes"]), {"inference", "final"})
            expected_entry_fields = {
                "mode",
                "packaged",
                "archive_sha256",
                "uploaded",
                "published",
                "last_exit_code",
            }
            self.assertEqual(set(persisted["modes"]["inference"]), expected_entry_fields)
            self.assertEqual(set(persisted["modes"]["final"]), expected_entry_fields)
            self.assertEqual(
                load_delivery_state(state_path, mode="inference").archive_sha256,
                inference_sha,
            )

    def test_delivery_state_rejects_corruption_types_invariants_and_unsafe_files(self):
        _, ResultContractError, load_delivery_state, _, record_delivery_stage = self._api()
        sha = "a" * 64
        valid_entry = {
            "mode": "inference",
            "packaged": True,
            "archive_sha256": sha,
            "uploaded": False,
            "published": False,
            "last_exit_code": 31,
        }
        cases = {
            "duplicate": '{"mode":"inference","mode":"inference"}\n',
            "nan": '{"mode":"inference","packaged":NaN}\n',
            "excessive-json-depth": "[" * 10000 + "0" + "]" * 10000 + "\n",
            "extra": json.dumps({**valid_entry, "future": True}) + "\n",
            "bool-code": json.dumps({**valid_entry, "last_exit_code": True}) + "\n",
            "uppercase-sha": json.dumps({**valid_entry, "archive_sha256": "A" * 64}) + "\n",
            "sha-without-package": json.dumps(
                {
                    **valid_entry,
                    "packaged": False,
                    "archive_sha256": sha,
                    "last_exit_code": 30,
                }
            )
            + "\n",
            "upload-without-package": json.dumps(
                {
                    **valid_entry,
                    "packaged": False,
                    "archive_sha256": None,
                    "uploaded": True,
                    "last_exit_code": None,
                }
            )
            + "\n",
            "wrong-failure-position": json.dumps(
                {**valid_entry, "uploaded": True, "last_exit_code": 31}
            )
            + "\n",
            "envelope-extra": json.dumps(
                {
                    "schema": "scammark-delivery-state-v1",
                    "modes": {
                        "inference": valid_entry,
                        "final": {
                            **valid_entry,
                            "mode": "final",
                            "archive_sha256": "b" * 64,
                        },
                    },
                    "future": True,
                }
            )
            + "\n",
            "envelope-missing-mode": json.dumps(
                {
                    "schema": "scammark-delivery-state-v1",
                    "modes": {"inference": valid_entry},
                }
            )
            + "\n",
            "envelope-key-mode-mismatch": json.dumps(
                {
                    "schema": "scammark-delivery-state-v1",
                    "modes": {
                        "inference": valid_entry,
                        "final": valid_entry,
                    },
                }
            )
            + "\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "delivery_state.json"
            for label, text in cases.items():
                with self.subTest(case=label):
                    state_path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ResultContractError):
                        load_delivery_state(state_path, mode="inference")

            target = root / "target.json"
            target.write_text(json.dumps(valid_entry) + "\n", encoding="utf-8")
            state_path.unlink(missing_ok=True)
            state_path.symlink_to(target)
            with self.assertRaises(ResultContractError):
                load_delivery_state(state_path)
            with self.assertRaises(ResultContractError):
                record_delivery_stage(
                    state_path,
                    stage="upload",
                    status="complete",
                    archive_sha256=sha,
                )
            state_path.unlink()

            os.link(target, state_path)
            with self.assertRaises(ResultContractError):
                load_delivery_state(state_path)
            with self.assertRaises(ResultContractError):
                record_delivery_stage(
                    state_path,
                    stage="upload",
                    status="complete",
                    archive_sha256=sha,
                )
            state_path.unlink()

            if hasattr(os, "mkfifo"):
                os.mkfifo(state_path)
                with self.assertRaises(ResultContractError):
                    load_delivery_state(state_path)
                state_path.unlink()

            real_parent = root / "real-parent"
            real_parent.mkdir()
            (real_parent / "delivery_state.json").write_text(
                json.dumps(valid_entry) + "\n", encoding="utf-8"
            )
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(ResultContractError):
                load_delivery_state(linked_parent / "delivery_state.json")

    def test_delivery_transitions_are_monotonic_idempotent_and_hash_immutable(self):
        (
            _,
            ResultContractError,
            load_delivery_state,
            next_delivery_stages,
            record_delivery_stage,
        ) = self._api()
        sha = "a" * 64

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "delivery_state.json"
            failed_package = record_delivery_stage(
                state_path,
                stage="package",
                status="failed",
                exit_code=30,
            )
            self.assertEqual(failed_package.last_exit_code, 30)
            packaged = record_delivery_stage(
                state_path,
                stage="package",
                status="complete",
                archive_sha256=sha,
            )
            self.assertTrue(packaged.packaged)
            self.assertIsNone(packaged.last_exit_code)
            self.assertEqual(
                record_delivery_stage(
                    state_path,
                    stage="package",
                    status="complete",
                    archive_sha256=sha,
                ),
                packaged,
            )
            with self.assertRaises(ResultContractError):
                record_delivery_stage(
                    state_path,
                    stage="package",
                    status="complete",
                    archive_sha256="b" * 64,
                )
            with self.assertRaises(ResultContractError):
                record_delivery_stage(
                    state_path,
                    stage="package",
                    status="failed",
                    exit_code=30,
                )

            failed_upload = record_delivery_stage(
                state_path,
                stage="upload",
                status="failed",
                archive_sha256=sha,
                exit_code=31,
            )
            self.assertEqual(next_delivery_stages(failed_upload), ("upload", "publish"))
            self.assertTrue(failed_upload.packaged)
            self.assertFalse(failed_upload.uploaded)
            with self.assertRaises(ResultContractError):
                record_delivery_stage(
                    state_path,
                    stage="publish",
                    status="failed",
                    exit_code=32,
                )
            uploaded = record_delivery_stage(
                state_path,
                stage="upload",
                status="complete",
                archive_sha256=sha,
            )
            self.assertTrue(uploaded.uploaded)
            self.assertIsNone(uploaded.last_exit_code)
            failed_publish = record_delivery_stage(
                state_path,
                stage="publish",
                status="failed",
                archive_sha256=sha,
                exit_code=32,
            )
            self.assertTrue(failed_publish.uploaded)
            self.assertFalse(failed_publish.published)
            published = record_delivery_stage(
                state_path,
                stage="publish",
                status="complete",
                archive_sha256=sha,
            )
            self.assertTrue(published.published)
            self.assertIsNone(published.last_exit_code)
            self.assertEqual(next_delivery_stages(published), ())
            self.assertEqual(load_delivery_state(state_path), published)

            for stage, status, exit_code in (
                ("upload", "failed", 31),
                ("publish", "failed", 32),
                ("upload", "failed", 32),
            ):
                with self.subTest(stage=stage, status=status, exit=exit_code):
                    with self.assertRaises(ResultContractError):
                        record_delivery_stage(
                            state_path,
                            stage=stage,
                            status=status,
                            exit_code=exit_code,
                        )

    def test_failed_delivery_record_preserves_bytes_and_leaves_no_temporary_residue(self):
        _, ResultContractError, _, _, record_delivery_stage = self._api()
        sha = "a" * 64

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "delivery_state.json"
            record_delivery_stage(
                state_path,
                stage="package",
                status="complete",
                archive_sha256=sha,
            )
            original = state_path.read_bytes()
            before = {path.name for path in root.iterdir()}
            with self.assertRaises(ResultContractError):
                record_delivery_stage(
                    state_path,
                    stage="upload",
                    status="failed",
                    archive_sha256="b" * 64,
                    exit_code=31,
                )
            self.assertEqual(state_path.read_bytes(), original)
            self.assertEqual({path.name for path in root.iterdir()}, before)

    def test_concurrent_mode_records_do_not_lose_either_entry(self):
        _, _, load_delivery_state, _, _ = self._api()
        inference_sha = "a" * 64
        final_sha = "b" * 64

        with tempfile.TemporaryDirectory() as tmp:
            for iteration in range(3):
                state_path = Path(tmp) / f"delivery_state-{iteration}.json"
                processes = []
                for mode, digest in (
                    ("inference", inference_sha),
                    ("final", final_sha),
                ):
                    processes.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "scammark.result_contract",
                                "record-delivery",
                                "--state-path",
                                str(state_path),
                                "--mode",
                                mode,
                                "--stage",
                                "package",
                                "--status",
                                "complete",
                                "--archive-sha256",
                                digest,
                            ],
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=self._subprocess_environment(),
                        )
                    )
                for process in processes:
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, stderr)
                    self.assertEqual(json.loads(stdout)["packaged"], True)

                self.assertEqual(
                    load_delivery_state(
                        state_path, mode="inference"
                    ).archive_sha256,
                    inference_sha,
                )
                self.assertEqual(
                    load_delivery_state(state_path, mode="final").archive_sha256,
                    final_sha,
                )

    def test_delivery_event_is_allowlisted_unique_and_never_authoritative(self):
        _, _, load_delivery_state, _, record_delivery_stage = self._api()
        sha = "a" * 64

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "delivery_state.json"
            event_path = root / "run_events.jsonl"
            for status, stage, exit_code in (
                ("complete", "package", None),
                ("failed", "upload", 31),
            ):
                record_delivery_stage(
                    state_path,
                    stage=stage,
                    status=status,
                    archive_sha256=sha,
                    exit_code=exit_code,
                    event_path=event_path,
                    runner_attempt_id="runner-safe-1",
                )
            record_delivery_stage(
                root / "final-delivery-state.json",
                mode="final",
                stage="package",
                status="complete",
                archive_sha256=sha,
                event_path=event_path,
                runner_attempt_id="runner-safe-final",
            )
            rows = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["event"] for row in rows}, {"delivery_stage"})
            self.assertEqual(len({row["attempt_id"] for row in rows}), 3)
            self.assertTrue(all(row["run_mode"] == "full" for row in rows))
            self.assertEqual(
                [row["delivery_mode"] for row in rows],
                ["inference", "inference", "final"],
            )
            self.assertEqual(
                [row["runner_attempt_id"] for row in rows],
                ["runner-safe-1", "runner-safe-1", "runner-safe-final"],
            )
            self.assertTrue(all(row["message"] == "delivery stage state recorded" for row in rows))
            serialized = json.dumps(rows)
            self.assertNotIn("Authorization", serialized)
            self.assertNotIn("Bearer", serialized)
            self.assertNotIn("secret", serialized.lower())

            invalid_event_parent = root / "not-a-directory"
            invalid_event_parent.write_text("regular file\n", encoding="utf-8")
            completed = record_delivery_stage(
                state_path,
                stage="upload",
                status="complete",
                archive_sha256=sha,
                event_path=invalid_event_parent / "events.jsonl",
                runner_attempt_id="runner-safe-2",
            )
            self.assertTrue(completed.uploaded)
            self.assertEqual(load_delivery_state(state_path), completed)

            event_target = root / "event-target.txt"
            event_target.write_text("event target remains unchanged\n", encoding="utf-8")
            event_link = root / "linked-events.jsonl"
            event_link.symlink_to(event_target)
            failed_publish = record_delivery_stage(
                state_path,
                stage="publish",
                status="failed",
                archive_sha256=sha,
                exit_code=32,
                event_path=event_link,
                runner_attempt_id="runner-safe-3",
            )
            self.assertEqual(failed_publish.last_exit_code, 32)
            self.assertEqual(
                event_target.read_text(encoding="utf-8"),
                "event target remains unchanged\n",
            )

            external_event_parent = root / "external-event-parent"
            external_event_parent.mkdir()
            external_parent_target = external_event_parent / "events.jsonl"
            external_parent_target.write_text(
                "parent target remains unchanged\n",
                encoding="utf-8",
            )
            linked_event_parent = root / "linked-event-parent"
            linked_event_parent.symlink_to(
                external_event_parent,
                target_is_directory=True,
            )
            repeated_publish = record_delivery_stage(
                state_path,
                stage="publish",
                status="complete",
                archive_sha256=sha,
                event_path=linked_event_parent / "events.jsonl",
                runner_attempt_id="runner-safe-5",
            )
            self.assertTrue(repeated_publish.published)
            self.assertEqual(
                external_parent_target.read_text(encoding="utf-8"),
                "parent target remains unchanged\n",
            )

            event_link.unlink()
            os.link(event_target, event_link)
            published = record_delivery_stage(
                state_path,
                stage="publish",
                status="complete",
                archive_sha256=sha,
                event_path=event_link,
                runner_attempt_id="runner-safe-4",
            )
            self.assertEqual(published, repeated_publish)
            self.assertEqual(
                event_target.read_text(encoding="utf-8"),
                "event target remains unchanged\n",
            )

    def test_delivery_event_append_never_reopens_a_checked_path(self):
        import scammark.result_contract as result_contract

        _, _, load_delivery_state, _, record_delivery_stage = self._api()
        sha = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "delivery_state.json"
            event_path = root / "run_events.jsonl"
            external_target = root / "external-target.txt"
            event_path.write_text("", encoding="utf-8")
            external_target.write_text(
                "external target remains unchanged\n",
                encoding="utf-8",
            )
            record_delivery_stage(
                state_path,
                stage="package",
                status="complete",
                archive_sha256=sha,
            )
            original_open_parent = result_contract._open_secure_parent_directory
            injection_count = 0

            def swap_final_entry_after_parent_open(path, *, missing_ok):
                nonlocal injection_count
                opened = original_open_parent(path, missing_ok=missing_ok)
                if Path(path) != event_path:
                    return opened
                injection_count += 1
                event_path.unlink()
                event_path.symlink_to(external_target)
                return opened

            with mock.patch.object(
                result_contract,
                "_open_secure_parent_directory",
                side_effect=swap_final_entry_after_parent_open,
            ):
                failed_upload = record_delivery_stage(
                    state_path,
                    stage="upload",
                    status="failed",
                    archive_sha256=sha,
                    exit_code=31,
                    event_path=event_path,
                    runner_attempt_id="runner-safe-race",
                )

            self.assertEqual(injection_count, 1)
            self.assertEqual(failed_upload.last_exit_code, 31)
            self.assertEqual(load_delivery_state(state_path).last_exit_code, 31)
            self.assertEqual(
                external_target.read_text(encoding="utf-8"),
                "external target remains unchanged\n",
            )

    def test_delivery_event_uses_pinned_parent_after_parent_path_swap(self):
        import scammark.result_contract as result_contract

        _, _, load_delivery_state, _, record_delivery_stage = self._api()
        sha = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "delivery_state.json"
            event_parent = root / "event-parent"
            renamed_parent = root / "event-parent-renamed"
            external_parent = root / "external-parent"
            event_parent.mkdir()
            external_parent.mkdir()
            event_path = event_parent / "events.jsonl"
            event_path.write_text("", encoding="utf-8")
            external_target = external_parent / "events.jsonl"
            external_target.write_text(
                "external parent target remains unchanged\n",
                encoding="utf-8",
            )
            record_delivery_stage(
                state_path,
                stage="package",
                status="complete",
                archive_sha256=sha,
            )
            original_open_parent = result_contract._open_secure_parent_directory
            injection_count = 0

            def swap_parent_path_after_secure_open(path, *, missing_ok):
                nonlocal injection_count
                opened = original_open_parent(path, missing_ok=missing_ok)
                if Path(path) != event_path:
                    return opened
                injection_count += 1
                event_parent.rename(renamed_parent)
                event_parent.symlink_to(external_parent, target_is_directory=True)
                return opened

            with mock.patch.object(
                result_contract,
                "_open_secure_parent_directory",
                side_effect=swap_parent_path_after_secure_open,
            ):
                failed_upload = record_delivery_stage(
                    state_path,
                    stage="upload",
                    status="failed",
                    archive_sha256=sha,
                    exit_code=31,
                    event_path=event_path,
                    runner_attempt_id="runner-safe-parent-race",
                )

            self.assertEqual(injection_count, 1)
            self.assertEqual(failed_upload.last_exit_code, 31)
            self.assertEqual(load_delivery_state(state_path).last_exit_code, 31)
            self.assertEqual(
                external_target.read_text(encoding="utf-8"),
                "external parent target remains unchanged\n",
            )
            appended_rows = (
                renamed_parent / "events.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(appended_rows), 1)
            self.assertEqual(json.loads(appended_rows[0])["event"], "delivery_stage")

    def test_delivery_event_inserts_separator_after_unterminated_tail(self):
        _, _, load_delivery_state, _, record_delivery_stage = self._api()
        sha = "a" * 64
        prior_row = '{"event":"warning","message":"partial-tail"}'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "delivery_state.json"
            event_path = root / "run_events.jsonl"
            event_path.write_text(prior_row, encoding="utf-8")

            recorded = record_delivery_stage(
                state_path,
                stage="package",
                status="complete",
                archive_sha256=sha,
                event_path=event_path,
                runner_attempt_id="runner-safe-tail",
            )

            self.assertEqual(load_delivery_state(state_path), recorded)
            lines = event_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], prior_row)
            self.assertEqual(len(lines), 2)
            appended = json.loads(lines[1])
            self.assertEqual(appended["event"], "delivery_stage")
            self.assertEqual(appended["runner_attempt_id"], "runner-safe-tail")

    def test_delivery_cli_load_record_and_validate_are_offline_and_deterministic(self):
        from tests.result_fixture import CompleteResultFixture

        sha = "a" * 64
        with CompleteResultFixture() as fixture:
            state_path = fixture.root / "delivery_state.json"
            load = self._run_module(
                "load-delivery",
                "--state-path",
                str(state_path),
                "--mode",
                "inference",
            )
            self.assertEqual(load.returncode, 0, load.stderr)
            self.assertEqual(json.loads(load.stdout)["packaged"], False)
            self.assertFalse(state_path.exists())

            record = self._run_module(
                "record-delivery",
                "--state-path",
                str(state_path),
                "--mode",
                "inference",
                "--stage",
                "package",
                "--status",
                "complete",
                "--archive-sha256",
                sha,
                "--event-path",
                str(fixture.full / "delivery_events.jsonl"),
                "--runner-attempt-id",
                "runner-safe-cli",
            )
            self.assertEqual(record.returncode, 0, record.stderr)
            self.assertEqual(json.loads(record.stdout)["archive_sha256"], sha)

            validate = self._run_module(
                "validate-full",
                "--full-dir",
                str(fixture.full),
                "--mode",
                "inference",
            )
            self.assertEqual(validate.returncode, 0, validate.stderr)
            self.assertEqual(
                json.loads(validate.stdout),
                {"mode": "inference", "sample_count": 5658},
            )

    def test_build_archive_cli_supports_both_modes_and_secret_stdin_is_safe(self):
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        with CompleteResultFixture() as fixture:
            common = (
                "--full-dir",
                str(fixture.full),
                "--state-dir",
                str(fixture.state),
                "--release-tag",
                EXPERIMENT_RELEASE_TAG,
                "--repository",
                "nlixsys/ScamWeb-V",
                "--git-commit",
                "a" * 40,
            )
            inference = self._run_module(
                "build-archive",
                "--mode",
                "inference",
                *common,
                "--log-path",
                str(fixture.log),
                "--config-path",
                str(fixture.config),
                "--frozen-dir",
                str(fixture.frozen),
                "--destination",
                str(fixture.destination),
                "--secret-values-stdin-json",
                stdin="[]\n",
            )
            self.assertEqual(inference.returncode, 0, inference.stderr)
            inference_output = json.loads(inference.stdout)
            self.assertEqual(inference_output["archive"], fixture.destination.name)
            self.assertRegex(inference_output["sha256"], r"^[0-9a-f]{64}$")

            final = self._run_module(
                "build-archive",
                "--mode",
                "final",
                *common,
                "--destination",
                str(fixture.final_destination),
                "--bertscore-asset-status-path",
                str(fixture.bertscore_asset_status),
                "--finalize-log-path",
                str(fixture.finalize_log),
            )
            self.assertEqual(final.returncode, 0, final.stderr)
            final_output = json.loads(final.stdout)
            self.assertEqual(final_output["archive"], fixture.final_destination.name)
            self.assertRegex(final_output["sha256"], r"^[0-9a-f]{64}$")

        marker = "unit-private-cli-value-93m"
        with CompleteResultFixture() as fixture:
            fixture.log.write_text(marker + "\n", encoding="utf-8")
            rejected = self._run_module(
                "build-archive",
                "--mode",
                "inference",
                "--full-dir",
                str(fixture.full),
                "--state-dir",
                str(fixture.state),
                "--log-path",
                str(fixture.log),
                "--config-path",
                str(fixture.config),
                "--frozen-dir",
                str(fixture.frozen),
                "--destination",
                str(fixture.destination),
                "--release-tag",
                EXPERIMENT_RELEASE_TAG,
                "--repository",
                "nlixsys/ScamWeb-V",
                "--git-commit",
                "a" * 40,
                "--secret-values-stdin-json",
                stdin=json.dumps([marker]),
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn(marker, rejected.stdout)
            self.assertNotIn(marker, rejected.stderr)
            self.assertFalse(fixture.destination.exists())

            argv_rejected = self._run_module(
                "build-archive",
                "--secret-value",
                marker,
            )
            self.assertNotEqual(argv_rejected.returncode, 0)
            self.assertNotIn(marker, argv_rejected.stdout)
            self.assertNotIn(marker, argv_rejected.stderr)

    def test_build_cli_enforces_secret_stdin_cap_in_utf8_bytes(self):
        from tests.result_fixture import CompleteResultFixture, EXPERIMENT_RELEASE_TAG

        private_prefix = "multibyte-private-marker"
        oversized_value = private_prefix + ("界" * 22000)
        secret_input = json.dumps([oversized_value], ensure_ascii=False)
        self.assertLess(len(secret_input), 64 * 1024)
        self.assertGreater(len(secret_input.encode("utf-8")), 64 * 1024)

        with CompleteResultFixture() as fixture:
            rejected = self._run_module(
                "build-archive",
                "--mode",
                "inference",
                "--full-dir",
                str(fixture.full),
                "--state-dir",
                str(fixture.state),
                "--log-path",
                str(fixture.log),
                "--config-path",
                str(fixture.config),
                "--frozen-dir",
                str(fixture.frozen),
                "--destination",
                str(fixture.destination),
                "--release-tag",
                EXPERIMENT_RELEASE_TAG,
                "--repository",
                "nlixsys/ScamWeb-V",
                "--git-commit",
                "a" * 40,
                "--secret-values-stdin-json",
                stdin=secret_input,
            )

            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn(private_prefix, rejected.stdout)
            self.assertNotIn(private_prefix, rejected.stderr)
            self.assertFalse(fixture.destination.exists())

    def test_secret_stdin_rejects_raw_invalid_utf8_independent_of_text_encoding(self):
        import scammark.result_contract as result_contract

        _, ResultContractError, _, _, _ = self._api()
        raw_input = io.BytesIO(b'["\xff"]')
        latin1_stdin = io.TextIOWrapper(raw_input, encoding="latin-1")
        try:
            with mock.patch.object(sys, "stdin", latin1_stdin):
                with self.assertRaises(ResultContractError) as caught:
                    result_contract._secret_values_from_stdin(True)
        finally:
            latin1_stdin.detach()

        self.assertNotIn("ÿ", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
