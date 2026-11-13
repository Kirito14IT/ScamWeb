import io
import json
import os
import runpy
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


def make_tracker(settings):
    from scammark.observability import ProgressTracker

    return ProgressTracker(
        settings,
        total_samples=100,
        api_call_limit=50000,
        pricing={
            "text": {
                "currency": "USD",
                "input_per_million": 1,
                "output_per_million": 0,
            },
            "vision": {
                "currency": "CNY",
                "input_per_million": 1,
                "output_per_million": 0,
            },
        },
        text_model="gpt-4o-mini",
        vision_model="qwen3-vl-flash",
    )


def write_minimal_status_config(root):
    output = root / "scammark_qwen3_dashscope_server_full_v2"
    config = root / "v2.toml"
    config.write_text(
        "[experiment]\nexpected_full_sample_count = 5658\napi_call_limit = 50000\n"
        f'[paths]\noutput_root = "{output.as_posix()}"\n',
        encoding="utf-8",
    )
    return config


def fake_absent_tmux(*args, **kwargs):
    return subprocess.CompletedProcess(args[0], 1, "", "missing")


def snapshot_filesystem(root):
    snapshot = {}
    for path in sorted(root.rglob("*")):
        stat = path.stat()
        snapshot[path.relative_to(root).as_posix()] = {
            "kind": "directory" if path.is_dir() else "file",
            "bytes": None if path.is_dir() else path.read_bytes(),
            "mtime_ns": stat.st_mtime_ns,
        }
    return snapshot


def write_status_checkpoint(
    checkpoint_dir,
    sample_id,
    config_hash,
    *,
    complete=(),
    errors=(),
    binary=None,
):
    stages = {}
    for stage in complete:
        output = (
            {"binary": binary}
            if stage == "binary" and binary is not None
            else {"safe": True}
        )
        stages[stage] = {"status": "complete", "output": output}
    for stage in errors:
        stages[stage] = {
            "status": "error",
            "error": {"error_type": "RuntimeError"},
        }
    (checkpoint_dir / f"{sample_id}.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "config_hash": config_hash,
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )


def make_status_event(event, sequence, timestamp, elapsed, **fields):
    parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    local_timestamp = parsed_timestamp.astimezone(
        timezone(timedelta(hours=8))
    ).isoformat()
    return {
        "schema": "scammark-run-event-v1",
        "event": event,
        "experiment_version": "full-v2",
        "run_mode": "full",
        "attempt_id": "status-attempt",
        "sequence": sequence,
        "timestamp_utc": timestamp,
        "timestamp_local": local_timestamp,
        "process_elapsed_seconds": elapsed,
        **fields,
    }


def write_tracker_status_config(root, *, total=2):
    output = root / "scammark_qwen3_dashscope_server_full_v2"
    config = root / "tracker-v2.toml"
    config.write_text(
        f"[experiment]\nexpected_full_sample_count = {total}\n"
        "api_call_limit = 50000\n"
        '[models]\ntext = "gpt-4o-mini"\nvision = "qwen3-vl-flash"\n'
        f'[paths]\noutput_root = "{output.as_posix()}"\n'
        "[observability]\neta_warmup_samples = 1\neta_history_size = 50\n"
        "cost_full_update_samples = 1\nbreaker_consecutive_backend_errors = 5\n"
        "[pricing.text]\ncurrency = \"USD\"\ninput_per_million = 1\n"
        "output_per_million = 0\n"
        "[pricing.vision]\ncurrency = \"CNY\"\ninput_per_million = 2\n"
        "output_per_million = 0\n",
        encoding="utf-8",
    )
    return config


class ObservabilityTests(unittest.TestCase):
    def test_query_tmux_distinguishes_absent_running_and_dead(self):
        from scammark.run_status import query_tmux

        calls = []

        def running(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, "0|123|\n", "")

        def dead(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, "1|123|130\n", "")

        absent = query_tmux(
            "scammark-full-v2", runner=fake_absent_tmux
        )
        live = query_tmux("scammark-full-v2", runner=running)
        stopped = query_tmux("scammark-full-v2", runner=dead)

        self.assertEqual(absent.state, "absent")
        self.assertEqual((live.state, live.pid, live.exit_code), ("running", 123, None))
        self.assertEqual((stopped.state, stopped.pid, stopped.exit_code), ("dead", 123, 130))
        self.assertEqual(
            calls,
            [
                (
                    [
                        "tmux",
                        "display-message",
                        "-p",
                        "-t",
                        "=scammark-full-v2:0.0",
                        "#{pane_dead}|#{pane_pid}|#{pane_dead_status}",
                    ],
                    {"capture_output": True, "text": True, "check": False},
                )
            ],
        )

    def test_collect_status_is_read_only_when_state_does_not_exist(self):
        from scammark.run_status import collect_status

        original_open = Path.open

        def read_only_open(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                raise AssertionError(f"status attempted write-open: {mode}")
            return original_open(path, mode, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            before = snapshot_filesystem(root)
            with (
                mock.patch(
                    "scammark.experiment_state.ExperimentState",
                    side_effect=AssertionError("status constructed ExperimentState"),
                ),
                mock.patch(
                    "scammark.observability.RunEventLogger",
                    side_effect=AssertionError("status constructed RunEventLogger"),
                ),
                mock.patch.object(
                    Path,
                    "mkdir",
                    side_effect=AssertionError("status attempted mkdir"),
                ),
                mock.patch.object(Path, "open", new=read_only_open),
                mock.patch.object(
                    Path,
                    "write_text",
                    side_effect=AssertionError("status attempted write_text"),
                ),
                mock.patch.object(
                    Path,
                    "write_bytes",
                    side_effect=AssertionError("status attempted write_bytes"),
                ),
                mock.patch.object(
                    Path,
                    "touch",
                    side_effect=AssertionError("status attempted touch"),
                ),
            ):
                status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )
            after = snapshot_filesystem(root)

        self.assertEqual(status["tmux"]["state"], "absent")
        self.assertEqual(after, before)

    def test_status_next_action_priority_table(self):
        from scammark.run_status import choose_next_action

        cases = [
            ({"tmux": {"state": "running"}}, "monitor"),
            (
                {"delivery_failure": 30},
                "bash scripts/server/recover_delivery.sh package",
            ),
            (
                {"delivery_failure": 31},
                "bash scripts/server/recover_delivery.sh upload",
            ),
            (
                {"delivery_failure": 32},
                "bash scripts/server/recover_delivery.sh publish",
            ),
            (
                {"exit_code": 20},
                "replenish credentials/balance, then resume",
            ),
            (
                {"exit_code": 21},
                "physical call cap exhausted; do not resume this version",
            ),
            (
                {"exit_code": 22},
                "inspect errors, then start_full.sh --ack-breaker",
            ),
            (
                {"exit_code": 23},
                "inspect contract drift; do not force resume",
            ),
            ({"exit_code": 2}, "inspect failed acceptance gates"),
            (
                {"exit_code": 130, "graceful_stop_event": True},
                "resume inference",
            ),
            (
                {"exit_code": 143, "graceful_stop_event": True},
                "resume inference",
            ),
            (
                {"exit_code": 130, "graceful_stop_event": False},
                "inspect forced interruption and orphan call",
            ),
            (
                {"exit_code": 143, "graceful_stop_event": False},
                "inspect forced interruption and orphan call",
            ),
            (
                {"inference_complete": True, "bertscore_complete": False},
                "bash scripts/server/recover_delivery.sh package",
            ),
            (
                {
                    "inference_complete": True,
                    "bertscore_complete": False,
                    "delivery_modes": {
                        "inference": {"stage": "publish", "status": "complete"},
                        "final": {"stage": "package", "status": "pending"},
                    },
                },
                "finalize BERTScore",
            ),
            (
                {"preflight_complete": True, "smoke_complete": False},
                "run paid 10-page smoke",
            ),
        ]
        for status, expected in cases:
            with self.subTest(status=status):
                self.assertEqual(choose_next_action(status), expected)

        self.assertEqual(
            choose_next_action(
                {
                    "tmux": {"state": "running"},
                    "delivery_failure": 30,
                    "exit_code": 20,
                }
            ),
            "monitor",
        )
        self.assertIn(
            "contract",
            choose_next_action(
                {
                    "calls": {"corrupt": True},
                    "errors": [
                        {
                            "message": (
                                "Bearer sk-STATUS-SECRET-12345678 "
                                "raw_response=RAW_RESPONSE_SECRET"
                            )
                        }
                    ],
                }
            ),
        )
        actions = "\n".join(choose_next_action(status) for status, _ in cases)
        self.assertNotIn("STATUS-SECRET", actions)
        self.assertNotIn("RAW_RESPONSE_SECRET", actions)

    def test_render_status_shows_only_remaining_cost(self):
        from scammark.run_status import render_status

        text = render_status(
            {
                "tmux": {"state": "dead", "pid": 123, "exit_code": 20},
                "samples": {
                    "complete": 14,
                    "partial": 1,
                    "error": 0,
                    "total": 5658,
                },
                "stages": {"complete": 104, "reused": 0, "failed": 1},
                "current": {"sample_id": "156", "stage": "binary"},
                "last_safe": {"sample_id": "156", "stage": "cross"},
                "calls": {
                    "reserved": 104,
                    "finished": 103,
                    "outstanding": 1,
                    "remaining": 49896,
                },
                "eta_seconds": 3600,
                "remaining_cost_est": {"USD": 1.84, "CNY": 19.76},
                "accrued_cost": {"USD": 99},
                "spent": 88,
                "estimated_total": 77,
                "per_sample_cost": 66,
                "final_cost": 55,
                "errors": [
                    {"category": "auth_billing", "message": "HTTP 402"}
                ],
                "breaker": {"consecutive": 0, "opened": False},
                "next_action": "replenish credentials/balance, then resume",
            }
        )
        self.assertIn("complete: 14 / 5658", text)
        self.assertIn("current: sample=156 stage=binary", text)
        self.assertIn(
            "next action: replenish credentials/balance, then resume",
            text,
        )
        self.assertIn(
            "remaining cost estimate: CNY 19.76 + USD 1.84",
            text,
        )
        self.assertLess(text.index("CNY 19.76"), text.index("USD 1.84"))
        for forbidden in (
            "accrued",
            "spent",
            "estimated total",
            "per sample",
            "per-sample",
            "final cost",
            "99",
            "88",
            "77",
            "66",
            "55",
        ):
            self.assertNotIn(forbidden, text.lower())

    def test_status_ignores_only_truncated_event_tail_but_flags_ledger_corruption(self):
        from scammark.run_status import collect_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            full, state = output / "full", output / "state_full"
            full.mkdir(parents=True)
            state.mkdir(parents=True)
            (full / "run_events.jsonl").write_text(
                '{"event":"sample_complete","sample_id":"1"}\n{"event":',
                encoding="utf-8",
            )
            (state / "api_call_ledger.jsonl").write_text(
                '{"event":"reserved"', encoding="utf-8"
            )
            status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

        self.assertEqual(
            status["event_warnings"],
            ["ignored truncated final event row"],
        )
        self.assertTrue(status["calls"]["corrupt"])
        self.assertEqual(status["calls"]["reserved"], 1)
        self.assertEqual(status["calls"]["outstanding"], 1)
        self.assertIn("contract", status["next_action"])

    def test_read_jsonl_tail_warns_distinctly_for_earlier_malformed_rows(self):
        from scammark.run_status import read_jsonl_tail

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            trusted = {
                "event": "summary",
                "eta_seconds": 42,
                "remaining_cost_est": {"USD": 1.5},
            }
            path.write_text(
                json.dumps(trusted)
                + '\n{"event":"summary","eta_seconds":'
                + '\n{"event":"error","message":',
                encoding="utf-8",
            )
            rows, warnings = read_jsonl_tail(path)

        self.assertEqual(rows, [trusted])
        self.assertEqual(
            warnings,
            ["malformed event row 2", "ignored truncated final event row"],
        )
        self.assertEqual(
            warnings.count("ignored truncated final event row"),
            1,
        )

    def test_read_jsonl_tail_bounds_non_objects_oversized_rows_and_huge_integers(self):
        from scammark.run_status import read_jsonl_tail

        secret = "RAW_RESPONSE_EVENT_SECRET"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            valid = {"event": "heartbeat", "health": "ok"}
            path.write_text(
                "[]\n"
                + json.dumps(
                    {
                        "event": "error",
                        "message": secret + ("x" * 300_000),
                    }
                )
                + "\n"
                + '{"event":"summary","eta_seconds":'
                + ("9" * 5_000)
                + "}\n"
                + json.dumps(valid)
                + "\n",
                encoding="utf-8",
            )
            rows, warnings = read_jsonl_tail(path)

        self.assertEqual(rows, [valid])
        self.assertEqual(len(warnings), 3)
        self.assertTrue(any("non-object" in warning for warning in warnings))
        self.assertTrue(any("oversized" in warning for warning in warnings))
        self.assertTrue(any("malformed" in warning for warning in warnings))
        rendered = json.dumps({"rows": rows, "warnings": warnings})
        self.assertNotIn(secret, rendered)
        self.assertLess(len(rendered), 2_000)

    def test_status_uses_checkpoint_authoritative_shared_sample_progress(self):
        from scammark.experiment_state import (
            scan_checkpoints,
            summarize_sample_progress,
        )
        from scammark.run_status import collect_status

        base = ("url", "text", "visual", "cross", "binary")
        seven = (*base, "type", "grounding")
        manifest = [
            {"sample_id": "benign-five", "binary": "benign"},
            {"sample_id": "gold-scam-seven", "binary": "scam"},
            {"sample_id": "benign-fp-seven", "binary": "benign"},
            {"sample_id": "partial", "binary": "benign"},
            {"sample_id": "required-error", "binary": "benign"},
        ]
        config_hash = "a" * 64
        opened = []
        original_open = Path.open

        def reject_v1_open(path, *args, **kwargs):
            candidate = Path(path)
            opened.append(candidate)
            if "scammark_qwen3_dashscope_server_full" in candidate.parts:
                raise AssertionError(f"status opened v1 runtime path: {candidate}")
            return original_open(candidate, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            full = output / "full"
            checkpoints = output / "state_full" / "checkpoints"
            full.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            (full / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            (full / "predictions.jsonl").write_text(
                json.dumps({"sample_id": "benign-five"}) + "\n",
                encoding="utf-8",
            )
            (output / "state_full" / "run_config.json").write_text(
                json.dumps({"config_hash": config_hash, "config": {}}),
                encoding="utf-8",
            )
            write_status_checkpoint(
                checkpoints,
                "benign-five",
                config_hash,
                complete=base,
                binary="benign",
            )
            write_status_checkpoint(
                checkpoints,
                "gold-scam-seven",
                config_hash,
                complete=seven,
                binary="benign",
            )
            write_status_checkpoint(
                checkpoints,
                "benign-fp-seven",
                config_hash,
                complete=seven,
                binary="scam",
            )
            write_status_checkpoint(
                checkpoints,
                "partial",
                config_hash,
                complete=("url",),
            )
            write_status_checkpoint(
                checkpoints,
                "required-error",
                config_hash,
                errors=("text",),
            )
            write_status_checkpoint(
                checkpoints,
                "wrong-hash",
                "b" * 64,
                complete=base,
                binary="benign",
            )
            v1 = root / "scammark_qwen3_dashscope_server_full" / "full"
            v1.mkdir(parents=True)
            (v1 / "manifest.jsonl").write_text(
                '{"sample_id":"V1_SENTINEL","binary":"benign"}\n',
                encoding="utf-8",
            )

            with (
                mock.patch.object(Path, "open", new=reject_v1_open),
                mock.patch(
                    "scammark.run_status.scan_checkpoints",
                    wraps=scan_checkpoints,
                ) as checkpoint_scanner,
                mock.patch(
                    "scammark.run_status.summarize_sample_progress",
                    wraps=summarize_sample_progress,
                ) as progress_summary,
            ):
                status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        checkpoint_scanner.assert_called_once_with(
            checkpoints,
            config_hash=config_hash,
        )
        progress_summary.assert_called_once()
        self.assertEqual(
            status["samples"],
            {
                "complete": 3,
                "partial": 1,
                "error": 1,
                "total": 5,
            },
        )
        self.assertEqual(
            status["required_stage_counts"],
            {
                "benign-five": 5,
                "gold-scam-seven": 7,
                "benign-fp-seven": 7,
                "partial": 5,
                "required-error": 5,
            },
        )
        self.assertEqual(status["stages"]["complete"], 20)
        self.assertEqual(status["stages"]["failed"], 1)
        self.assertEqual(status["prediction_rows"], 1)
        self.assertEqual(
            status["checkpoint_corrupt"],
            ["wrong-hash.json"],
        )
        self.assertNotIn(
            "scammark_qwen3_dashscope_server_full",
            {part for path in opened for part in path.parts},
        )
        self.assertNotIn("V1_SENTINEL", json.dumps(status))

    def test_status_uses_only_configured_frozen_manifest_and_rejects_v1_symlink(self):
        from scammark.run_status import collect_status

        base = ("url", "text", "visual", "cross", "binary")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen" / "manifest.jsonl"
            frozen.parent.mkdir()
            frozen.write_text(
                '{"sample_id":"frozen-benign","binary":"benign"}\n',
                encoding="utf-8",
            )
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            config = root / "v2.toml"
            config.write_text(
                "[experiment]\nexpected_full_sample_count = 5658\n"
                "api_call_limit = 50000\n"
                f'[data]\nmanifest = "{frozen.as_posix()}"\n'
                f'[paths]\noutput_root = "{output.as_posix()}"\n',
                encoding="utf-8",
            )
            checkpoints = output / "state_full" / "checkpoints"
            checkpoints.mkdir(parents=True)
            config_hash = "c" * 64
            (output / "state_full" / "run_config.json").write_text(
                json.dumps({"config_hash": config_hash}), encoding="utf-8"
            )
            write_status_checkpoint(
                checkpoints,
                "frozen-benign",
                config_hash,
                complete=base,
                binary="benign",
            )

            status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

            v1 = root / "scammark_qwen3_dashscope_server_full"
            v1.mkdir()
            linked_v2 = root / "linked" / "scammark_qwen3_dashscope_server_full_v2"
            linked_v2.parent.mkdir()
            linked_v2.symlink_to(v1, target_is_directory=True)
            bad_config = root / "bad-v2.toml"
            bad_config.write_text(
                "[experiment]\nexpected_full_sample_count = 5658\n"
                "api_call_limit = 50000\n"
                f'[paths]\noutput_root = "{linked_v2.as_posix()}"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "full-v2"):
                collect_status(
                    bad_config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        self.assertEqual(
            status["samples"],
            {"complete": 1, "partial": 0, "error": 0, "total": 1},
        )

    def test_status_rejects_nested_output_symlink_before_opening_v1(self):
        from scammark.run_status import collect_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            output.mkdir()
            v1_full = (
                root / "scammark_qwen3_dashscope_server_full" / "full"
            )
            v1_full.mkdir(parents=True)
            sentinel = v1_full / "manifest.jsonl"
            sentinel.write_text(
                '{"sample_id":"V1_NESTED_SENTINEL","binary":"benign"}\n',
                encoding="utf-8",
            )
            (output / "full").symlink_to(v1_full, target_is_directory=True)
            original_open = Path.open

            def reject_sentinel_open(path, *args, **kwargs):
                if Path(path).resolve(strict=False) == sentinel:
                    raise AssertionError("status opened nested v1 sentinel")
                return original_open(path, *args, **kwargs)

            with (
                mock.patch.object(Path, "open", new=reject_sentinel_open),
                self.assertRaisesRegex(ValueError, "v2"),
            ):
                collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

    def test_status_fails_closed_for_hostile_or_unreadable_durable_ledger(self):
        from scammark.run_status import collect_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            ledger = (
                root
                / "scammark_qwen3_dashscope_server_full_v2"
                / "state_full"
                / "api_call_ledger.jsonl"
            )
            ledger.parent.mkdir(parents=True)
            ledger.write_text(
                '{"event":"reserved","call_id":'
                + ("9" * 5_000)
                + "}\n",
                encoding="utf-8",
            )

            huge_integer_status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

            ledger.write_text(
                '{"event":"reserved","call_id":"safe"}\n',
                encoding="utf-8",
            )
            original_read_text = Path.read_text

            def fail_ledger_read(path, *args, **kwargs):
                if Path(path) == ledger:
                    raise OSError("durable ledger unavailable")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", new=fail_ledger_read):
                unreadable_status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        for status in (huge_integer_status, unreadable_status):
            with self.subTest(status=status):
                self.assertTrue(status["calls"]["corrupt"])
                self.assertGreaterEqual(status["calls"]["reserved"], 1)
                self.assertGreaterEqual(status["calls"]["outstanding"], 1)
                self.assertIn("contract", status["next_action"])

    def test_status_requires_valid_run_config_hash_for_existing_durable_state(self):
        from scammark.run_status import collect_status

        base = ("url", "text", "visual", "cross", "binary")
        config_hash = "e" * 64
        for run_config_mode in ("missing", "corrupt"):
            with self.subTest(run_config_mode=run_config_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = write_tracker_status_config(root, total=1)
                output = root / "scammark_qwen3_dashscope_server_full_v2"
                full = output / "full"
                state = output / "state_full"
                checkpoints = state / "checkpoints"
                full.mkdir(parents=True)
                checkpoints.mkdir(parents=True)
                (full / "manifest.jsonl").write_text(
                    '{"sample_id":"durable","binary":"benign"}\n',
                    encoding="utf-8",
                )
                write_status_checkpoint(
                    checkpoints,
                    "durable",
                    config_hash,
                    complete=base,
                    binary="benign",
                )
                (state / "api_call_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "event": "reserved",
                            "call_id": "durable-call",
                            "config_hash": config_hash,
                            "context": {
                                "sample_id": "durable",
                                "stage": "url",
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if run_config_mode == "corrupt":
                    (state / "run_config.json").write_text(
                        '{"config_hash":"invalid"}', encoding="utf-8"
                    )

                status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

            self.assertEqual(status["samples"]["complete"], 0)
            self.assertEqual(status["stages"]["complete"], 0)
            self.assertTrue(status["calls"]["corrupt"])
            self.assertGreaterEqual(status["calls"]["reserved"], 1)
            self.assertGreaterEqual(status["calls"]["outstanding"], 1)
            self.assertIn("run_config.json", status["checkpoint_corrupt"])
            self.assertIn("contract", status["next_action"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            empty = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )
        self.assertFalse(empty["calls"]["corrupt"])
        self.assertEqual(empty["checkpoint_corrupt"], [])

    def test_status_checkpoint_preflight_rejects_unsafe_collections(self):
        from scammark.experiment_state import scan_checkpoints
        from scammark.run_status import collect_status

        config_hash = "f" * 64
        complete_binary = {
            "url": {"status": "complete", "output": {}},
            "text": {"status": "complete", "output": {}},
            "visual": {"status": "complete", "output": {}},
            "cross": {"status": "complete", "output": {}},
            "binary": {
                "status": "complete",
                "output": {"binary": "benign"},
            },
        }

        def prepare(root):
            config = write_tracker_status_config(root, total=1)
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            full = output / "full"
            checkpoints = output / "state_full" / "checkpoints"
            full.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            (full / "manifest.jsonl").write_text(
                '{"sample_id":"duplicate","binary":"benign"}\n',
                encoding="utf-8",
            )
            (output / "state_full" / "run_config.json").write_text(
                json.dumps({"config_hash": config_hash}), encoding="utf-8"
            )
            return config, checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            config, checkpoints = prepare(Path(tmp))
            payloads = (
                ("duplicate-a.json", complete_binary),
                ("duplicate-no-binary.json", {"url": complete_binary["url"]}),
                (
                    "duplicate-conflict.json",
                    {
                        **complete_binary,
                        "binary": {
                            "status": "complete",
                            "output": {"binary": "scam"},
                        },
                    },
                ),
            )
            for name, stages in payloads:
                (checkpoints / name).write_text(
                    json.dumps(
                        {
                            "sample_id": "duplicate",
                            "config_hash": config_hash,
                            "stages": stages,
                        }
                    ),
                    encoding="utf-8",
                )
            with mock.patch(
                "scammark.run_status.scan_checkpoints",
                wraps=scan_checkpoints,
            ) as checkpoint_scanner:
                duplicate_status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        checkpoint_scanner.assert_not_called()
        self.assertEqual(duplicate_status["samples"]["complete"], 0)
        self.assertEqual(duplicate_status["stages"]["complete"], 0)
        self.assertEqual(
            duplicate_status["checkpoint_corrupt"],
            [
                "duplicate-a.json",
                "duplicate-conflict.json",
                "duplicate-no-binary.json",
            ],
        )
        self.assertIn("contract", duplicate_status["next_action"])

        with tempfile.TemporaryDirectory() as tmp:
            config, checkpoints = prepare(Path(tmp))
            oversized = {
                "sample_id": "duplicate",
                "config_hash": config_hash,
                "stages": complete_binary,
                "padding": "x" * (4 * 1024 * 1024),
            }
            (checkpoints / "oversized.json").write_text(
                json.dumps(oversized), encoding="utf-8"
            )
            with mock.patch(
                "scammark.run_status.scan_checkpoints",
                wraps=scan_checkpoints,
            ) as checkpoint_scanner:
                oversized_status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        checkpoint_scanner.assert_not_called()
        self.assertEqual(oversized_status["stages"]["complete"], 0)
        self.assertIn("oversized.json", oversized_status["checkpoint_corrupt"])

        with tempfile.TemporaryDirectory() as tmp:
            config, checkpoints = prepare(Path(tmp))
            (checkpoints / "bad-shape.json").write_text(
                json.dumps(
                    {
                        "sample_id": "duplicate",
                        "config_hash": config_hash,
                        "stages": [],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "scammark.run_status.scan_checkpoints",
                wraps=scan_checkpoints,
            ) as checkpoint_scanner:
                bad_shape_status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        checkpoint_scanner.assert_not_called()
        self.assertEqual(bad_shape_status["stages"]["complete"], 0)
        self.assertEqual(
            bad_shape_status["checkpoint_corrupt"], ["bad-shape.json"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            config, checkpoints = prepare(Path(tmp))
            (checkpoints / "first.json").write_text(
                json.dumps(
                    {
                        "sample_id": "first",
                        "config_hash": config_hash,
                        "stages": {},
                    }
                ),
                encoding="utf-8",
            )
            (checkpoints / "second.json").write_text(
                json.dumps(
                    {
                        "sample_id": "second",
                        "config_hash": config_hash,
                        "stages": {},
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch("scammark.run_status._CHECKPOINT_MAX_FILES", 1),
                mock.patch(
                    "scammark.run_status.scan_checkpoints",
                    wraps=scan_checkpoints,
                ) as checkpoint_scanner,
            ):
                bounded_status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        checkpoint_scanner.assert_not_called()
        self.assertEqual(bounded_status["stages"]["complete"], 0)
        self.assertTrue(bounded_status["checkpoint_corrupt"])
        self.assertIn("contract", bounded_status["next_action"])

    def test_status_inference_completion_requires_durable_prediction_alignment(self):
        from scammark.run_status import collect_status

        base = ("url", "text", "visual", "cross", "binary")
        config_hash = "1" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_tracker_status_config(root, total=2)
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            full = output / "full"
            checkpoints = output / "state_full" / "checkpoints"
            full.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            (full / "manifest.jsonl").write_text(
                '{"sample_id":"alpha","binary":"benign"}\n'
                '{"sample_id":"beta","binary":"benign"}\n',
                encoding="utf-8",
            )
            (output / "state_full" / "run_config.json").write_text(
                json.dumps({"config_hash": config_hash}), encoding="utf-8"
            )
            write_status_checkpoint(
                checkpoints,
                "alpha",
                config_hash,
                complete=base,
                binary="benign",
            )
            (full / "metrics.json").write_text(
                json.dumps(
                    {"engineering": {"acceptance_gates": {"passed": True}}}
                ),
                encoding="utf-8",
            )
            (full / "run_events.jsonl").write_text(
                json.dumps(
                    make_status_event(
                        "run_completed",
                        1,
                        "2026-07-13T12:00:00+00:00",
                        30,
                        sample_count=2,
                        gates_passed=True,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            predictions = full / "predictions.jsonl"
            predictions.write_text(
                '{"sample_id":"alpha"}\n{"sample_id":"alpha"}\n',
                encoding="utf-8",
            )
            duplicate_and_stale = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

            write_status_checkpoint(
                checkpoints,
                "beta",
                config_hash,
                complete=base,
                binary="benign",
            )
            predictions.write_text(
                '{"sample_id":"alpha"}\n{"sample_id":"unknown"}\n',
                encoding="utf-8",
            )
            unknown_id = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

            predictions.write_text(
                '{"sample_id":"alpha"}\n', encoding="utf-8"
            )
            missing_id = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

            predictions.write_text(
                '{"sample_id":"alpha"}\n{"sample_id":"beta"}\n',
                encoding="utf-8",
            )
            aligned = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

        self.assertFalse(duplicate_and_stale["inference_complete"])
        self.assertEqual(duplicate_and_stale["samples"]["complete"], 1)
        self.assertIn(
            "predictions.jsonl", duplicate_and_stale["checkpoint_corrupt"]
        )
        self.assertFalse(unknown_id["inference_complete"])
        self.assertIn("predictions.jsonl", unknown_id["checkpoint_corrupt"])
        self.assertFalse(missing_id["inference_complete"])
        self.assertEqual(missing_id["prediction_rows"], 1)
        self.assertTrue(aligned["inference_complete"])
        self.assertEqual(aligned["samples"]["complete"], 2)

    def test_status_actions_and_stop_facts_use_live_state_and_latest_attempt(self):
        from scammark.run_status import choose_next_action, collect_status

        self.assertEqual(
            choose_next_action(
                {"calls": {"corrupt": False, "remaining": 0}}
            ),
            "physical call cap exhausted; do not resume this version",
        )
        self.assertEqual(
            choose_next_action({"breaker": {"opened": True}}),
            "inspect errors, then start_full.sh --ack-breaker",
        )
        self.assertEqual(
            choose_next_action(
                {
                    "tmux": {"state": "running"},
                    "calls": {"remaining": 0},
                    "breaker": {"opened": True},
                }
            ),
            "monitor",
        )
        self.assertIn(
            "contract",
            choose_next_action(
                {
                    "calls": {"corrupt": True, "remaining": 0},
                    "delivery_failure": 31,
                }
            ),
        )

        events = [
            make_status_event(
                "run_started",
                1,
                "2026-07-13T12:00:00+00:00",
                0,
                attempt_id="attempt-old",
            ),
            make_status_event(
                "shell_exited",
                2,
                "2026-07-13T12:01:00+00:00",
                10,
                attempt_id="shell-old",
                runner_attempt_id="attempt-old",
                exit_code=0,
                forced=False,
            ),
            make_status_event(
                "run_started",
                1,
                "2026-07-13T12:02:00+00:00",
                0,
                attempt_id="attempt-new",
            ),
            make_status_event(
                "run_stopped",
                2,
                "2026-07-13T12:03:00+00:00",
                20,
                attempt_id="attempt-new",
                reason="signal",
                category="graceful_stop",
                exit_code=130,
                forced=False,
            ),
            make_status_event(
                "shell_exited",
                3,
                "2026-07-13T12:04:00+00:00",
                30,
                attempt_id="shell-foreign",
                runner_attempt_id="attempt-old",
                exit_code=143,
                forced=True,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            event_path = (
                root
                / "scammark_qwen3_dashscope_server_full_v2"
                / "full"
                / "run_events.jsonl"
            )
            event_path.parent.mkdir(parents=True)
            event_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

        self.assertEqual(status["exit_code"], 130)
        self.assertTrue(status["graceful_stop_event"])
        self.assertFalse(status["forced_stop"])
        self.assertEqual(status["stop_reason"], "signal")
        self.assertEqual(status["next_action"], "resume inference")

    def test_status_projects_delivery_stages_and_prioritizes_recovery(self):
        from scammark.run_status import _project_events, choose_next_action

        delivery = make_status_event(
            "delivery_stage",
            1,
            "2026-07-13T12:00:00+00:00",
            1,
            stage="upload",
            status="failed",
            exit_code=31,
            delivery_mode="final",
        )
        invalid_delivery = make_status_event(
            "delivery_stage",
            2,
            "2026-07-13T12:00:01+00:00",
            2,
            stage="upload",
            status="failed",
            exit_code=31,
            delivery_mode="url",
        )
        inference_with_delivery_stage = make_status_event(
            "stage_started",
            3,
            "2026-07-13T12:00:02+00:00",
            3,
            sample_id="safe-id",
            stage="upload",
            model="gpt-4o-mini",
        )

        projected = _project_events(
            [delivery, invalid_delivery, inference_with_delivery_stage],
            expected_run_mode="full",
        )

        self.assertEqual(projected[0]["stage"], "upload")
        self.assertEqual(projected[0]["exit_code"], 31)
        self.assertEqual(projected[0]["delivery_mode"], "final")
        self.assertEqual(len(projected), 2)
        self.assertNotIn("stage", projected[1])
        self.assertEqual(
            choose_next_action(
                {
                    "delivery_failure": 31,
                    "calls": {"corrupt": False, "remaining": 0},
                    "breaker": {"opened": True},
                    "inference_complete": True,
                    "bertscore_complete": False,
                }
            ),
            "bash scripts/server/recover_delivery.sh upload",
        )
        self.assertEqual(
            choose_next_action(
                {
                    "tmux": {"state": "running"},
                    "delivery_failure": 31,
                    "calls": {"remaining": 0},
                }
            ),
            "monitor",
        )
        self.assertIn(
            "contract",
            choose_next_action(
                {
                    "delivery_failure": 31,
                    "calls": {"corrupt": True, "remaining": 0},
                }
            ),
        )

    def test_status_missing_delivery_state_falls_back_to_events_by_mode(self):
        from scammark.run_status import _delivery_status, _project_events

        events = _project_events(
            [
                make_status_event(
                    "delivery_stage",
                    1,
                    "2026-07-13T12:00:00+00:00",
                    1,
                    stage="publish",
                    status="complete",
                ),
                make_status_event(
                    "delivery_stage",
                    2,
                    "2026-07-13T12:00:01+00:00",
                    2,
                    stage="upload",
                    status="failed",
                    exit_code=31,
                    delivery_mode="final",
                ),
            ],
            expected_run_mode="full",
        )
        with tempfile.TemporaryDirectory() as tmp:
            modes, selected, failure, corrupt = _delivery_status(
                Path(tmp),
                events=events,
            )

        self.assertEqual(
            modes,
            {
                "inference": {"stage": "publish", "status": "complete"},
                "final": {"stage": "upload", "status": "failed", "exit_code": 31},
            },
        )
        self.assertEqual(selected, modes["inference"])
        self.assertIsNone(failure)
        self.assertFalse(corrupt)

    def test_status_validates_manifest_ids_and_projects_ledger_stage_keys(self):
        from scammark.run_status import collect_status

        config_hash = "3" * 64
        invalid_manifests = (
            (
                [{"sample_id": "MANIFEST ID SENTINEL", "binary": "benign"}],
                "MANIFEST ID SENTINEL",
            ),
            (
                [{"sample_id": "sk-MANIFESTSECRET12345678", "binary": "benign"}],
                "MANIFESTSECRET12345678",
            ),
            (
                [{"sample_id": "safe-id", "binary": "unknown"}],
                None,
            ),
            (
                [
                    {"sample_id": "duplicate-id", "binary": "benign"},
                    {"sample_id": "duplicate-id", "binary": "benign"},
                ],
                None,
            ),
        )
        for rows, sentinel in invalid_manifests:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = write_tracker_status_config(root, total=len(rows))
                output = root / "scammark_qwen3_dashscope_server_full_v2"
                full = output / "full"
                state = output / "state_full"
                full.mkdir(parents=True)
                state.mkdir(parents=True)
                (full / "manifest.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                (state / "run_config.json").write_text(
                    json.dumps({"config_hash": config_hash}), encoding="utf-8"
                )
                status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

            rendered = json.dumps(status, sort_keys=True)
            self.assertEqual(status["required_stage_counts"], {})
            self.assertIn("manifest.jsonl", status["checkpoint_corrupt"])
            self.assertIn("contract", status["next_action"])
            if sentinel is not None:
                self.assertNotIn(sentinel, rendered)

        ledger_stage = "LEDGER_STAGE_SENTINEL"
        replay_stage = "REPLAY_STAGE_SENTINEL"
        huge_stage = "H" * 1_000_000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_tracker_status_config(root, total=1)
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            full = output / "full"
            state = output / "state_full"
            full.mkdir(parents=True)
            state.mkdir(parents=True)
            (full / "manifest.jsonl").write_text(
                '{"sample_id":"safe-id","binary":"benign"}\n',
                encoding="utf-8",
            )
            (state / "run_config.json").write_text(
                json.dumps({"config_hash": config_hash}), encoding="utf-8"
            )
            ledger_rows = []
            for call_id, sample_id, stage in (
                ("known", "safe-id", "visual"),
                ("unexpected", "LEDGER_SAMPLE_SENTINEL", ledger_stage),
                ("huge", "safe-id", huge_stage),
                ("replay", "safe-id", replay_stage),
            ):
                ledger_rows.append(
                    {
                        "event": "reserved",
                        "call_id": call_id,
                        "config_hash": config_hash,
                        "context": {"sample_id": sample_id, "stage": stage},
                    }
                )
            ledger_rows.append(
                {
                    "event": "finished",
                    "call_id": "replay",
                    "config_hash": config_hash,
                    "status": "completed",
                }
            )
            (state / "api_call_ledger.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in ledger_rows),
                encoding="utf-8",
            )
            status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

        rendered = json.dumps(status, sort_keys=True)
        self.assertEqual(status["calls"]["reserved"], 4)
        self.assertEqual(status["calls"]["finished"], 1)
        self.assertEqual(status["calls"]["outstanding"], 3)
        self.assertEqual(
            status["calls"]["outstanding_by_stage"],
            {"unknown": 2, "visual": 1},
        )
        self.assertEqual(
            status["calls"]["replay_required_by_stage"], {"unknown": 1}
        )
        self.assertTrue(status["calls"]["corrupt"])
        self.assertIn("contract", status["next_action"])
        for sentinel in (
            ledger_stage,
            replay_stage,
            "LEDGER_SAMPLE_SENTINEL",
            "H" * 100,
        ):
            self.assertNotIn(sentinel, rendered)
        self.assertLess(len(rendered), 20_000)

    def test_status_bounds_and_normalizes_every_ledger_scalar(self):
        from scammark.experiment_state import CallLedgerAudit
        from scammark.run_status import (
            _STATUS_INTEGER_LIMIT,
            _project_ledger_audit,
            collect_status,
        )

        ordinary = CallLedgerAudit(
            reserved=5,
            finished=3,
            completed=2,
            transport_error=1,
            outstanding=2,
            outstanding_by_stage={"url": 2},
            outstanding_contexts=(),
            finished_without_checkpoint=frozenset(),
            replay_required_by_stage={"text": 1},
            corrupt=False,
        )
        self.assertEqual(_project_ledger_audit(ordinary), ordinary)

        class HostileInt(int):
            pass

        huge = 10**3_999
        unsafe = CallLedgerAudit(
            reserved=huge,
            finished=True,
            completed=-1,
            transport_error=HostileInt(2),
            outstanding=huge,
            outstanding_by_stage={"url": 1},
            outstanding_contexts=(),
            finished_without_checkpoint=frozenset(),
            replay_required_by_stage={},
            corrupt=False,
        )
        projected = _project_ledger_audit(unsafe)
        scalars = (
            projected.reserved,
            projected.finished,
            projected.completed,
            projected.transport_error,
            projected.outstanding,
        )
        self.assertTrue(projected.corrupt)
        self.assertTrue(
            all(
                type(value) is int and 0 <= value <= _STATUS_INTEGER_LIMIT
                for value in scalars
            )
        )
        self.assertLessEqual(
            projected.completed + projected.transport_error,
            projected.finished,
        )
        self.assertLessEqual(
            projected.finished + projected.outstanding,
            projected.reserved,
        )

        config_hash = "4" * 64
        huge_decimal = "1" + ("0" * 3_999)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_tracker_status_config(root, total=1)
            state = (
                root
                / "scammark_qwen3_dashscope_server_full_v2"
                / "state_full"
            )
            state.mkdir(parents=True)
            (state / "run_config.json").write_text(
                json.dumps({"config_hash": config_hash}), encoding="utf-8"
            )
            ledger_row = {
                "event": "reserved",
                "call_id": "legacy-baseline",
                "config_hash": config_hash,
                "legacy_api_calls_before_ledger": huge,
                "context": {"sample_id": "safe-id", "stage": "url"},
            }
            (state / "api_call_ledger.jsonl").write_text(
                json.dumps(ledger_row) + "\n", encoding="utf-8"
            )
            status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

        serialized = json.dumps(status, sort_keys=True)
        self.assertEqual(status["calls"]["reserved"], _STATUS_INTEGER_LIMIT)
        self.assertEqual(status["calls"]["outstanding"], 1)
        self.assertEqual(status["calls"]["remaining"], 0)
        self.assertTrue(status["calls"]["corrupt"])
        self.assertIn("contract", status["next_action"])
        self.assertNotIn(huge_decimal, serialized)
        self.assertLess(len(serialized), 20_000)

    def test_status_latest_valid_summary_lifecycle_errors_breaker_and_security(self):
        from scammark.run_status import collect_status, render_status

        secret = "sk-STATUSSECRET12345678"
        raw_secret = "RAW_RESPONSE_STATUS_SENTINEL"
        base64_secret = "BASE64_STATUS_SENTINEL"
        query_secret = "QUERY_STATUS_SENTINEL"
        workspace = "private-status-workspace"
        events = [
            make_status_event(
                "run_started",
                1,
                "2026-07-13T12:00:00+00:00",
                0,
            ),
            make_status_event(
                "summary",
                2,
                "2026-07-13T12:02:00+00:00",
                120,
                eta_seconds=999,
                rate_per_minute=1.0,
                remaining_cost_est={"USD": 9.99},
            ),
            make_status_event(
                "summary",
                3,
                "2026-07-13T12:04:00+00:00",
                240,
                completed_samples=14,
                sample_total=5658,
                rate_per_minute=2.5,
                eta_seconds=3600,
                preliminary_eta_seconds=4200,
                projected_completion="2026-07-13T13:04:00+00:00",
                remaining_cost_est={
                    "USD": 1.84,
                    "CNY": 19.76,
                    "EUR": 123456.78,
                    "accrued": "ACCRUED_COST_SENTINEL",
                    "total": "TOTAL_COST_SENTINEL",
                    "per_sample": "PER_SAMPLE_COST_SENTINEL",
                },
                cost_calculation={
                    "accrued_cost": "ACCRUED_COST_SENTINEL",
                    "estimated_total": "TOTAL_COST_SENTINEL",
                    "per_sample_cost": "PER_SAMPLE_COST_SENTINEL",
                },
            ),
            make_status_event(
                "stage_completed",
                4,
                "2026-07-13T12:05:00+00:00",
                300,
                sample_id="safe-sample",
                stage="cross",
                model="qwen3-vl-flash",
                elapsed_seconds=4,
                usage={"prompt_tokens": 10, "completion_tokens": 2},
            ),
            make_status_event(
                "stage_started",
                5,
                "2026-07-13T12:06:00+00:00",
                360,
                sample_id="current-sample",
                stage="binary",
                model="qwen3-vl-flash",
            ),
        ]
        for index in range(7):
            events.append(
                make_status_event(
                    "error",
                    6 + index,
                    f"2026-07-13T12:0{7 + index}:00+00:00"
                    if index < 3
                    else f"2026-07-13T12:{10 + index}:00+00:00",
                    420 + index * 30,
                    category=f"category-{index}",
                    error_type="ProviderError",
                    http_status=503,
                    message=(
                        f"Bearer {secret} raw_response={raw_secret} "
                        f"data:image/png;base64,{base64_secret} "
                        f"https://{workspace}.cn-beijing.maas.aliyuncs.com/v1"
                        f"?token={query_secret} Traceback (most recent call last): secret"
                    ),
                    raw_response=raw_secret,
                    prompt="PROMPT_STATUS_SENTINEL",
                    image=f"data:image/png;base64,{base64_secret}",
                )
            )
        for index in range(5):
            events.append(
                make_status_event(
                    "sample_complete",
                    20 + index,
                    f"2026-07-13T12:{14 + index}:00+00:00",
                    720 + index * 30,
                    sample_id=f"backend-{index}",
                    backend_error=True,
                )
            )
        events.append(
            make_status_event(
                "breaker_opened",
                30,
                "2026-07-13T12:15:00+00:00",
                900,
                consecutive=5,
                sample_id="backend-4",
                message=f"Bearer {secret}",
            )
        )
        events.append(
            make_status_event(
                "summary",
                31,
                "2026-07-13T12:16:00+00:00",
                960,
                eta_seconds=-1,
                rate_per_minute=7.5,
                remaining_cost_est={"USD": "not-a-number"},
            )
        )

        deep = {"event": "summary", "eta_seconds": 1}
        nested = {"secret": "DEEP_EVENT_SENTINEL"}
        for _ in range(12):
            nested = {"nested": nested}
        deep["cost_calculation"] = nested

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            event_path = (
                root
                / "scammark_qwen3_dashscope_server_full_v2"
                / "full"
                / "run_events.jsonl"
            )
            event_path.parent.mkdir(parents=True)
            checkpoint_secret = "sk-CHECKPOINTSECRET12345678"
            checkpoint_dir = event_path.parent.parent / "state_full" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / f"{checkpoint_secret}.json").write_text(
                '{"sample_id":', encoding="utf-8"
            )
            event_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events)
                + '{"event":"summary","eta_seconds":NaN}\n'
                + '{"event":"summary","eta_seconds":1e999}\n'
                + json.dumps(deep)
                + "\n"
                + '{"event":"summary","eta_seconds":'
                + ("9" * 5_000)
                + "}\n"
                + '{"event":"summary","eta_seconds":',
                encoding="utf-8",
            )

            with mock.patch(
                "scammark.run_status._utc_now",
                return_value=datetime(
                    2026, 7, 13, 12, 20, 0, tzinfo=timezone.utc
                ),
            ):
                status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )
            rendered = render_status(status)

        self.assertEqual(status["eta_seconds"], 3600)
        self.assertEqual(status["preliminary_eta_seconds"], 4200)
        self.assertEqual(status["throughput"], 2.5)
        self.assertEqual(
            status["projected_completion"],
            "2026-07-13T13:04:00+00:00",
        )
        self.assertEqual(
            status["remaining_cost_est"],
            {"USD": 1.84, "CNY": 19.76},
        )
        self.assertEqual(status["started_at"], "2026-07-13T12:00:00+00:00")
        self.assertEqual(status["elapsed_seconds"], 900)
        self.assertEqual(status["event_age_seconds"], 300)
        self.assertEqual(
            status["current"],
            {"sample_id": "current-sample", "stage": "binary"},
        )
        self.assertEqual(
            status["last_safe"],
            {"sample_id": "safe-sample", "stage": "cross"},
        )
        self.assertEqual(len(status["errors"]), 5)
        self.assertEqual(
            [item["category"] for item in status["errors"]],
            [f"category-{index}" for index in range(2, 7)],
        )
        self.assertEqual(status["breaker"], {"consecutive": 5, "opened": True})
        self.assertIn(
            "ignored truncated final event row",
            status["event_warnings"],
        )
        self.assertIn(
            "remaining cost estimate: CNY 19.76 + USD 1.84",
            rendered,
        )
        self.assertIn("event age: 00:05:00", rendered)
        serialized = json.dumps(status, ensure_ascii=False) + rendered
        for forbidden in (
            secret,
            raw_secret,
            base64_secret,
            query_secret,
            workspace,
            "PROMPT_STATUS_SENTINEL",
            "DEEP_EVENT_SENTINEL",
            "ACCRUED_COST_SENTINEL",
            "TOTAL_COST_SENTINEL",
            "PER_SAMPLE_COST_SENTINEL",
            checkpoint_secret,
            "Traceback",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_status_event_projection_requires_exact_source_provenance(self):
        from scammark.run_status import _project_events

        valid_full = make_status_event(
            "summary",
            1,
            "2026-07-13T12:00:00+00:00",
            1,
            eta_seconds=10,
        )
        wrong_schema = dict(valid_full, schema="other-schema", eta_seconds=20)
        missing_schema = dict(valid_full, eta_seconds=30)
        missing_schema.pop("schema")
        wrong_version = dict(
            valid_full,
            experiment_version="full-v1",
            eta_seconds=40,
        )
        pilot_row = dict(valid_full, run_mode="pilot", eta_seconds=50)
        invalid_core = []
        for field in (
            "attempt_id",
            "sequence",
            "timestamp_utc",
            "process_elapsed_seconds",
        ):
            missing = dict(valid_full, eta_seconds=60)
            missing.pop(field)
            invalid_core.append(missing)
        invalid_core.extend(
            (
                dict(valid_full, attempt_id="", eta_seconds=70),
                dict(valid_full, sequence=0, eta_seconds=80),
                dict(valid_full, timestamp_utc="not-a-timestamp", eta_seconds=90),
                dict(valid_full, process_elapsed_seconds=-1, eta_seconds=100),
            )
        )
        missing_local = dict(valid_full, eta_seconds=110)
        missing_local.pop("timestamp_local")
        invalid_core.extend(
            (
                missing_local,
                dict(
                    valid_full,
                    timestamp_local="not-a-timestamp",
                    eta_seconds=120,
                ),
                dict(
                    valid_full,
                    timestamp_local="2026-07-13T19:00:00+07:00",
                    eta_seconds=130,
                ),
                dict(
                    valid_full,
                    timestamp_local="2026-07-13T20:00:01+08:00",
                    eta_seconds=140,
                ),
            )
        )

        projected_full = _project_events(
            [
                wrong_schema,
                missing_schema,
                wrong_version,
                pilot_row,
                *invalid_core,
                valid_full,
            ],
            expected_run_mode="full",
        )
        projected_pilot = _project_events(
            [valid_full, pilot_row],
            expected_run_mode="pilot",
        )

        self.assertEqual(
            [event["eta_seconds"] for event in projected_full],
            [10],
        )
        self.assertEqual(
            [event["eta_seconds"] for event in projected_pilot],
            [50],
        )

    def test_status_renderer_and_tmux_fail_closed_for_hostile_objects(self):
        from scammark.run_status import query_tmux, render_status

        class HostileResult:
            returncode = 0

            @property
            def stdout(self):
                raise RuntimeError("RAW_RESPONSE_HOSTILE_RESULT")

        class HostileMapping(dict):
            def get(self, key, default=None):
                raise RuntimeError("RAW_RESPONSE_HOSTILE_MAPPING")

        self.assertEqual(
            query_tmux("scammark-full-v2", runner=lambda *args, **kwargs: HostileResult()).state,
            "absent",
        )
        self.assertEqual(render_status(HostileMapping()), "tmux: absent")

        rendered = render_status(
            {
                "tmux": {"state": "dead"},
                "eta_seconds": 10**100_000,
                "remaining_cost_est": {
                    "USD": float("nan"),
                    "CNY": float("inf"),
                },
                "errors": [
                    {
                        "category": "unexpected",
                        "message": (
                            "raw_response=RAW_RESPONSE_RENDER_SENTINEL "
                            "Traceback (most recent call last): secret"
                        ),
                    }
                ],
                "next_action": "Bearer sk-HOSTILERENDER12345678",
            }
        )
        for forbidden in (
            "RAW_RESPONSE_RENDER_SENTINEL",
            "Traceback",
            "HOSTILERENDER",
            "remaining cost estimate",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_status_renderer_bounds_every_exact_integer_field(self):
        from scammark.run_status import render_status

        class HostileInt(int):
            def __str__(self):
                raise RuntimeError("hostile integer string conversion")

            def __format__(self, format_spec):
                raise RuntimeError("hostile integer formatting")

        unsafe_values = (
            10**5_000,
            True,
            -1,
            HostileInt(7),
        )
        for unsafe in unsafe_values:
            rendered = render_status(
                {
                    "tmux": {
                        "state": "dead",
                        "pid": unsafe,
                        "exit_code": unsafe,
                    },
                    "samples": {
                        "complete": unsafe,
                        "partial": unsafe,
                        "error": unsafe,
                        "total": unsafe,
                    },
                    "stages": {
                        "complete": unsafe,
                        "reused": unsafe,
                        "failed": unsafe,
                    },
                    "calls": {
                        "reserved": unsafe,
                        "finished": unsafe,
                        "outstanding": unsafe,
                        "remaining": unsafe,
                    },
                    "breaker": {"consecutive": unsafe, "opened": True},
                    "next_action": "safe action",
                }
            )
            self.assertEqual(
                rendered,
                "tmux: dead\nnext action: safe action",
            )
            self.assertLess(len(rendered), 100)

        huge = unsafe_values[0]
        mixed = render_status(
            {
                "tmux": {"state": "dead", "pid": huge, "exit_code": 20},
                "samples": {
                    "complete": 1,
                    "partial": 0,
                    "error": 0,
                    "total": 2,
                },
                "stages": {"complete": 5, "reused": huge, "failed": 0},
                "calls": {
                    "reserved": huge,
                    "finished": 2,
                    "outstanding": -1,
                    "remaining": HostileInt(7),
                },
                "breaker": {"consecutive": huge, "opened": False},
            }
        )
        self.assertIn("tmux: dead exit=20", mixed)
        self.assertIn("complete: 1 / 2", mixed)
        self.assertNotIn("stages:", mixed)
        self.assertIn("calls: finished=2", mixed)
        self.assertNotIn("breaker:", mixed)

    def test_status_without_summary_replays_pilot_and_full_in_memory_read_only(self):
        from scammark.observability import (
            ProgressTracker,
            project_remaining_stage_counts,
        )
        from scammark.run_status import collect_status

        base = ("url", "text", "visual", "cross", "binary")
        models = {
            "url": "gpt-4o-mini",
            "text": "gpt-4o-mini",
            "visual": "qwen3-vl-flash",
            "cross": "qwen3-vl-flash",
            "binary": "qwen3-vl-flash",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_tracker_status_config(root)
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            full = output / "full"
            pilot = output / "pilot_10"
            checkpoints = output / "state_full" / "checkpoints"
            full.mkdir(parents=True)
            pilot.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            manifest = [
                {"sample_id": "done", "binary": "benign"},
                {"sample_id": "pending", "binary": "benign"},
            ]
            (full / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            config_hash = "d" * 64
            (output / "state_full" / "run_config.json").write_text(
                json.dumps({"config_hash": config_hash}), encoding="utf-8"
            )
            write_status_checkpoint(
                checkpoints,
                "done",
                config_hash,
                complete=base,
                binary="benign",
            )

            pilot_events = []
            full_events = []
            for index, stage in enumerate(base, start=1):
                pilot_events.append(
                    make_status_event(
                        "stage_completed",
                        index,
                        f"2026-07-13T10:00:{index:02d}+00:00",
                        index * 3,
                        sample_id="pilot",
                        stage=stage,
                        model=models[stage],
                        elapsed_seconds=3,
                        usage={
                            "prompt_tokens": 1_000_000,
                            "completion_tokens": 0,
                        },
                        run_mode="pilot",
                    )
                )
                full_events.append(
                    make_status_event(
                        "stage_completed",
                        index,
                        f"2026-07-13T11:00:{index:02d}+00:00",
                        index * 2,
                        sample_id="done",
                        stage=stage,
                        model=models[stage],
                        elapsed_seconds=2,
                        usage={
                            "prompt_tokens": 1_000_000,
                            "completion_tokens": 0,
                        },
                    )
                )
            full_events.append(
                make_status_event(
                    "sample_complete",
                    6,
                    "2026-07-13T11:00:12+00:00",
                    12,
                    sample_id="done",
                    sample_index=1,
                    sample_total=2,
                    backend_error=False,
                )
            )
            (pilot / "run_events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in pilot_events),
                encoding="utf-8",
            )
            (full / "run_events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in full_events),
                encoding="utf-8",
            )
            (pilot / "metrics.json").write_text(
                json.dumps(
                    {
                        "binary": {
                            "confusion": {"tp": 0, "fp": 0, "tn": 10, "fn": 0}
                        },
                        "engineering": {
                            "sample_count": 10,
                            "acceptance_gates": {"passed": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            before = snapshot_filesystem(root)
            original_open = Path.open

            def read_only_open(path, mode="r", *args, **kwargs):
                if any(flag in mode for flag in ("w", "a", "x", "+")):
                    raise AssertionError(f"status attempted write-open: {mode}")
                return original_open(path, mode, *args, **kwargs)

            with (
                mock.patch(
                    "scammark.experiment_state.ExperimentState",
                    side_effect=AssertionError("status constructed ExperimentState"),
                ),
                mock.patch(
                    "scammark.observability.RunEventLogger",
                    side_effect=AssertionError("status constructed RunEventLogger"),
                ),
                mock.patch.object(
                    Path,
                    "mkdir",
                    side_effect=AssertionError("status attempted mkdir"),
                ),
                mock.patch.object(Path, "open", new=read_only_open),
                mock.patch.object(
                    Path,
                    "write_text",
                    side_effect=AssertionError("status attempted write_text"),
                ),
                mock.patch.object(
                    Path,
                    "write_bytes",
                    side_effect=AssertionError("status attempted write_bytes"),
                ),
                mock.patch(
                    "scammark.run_status.ProgressTracker",
                    wraps=ProgressTracker,
                ) as tracker_type,
                mock.patch(
                    "scammark.run_status.project_remaining_stage_counts",
                    wraps=project_remaining_stage_counts,
                ) as projection,
            ):
                status = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )
            after = snapshot_filesystem(root)

        self.assertEqual(after, before)
        tracker_type.assert_called_once()
        projection.assert_called_once()
        self.assertEqual(
            projection.call_args.kwargs["smoke_benign_false_positive_rate"],
            0.0,
        )
        self.assertEqual(
            status["remaining_stage_counts"],
            {
                "url": 1,
                "text": 1,
                "visual": 1,
                "cross": 1,
                "binary": 1,
                "type": 0,
                "grounding": 0,
            },
        )
        self.assertEqual(status["eta_seconds"], 10)
        self.assertEqual(status["preliminary_eta_seconds"], 15)
        self.assertEqual(
            status["remaining_cost_est"],
            {"USD": 2.0, "CNY": 6.0},
        )

    def test_status_without_observations_does_not_fabricate_zero_estimates(self):
        from scammark.run_status import collect_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_tracker_status_config(root, total=1)
            manifest = (
                root
                / "scammark_qwen3_dashscope_server_full_v2"
                / "full"
                / "manifest.jsonl"
            )
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '{"sample_id":"pending","binary":"benign"}\n',
                encoding="utf-8",
            )

            status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

        self.assertIsNone(status["eta_seconds"])
        self.assertIsNone(status["preliminary_eta_seconds"])
        self.assertEqual(status["remaining_cost_est"], {})
        self.assertEqual(
            status["remaining_stage_counts"],
            {
                "url": 1,
                "text": 1,
                "visual": 1,
                "cross": 1,
                "binary": 1,
                "type": 1,
                "grounding": 1,
            },
        )

    def test_status_lifecycle_artifacts_and_delivery_choose_exact_actions(self):
        from scammark.run_status import choose_next_action, collect_status

        def dead(exit_code):
            return lambda args, **kwargs: subprocess.CompletedProcess(
                args, 0, f"1|321|{exit_code}\n", ""
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_tracker_status_config(root, total=1)
            output = root / "scammark_qwen3_dashscope_server_full_v2"
            full = output / "full"
            pilot = output / "pilot_10"
            full.mkdir(parents=True)
            event_path = full / "run_events.jsonl"

            graceful = make_status_event(
                "run_stopped",
                1,
                "2026-07-13T12:00:00+00:00",
                10,
                reason="signal",
                category="graceful_stop",
                exit_code=130,
                forced=False,
            )
            event_path.write_text(json.dumps(graceful) + "\n", encoding="utf-8")
            graceful_status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=dead(130),
            )

            forced = make_status_event(
                "shell_exited",
                2,
                "2026-07-13T12:01:00+00:00",
                20,
                exit_code=143,
                forced=True,
                runner_attempt_id="status-attempt",
            )
            event_path.write_text(json.dumps(forced) + "\n", encoding="utf-8")
            forced_status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=dead(143),
            )

            event_path.write_text("", encoding="utf-8")
            (full / "preflight.json").write_text(
                json.dumps({"status": "validated", "selected_sample_count": 1}),
                encoding="utf-8",
            )
            preflight_status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

            pilot.mkdir()
            (pilot / "metrics.json").write_text(
                json.dumps(
                    {
                        "binary": {
                            "confusion": {"tp": 1, "fp": 0, "tn": 9, "fn": 0}
                        },
                        "engineering": {
                            "sample_count": 10,
                            "acceptance_gates": {"passed": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            smoke_status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

            config_hash = "2" * 64
            checkpoints = output / "state_full" / "checkpoints"
            checkpoints.mkdir(parents=True)
            (output / "state_full" / "run_config.json").write_text(
                json.dumps({"config_hash": config_hash}), encoding="utf-8"
            )
            (full / "manifest.jsonl").write_text(
                '{"sample_id":"complete","binary":"benign"}\n',
                encoding="utf-8",
            )
            (full / "predictions.jsonl").write_text(
                '{"sample_id":"complete"}\n', encoding="utf-8"
            )
            write_status_checkpoint(
                checkpoints,
                "complete",
                config_hash,
                complete=("url", "text", "visual", "cross", "binary"),
                binary="benign",
            )
            run_complete = make_status_event(
                "run_completed",
                3,
                "2026-07-13T12:02:00+00:00",
                30,
                sample_count=1,
                gates_passed=True,
            )
            event_path.write_text(
                json.dumps(run_complete) + "\n", encoding="utf-8"
            )
            inference_status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

            (full / "metrics_final.json").write_text(
                json.dumps(
                    {
                        "rationale": {
                            "end_to_end": {"bertscore_status": "computed"}
                        },
                        "engineering": {"prediction_sample_count": 1},
                    }
                ),
                encoding="utf-8",
            )
            (full / "delivery_state.json").write_text(
                json.dumps(
                    {
                        "mode": "inference",
                        "packaged": True,
                        "archive_sha256": "a" * 64,
                        "uploaded": True,
                        "published": False,
                        "last_exit_code": 32,
                    }
                ),
                encoding="utf-8",
            )
            delivery_status = collect_status(
                config,
                tmux_session="scammark-full-v2",
                tmux_runner=fake_absent_tmux,
            )

        self.assertEqual(graceful_status["exit_code"], 130)
        self.assertTrue(graceful_status["graceful_stop_event"])
        self.assertEqual(graceful_status["next_action"], "resume inference")
        self.assertEqual(forced_status["exit_code"], 143)
        self.assertFalse(forced_status["graceful_stop_event"])
        self.assertEqual(
            forced_status["next_action"],
            "inspect forced interruption and orphan call",
        )
        self.assertTrue(preflight_status["preflight_complete"])
        self.assertFalse(preflight_status["smoke_complete"])
        self.assertEqual(
            preflight_status["next_action"],
            "run paid 10-page smoke",
        )
        self.assertTrue(smoke_status["smoke_complete"])
        self.assertIn("resume inference", smoke_status["next_action"])
        self.assertTrue(inference_status["inference_complete"])
        self.assertFalse(inference_status["bertscore_complete"])
        self.assertEqual(
            inference_status["next_action"],
            "bash scripts/server/recover_delivery.sh package",
        )
        self.assertTrue(delivery_status["bertscore_complete"])
        self.assertEqual(delivery_status["delivery_failure"], 32)
        self.assertEqual(
            delivery_status["next_action"],
            "bash scripts/server/recover_delivery.sh publish",
        )
        self.assertIn(
            "contract",
            choose_next_action(
                {"calls": {"corrupt": True}, "delivery_failure": 31}
            ),
        )

    def test_status_reads_two_mode_delivery_state_without_writing_and_prioritizes_modes(self):
        from scammark.run_status import collect_status

        def entry(
            mode,
            *,
            packaged=False,
            uploaded=False,
            published=False,
            exit_code=None,
        ):
            return {
                "mode": mode,
                "packaged": packaged,
                "archive_sha256": "a" * 64 if packaged else None,
                "uploaded": uploaded,
                "published": published,
                "last_exit_code": exit_code,
            }

        def envelope(inference, final):
            return {
                "schema": "scammark-delivery-state-v1",
                "modes": {"inference": inference, "final": final},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            full = root / "scammark_qwen3_dashscope_server_full_v2" / "full"
            full.mkdir(parents=True)
            state_path = full / "delivery_state.json"
            state_path.write_text(
                json.dumps(
                    envelope(
                        entry(
                            "inference",
                            packaged=True,
                            exit_code=31,
                        ),
                        entry("final"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            before = snapshot_filesystem(root)
            with (
                mock.patch(
                    "scammark.run_status._inference_complete",
                    return_value=True,
                ),
                mock.patch(
                    "scammark.run_status._bertscore_complete",
                    return_value=False,
                ),
            ):
                inference_failed = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )
            self.assertEqual(snapshot_filesystem(root), before)

            actions = {}
            for final_stage, final_entry in (
                ("package", entry("final")),
                ("upload", entry("final", packaged=True)),
                (
                    "publish",
                    entry("final", packaged=True, uploaded=True),
                ),
            ):
                state_path.write_text(
                    json.dumps(
                        envelope(
                            entry(
                                "inference",
                                packaged=True,
                                uploaded=True,
                                published=True,
                            ),
                            final_entry,
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with (
                    mock.patch(
                        "scammark.run_status._inference_complete",
                        return_value=True,
                    ),
                    mock.patch(
                        "scammark.run_status._bertscore_complete",
                        return_value=True,
                    ),
                ):
                    status = collect_status(
                        config,
                        tmux_session="scammark-full-v2",
                        tmux_runner=fake_absent_tmux,
                    )
                actions[final_stage] = status["next_action"]

            state_path.write_text(
                json.dumps(
                    envelope(
                        entry(
                            "inference",
                            packaged=True,
                            uploaded=True,
                            published=True,
                        ),
                        {
                            **entry("final"),
                            "packaged": 1,
                            "archive_sha256": "a" * 64,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch(
                    "scammark.run_status._inference_complete",
                    return_value=True,
                ),
                mock.patch(
                    "scammark.run_status._bertscore_complete",
                    return_value=True,
                ),
            ):
                corrupt = collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        self.assertEqual(inference_failed["delivery_failure"], 31)
        self.assertEqual(
            inference_failed["next_action"],
            "bash scripts/server/recover_delivery.sh upload",
        )
        self.assertEqual(
            actions,
            {
                "package": "bash scripts/server/recover_delivery.sh --mode final package",
                "upload": "bash scripts/server/recover_delivery.sh --mode final upload",
                "publish": "bash scripts/server/recover_delivery.sh --mode final publish",
            },
        )
        self.assertTrue(corrupt["delivery_corrupt"])
        self.assertIn("contract", corrupt["next_action"])

    def test_status_delivery_state_strict_read_rejects_aliases_and_duplicate_keys(self):
        from scammark.run_status import collect_status

        def collect(config):
            with (
                mock.patch(
                    "scammark.run_status._inference_complete",
                    return_value=True,
                ),
                mock.patch(
                    "scammark.run_status._bertscore_complete",
                    return_value=True,
                ),
            ):
                return collect_status(
                    config,
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )

        final_entry = {
            "mode": "final",
            "packaged": True,
            "archive_sha256": "b" * 64,
            "uploaded": False,
            "published": False,
            "last_exit_code": None,
        }
        inference_entry = {
            **final_entry,
            "mode": "inference",
            "archive_sha256": "a" * 64,
        }
        envelope = {
            "schema": "scammark-delivery-state-v1",
            "modes": {
                "inference": inference_entry,
                "final": final_entry,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            full = root / "scammark_qwen3_dashscope_server_full_v2" / "full"
            full.mkdir(parents=True)
            state_path = full / "delivery_state.json"

            state_path.write_text(
                json.dumps(final_entry) + "\n",
                encoding="utf-8",
            )
            before = snapshot_filesystem(root)
            legacy_final = collect(config)
            self.assertEqual(snapshot_filesystem(root), before)
            self.assertEqual(
                legacy_final["delivery_modes"]["inference"],
                {"stage": "package", "status": "pending"},
            )
            self.assertEqual(
                legacy_final["delivery_modes"]["final"],
                {"stage": "upload", "status": "pending"},
            )

            state_path.write_text(
                '{"stage":"publish","status":"failed","exit_code":32}\n',
                encoding="utf-8",
            )
            relaxed_stage = collect(config)
            self.assertTrue(relaxed_stage["delivery_corrupt"])
            self.assertIn("contract", relaxed_stage["next_action"])
            state_path.write_text(
                '{"packaged":true,"uploaded":false,"published":false}\n',
                encoding="utf-8",
            )
            relaxed_flags = collect(config)
            self.assertTrue(relaxed_flags["delivery_corrupt"])
            self.assertIn("contract", relaxed_flags["next_action"])

            duplicate = json.dumps(envelope).replace(
                '"schema": "scammark-delivery-state-v1",',
                '"schema": "scammark-delivery-state-v1", '
                '"schema": "scammark-delivery-state-v1",',
                1,
            )
            state_path.write_text(duplicate + "\n", encoding="utf-8")
            duplicate_before = state_path.read_bytes()
            duplicate_status = collect(config)
            self.assertEqual(state_path.read_bytes(), duplicate_before)
            self.assertTrue(duplicate_status["delivery_corrupt"])
            self.assertIn("contract", duplicate_status["next_action"])

            target = full / "delivery-state-target.json"
            target.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
            target_before = (target.read_bytes(), target.stat().st_mtime_ns)
            state_path.unlink()
            state_path.symlink_to(target.name)
            symlink_status = collect(config)
            self.assertTrue(state_path.is_symlink())
            self.assertEqual(
                (target.read_bytes(), target.stat().st_mtime_ns),
                target_before,
            )
            self.assertTrue(symlink_status["delivery_corrupt"])
            self.assertIn("contract", symlink_status["next_action"])

            state_path.unlink()
            os.link(target, state_path)
            hardlink_status = collect(config)
            self.assertEqual(
                (target.read_bytes(), target.stat().st_mtime_ns),
                target_before,
            )
            self.assertEqual(target.stat().st_nlink, 2)
            self.assertTrue(hardlink_status["delivery_corrupt"])
            self.assertIn("contract", hardlink_status["next_action"])

            state_path.unlink()
            missing_target = full / "missing-delivery-state.json"
            state_path.symlink_to(missing_target.name)
            (full / "run_events.jsonl").write_text(
                json.dumps(
                    make_status_event(
                        "delivery_stage",
                        1,
                        "2026-07-13T12:00:00+00:00",
                        1,
                        stage="upload",
                        status="failed",
                        exit_code=31,
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            broken_symlink_status = collect(config)
            self.assertTrue(state_path.is_symlink())
            self.assertFalse(missing_target.exists())
            self.assertTrue(broken_symlink_status["delivery_corrupt"])
            self.assertIn("contract", broken_symlink_status["next_action"])

    def test_status_cli_requires_arguments_and_is_read_only_for_all_tmux_states(self):
        from scammark.run_status import main

        for argv in ([], ["--config", "v2.toml"], ["--tmux-session", "v2"]):
            with (
                self.subTest(argv=argv),
                mock.patch("sys.stderr", new=io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                main(argv)
            self.assertEqual(caught.exception.code, 2)

        outputs = {
            "absent": subprocess.CompletedProcess([], 1, "", "missing"),
            "running": subprocess.CompletedProcess([], 0, "0|456|\n", ""),
            "dead": subprocess.CompletedProcess([], 0, "1|456|143\n", ""),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            before = snapshot_filesystem(root)
            original_open = Path.open

            def read_only_open(path, mode="r", *args, **kwargs):
                if any(flag in mode for flag in ("w", "a", "x", "+")):
                    raise AssertionError(f"CLI attempted write-open: {mode}")
                return original_open(path, mode, *args, **kwargs)

            for state, completed in outputs.items():
                stream = io.StringIO()
                with (
                    self.subTest(state=state),
                    mock.patch(
                        "scammark.run_status.subprocess.run",
                        return_value=completed,
                    ),
                    mock.patch.object(Path, "open", new=read_only_open),
                    mock.patch.object(
                        Path,
                        "mkdir",
                        side_effect=AssertionError("CLI attempted mkdir"),
                    ),
                    mock.patch.object(
                        Path,
                        "write_text",
                        side_effect=AssertionError("CLI attempted write_text"),
                    ),
                    mock.patch("sys.stdout", new=stream),
                ):
                    result = main(
                        [
                            "--config",
                            str(config),
                            "--tmux-session",
                            "scammark-full-v2",
                        ]
                    )
                self.assertEqual(result, 0)
                self.assertIn(f"tmux: {state}", stream.getvalue())
            after = snapshot_filesystem(root)

        self.assertEqual(after, before)

    def test_status_module_main_guard_uses_injected_tmux_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_minimal_status_config(root)
            before = snapshot_filesystem(root)
            stream = io.StringIO()
            argv = [
                "scammark.run_status",
                "--config",
                str(config),
                "--tmux-session",
                "scammark-full-v2",
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch("subprocess.run") as runner,
                    mock.patch("sys.stdout", new=stream),
                    self.assertRaises(SystemExit) as caught,
                ):
                    runner.return_value = subprocess.CompletedProcess(
                        [], 1, "", "missing"
                    )
                    runpy.run_module("scammark.run_status", run_name="__main__")
            after = snapshot_filesystem(root)

        self.assertEqual(caught.exception.code, 0)
        self.assertIn("tmux: absent", stream.getvalue())
        self.assertEqual(after, before)

    def test_status_cli_config_failure_has_no_traceback_or_secret(self):
        from scammark.run_status import main

        secret = "sk-CLI-CONFIG-SECRET-12345678"
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "invalid.toml"
            config.write_text(
                f'[paths]\noutput_root = "{secret}"\ninvalid = [',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("sys.stdout", new=stdout),
                mock.patch("sys.stderr", new=stderr),
                self.assertRaises(SystemExit) as caught,
            ):
                main(
                    [
                        "--config",
                        str(config),
                        "--tmux-session",
                        "scammark-full-v2",
                    ]
                )

        self.assertEqual(caught.exception.code, 2)
        rendered = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(secret, rendered)
        self.assertNotIn("Traceback", rendered)

    def test_failure_classification_distinguishes_quota_429_from_transient_429(self):
        from scammark.experiment_state import (
            ApiCallLimitError,
            ExperimentContractError,
        )
        from scammark.observability import classify_failure
        from scammark.reliability import (
            PermanentRequestError,
            TransientRequestError,
        )

        cases = [
            (
                PermanentRequestError("HTTP 401: invalid api key"),
                ("auth_billing", True, 20),
            ),
            (
                PermanentRequestError("HTTP 402: insufficient balance"),
                ("auth_billing", True, 20),
            ),
            (
                PermanentRequestError(
                    "HTTP 429: quota exhausted for account"
                ),
                ("auth_billing", True, 20),
            ),
            (
                TransientRequestError("HTTP 429: rate limited"),
                ("transient", False, None),
            ),
            (
                PermanentRequestError("HTTP 403: content policy refusal"),
                ("sample_permanent", False, None),
            ),
            (
                PermanentRequestError(
                    "HTTP 403: content policy refusal for this account"
                ),
                ("sample_permanent", False, None),
            ),
            (
                PermanentRequestError("HTTP 403: account is disabled"),
                ("auth_billing", True, 20),
            ),
            (
                ApiCallLimitError("Physical API call cap reached (50000)"),
                ("api_budget", True, 21),
            ),
            (
                ExperimentContractError("configuration changed"),
                ("contract_drift", True, 23),
            ),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__, message=str(error)):
                result = classify_failure(error)
                self.assertEqual(
                    (result.category, result.systemic, result.exit_code),
                    expected,
                )

    def test_failure_classification_walks_cycles_and_binds_typed_errors_first(self):
        from scammark.experiment_state import (
            ApiCallLimitError,
            CheckpointContractError,
        )
        from scammark.observability import classify_failure
        from scammark.reliability import TransientRequestError

        outer = TransientRequestError("HTTP 503: retry later")
        cap = ApiCallLimitError("physical cap reached")
        contract = CheckpointContractError("checkpoint changed")
        outer.__cause__ = contract
        outer.__context__ = cap
        contract.__context__ = outer

        result = classify_failure(outer)

        self.assertEqual(result.category, "api_budget")
        self.assertEqual(result.exit_code, 21)
        self.assertIsNone(result.http_status)

    def test_failure_classification_extracts_only_explicit_http_statuses(self):
        from scammark.observability import classify_failure

        class Response:
            status_code = 503

        class ResponseError(RuntimeError):
            response = Response()

        class MisleadingError(RuntimeError):
            status_code = True

        cases = [
            (ResponseError("provider unavailable"), ("transient", 503)),
            (RuntimeError("HTTP status=408 timeout"), ("transient", 408)),
            (RuntimeError("budget 50000 reached"), ("unexpected", None)),
            (MisleadingError("ordinary failure"), ("unexpected", None)),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                result = classify_failure(error)
                self.assertEqual((result.category, result.http_status), expected)

    def test_failure_classification_covers_transport_sample_local_and_unexpected(self):
        from scammark.observability import classify_failure

        cases = [
            (TimeoutError("read timed out"), "transient"),
            (ConnectionError("connection reset by peer"), "transient"),
            (ValueError("input schema is invalid"), "sample_permanent"),
            (RuntimeError("moderation rejected this sample"), "sample_permanent"),
            (RuntimeError("ordinary local bug"), "unexpected"),
        ]
        for error, category in cases:
            with self.subTest(error=type(error).__name__, category=category):
                result = classify_failure(error)
                self.assertEqual(result.category, category)
                self.assertFalse(result.systemic)
                self.assertIsNone(result.exit_code)

    def test_failure_classification_handles_real_key_text_permanent_4xx_and_error_types(self):
        from scammark.observability import classify_failure
        from scammark.reliability import PermanentRequestError

        class SchemaRepairError(RuntimeError):
            pass

        class ModerationError(RuntimeError):
            pass

        class InputValidationError(RuntimeError):
            pass

        cases = [
            (
                PermanentRequestError(
                    "OPENAI_API_KEY is not set; configure an API key"
                ),
                ("auth_billing", None),
            ),
            (
                PermanentRequestError("HTTP 403: forbidden"),
                ("sample_permanent", 403),
            ),
            (
                PermanentRequestError("HTTP 422: rejected for this item"),
                ("sample_permanent", 422),
            ),
            (SchemaRepairError("stage failed"), ("sample_permanent", None)),
            (ModerationError("stage failed"), ("sample_permanent", None)),
            (InputValidationError("stage failed"), ("sample_permanent", None)),
            (SchemaRepairError("HTTP 408: stage failed"), ("transient", 408)),
            (ModerationError("HTTP 429: stage failed"), ("transient", 429)),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__, message=str(error)):
                result = classify_failure(error)
                self.assertEqual((result.category, result.http_status), expected)

    def test_failure_classification_handles_common_quota_bodies_without_broadening_404(self):
        from scammark.observability import classify_failure
        from scammark.reliability import PermanentRequestError

        cases = [
            (
                PermanentRequestError(
                    "HTTP 429: You exceeded your current quota, check billing"
                ),
                "auth_billing",
            ),
            (
                PermanentRequestError(
                    'HTTP 429: {"code":"insufficient_quota"}'
                ),
                "auth_billing",
            ),
            (
                PermanentRequestError(
                    'HTTP 429: {"code":"quota_exceeded"}'
                ),
                "transient",
            ),
            (
                PermanentRequestError(
                    'HTTP 429: {"code":"credit_exhausted"}'
                ),
                "auth_billing",
            ),
            (
                PermanentRequestError(
                    "HTTP 429: Your account has run out of credits"
                ),
                "auth_billing",
            ),
            (
                PermanentRequestError(
                    "HTTP 429: credit balance is too low"
                ),
                "auth_billing",
            ),
            (PermanentRequestError("HTTP 404"), "unexpected"),
            (
                PermanentRequestError(
                    "HTTP 422: rejected for this input item"
                ),
                "sample_permanent",
            ),
        ]
        for error, category in cases:
            with self.subTest(message=str(error)):
                self.assertEqual(classify_failure(error).category, category)

    def test_failure_classification_makes_every_non_auth_403_sample_permanent(self):
        from scammark.observability import classify_failure
        from scammark.reliability import (
            PermanentRequestError,
            TransientRequestError,
        )

        cases = [
            (TransientRequestError("HTTP 403: forbidden"), "sample_permanent"),
            (RuntimeError("HTTP 403: forbidden"), "sample_permanent"),
            (
                TransientRequestError(
                    "HTTP 403: content policy refusal for this account"
                ),
                "sample_permanent",
            ),
            (
                TransientRequestError(
                    "HTTP 403: content policy refusal; account is deactivated"
                ),
                "sample_permanent",
            ),
            (
                PermanentRequestError("HTTP 403: account is disabled"),
                "auth_billing",
            ),
            (
                TransientRequestError("HTTP 403: invalid api key"),
                "auth_billing",
            ),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__, message=str(error)):
                self.assertEqual(classify_failure(error).category, expected)

    def test_failure_classification_content_filter_precedes_account_403(self):
        from scammark.observability import classify_failure

        cases = [
            (
                RuntimeError(
                    'HTTP 403: {"code":"content_filter"}; '
                    "account is deactivated"
                ),
                "sample_permanent",
            ),
            (
                RuntimeError(
                    'HTTP 403: {"code":"content_policy"}; '
                    "account is disabled"
                ),
                "sample_permanent",
            ),
            (
                RuntimeError('HTTP 403: {"code":"account_deactivated"}'),
                "auth_billing",
            ),
            (
                RuntimeError('HTTP 403: {"code":"invalid_api_key"}'),
                "auth_billing",
            ),
        ]
        for error, expected in cases:
            with self.subTest(message=str(error)):
                self.assertEqual(classify_failure(error).category, expected)

    def test_failure_classification_reads_response_body_for_auth_not_rate_limit(self):
        from scammark.observability import classify_failure

        class Response:
            status_code = 429

            def __init__(self, body):
                self.body = body

        class ResponseFailure(RuntimeError):
            def __init__(self, body):
                super().__init__("provider request failed")
                self.response = Response(body)

        cases = [
            (
                RuntimeError('HTTP 429: {"code":"invalid_api_key"}'),
                "auth_billing",
            ),
            (
                RuntimeError('HTTP 429: {"code":"missing_api_key"}'),
                "auth_billing",
            ),
            (
                RuntimeError('HTTP 403: {"code":"account_deactivated"}'),
                "auth_billing",
            ),
            (
                ResponseFailure('{"code":"quota_exceeded"}'),
                "transient",
            ),
            (
                ResponseFailure(b'{"code":"invalid_api_key"}'),
                "auth_billing",
            ),
            (
                ResponseFailure('{"code":"rate_limit_exceeded"}'),
                "transient",
            ),
        ]
        for error, expected in cases:
            with self.subTest(body=error.response.body if hasattr(error, "response") else str(error)):
                self.assertEqual(classify_failure(error).category, expected)

    def test_failure_classification_merges_direct_and_response_analysis_fields(self):
        from scammark.observability import classify_failure

        class Response:
            status_code = 429
            body = "generic provider failure"
            text = '{"code":"quota_exceeded"}'

        class ResponseFieldsError(RuntimeError):
            response = Response()

        direct = RuntimeError("provider request failed")
        direct.status_code = 429
        direct.body = "generic provider failure"
        direct.response_body = b'{"code":"invalid_api_key"}'
        direct.text = "provider request failed"

        for error, expected in (
            (ResponseFieldsError("request failed"), "transient"),
            (direct, "auth_billing"),
        ):
            with self.subTest(error=type(error).__name__):
                result = classify_failure(error)
                self.assertEqual(result.category, expected)
                self.assertNotIn("quota_exceeded", result.message)
                self.assertNotIn("invalid_api_key", result.message)

    def test_failure_classification_projects_bounded_safe_structured_bodies(self):
        from scammark.observability import classify_failure

        class Response:
            status_code = 429

            def __init__(self, body):
                self.body = body

        class BodyError(RuntimeError):
            def __init__(self, body):
                super().__init__("provider request failed")
                self.response = Response(body)

        class MustNotStringify:
            def __init__(self):
                self.calls = 0

            def __str__(self):
                self.calls += 1
                raise RuntimeError("arbitrary object was stringified")

        classified_bodies = [
            ({"error": {"code": "insufficient_quota"}}, "auth_billing"),
            ([{"code": "quota_exceeded"}], "transient"),
            (({"code": "credit_exhausted"},), "auth_billing"),
        ]
        for body, expected in classified_bodies:
            with self.subTest(body_type=type(body).__name__):
                result = classify_failure(BodyError(body))
                self.assertEqual(result.category, expected)
                self.assertNotIn("quota", result.message.lower())

        opaque = MustNotStringify()
        cyclic = []
        cyclic.append(cyclic)
        safe_body = {
            "code": "rate_limit_exceeded",
            "opaque": opaque,
            "cyclic": cyclic,
            "primitive": 17,
        }
        self.assertEqual(
            classify_failure(BodyError(safe_body)).category,
            "transient",
        )
        self.assertEqual(opaque.calls, 0)

    def test_failure_classification_bounds_hostile_structures_without_raising(self):
        from scammark import observability

        class Response:
            status_code = 429

            def __init__(self, body):
                self.body = body

        class BodyError(RuntimeError):
            def __init__(self, body):
                super().__init__("provider request failed")
                self.response = Response(body)

        huge_integer = 10**5000
        cyclic_list = []
        cyclic_list.append(cyclic_list)
        cyclic_dict = {"cycle": cyclic_list, "huge": huge_integer}
        huge_list = [huge_integer, cyclic_dict] + [None] * 50_000

        for body in (huge_integer, cyclic_dict, huge_list):
            with self.subTest(body_type=type(body).__name__):
                result = observability.classify_failure(BodyError(body))
                self.assertEqual(result.category, "transient")

        with mock.patch.object(
            observability.json,
            "dumps",
            side_effect=ValueError("forced serializer rejection"),
        ):
            result = observability.classify_failure(
                BodyError({"detail": "generic provider failure"})
            )
        self.assertEqual(result.category, "transient")

    def test_failure_classification_priority_fetches_late_semantic_keys(self):
        from scammark.observability import classify_failure

        class Response:
            status_code = 429

            def __init__(self, body):
                self.body = body

        class BodyError(RuntimeError):
            def __init__(self, body):
                super().__init__("provider request failed")
                self.response = Response(body)

        priority_keys = (
            "error",
            "code",
            "error_code",
            "message",
            "type",
            "status",
            "detail",
        )
        for priority_key in priority_keys:
            body = {
                f"padding_{index}": "generic"
                for index in range(10_000)
            }
            body[priority_key] = "insufficient_quota"
            with self.subTest(priority_key=priority_key):
                result = classify_failure(BodyError(body))
                self.assertEqual(result.category, "auth_billing")

        nested = {
            f"padding_{index}": "generic" for index in range(40)
        }
        nested["code"] = "insufficient_quota"
        result = classify_failure(BodyError({"error": nested}))
        self.assertEqual(result.category, "auth_billing")

    def test_failure_classification_scalar_keys_survive_error_budget_noise(self):
        from scammark.observability import classify_failure

        class Response:
            status_code = 429

            def __init__(self, body):
                self.body = body

        class BodyError(RuntimeError):
            def __init__(self, body):
                super().__init__("provider request failed")
                self.response = Response(body)

        semantic_keys = (
            "code",
            "error_code",
            "status",
            "type",
            "message",
            "detail",
        )
        for semantic_key in semantic_keys:
            error_noise = [
                ["generic"] * 24 for _ in range(24)
            ]
            body = {
                "error": error_noise,
                semantic_key: "insufficient_quota",
            }
            with self.subTest(semantic_key=semantic_key):
                result = classify_failure(BodyError(body))
                self.assertEqual(result.category, "auth_billing")

        nested_error = {
            "error": [["generic"] * 24 for _ in range(24)],
            "code": "insufficient_quota",
        }
        result = classify_failure(BodyError({"error": nested_error}))
        self.assertEqual(result.category, "auth_billing")

    def test_failure_classification_priority_siblings_share_budget_fairly(self):
        from scammark.observability import classify_failure

        class Response:
            status_code = 429

            def __init__(self, body):
                self.body = body

        class BodyError(RuntimeError):
            def __init__(self, body):
                super().__init__("provider request failed")
                self.response = Response(body)

        for noisy_key, scalar_key in (
            ("code", "error_code"),
            ("message", "detail"),
        ):
            body = {
                noisy_key: [["generic"] * 24 for _ in range(24)],
                scalar_key: "insufficient_quota",
            }
            with self.subTest(
                noisy_key=noisy_key,
                scalar_key=scalar_key,
            ):
                result = classify_failure(BodyError(body))
                self.assertEqual(result.category, "auth_billing")

        complex_siblings = {
            "code": [["generic"] * 24 for _ in range(24)],
            "error": {"code": "insufficient_quota"},
        }
        result = classify_failure(BodyError(complex_siblings))
        self.assertEqual(result.category, "auth_billing")

    def test_failure_classification_bounds_large_text_before_normalizing(self):
        import tracemalloc

        from scammark.observability import classify_failure

        class Response:
            status_code = 429

            def __init__(self, body):
                self.body = body

        class BodyError(RuntimeError):
            def __init__(self, body):
                super().__init__("provider request failed")
                self.response = Response(body)

        classify_failure(BodyError("warmup"))
        large_values = ("x" * 2_000_000, b"x" * 2_000_000)
        for body in large_values:
            with self.subTest(body_type=type(body).__name__):
                tracemalloc.start()
                try:
                    result = classify_failure(BodyError(body))
                    _, peak_bytes = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertEqual(result.category, "transient")
                self.assertLess(peak_bytes, 1_000_000)

    def test_failure_classification_returns_frozen_safe_nonempty_messages(self):
        from scammark.observability import (
            FailureClassification,
            classify_failure,
        )

        class UnprintableError(RuntimeError):
            def __str__(self):
                raise RuntimeError("broken exception formatter")

        unprintable = UnprintableError()
        unprintable.__cause__ = unprintable
        fallback = classify_failure(unprintable)
        self.assertTrue(fallback.message)

        unsafe = RuntimeError(
            "HTTP 503: Authorization: Bearer token-value; "
            "prompt=PROMPT_SENTINEL raw_response=RAW_RESPONSE_SENTINEL "
            "https://provider.example/v1/chat/completions?api_key=secret "
            "Traceback (most recent call last): TRACEBACK_SENTINEL\x00"
        )
        result = classify_failure(unsafe)
        self.assertIsInstance(result, FailureClassification)
        self.assertEqual(
            tuple(FailureClassification.__dataclass_fields__),
            ("category", "systemic", "exit_code", "http_status", "message"),
        )
        for sentinel in (
            "token-value",
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "/v1/chat/completions",
            "api_key=secret",
            "TRACEBACK_SENTINEL",
            "\x00",
        ):
            self.assertNotIn(sentinel, result.message)
        with self.assertRaises(AttributeError):
            result.category = "changed"

    def test_failure_messages_never_return_http_bodies_or_multiword_prompts(self):
        from scammark.observability import classify_failure

        http_result = classify_failure(
            RuntimeError(
                "HTTP 500: RAW_PROVIDER_BODY_SENTINEL arbitrary body words"
            )
        )
        self.assertEqual(
            (http_result.category, http_result.http_status),
            ("transient", 500),
        )
        self.assertNotIn("RAW_PROVIDER_BODY_SENTINEL", http_result.message)
        self.assertNotIn("arbitrary body words", http_result.message)

        prompt_result = classify_failure(
            RuntimeError(
                "schema failure prompt=FIRST_PROMPT_WORD "
                "SECOND_PROMPT_WORD THIRD_PROMPT_WORD"
            )
        )
        self.assertEqual(prompt_result.category, "sample_permanent")
        for sentinel in (
            "FIRST_PROMPT_WORD",
            "SECOND_PROMPT_WORD",
            "THIRD_PROMPT_WORD",
        ):
            self.assertNotIn(sentinel, prompt_result.message)

    def test_failure_messages_redact_json_quoted_fields_and_basic_authorization(self):
        from scammark.observability import classify_failure

        result = classify_failure(
            RuntimeError(
                'schema failure {"prompt":"JSON_PROMPT_SECRET words",'
                '"raw_response":"JSON_RESPONSE_SECRET words",'
                '"authorization":"Basic BASIC_AUTH_SECRET"}'
            )
        )
        self.assertEqual(result.category, "sample_permanent")
        for sentinel in (
            "JSON_PROMPT_SECRET",
            "JSON_RESPONSE_SECRET",
            "BASIC_AUTH_SECRET",
        ):
            self.assertNotIn(sentinel, result.message)

    def test_failure_classification_stringifies_each_exception_only_once(self):
        from scammark.observability import classify_failure

        class StatefulError(RuntimeError):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def __str__(self):
                self.calls += 1
                if self.calls == 1:
                    return "invalid api key"
                return "SECOND_CALL_UNLABELED_SECRET"

        error = StatefulError()
        result = classify_failure(error)

        self.assertEqual(result.category, "auth_billing")
        self.assertEqual(error.calls, 1)
        self.assertNotIn("SECOND_CALL_UNLABELED_SECRET", result.message)

    def test_breaker_reconstructs_four_errors_and_opens_on_next_error(self):
        from scammark.observability import FailureCircuitBreaker

        prior = [
            {
                "event": "sample_complete",
                "backend_error": True,
                "sample_id": str(index),
            }
            for index in range(1, 5)
        ]
        breaker = FailureCircuitBreaker.from_events(prior, threshold=5)
        self.assertEqual(breaker.consecutive, 4)
        self.assertTrue(breaker.record(backend_error=True))
        self.assertTrue(breaker.opened)

    def test_breaker_same_id_success_resets_and_errors_can_start_new_window(self):
        from scammark.observability import FailureCircuitBreaker

        breaker = FailureCircuitBreaker.from_events(
            [
                {"event": "sample_complete", "backend_error": True, "sample_id": "same"},
                {"event": "sample_complete", "backend_error": True, "sample_id": "other"},
                {"event": "sample_complete", "backend_error": False, "sample_id": "same"},
                {"event": "sample_complete", "backend_error": True, "sample_id": "same"},
            ],
            threshold=5,
        )

        self.assertEqual((breaker.consecutive, breaker.opened), (1, False))

    def test_breaker_sample_record_deduplicates_and_ack_clears_active_window(self):
        from scammark.observability import FailureCircuitBreaker

        breaker = FailureCircuitBreaker(threshold=2)
        self.assertFalse(
            breaker.record_sample(sample_id="same", backend_error=True)
        )
        self.assertFalse(
            breaker.record_sample(sample_id="same", backend_error=True)
        )
        self.assertTrue(
            breaker.record_sample(sample_id="other", backend_error=True)
        )

        breaker.acknowledge(reason="operator checked breaker")
        self.assertFalse(
            breaker.record_sample(sample_id="same", backend_error=True)
        )
        self.assertEqual((breaker.consecutive, breaker.opened), (1, False))

    def test_open_breaker_requires_explicit_ack_and_records_exact_reset(self):
        from scammark.observability import (
            BreakerAcknowledgementRequired,
            FailureCircuitBreaker,
        )

        prior = [{"event": "breaker_opened", "consecutive": 5}]
        breaker = FailureCircuitBreaker.from_events(prior, threshold=5)
        with self.assertRaises(BreakerAcknowledgementRequired) as caught:
            breaker.require_start_allowed(acknowledged=False)
        self.assertEqual(caught.exception.exit_code, 22)
        breaker.require_start_allowed(acknowledged=True)
        self.assertTrue(breaker.opened)

        event = breaker.acknowledge(
            reason=(
                "operator inspected prompt=PROMPT_SENTINEL at "
                "https://provider.example/v1/chat/completions?key=secret"
            )
        )
        self.assertEqual(
            event,
            {
                "event": "breaker_reset_ack",
                "reason": "operator inspected <redacted-field>",
                "prior_consecutive": 5,
            },
        )
        self.assertEqual((breaker.consecutive, breaker.opened), (0, False))

    def test_breaker_reconstruction_uses_latest_success_reset_and_open_events(self):
        from scammark.observability import FailureCircuitBreaker

        prior = [
            {"event": "sample_complete", "backend_error": True, "sample_id": "1"},
            {"event": "sample_complete", "backend_error": True, "sample_id": "2"},
            {"event": "sample_complete", "backend_error": False, "sample_id": "3"},
            {"event": "sample_complete", "backend_error": True, "sample_id": "4"},
            {"event": "breaker_opened", "consecutive": 2},
            {"event": "sample_complete", "backend_error": False, "sample_id": "5"},
            {
                "event": "breaker_reset_ack",
                "reason": "checked",
                "prior_consecutive": 0,
            },
            {"event": "sample_complete", "backend_error": True, "sample_id": "6"},
            {"event": "sample_complete", "backend_error": True, "sample_id": "7"},
        ]
        breaker = FailureCircuitBreaker.from_events(prior, threshold=5)
        self.assertEqual((breaker.consecutive, breaker.opened), (2, False))

        opened = FailureCircuitBreaker.from_events(
            [
                *prior,
                {"event": "breaker_opened", "consecutive": 5},
                {
                    "event": "sample_complete",
                    "backend_error": False,
                    "sample_id": "8",
                },
            ],
            threshold=5,
        )
        self.assertEqual(opened.consecutive, 0)
        self.assertTrue(opened.opened)

    def test_breaker_ignores_duplicate_and_malformed_sample_events(self):
        from scammark.observability import FailureCircuitBreaker

        valid = {
            "event": "sample_complete",
            "backend_error": True,
            "sample_id": "one",
        }
        prior = [
            None,
            "not-an-event",
            {},
            valid,
            dict(valid),
            {
                "event": "sample_complete",
                "backend_error": "true",
                "sample_id": "two",
            },
            {
                "event": "sample_complete",
                "backend_error": True,
                "sample_id": "",
            },
            {
                "event": "sample_complete",
                "backend_error": True,
                "sample_id": "two",
            },
            {
                "event": "sample_complete",
                "backend_error": True,
                "sample_id": "three",
                "attempt_id": "attempt-a",
                "sequence": 7,
            },
            {
                "event": "sample_complete",
                "backend_error": True,
                "sample_id": "different-corrupt-copy",
                "attempt_id": "attempt-a",
                "sequence": 7,
            },
        ]

        breaker = FailureCircuitBreaker.from_events(prior, threshold=4)

        self.assertEqual((breaker.consecutive, breaker.opened), (3, False))

    def test_breaker_allows_same_sample_again_after_success_or_reset_segment(self):
        from scammark.observability import FailureCircuitBreaker

        reset = {
            "event": "breaker_reset_ack",
            "reason": "operator checked",
            "prior_consecutive": 3,
        }
        cases = [
            (
                "sample_complete",
                [
                    {
                        "event": "sample_complete",
                        "backend_error": True,
                        "sample_id": "retried",
                    },
                    {
                        "event": "sample_complete",
                        "backend_error": False,
                        "sample_id": "success",
                    },
                    {
                        "event": "sample_complete",
                        "backend_error": True,
                        "sample_id": "retried",
                    },
                    {
                        "event": "sample_complete",
                        "backend_error": True,
                        "sample_id": "next",
                    },
                ],
            ),
            (
                "breaker_reset_ack",
                [
                    {
                        "event": "sample_complete",
                        "backend_error": True,
                        "sample_id": "retried",
                    },
                    {
                        "event": "sample_complete",
                        "backend_error": True,
                        "sample_id": "open-2",
                    },
                    {
                        "event": "sample_complete",
                        "backend_error": True,
                        "sample_id": "open-3",
                    },
                    reset,
                    {
                        "event": "sample_complete",
                        "backend_error": True,
                        "sample_id": "retried",
                    },
                    {
                        "event": "sample_complete",
                        "backend_error": True,
                        "sample_id": "next",
                    },
                ],
            ),
        ]
        for boundary, prior in cases:
            with self.subTest(boundary=boundary):
                breaker = FailureCircuitBreaker.from_events(
                    prior, threshold=3
                )
                self.assertEqual(
                    (breaker.consecutive, breaker.opened),
                    (2, False),
                )

    def test_breaker_malformed_open_rows_fail_closed_until_valid_ack(self):
        from scammark.observability import FailureCircuitBreaker

        for consecutive in (None, 0, -1, True, "5"):
            row = {"event": "breaker_opened"}
            if consecutive is not None:
                row["consecutive"] = consecutive
            with self.subTest(consecutive=consecutive):
                breaker = FailureCircuitBreaker.from_events(
                    [row], threshold=5
                )
                self.assertEqual(
                    (breaker.consecutive, breaker.opened),
                    (5, True),
                )
                acknowledged = FailureCircuitBreaker.from_events(
                    [
                        row,
                        {
                            "event": "breaker_reset_ack",
                            "reason": "operator checked",
                            "prior_consecutive": 5,
                        },
                    ],
                    threshold=5,
                )
                self.assertEqual(
                    (acknowledged.consecutive, acknowledged.opened),
                    (0, False),
                )

        valid = FailureCircuitBreaker.from_events(
            [{"event": "breaker_opened", "consecutive": 1}],
            threshold=5,
        )
        self.assertTrue(valid.opened)

    def test_breaker_record_is_atomic_at_threshold_with_two_threads(self):
        from scammark.observability import FailureCircuitBreaker

        ready = threading.Barrier(2)
        breaker = FailureCircuitBreaker(threshold=2)

        class CoordinatedInt(int):
            def __new__(cls, value):
                return super().__new__(cls, value)

            def __add__(self, other):
                lock = getattr(breaker, "_lock", None)
                lock_is_held = bool(
                    lock is not None
                    and getattr(lock, "_is_owned", lambda: False)()
                )
                if not lock_is_held:
                    ready.wait(timeout=1)
                return int(self) + other

        breaker.consecutive = CoordinatedInt(0)
        threads = [
            threading.Thread(
                target=breaker.record,
                kwargs={"backend_error": True},
            )
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual((breaker.consecutive, breaker.opened), (2, True))

    def test_breaker_acknowledge_and_record_have_only_serializable_outcomes(self):
        from scammark.observability import FailureCircuitBreaker

        breaker = FailureCircuitBreaker.from_events(
            [{"event": "breaker_opened", "consecutive": 2}],
            threshold=2,
        )
        record_started = threading.Event()
        allow_record = threading.Event()
        acknowledgement = []

        class PausingInt(int):
            def __new__(cls, value):
                return super().__new__(cls, value)

            def __add__(self, other):
                lock = getattr(breaker, "_lock", None)
                lock_is_held = bool(
                    lock is not None
                    and getattr(lock, "_is_owned", lambda: False)()
                )
                record_started.set()
                if not lock_is_held:
                    allow_record.wait(timeout=1)
                return int(self) + other

        breaker.consecutive = PausingInt(2)

        def record():
            breaker.record(backend_error=True)

        def acknowledge():
            record_started.wait(timeout=1)
            acknowledgement.append(
                breaker.acknowledge(reason="operator checked")
            )
            allow_record.set()

        record_thread = threading.Thread(target=record)
        ack_thread = threading.Thread(target=acknowledge)
        record_thread.start()
        ack_thread.start()
        record_thread.join(timeout=1)
        ack_thread.join(timeout=1)

        self.assertTrue(all(
            not thread.is_alive() for thread in (record_thread, ack_thread)
        ))
        observed = (
            breaker.consecutive,
            breaker.opened,
            acknowledgement[0]["prior_consecutive"],
        )
        self.assertIn(observed, {(0, False, 3), (1, False, 2)})

    def test_breaker_ack_reason_summarizes_provider_body_but_keeps_operator_text(self):
        from scammark.observability import FailureCircuitBreaker

        unsafe = FailureCircuitBreaker.from_events(
            [{"event": "breaker_opened", "consecutive": 5}],
            threshold=5,
        ).acknowledge(
            reason=(
                "HTTP 500: RAW_PROVIDER_BODY_SENTINEL "
                "prompt=PROMPT_SENTINEL"
            )
        )
        ordinary = FailureCircuitBreaker.from_events(
            [{"event": "breaker_opened", "consecutive": 5}],
            threshold=5,
        ).acknowledge(reason="operator inspected sanitized errors")

        self.assertEqual(
            unsafe["reason"],
            "operator acknowledged breaker after HTTP 500 failure",
        )
        self.assertLessEqual(len(unsafe["reason"]), 100)
        self.assertEqual(
            ordinary["reason"], "operator inspected sanitized errors"
        )
        self.assertNotIn("RAW_PROVIDER_BODY_SENTINEL", str(unsafe))
        self.assertNotIn("PROMPT_SENTINEL", str(unsafe))

    def test_breaker_ack_reason_redacts_generic_bodies_and_secret_fields(self):
        from scammark.observability import FailureCircuitBreaker

        reasons = [
            "provider body: PROVIDER_BODY_SECRET",
            "unknown raw body: UNKNOWN_BODY_SECRET",
            "operator checked token=TOKEN_SECRET trailing words",
            "operator checked key=KEY_SECRET trailing words",
            "operator checked password=PASSWORD_SECRET trailing words",
            "operator checked authorization=AUTH_SECRET trailing words",
        ]
        projected = []
        for reason in reasons:
            breaker = FailureCircuitBreaker.from_events(
                [{"event": "breaker_opened", "consecutive": 5}],
                threshold=5,
            )
            projected.append(breaker.acknowledge(reason=reason)["reason"])

        self.assertEqual(
            projected[:2],
            [
                "operator acknowledged breaker after provider failure",
                "operator acknowledged breaker after provider failure",
            ],
        )
        persisted = " ".join(projected)
        for sentinel in (
            "PROVIDER_BODY_SECRET",
            "UNKNOWN_BODY_SECRET",
            "TOKEN_SECRET",
            "KEY_SECRET",
            "PASSWORD_SECRET",
            "AUTH_SECRET",
            "trailing words",
        ):
            self.assertNotIn(sentinel, persisted)
        ordinary = FailureCircuitBreaker.from_events(
            [{"event": "breaker_opened", "consecutive": 5}],
            threshold=5,
        ).acknowledge(reason="operator inspected sanitized errors")
        self.assertEqual(
            ordinary["reason"], "operator inspected sanitized errors"
        )

    def test_breaker_ack_reason_summarizes_http_error_and_generic_secret(self):
        from scammark.observability import FailureCircuitBreaker

        reasons = [
            "HTTPError 500: TOPSECRET_BODY words",
            "provider-error 502: PROVIDER_ERROR_BODY words",
            "status_code=503: STATUS_BODY words",
            "operator checked secret=GENERIC_SECRET trailing words",
        ]
        projected = []
        for reason in reasons:
            breaker = FailureCircuitBreaker.from_events(
                [{"event": "breaker_opened", "consecutive": 5}],
                threshold=5,
            )
            projected.append(breaker.acknowledge(reason=reason)["reason"])

        self.assertEqual(
            projected[:3],
            [
                "operator acknowledged breaker after HTTP 500 failure",
                "operator acknowledged breaker after HTTP 502 failure",
                "operator acknowledged breaker after HTTP 503 failure",
            ],
        )
        persisted = " ".join(projected)
        for sentinel in (
            "TOPSECRET_BODY",
            "PROVIDER_ERROR_BODY",
            "STATUS_BODY",
            "GENERIC_SECRET",
            "trailing words",
        ):
            self.assertNotIn(sentinel, persisted)

    def test_breaker_ack_reason_summarizes_remaining_body_variants(self):
        from scammark.observability import FailureCircuitBreaker

        cases = [
            (
                "HTTP error 504: HTTP_SPACE_ERROR_BODY",
                "operator acknowledged breaker after HTTP 504 failure",
            ),
            (
                "HTTP-error 505: HTTP_HYPHEN_ERROR_BODY",
                "operator acknowledged breaker after HTTP 505 failure",
            ),
            (
                "body: BARE_BODY_SECRET",
                "operator acknowledged breaker after provider failure",
            ),
            (
                "response: BARE_RESPONSE_SECRET",
                "operator acknowledged breaker after provider failure",
            ),
            (
                '{"detail":"JSON_DETAIL_SECRET"}',
                "operator acknowledged breaker after provider failure",
            ),
            (
                '{"body":"JSON_BODY_SECRET"}',
                "operator acknowledged breaker after provider failure",
            ),
            (
                '{"response":"JSON_RESPONSE_SECRET"}',
                "operator acknowledged breaker after provider failure",
            ),
            (
                '[{"detail":"JSON_ARRAY_DETAIL_SECRET"}]',
                "operator acknowledged breaker after provider failure",
            ),
        ]
        projected = []
        for reason, expected in cases:
            with self.subTest(reason=reason):
                breaker = FailureCircuitBreaker.from_events(
                    [{"event": "breaker_opened", "consecutive": 5}],
                    threshold=5,
                )
                actual = breaker.acknowledge(reason=reason)["reason"]
                projected.append(actual)
                self.assertEqual(actual, expected)

        persisted = " ".join(projected)
        for sentinel in (
            "HTTP_SPACE_ERROR_BODY",
            "HTTP_HYPHEN_ERROR_BODY",
            "BARE_BODY_SECRET",
            "BARE_RESPONSE_SECRET",
            "JSON_DETAIL_SECRET",
            "JSON_BODY_SECRET",
            "JSON_RESPONSE_SECRET",
            "JSON_ARRAY_DETAIL_SECRET",
        ):
            self.assertNotIn(sentinel, persisted)

    def test_breaker_ack_reason_redacts_remaining_credential_labels(self):
        from scammark.observability import FailureCircuitBreaker

        cases = [
            (
                "operator checked credential=CREDENTIAL_SECRET trailing words",
                "CREDENTIAL_SECRET",
            ),
            (
                "operator checked credentials=CREDENTIALS_SECRET trailing words",
                "CREDENTIALS_SECRET",
            ),
            (
                "operator checked refresh_token=REFRESH_SECRET trailing words",
                "REFRESH_SECRET",
            ),
            (
                "operator checked session_token=SESSION_SECRET trailing words",
                "SESSION_SECRET",
            ),
            (
                "operator checked private_key=PRIVATE_KEY_SECRET trailing words",
                "PRIVATE_KEY_SECRET",
            ),
        ]
        projected = []
        for reason, sentinel in cases:
            with self.subTest(reason=reason):
                breaker = FailureCircuitBreaker.from_events(
                    [{"event": "breaker_opened", "consecutive": 5}],
                    threshold=5,
                )
                actual = breaker.acknowledge(reason=reason)["reason"]
                projected.append(actual)
                self.assertEqual(actual, "operator checked <redacted-field>")
                self.assertNotIn(sentinel, actual)

        persisted = " ".join(projected)
        for sentinel in (
            "CREDENTIAL_SECRET",
            "CREDENTIALS_SECRET",
            "REFRESH_SECRET",
            "SESSION_SECRET",
            "PRIVATE_KEY_SECRET",
            "trailing words",
        ):
            self.assertNotIn(sentinel, persisted)

    def test_breaker_ack_reason_redacts_exact_auth_labels_only(self):
        from scammark.observability import FailureCircuitBreaker

        cases = [
            "operator checked auth=AUTH_SECRET trailing words",
            "operator checked auth_token=AUTH_TOKEN_SECRET trailing words",
            "operator checked auth-token=AUTH_HYPHEN_SECRET trailing words",
            "operator checked auth token=AUTH_SPACE_SECRET trailing words",
        ]
        for reason in cases:
            with self.subTest(reason=reason):
                breaker = FailureCircuitBreaker.from_events(
                    [{"event": "breaker_opened", "consecutive": 5}],
                    threshold=5,
                )
                actual = breaker.acknowledge(reason=reason)["reason"]
                self.assertEqual(actual, "operator checked <redacted-field>")

        for ordinary in (
            "operator reviewed authentication=expected",
            "operator checked author=alice",
        ):
            with self.subTest(ordinary=ordinary):
                breaker = FailureCircuitBreaker.from_events(
                    [{"event": "breaker_opened", "consecutive": 5}],
                    threshold=5,
                )
                actual = breaker.acknowledge(reason=ordinary)["reason"]
                self.assertEqual(actual, ordinary)

    def test_breaker_ignores_adversarial_string_subclass_fields(self):
        from scammark.observability import FailureCircuitBreaker

        class ExplosiveString(str):
            def __eq__(self, other):
                raise RuntimeError("comparison escaped")

            def __hash__(self):
                raise RuntimeError("hash escaped")

            def strip(self, *args, **kwargs):
                raise RuntimeError("strip escaped")

        events = [
            {"event": ExplosiveString("sample_complete")},
            {
                "event": "breaker_reset_ack",
                "reason": ExplosiveString("checked"),
                "prior_consecutive": 1,
            },
            {
                "event": "sample_complete",
                "backend_error": True,
                "sample_id": ExplosiveString("sample"),
            },
        ]

        breaker = FailureCircuitBreaker.from_events(events, threshold=5)

        self.assertEqual((breaker.consecutive, breaker.opened), (0, False))

    def test_breaker_replay_rejects_control_only_ack_reason(self):
        from scammark.observability import FailureCircuitBreaker

        breaker = FailureCircuitBreaker.from_events(
            [
                {"event": "breaker_opened", "consecutive": 5},
                {
                    "event": "breaker_reset_ack",
                    "reason": "\x00\n",
                    "prior_consecutive": 5,
                },
            ],
            threshold=5,
        )

        self.assertEqual((breaker.consecutive, breaker.opened), (5, True))

    def test_breaker_replay_accepts_only_open_state_with_exact_prior_count(self):
        from scammark.observability import FailureCircuitBreaker

        errors = [
            {
                "event": "sample_complete",
                "backend_error": True,
                "sample_id": str(index),
            }
            for index in range(1, 6)
        ]
        ack_zero = {
            "event": "breaker_reset_ack",
            "reason": "operator checked",
            "prior_consecutive": 0,
        }
        ack_five = {**ack_zero, "prior_consecutive": 5}

        cases = [
            ([*errors, ack_zero], (5, True)),
            (
                [
                    {"event": "breaker_opened", "consecutive": 5},
                    ack_zero,
                ],
                (5, True),
            ),
            ([*errors[:4], {**ack_zero, "prior_consecutive": 4}], (4, False)),
            ([*errors, ack_five], (0, False)),
            (
                [
                    {"event": "breaker_opened", "consecutive": 5},
                    ack_five,
                ],
                (0, False),
            ),
        ]
        for events, expected in cases:
            with self.subTest(events=events):
                breaker = FailureCircuitBreaker.from_events(
                    events, threshold=5
                )
                self.assertEqual(
                    (breaker.consecutive, breaker.opened), expected
                )

        with self.assertRaisesRegex(ValueError, "closed"):
            FailureCircuitBreaker(threshold=5).acknowledge(
                reason="operator checked"
            )

    def test_breaker_opens_at_exact_threshold_and_success_resets_only_count(self):
        from scammark.observability import FailureCircuitBreaker

        breaker = FailureCircuitBreaker(threshold=2)
        self.assertFalse(breaker.record(backend_error=True))
        self.assertEqual(breaker.consecutive, 1)
        self.assertTrue(breaker.record(backend_error=True))
        self.assertTrue(breaker.opened)
        self.assertFalse(breaker.record(backend_error=False))
        self.assertEqual(breaker.consecutive, 0)
        self.assertTrue(breaker.opened)

    def test_breaker_validates_threshold_flags_and_ack_reason(self):
        from scammark.observability import FailureCircuitBreaker

        for invalid in (0, -1, True, 1.5, "5"):
            with self.subTest(threshold=invalid), self.assertRaises(ValueError):
                FailureCircuitBreaker(threshold=invalid)
            with self.subTest(from_events=invalid), self.assertRaises(ValueError):
                FailureCircuitBreaker.from_events([], threshold=invalid)

        breaker = FailureCircuitBreaker(threshold=1)
        for invalid in (None, 1, "true"):
            with self.subTest(backend_error=invalid), self.assertRaises(ValueError):
                breaker.record(backend_error=invalid)
        with self.assertRaises(ValueError):
            breaker.require_start_allowed(acknowledged="yes")
        with self.assertRaises(ValueError):
            breaker.acknowledge(reason="  \x00  ")

    def test_stop_controller_first_signal_waits_for_boundary_second_forces_code(self):
        from scammark.observability import (
            GracefulStopRequested,
            StopController,
        )

        forced = []
        controller = StopController(force_exit=forced.append)
        controller.handle_signal(signal.SIGINT, None)
        with self.assertRaises(GracefulStopRequested) as caught:
            controller.raise_if_requested("before_request")
        self.assertEqual(caught.exception.signum, signal.SIGINT)
        self.assertEqual(caught.exception.exit_code, 130)
        self.assertEqual(forced, [])
        controller.handle_signal(signal.SIGINT, None)
        self.assertEqual(forced, [130])

    def test_stop_controller_locks_first_signal_and_forces_once_outside_lock(self):
        from scammark.observability import StopController

        forced = []
        lock_was_available = []
        controller = None

        def force_exit(exit_code):
            acquired = controller._lock.acquire(blocking=False)
            lock_was_available.append(acquired)
            if acquired:
                controller._lock.release()
            forced.append(exit_code)

        controller = StopController(force_exit=force_exit)
        controller.handle_signal(signal.SIGTERM, None)
        controller.handle_signal(signal.SIGINT, None)
        controller.handle_signal(signal.SIGINT, None)

        self.assertEqual(forced, [143])
        self.assertEqual(lock_was_available, [True])

    def test_stop_controller_signal_handler_is_reentrant_with_controller_lock(self):
        from scammark.observability import StopController

        controller = StopController(force_exit=lambda code: None)
        completed = threading.Event()

        def interrupt_while_locked():
            with controller._lock:
                controller.handle_signal(signal.SIGINT, None)
            completed.set()

        thread = threading.Thread(target=interrupt_while_locked, daemon=True)
        thread.start()
        thread.join(timeout=0.2)
        self.assertTrue(completed.is_set())

    def test_stop_controller_remembers_forced_state_when_force_callable_raises(self):
        from scammark.observability import StopController

        class ForcedExit(Exception):
            pass

        forced = []

        def force_exit(exit_code):
            forced.append(exit_code)
            raise ForcedExit

        controller = StopController(force_exit=force_exit)
        controller.handle_signal(signal.SIGINT, None)
        with self.assertRaises(ForcedExit):
            controller.handle_signal(signal.SIGTERM, None)
        controller.handle_signal(signal.SIGTERM, None)
        self.assertEqual(forced, [130])

    def test_graceful_stop_sanitizes_and_requires_nonempty_boundary(self):
        from scammark.observability import (
            GracefulStopRequested,
            StopController,
        )

        controller = StopController(force_exit=lambda code: None)
        self.assertIsNone(controller.raise_if_requested("before_request"))
        controller.handle_signal(signal.SIGTERM, None)
        with self.assertRaises(GracefulStopRequested) as caught:
            controller.raise_if_requested(
                "after_stage prompt=PROMPT_SENTINEL "
                "https://provider.example/v1/chat/completions?key=secret\x00"
            )
        self.assertEqual(caught.exception.exit_code, 143)
        self.assertEqual(
            caught.exception.boundary,
            "after_stage <redacted-field>",
        )
        self.assertNotIn("PROMPT_SENTINEL", str(caught.exception))
        self.assertNotIn("/v1/chat/completions", str(caught.exception))
        with self.assertRaises(ValueError):
            GracefulStopRequested(signal.SIGINT, " \x00 ")

    def test_control_reasons_drop_multiword_sensitive_values(self):
        from scammark.observability import (
            FailureCircuitBreaker,
            GracefulStopRequested,
            StopController,
        )

        breaker = FailureCircuitBreaker.from_events(
            [{"event": "breaker_opened", "consecutive": 5}],
            threshold=5,
        )
        reset = breaker.acknowledge(
            reason=(
                "checked raw_response=FIRST_RESPONSE_WORD "
                "SECOND_RESPONSE_WORD THIRD_RESPONSE_WORD"
            )
        )
        controller = StopController(force_exit=lambda code: None)
        controller.handle_signal(signal.SIGINT, None)
        with self.assertRaises(GracefulStopRequested) as caught:
            controller.raise_if_requested(
                "before_request prompt=FIRST_PROMPT_WORD "
                "SECOND_PROMPT_WORD THIRD_PROMPT_WORD"
            )
        persisted_text = reset["reason"] + " " + caught.exception.boundary
        for sentinel in (
            "FIRST_RESPONSE_WORD",
            "SECOND_RESPONSE_WORD",
            "THIRD_RESPONSE_WORD",
            "FIRST_PROMPT_WORD",
            "SECOND_PROMPT_WORD",
            "THIRD_PROMPT_WORD",
        ):
            self.assertNotIn(sentinel, persisted_text)

    def test_stop_controller_install_close_preserve_handlers_and_are_idempotent(self):
        from scammark import observability

        original_int = object()
        original_term = lambda signum, frame: None
        handlers = {
            signal.SIGINT: original_int,
            signal.SIGTERM: original_term,
        }
        calls = []

        def fake_getsignal(signum):
            return handlers[signum]

        def fake_signal(signum, handler):
            previous = handlers[signum]
            handlers[signum] = handler
            calls.append((signum, handler))
            return previous

        controller = observability.StopController(force_exit=lambda code: None)
        with mock.patch.object(
            observability.signal, "getsignal", side_effect=fake_getsignal
        ), mock.patch.object(
            observability.signal, "signal", side_effect=fake_signal
        ):
            controller.install()
            controller.install()
            self.assertIs(handlers[signal.SIGINT].__self__, controller)
            self.assertIs(handlers[signal.SIGTERM].__self__, controller)
            controller.close()
            controller.close()

        self.assertIs(handlers[signal.SIGINT], original_int)
        self.assertIs(handlers[signal.SIGTERM], original_term)
        self.assertEqual([item[0] for item in calls], [
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGTERM,
            signal.SIGINT,
        ])

    def test_stop_controller_rolls_back_partial_install_and_retries_close_failure(self):
        from scammark import observability

        original_int = object()
        original_term = object()
        handlers = {
            signal.SIGINT: original_int,
            signal.SIGTERM: original_term,
        }
        fail_install_term = [True]

        def fake_getsignal(signum):
            return handlers[signum]

        controller = observability.StopController(force_exit=lambda code: None)

        def fake_signal(signum, handler):
            if (
                signum == signal.SIGTERM
                and getattr(handler, "__self__", None) is controller
                and fail_install_term[0]
            ):
                raise RuntimeError("install failed")
            previous = handlers[signum]
            handlers[signum] = handler
            return previous

        with mock.patch.object(
            observability.signal, "getsignal", side_effect=fake_getsignal
        ), mock.patch.object(
            observability.signal, "signal", side_effect=fake_signal
        ):
            with self.assertRaisesRegex(RuntimeError, "install failed"):
                controller.install()
            self.assertIs(handlers[signal.SIGINT], original_int)
            self.assertIs(handlers[signal.SIGTERM], original_term)

            fail_install_term[0] = False
            controller.install()
            fail_restore_int = [True]

            def flaky_restore(signum, handler):
                if (
                    signum == signal.SIGINT
                    and handler is original_int
                    and fail_restore_int[0]
                ):
                    raise RuntimeError("restore failed")
                return fake_signal(signum, handler)

            with mock.patch.object(
                observability.signal, "signal", side_effect=flaky_restore
            ):
                with self.assertRaisesRegex(RuntimeError, "restore failed"):
                    controller.close()
                self.assertIs(handlers[signal.SIGTERM], original_term)
                fail_restore_int[0] = False
                controller.close()
        self.assertIs(handlers[signal.SIGINT], original_int)

    def test_stop_controller_partial_rollback_never_reports_installed(self):
        from scammark import observability

        original_int = object()
        original_term = object()
        handlers = {
            signal.SIGINT: original_int,
            signal.SIGTERM: original_term,
        }
        fail_install_term = [True]
        fail_restore_int = [True]

        def fake_getsignal(signum):
            return handlers[signum]

        controller = observability.StopController(force_exit=lambda code: None)

        def fake_signal(signum, handler):
            installing = getattr(handler, "__self__", None) is controller
            if (
                signum == signal.SIGTERM
                and installing
                and fail_install_term[0]
            ):
                raise RuntimeError("install failed")
            if (
                signum == signal.SIGINT
                and handler is original_int
                and fail_restore_int[0]
            ):
                raise RuntimeError("rollback failed")
            previous = handlers[signum]
            handlers[signum] = handler
            return previous

        with mock.patch.object(
            observability.signal, "getsignal", side_effect=fake_getsignal
        ), mock.patch.object(
            observability.signal, "signal", side_effect=fake_signal
        ):
            with self.assertRaisesRegex(RuntimeError, "install failed"):
                controller.install()
            self.assertIs(handlers[signal.SIGINT].__self__, controller)
            self.assertIs(handlers[signal.SIGTERM], original_term)

            with self.assertRaisesRegex(RuntimeError, "partial"):
                controller.install()
            with self.assertRaisesRegex(RuntimeError, "rollback failed"):
                controller.close()

            fail_restore_int[0] = False
            controller.close()
            self.assertIs(handlers[signal.SIGINT], original_int)
            self.assertIs(handlers[signal.SIGTERM], original_term)

            fail_install_term[0] = False
            controller.install()
            self.assertIs(handlers[signal.SIGINT].__self__, controller)
            self.assertIs(handlers[signal.SIGTERM].__self__, controller)
            controller.close()

        self.assertIs(handlers[signal.SIGINT], original_int)
        self.assertIs(handlers[signal.SIGTERM], original_term)

    def test_stop_controller_install_and_close_reject_non_main_thread(self):
        from scammark import observability

        signal_calls = []
        errors = []
        controller = observability.StopController(force_exit=lambda code: None)

        def invoke(method):
            try:
                method()
            except Exception as error:
                errors.append(error)

        with mock.patch.object(
            observability.signal,
            "signal",
            side_effect=lambda *args: signal_calls.append(args),
        ):
            install_thread = threading.Thread(target=invoke, args=(controller.install,))
            close_thread = threading.Thread(target=invoke, args=(controller.close,))
            install_thread.start()
            install_thread.join()
            close_thread.start()
            close_thread.join()

        self.assertEqual(len(errors), 2)
        self.assertTrue(all(isinstance(error, RuntimeError) for error in errors))
        self.assertEqual(signal_calls, [])

    def test_task_five_interfaces_and_exit_codes_are_public(self):
        from scammark import observability

        expected = {
            "ApiBudgetStop",
            "AuthBillingStop",
            "BreakerAcknowledgementRequired",
            "CircuitBreakerStop",
            "ContractDriftStop",
            "FailureCircuitBreaker",
            "FailureClassification",
            "GracefulStopRequested",
            "RunControlError",
            "StopController",
            "classify_failure",
        }
        self.assertLessEqual(expected, set(observability.__all__))
        self.assertEqual(
            [
                observability.RunControlError.exit_code,
                observability.AuthBillingStop.exit_code,
                observability.ApiBudgetStop.exit_code,
                observability.CircuitBreakerStop.exit_code,
                observability.ContractDriftStop.exit_code,
            ],
            [1, 20, 21, 22, 23],
        )
        self.assertTrue(
            issubclass(
                observability.BreakerAcknowledgementRequired,
                observability.CircuitBreakerStop,
            )
        )

    def test_task_four_interfaces_are_public_exports(self):
        from scammark import observability

        self.assertLessEqual(
            {
                "HeartbeatWorker",
                "ProgressSnapshot",
                "ProgressTracker",
                "StageObservation",
                "SummaryScheduler",
                "project_remaining_stage_counts",
            },
            set(observability.__all__),
        )

    def test_eta_uses_stage_mix_excludes_reuse_and_warms_after_five_full_samples(self):
        from scammark.observability import ObservabilitySettings, ProgressTracker

        tracker = ProgressTracker(
            ObservabilitySettings(eta_warmup_samples=5),
            total_samples=100,
            api_call_limit=50000,
            pricing={
                "text": {
                    "currency": "USD",
                    "input_per_million": 1,
                    "output_per_million": 2,
                },
                "vision": {
                    "currency": "CNY",
                    "input_per_million": 3,
                    "output_per_million": 4,
                },
            },
            text_model="gpt-4o-mini",
            vision_model="qwen3-vl-flash",
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={},
            elapsed_seconds=2,
            source="smoke",
        )
        tracker.record_stage(
            "binary",
            model="gpt-4o-mini",
            usage={},
            elapsed_seconds=3,
            source="smoke",
        )
        tracker.record_stage(
            "type",
            model="gpt-4o-mini",
            usage={},
            elapsed_seconds=5,
            source="smoke",
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={},
            elapsed_seconds=999,
            reused=True,
        )
        remaining = {"url": 10, "binary": 10, "type": 4}

        self.assertIsNone(
            tracker.snapshot(remaining_stage_counts=remaining).eta_seconds
        )
        self.assertEqual(
            tracker.snapshot(
                remaining_stage_counts=remaining
            ).preliminary_eta_seconds,
            70,
        )
        for index in range(1, 6):
            tracker.complete_sample(str(index), index=index, backend_error=False)
        self.assertEqual(
            tracker.snapshot(remaining_stage_counts=remaining).eta_seconds,
            70,
        )

    def test_remaining_stage_projection_uses_truth_scams_and_observed_benign_fp_rate(self):
        from scammark.observability import project_remaining_stage_counts

        manifest = [
            *(
                {"sample_id": f"s{index}", "binary": "scam"}
                for index in range(3)
            ),
            *(
                {"sample_id": f"b{index}", "binary": "benign"}
                for index in range(7)
            ),
        ]
        counts = project_remaining_stage_counts(
            manifest,
            complete_stages=frozenset({("s0", "url")}),
            binary_outputs={},
            observed_benign_false_positive_rate=0.20,
            smoke_benign_false_positive_rate=0.10,
        )
        self.assertEqual(counts["url"], 9)
        self.assertEqual(counts["binary"], 10)
        self.assertEqual(counts["type"], 5)
        self.assertEqual(counts["grounding"], 5)

    def test_projection_accepts_zero_observed_rate_without_double_counting(self):
        from scammark.observability import project_remaining_stage_counts

        manifest = [
            {"sample_id": "truth-scam", "binary": "scam"},
            {"sample_id": "observed-fp", "binary": "benign"},
            {"sample_id": "observed-tn", "binary": "benign"},
            {"sample_id": "unresolved-1", "binary": "benign"},
            {"sample_id": "unresolved-2", "binary": "benign"},
        ]
        counts = project_remaining_stage_counts(
            manifest,
            complete_stages=frozenset(
                {
                    ("truth-scam", "type"),
                    ("observed-fp", "grounding"),
                }
            ),
            binary_outputs={
                "observed-fp": "scam",
                "observed-tn": "benign",
            },
            observed_benign_false_positive_rate=0.0,
            smoke_benign_false_positive_rate=1.0,
        )

        self.assertEqual(counts["type"], 1)
        self.assertEqual(counts["grounding"], 1)

    def test_remaining_projection_rejects_invalid_manifest_contract(self):
        from scammark.observability import project_remaining_stage_counts

        invalid_manifests = (
            [{"sample_id": "missing"}],
            [{"sample_id": "   ", "binary": "benign"}],
            [{"sample_id": "invalid", "binary": "unknown"}],
            [
                {"sample_id": "duplicate", "binary": "benign"},
                {"sample_id": "duplicate", "binary": "scam"},
            ],
        )
        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                project_remaining_stage_counts(
                    manifest,
                    complete_stages=frozenset(),
                    binary_outputs={},
                    observed_benign_false_positive_rate=None,
                    smoke_benign_false_positive_rate=0.1,
                )

    def test_remaining_projection_rejects_invalid_binary_outputs_contract(self):
        from scammark.observability import project_remaining_stage_counts

        manifest = [{"sample_id": "known", "binary": "benign"}]
        for binary_outputs in (
            {"known": "unknown"},
            {"unknown-id": "scam"},
        ):
            with (
                self.subTest(binary_outputs=binary_outputs),
                self.assertRaises(ValueError),
            ):
                project_remaining_stage_counts(
                    manifest,
                    complete_stages=frozenset(),
                    binary_outputs=binary_outputs,
                    observed_benign_false_positive_rate=None,
                    smoke_benign_false_positive_rate=0.1,
                )

    def test_remaining_cost_keeps_currencies_separate_and_adds_replay_allowance(self):
        from scammark.observability import ObservabilitySettings, ProgressTracker

        tracker = ProgressTracker(
            ObservabilitySettings(cost_full_update_samples=25),
            total_samples=10,
            api_call_limit=50000,
            pricing={
                "text": {
                    "currency": "USD",
                    "input_per_million": 1,
                    "output_per_million": 2,
                },
                "vision": {
                    "currency": "CNY",
                    "input_per_million": 3,
                    "output_per_million": 4,
                },
            },
            text_model="gpt-4o-mini",
            vision_model="qwen3-vl-flash",
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 1000, "completion_tokens": 100},
            elapsed_seconds=2,
            source="smoke",
        )
        tracker.record_stage(
            "visual",
            model="qwen3-vl-flash",
            usage={"prompt_tokens": 2000, "completion_tokens": 200},
            elapsed_seconds=4,
            source="smoke",
        )
        estimate = tracker.estimate_remaining_cost(
            {"url": 9, "visual": 9},
            outstanding_by_stage={"visual": 1},
            replay_required_by_stage={"url": 1},
        )
        self.assertAlmostEqual(estimate["USD"], 0.0120)
        self.assertAlmostEqual(estimate["CNY"], 0.0680)

    def test_full_cost_mean_replaces_smoke_only_on_twenty_fifth_full_sample(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(
            ObservabilitySettings(cost_full_update_samples=25)
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 0},
            elapsed_seconds=2,
            source="smoke",
        )
        for index in range(1, 25):
            tracker.record_stage(
                "url",
                model="gpt-4o-mini",
                usage={"prompt_tokens": 2_000_000, "completion_tokens": 0},
                elapsed_seconds=2,
                source="full",
            )
            tracker.complete_sample(
                str(index), index=index, backend_error=False
            )
        self.assertEqual(
            tracker.estimate_remaining_cost(
                {"url": 1},
                outstanding_by_stage={},
                replay_required_by_stage={},
            )["USD"],
            1.0,
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 2_000_000, "completion_tokens": 0},
            elapsed_seconds=2,
            source="full",
        )
        tracker.complete_sample("25", index=25, backend_error=False)
        self.assertEqual(
            tracker.estimate_remaining_cost(
                {"url": 1},
                outstanding_by_stage={},
                replay_required_by_stage={},
            )["USD"],
            2.0,
        )

    def test_single_sided_usage_is_not_a_complete_cost_observation(self):
        from scammark.observability import ObservabilitySettings

        for usage in (
            {"input_tokens": 1_000_000},
            {"output_tokens": 1_000_000},
        ):
            tracker = make_tracker(ObservabilitySettings())
            tracker.record_stage(
                "url",
                model="gpt-4o-mini",
                usage=usage,
                elapsed_seconds=1,
                source="smoke",
            )
            with (
                self.subTest(usage=usage),
                self.assertRaisesRegex(ValueError, "cost observation"),
            ):
                tracker.estimate_remaining_cost(
                    {"url": 1},
                    outstanding_by_stage={},
                    replay_required_by_stage={},
                )

    def test_cost_projection_rejects_each_unestimated_positive_allowance(self):
        from scammark.observability import ObservabilitySettings

        for remaining, outstanding, replay in (
            ({"url": 1}, {}, {}),
            ({}, {"url": 1}, {}),
            ({}, {}, {"url": 1}),
        ):
            tracker = make_tracker(ObservabilitySettings())
            with (
                self.subTest(
                    remaining=remaining,
                    outstanding=outstanding,
                    replay=replay,
                ),
                self.assertRaisesRegex(ValueError, "cost observation"),
            ):
                tracker.estimate_remaining_cost(
                    remaining,
                    outstanding_by_stage=outstanding,
                    replay_required_by_stage=replay,
                )

    def test_full_only_cost_is_unavailable_until_twenty_fifth_success(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(
            ObservabilitySettings(cost_full_update_samples=25)
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
            elapsed_seconds=1,
            source="full",
        )
        for completed in (0, 24):
            for index in range(tracker.snapshot().completed_samples + 1, completed + 1):
                tracker.complete_sample(
                    str(index), index=index, backend_error=False
                )
            with self.subTest(completed=completed), self.assertRaisesRegex(
                ValueError, "cost observation"
            ):
                tracker.estimate_remaining_cost(
                    {"url": 1},
                    outstanding_by_stage={},
                    replay_required_by_stage={},
                )

        tracker.complete_sample("25", index=25, backend_error=False)
        self.assertEqual(
            tracker.estimate_remaining_cost(
                {"url": 1},
                outstanding_by_stage={},
                replay_required_by_stage={},
            )["USD"],
            1.0,
        )

    def test_summary_scheduler_triggers_on_count_or_time_once(self):
        from scammark.observability import (
            ObservabilitySettings,
            SummaryScheduler,
        )

        scheduler = SummaryScheduler(
            ObservabilitySettings(
                summary_sample_interval=25,
                summary_seconds=300,
            ),
            started_at=0,
        )
        self.assertFalse(scheduler.should_emit(completed_samples=24, now=299))
        self.assertTrue(scheduler.should_emit(completed_samples=25, now=299))
        self.assertFalse(scheduler.should_emit(completed_samples=25, now=300))
        self.assertTrue(scheduler.should_emit(completed_samples=26, now=599))

        exact = SummaryScheduler(
            ObservabilitySettings(
                summary_sample_interval=25,
                summary_seconds=300,
            ),
            started_at=0,
        )
        self.assertTrue(exact.should_emit(completed_samples=25, now=300))
        self.assertFalse(exact.should_emit(completed_samples=25, now=300))

    def test_summary_scheduler_rejects_regression_after_non_emitting_observation(self):
        from scammark.observability import (
            ObservabilitySettings,
            SummaryScheduler,
        )

        scheduler = SummaryScheduler(ObservabilitySettings(), started_at=0)
        self.assertFalse(
            scheduler.should_emit(completed_samples=10, now=100)
        )
        with self.assertRaisesRegex(ValueError, "completed_samples"):
            scheduler.should_emit(completed_samples=9, now=101)
        with self.assertRaisesRegex(ValueError, "now"):
            scheduler.should_emit(completed_samples=10, now=99)

    def test_summary_scheduler_concurrent_boundary_emits_once(self):
        from scammark.observability import (
            ObservabilitySettings,
            SummaryScheduler,
        )

        scheduler = SummaryScheduler(ObservabilitySettings(), started_at=0)
        barrier = threading.Barrier(3)
        results = []

        def probe():
            barrier.wait()
            results.append(
                scheduler.should_emit(completed_samples=25, now=300)
            )

        threads = [threading.Thread(target=probe) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(sorted(results), [False, True])

    def test_heartbeat_emits_at_sixty_seconds_and_not_more_often(self):
        from scammark.observability import (
            HeartbeatWorker,
            ObservabilitySettings,
        )

        events = []
        snapshot = lambda: {
            "sample_id": "884",
            "stage": "grounding",
            "last_sample_completed_at": 100.0,
            "api_calls": 866,
            "outstanding": 1,
        }
        worker = HeartbeatWorker(
            ObservabilitySettings(heartbeat_stall_seconds=60),
            snapshot=snapshot,
            emit=lambda name, **fields: events.append((name, fields)),
            monotonic=lambda: 159.0,
        )
        worker.tick()
        self.assertEqual(events, [])
        worker.monotonic = lambda: 160.0
        worker.tick()
        worker.tick()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1]["health"], "waiting_api")

    def test_heartbeat_uses_mapping_or_bound_tracker_api_call_limit(self):
        from scammark.observability import (
            HeartbeatWorker,
            ObservabilitySettings,
            ProgressTracker,
        )

        tracker = ProgressTracker(
            ObservabilitySettings(),
            total_samples=10,
            api_call_limit=50000,
            pricing={
                "text": {
                    "currency": "USD",
                    "input_per_million": 1,
                    "output_per_million": 1,
                },
                "vision": {
                    "currency": "CNY",
                    "input_per_million": 1,
                    "output_per_million": 1,
                },
            },
            text_model="text-model",
            vision_model="vision-model",
            monotonic=lambda: 0,
        )
        events = []
        for snapshot, configured_limit, expected_limit in (
            (
                lambda: {
                    "last_sample_completed_at": 0,
                    "api_call_limit": 123,
                },
                999,
                123,
            ),
            (tracker.snapshot, 999, 50000),
            (lambda: {"last_sample_completed_at": 0}, 456, 456),
        ):
            HeartbeatWorker(
                ObservabilitySettings(),
                snapshot=snapshot,
                emit=lambda name, **fields: events.append(fields),
                monotonic=lambda: 60,
                api_call_limit=configured_limit,
            ).tick()
            self.assertEqual(events[-1]["api_call_limit"], expected_limit)

    def test_heartbeat_defaults_unbound_snapshot_to_global_api_call_limit(self):
        from scammark.observability import HeartbeatWorker, ObservabilitySettings

        events = []
        worker = HeartbeatWorker(
            ObservabilitySettings(),
            snapshot=lambda: {"last_sample_completed_at": 0},
            emit=lambda name, **fields: events.append(fields),
            monotonic=lambda: 60,
        )
        worker.tick()
        self.assertEqual(events[0]["api_call_limit"], 50000)

    def test_heartbeat_polls_every_tick_uses_sample_completion_and_calls_outside_lock(self):
        from scammark.observability import (
            HeartbeatWorker,
            ObservabilitySettings,
            ProgressTracker,
        )

        clock = [0.0]
        tracker = ProgressTracker(
            ObservabilitySettings(),
            total_samples=2,
            api_call_limit=10,
            pricing={
                "text": {
                    "currency": "USD",
                    "input_per_million": 1,
                    "output_per_million": 1,
                },
                "vision": {
                    "currency": "CNY",
                    "input_per_million": 1,
                    "output_per_million": 1,
                },
            },
            text_model="text-model",
            vision_model="vision-model",
            monotonic=lambda: clock[0],
        )
        clock[0] = 1
        tracker.begin_sample("sample-1", index=1)
        clock[0] = 2
        tracker.begin_stage("url")
        clock[0] = 3
        tracker.reserve_call("url")
        clock[0] = 59
        tracker.record_stage(
            "url",
            model="text-model",
            usage={},
            elapsed_seconds=56,
        )

        polls = []
        events = []
        worker = None

        def outside_lock(target):
            acquired = worker._lock.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired:
                worker._lock.release()
            target.append(clock[0])

        worker = HeartbeatWorker(
            ObservabilitySettings(heartbeat_stall_seconds=60),
            snapshot=tracker.snapshot,
            emit=lambda name, **fields: outside_lock(events),
            monotonic=lambda: clock[0],
            on_poll=lambda now: outside_lock(polls),
        )
        clock[0] = 60
        worker.tick()
        worker.tick()

        snapshot = tracker.snapshot()
        self.assertEqual(snapshot.last_progress_at, 59)
        self.assertEqual(snapshot.last_sample_completed_at, 0)
        self.assertEqual(polls, [60, 60])
        self.assertEqual(events, [60])

    def test_heartbeat_survives_repeated_on_poll_failures(self):
        from scammark.observability import HeartbeatWorker, ObservabilitySettings

        clock = [60.0]
        polls = []
        events = []

        def broken_summary(now):
            polls.append(now)
            raise RuntimeError("summary failed")

        worker = HeartbeatWorker(
            ObservabilitySettings(),
            snapshot=lambda: {
                "last_sample_completed_at": 0,
                "outstanding": 1,
            },
            emit=lambda name, **fields: events.append(fields),
            monotonic=lambda: clock[0],
            on_poll=broken_summary,
        )
        worker.tick()
        clock[0] = 120
        worker.tick()

        self.assertEqual(polls, [60, 120])
        self.assertEqual(len(events), 2)

    def test_heartbeat_start_and_close_are_idempotent_and_daemonized(self):
        from scammark import observability

        created = []

        class FinishedThread:
            def __init__(self, *, target, name, daemon):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.join_count = 0
                created.append(self)

            def start(self):
                return None

            def is_alive(self):
                return False

            def join(self):
                self.join_count += 1

        worker = observability.HeartbeatWorker(
            observability.ObservabilitySettings(),
            snapshot=lambda: {
                "last_sample_completed_at": 0,
                "outstanding": 0,
            },
            emit=lambda *args, **kwargs: None,
        )
        with mock.patch.object(
            observability.threading,
            "Thread",
            FinishedThread,
        ):
            worker.start()
            worker.start()
            worker.close()
            worker.close()

        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].daemon)
        self.assertEqual(created[0].join_count, 1)

    def test_progress_tracker_records_lifecycle_calls_retries_and_completion(self):
        from scammark.observability import ObservabilitySettings, ProgressTracker

        times = iter((100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0))
        tracker = ProgressTracker(
            ObservabilitySettings(),
            total_samples=10,
            api_call_limit=20,
            pricing={
                "text": {
                    "currency": "USD",
                    "input_per_million": 1,
                    "output_per_million": 1,
                },
                "vision": {
                    "currency": "CNY",
                    "input_per_million": 1,
                    "output_per_million": 1,
                },
            },
            text_model="text-model",
            vision_model="vision-model",
            monotonic=lambda: next(times),
        )

        tracker.begin_sample("sample-1", index=1)
        tracker.begin_stage("url")
        tracker.reserve_call("url")
        tracker.record_retry()
        tracker.finish_call("url", transport_error=False)
        in_progress = tracker.snapshot()
        self.assertEqual(in_progress.current_sample_id, "sample-1")
        self.assertEqual(in_progress.current_stage, "url")
        self.assertEqual(in_progress.api_calls, 1)
        self.assertEqual(in_progress.outstanding_calls, 0)
        self.assertEqual(in_progress.retries, 1)
        self.assertEqual(in_progress.last_progress_at, 105.0)
        self.assertEqual(in_progress.last_sample_completed_at, 100.0)

        tracker.complete_sample("sample-1", index=1, backend_error=True)
        completed = tracker.snapshot()
        self.assertEqual(completed.completed_samples, 1)
        self.assertIsNone(completed.current_sample_id)
        self.assertIsNone(completed.current_stage)
        self.assertEqual(completed.backend_errors, 1)
        self.assertEqual(completed.last_sample_completed_at, 106.0)

    def test_completed_sample_id_and_index_cannot_begin_again(self):
        from scammark.observability import ObservabilitySettings

        for sample_id, index in (("done", 2), ("new", 1)):
            with self.subTest(sample_id=sample_id, index=index):
                tracker = make_tracker(ObservabilitySettings())
                tracker.complete_sample("done", index=1, backend_error=False)
                with self.assertRaisesRegex(
                    RuntimeError, "already completed"
                ):
                    tracker.begin_sample(sample_id, index=index)
                snapshot = tracker.snapshot()
                self.assertEqual(snapshot.completed_samples, 1)
                self.assertIsNone(snapshot.current_sample_id)

    def test_complete_sample_rejects_duplicate_id_or_index_atomically(self):
        from scammark.observability import ObservabilitySettings

        for sample_id, index in (("done", 2), ("new", 1)):
            with self.subTest(sample_id=sample_id, index=index):
                tracker = make_tracker(ObservabilitySettings())
                tracker.complete_sample("done", index=1, backend_error=False)
                before = tracker.snapshot()
                with self.assertRaisesRegex(
                    RuntimeError, "already completed"
                ):
                    tracker.complete_sample(
                        sample_id, index=index, backend_error=False
                    )
                after = tracker.snapshot()
                self.assertEqual(after.completed_samples, 1)
                self.assertEqual(
                    after.last_sample_completed_at,
                    before.last_sample_completed_at,
                )

    def test_duplicate_completion_does_not_advance_full_sample_thresholds(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(
            ObservabilitySettings(
                eta_warmup_samples=5,
                cost_full_update_samples=5,
            )
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
            elapsed_seconds=1,
            source="smoke",
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"input_tokens": 2_000_000, "output_tokens": 0},
            elapsed_seconds=2,
            source="full",
        )
        tracker.complete_sample("one", index=1, backend_error=False)
        for attempt in range(4):
            with self.subTest(attempt=attempt), self.assertRaisesRegex(
                RuntimeError, "already completed"
            ):
                tracker.complete_sample("one", index=1, backend_error=False)

        snapshot = tracker.snapshot(remaining_stage_counts={"url": 1})
        self.assertEqual(snapshot.completed_samples, 1)
        self.assertIsNone(snapshot.eta_seconds)
        self.assertEqual(
            tracker.estimate_remaining_cost(
                {"url": 1},
                outstanding_by_stage={},
                replay_required_by_stage={},
            )["USD"],
            1.0,
        )

    def test_zero_second_smoke_mean_is_not_replaced_by_full_fallback(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(ObservabilitySettings(eta_warmup_samples=1))
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={},
            elapsed_seconds=0,
            source="smoke",
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={},
            elapsed_seconds=10,
            source="full",
        )
        tracker.complete_sample("1", index=1, backend_error=False)

        snapshot = tracker.snapshot(remaining_stage_counts={"url": 1})
        self.assertEqual(snapshot.preliminary_eta_seconds, 0)
        self.assertEqual(snapshot.eta_seconds, 10)

    def test_full_only_eta_is_hidden_until_warmup_without_preliminary_fallback(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(ObservabilitySettings(eta_warmup_samples=5))
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={},
            elapsed_seconds=7,
            source="full",
        )
        cold = tracker.snapshot(remaining_stage_counts={"url": 1})
        self.assertIsNone(cold.preliminary_eta_seconds)
        self.assertIsNone(cold.eta_seconds)

        for index in range(1, 5):
            tracker.complete_sample(
                str(index), index=index, backend_error=False
            )
        warming = tracker.snapshot(remaining_stage_counts={"url": 1})
        self.assertIsNone(warming.preliminary_eta_seconds)
        self.assertIsNone(warming.eta_seconds)

        tracker.complete_sample("5", index=5, backend_error=False)
        warm = tracker.snapshot(remaining_stage_counts={"url": 1})
        self.assertIsNone(warm.preliminary_eta_seconds)
        self.assertEqual(warm.eta_seconds, 7)

    def test_successful_stage_from_backend_error_sample_enters_full_means(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(
            ObservabilitySettings(
                eta_warmup_samples=1,
                cost_full_update_samples=1,
            )
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
            elapsed_seconds=1,
            source="smoke",
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"input_tokens": 2_000_000, "output_tokens": 0},
            elapsed_seconds=9,
            source="full",
        )
        tracker.complete_sample("1", index=1, backend_error=True)
        tracker.complete_sample("2", index=2, backend_error=False)

        snapshot = tracker.snapshot(remaining_stage_counts={"url": 1})
        self.assertEqual(snapshot.eta_seconds, 9)
        self.assertEqual(
            tracker.estimate_remaining_cost(
                {"url": 1},
                outstanding_by_stage={},
                replay_required_by_stage={},
            )["USD"],
            2.0,
        )

    def test_missing_full_usage_keeps_smoke_cost_fallback(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(
            ObservabilitySettings(cost_full_update_samples=1)
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 0},
            elapsed_seconds=1,
            source="smoke",
        )
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={},
            elapsed_seconds=2,
            source="full",
        )
        tracker.complete_sample("1", index=1, backend_error=False)

        self.assertEqual(
            tracker.estimate_remaining_cost(
                {"url": 1},
                outstanding_by_stage={},
                replay_required_by_stage={},
            )["USD"],
            1.0,
        )

    def test_missing_usage_still_advances_bounded_cost_history_window(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(ObservabilitySettings(eta_history_size=2))
        for input_tokens in (1_000_000, None, 3_000_000):
            usage = (
                {}
                if input_tokens is None
                else {"input_tokens": input_tokens, "output_tokens": 0}
            )
            tracker.record_stage(
                "url",
                model="gpt-4o-mini",
                usage=usage,
                elapsed_seconds=1,
                source="smoke",
            )

        self.assertEqual(
            tracker.estimate_remaining_cost(
                {"url": 1},
                outstanding_by_stage={},
                replay_required_by_stage={},
            )["USD"],
            3.0,
        )

    def test_invalid_stage_observations_are_rejected_without_polluting_means(self):
        from scammark.observability import ObservabilitySettings

        tracker = make_tracker(ObservabilitySettings())
        tracker.record_stage(
            "url",
            model="gpt-4o-mini",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
            elapsed_seconds=2,
            source="smoke",
        )
        invalid = (
            {"usage": {}, "elapsed_seconds": -1},
            {"usage": {}, "elapsed_seconds": float("nan")},
            {
                "usage": {"prompt_tokens": -1},
                "elapsed_seconds": 3,
            },
            {
                "usage": {"input_tokens": float("inf")},
                "elapsed_seconds": 3,
            },
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                tracker.record_stage(
                    "url",
                    model="gpt-4o-mini",
                    source="smoke",
                    **values,
                )

        self.assertEqual(
            tracker.snapshot(
                remaining_stage_counts={"url": 1}
            ).preliminary_eta_seconds,
            2,
        )
        self.assertEqual(
            tracker.estimate_remaining_cost(
                {"url": 1},
                outstanding_by_stage={},
                replay_required_by_stage={},
            )["USD"],
            1.0,
        )

    def test_progress_tracker_rejects_unreportable_pricing_currency(self):
        from scammark.observability import ObservabilitySettings, ProgressTracker

        with self.assertRaisesRegex(ValueError, "currency"):
            ProgressTracker(
                ObservabilitySettings(),
                total_samples=1,
                api_call_limit=10,
                pricing={
                    "text": {
                        "currency": "EUR",
                        "input_per_million": 1,
                        "output_per_million": 1,
                    },
                    "vision": {
                        "currency": "CNY",
                        "input_per_million": 1,
                        "output_per_million": 1,
                    },
                },
                text_model="text-model",
                vision_model="vision-model",
            )

    def test_settings_redaction_duration_and_human_rendering_helpers(self):
        from scammark.observability import (
            ObservabilitySettings,
            format_duration,
            render_human_event,
            sanitize_payload,
            sanitize_text,
        )

        settings = ObservabilitySettings.from_mapping({"summary_sample_interval": 7})
        self.assertEqual(settings.summary_sample_interval, 7)
        with self.assertRaisesRegex(ValueError, "Unknown observability settings"):
            ObservabilitySettings.from_mapping({"unknown": True})

        secret_text = (
            "Authorization: Bearer token-value; sk-12345678 "
            "api_key=key-value data:image/png;base64,AAAA "
            "https://workspace.cn-beijing.maas.aliyuncs.com/path?token=secret"
        )
        redacted = sanitize_text(secret_text)
        for sentinel in (
            "token-value",
            "sk-12345678",
            "key-value",
            "AAAA",
            "workspace.cn-beijing",
            "token=secret",
        ):
            self.assertNotIn(sentinel, redacted)
        self.assertEqual(
            sanitize_payload(
                {
                    "Prompt": "never persist",
                    "nested": [{"api-key": "never persist"}],
                }
            ),
            {
                "Prompt": "<redacted-field>",
                "nested": [{"api-key": "<redacted-field>"}],
            },
        )
        self.assertEqual(format_duration(-1), "00:00:00")
        self.assertEqual(format_duration(3661), "01:01:01")
        self.assertIsNone(render_human_event({"event": "stage_started"}))
        summary = render_human_event(
            {
                "event": "summary",
                "completed_samples": 25,
                "sample_total": 100,
                "rate_per_minute": 2,
                "eta_seconds": 300,
                "remaining_cost_est": {
                    "USD": 1.84,
                    "CNY": 19.76,
                    "EUR": 7,
                    "accrued": 6,
                },
                "cost_calculation": {
                    "accrued_cost": 999,
                    "per_sample_cost": 888,
                    "final_cost": 777,
                },
            }
        )
        self.assertIn("USD 1.84", summary)
        self.assertIn("CNY 19.76", summary)
        for forbidden in (
            "EUR",
            "accrued",
            "per_sample",
            "final",
            "999",
            "888",
            "777",
        ):
            self.assertNotIn(forbidden, summary)

    def test_event_contract_redacts_denied_fields_and_endpoint_details(self):
        from scammark.observability import ObservabilitySettings, RunEventLogger

        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_events.jsonl"
            logger = RunEventLogger(
                path,
                ObservabilitySettings(),
                stream=stream,
                utc_now=lambda: datetime(2026, 7, 12, 12, 15, 30, tzinfo=timezone.utc),
                monotonic=lambda: 100.0,
                attempt_id="attempt-test",
                experiment_version="full-v2",
                run_mode="full",
            )
            logger.emit(
                "sample_complete",
                critical=True,
                sample_index=123,
                sample_total=5658,
                sample_id="883",
                result="scam",
                api_calls=861,
                api_call_limit=50000,
                eta_seconds=170400,
                health="ok",
                prompt="PROMPT_SENTINEL_MUST_NOT_PERSIST",
                raw_response="RAW_RESPONSE_SENTINEL_MUST_NOT_PERSIST",
                authorization="Bearer sk-live-secret",
                endpoint="https://workspace.cn-beijing.maas.aliyuncs.com/path?token=secret",
                image="data:image/png;base64,AAAA",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        persisted = json.dumps(payload, ensure_ascii=False)
        for sentinel in (
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "sk-live-secret",
            "token=secret",
            "workspace.cn-beijing",
            "AAAA",
        ):
            self.assertNotIn(sentinel, persisted)
        self.assertEqual(payload["schema"], "scammark-run-event-v1")
        self.assertEqual(payload["event"], "sample_complete")
        self.assertEqual(payload["experiment_version"], "full-v2")
        self.assertEqual(payload["run_mode"], "full")
        self.assertEqual(payload["attempt_id"], "attempt-test")
        self.assertEqual(payload["sequence"], 1)
        self.assertEqual(payload["timestamp_local"], "2026-07-12T20:15:30+08:00")
        self.assertIn("123/5658", stream.getvalue())
        self.assertNotIn("cost", stream.getvalue().lower())

    def test_logger_appends_attempts_and_thread_rows_do_not_interleave(self):
        from scammark.observability import ObservabilitySettings, RunEventLogger

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            first = RunEventLogger(
                path,
                ObservabilitySettings(),
                stream=io.StringIO(),
                attempt_id="a",
            )
            first.emit("heartbeat", health="waiting_api", wait_seconds=60)
            second = RunEventLogger(
                path,
                ObservabilitySettings(),
                stream=io.StringIO(),
                attempt_id="b",
            )
            threads = [
                threading.Thread(
                    target=second.emit,
                    args=("heartbeat",),
                    kwargs={"health": "waiting_api", "wait_seconds": index},
                )
                for index in range(20)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 21)
        self.assertEqual(rows[0]["attempt_id"], "a")
        self.assertEqual({row["attempt_id"] for row in rows[1:]}, {"b"})
        self.assertEqual(
            sorted(row["sequence"] for row in rows[1:]),
            list(range(1, 21)),
        )

    def test_critical_events_fsync_and_safe_sink_never_changes_control_flow(self):
        from scammark.observability import (
            ObservabilitySettings,
            RunEventLogger,
            SafeEventSink,
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch("os.fsync") as fsync:
            logger = RunEventLogger(
                Path(tmp) / "events.jsonl",
                ObservabilitySettings(),
                stream=io.StringIO(),
            )
            logger.emit("heartbeat", health="waiting_api", wait_seconds=60)
            self.assertEqual(fsync.call_count, 0)
            logger.emit("run_stopped", reason="graceful_pause", exit_code=130)
            self.assertEqual(fsync.call_count, 1)

        sink = SafeEventSink(
            type(
                "Broken",
                (),
                {
                    "emit": lambda *args, **kwargs: (
                        _ for _ in ()
                    ).throw(OSError("disk full"))
                },
            )()
        )
        self.assertIsNone(sink.emit("sample_started", sample_id="1"))
        self.assertTrue(sink.degraded)

    def test_logger_rejects_unknown_contract_values_and_drops_unknown_fields(self):
        from scammark.observability import ObservabilitySettings, RunEventLogger

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            logger = RunEventLogger(
                path,
                ObservabilitySettings(),
                stream=io.StringIO(),
                attempt_id="contract-test",
            )
            with self.assertRaisesRegex(ValueError, "Unknown event"):
                logger.emit("unknown_event")
            with self.assertRaisesRegex(ValueError, "Unknown health"):
                logger.emit("heartbeat", health="unknown", wait_seconds=1)
            logger.emit(
                "heartbeat",
                health="ok",
                wait_seconds=1,
                endpoint="https://workspace.cn-beijing.maas.aliyuncs.com/?token=secret",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("endpoint", payload)
        self.assertEqual(payload["sequence"], 1)

    def test_safe_sink_warns_once_and_sanitizes_the_failure(self):
        from scammark.observability import SafeEventSink

        fallback = io.StringIO()
        broken = type(
            "Broken",
            (),
            {
                "emit": lambda *args, **kwargs: (_ for _ in ()).throw(
                    OSError("Bearer sk-12345678")
                )
            },
        )()
        sink = SafeEventSink(broken, fallback_stream=fallback)

        sink.emit("sample_started", sample_id="1")
        sink.emit("sample_started", sample_id="2")

        warning = fallback.getvalue()
        self.assertEqual(warning.count("observability degraded"), 1)
        self.assertNotIn("sk-12345678", warning)

    def test_safe_sink_swallows_an_unprintable_logger_exception(self):
        from scammark.observability import SafeEventSink

        class UnprintableError(Exception):
            def __str__(self):
                raise RuntimeError("broken exception formatter")

        broken = type(
            "Broken",
            (),
            {
                "emit": lambda *args, **kwargs: (_ for _ in ()).throw(
                    UnprintableError()
                )
            },
        )()
        fallback = io.StringIO()
        sink = SafeEventSink(broken, fallback_stream=fallback)

        self.assertIsNone(sink.emit("sample_started", sample_id="1"))
        self.assertTrue(sink.degraded)
        self.assertIn("<unprintable>", fallback.getvalue())

    def test_external_append_and_shell_cli_use_allowlisted_typed_fields(self):
        from scammark.observability import append_external_event

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            append_external_event(
                path,
                "shell_started",
                attempt_id="direct-shell-attempt",
                fields={
                    "pid": 10,
                    "starttime": 20,
                    "runner_attempt_id": "direct-runner",
                    "prompt": "PROMPT_SENTINEL",
                },
            )
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(Path.cwd() / "src") + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scammark.observability",
                    "append",
                    "--path",
                    str(path),
                    "--event",
                    "shell_exited",
                    "--attempt-id",
                    "runner-a",
                    "--field",
                    "pid=123",
                    "--field",
                    "exit_code=0",
                    "--field",
                    "forced=false",
                ],
                cwd=Path.cwd(),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(rows), 2)
        self.assertNotIn("prompt", rows[0])
        self.assertEqual(rows[1]["pid"], 123)
        self.assertEqual(rows[1]["exit_code"], 0)
        self.assertIs(rows[1]["forced"], False)
        self.assertEqual(rows[1]["runner_attempt_id"], "runner-a")
        self.assertRegex(
            rows[1]["attempt_id"],
            r"^runner-a-shell-\d+-[0-9a-f]{32}$",
        )
        self.assertEqual(rows[1]["sequence"], 1)

    def test_shell_cli_nonce_prevents_same_pid_attempt_collisions(self):
        from scammark import observability

        nonce_values = ["a" * 32, "b" * 32]
        generated = [
            type("GeneratedUuid", (), {"hex": value})()
            for value in nonce_values
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            arguments = [
                "append",
                "--path",
                str(path),
                "--event",
                "shell_started",
                "--attempt-id",
                "runner-a",
            ]
            with mock.patch.object(
                observability.os, "getpid", return_value=4242
            ), mock.patch.object(
                observability.uuid, "uuid4", side_effect=generated
            ):
                self.assertEqual(observability.main(arguments), 0)
                self.assertEqual(observability.main(arguments), 0)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [row["attempt_id"] for row in rows],
            [
                f"runner-a-shell-4242-{nonce_values[0]}",
                f"runner-a-shell-4242-{nonce_values[1]}",
            ],
        )
        self.assertEqual([row["sequence"] for row in rows], [1, 1])

    def test_shell_cli_preserves_unique_suffix_for_long_runner_id(self):
        from scammark import observability

        runner_attempt_id = "r" * 240
        nonce_values = ["c" * 32, "d" * 32]
        generated = [
            type("GeneratedUuid", (), {"hex": value})()
            for value in nonce_values
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            arguments = [
                "append",
                "--path",
                str(path),
                "--event",
                "shell_started",
                "--attempt-id",
                runner_attempt_id,
            ]
            with mock.patch.object(
                observability.os, "getpid", return_value=4242
            ), mock.patch.object(
                observability.uuid, "uuid4", side_effect=generated
            ):
                self.assertEqual(observability.main(arguments), 0)
                self.assertEqual(observability.main(arguments), 0)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        attempt_ids = [row["attempt_id"] for row in rows]
        self.assertNotEqual(attempt_ids[0], attempt_ids[1])
        self.assertTrue(all(len(value) <= 200 for value in attempt_ids))
        self.assertTrue(all("-shell-4242-" in value for value in attempt_ids))
        self.assertTrue(attempt_ids[0].endswith(nonce_values[0]))
        self.assertTrue(attempt_ids[1].endswith(nonce_values[1]))
        self.assertEqual([row["sequence"] for row in rows], [1, 1])


if __name__ == "__main__":
    unittest.main()
