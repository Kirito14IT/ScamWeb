import hashlib
import io
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.scammark_full_fixture import FullRunFixture

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ScamMarkExperimentTests(unittest.TestCase):
    def test_task10_parser_accepts_explicit_breaker_acknowledgement(self):
        from scammark import run_experiment

        with patch.object(
            sys,
            "argv",
            [
                "run_experiment",
                "--config",
                "configs/qwen3_dashscope_server_full_v2.toml",
                "--full",
                "--resume",
                "--ack-breaker",
            ],
        ):
            args = run_experiment.parse_args()

        self.assertTrue(args.ack_breaker)

    def test_task10_main_maps_control_exceptions_to_stable_exit_codes(self):
        import argparse

        from scammark import run_experiment
        from scammark.observability import (
            ApiBudgetStop,
            AuthBillingStop,
            BreakerAcknowledgementRequired,
            CircuitBreakerStop,
            ContractDriftStop,
            GracefulStopRequested,
        )

        args = argparse.Namespace(
            config="configs/qwen3_dashscope_server_full_v2.toml",
            pilot=None,
            full=True,
            resume=True,
            validate_only=False,
            ack_breaker=False,
        )
        cases = (
            (AuthBillingStop("billing"), 20),
            (ApiBudgetStop("cap"), 21),
            (CircuitBreakerStop("breaker"), 22),
            (BreakerAcknowledgementRequired("ack"), 22),
            (ContractDriftStop("contract"), 23),
            (GracefulStopRequested(signal.SIGINT, "before_request"), 130),
            (GracefulStopRequested(signal.SIGTERM, "after_stage"), 143),
        )
        for error, expected in cases:
            with self.subTest(error=type(error).__name__), patch.object(
                run_experiment, "parse_args", return_value=args
            ), patch.object(
                run_experiment, "run", side_effect=error
            ), patch.object(
                sys, "stderr", new_callable=io.StringIO
            ) as stderr:
                self.assertEqual(run_experiment.main(), expected)
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_task10_main_preserves_success_gate_validate_and_unexpected_exits(self):
        import argparse

        from scammark import run_experiment

        base = dict(
            config="fixture.toml",
            pilot=None,
            full=True,
            resume=False,
            ack_breaker=False,
        )
        results = (
            (
                argparse.Namespace(**base, validate_only=False),
                {"output_dir": "fixture", "gates": {"passed": True}},
                0,
            ),
            (
                argparse.Namespace(**base, validate_only=False),
                {"output_dir": "fixture", "gates": {"passed": False}},
                2,
            ),
            (
                argparse.Namespace(**base, validate_only=True),
                {"output_dir": "fixture", "preflight": {"status": "validated"}},
                0,
            ),
        )
        for args, result, expected in results:
            with self.subTest(expected=expected), patch.object(
                run_experiment, "parse_args", return_value=args
            ), patch.object(run_experiment, "run", return_value=result), patch.object(
                sys, "stdout", new_callable=io.StringIO
            ):
                self.assertEqual(run_experiment.main(), expected)

        secret = (
            "Bearer sk-SENTINEL-12345678 RAW_PROVIDER_BODY "
            "https://workspace-private.invalid/path?token=secret"
        )
        args = argparse.Namespace(**base, validate_only=False)
        with patch.object(
            run_experiment, "parse_args", return_value=args
        ), patch.object(
            run_experiment, "run", side_effect=RuntimeError(secret)
        ), patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(run_experiment.main(), 1)
        rendered = stderr.getvalue()
        self.assertIn("RuntimeError", rendered)
        self.assertLessEqual(len(rendered), 700)
        self.assertNotIn("Traceback", rendered)
        for sentinel in (
            "sk-SENTINEL-12345678",
            "RAW_PROVIDER_BODY",
            "workspace-private",
            "token=secret",
        ):
            self.assertNotIn(sentinel, rendered)

    def test_task10_systemic_failures_stop_before_next_request_without_partial_row(self):
        systemic_cases = (
            (401, "invalid api key", 20),
            (402, "insufficient balance", 20),
            (403, "account is disabled", 20),
            (429, "quota exhausted for account", 20),
        )
        for status, message, expected_code in systemic_cases:
            with self.subTest(status=status), FullRunFixture(
                sample_count=3
            ) as fixture:
                fixture.respond(
                    sample_id="2",
                    outcome={
                        "kind": "http_error",
                        "status": status,
                        "message": message,
                    },
                )
                fixture.create_v1_sentinel_that_raises_on_open()

                code = fixture.run_main(resume=False)
                events = fixture.load_events()

                self.assertEqual(code, expected_code)
                self.assertEqual(fixture.requested_sample_ids(), ["1", "2"])
                self.assertEqual(
                    fixture.calls_for("1"),
                    ["url", "text", "visual", "cross", "binary"],
                )
                self.assertEqual(fixture.calls_for("2"), ["url"])
                self.assertEqual(
                    [row["sample_id"] for row in fixture.load_predictions()],
                    ["1"],
                )
                self.assertEqual(
                    sum(row["event"] == "run_stopped" for row in events), 1
                )
                self.assertEqual(events[-1]["event"], "run_stopped")
                self.assertEqual(fixture.socket_attempt_count, 0)
                self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_wrapped_api_cap_stops_without_partial_row_or_later_request(self):
        from scammark.experiment_state import ApiCallLimitError, ExperimentState

        original_reserve = ExperimentState.reserve_api_call

        def cap_second_sample(state, context, *, max_calls):
            if str(context.get("sample_id") or "") == "2":
                raise ApiCallLimitError("Physical API call cap reached (fixture)")
            return original_reserve(state, context, max_calls=max_calls)

        with FullRunFixture(sample_count=3) as fixture, patch.object(
            ExperimentState, "reserve_api_call", new=cap_second_sample
        ):
            fixture.create_v1_sentinel_that_raises_on_open()
            code = fixture.run_main(resume=False)
            events = fixture.load_events()

            self.assertEqual(code, 21)
            self.assertEqual(fixture.requested_sample_ids(), ["1"])
            self.assertEqual(
                [row["sample_id"] for row in fixture.load_predictions()], ["1"]
            )
            self.assertEqual(
                sum(row["event"] == "run_stopped" for row in events), 1
            )
            self.assertEqual(events[-1]["event"], "run_stopped")
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_preexisting_exhausted_api_budget_flushes_without_request(self):
        from scammark.experiment_state import ExperimentState

        with FullRunFixture(sample_count=1) as fixture, patch.object(
            ExperimentState, "total_api_calls", return_value=50_000
        ):
            fixture.create_v1_sentinel_that_raises_on_open()
            code = fixture.run_main(resume=True)
            events = fixture.load_events()

            self.assertEqual(code, 21)
            self.assertEqual(fixture.request_count, 0)
            self.assertEqual(
                [row["event"] for row in events][-2:],
                ["prediction_flushed", "run_stopped"],
            )
            self.assertEqual(events[-1]["category"], "api_budget")
            self.assertEqual(events[-1]["exit_code"], 21)
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_sample_local_failures_persist_partial_and_continue(self):
        local_cases = (
            {"kind": "http_error", "status": 403, "message": "content policy refusal"},
            {"kind": "http_error", "status": 403, "message": "moderation refusal"},
            {"kind": "http_error", "status": 400, "message": "invalid sample input"},
            {"kind": "schema_error", "message": "invalid output schema"},
        )
        for outcome in local_cases:
            with self.subTest(outcome=outcome), FullRunFixture(
                sample_count=2
            ) as fixture:
                fixture.respond(sample_id="1", outcome=outcome)
                fixture.create_v1_sentinel_that_raises_on_open()

                result = fixture.run(resume=False)
                records = fixture.load_predictions()

                self.assertTrue(result["gates"]["passed"])
                self.assertEqual(fixture.requested_sample_ids(), ["1", "2"])
                self.assertEqual([row["sample_id"] for row in records], ["1", "2"])
                self.assertTrue(records[0]["prediction"]["backend_error"])
                self.assertFalse(records[1]["prediction"].get("backend_error", False))
                self.assertEqual(fixture.socket_attempt_count, 0)
                self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_four_prior_errors_plus_fifth_flushes_before_breaker_stop(self):
        with FullRunFixture(sample_count=6) as fixture:
            for sample_id in ("1", "2", "3", "4", "5"):
                fixture.respond(
                    sample_id=sample_id,
                    outcome={"kind": "schema_error", "message": "invalid output"},
                )
            fixture.request_stop_before_sample("5")
            fixture.create_v1_sentinel_that_raises_on_open()

            self.assertEqual(fixture.run_main(resume=False), 130)
            prior_event_count = len(fixture.load_events())
            code = fixture.run_main(resume=True)
            events = fixture.load_events()
            names = [
                row["event"] for row in events[prior_event_count:]
            ]

            self.assertEqual(code, 22)
            self.assertEqual(fixture.calls_for("6"), [])
            self.assertEqual(
                [row["sample_id"] for row in fixture.load_predictions()],
                ["1", "2", "3", "4", "5"],
            )
            self.assertEqual(names[-3:], [
                "prediction_flushed",
                "breaker_opened",
                "run_stopped",
            ])
            self.assertEqual(names.count("breaker_opened"), 1)
            self.assertEqual(names.count("run_stopped"), 1)
            self.assertIn("breaker_opened", fixture._critical_events)
            self.assertIn("run_stopped", fixture._critical_events)
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_prior_open_breaker_requires_ack_and_ack_precedes_requests(self):
        with FullRunFixture(sample_count=2) as fixture:
            fixture.seed_breaker_opened()
            fixture.create_v1_sentinel_that_raises_on_open()

            self.assertEqual(fixture.run_main(resume=True, ack_breaker=False), 22)
            self.assertEqual(fixture.request_count, 0)
            unack_events = fixture.load_events()
            self.assertEqual(unack_events[-1]["event"], "run_stopped")

            self.assertEqual(fixture.run_main(resume=True, ack_breaker=True), 0)
            events = fixture.load_events()
            names = [row["event"] for row in events]
            ack_index = max(
                index
                for index, name in enumerate(names)
                if name == "breaker_reset_ack"
            )
            first_request_index = next(
                index
                for index, name in enumerate(names)
                if index > ack_index and name == "api_call_reserved"
            )

            self.assertLess(ack_index, first_request_index)
            self.assertIn("breaker_reset_ack", fixture._critical_events)
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_breaker_deduplicates_prior_rows_and_success_resets_count(self):
        with FullRunFixture(sample_count=6) as fixture:
            fixture.seed_sample_events(backend_errors=4)
            fixture.seed_predictions(count=4)
            assert fixture.output_root is not None
            event_path = fixture.output_root / "full" / "run_events.jsonl"
            prior_lines = event_path.read_text(encoding="utf-8").splitlines()
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(prior_lines[-1] + "\n")
            fixture.respond(
                sample_id="6",
                outcome={"kind": "schema_error", "message": "invalid output"},
            )
            fixture.create_v1_sentinel_that_raises_on_open()

            code = fixture.run_main(resume=True)
            events = fixture.load_events()
            sample_six = [
                row
                for row in events
                if row["event"] == "sample_complete"
                and row.get("sample_id") == "6"
            ][-1]

            self.assertEqual(code, 0)
            self.assertEqual(fixture.requested_sample_ids(), ["5", "6"])
            self.assertEqual(sample_six["breaker_consecutive"], 1)
            self.assertNotIn("breaker_opened", [row["event"] for row in events])
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_real_error_checkpoints_deduplicate_and_crash_success_resets_once(self):
        with FullRunFixture(sample_count=6) as fixture:
            for sample_id in ("1", "2", "3", "4"):
                fixture.respond(
                    sample_id=sample_id,
                    outcome={"kind": "schema_error", "message": "invalid output"},
                )
            fixture.request_stop_before_sample("5")
            self.assertEqual(fixture.run_main(resume=False), 130)

            fixture.request_stop_before_sample("5")
            self.assertEqual(fixture.run_main(resume=True), 130)
            events = fixture.load_events()
            self.assertFalse(
                any(row["event"] == "breaker_opened" for row in events)
            )
            latest_four = [
                row
                for row in events
                if row["event"] == "sample_complete"
                and row.get("sample_id") == "4"
            ][-1]
            self.assertEqual(latest_four["breaker_consecutive"], 4)

            fixture.respond(
                sample_id="1",
                outcome={"kind": "success", "binary": "benign"},
            )
            fixture._drop_sample_complete_ids.add("1")
            fixture.request_stop_before_sample("2")
            self.assertEqual(fixture.run_main(resume=True), 130)

            fixture.request_stop_before_sample("4")
            self.assertEqual(fixture.run_main(resume=True), 130)
            events = fixture.load_events()
            latest = {
                sample_id: [
                    row
                    for row in events
                    if row["event"] == "sample_complete"
                    and row.get("sample_id") == sample_id
                ][-1]
                for sample_id in ("1", "2", "3")
            }
            self.assertEqual(latest["1"]["breaker_consecutive"], 0)
            self.assertEqual(latest["2"]["breaker_consecutive"], 1)
            self.assertEqual(latest["3"]["breaker_consecutive"], 2)

            fixture.request_stop_before_sample("2")
            self.assertEqual(fixture.run_main(resume=True), 130)
            replayed = [
                row
                for row in fixture.load_events()
                if row["event"] == "sample_complete"
                and row.get("sample_id") == "1"
            ][-1]
            self.assertEqual(replayed["breaker_consecutive"], 2)
            self.assertNotIn("backend_error", replayed)
            self.assertEqual(fixture.socket_attempt_count, 0)

    def test_task10_invalid_breaker_ack_does_not_recount_real_error_checkpoint(self):
        with FullRunFixture(sample_count=5) as fixture:
            for sample_id in ("1", "2", "3", "4"):
                fixture.respond(
                    sample_id=sample_id,
                    outcome={"kind": "schema_error", "message": "invalid output"},
                )
            fixture.request_stop_before_sample("5")
            self.assertEqual(fixture.run_main(resume=False), 130)

            assert fixture.output_root is not None
            event_path = fixture.output_root / "full" / "run_events.jsonl"
            invalid_ack = fixture.load_events()[-1]
            invalid_ack.update(
                {
                    "event": "breaker_reset_ack",
                    "attempt_id": "fixture-invalid-ack",
                    "sequence": 1,
                    "reason": "",
                    "prior_consecutive": 4,
                }
            )
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(invalid_ack, ensure_ascii=False) + "\n")

            fixture.request_stop_before_sample("5")
            self.assertEqual(fixture.run_main(resume=True), 130)
            events = fixture.load_events()
            self.assertFalse(
                any(row["event"] == "breaker_opened" for row in events)
            )
            replayed_four = [
                row
                for row in events
                if row["event"] == "sample_complete"
                and row.get("sample_id") == "4"
            ][-1]
            self.assertEqual(replayed_four["breaker_consecutive"], 4)
            self.assertEqual(fixture.socket_attempt_count, 0)

    def test_task10_unprintable_transient_becomes_safe_partial_and_continues(self):
        hostile = (
            "UNPRINTABLE_TRANSIENT_SENTINEL PROMPT_SENTINEL "
            "RAW_RESPONSE_SENTINEL"
        )
        with FullRunFixture(sample_count=2) as fixture:
            assert fixture.config_path is not None
            fixture.config_path.write_text(
                fixture.config_path.read_text(encoding="utf-8").replace(
                    "backoff_seconds = [2, 4, 8]",
                    "backoff_seconds = [0, 0, 0]",
                    1,
                ),
                encoding="utf-8",
            )
            fixture.respond(
                sample_id="1",
                outcome={"kind": "unprintable_transient", "message": hostile},
            )

            result = fixture.run(resume=False)
            records = fixture.load_predictions()
            persisted = fixture.all_durable_and_human_text()

            self.assertTrue(result["gates"]["passed"])
            self.assertEqual([row["sample_id"] for row in records], ["1", "2"])
            self.assertTrue(records[0]["prediction"]["backend_error"])
            self.assertEqual(fixture.requested_sample_ids(), ["1", "2"])
            for sentinel in (
                "UNPRINTABLE_TRANSIENT_SENTINEL",
                "PROMPT_SENTINEL",
                "RAW_RESPONSE_SENTINEL",
            ):
                self.assertNotIn(sentinel, persisted)
            self.assertEqual(fixture.socket_attempt_count, 0)

    def test_task10_sigint_pause_before_first_new_record_preserves_resume_bytes(self):
        with FullRunFixture(sample_count=120) as fixture:
            fixture.seed_predictions(count=100)
            fixture.request_stop_before_sample("1")
            fixture.create_v1_sentinel_that_raises_on_open()
            assert fixture.output_root is not None
            predictions_path = fixture.output_root / "full" / "predictions.jsonl"
            before = predictions_path.read_bytes()
            fd_before = len(os.listdir("/proc/self/fd"))
            handlers_before = {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }

            code = fixture.run_main(resume=True)
            after = predictions_path.read_bytes()
            events = fixture.load_events()
            names = [row["event"] for row in events]

            self.assertEqual(code, 130)
            self.assertEqual(fixture.request_count, 0)
            self.assertEqual(len(fixture.load_predictions()), 100)
            self.assertEqual(after, before)
            self.assertEqual(names[-3:], [
                "stop_requested",
                "prediction_flushed",
                "run_stopped",
            ])
            self.assertIn("stop_requested", fixture._critical_events)
            flushed = events[-2]
            self.assertEqual(flushed["record_count"], 100)
            self.assertEqual(
                flushed["path_sha256"], hashlib.sha256(after).hexdigest()
            )
            self.assertEqual(
                {signum: signal.getsignal(signum) for signum in handlers_before},
                handlers_before,
            )
            self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
            self.assertTrue(
                all(not client._http_clients for client in fixture._clients)
            )
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_sigterm_after_completed_record_flushes_then_resume_runs_missing(self):
        with FullRunFixture(sample_count=3) as fixture:
            fixture._stop_signum = signal.SIGTERM
            fixture.request_stop_before_sample("2")
            fixture.create_v1_sentinel_that_raises_on_open()
            fd_before = len(os.listdir("/proc/self/fd"))
            handlers_before = {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }

            code = fixture.run_main(resume=False)
            stopped_events = fixture.load_events()
            stopped_names = [row["event"] for row in stopped_events]

            self.assertEqual(code, 143)
            self.assertEqual(fixture.requested_sample_ids(), ["1"])
            self.assertEqual(
                [row["sample_id"] for row in fixture.load_predictions()], ["1"]
            )
            self.assertEqual(stopped_names[-3:], [
                "stop_requested",
                "prediction_flushed",
                "run_stopped",
            ])
            self.assertIn("stop_requested", fixture._critical_events)
            self.assertNotIn(
                "2", [row["sample_id"] for row in fixture.load_predictions()]
            )
            sample_one_calls = list(fixture.calls_for("1"))

            self.assertEqual(fixture.run_main(resume=True), 0)
            self.assertEqual(fixture.calls_for("1"), sample_one_calls)
            self.assertEqual(fixture.requested_sample_ids(), ["1", "2", "3"])
            self.assertEqual(
                [row["sample_id"] for row in fixture.load_predictions()],
                ["1", "2", "3"],
            )
            self.assertEqual(
                {signum: signal.getsignal(signum) for signum in handlers_before},
                handlers_before,
            )
            self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
            self.assertTrue(
                all(not client._http_clients for client in fixture._clients)
            )
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_hostile_failures_are_absent_from_all_durable_and_human_paths(self):
        sentinels = (
            "sk-SENTINEL-12345678",
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "BASE64SENTINEL",
            "workspace-private",
            "token=secret",
            "Traceback (most recent call last)",
        )
        hostile = (
            "Authorization: Bearer sk-SENTINEL-12345678 "
            "PROMPT_SENTINEL RAW_RESPONSE_SENTINEL "
            "data:image/png;base64,BASE64SENTINEL "
            "https://workspace-private.cn-beijing.maas.aliyuncs.com/path?token=secret "
            "Traceback (most recent call last)"
        )
        outcomes = (
            {"kind": "schema_error", "message": hostile},
            {
                "kind": "http_error",
                "status": 403,
                "message": f"content policy refusal {hostile}",
            },
        )
        for outcome in outcomes:
            with self.subTest(kind=outcome["kind"]), FullRunFixture(1) as fixture:
                fixture.respond(sample_id="1", outcome=outcome)
                fixture.create_v1_sentinel_that_raises_on_open()
                result = fixture.run(resume=False)
                persisted = fixture.all_durable_and_human_text()

                self.assertTrue(result["gates"]["passed"])
                self.assertTrue(
                    fixture.load_predictions()[0]["prediction"]["backend_error"]
                )
                for sentinel in sentinels:
                    self.assertNotIn(sentinel, persisted)
                self.assertEqual(fixture.socket_attempt_count, 0)
                self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_hostile_typed_control_stderr_is_generic_and_traceback_free(self):
        import argparse

        from scammark import run_experiment
        from scammark.observability import AuthBillingStop

        hostile = (
            "Authorization: Bearer sk-SENTINEL-12345678 "
            "PROMPT_SENTINEL RAW_RESPONSE_SENTINEL workspace-private "
            "Traceback (most recent call last)"
        )
        args = argparse.Namespace(
            config="fixture.toml",
            pilot=None,
            full=True,
            resume=True,
            validate_only=False,
            ack_breaker=False,
        )
        with patch.object(
            run_experiment, "parse_args", return_value=args
        ), patch.object(
            run_experiment, "run", side_effect=AuthBillingStop(hostile)
        ), patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(run_experiment.main(), 20)

        rendered = stderr.getvalue()
        self.assertIn("AuthBillingStop", rendered)
        self.assertLessEqual(len(rendered), 700)
        for sentinel in (
            "sk-SENTINEL-12345678",
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "workspace-private",
            "Traceback",
        ):
            self.assertNotIn(sentinel, rendered)

    def test_task10_bootstrap_contract_failures_exit_23_with_one_safe_stop(self):
        cases = (
            "state_config",
            "checkpoint",
            "ledger",
            "predictions",
            "predictions_syntax",
            "frozen_manifest",
        )
        for case in cases:
            with self.subTest(case=case), FullRunFixture(2) as fixture:
                assert fixture.output_root is not None
                if case == "state_config":
                    state_root = fixture.output_root / "state_full"
                    state_root.mkdir(parents=True)
                    (state_root / "run_config.json").write_text(
                        "{MALFORMED_STATE_SENTINEL", encoding="utf-8"
                    )
                elif case == "checkpoint":
                    fixture.seed_partial_checkpoint("1", ("url",))
                    checkpoint = next(
                        (fixture.output_root / "state_full" / "checkpoints").rglob(
                            "*.json"
                        )
                    )
                    checkpoint.write_text(
                        "{MALFORMED_CHECKPOINT_SENTINEL", encoding="utf-8"
                    )
                elif case == "ledger":
                    state = fixture._state()
                    state.call_ledger.write_text(
                        '{"event":"reserved","MALFORMED_LEDGER_SENTINEL":',
                        encoding="utf-8",
                    )
                elif case in {"predictions", "predictions_syntax"}:
                    fixture.seed_predictions(1)
                    predictions = fixture.output_root / "full" / "predictions.jsonl"
                    if case == "predictions_syntax":
                        predictions.write_text(
                            '{"sample_id":"1"}\n{MALFORMED_PREDICTION_SENTINEL',
                            encoding="utf-8",
                        )
                    else:
                        with predictions.open("a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        "sample_id": "1",
                                        "MALFORMED_PREDICTION_SENTINEL": True,
                                    }
                                )
                                + "\n"
                            )
                else:
                    assert fixture._frozen_root is not None
                    manifest = fixture._frozen_root / "scamweb_v_manifest.jsonl"
                    manifest.write_text(
                        manifest.read_text(encoding="utf-8")
                        + "MALFORMED_MANIFEST_SENTINEL\n",
                        encoding="utf-8",
                    )
                fixture.create_v1_sentinel_that_raises_on_open()

                code = fixture.run_main(resume=True)
                events = fixture.load_events()
                stops = [row for row in events if row["event"] == "run_stopped"]
                persisted = (
                    json.dumps(events, ensure_ascii=False)
                    + fixture.stdout.getvalue()
                    + fixture._stderr.getvalue()
                )

                self.assertEqual(code, 23)
                self.assertEqual(len(stops), 1)
                self.assertEqual(stops[0]["category"], "contract_drift")
                self.assertEqual(stops[0]["exit_code"], 23)
                self.assertFalse(stops[0]["forced"])
                self.assertTrue(stops[0]["boundary"])
                self.assertEqual(
                    [row["event"] for row in events].count("prediction_flushed"),
                    0,
                )
                self.assertNotIn("MALFORMED_", persisted)
                self.assertNotIn("Traceback", persisted)
                self.assertEqual(fixture.socket_attempt_count, 0)
                self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_pre_sink_config_contract_exits_23_without_event_or_traceback(self):
        mutations = (
            ("[models]", "[models_missing]"),
            ("allow_full_run = true", "allow_full_run = false"),
        )
        for before, after in mutations:
            with self.subTest(before=before), FullRunFixture(1) as fixture:
                assert fixture.config_path is not None
                fixture.config_path.write_text(
                    fixture.config_path.read_text(encoding="utf-8").replace(
                        before, after, 1
                    ),
                    encoding="utf-8",
                )
                fixture.create_v1_sentinel_that_raises_on_open()

                code = fixture.run_main(resume=False)
                persisted = fixture.all_durable_and_human_text()

                self.assertEqual(code, 23)
                self.assertEqual(fixture.load_events(), [])
                self.assertNotIn("Traceback", persisted)
                self.assertEqual(fixture.request_count, 0)
                self.assertEqual(fixture.socket_attempt_count, 0)
                self.assertEqual(fixture.v1_open_count, 0)

    def test_task10_late_evaluation_and_metrics_oserrors_remain_unexpected_exit_1(self):
        from scammark import run_experiment

        for phase in ("evaluation", "metrics"):
            with self.subTest(phase=phase), FullRunFixture(1) as fixture:
                fixture.create_v1_sentinel_that_raises_on_open()
                if phase == "evaluation":
                    patches = patch.object(
                        run_experiment,
                        "evaluate_prediction_records",
                        side_effect=OSError("LATE_EVALUATION_SENTINEL"),
                    )
                else:
                    original_write = run_experiment._atomic_write_json

                    def fail_metrics(path, payload):
                        if Path(path).name == "metrics.json":
                            raise OSError("LATE_METRICS_SENTINEL")
                        return original_write(path, payload)

                    patches = patch.object(
                        run_experiment,
                        "_atomic_write_json",
                        new=fail_metrics,
                    )
                with patches:
                    code = fixture.run_main(resume=False)
                events = fixture.load_events()
                persisted = fixture.all_durable_and_human_text()

                self.assertEqual(code, 1)
                self.assertFalse(
                    any(
                        row["event"] == "run_stopped"
                        and row.get("category") == "contract_drift"
                        for row in events
                    )
                )
                self.assertNotIn("Traceback", persisted)
                self.assertNotIn("LATE_", persisted)
                self.assertEqual(fixture.socket_attempt_count, 0)
                self.assertEqual(fixture.v1_open_count, 0)

    def test_review_exact_v2_identity_enforces_normal_and_validate_namespace(self):
        from scammark import run_experiment
        from scammark.experiment_state import ExperimentContractError

        for validate_only in (False, True):
            with self.subTest(validate_only=validate_only), FullRunFixture(1) as fixture:
                self.assertIsNotNone(fixture.config_path)
                text = fixture.config_path.read_text(encoding="utf-8").replace(
                    'name = "scammark-qwen3-dashscope-server-full-v2"',
                    'name = "scammark-qwen3-dashscope-server-full-v2-alias"',
                    1,
                )
                fixture.config_path.write_text(text, encoding="utf-8")

                with self.assertRaises(ExperimentContractError):
                    if validate_only:
                        run_experiment.run(
                            fixture.config_path,
                            None,
                            False,
                            validate_only=True,
                        )
                    else:
                        fixture.run(resume=False)
                self.assertEqual(fixture.request_count, 0)
                self.assertEqual(fixture.socket_attempt_count, 0)

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.config_path)
            self.assertIsNotNone(fixture.output_root)
            self.assertIsNotNone(fixture.v1_root)
            text = fixture.config_path.read_text(encoding="utf-8")
            text = text.replace(
                f'output_root = "{fixture.output_root.as_posix()}"',
                f'output_root = "{fixture.v1_root.as_posix()}"',
                1,
            )
            text = text.replace(
                f'image_cache = "{(fixture.output_root / "image_cache").as_posix()}"',
                f'image_cache = "{(fixture.v1_root / "image_cache").as_posix()}"',
                1,
            )
            fixture.config_path.write_text(text, encoding="utf-8")

            with self.assertRaises(ExperimentContractError):
                run_experiment.run(
                    fixture.config_path,
                    None,
                    False,
                    validate_only=True,
                )
            self.assertEqual(fixture.request_count, 0)
            self.assertEqual(fixture.socket_attempt_count, 0)

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.config_path)
            legacy_root = fixture.root / "outputs" / "legacy-validate"
            text = fixture.config_path.read_text(encoding="utf-8")
            text = text.replace(
                'name = "scammark-qwen3-dashscope-server-full-v2"',
                'name = "legacy-validate-only-fixture"',
                1,
            )
            text = text.replace(
                f'output_root = "{fixture.output_root.as_posix()}"',
                f'output_root = "{legacy_root.as_posix()}"',
                1,
            )
            text = text.replace(
                f'image_cache = "{(fixture.output_root / "image_cache").as_posix()}"',
                f'image_cache = "{(legacy_root / "image_cache").as_posix()}"',
                1,
            )
            fixture.config_path.write_text(text, encoding="utf-8")

            result = run_experiment.run(
                fixture.config_path,
                None,
                False,
                validate_only=True,
            )

            self.assertEqual(result["preflight"]["status"], "validated")
            self.assertEqual(fixture.request_count, 0)
            self.assertEqual(fixture.socket_attempt_count, 0)

    def test_review_v2_runtime_symlinks_are_rejected_before_touch_or_request(self):
        from scammark.experiment_state import ExperimentContractError

        cases = (
            ("full/run_events.jsonl", False),
            ("state_full", True),
            ("state_full/run_config.json", False),
            ("state_full/checkpoints", True),
            ("state_full/api_call_ledger.jsonl", False),
            ("pilot_10/run_events.jsonl", False),
            ("pilot_10/metrics.json", False),
            ("full/predictions.jsonl", False),
            ("full/human_evaluation_candidates.csv", False),
            ("full/full_summary.md", False),
            ("image_cache", True),
        )
        for relative, is_directory in cases:
            with self.subTest(relative=relative), FullRunFixture(1) as fixture:
                self.assertIsNotNone(fixture.output_root)
                self.assertIsNotNone(fixture.v1_root)
                target = fixture.v1_root / "review-target"
                target.parent.mkdir(parents=True, exist_ok=True)
                if is_directory:
                    target.mkdir()
                    (target / "sentinel.txt").write_text(
                        "unchanged", encoding="utf-8"
                    )
                    before = tuple(sorted(os.listdir(target)))
                else:
                    target.write_bytes(b'{"sentinel":"unchanged"}\n')
                    with open(target, "rb") as handle:
                        before = handle.read()
                local = fixture.output_root / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                local.symlink_to(target, target_is_directory=is_directory)
                fixture.create_v1_sentinel_that_raises_on_open()

                with self.assertRaises(ExperimentContractError):
                    fixture.run(resume=True)

                if is_directory:
                    after = tuple(sorted(os.listdir(target)))
                else:
                    with open(target, "rb") as handle:
                        after = handle.read()
                self.assertEqual(after, before)
                self.assertEqual(fixture.request_count, 0)
                self.assertEqual(fixture.socket_attempt_count, 0)
                self.assertEqual(fixture.v1_open_count, 0)

    def test_review_validate_only_rejects_v2_child_symlink_without_touching_target(self):
        from scammark import run_experiment
        from scammark.experiment_state import ExperimentContractError

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.output_root)
            self.assertIsNotNone(fixture.v1_root)
            target = fixture.v1_root / "validate-predictions.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'{"sentinel":"unchanged"}\n')
            local = fixture.output_root / "full" / "predictions.jsonl"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.symlink_to(target)
            fixture.create_v1_sentinel_that_raises_on_open()

            with self.assertRaises(ExperimentContractError):
                run_experiment.run(
                    fixture.config_path,
                    None,
                    False,
                    validate_only=True,
                )

            with open(target, "rb") as handle:
                self.assertEqual(
                    handle.read(), b'{"sentinel":"unchanged"}\n'
                )
            self.assertEqual(fixture.request_count, 0)
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)

    def test_rereview_hard_linked_run_events_preserves_external_inode(self):
        from scammark.experiment_state import ExperimentContractError

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.output_root)
            self.assertIsNotNone(fixture.v1_root)
            target = fixture.v1_root / "hard-linked-events-target.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            local = fixture.output_root / "full" / "run_events.jsonl"
            local.parent.mkdir(parents=True, exist_ok=True)
            os.link(target, local)
            before = target.read_bytes()
            before_nlink = target.stat().st_nlink
            fixture.create_v1_sentinel_that_raises_on_open()

            caught = None
            try:
                fixture.run(resume=True)
            except BaseException as exc:  # noqa: BLE001 - inspect exact boundary type.
                caught = exc
            with open(target, "rb") as handle:
                after = handle.read()

            self.assertEqual(
                (
                    after,
                    target.stat().st_nlink,
                    fixture.request_count,
                    fixture.socket_attempt_count,
                    fixture.v1_open_count,
                ),
                (before, before_nlink, 0, 0, 0),
            )
            self.assertIsInstance(caught, ExperimentContractError)

    def test_rereview_recursive_runtime_hard_links_are_rejected_untouched(self):
        from scammark.experiment_state import ExperimentContractError

        cases = (
            "full/nested/deep/runtime.bin",
            "state_full/nested/foreign.bin",
            "pilot_10/run_events.jsonl",
            "pilot_10/metrics.json",
            "full/predictions.jsonl",
            "full/human_evaluation_candidates.csv",
            "full/full_summary.md",
        )
        for relative in cases:
            with self.subTest(relative=relative), FullRunFixture(1) as fixture:
                self.assertIsNotNone(fixture.output_root)
                self.assertIsNotNone(fixture.v1_root)
                target = fixture.v1_root / "recursive-hard-link-target"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"review-hard-link-target\n")
                local = fixture.output_root / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                os.link(target, local)
                before = target.read_bytes()
                before_nlink = target.stat().st_nlink
                fixture.create_v1_sentinel_that_raises_on_open()

                caught = None
                try:
                    fixture.run(resume=True)
                except BaseException as exc:  # noqa: BLE001 - inspect boundary type.
                    caught = exc
                with open(target, "rb") as handle:
                    after = handle.read()

                self.assertEqual(
                    (
                        after,
                        target.stat().st_nlink,
                        fixture.request_count,
                        fixture.socket_attempt_count,
                        fixture.v1_open_count,
                    ),
                    (before, before_nlink, 0, 0, 0),
                )
                self.assertIsInstance(caught, ExperimentContractError)

    def test_rereview_event_validation_to_open_race_fails_closed(self):
        from scammark import run_experiment
        from scammark.experiment_state import ExperimentContractError

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.output_root)
            self.assertIsNotNone(fixture.v1_root)
            events_path = fixture.output_root / "full" / "run_events.jsonl"
            target = fixture.v1_root / "race-events-target.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            original_validate = run_experiment._validate_runtime_path
            raced = []

            def validate_then_swap(output_root, path):
                result = original_validate(output_root, path)
                if Path(path) == events_path and not raced:
                    events_path.parent.mkdir(parents=True, exist_ok=True)
                    events_path.symlink_to(target)
                    raced.append(True)
                return result

            fixture.create_v1_sentinel_that_raises_on_open()
            caught = None
            with patch.object(
                run_experiment,
                "_validate_runtime_path",
                new=validate_then_swap,
            ):
                try:
                    fixture.run(resume=True)
                except BaseException as exc:  # noqa: BLE001 - inspect boundary type.
                    caught = exc
            with open(target, "rb") as handle:
                after = handle.read()

            self.assertEqual(raced, [True])
            self.assertEqual(
                (
                    after,
                    fixture.request_count,
                    fixture.socket_attempt_count,
                    fixture.v1_open_count,
                ),
                (b"", 0, 0, 0),
            )
            self.assertIsInstance(caught, ExperimentContractError)

    def test_final_review_events_ancestor_swap_never_writes_v1(self):
        from scammark import run_experiment
        from scammark.experiment_state import ExperimentContractError

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.output_root)
            self.assertIsNotNone(fixture.v1_root)
            logical_full = fixture.output_root / "full"
            logical_full.mkdir(parents=True)
            displaced_full = fixture.output_root / "full-before-review-swap"
            target_dir = fixture.v1_root / "ancestor-events-target"
            target_dir.mkdir(parents=True, exist_ok=True)
            events_path = logical_full / "run_events.jsonl"
            original_validate = run_experiment._validate_runtime_path
            raced = []

            def validate_then_swap(output_root, path):
                result = original_validate(output_root, path)
                if Path(path) == events_path and not raced:
                    logical_full.rename(displaced_full)
                    logical_full.symlink_to(target_dir, target_is_directory=True)
                    raced.append(True)
                return result

            fixture.create_v1_sentinel_that_raises_on_open()
            caught = None
            with patch.object(
                run_experiment,
                "_validate_runtime_path",
                new=validate_then_swap,
            ):
                try:
                    fixture.run(resume=True)
                except BaseException as exc:  # noqa: BLE001 - inspect boundary type.
                    caught = exc

            self.assertEqual(raced, [True])
            self.assertEqual(tuple(sorted(os.listdir(target_dir))), ())
            self.assertEqual(fixture.request_count, 0)
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)
            self.assertIsInstance(caught, ExperimentContractError)

    def test_final_review_validate_only_ancestor_swap_never_writes_v1(self):
        from scammark import run_experiment
        from scammark.experiment_state import ExperimentContractError

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.output_root)
            self.assertIsNotNone(fixture.v1_root)
            logical_full = fixture.output_root / "full"
            logical_full.mkdir(parents=True)
            displaced_full = fixture.output_root / "full-before-validate-swap"
            target_dir = fixture.v1_root / "ancestor-validate-target"
            target_dir.mkdir(parents=True, exist_ok=True)
            original_validate = run_experiment._validate_v2_runtime_namespace
            raced = []

            def validate_then_swap(*args, **kwargs):
                result = original_validate(*args, **kwargs)
                logical_full.rename(displaced_full)
                logical_full.symlink_to(target_dir, target_is_directory=True)
                raced.append(True)
                return result

            caught = None
            with patch.object(
                run_experiment,
                "_validate_v2_runtime_namespace",
                new=validate_then_swap,
            ):
                try:
                    run_experiment.run(
                        fixture.config_path,
                        None,
                        False,
                        validate_only=True,
                    )
                except BaseException as exc:  # noqa: BLE001 - inspect boundary type.
                    caught = exc

            self.assertEqual(raced, [True])
            self.assertEqual(tuple(sorted(os.listdir(target_dir))), ())
            self.assertEqual(fixture.request_count, 0)
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)
            self.assertIsInstance(caught, ExperimentContractError)

    def test_rereview_child_mount_and_reparse_markers_are_rejected(self):
        import stat

        from scammark import run_experiment
        from scammark.experiment_state import ExperimentContractError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scammark_qwen3_dashscope_server_full_v2"
            child = root / "full" / "child"
            child.mkdir(parents=True)
            with patch.object(
                run_experiment.os.path,
                "ismount",
                side_effect=lambda path: Path(path) == child,
            ), self.assertRaises(ExperimentContractError):
                run_experiment._reject_existing_runtime_symlinks(root)

        class ReparseStat:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.st_file_attributes = getattr(
                    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
                )

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scammark_qwen3_dashscope_server_full_v2"
            child = root / "full" / "child"
            child.mkdir(parents=True)
            real_lstat = os.lstat

            def marked_lstat(path, *args, **kwargs):
                result = real_lstat(path, *args, **kwargs)
                return ReparseStat(result) if Path(path) == child else result

            with patch.object(
                run_experiment.os, "lstat", new=marked_lstat
            ), patch.object(
                run_experiment.os.path, "ismount", return_value=False
            ), self.assertRaises(ExperimentContractError):
                run_experiment._reject_existing_runtime_symlinks(root)

    def test_review_summary_refreshes_replay_allowance_after_checkpoint(self):
        from scammark import run_experiment

        estimator_calls = []

        def record_estimate(
            tracker,
            remaining_stage_counts,
            *,
            outstanding_by_stage,
            replay_required_by_stage,
        ):
            estimator_calls.append(
                {
                    "outstanding": dict(outstanding_by_stage),
                    "replay": dict(replay_required_by_stage),
                }
            )
            return {"USD": 0.0, "CNY": 0.0}

        with FullRunFixture(1) as fixture:
            state = fixture._state()
            call_id = state.reserve_api_call(
                {"sample_id": "1", "stage": "text"}, max_calls=50_000
            )
            state.finish_api_call(call_id, status="completed")
            with patch.object(
                run_experiment.SummaryScheduler,
                "should_emit",
                return_value=True,
            ), patch.object(
                run_experiment.ProgressTracker,
                "estimate_remaining_cost",
                new=record_estimate,
            ):
                fixture.run(resume=True)
            summary = next(
                row
                for row in reversed(fixture.load_events())
                if row["event"] == "summary"
            )

        self.assertEqual(estimator_calls[-1]["replay"], {})
        self.assertEqual(estimator_calls[-1]["outstanding"], {})
        self.assertEqual(
            summary["cost_calculation"]["replay_required_by_stage"], {}
        )
        self.assertEqual(
            summary["cost_calculation"]["outstanding_by_stage"], {}
        )

    def test_review_structured_cost_calculation_has_used_means_and_safe_inputs(self):
        with FullRunFixture(26) as fixture:
            fixture.run(resume=False)
            summary = next(
                row for row in fixture.load_events() if row["event"] == "summary"
            )
            calculation = summary["cost_calculation"]

        self.assertEqual(
            set(calculation),
            {
                "stage_means",
                "population",
                "pricing",
                "retry_assumptions",
                "outstanding_by_stage",
                "replay_required_by_stage",
                "calculated_at_process_seconds",
            },
        )
        for stage in ("url", "text", "visual", "cross", "binary"):
            mean = calculation["stage_means"][stage]
            self.assertEqual(mean["source"], "full")
            self.assertEqual(mean["observation_count"], 25)
            self.assertEqual(mean["mean_input_tokens"], 10.0)
            self.assertEqual(mean["mean_output_tokens"], 2.0)
            self.assertGreaterEqual(mean["mean_elapsed_seconds"], 0.0)
            self.assertIn(mean["currency"], {"USD", "CNY"})
            self.assertGreaterEqual(mean["mean_cost"], 0.0)
        self.assertEqual(calculation["population"]["manifest_samples"], 26)
        self.assertEqual(
            calculation["population"]["remaining_stage_counts"],
            {"url": 1, "text": 1, "visual": 1, "cross": 1, "binary": 1, "type": 0, "grounding": 0},
        )
        self.assertEqual(
            calculation["pricing"],
            {
                "text": {
                    "currency": "USD",
                    "input_per_million": 0.1,
                    "output_per_million": 0.42,
                },
                "vision": {
                    "currency": "CNY",
                    "input_per_million": 0.15,
                    "output_per_million": 1.5,
                },
            },
        )
        self.assertEqual(
            calculation["retry_assumptions"],
            {"current_retry_count": 0, "usage_included_in_stage_means": True},
        )
        serialized = json.dumps(calculation, ensure_ascii=False).lower()
        for forbidden in (
            "accrued",
            "spent",
            "estimated_total",
            "per_sample",
            "api_key",
            "prompt",
            "response",
            "base64",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_review_full_event_history_keeps_early_retry_once_beyond_tail(self):
        from scammark.observability import EVENT_SCHEMA

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.output_root)
            path = fixture.output_root / "full" / "run_events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            common = {
                "schema": EVENT_SCHEMA,
                "experiment_version": "full-v2",
                "run_mode": "full",
                "attempt_id": "review-prior-attempt",
                "timestamp_utc": "2026-07-12T12:00:00+00:00",
                "timestamp_local": "2026-07-12T20:00:00+08:00",
                "process_elapsed_seconds": 1.0,
            }
            retry = {
                **common,
                "event": "retry_scheduled",
                "sequence": 1,
                "sample_id": "1",
                "stage": "url",
                "model": "gpt-4o-mini",
                "attempt": 1,
                "next_attempt": 2,
                "delay_seconds": 2.0,
                "error_type": "TransientRequestError",
                "message": "transient provider or transport failure",
            }
            rows = [json.dumps(retry), json.dumps(retry)]
            padding = "x" * 900
            for sequence in range(2, 10_052):
                rows.append(
                    json.dumps(
                        {
                            **common,
                            "event": "sample_started",
                            "sequence": sequence,
                            "sample_index": 1,
                            "sample_total": 1,
                            "sample_id": "1",
                            "padding": padding,
                        }
                    )
                )
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            self.assertGreater(path.stat().st_size, 8 * 1024 * 1024)

            fixture.run(resume=False)
            completed = next(
                row
                for row in reversed(fixture.load_events())
                if row["event"] == "sample_complete"
            )

        self.assertEqual(completed["retry_count"], 1)

    def test_review_malformed_run_config_becomes_contract_stop_not_generic_error(self):
        from scammark.experiment_state import ExperimentContractError

        with FullRunFixture(1) as fixture:
            self.assertIsNotNone(fixture.output_root)
            state_root = fixture.output_root / "state_full"
            state_root.mkdir(parents=True)
            (state_root / "run_config.json").write_text(
                "{malformed run config", encoding="utf-8"
            )

            with self.assertRaises(ExperimentContractError):
                fixture.run(resume=True)
            events = fixture.load_events()
            persisted = json.dumps(events, ensure_ascii=False)

        self.assertEqual(events[-1]["event"], "run_stopped")
        self.assertEqual(events[-1]["category"], "contract_drift")
        self.assertEqual(events[-1]["reason"], "state_construction_failed")
        self.assertNotIn("malformed run config", persisted)
        self.assertNotIn("traceback", persisted.lower())

    def test_review_heartbeat_activity_tracks_reused_stage_without_warming(self):
        from scammark.observability import HeartbeatWorker, RunEventLogger

        holder = {}
        observed = []
        original_emit = RunEventLogger.emit

        def capture_start(worker):
            holder["worker"] = worker

        def capture_emit(logger, event, *, critical=False, **fields):
            if event in {"sample_started", "stage_reused", "sample_complete"}:
                worker = holder.get("worker")
                if worker is not None:
                    observed.append((event, dict(fields), dict(worker.snapshot())))
            return original_emit(
                logger, event, critical=critical, **fields
            )

        with FullRunFixture(1) as fixture:
            fixture.seed_complete_checkpoint("1")
            with patch.object(HeartbeatWorker, "start", new=capture_start), patch.object(
                RunEventLogger, "emit", new=capture_emit
            ):
                fixture.run(resume=True)

        started = next(item for item in observed if item[0] == "sample_started")
        reused = [item for item in observed if item[0] == "stage_reused"]
        completed = next(item for item in observed if item[0] == "sample_complete")
        self.assertEqual(started[2]["current_sample_id"], "1")
        self.assertIsNone(started[2]["current_stage"])
        self.assertEqual(len(reused), 7)
        for _, fields, snapshot in reused:
            self.assertEqual(snapshot["current_sample_id"], "1")
            self.assertEqual(snapshot["current_stage"], fields["stage"])
            self.assertIsNone(snapshot["eta_seconds"])
        self.assertIsNone(completed[2]["current_sample_id"])
        self.assertIsNone(completed[2]["current_stage"])
        self.assertGreaterEqual(
            completed[2]["last_sample_completed_at"],
            started[2]["last_sample_completed_at"],
        )

    def test_review_prediction_merge_rejects_custom_mapping_rows(self):
        from collections import UserDict

        from scammark.experiment_state import ExperimentContractError
        from scammark.run_experiment import _merge_prediction_records

        class DictSubclass(dict):
            pass

        for row in (
            UserDict({"sample_id": "1"}),
            DictSubclass(sample_id="1"),
        ):
            with self.subTest(row_type=type(row).__name__), self.assertRaises(
                ExperimentContractError
            ):
                _merge_prediction_records([row], [], ["1"])

    def test_post_prediction_projection_and_cost_failures_are_observational(self):
        import scammark.run_experiment as run_experiment

        original_projection = run_experiment.project_remaining_stage_counts
        projection_calls = [0]

        def fail_post_prediction_projection(*args, **kwargs):
            projection_calls[0] += 1
            if projection_calls[0] > 1:
                raise OSError("post-prediction projection failure")
            return original_projection(*args, **kwargs)

        with FullRunFixture(sample_count=1) as fixture, patch.object(
            run_experiment,
            "project_remaining_stage_counts",
            new=fail_post_prediction_projection,
        ):
            result = fixture.run(resume=False)
            predictions = fixture.load_predictions()
            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(len(predictions), 1)
            self.assertEqual(
                fixture.calls_for("1"),
                ["url", "text", "visual", "cross", "binary"],
            )

        with FullRunFixture(sample_count=1) as fixture, patch.object(
            run_experiment.SummaryScheduler,
            "should_emit",
            return_value=True,
        ), patch.object(
            run_experiment.ProgressTracker,
            "estimate_remaining_cost",
            side_effect=OSError("cost projection failure"),
        ):
            result = fixture.run(resume=False)
            predictions = fixture.load_predictions()
            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(len(predictions), 1)
            self.assertEqual(
                fixture.calls_for("1"),
                ["url", "text", "visual", "cross", "binary"],
            )

    def test_runner_reads_only_safe_pilot_10_metrics_false_positive_fallback(self):
        from scammark.run_experiment import _safe_pilot_false_positive_rate

        with FullRunFixture(sample_count=1) as fixture:
            self.assertIsNotNone(fixture.output_root)
            metrics_path = fixture.output_root / "pilot_10" / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(
                    {"binary": {"confusion": {"fp": 1, "tn": 3}}}
                ),
                encoding="utf-8",
            )
            fixture.create_v1_sentinel_that_raises_on_open()

            self.assertEqual(
                _safe_pilot_false_positive_rate(fixture.output_root), 0.25
            )
            self.assertEqual(fixture.v1_open_count, 0)
            metrics_path.write_text("{malformed", encoding="utf-8")
            self.assertIsNone(
                _safe_pilot_false_positive_rate(fixture.output_root)
            )

    def test_run_started_keeps_sanitized_config_hash_identity(self):
        with FullRunFixture(sample_count=1) as fixture:
            fixture.run(resume=False)
            started = fixture.load_events()[0]

        self.assertEqual(started["event"], "run_started")
        self.assertRegex(started["config_sha256"], r"^[0-9a-f]{64}$")

    def test_full_run_fixture_future_api_is_stable_local_and_restores_guards(self):
        import inspect
        import socket

        original_cwd = Path.cwd()
        original_socket_boundary = socket.create_connection
        expected_methods = {
            "run",
            "run_main",
            "respond",
            "seed_complete_checkpoint",
            "seed_partial_checkpoint",
            "seed_predictions",
            "seed_sample_events",
            "seed_breaker_opened",
            "request_stop_before_sample",
            "create_v1_sentinel_that_raises_on_open",
            "load_events",
            "load_predictions",
            "calls_for",
            "requested_sample_ids",
            "all_durable_and_human_text",
        }
        self.assertTrue(
            expected_methods
            <= {
                name
                for name, value in inspect.getmembers(
                    FullRunFixture, predicate=inspect.isfunction
                )
            }
        )

        with FullRunFixture(sample_count=3) as fixture:
            fixture.respond(
                sample_id="3",
                outcome={"kind": "success", "binary": "benign"},
            )
            fixture.seed_complete_checkpoint(sample_id="1")
            fixture.seed_partial_checkpoint(
                sample_id="2", complete_stages=("url", "text")
            )
            fixture.seed_predictions(count=2)
            fixture.seed_sample_events(backend_errors=1)
            fixture.request_stop_before_sample("future-only")
            fixture.create_v1_sentinel_that_raises_on_open()

            result = fixture.run(resume=True)
            fixture.seed_breaker_opened()
            code = fixture.run_main(resume=True, ack_breaker=True)
            events = fixture.load_events()
            predictions = fixture.load_predictions()
            durable_text = fixture.all_durable_and_human_text()

            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(code, 0)
            self.assertTrue(events)
            self.assertEqual(len(predictions), 3)
            self.assertGreaterEqual(fixture.request_count, 1)
            self.assertTrue(fixture.calls_for("2"))
            self.assertIn("2", fixture.requested_sample_ids())
            self.assertIn("run_started", durable_text)
            self.assertIs(fixture.stdout, fixture.stdout)
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)
            with self.assertRaises(AttributeError):
                fixture.request_count = 99
            with self.assertRaises(AttributeError):
                fixture.stdout = io.StringIO()

        self.assertEqual(Path.cwd(), original_cwd)
        self.assertIs(socket.create_connection, original_socket_boundary)

    def test_run_started_precedes_early_state_contract_hash_failure(self):
        primary = RuntimeError("early contract hash failure")
        with FullRunFixture(sample_count=1) as fixture, patch(
            "scammark.run_experiment._selected_image_hashes",
            side_effect=primary,
        ):
            with self.assertRaises(RuntimeError) as caught:
                fixture.run(resume=False)
            events = fixture.load_events()
            self.assertFalse((fixture.output_root / "state_full").exists())

        self.assertIs(caught.exception, primary)
        self.assertEqual(events[0]["event"], "run_started")
        self.assertNotIn("traceback", json.dumps(events).lower())

    def test_state_or_client_construction_failure_restores_signal_handlers(self):
        import signal

        before = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        state_error = RuntimeError("state construction primary")
        with FullRunFixture(sample_count=1) as fixture, patch(
            "scammark.run_experiment.ExperimentState", side_effect=state_error
        ):
            with self.assertRaises(RuntimeError) as caught:
                fixture.run(resume=False)
        self.assertIs(caught.exception, state_error)
        self.assertEqual(
            before,
            {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGINT, signal.SIGTERM)
            },
        )

        client_error = RuntimeError("client construction primary")
        with FullRunFixture(sample_count=1) as fixture, patch.object(
            fixture, "_make_client", side_effect=client_error
        ):
            with self.assertRaises(RuntimeError) as caught:
                fixture.run(resume=False)
        self.assertIs(caught.exception, client_error)
        self.assertEqual(
            before,
            {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGINT, signal.SIGTERM)
            },
        )

    def test_event_tracker_and_heartbeat_failures_are_observational(self):
        with FullRunFixture(sample_count=1) as fixture, patch(
            "scammark.observability.RunEventLogger.emit",
            side_effect=OSError("event sink body PROMPT_SENTINEL"),
        ):
            result = fixture.run(resume=False)
            predictions = fixture.load_predictions()
            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(fixture.calls_for("1"), [
                "url", "text", "visual", "cross", "binary"
            ])
            self.assertEqual(predictions[0]["prediction"]["binary"], "benign")

        with FullRunFixture(sample_count=1) as fixture, patch(
            "scammark.run_experiment.ProgressTracker.record_stage",
            side_effect=OSError("tracker failure"),
        ):
            result = fixture.run(resume=False)
            names = [row["event"] for row in fixture.load_events()]
            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(names.count("stage_completed"), 5)
            self.assertEqual(
                fixture.calls_for("1"),
                ["url", "text", "visual", "cross", "binary"],
            )

        calls = {"start": 0, "tick": 0, "close": 0}

        def fail_tick(worker):
            calls["tick"] += 1
            raise OSError("heartbeat tick failure")

        def fail_start(worker):
            calls["start"] += 1
            worker.tick()

        def fail_close(worker):
            calls["close"] += 1
            raise OSError("heartbeat close failure")

        with FullRunFixture(sample_count=1) as fixture, patch(
            "scammark.run_experiment.HeartbeatWorker.tick", new=fail_tick
        ), patch(
            "scammark.run_experiment.HeartbeatWorker.start", new=fail_start
        ), patch(
            "scammark.run_experiment.HeartbeatWorker.close", new=fail_close
        ):
            result = fixture.run(resume=False)
            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(
                fixture.calls_for("1"),
                ["url", "text", "visual", "cross", "binary"],
            )
        self.assertEqual(calls, {"start": 1, "tick": 1, "close": 1})

        activity_error = OSError("activity PROMPT_SENTINEL")
        aggregate_error = OSError("aggregate PROMPT_SENTINEL")
        with FullRunFixture(sample_count=1) as fixture, patch(
            "scammark.run_experiment._ProgressActivity.begin_sample",
            side_effect=activity_error,
        ), patch(
            "scammark.run_experiment._ProgressActivity.begin_stage",
            side_effect=activity_error,
        ), patch(
            "scammark.run_experiment._ProgressActivity.complete_sample",
            side_effect=activity_error,
        ), patch(
            "scammark.run_experiment._record_safe_stage_aggregate",
            side_effect=aggregate_error,
        ):
            result = fixture.run(resume=False)
            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(
                fixture.calls_for("1"),
                ["url", "text", "visual", "cross", "binary"],
            )

    def test_primary_evaluation_exception_survives_all_cleanup_failures(self):
        import signal

        from scammark.model_client import OpenAIAPIModelClient
        from scammark.observability import HeartbeatWorker, StopController

        primary = RuntimeError("evaluation primary")
        calls = {"heartbeat": 0, "client": 0, "controller": 0}
        original_controller_close = StopController.close

        def heartbeat_close(worker):
            calls["heartbeat"] += 1
            raise OSError("heartbeat cleanup")

        def client_close(client):
            calls["client"] += 1
            raise OSError("client cleanup")

        def controller_close(controller):
            calls["controller"] += 1
            original_controller_close(controller)
            raise OSError("controller cleanup")

        before = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        with FullRunFixture(sample_count=1) as fixture, patch(
            "scammark.run_experiment.evaluate_prediction_records",
            side_effect=primary,
        ), patch.object(
            HeartbeatWorker, "close", new=heartbeat_close
        ), patch.object(
            OpenAIAPIModelClient, "close", new=client_close
        ), patch.object(
            StopController, "close", new=controller_close
        ):
            with self.assertRaises(RuntimeError) as caught:
                fixture.run(resume=False)

        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, {"heartbeat": 1, "client": 1, "controller": 1})
        self.assertEqual(
            before,
            {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGINT, signal.SIGTERM)
            },
        )

    def test_metrics_write_primary_exception_is_preserved_after_prediction_flush(self):
        primary = OSError("metrics write primary")

        def fail_metrics(path, payload):
            if Path(path).name == "metrics.json":
                raise primary
            raise AssertionError(f"unexpected JSON write: {path}")

        with FullRunFixture(sample_count=1) as fixture, patch(
            "scammark.run_experiment._atomic_write_json", new=fail_metrics
        ):
            with self.assertRaises(OSError) as caught:
                fixture.run(resume=False)
            predictions = fixture.load_predictions()
            names = [row["event"] for row in fixture.load_events()]

        self.assertIs(caught.exception, primary)
        self.assertEqual(len(predictions), 1)
        self.assertIn("prediction_flushed", names)
        self.assertNotIn("metrics_written", names)
        self.assertNotIn("run_completed", names)

    def test_resume_reuse_does_not_warm_eta_before_five_new_successes(self):
        from scammark.observability import ObservabilitySettings, RunEventLogger

        with FullRunFixture(sample_count=7) as fixture:
            self.assertIsNotNone(fixture.output_root)
            logger = RunEventLogger(
                fixture.output_root / "pilot_10" / "run_events.jsonl",
                ObservabilitySettings(),
                stream=None,
                attempt_id="fixture-smoke",
                run_mode="pilot",
            )
            for stage in (
                "url",
                "text",
                "visual",
                "cross",
                "binary",
                "type",
                "grounding",
            ):
                logger.emit(
                    "stage_completed",
                    sample_id="smoke-1",
                    stage=stage,
                    model=(
                        "gpt-4o-mini"
                        if stage in {"url", "text"}
                        else "qwen3-vl-flash"
                    ),
                    elapsed_seconds=2.0,
                    usage={"prompt_tokens": 100, "completion_tokens": 10},
                    attempt_count=1,
                )
            fixture.seed_complete_checkpoint(sample_id="1")

            fixture.run(resume=True)
            completed = [
                row
                for row in fixture.load_events()
                if row["event"] == "sample_complete"
            ]

            self.assertEqual(completed[0]["sample_id"], "1")
            self.assertIsNone(completed[0].get("eta_seconds"))
            self.assertIsNotNone(completed[0].get("preliminary_eta_seconds"))
            self.assertIsNone(completed[4].get("eta_seconds"))
            self.assertIsNotNone(completed[4].get("preliminary_eta_seconds"))
            self.assertIsNotNone(completed[5].get("eta_seconds"))

    def test_combined_snapshot_heartbeat_and_cost_allowances_are_baseline_safe(self):
        from scammark.observability import (
            HeartbeatWorker,
            ObservabilitySettings,
            ProgressTracker,
            SummaryScheduler,
        )
        from scammark.run_experiment import _combined_progress_snapshot

        now = [0.0]
        settings = ObservabilitySettings(
            heartbeat_poll_seconds=1,
            heartbeat_stall_seconds=5,
            summary_sample_interval=2,
            summary_seconds=10,
            eta_warmup_samples=1,
            cost_full_update_samples=1,
        )
        tracker = ProgressTracker(
            settings,
            total_samples=3,
            api_call_limit=100,
            pricing={
                "text": {
                    "currency": "USD",
                    "input_per_million": 1.0,
                    "output_per_million": 2.0,
                },
                "vision": {
                    "currency": "CNY",
                    "input_per_million": 3.0,
                    "output_per_million": 4.0,
                },
            },
            text_model="text-model",
            vision_model="vision-model",
            monotonic=lambda: now[0],
        )
        tracker.record_stage(
            "url",
            model="text-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            elapsed_seconds=2,
            source="smoke",
        )
        tracker.record_stage(
            "visual",
            model="vision-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            elapsed_seconds=3,
            source="smoke",
        )
        tracker.begin_sample("new-1", index=1)
        tracker.begin_stage("url")
        tracker.reserve_call("url")
        tracker.record_retry()
        combined = _combined_progress_snapshot(
            tracker,
            remaining_stage_counts={"url": 1, "visual": 1},
            durable_completed=7,
            durable_reserved=10,
            durable_outstanding=2,
            durable_retries=3,
            api_call_limit=100,
        )

        self.assertEqual(combined["completed_samples"], 7)
        self.assertEqual(combined["api_calls"], 11)
        self.assertEqual(combined["outstanding_calls"], 3)
        self.assertEqual(combined["retries"], 4)
        emitted = []
        now[0] = 6.0
        HeartbeatWorker(
            settings,
            snapshot=lambda: combined,
            emit=lambda event, **fields: emitted.append((event, fields)),
            monotonic=lambda: now[0],
            api_call_limit=100,
        ).tick()
        self.assertEqual(emitted[0][0], "heartbeat")
        self.assertEqual(emitted[0][1]["api_calls"], 11)
        self.assertEqual(emitted[0][1]["outstanding"], 3)

        tracker.finish_call("url", transport_error=False)
        tracker.record_stage(
            "url",
            model="text-model",
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 0},
            elapsed_seconds=100,
            reused=True,
        )
        tracker.complete_sample("new-1", index=1, backend_error=False)
        fallback_cost = tracker.estimate_remaining_cost(
            {"url": 1, "visual": 1},
            outstanding_by_stage={"url": 2},
            replay_required_by_stage={"url": 3},
        )
        self.assertAlmostEqual(fallback_cost["USD"], 6 * 20 / 1_000_000)
        self.assertAlmostEqual(fallback_cost["CNY"], 50 / 1_000_000)

        scheduler = SummaryScheduler(settings, started_at=0)
        self.assertFalse(scheduler.should_emit(completed_samples=1, now=1))
        self.assertTrue(scheduler.should_emit(completed_samples=2, now=2))
        self.assertFalse(scheduler.should_emit(completed_samples=2, now=9))
        self.assertTrue(scheduler.should_emit(completed_samples=2, now=12))

    def test_runner_filters_forbidden_cost_fields_from_human_callbacks(self):
        with FullRunFixture(sample_count=26) as fixture, patch(
            "scammark.run_experiment.ProgressTracker.estimate_remaining_cost",
            return_value={
                "USD": 1.25,
                "CNY": 2.5,
                "accrued": 999,
                "spent": 999,
                "estimated_total": 999,
                "per_sample": 999,
            },
        ):
            fixture.run(resume=False)
            human = fixture.stdout.getvalue().lower()

        self.assertIn("remaining_cost_est~usd 1.25 + cny 2.50", human)
        for forbidden in (
            "accrued",
            "spent",
            "estimated_total",
            "per_sample",
        ):
            self.assertNotIn(forbidden, human)

    def test_runner_summary_contains_remaining_cost_only(self):
        with FullRunFixture(sample_count=26) as fixture:
            fixture.run(resume=False)
            human = fixture.stdout.getvalue().lower()
            summaries = [
                line for line in human.splitlines() if "summary" in line
            ]
            events = fixture.load_events()

            self.assertGreaterEqual(len(summaries), 1)
            self.assertTrue(
                all("remaining_cost_est" in line for line in summaries)
            )
            for forbidden in (
                "accrued",
                "spent",
                "final cost",
                "per-sample cost",
                "estimated_total",
            ):
                self.assertNotIn(forbidden, human)
            summary_events = [
                row for row in events if row["event"] == "summary"
            ]
            self.assertGreaterEqual(len(summary_events), 1)
            self.assertEqual(summary_events[0]["completed_samples"], 25)
            self.assertEqual(
                set(summary_events[0]["remaining_cost_est"]), {"USD", "CNY"}
            )

    def test_resume_prediction_flushes_are_nonshrinking_atomic_and_lifecycle_ordered(self):
        with FullRunFixture(sample_count=120) as fixture:
            fixture.seed_predictions(count=100)
            fixture.seed_complete_checkpoint(sample_id="101")

            fixture.run(resume=True)
            snapshots = fixture.prediction_snapshots
            events = fixture.load_events()

            self.assertEqual(fixture.prediction_read_count, 1)
            self.assertGreaterEqual(len(snapshots), 5)
            lengths = [len(snapshot) for snapshot in snapshots]
            self.assertEqual(lengths[:5], [100, 100, 100, 100, 120])
            self.assertEqual(lengths, sorted(lengths))
            for snapshot in snapshots:
                ids = [row["sample_id"] for row in snapshot]
                self.assertEqual(ids, [str(index) for index in range(1, len(ids) + 1)])
                self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(
                all("fixture_seed" not in row for row in snapshots[0][:25])
            )
            self.assertTrue(
                all(row.get("fixture_seed") for row in snapshots[0][25:])
            )

            audit = next(row for row in events if row["event"] == "resume_audit")
            self.assertEqual(audit["complete_samples"], 1)
            self.assertEqual(fixture.calls_for("101"), [])
            names = [row["event"] for row in events]
            self.assertEqual(
                names.count("prediction_flushed"), len(snapshots)
            )
            self.assertLess(names.index("metrics_written"), names.index("run_completed"))
            self.assertEqual(names[-1], "run_completed")
            self.assertTrue(fixture.artifact_checks)
            self.assertTrue(all(present for _, present in fixture.artifact_checks))

    def test_fake_resume_emits_audit_reuses_stages_and_never_reads_v1(self):
        with FullRunFixture(
            sample_count=3,
            output_name="scammark_qwen3_dashscope_server_full_v2",
        ) as fixture:
            fixture.seed_complete_checkpoint(sample_id="1")
            fixture.seed_partial_checkpoint(
                sample_id="2",
                complete_stages=("url", "text", "visual", "cross"),
            )
            fixture.create_v1_sentinel_that_raises_on_open()

            result = fixture.run(resume=True)
            events = fixture.load_events()

            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(events[0]["event"], "run_started")
            audit = next(row for row in events if row["event"] == "resume_audit")
            first_reservation = next(
                index
                for index, row in enumerate(events)
                if row["event"] == "api_call_reserved"
            )
            self.assertLess(events.index(audit), first_reservation)
            self.assertEqual(audit["complete_samples"], 1)
            self.assertEqual(audit["partial_samples"], 1)
            self.assertEqual(audit["complete_stages"], 11)
            self.assertEqual(audit["reserved"], 11)
            self.assertEqual(audit["finished"], 0)
            self.assertGreaterEqual(
                sum(row["event"] == "stage_reused" for row in events), 11
            )
            self.assertEqual(fixture.calls_for("1"), [])
            self.assertEqual(
                fixture.calls_for("2"), ["binary", "type", "grounding"]
            )
            self.assertEqual(fixture.socket_attempt_count, 0)
            self.assertEqual(fixture.v1_open_count, 0)
    def test_merge_prediction_records_never_truncates_prior_resume_progress(self):
        from scammark.run_experiment import _merge_prediction_records

        manifest_ids = [str(index) for index in range(1, 121)]
        existing = [
            {"sample_id": str(index), "binary": "benign"}
            for index in range(1, 101)
        ]
        updates = [
            {"sample_id": str(index), "binary": "scam"}
            for index in range(1, 26)
        ]

        merged = _merge_prediction_records(existing, updates, manifest_ids)

        self.assertEqual(len(merged), 100)
        self.assertEqual(
            [row["sample_id"] for row in merged], manifest_ids[:100]
        )
        self.assertTrue(
            all(row["binary"] == "scam" for row in merged[:25])
        )
        self.assertTrue(
            all(row["binary"] == "benign" for row in merged[25:])
        )

    def test_merge_rejects_duplicates_or_ids_outside_manifest(self):
        from scammark.experiment_state import ExperimentContractError
        from scammark.run_experiment import _merge_prediction_records

        with self.assertRaises(ExperimentContractError):
            _merge_prediction_records(
                [{"sample_id": "1"}, {"sample_id": "1"}], [], ["1"]
            )
        with self.assertRaises(ExperimentContractError):
            _merge_prediction_records(
                [{"sample_id": "outside"}], [], ["1"]
            )

    def test_merge_rejects_duplicate_or_empty_manifest_ids(self):
        from scammark.experiment_state import ExperimentContractError
        from scammark.run_experiment import _merge_prediction_records

        for manifest_ids in (["1", "1"], ["1", ""]):
            with self.subTest(manifest_ids=manifest_ids), self.assertRaises(
                ExperimentContractError
            ):
                _merge_prediction_records([], [], manifest_ids)

    def test_merge_rejects_non_mapping_rows_and_duplicate_updates(self):
        from scammark.experiment_state import ExperimentContractError
        from scammark.run_experiment import _merge_prediction_records

        with self.assertRaises(ExperimentContractError):
            _merge_prediction_records(["not-a-mapping"], [], ["1"])
        with self.assertRaises(ExperimentContractError):
            _merge_prediction_records(
                [], [{"sample_id": "1"}, {"id": "1"}], ["1"]
            )

    def test_merge_rejects_hostile_ids_with_bounded_sanitized_error(self):
        from scammark.experiment_state import ExperimentContractError
        from scammark.run_experiment import _merge_prediction_records

        hostile = (
            "Bearer sk-SENTINEL-12345678 "
            "https://workspace-private.cn-beijing.maas.aliyuncs.com/path?token=secret "
            + "x" * 2_000
        )
        with self.assertRaises(ExperimentContractError) as caught:
            _merge_prediction_records([{"sample_id": hostile}], [], ["1"])

        message = str(caught.exception)
        self.assertLessEqual(len(message), 300)
        for forbidden in (
            "sk-SENTINEL-12345678",
            "workspace-private",
            "token=secret",
        ):
            self.assertNotIn(forbidden, message)

    def test_merge_returns_copies_in_manifest_order_without_mutating_inputs(self):
        from scammark.run_experiment import _merge_prediction_records

        existing = [{"sample_id": "2", "binary": "benign"}]
        updates = [{"sample_id": "1", "binary": "scam"}]
        existing_before = json.loads(json.dumps(existing))
        updates_before = json.loads(json.dumps(updates))

        merged = _merge_prediction_records(existing, updates, ["1", "2"])

        self.assertEqual([row["sample_id"] for row in merged], ["1", "2"])
        self.assertEqual(existing, existing_before)
        self.assertEqual(updates, updates_before)
        self.assertIsNot(merged[0], updates[0])
        self.assertIsNot(merged[1], existing[0])

    def test_implementation_source_hashes_cover_every_v2_runtime_module(self):
        from scammark.run_experiment import _implementation_source_hashes

        expected = (
            "run_experiment.py",
            "pipeline.py",
            "model_client.py",
            "experiment_state.py",
            "input_budget.py",
            "image_preprocessing.py",
            "mhtml_policy.py",
            "prompts.py",
            "prediction_schema.py",
            "reliability.py",
            "metrics.py",
            "evaluate.py",
            "observability.py",
            "run_status.py",
            "result_contract.py",
        )
        hashes = _implementation_source_hashes()

        self.assertEqual(tuple(hashes), expected)
        for name, digest in hashes.items():
            with self.subTest(name=name):
                path = SRC / "scammark" / name
                self.assertTrue(path.is_file())
                self.assertEqual(
                    digest,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )

    def test_workspace_endpoint_identity_is_hashed_and_redacted_from_artifacts(self):
        from scammark.model_client import endpoint_identity
        from scammark.run_experiment import sanitized_contract_config

        workspace = "ws-private-identifier"
        endpoint = (
            f"https://{workspace}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        )
        identity = endpoint_identity(endpoint)

        self.assertEqual(identity["host"], "*.cn-beijing.maas.aliyuncs.com")
        self.assertEqual(len(identity["sha256"]), 64)
        self.assertNotIn(workspace, json.dumps(identity))

        config = {
            "endpoints": {
                "text_base_url": "https://cn.crazyrouter.com/v1",
                "vision_base_url": endpoint,
                "text_provider": "Crazyrouter",
                "vision_provider": "Alibaba Cloud Model Studio",
            },
            "experiment": {"name": "redaction-test"},
        }
        sanitized = sanitized_contract_config(config)
        serialized = json.dumps(sanitized)
        self.assertNotIn(workspace, serialized)
        self.assertEqual(
            sanitized["endpoints"]["vision_base_url"],
            "https://*.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(
            sanitized["endpoints"]["vision_endpoint_sha256"],
            identity["sha256"],
        )

    def test_frozen_asset_verifier_checks_internal_hashes_and_counts(self):
        from scammark.frozen_assets import sha256_file, verify_frozen_assets

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = root / "scamweb_v_manifest.jsonl"
            evidence = root / "web_evidence.jsonl"
            manifest.write_text('{"sample_id":"1"}\n', encoding="utf-8")
            evidence.write_text('{"sample_id":"1"}\n', encoding="utf-8")
            (root / "dataset_version.json").write_text(
                json.dumps(
                    {
                        "dataset_version": "test-v1",
                        "sample_count": 1,
                        "sha256": {
                            "frozen_manifest": sha256_file(manifest),
                            "frozen_evidence": sha256_file(evidence),
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = verify_frozen_assets(root)
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["dataset_version"], "test-v1")

            evidence.write_text('{"sample_id":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "web_evidence.jsonl SHA256 mismatch"):
                verify_frozen_assets(root)

    def test_dataset_path_mapper_relocates_windows_paths_to_linux(self):
        from scammark.dataset_paths import DatasetPathMapper

        mapper = DatasetPathMapper(
            source_root="D:/datasets/scamweb_dataset",
            runtime_root="/data/scamweb_dataset",
        )
        record = mapper.relocate_record(
            {
                "sample_id": "1",
                "image_path": r"D:\datasets\scamweb_dataset\dataset_scam_gold\截图_scam\1.png",
                "mhtml_path": r"D:\datasets\scamweb_dataset\dataset_scam_gold\mhtml_scam\1.mhtml",
                "ai_json_path": "",
            }
        )

        self.assertEqual(
            record["image_path"],
            "/data/scamweb_dataset/dataset_scam_gold/截图_scam/1.png",
        )
        self.assertEqual(
            record["mhtml_path"],
            "/data/scamweb_dataset/dataset_scam_gold/mhtml_scam/1.mhtml",
        )
        self.assertEqual(record["ai_json_path"], "")
        self.assertEqual(record["sample_id"], "1")

    def test_dataset_path_mapper_rejects_unexpected_absolute_source(self):
        from scammark.dataset_paths import DatasetPathMapper

        mapper = DatasetPathMapper(
            source_root="D:/datasets/scamweb_dataset",
            runtime_root="/data/scamweb_dataset",
        )

        with self.assertRaisesRegex(ValueError, "outside the frozen source root"):
            mapper.relocate_record({"image_path": r"E:\other\page.png"})

    def test_budgeted_prompt_retains_priority_fields_and_caps_tokens(self):
        from scammark.input_budget import build_budgeted_stage_prompt, count_tokens

        sample = {
            "url": "https://secure.example/login",
            "title": "Account Verification",
            "mhtml_audit_status": "valid",
            "visible_text": "verify account now " * 4000,
            "meta": {"huge": "x" * 1_700_000, "description": "official account verification"},
            "forms": [{"fields": ["password", "card"]}] * 100,
            "links": [f"https://redirect.example/{index}" for index in range(200)],
        }
        prompt, metadata = build_budgeted_stage_prompt("text", sample, {}, max_tokens=2048)

        self.assertLessEqual(count_tokens(prompt), 2048)
        self.assertNotIn("https://secure.example/login", prompt)
        self.assertIn("Account Verification", prompt)
        self.assertTrue(metadata["truncated"])
        self.assertLess(len(prompt), 100_000)

    def test_mhtml_policy_fields_never_enter_model_prompts(self):
        from scammark.prompts import build_stage_prompt

        prompt = build_stage_prompt(
            "text",
            {
                "url": "https://example.test",
                "title": "Example",
                "mhtml_audit_status": "mhtml_mismatch",
                "mhtml_used": False,
            },
            {},
        )

        self.assertNotIn("mhtml_audit_status", prompt)
        self.assertNotIn("mhtml_mismatch", prompt)
        self.assertNotIn("mhtml_used", prompt)

    def test_budgeted_prompt_preserves_every_image_geometry_record(self):
        from scammark.input_budget import build_budgeted_stage_prompt

        image_inputs = [
            {
                "image_index": index,
                "kind": "overview" if index == 0 else "tile",
                "crop_xyxy_in_original": [0, index * 1000, 1200, (index + 1) * 1000],
                "original_size": [1200, 8000],
                "submitted_size": [600, 500],
            }
            for index in range(8)
        ]
        prompt, metadata = build_budgeted_stage_prompt(
            "grounding",
            {"url": "https://example.test", "title": "Example", "visible_text": "x " * 20000},
            {
                "image_inputs": image_inputs,
                "prompt_stages": {"visual": {"evidence_summary": "risk " * 20000}},
            },
            max_tokens=1800,
        )

        self.assertFalse(metadata["truncated"])
        for index in range(8):
            self.assertIn(f'"image_index": {index}', prompt)
        self.assertIn('"crop_xyxy_in_original"', prompt)

    def test_long_screenshot_bundle_stays_within_budget_and_maps_tile_box(self):
        from PIL import Image

        from scammark.image_preprocessing import ImagePolicy, map_bbox_1000, prepare_image_bundle

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "long.png"
            Image.new("RGB", (800, 8000), "white").save(source)
            policy = ImagePolicy(
                overview_pixels=200_000,
                total_pixels=1_000_000,
                max_long_side=3000,
                max_tiles=4,
                overlap_ratio=0.10,
                max_base64_bytes=1_000_000,
            )
            bundle = prepare_image_bundle(source, root / "cache", policy)

            self.assertGreater(len(bundle.assets), 1)
            self.assertLessEqual(sum(asset.width * asset.height for asset in bundle.assets), 1_000_000)
            tile = bundle.assets[1]
            mapped = map_bbox_1000(tile, [0, 0, 1000, 1000])
            self.assertAlmostEqual(mapped[0], tile.crop_xyxy[0], delta=1.0)
            self.assertAlmostEqual(mapped[1], tile.crop_xyxy[1], delta=1.0)
            self.assertAlmostEqual(mapped[2], tile.crop_xyxy[2], delta=1.0)
            self.assertAlmostEqual(mapped[3], tile.crop_xyxy[3], delta=1.0)
            self.assertTrue(all(asset.data_url_size <= 1_000_000 for asset in bundle.assets))

    def test_regular_screenshot_uses_overview_not_full_visual_budget(self):
        from PIL import Image

        from scammark.image_preprocessing import ImagePolicy, prepare_image_bundle

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "regular.png"
            Image.new("RGB", (2000, 3000), "white").save(source)
            bundle = prepare_image_bundle(
                source,
                root / "cache",
                ImagePolicy(overview_pixels=1_000_000, total_pixels=8_000_000),
            )

        self.assertEqual(len(bundle.assets), 1)
        self.assertLessEqual(bundle.assets[0].width * bundle.assets[0].height, 1_000_000)

    def test_long_bundle_submits_one_contact_sheet_and_keeps_logical_tile_indexes(self):
        from PIL import Image

        from scammark.image_preprocessing import ImagePolicy, assets_for_stage, prepare_image_bundle
        from scammark.model_client import _image_asset_indexes
        from scammark.pipeline import ScamMarkPipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "long-contact.png"
            Image.new("RGB", (800, 10000), "white").save(source)
            policy = ImagePolicy(
                overview_pixels=200_000,
                total_pixels=1_000_000,
                contact_sheet_pixels=800_000,
                contact_sheet_base64_bytes=500_000,
                max_long_side=3000,
                max_tiles=4,
            )
            bundle = prepare_image_bundle(source, root / "cache", policy)
            submitted = assets_for_stage(bundle, "visual")
            prompt_context = ScamMarkPipeline._bundle_prompt_context(bundle, "visual")

        self.assertGreater(len(bundle.assets), 1)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0].kind, "contact_sheet")
        self.assertEqual(_image_asset_indexes(submitted), {asset.index for asset in bundle.assets})
        self.assertLessEqual(submitted[0].width * submitted[0].height, 800_000)
        self.assertLessEqual(submitted[0].data_url_size, 500_000)
        self.assertEqual({item["image_index"] for item in prompt_context}, set(range(len(bundle.assets))))
        self.assertTrue(all(item["submission_mode"] == "contact_sheet" for item in prompt_context))
        self.assertTrue(all(len(item["contact_sheet_panel_xyxy"]) == 4 for item in prompt_context))

    def test_grounding_prompt_requests_tile_relative_normalized_boxes(self):
        from scammark.prompts import build_stage_prompt

        prompt = build_stage_prompt(
            "grounding",
            {"url": "https://example.test", "title": "Example"},
            {
                "image_inputs": [
                    {
                        "image_index": 1,
                        "kind": "tile",
                        "crop_xyxy_in_original": [0, 1000, 800, 3000],
                    }
                ]
            },
        )

        self.assertIn("bbox_1000", prompt)
        self.assertIn("image_index", prompt)
        self.assertIn("mapped back", prompt)

    def test_prompts_bound_json_strings_and_explain_contact_sheet_coordinates(self):
        from scammark.prompts import build_stage_prompt

        prompt = build_stage_prompt(
            "grounding",
            {"url": "https://example.test"},
            {
                "image_inputs": [
                    {
                        "image_index": 1,
                        "submission_mode": "contact_sheet",
                        "contact_sheet_panel_xyxy": [10, 20, 200, 400],
                    }
                ]
            },
        )

        self.assertIn("escape internal double quotes", prompt)
        self.assertIn("contact sheet", prompt.lower())
        self.assertIn("within the selected panel", prompt)

    def test_url_and_text_analyzers_receive_only_their_own_modalities(self):
        from scammark.prompts import build_stage_prompt

        raw_page_marker = "RAW_PAGE_TEXT_MUST_NOT_REACH_URL"
        prior_url_marker = "PRIOR_URL_OUTPUT_MUST_NOT_REACH_TEXT"
        sample = {
            "url": "https://example.test/login",
            "title": "Account notice",
            "visible_text": raw_page_marker,
            "forms": [{"fields": ["password"]}],
        }
        previous = {
            "prompt_stages": {
                "url": {
                    "risk_level": "high",
                    "evidence_summary": prior_url_marker,
                }
            }
        }

        url_prompt = build_stage_prompt("url", sample, previous)
        text_prompt = build_stage_prompt("text", sample, previous)

        self.assertIn("https://example.test/login", url_prompt)
        self.assertNotIn(raw_page_marker, url_prompt)
        self.assertIn(raw_page_marker, text_prompt)
        self.assertNotIn(prior_url_marker, text_prompt)

    def test_visual_prompt_uses_image_geometry_without_raw_or_prior_text(self):
        from scammark.prompts import build_stage_prompt

        raw_marker = "RAW_EXPLICIT_PAGE_QUOTE_MUST_NOT_REACH_QWEN"
        prior_marker = "PRIOR_FREE_TEXT_QUOTE_MUST_NOT_REACH_QWEN"
        prompt = build_stage_prompt(
            "visual",
            {
                "url": "https://example.test",
                "title": raw_marker,
                "visible_text": raw_marker,
                "meta": {"description": raw_marker},
                "image_size": [1881, 2434],
            },
            {
                "heuristic": {"raw": prior_marker},
                "prompt_stages": {
                    "url": {"evidence_summary": prior_marker},
                    "text": {"claims": [prior_marker], "evidence_summary": prior_marker},
                },
                "image_inputs": [
                    {
                        "image_index": 0,
                        "kind": "overview",
                        "crop_xyxy_in_original": [0, 0, 1881, 2434],
                    }
                ],
            },
        )

        self.assertNotIn(raw_marker, prompt)
        self.assertNotIn(prior_marker, prompt)
        self.assertIn('"image_index": 0', prompt)
        self.assertIn('"image_size"', prompt)

    def test_downstream_qwen_prompts_use_controlled_features_not_free_text_quotes(self):
        from scammark.prompts import build_stage_prompt

        marker = "UNTRUSTED_FREE_TEXT_QUOTE_MUST_NOT_REACH_QWEN"
        sample = {
            "url": "https://example.test",
            "title": marker,
            "visible_text": marker,
            "image_size": [1000, 1200],
        }
        analyses = {
            "heuristic": {"raw": marker},
            "prompt_stages": {
                "url": {
                    "risk_level": "high",
                    "domain_observations": [marker],
                    "evidence_summary": marker,
                },
                "text": {
                    "claims": [marker],
                    "candidate_scam_types": ["Phishing", marker],
                    "evidence_summary": marker,
                },
                "visual": {
                    "page_genre": "adult_or_romance",
                    "visual_scam_cues": [marker],
                    "evidence_summary": marker,
                },
                "cross": {
                    "risk_level": "high",
                    "conflicts": [marker],
                    "evidence_summary": marker,
                },
                "binary": {"binary": "scam", "reasoning": marker},
                "type": {
                    "scam_type_id": 23,
                    "scam_type": "Phishing",
                    "reasoning": marker,
                },
            },
            "image_inputs": [{"image_index": 0, "kind": "overview"}],
        }

        for stage in ("cross", "binary", "type", "grounding"):
            with self.subTest(stage=stage):
                prompt = build_stage_prompt(stage, sample, analyses)
                self.assertNotIn(marker, prompt)
                self.assertIn("Phishing", prompt)

    def test_type_prompt_uses_numbered_closed_set_and_removes_explicit_raw_context(self):
        from scammark.prompts import build_stage_prompt

        explicit = "verbatim explicit adult title that must not reach type"
        prompt = build_stage_prompt(
            "type",
            {"url": "https://example.test", "title": "Example 18+", "visible_text": explicit},
            {
                "heuristic": {"raw": explicit},
                "prompt_stages": {
                    "visual": {
                        "page_genre": "adult_or_romance",
                        "visual_scam_cues": [explicit],
                        "evidence_summary": "Adult lure with off-platform contact.",
                    },
                    "binary": {"binary": "scam", "reasoning": explicit},
                },
            },
        )

        self.assertIn('"scam_type_id"', prompt)
        self.assertIn('"id": 1', prompt)
        self.assertIn('"label":', prompt)
        self.assertNotIn("Adult lure with off-platform contact.", prompt)
        self.assertIn('"page_genre": "adult_or_romance"', prompt)
        self.assertIn('"visual_scam_cues_present": true', prompt)
        self.assertNotIn(explicit, prompt)

    def test_type_parser_deterministically_decodes_id_instead_of_guessing_variant(self):
        from scammark.model_client import _parse_and_validate_stage_output
        from scammark.taxonomy import CANONICAL_LABELS

        expected = "Business Impersonation Scam"
        type_id = CANONICAL_LABELS.index(expected) + 1
        result = _parse_and_validate_stage_output(
            "type",
            json.dumps(
                {
                    "scam_type_id": type_id,
                    "scam_type": "Brand Impersonation Scam",
                    "reasoning": "A brand is impersonated.",
                }
            ),
        )

        self.assertEqual(result["scam_type"], expected)
        self.assertEqual(result["reported_scam_type"], "Brand Impersonation Scam")

    def test_type_routes_to_qwen_without_attaching_the_raw_screenshot(self):
        from scammark.image_preprocessing import ImageAsset, ImageBundle, assets_for_stage

        asset = ImageAsset(
            index=0,
            kind="overview",
            path="overview.png",
            width=100,
            height=100,
            crop_xyxy=[0, 0, 100, 100],
            source_size=[100, 100],
            mime_type="image/png",
            data_url_size=100,
        )
        bundle = ImageBundle("page.png", [100, 100], [asset], {})

        self.assertEqual(assets_for_stage(bundle, "type"), [])

    def test_contact_sheet_uses_its_stricter_base64_engineering_gate(self):
        from scammark.run_experiment import _image_payload_within_limits

        policy = {
            "total_pixels": 8_388_608,
            "max_base64_bytes": 10_000_000,
            "contact_sheet_base64_bytes": 2_500_000,
        }
        accepted = {
            "total_pixels": 7_936_000,
            "assets": [{"kind": "contact_sheet", "data_url_size": 2_500_000}],
        }
        rejected = {
            "total_pixels": 7_936_000,
            "assets": [{"kind": "contact_sheet", "data_url_size": 2_500_001}],
        }

        self.assertTrue(_image_payload_within_limits(accepted, policy))
        self.assertFalse(_image_payload_within_limits(rejected, policy))

    def test_retry_policy_recovers_transient_error_and_records_attempts(self):
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise TransientRequestError("TLS EOF")
            return {"ok": True}

        result, metadata = execute_with_retry(
            operation,
            RequestPolicy(max_attempts=4, backoff_seconds=(0, 0, 0)),
            sleep_fn=lambda _: None,
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(metadata["attempt_count"], 3)
        self.assertEqual(metadata["retry_count"], 2)

    def test_retry_policy_does_not_retry_permanent_error(self):
        from scammark.reliability import PermanentRequestError, RequestPolicy, execute_with_retry

        calls = []

        def operation():
            calls.append(1)
            raise PermanentRequestError("HTTP 404")

        with self.assertRaises(PermanentRequestError) as raised:
            execute_with_retry(
                operation,
                RequestPolicy(max_attempts=4, backoff_seconds=(0, 0, 0)),
                sleep_fn=lambda _: None,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(raised.exception.request_metadata["attempt_count"], 1)

    def test_retry_policy_respects_retry_after(self):
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        calls = []
        sleeps = []

        def operation():
            calls.append(1)
            if len(calls) == 1:
                raise TransientRequestError("rate limited", retry_after_seconds=7.0)
            return "ok"

        result, _ = execute_with_retry(
            operation,
            RequestPolicy(max_attempts=2, backoff_seconds=(2,), full_jitter=False),
            sleep_fn=sleeps.append,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [7.0])

    def test_retry_callback_preserves_attempt_count_and_exact_delays(self):
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        attempts = []
        sleeps = []
        events = []

        def operation():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 4:
                raise TransientRequestError("HTTP 503: temporary")
            return "ok"

        result, metadata = execute_with_retry(
            operation,
            RequestPolicy(
                max_attempts=4,
                backoff_seconds=(2, 4, 8),
                full_jitter=False,
            ),
            sleep_fn=sleeps.append,
            event_callback=lambda event, payload: events.append((event, payload)),
        )

        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [2, 4, 8])
        self.assertEqual(metadata["attempt_count"], 4)
        self.assertEqual([name for name, _ in events], ["retry_scheduled"] * 3)
        self.assertEqual(
            [(payload["attempt"], payload["next_attempt"], payload["delay_seconds"]) for _, payload in events],
            [(1, 2, 2), (2, 3, 4), (3, 4, 8)],
        )

    def test_retry_callback_failure_cannot_change_attempts_sleep_or_result(self):
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        attempts = []
        sleeps = []

        def operation():
            attempts.append(len(attempts) + 1)
            if len(attempts) == 1:
                raise TransientRequestError("temporary")
            return "provider-result"

        def broken_callback(event, payload):
            payload["delay_seconds"] = 999
            raise OSError("event sink unavailable")

        result, metadata = execute_with_retry(
            operation,
            RequestPolicy(max_attempts=2, backoff_seconds=(3,), full_jitter=False),
            sleep_fn=sleeps.append,
            event_callback=broken_callback,
        )

        self.assertEqual(result, "provider-result")
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(sleeps, [3])
        self.assertEqual(metadata["attempt_count"], 2)

    def test_retry_callback_and_returned_attempt_errors_redact_provider_details(self):
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        secret = "sk-SENTINEL-12345678"
        provider_detail = (
            f"HTTP 503: Bearer {secret} PROMPT_SENTINEL RAW_RESPONSE_SENTINEL "
            "data:image/png;base64,AAAA "
            "https://workspace-private.cn-beijing.maas.aliyuncs.com/p?token=x"
        )
        calls = []
        events = []

        def operation():
            calls.append(1)
            if len(calls) == 1:
                raise TransientRequestError(provider_detail)
            return "ok"

        result, metadata = execute_with_retry(
            operation,
            RequestPolicy(max_attempts=2, backoff_seconds=(0,), full_jitter=False),
            sleep_fn=lambda _: None,
            event_callback=lambda event, payload: events.append((event, payload)),
        )
        serialized = json.dumps(
            {"events": events, "attempt_errors": metadata["attempt_errors"]},
            ensure_ascii=False,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(metadata["attempt_errors"][0]["error_type"], "TransientRequestError")
        for forbidden in (
            secret,
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "AAAA",
            "token=x",
            "workspace-private",
            "/p",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_retry_callback_preserves_jitter_call_and_retry_after_precedence(self):
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        calls = []
        random_calls = []
        sleeps = []
        events = []

        def operation():
            calls.append(1)
            if len(calls) == 1:
                raise TransientRequestError("rate limited", retry_after_seconds=7.0)
            return "ok"

        def random_fn(lower, upper):
            random_calls.append((lower, upper))
            return 1.5

        result, _ = execute_with_retry(
            operation,
            RequestPolicy(max_attempts=2, backoff_seconds=(8,), full_jitter=True),
            sleep_fn=sleeps.append,
            random_fn=random_fn,
            event_callback=lambda event, payload: events.append((event, dict(payload))),
        )

        self.assertEqual(result, "ok")
        self.assertEqual(random_calls, [(0.0, 8)])
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(events[0][1]["delay_seconds"], 7.0)

    def test_unprintable_retry_exception_keeps_original_exception_and_safe_metadata(self):
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        class UnprintableTransient(TransientRequestError):
            def __str__(self):
                raise RuntimeError("PROMPT_SENTINEL")

        error = UnprintableTransient("private")
        with self.assertRaises(UnprintableTransient) as raised:
            execute_with_retry(
                lambda: (_ for _ in ()).throw(error),
                RequestPolicy(max_attempts=1, backoff_seconds=()),
            )

        self.assertIs(raised.exception, error)
        self.assertEqual(
            raised.exception.request_metadata["attempt_errors"],
            [
                {
                    "attempt": 1,
                    "error_type": "TransientRequestError",
                    "message": "transient provider or transport failure",
                }
            ],
        )

    def test_mhtml_policy_hides_untrusted_evidence(self):
        from scammark.mhtml_policy import apply_mhtml_policy

        sample = {
            "sample_id": "demo",
            "url": "https://example.test",
            "title": "Example",
            "visible_text": "unrelated archived page",
            "links": ["https://wrong.test"],
            "mhtml_audit_status": "mhtml_mismatch",
        }
        filtered = apply_mhtml_policy(sample)

        self.assertEqual(filtered["url"], sample["url"])
        self.assertEqual(filtered["title"], sample["title"])
        self.assertNotIn("visible_text", filtered)
        self.assertNotIn("links", filtered)
        self.assertFalse(filtered["mhtml_used"])

    def test_pipeline_heuristics_cannot_reintroduce_untrusted_mhtml_text(self):
        from scammark.pipeline import ScamMarkPipeline

        class PromptCaptureClient:
            def __init__(self):
                self.prompts = []

            def complete_json(self, stage, prompt, sample, include_image=False):
                self.prompts.append(prompt)
                if stage == "binary":
                    return {"binary": "benign", "reasoning": "no trusted evidence"}
                return {"evidence_summary": "none"}

        client = PromptCaptureClient()
        ScamMarkPipeline(model_client=client).predict(
            {
                "sample_id": "untrusted-1",
                "url": "https://example.test",
                "title": "Example",
                "mhtml_audit_status": "mhtml_mismatch",
                "visible_text": "UNTRUSTED_ARCHIVE_MARKER guaranteed crypto profit login",
                "buttons": ["UNTRUSTED_ARCHIVE_MARKER"],
            }
        )

        self.assertNotIn("UNTRUSTED_ARCHIVE_MARKER", "\n".join(client.prompts))

    def test_mhtml_audit_rules_are_conservative(self):
        from scammark.audit_mhtml import classify_mhtml_record

        manifest = {"sample_id": "one", "url": "https://shop.example", "title": "Example Shop"}
        valid = classify_mhtml_record(
            manifest,
            {
                "parse_status": "ok",
                "url_domain_observed": True,
                "archive_url": "https://shop.example/home",
                "title": "Example Shop",
                "visible_text": "Example Shop products",
            },
        )
        redirect = classify_mhtml_record(
            manifest,
            {
                "parse_status": "ok",
                "url_domain_observed": False,
                "archive_url": "https://example.com/store",
                "title": "Example Shop",
                "visible_text": "Example Shop products",
            },
        )
        mismatch = classify_mhtml_record(
            manifest,
            {
                "parse_status": "ok",
                "url_domain_observed": False,
                "archive_url": "https://parking.invalid",
                "title": "Domain for sale",
                "visible_text": "This domain is parked and for sale",
            },
        )

        self.assertEqual(valid["status"], "valid")
        self.assertEqual(redirect["status"], "valid_redirect")
        self.assertEqual(mismatch["status"], "mhtml_mismatch")

    def test_accepted_recapture_replaces_stale_mhtml_evidence(self):
        from scammark.audit_mhtml import load_recaptured_evidence

        with tempfile.TemporaryDirectory() as tmp_dir:
            mhtml_path = Path(tmp_dir) / "recapture.mhtml"
            mhtml_path.write_text(
                "MIME-Version: 1.0\n"
                "Content-Type: multipart/related; boundary=demo\n\n"
                "--demo\n"
                "Content-Type: text/html; charset=utf-8\n"
                "Content-Location: https://new.example/home\n\n"
                "<html><head><title>Fresh Capture</title></head>"
                "<body><form><input name='password'></form>Fresh page text</body></html>\n"
                "--demo--\n",
                encoding="utf-8",
            )

            evidence = load_recaptured_evidence(
                {"mhtml_path": str(mhtml_path), "final_url": "https://new.example/home"},
                {"url": "https://new.example/home", "binary": "benign", "raw_id": "1"},
                "sample-1",
            )

        self.assertEqual(evidence["parse_status"], "ok")
        self.assertEqual(evidence["title"], "Fresh Capture")
        self.assertIn("Fresh page text", evidence["visible_text"])
        self.assertEqual(evidence["sample_id"], "sample-1")

    def test_frozen_box_rows_are_derived_deterministically(self):
        from scammark.audit_mhtml import build_box_rows

        rows = build_box_rows(
            [
                {
                    "sample_id": "one",
                    "binary": "scam",
                    "scam_type": "Phishing",
                    "boxes": [
                        {"xyxy": [1, 2, 3, 4], "reason": "login form", "source": "human_json"}
                    ],
                },
                {"sample_id": "two", "binary": "benign", "boxes": []},
            ]
        )

        self.assertEqual(rows, [{
            "sample_id": "one",
            "binary": "scam",
            "box_id": 1,
            "xyxy": [1.0, 2.0, 3.0, 4.0],
            "reason": "login form",
            "box_label": "Phishing",
            "source": "human_json",
        }])

    def test_qwen_mhtml_audit_separates_identity_from_gold_conflict(self):
        from PIL import Image

        from scammark.audit_mhtml import _qwen_audit

        class AuditClient:
            def __init__(self):
                self.requests = []

            def complete_json_with_metadata(self, **kwargs):
                self.requests.append(kwargs)
                if len(self.requests) == 1:
                    return {
                        "status": "uncertain",
                        "confidence": 0.5,
                        "reasons": ["ambiguous identity"],
                    }, {"attempt_count": 1, "usage": {"prompt_tokens": 10}}
                return {
                    "gold_input_conflict": False,
                    "confidence": 0.7,
                    "reasons": ["annotation agrees with screenshot"],
                }, {"attempt_count": 1, "usage": {"prompt_tokens": 8}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "page.png"
            Image.new("RGB", (800, 6000), "white").save(image_path)
            client = AuditClient()
            result = _qwen_audit(
                {
                    "sample_id": "audit-1",
                    "url": "https://example.test",
                    "image_path": str(image_path),
                    "binary": "scam",
                    "scam_type": "Phishing",
                    "boxes": [{"xyxy": [1, 2, 3, 4], "reason": "login"}],
                },
                {"archive_url": "https://other.test", "title": "Other", "visible_text": "text"},
                client,
                Path(tmp_dir) / "cache",
            )

        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(client.requests[0]["image_assets"]), 1)
        self.assertEqual(client.requests[0]["max_output_tokens"], 512)
        self.assertNotIn("Phishing", client.requests[0]["prompt"])
        self.assertNotIn("gold", client.requests[0]["prompt"].lower())
        self.assertIn("Phishing", client.requests[1]["prompt"])
        self.assertFalse(result["gold_input_conflict"])
        self.assertEqual(result["request_metadata"]["identity"]["attempt_count"], 1)

    def test_gold_conflict_audit_failure_does_not_discard_identity_status(self):
        from PIL import Image

        from scammark.audit_mhtml import _qwen_audit

        class AuditClient:
            def __init__(self):
                self.calls = 0

            def complete_json_with_metadata(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "status": "valid_redirect",
                        "confidence": 0.9,
                        "reasons": ["same page after redirect"],
                    }, {"attempt_count": 1}
                raise RuntimeError("conflict endpoint unavailable")

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "page.png"
            Image.new("RGB", (300, 300), "white").save(image_path)
            result = _qwen_audit(
                {
                    "sample_id": "audit-2",
                    "binary": "benign",
                    "url": "https://example.test",
                    "image_path": str(image_path),
                },
                {"archive_url": "https://redirect.test", "title": "Example", "visible_text": "text"},
                AuditClient(),
                Path(tmp_dir) / "cache",
            )

        self.assertEqual(result["status"], "valid_redirect")
        self.assertFalse(result["gold_input_conflict"])
        self.assertEqual(result["gold_conflict_check_status"], "error")
        self.assertEqual(result["request_metadata"]["gold_conflict"]["status"], "error")

    def test_checkpoint_refuses_config_drift_and_resumes_completed_stages(self):
        from scammark.experiment_state import ExperimentState

        with tempfile.TemporaryDirectory() as tmp_dir:
            state = ExperimentState(Path(tmp_dir), {"model": "qwen3-vl-flash", "temperature": 0})
            state.save_stage("sample-1", "url", {"risk_level": "low"}, {"attempt_count": 1})

            resumed = ExperimentState(Path(tmp_dir), {"model": "qwen3-vl-flash", "temperature": 0})
            self.assertEqual(resumed.load_stages("sample-1")["url"]["output"]["risk_level"], "low")
            with self.assertRaises(ValueError):
                ExperimentState(Path(tmp_dir), {"model": "gpt-4o", "temperature": 0})

    def test_api_call_ledger_counts_reservations_across_crashes(self):
        from scammark.experiment_state import ExperimentState

        with tempfile.TemporaryDirectory() as tmp_dir:
            state = ExperimentState(Path(tmp_dir), {"model": "qwen3-vl-flash"})
            call_id = state.reserve_api_call(
                {"stage": "visual", "request_fingerprint": "abc"},
                max_calls=2,
            )
            self.assertEqual(state.total_api_calls(), 1)

            resumed = ExperimentState(Path(tmp_dir), {"model": "qwen3-vl-flash"})
            resumed.finish_api_call(call_id, status="transport_error", metadata={"error": "disconnect"})
            resumed.reserve_api_call({"stage": "visual", "request_fingerprint": "def"}, max_calls=2)
            self.assertEqual(resumed.total_api_calls(), 2)
            with self.assertRaises(RuntimeError):
                resumed.reserve_api_call({"stage": "visual"}, max_calls=2)

    def test_call_ledger_audit_counts_orphans_replay_and_corrupt_tail(self):
        from scammark.experiment_state import ExperimentState, scan_call_ledger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ExperimentState(root, {"config": "v2"})
            done = state.reserve_api_call({"sample_id": "1", "stage": "url"}, max_calls=10)
            state.finish_api_call(done, status="completed")
            state.reserve_api_call({"sample_id": "2", "stage": "text"}, max_calls=10)
            with state.call_ledger.open("a", encoding="utf-8") as handle:
                handle.write('{"event":"reserved"')

            audit = scan_call_ledger(
                state.call_ledger,
                config_hash=state.config_hash,
                checkpoint_stages=frozenset(),
            )

        self.assertEqual(audit.reserved, 3)  # corrupt tail is one conservative call
        self.assertEqual(audit.finished, 1)
        self.assertEqual(audit.outstanding, 2)
        self.assertEqual(audit.outstanding_by_stage, {"text": 1, "unknown": 1})
        self.assertEqual(audit.finished_without_checkpoint, {("1", "url")})
        self.assertTrue(audit.corrupt)

    def test_ledger_append_separates_truncated_tail_from_new_reservation(self):
        from scammark.experiment_state import ExperimentState

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "v2"})
            truncated = '{"event":"reserved"'
            state.call_ledger.write_text(truncated, encoding="utf-8")

            state.reserve_api_call(
                {"sample_id": "new", "stage": "url"},
                max_calls=10,
            )

            rows = state.call_ledger.read_text(encoding="utf-8").splitlines()
            audit = state.api_call_audit()

        self.assertEqual(rows[0], truncated)
        self.assertEqual(len(rows), 2)
        self.assertEqual((audit.reserved, audit.outstanding), (2, 2))
        self.assertTrue(audit.corrupt)

    def test_concurrent_reservations_enforce_cap_in_one_locked_transaction(self):
        import threading

        from scammark.experiment_state import ApiCallLimitError, ExperimentState

        for separate_instances in (False, True):
            for attempt in range(20):
                with self.subTest(
                    separate_instances=separate_instances,
                    attempt=attempt,
                ), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    first = ExperimentState(root, {"config": "concurrent-cap"})
                    second = (
                        ExperimentState(root, {"config": "concurrent-cap"})
                        if separate_instances
                        else first
                    )
                    states = (first, second)
                    barrier = threading.Barrier(3)
                    results: list[object] = []
                    results_lock = threading.Lock()

                    def reserve(index: int) -> None:
                        barrier.wait()
                        try:
                            states[index].reserve_api_call(
                                {"sample_id": str(index), "stage": "url"},
                                max_calls=1,
                            )
                            result: object = "reserved"
                        except ApiCallLimitError:
                            result = "limited"
                        except Exception as error:  # pragma: no cover - assertion reports it
                            result = error
                        with results_lock:
                            results.append(result)

                    threads = [
                        threading.Thread(target=reserve, args=(index,))
                        for index in range(2)
                    ]
                    for thread in threads:
                        thread.start()
                    barrier.wait()
                    for thread in threads:
                        thread.join(timeout=5)

                    self.assertFalse(any(thread.is_alive() for thread in threads))
                    self.assertEqual(sorted(str(result) for result in results), ["limited", "reserved"])
                    self.assertEqual(first.api_call_audit().reserved, 1)

    def test_call_ledger_duplicate_reservations_remain_distinct_evidence(self):
        from scammark.experiment_state import scan_call_ledger

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            config_hash = "config-v2"
            rows = (
                {
                    "event": "reserved",
                    "call_id": "duplicate",
                    "config_hash": config_hash,
                    "context": {"sample_id": "1", "stage": "url"},
                },
                {
                    "event": "reserved",
                    "call_id": "duplicate",
                    "config_hash": config_hash,
                    "context": {"sample_id": "2", "stage": "text"},
                },
                {
                    "event": "finished",
                    "call_id": "duplicate",
                    "config_hash": config_hash,
                    "status": "completed",
                },
            )
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            audit = scan_call_ledger(
                ledger,
                config_hash=config_hash,
                checkpoint_stages=frozenset(),
            )

        self.assertEqual((audit.reserved, audit.finished, audit.outstanding), (2, 1, 1))
        self.assertEqual(audit.outstanding_by_stage, {"text": 1})
        self.assertEqual(audit.finished_without_checkpoint, {("1", "url")})
        self.assertEqual(audit.replay_required_by_stage, {"url": 1})
        self.assertTrue(audit.corrupt)

    def test_call_ledger_unmatched_finish_adds_one_conservative_call(self):
        from scammark.experiment_state import scan_call_ledger

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            config_hash = "config-v2"
            ledger.write_text(
                json.dumps(
                    {
                        "event": "finished",
                        "call_id": "missing-reservation",
                        "config_hash": config_hash,
                        "status": "completed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            audit = scan_call_ledger(
                ledger,
                config_hash=config_hash,
                checkpoint_stages=frozenset(),
            )

        self.assertEqual((audit.reserved, audit.finished, audit.completed), (1, 1, 1))
        self.assertEqual(audit.outstanding, 0)
        self.assertEqual(audit.finished_without_checkpoint, {("unknown", "unknown")})
        self.assertEqual(audit.replay_required_by_stage, {"unknown": 1})
        self.assertTrue(audit.corrupt)

    def test_call_ledger_missing_hash_counts_but_foreign_hash_is_ignored(self):
        from scammark.experiment_state import scan_call_ledger

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            config_hash = "config-v2"
            rows = (
                {"event": "reserved", "call_id": "missing-hash"},
                {
                    "event": "reserved",
                    "call_id": "foreign",
                    "config_hash": "other-config",
                    "context": {"sample_id": "foreign", "stage": "url"},
                },
                {
                    "event": "reserved",
                    "call_id": "current",
                    "config_hash": config_hash,
                    "context": {"sample_id": "1", "stage": "url"},
                },
                [],
            )
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            audit = scan_call_ledger(
                ledger,
                config_hash=config_hash,
                checkpoint_stages=frozenset(),
            )

        self.assertEqual((audit.reserved, audit.outstanding), (3, 3))
        self.assertEqual(audit.outstanding_by_stage, {"url": 1, "unknown": 2})
        self.assertTrue(audit.corrupt)

    def test_call_ledger_unknown_reservation_context_marks_corruption(self):
        from scammark.experiment_state import scan_call_ledger

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            config_hash = "config-v2"
            ledger.write_text(
                json.dumps(
                    {
                        "event": "reserved",
                        "call_id": "invalid-context",
                        "config_hash": config_hash,
                        "context": "not-a-mapping",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            audit = scan_call_ledger(
                ledger,
                config_hash=config_hash,
                checkpoint_stages=frozenset(),
            )

        self.assertEqual((audit.reserved, audit.outstanding), (1, 1))
        self.assertEqual(audit.outstanding_by_stage, {"unknown": 1})
        self.assertTrue(audit.corrupt)

    def test_durable_ledger_metadata_uses_explicit_safe_projections(self):
        from scammark.experiment_state import ExperimentState

        secret = "sk-SENTINEL-12345678"
        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "projection"})
            call_id = state.reserve_api_call(
                {
                    "sample_id": "1",
                    "stage": "url",
                    "model": "gpt-4o-mini",
                    "provider": "Crazyrouter",
                    "endpoint_host": "workspace-private.cn-beijing.maas.aliyuncs.com",
                    "endpoint_sha256": "a" * 64,
                    "request_fingerprint": "b" * 64,
                    "schema_repair": True,
                    "request_prompt": "PROMPT_SENTINEL",
                    "access_token": secret,
                    "endpoint": "https://workspace-private.example/path?token=secret",
                    "image_payload": "data:image/png;base64,BASE64SENTINEL",
                    "unknown": {"raw_response": "RAW_RESPONSE_SENTINEL"},
                },
                max_calls=10,
            )
            state.finish_api_call(
                call_id,
                status="transport_error",
                metadata={
                    "error_type": "TransientRequestError",
                    "message": f"Bearer {secret} PROMPT_SENTINEL",
                    "usage": {
                        "prompt_tokens": 10,
                        "unknown": "RAW_RESPONSE_SENTINEL",
                    },
                    "response_model": "gpt-4o-mini-2024-07-18",
                    "response_id": "provider-private",
                    "request_prompt": "PROMPT_SENTINEL",
                    "raw_response": "RAW_RESPONSE_SENTINEL",
                    "endpoint": "https://workspace-private.example/path?token=secret",
                },
            )
            rows = [
                json.loads(row)
                for row in state.call_ledger.read_text(encoding="utf-8").splitlines()
            ]
            persisted = state.call_ledger.read_text(encoding="utf-8")

        self.assertEqual(
            rows[0]["context"],
            {
                "sample_id": "1",
                "stage": "url",
                "model": "gpt-4o-mini",
                "provider": "Crazyrouter",
                "endpoint_host": "*.cn-beijing.maas.aliyuncs.com",
                "endpoint_sha256": "a" * 64,
                "request_fingerprint": "b" * 64,
                "schema_repair": True,
            },
        )
        self.assertEqual(
            rows[1]["metadata"],
            {
                "usage": {"prompt_tokens": 10},
                "response_model": "gpt-4o-mini-2024-07-18",
                "error_type": "TransientRequestError",
                "message": "<redacted-field>",
            },
        )
        for forbidden in (
            secret,
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "BASE64SENTINEL",
            "workspace-private",
            "token=secret",
            "provider-private",
            "request_prompt",
            "access_token",
            '"endpoint"',
            "image_payload",
        ):
            self.assertNotIn(forbidden, persisted)

    def test_checkpoint_metadata_projects_safe_fields_and_drops_deep_sentinels(self):
        from scammark.experiment_state import ExperimentState

        secret = "sk-SENTINEL-12345678"
        safe_metadata = {
            "stage": "url",
            "model": "gpt-4o-mini",
            "provider": "Crazyrouter",
            "attempt_count": 2,
            "retry_count": 1,
            "api_call_count": 2,
            "cumulative_attempt_count": 3,
            "cumulative_api_call_count": 3,
            "elapsed_seconds": 1.25,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "unknown": "RAW_RESPONSE_SENTINEL",
            },
            "finish_reason": "stop",
            "response_model": "gpt-4o-mini-2024-07-18",
            "response_model_matches_request": True,
            "endpoint_host": "workspace-private.cn-beijing.maas.aliyuncs.com",
            "endpoint_sha256": "a" * 64,
            "request_fingerprint": "b" * 64,
            "schema_repair_attempted": True,
            "attempt_errors": [
                {
                    "attempt": 1,
                    "error_type": "TransientRequestError",
                    "message": f"Bearer {secret} PROMPT_SENTINEL",
                    "raw_response": "RAW_RESPONSE_SENTINEL",
                }
            ],
            "schema_repair_error": "PROMPT_SENTINEL",
            "prompt": {
                "final_tokens": 100,
                "truncated": False,
                "raw": "PROMPT_SENTINEL",
            },
            "request_prompt": "PROMPT_SENTINEL",
            "access_token": secret,
            "raw_response": "RAW_RESPONSE_SENTINEL",
            "endpoint": "https://workspace-private.example/path?token=secret",
            "image": {
                "asset_count": 1,
                "total_pixels": 100_000,
                "total_base64_bytes": 200,
                "assets": [
                    {
                        "submitted_index": 0,
                        "kind": "overview",
                        "submitted_size": [100, 1000],
                        "data_url_size": 200,
                        "logical_indexes": [0],
                        "data_url": "data:image/png;base64,BASE64SENTINEL",
                    }
                ],
                "logical_inputs": [
                    {
                        "image_index": 0,
                        "kind": "overview",
                        "original_size": [100, 1000],
                        "submitted_size": [100, 1000],
                        "data_url_size": 200,
                        "path": "/private/PROMPT_SENTINEL.png",
                    }
                ],
                "data": "data:image/png;base64,BASE64SENTINEL",
            },
            "unknown": {"request_prompt": "PROMPT_SENTINEL"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "projection"})
            state.save_stage("1", "url", {"risk_level": "low"}, safe_metadata)
            state.save_stage_error(
                "1",
                "text",
                RuntimeError(f"Bearer {secret} PROMPT_SENTINEL"),
                safe_metadata,
            )
            checkpoint_path = next(state.checkpoints.glob("*.json"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            persisted = json.dumps(checkpoint, ensure_ascii=False)
            checkpoint["stages"]["url"]["request_metadata"]["request_prompt"] = (
                "PROMPT_SENTINEL"
            )
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            loaded_after_injection = state.load_stages("1")

        successful = checkpoint["stages"]["url"]["request_metadata"]
        failed = checkpoint["stages"]["text"]["request_metadata"]
        for metadata in (successful, failed):
            self.assertEqual(metadata["attempt_count"], 2)
            self.assertEqual(metadata["cumulative_api_call_count"], 3)
            self.assertEqual(
                metadata["usage"],
                {"prompt_tokens": 10, "completion_tokens": 2},
            )
            self.assertEqual(
                metadata["endpoint_host"],
                "*.cn-beijing.maas.aliyuncs.com",
            )
            self.assertEqual(metadata["request_fingerprint"], "b" * 64)
            self.assertEqual(
                metadata["attempt_errors"],
                [
                    {
                        "attempt": 1,
                        "error_type": "TransientRequestError",
                        "message": "<redacted-field>",
                    }
                ],
            )
            self.assertEqual(metadata["schema_repair_error"], "<redacted-field>")
            self.assertEqual(
                metadata["prompt"],
                {"final_tokens": 100, "truncated": False},
            )
            self.assertEqual(
                metadata["image"],
                {
                    "asset_count": 1,
                    "total_pixels": 100_000,
                    "total_base64_bytes": 200,
                    "assets": [
                        {
                            "submitted_index": 0,
                            "kind": "overview",
                            "submitted_size": [100, 1000],
                            "data_url_size": 200,
                            "logical_indexes": [0],
                        }
                    ],
                    "logical_inputs": [
                        {
                            "image_index": 0,
                            "kind": "overview",
                            "original_size": [100, 1000],
                            "submitted_size": [100, 1000],
                            "data_url_size": 200,
                        }
                    ],
                },
            )
        self.assertNotIn(
            "request_prompt",
            loaded_after_injection["url"]["request_metadata"],
        )
        for forbidden in (
            secret,
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "BASE64SENTINEL",
            "workspace-private",
            "token=secret",
            "request_prompt",
            "access_token",
            '"endpoint"',
            '"raw":',
            '"data_url":',
            '"path":',
            '"unknown"',
        ):
            self.assertNotIn(forbidden, persisted)

    def test_first_ledger_reservation_freezes_legacy_checkpoint_call_baseline(self):
        from scammark.experiment_state import ApiCallLimitError, ExperimentState

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "legacy-transition"})
            state.save_stage(
                "legacy",
                "url",
                {"risk_level": "low"},
                {"cumulative_api_call_count": 49_999},
            )
            self.assertEqual(state.total_api_calls(), 49_999)

            state.reserve_api_call(
                {"sample_id": "new", "stage": "text"},
                max_calls=50_000,
            )
            first_reservation = json.loads(
                state.call_ledger.read_text(encoding="utf-8").splitlines()[0]
            )

            state.save_stage(
                "legacy",
                "url",
                {"risk_level": "low"},
                {"cumulative_api_call_count": 50_000},
            )
            audit = state.api_call_audit()

            self.assertEqual(
                first_reservation["legacy_api_calls_before_ledger"],
                49_999,
            )
            self.assertEqual(audit.reserved, 50_000)
            self.assertEqual(state.total_api_calls(), 50_000)
            with self.assertRaises(ApiCallLimitError):
                state.reserve_api_call(
                    {"sample_id": "blocked", "stage": "visual"},
                    max_calls=50_000,
                )

    def test_blocked_first_reservation_keeps_legacy_fallback_visible(self):
        from scammark.experiment_state import ApiCallLimitError, ExperimentState

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "legacy-at-cap"})
            state.save_stage(
                "legacy",
                "url",
                {"risk_level": "low"},
                {"cumulative_api_call_count": 50_000},
            )

            with self.assertRaises(ApiCallLimitError):
                state.reserve_api_call(
                    {"sample_id": "blocked", "stage": "text"},
                    max_calls=50_000,
                )

            self.assertEqual(state.total_api_calls(), 50_000)
            self.assertEqual(state.call_ledger.read_text(encoding="utf-8"), "")

    def test_checkpoint_audit_classifies_stages_and_redacts_failure_metadata(self):
        from scammark.experiment_state import ExperimentState, scan_checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "v2"})
            state.save_stage("1", "url", {"url": "safe"}, {})
            state.save_stage_error(
                "1",
                "text",
                RuntimeError("Bearer sk-secret PROMPT_SENTINEL"),
                {"raw_response": "RAW_RESPONSE_SENTINEL"},
            )
            audit = scan_checkpoints(state.checkpoints)
            persisted = "\n".join(
                path.read_text(encoding="utf-8") for path in state.checkpoints.glob("*.json")
            )

        self.assertEqual(audit.complete_stages, {("1", "url")})
        self.assertEqual(audit.error_stages, {("1", "text")})
        for sentinel in ("sk-secret", "PROMPT_SENTINEL", "RAW_RESPONSE_SENTINEL"):
            self.assertNotIn(sentinel, persisted)

    def test_checkpoint_audit_excludes_wrong_or_missing_config_hash_from_replay(self):
        from scammark.experiment_state import ExperimentState, scan_checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "v2"})
            state.save_stage("valid", "url", {"risk_level": "low"}, {})
            wrong_path = state.checkpoints / "wrong.json"
            missing_path = state.checkpoints / "missing.json"
            wrong_path.write_text(
                json.dumps(
                    {
                        "sample_id": "replay",
                        "config_hash": "other-config",
                        "stages": {
                            "text": {"status": "complete", "output": {"safe": True}}
                        },
                    }
                ),
                encoding="utf-8",
            )
            missing_path.write_text(
                json.dumps(
                    {
                        "sample_id": "missing",
                        "stages": {
                            "binary": {"status": "complete", "output": {"safe": True}}
                        },
                    }
                ),
                encoding="utf-8",
            )
            call_id = state.reserve_api_call(
                {"sample_id": "replay", "stage": "text"},
                max_calls=10,
            )
            state.finish_api_call(call_id, status="completed")

            checkpoints = scan_checkpoints(
                state.checkpoints,
                config_hash=state.config_hash,
            )
            ledger = state.api_call_audit()

        self.assertEqual(checkpoints.complete_stages, {("valid", "url")})
        self.assertEqual(checkpoints.corrupt_files, ("missing.json", "wrong.json"))
        self.assertEqual(ledger.finished_without_checkpoint, {("replay", "text")})
        self.assertEqual(ledger.replay_required_by_stage, {"text": 1})

    def test_sample_progress_uses_five_or_seven_stages_consistently(self):
        from scammark.experiment_state import summarize_sample_progress

        five = frozenset(
            ("benign", stage) for stage in ("url", "text", "visual", "cross", "binary")
        )
        seven = five | frozenset(("benign", stage) for stage in ("type", "grounding"))
        manifest = [{"sample_id": "benign", "binary": "benign"}]
        benign = summarize_sample_progress(
            manifest,
            complete_stages=five,
            error_stages=frozenset(),
            binary_outputs={"benign": "benign"},
        )
        forced = summarize_sample_progress(
            manifest,
            complete_stages=seven,
            error_stages=frozenset(),
            binary_outputs={"benign": "scam"},
        )
        self.assertEqual((benign.complete, benign.partial, benign.error), (1, 0, 0))
        self.assertEqual((forced.complete, forced.partial, forced.error), (1, 0, 0))

        scam_manifest = [{"sample_id": "scam", "binary": "scam"}]
        scam_five = frozenset(
            ("scam", stage) for stage in ("url", "text", "visual", "cross", "binary")
        )
        scam = summarize_sample_progress(
            scam_manifest,
            complete_stages=scam_five,
            error_stages=frozenset(),
            binary_outputs={"scam": "benign"},
        )
        self.assertEqual((scam.complete, scam.partial), (0, 1))

    def test_sample_progress_ignores_errors_outside_required_stage_set(self):
        from scammark.experiment_state import summarize_sample_progress

        progress = summarize_sample_progress(
            [{"sample_id": "benign", "binary": "benign"}],
            complete_stages=frozenset({("benign", "url")}),
            error_stages=frozenset(
                {
                    ("benign", "type"),
                    ("benign", "future-stage"),
                }
            ),
            binary_outputs={"benign": "benign"},
        )

        self.assertEqual((progress.complete, progress.partial, progress.error), (0, 1, 0))

    def test_stop_is_checked_before_checkpoint_reuse_and_prevents_requests(self):
        import signal

        from scammark.experiment_state import ExperimentState
        from scammark.observability import GracefulStopRequested
        from scammark.pipeline import ScamMarkPipeline

        responses = {
            "url": {"risk_level": "low", "evidence_summary": "url"},
            "text": {"evidence_summary": "text"},
            "visual": {"evidence_summary": "visual"},
            "cross": {"risk_level": "low"},
            "binary": {"binary": "benign", "reasoning": "binary"},
        }

        class Client:
            def __init__(self):
                self.calls = []

            def complete_json_with_metadata(self, stage, **kwargs):
                self.calls.append(stage)
                return responses[stage], {"attempt_count": 1, "usage": {}}

        class RecordingState(ExperimentState):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loads = []

            def load_stages(self, sample_id):
                self.loads.append(str(sample_id))
                return super().load_stages(sample_id)

        class StopBeforeReuse:
            def __init__(self):
                self.boundaries = []

            def raise_if_requested(self, boundary):
                self.boundaries.append(boundary)
                if boundary == "before_stage:url":
                    raise GracefulStopRequested(signal.SIGINT, boundary)

        events = []
        with tempfile.TemporaryDirectory() as tmp:
            state = RecordingState(Path(tmp), {"contract": "v2"})
            ScamMarkPipeline(model_client=Client(), experiment_state=state).predict(
                {"sample_id": "1", "title": "demo"}
            )
            state.loads.clear()
            resumed_client = Client()
            controller = StopBeforeReuse()
            resumed = ScamMarkPipeline(
                model_client=resumed_client,
                experiment_state=state,
                event_callback=lambda event, payload: events.append(
                    (event, dict(payload))
                ),
                stop_controller=controller,
            )
            with self.assertRaises(GracefulStopRequested):
                resumed.predict({"sample_id": "1", "title": "demo"})

        self.assertEqual(
            controller.boundaries,
            ["before_sample", "before_stage:url"],
        )
        self.assertEqual(state.loads, [])
        self.assertEqual(resumed_client.calls, [])
        self.assertFalse(
            any(name in {"stage_reused", "stage_started"} for name, _ in events)
        )

    def test_stop_after_final_stage_resumes_from_checkpoint_without_an_api_call(self):
        import signal

        from scammark.experiment_state import ExperimentState
        from scammark.observability import GracefulStopRequested
        from scammark.pipeline import ScamMarkPipeline
        from scammark.prompts import PROMPT_STAGES

        responses = {
            "url": {"risk_level": "high", "evidence_summary": "url"},
            "text": {"evidence_summary": "text"},
            "visual": {"evidence_summary": "visual"},
            "cross": {"risk_level": "high"},
            "binary": {"binary": "scam", "reasoning": "binary"},
            "type": {"scam_type": "Phishing", "reasoning": "type"},
            "grounding": {
                "description": "grounded",
                "reasoning": "grounding",
                "boxes": [],
            },
        }

        class Client:
            def __init__(self):
                self.calls = []

            def complete_json_with_metadata(self, stage, **kwargs):
                self.calls.append(stage)
                return responses[stage], {"attempt_count": 1, "usage": {}}

        class StopAfterGrounding:
            def __init__(self):
                self.boundaries = []

            def raise_if_requested(self, boundary):
                self.boundaries.append(boundary)
                if boundary == "after_stage:grounding":
                    raise GracefulStopRequested(signal.SIGINT, boundary)

        resumed_events = []
        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"contract": "v2"})
            first_client = Client()
            controller = StopAfterGrounding()
            first = ScamMarkPipeline(
                model_client=first_client,
                experiment_state=state,
                stop_controller=controller,
            )
            with self.assertRaises(GracefulStopRequested):
                first.predict({"sample_id": "1", "title": "demo"})

            self.assertEqual(
                state.load_stages("1")["grounding"]["status"],
                "complete",
            )
            resumed_client = Client()
            resumed = ScamMarkPipeline(
                model_client=resumed_client,
                experiment_state=state,
                event_callback=lambda event, payload: resumed_events.append(
                    (event, dict(payload))
                ),
            )
            resumed.predict({"sample_id": "1", "title": "demo"})

        reused = [payload for name, payload in resumed_events if name == "stage_reused"]
        self.assertEqual(first_client.calls, list(PROMPT_STAGES))
        self.assertEqual(controller.boundaries[-1], "after_stage:grounding")
        self.assertEqual(resumed_client.calls, [])
        self.assertEqual([payload["stage"] for payload in reused], list(PROMPT_STAGES))
        self.assertTrue(
            all(set(payload) == {"sample_id", "stage", "model"} for payload in reused)
        )
        self.assertFalse(any(name == "stage_started" for name, _ in resumed_events))

    def test_stage_success_and_reuse_boundaries_follow_durable_order(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline

        responses = {
            "url": {"risk_level": "low", "evidence_summary": "url"},
            "text": {"evidence_summary": "text"},
            "visual": {"evidence_summary": "visual"},
            "cross": {"risk_level": "low"},
            "binary": {"binary": "benign", "reasoning": "binary"},
        }
        trace = []
        events = []

        class Client:
            model = "fake-text-model"

            def complete_json_with_metadata(self, stage, **kwargs):
                trace.append(f"client_call:{stage}")
                return responses[stage], {
                    "model": "fake-text-model",
                    "provider": "Fake Provider",
                    "endpoint_host": "api.example.test",
                    "endpoint_sha256": "a" * 64,
                    "attempt_count": 1,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }

        class RecordingState(ExperimentState):
            def save_stage(self, sample_id, stage, output, request_metadata):
                super().save_stage(sample_id, stage, output, request_metadata)
                trace.append(f"durable_save:{stage}")

        class RecordingController:
            def raise_if_requested(self, boundary):
                trace.append(boundary)

        def record_event(event, payload):
            safe_payload = dict(payload)
            events.append((event, safe_payload))
            trace.append(f"{event}:{safe_payload.get('stage', '')}")

        sample = {"sample_id": "ordered", "title": "demo"}
        with tempfile.TemporaryDirectory() as tmp:
            state = RecordingState(Path(tmp), {"contract": "ordered"})
            ScamMarkPipeline(
                model_client=Client(),
                experiment_state=state,
                stop_controller=RecordingController(),
                event_callback=record_event,
            ).predict(sample)
            first_url_trace = [item for item in trace if item.endswith(":url")]
            completed_payload = next(
                (
                    payload
                    for name, payload in events
                    if name == "stage_completed" and payload.get("stage") == "url"
                ),
                None,
            )

            trace.clear()
            events.clear()
            resumed_client = Client()
            ScamMarkPipeline(
                model_client=resumed_client,
                experiment_state=state,
                stop_controller=RecordingController(),
                event_callback=record_event,
            ).predict(sample)
            reused_url_trace = [item for item in trace if item.endswith(":url")]

        self.assertEqual(
            first_url_trace,
            [
                "before_stage:url",
                "before_request:url",
                "stage_started:url",
                "client_call:url",
                "durable_save:url",
                "stage_completed:url",
                "after_stage:url",
            ],
        )
        self.assertEqual(
            reused_url_trace,
            ["before_stage:url", "stage_reused:url", "after_stage:url"],
        )
        self.assertIsInstance(completed_payload, dict)
        self.assertEqual(
            set(completed_payload),
            {
                "sample_id",
                "stage",
                "model",
                "provider",
                "endpoint_host",
                "endpoint_sha256",
                "elapsed_seconds",
                "usage",
                "attempt_count",
            },
        )
        self.assertEqual(
            completed_payload["usage"],
            {"prompt_tokens": 10, "completion_tokens": 2},
        )
        self.assertEqual(completed_payload["attempt_count"], 1)
        self.assertGreaterEqual(completed_payload["elapsed_seconds"], 0.0)

    def test_stop_before_request_prevents_stage_call_and_started_event(self):
        import signal

        from scammark.observability import GracefulStopRequested
        from scammark.pipeline import ScamMarkPipeline

        class Client:
            def __init__(self):
                self.calls = []

            def complete_json_with_metadata(self, stage, **kwargs):
                self.calls.append(stage)
                return {"evidence_summary": "unexpected"}, {"attempt_count": 1}

        class StopBeforeRequest:
            def __init__(self):
                self.boundaries = []

            def raise_if_requested(self, boundary):
                self.boundaries.append(boundary)
                if boundary == "before_request:url":
                    raise GracefulStopRequested(signal.SIGINT, boundary)

        client = Client()
        controller = StopBeforeRequest()
        events = []
        pipeline = ScamMarkPipeline(
            model_client=client,
            stop_controller=controller,
            event_callback=lambda event, payload: events.append((event, dict(payload))),
        )
        with self.assertRaises(GracefulStopRequested):
            pipeline.predict({"sample_id": "before-request", "title": "demo"})

        self.assertEqual(
            controller.boundaries,
            ["before_sample", "before_stage:url", "before_request:url"],
        )
        self.assertEqual(client.calls, [])
        self.assertFalse(any(name == "stage_started" for name, _ in events))

    def test_before_sample_precedes_preprocessing_and_checkpoint_loading(self):
        import signal

        from scammark.observability import GracefulStopRequested
        from scammark.pipeline import ScamMarkPipeline

        class Client:
            def complete_json(self, **kwargs):
                raise AssertionError("provider request must not run")

        class State:
            def load_stages(self, sample_id):
                raise AssertionError("checkpoint load must not run")

        class StopBeforeSample:
            def raise_if_requested(self, boundary):
                if boundary == "before_sample":
                    raise GracefulStopRequested(signal.SIGINT, boundary)

        pipeline = ScamMarkPipeline(
            model_client=Client(),
            experiment_state=State(),
            stop_controller=StopBeforeSample(),
        )
        with patch(
            "scammark.pipeline.apply_mhtml_policy",
            side_effect=AssertionError("sample preprocessing must not run"),
        ), self.assertRaises(GracefulStopRequested):
            pipeline.predict({"sample_id": "before-sample", "title": "demo"})

    def test_checkpoint_validation_precedes_image_preparation(self):
        from scammark.experiment_state import CheckpointContractError
        from scammark.pipeline import ScamMarkPipeline

        trace = []

        class Client:
            def complete_json(self, **kwargs):
                raise AssertionError("provider request must not run")

        class InvalidCheckpointState:
            def load_stages(self, sample_id):
                trace.append("checkpoint_load")
                raise CheckpointContractError("checkpoint validation failed")

        class RecordingController:
            def raise_if_requested(self, boundary):
                trace.append(boundary)

        class ImageFailingPipeline(ScamMarkPipeline):
            def _prepare_bundle(self, sample):
                trace.append("image_prepare")
                raise OSError("image preparation must follow checkpoint validation")

        pipeline = ImageFailingPipeline(
            model_client=Client(),
            experiment_state=InvalidCheckpointState(),
            stop_controller=RecordingController(),
            image_cache_dir="unused-cache",
        )
        error = None
        try:
            pipeline.predict(
                {
                    "sample_id": "checkpoint-first",
                    "title": "demo",
                    "image_path": "unused-image.png",
                }
            )
        except Exception as caught:
            error = caught

        self.assertIsInstance(error, CheckpointContractError)
        self.assertEqual(
            trace,
            ["before_sample", "before_stage:url", "checkpoint_load"],
        )

    def test_stop_before_stage_prevents_checkpoint_and_image_work(self):
        import signal

        from scammark.observability import GracefulStopRequested
        from scammark.pipeline import ScamMarkPipeline

        trace = []

        class Client:
            def complete_json(self, **kwargs):
                raise AssertionError("provider request must not run")

        class State:
            def load_stages(self, sample_id):
                trace.append("checkpoint_load")
                raise AssertionError("checkpoint load must not run")

        class StopBeforeStage:
            def raise_if_requested(self, boundary):
                trace.append(boundary)
                if boundary == "before_stage:url":
                    raise GracefulStopRequested(signal.SIGINT, boundary)

        class ImageFailingPipeline(ScamMarkPipeline):
            def _prepare_bundle(self, sample):
                trace.append("image_prepare")
                raise OSError("image preparation must not run")

        pipeline = ImageFailingPipeline(
            model_client=Client(),
            experiment_state=State(),
            stop_controller=StopBeforeStage(),
            image_cache_dir="unused-cache",
        )
        with self.assertRaises(GracefulStopRequested):
            pipeline.predict(
                {
                    "sample_id": "stop-before-stage",
                    "title": "demo",
                    "image_path": "unused-image.png",
                }
            )

        self.assertEqual(trace, ["before_sample", "before_stage:url"])

    def test_stage_failure_is_saved_before_sanitized_event_and_reraises_same_exception(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        secret = "sk-SENTINEL-12345678"
        failure_text = (
            f"Bearer {secret} PROMPT_SENTINEL RAW_RESPONSE_SENTINEL "
            "data:image/png;base64,BASE64SENTINEL "
            "https://workspace-private.cn-beijing.maas.aliyuncs.com/private/path?token=secret"
        )
        provider_error = TransientRequestError(failure_text)
        provider_error.request_metadata = {
            "model": "fake-text-model",
            "provider": "Fake Provider",
            "endpoint_host": "workspace-private.cn-beijing.maas.aliyuncs.com",
            "endpoint_sha256": "b" * 64,
            "attempt_count": 1,
            "api_call_count": 1,
            "usage": {"prompt_tokens": 99},
            "request_prompt": "PROMPT_SENTINEL",
            "raw_response": "RAW_RESPONSE_SENTINEL",
            "endpoint": (
                "https://workspace-private.cn-beijing.maas.aliyuncs.com/"
                "private/path?token=secret"
            ),
        }
        trace = []
        events = []

        class Client:
            model = "fake-text-model"

            def complete_json_with_metadata(self, stage, **kwargs):
                trace.append(f"client_call:{stage}")
                raise provider_error

        class RecordingState(ExperimentState):
            def save_stage_error(
                self,
                sample_id,
                stage,
                error,
                request_metadata,
            ):
                super().save_stage_error(
                    sample_id,
                    stage,
                    error,
                    request_metadata,
                )
                trace.append(f"durable_error:{stage}")

        class RecordingController:
            def raise_if_requested(self, boundary):
                trace.append(boundary)

        def record_event(event, payload):
            safe_payload = dict(payload)
            events.append((event, safe_payload))
            trace.append(f"{event}:{safe_payload.get('stage', '')}")

        with tempfile.TemporaryDirectory() as tmp:
            state = RecordingState(Path(tmp), {"contract": "failure-order"})
            pipeline = ScamMarkPipeline(
                model_client=Client(),
                experiment_state=state,
                stop_controller=RecordingController(),
                event_callback=record_event,
            )
            with self.assertRaises(TransientRequestError) as raised:
                pipeline.predict({"sample_id": "failure-1", "title": "demo"})
            checkpoint_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in state.checkpoints.glob("*.json")
            )

        failure_trace = [item for item in trace if item.endswith(":url")]
        failed_payload = next(
            (payload for name, payload in events if name == "stage_failed"),
            None,
        )
        self.assertIs(raised.exception, provider_error)
        self.assertEqual(
            failure_trace,
            [
                "before_stage:url",
                "before_request:url",
                "stage_started:url",
                "client_call:url",
                "durable_error:url",
                "stage_failed:url",
            ],
        )
        self.assertIsInstance(failed_payload, dict)
        self.assertEqual(
            set(failed_payload),
            {
                "sample_id",
                "stage",
                "model",
                "provider",
                "endpoint_host",
                "endpoint_sha256",
                "elapsed_seconds",
                "error_category",
                "error_type",
                "message",
            },
        )
        self.assertEqual(failed_payload["error_category"], "transient")
        self.assertEqual(failed_payload["error_type"], "TransientRequestError")
        self.assertEqual(
            failed_payload["message"],
            "transient provider or transport failure",
        )
        self.assertEqual(
            failed_payload["endpoint_host"],
            "*.cn-beijing.maas.aliyuncs.com",
        )
        persisted = checkpoint_text + json.dumps(events, ensure_ascii=False)
        for forbidden in (
            secret,
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "BASE64SENTINEL",
            "workspace-private",
            "token=secret",
            "/private/path",
        ):
            self.assertNotIn(forbidden, persisted)

    def test_stage_error_checkpoint_failure_emits_no_false_failed_event(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        provider_error = TransientRequestError("HTTP 503: temporary")
        provider_error.request_metadata = {"attempt_count": 1}
        calls = []
        events = []

        class Client:
            def complete_json_with_metadata(self, stage, **kwargs):
                calls.append(stage)
                raise provider_error

        class FailingErrorState(ExperimentState):
            def save_stage_error(self, *args, **kwargs):
                raise OSError("checkpoint storage unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            state = FailingErrorState(Path(tmp), {"contract": "failed-save"})
            pipeline = ScamMarkPipeline(
                model_client=Client(),
                experiment_state=state,
                event_callback=lambda event, payload: events.append(
                    (event, dict(payload))
                ),
            )
            with self.assertRaisesRegex(OSError, "checkpoint storage unavailable"):
                pipeline.predict({"sample_id": "failed-save", "title": "demo"})

        self.assertEqual(calls, ["url"])
        self.assertFalse(any(name == "stage_failed" for name, _ in events))

    def test_unreadable_resume_failure_counts_close_the_physical_call_cap(self):
        from scammark.experiment_state import ApiCallLimitError, ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        class CountReadFailure(BaseException):
            pass

        first_error = TransientRequestError("HTTP 503: first failure")
        first_error.request_metadata = {
            "attempt_count": 4,
            "api_call_count": 4,
        }

        class UnreadableCountError(TransientRequestError):
            @property
            def request_metadata(self):
                raise CountReadFailure("hostile request metadata")

        resumed_error = UnreadableCountError("HTTP 503: resumed failure")
        resumed_calls = []

        class FirstClient:
            def complete_json_with_metadata(self, **kwargs):
                raise first_error

        class ResumedClient:
            def complete_json_with_metadata(self, stage, **kwargs):
                resumed_calls.append(stage)
                raise resumed_error

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"contract": "count-cap"})
            with self.assertRaises(TransientRequestError) as first_raised:
                ScamMarkPipeline(
                    model_client=FirstClient(),
                    experiment_state=state,
                ).predict({"sample_id": "count-cap", "title": "demo"})
            with self.assertRaises(UnreadableCountError) as resumed_raised:
                ScamMarkPipeline(
                    model_client=ResumedClient(),
                    experiment_state=state,
                ).predict({"sample_id": "count-cap", "title": "demo"})

            metadata = state.load_stages("count-cap")["url"]["request_metadata"]
            total_before_reservation = state.total_api_calls()
            reservation_was_allowed = False
            try:
                state.reserve_api_call(
                    {"sample_id": "blocked", "stage": "url"},
                    max_calls=5,
                )
            except ApiCallLimitError:
                pass
            else:
                reservation_was_allowed = True

        self.assertIs(first_raised.exception, first_error)
        self.assertIs(resumed_raised.exception, resumed_error)
        self.assertEqual(resumed_calls, ["url"])
        self.assertEqual(
            (
                metadata.get("cumulative_attempt_count"),
                metadata.get("cumulative_api_call_count"),
                total_before_reservation,
                reservation_was_allowed,
            ),
            (5, 5, 5, False),
        )

    def test_invalid_and_oversized_failure_counts_are_conservative_and_json_safe(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        class CountSubclass(int):
            pass

        durable_count_limit = 1_000_000_000_000
        cases = (
            ("missing", {}, 1, 1),
            ("negative", {"attempt_count": -4, "api_call_count": -4}, 1, 1),
            ("string", {"attempt_count": "4", "api_call_count": "4"}, 1, 1),
            (
                "subclass",
                {
                    "attempt_count": CountSubclass(4),
                    "api_call_count": CountSubclass(4),
                },
                1,
                1,
            ),
            (
                "oversized",
                {"attempt_count": 10**5000, "api_call_count": 10**5000},
                durable_count_limit,
                durable_count_limit,
            ),
            ("explicit-zero-api", {"attempt_count": 1, "api_call_count": 0}, 1, 0),
        )

        for name, raw_metadata, expected_attempts, expected_api_calls in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                provider_error = TransientRequestError(f"HTTP 503: {name}")
                provider_error.request_metadata = raw_metadata

                class Client:
                    def complete_json_with_metadata(self, **kwargs):
                        raise provider_error

                state = ExperimentState(Path(tmp), {"contract": f"counts-{name}"})
                with self.assertRaises(TransientRequestError) as raised:
                    ScamMarkPipeline(
                        model_client=Client(),
                        experiment_state=state,
                    ).predict({"sample_id": name, "title": "demo"})
                metadata = state.load_stages(name)["url"]["request_metadata"]
                persisted = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in state.checkpoints.glob("*.json")
                )

                self.assertIs(raised.exception, provider_error)
                self.assertEqual(
                    (
                        metadata.get("cumulative_attempt_count"),
                        metadata.get("cumulative_api_call_count"),
                    ),
                    (expected_attempts, expected_api_calls),
                )
                json.loads(persisted)

    def test_nested_oversized_failure_metadata_is_bounded_before_checkpoint_persistence(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        durable_count_limit = 1_000_000_000_000
        oversized = 10**5000
        provider_error = TransientRequestError("HTTP 503: nested metadata")
        cases = (
            (
                "usage",
                {"usage": {"prompt_tokens": oversized}},
                ("usage", "prompt_tokens"),
            ),
            (
                "attempt-errors",
                {
                    "attempt_errors": [
                        {
                            "attempt": oversized,
                            "error_type": "TransientRequestError",
                            "message": "temporary",
                        }
                    ]
                },
                ("attempt_errors", 0, "attempt"),
            ),
            (
                "prompt",
                {"prompt": {"max_tokens": oversized}},
                ("prompt", "max_tokens"),
            ),
            (
                "image",
                {"image": {"asset_count": oversized}},
                ("image", "asset_count"),
            ),
        )

        for name, nested_metadata, value_path in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                provider_error.request_metadata = {
                    "attempt_count": 1,
                    "api_call_count": 1,
                    **nested_metadata,
                }
                calls = []

                class Client:
                    def complete_json_with_metadata(self, stage, **kwargs):
                        calls.append(stage)
                        raise provider_error

                state = ExperimentState(Path(tmp), {"contract": f"nested-{name}"})
                with self.assertRaises(TransientRequestError) as raised:
                    ScamMarkPipeline(
                        model_client=Client(),
                        experiment_state=state,
                    ).predict({"sample_id": name, "title": "demo"})

                checkpoint = next(state.checkpoints.glob("*.json"))
                persisted = checkpoint.read_text(encoding="utf-8")
                parsed_checkpoint = json.loads(persisted)
                stage = state.load_stages(name)["url"]
                metadata = stage["request_metadata"]
                bounded_value = metadata
                for path_component in value_path:
                    bounded_value = bounded_value[path_component]

                self.assertIs(raised.exception, provider_error)
                self.assertEqual(calls, ["url"])
                self.assertEqual(stage["status"], "error")
                self.assertEqual(parsed_checkpoint["stages"]["url"]["status"], "error")
                self.assertEqual(
                    tuple(
                        metadata.get(field)
                        for field in (
                            "attempt_count",
                            "api_call_count",
                            "cumulative_attempt_count",
                            "cumulative_api_call_count",
                        )
                    ),
                    (1, 1, 1, 1),
                )
                self.assertEqual(bounded_value, durable_count_limit)
                self.assertLess(len(persisted), 50_000)

    def test_raw_failure_metadata_is_preflighted_before_shared_projection(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        text_accesses = []

        class HostileText:
            def __str__(self):
                text_accesses.append("str")
                return "PROMPT_SENTINEL_RAW_PROVIDER_BODY"

        integer_accesses = []

        class HostileInteger(int):
            def __int__(self):
                integer_accesses.append("int")
                return 7

        mapping_accesses = []

        class HostileDict(dict):
            def __bool__(self):
                mapping_accesses.append("bool")
                return True

            def keys(self):
                mapping_accesses.append("keys")
                return super().keys()

            def __iter__(self):
                mapping_accesses.append("iter")
                return super().__iter__()

        cases = (
            (
                "hostile-text",
                {"provider": HostileText()},
                text_accesses,
            ),
            (
                "integer-subclass",
                {"usage": {"prompt_tokens": HostileInteger(7)}},
                integer_accesses,
            ),
            (
                "nan",
                {"provider": "must-drop", "elapsed_seconds": float("nan")},
                [],
            ),
            (
                "infinity",
                {"provider": "must-drop", "elapsed_seconds": float("inf")},
                [],
            ),
            (
                "mapping-subclass",
                HostileDict({"provider": "must-drop"}),
                mapping_accesses,
            ),
        )
        expected_fields = {
            "request_fingerprint",
            "attempt_count",
            "api_call_count",
            "cumulative_attempt_count",
            "cumulative_api_call_count",
        }
        provider_error = TransientRequestError("HTTP 503: unsafe raw metadata")

        for name, raw_metadata, accesses in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                provider_error.request_metadata = raw_metadata
                calls = []

                class Client:
                    def complete_json_with_metadata(self, stage, **kwargs):
                        calls.append(stage)
                        raise provider_error

                state = ExperimentState(Path(tmp), {"contract": f"preflight-{name}"})
                with self.assertRaises(TransientRequestError) as raised:
                    ScamMarkPipeline(
                        model_client=Client(),
                        experiment_state=state,
                    ).predict({"sample_id": name, "title": "demo"})

                checkpoints = list(state.checkpoints.glob("*.json"))
                self.assertEqual(len(checkpoints), 1)
                persisted = checkpoints[0].read_text(encoding="utf-8")
                parsed_checkpoint = json.loads(persisted)
                stage = state.load_stages(name)["url"]
                metadata = stage["request_metadata"]

                self.assertIs(raised.exception, provider_error)
                self.assertEqual(calls, ["url"])
                self.assertEqual(accesses, [])
                self.assertNotIn("PROMPT_SENTINEL_RAW_PROVIDER_BODY", persisted)
                self.assertEqual(stage["status"], "error")
                self.assertEqual(parsed_checkpoint["stages"]["url"]["status"], "error")
                self.assertEqual(set(metadata), expected_fields)
                self.assertRegex(metadata["request_fingerprint"], r"^[0-9a-f]{64}$")
                self.assertEqual(
                    tuple(
                        metadata[field]
                        for field in (
                            "attempt_count",
                            "api_call_count",
                            "cumulative_attempt_count",
                            "cumulative_api_call_count",
                        )
                    ),
                    (1, 1, 1, 1),
                )

    def test_raw_preflight_fallback_preserves_trusted_shallow_counts(self):
        from scammark.experiment_state import ApiCallLimitError, ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        provider_accesses = []

        class HostileProvider:
            def __str__(self):
                provider_accesses.append("str")
                return "PROMPT_SENTINEL_UNSAFE_PROVIDER_BODY"

        key_accesses = []

        class HostileKey:
            def __hash__(self):
                key_accesses.append("hash")
                return hash("attempt_count")

            def __eq__(self, other):
                key_accesses.append("eq")
                return False

        hostile_key = HostileKey()
        hostile_key_metadata = {
            "attempt_count": 4,
            "api_call_count": 4,
            hostile_key: "must-drop",
        }
        key_accesses.clear()
        oversized_metadata = {
            f"noise_{index}": 0
            for index in range(1_000)
        }
        oversized_metadata["attempt_count"] = 4
        oversized_metadata["api_call_count"] = 4
        oversized_metadata["provider"] = HostileProvider()
        durable_count_limit = 1_000_000_000_000
        cases = (
            (
                "hostile-provider",
                {
                    "attempt_count": 4,
                    "api_call_count": 4,
                    "provider": HostileProvider(),
                },
                provider_accesses,
                4,
            ),
            (
                "hostile-key",
                hostile_key_metadata,
                key_accesses,
                4,
            ),
            (
                "counts-after-scan-bound",
                oversized_metadata,
                provider_accesses,
                durable_count_limit,
            ),
        )
        expected_fields = {
            "request_fingerprint",
            "attempt_count",
            "api_call_count",
            "cumulative_attempt_count",
            "cumulative_api_call_count",
        }
        provider_error = TransientRequestError("HTTP 503: unsafe raw metadata")

        for name, raw_metadata, accesses, expected_count in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                provider_error.request_metadata = raw_metadata
                calls = []

                class Client:
                    def complete_json_with_metadata(self, stage, **kwargs):
                        calls.append(stage)
                        raise provider_error

                state = ExperimentState(Path(tmp), {"contract": f"counts-{name}"})
                with self.assertRaises(TransientRequestError) as raised:
                    ScamMarkPipeline(
                        model_client=Client(),
                        experiment_state=state,
                    ).predict({"sample_id": name, "title": "demo"})

                checkpoint = next(state.checkpoints.glob("*.json"))
                persisted = checkpoint.read_text(encoding="utf-8")
                parsed_checkpoint = json.loads(persisted)
                stage = state.load_stages(name)["url"]
                metadata = stage["request_metadata"]
                total_before_reservation = state.total_api_calls()
                reservation_was_allowed = False
                try:
                    state.reserve_api_call(
                        {"sample_id": "blocked", "stage": "url"},
                        max_calls=4,
                    )
                except ApiCallLimitError:
                    pass
                else:
                    reservation_was_allowed = True

                self.assertIs(raised.exception, provider_error)
                self.assertEqual(calls, ["url"])
                self.assertEqual(accesses, [])
                self.assertNotIn("PROMPT_SENTINEL_UNSAFE_PROVIDER_BODY", persisted)
                self.assertEqual(stage["status"], "error")
                self.assertEqual(parsed_checkpoint["stages"]["url"]["status"], "error")
                self.assertEqual(set(metadata), expected_fields)
                self.assertEqual(
                    (
                        metadata["attempt_count"],
                        metadata["api_call_count"],
                        metadata["cumulative_attempt_count"],
                        metadata["cumulative_api_call_count"],
                        total_before_reservation,
                        reservation_was_allowed,
                    ),
                    (
                        expected_count,
                        expected_count,
                        expected_count,
                        expected_count,
                        expected_count,
                        False,
                    ),
                )

    def test_failure_count_addition_and_projector_fallback_remain_conservative(self):
        from scammark.pipeline import _project_failure_request_metadata
        from scammark.reliability import TransientRequestError

        class ProjectionFailure(BaseException):
            pass

        provider_error = TransientRequestError("HTTP 503: temporary")
        provider_error.request_metadata = {
            "attempt_count": 4,
            "api_call_count": 4,
        }
        with patch(
            "scammark.pipeline.project_request_metadata",
            side_effect=ProjectionFailure("projection failed"),
        ):
            projected = _project_failure_request_metadata(
                provider_error,
                request_fingerprint="a" * 64,
                prior_attempts=999_999_999_999,
                prior_api_calls=999_999_999_999,
            )

        self.assertEqual(
            projected,
            {
                "request_fingerprint": "a" * 64,
                "attempt_count": 4,
                "api_call_count": 4,
                "cumulative_attempt_count": 1_000_000_000_000,
                "cumulative_api_call_count": 1_000_000_000_000,
            },
        )

    def test_stage_event_projection_baseexception_is_observational_on_success(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline

        class ProjectionFailure(BaseException):
            pass

        responses = {
            "url": {"risk_level": "low", "evidence_summary": "url"},
            "text": {"evidence_summary": "text"},
            "visual": {"evidence_summary": "visual"},
            "cross": {"risk_level": "low"},
            "binary": {"binary": "benign", "reasoning": "binary"},
        }

        class Client:
            def __init__(self):
                self.calls = []

            def complete_json_with_metadata(self, stage, **kwargs):
                self.calls.append(stage)
                return responses[stage], {
                    "attempt_count": 1,
                    "usage": {"prompt_tokens": 10},
                }

        class RecordingController:
            def __init__(self):
                self.boundaries = []

            def raise_if_requested(self, boundary):
                self.boundaries.append(boundary)

        sample = {"sample_id": "projection-success", "title": "demo"}
        with tempfile.TemporaryDirectory() as baseline_tmp:
            baseline_client = Client()
            baseline = ScamMarkPipeline(
                model_client=baseline_client,
                experiment_state=ExperimentState(
                    Path(baseline_tmp),
                    {"contract": "projection-success"},
                ),
            ).predict(sample)

        callback_calls = []
        with tempfile.TemporaryDirectory() as observed_tmp:
            observed_client = Client()
            observed_state = ExperimentState(
                Path(observed_tmp),
                {"contract": "projection-success"},
            )
            controller = RecordingController()
            with patch(
                "scammark.pipeline._project_pipeline_stage_event",
                side_effect=ProjectionFailure("PROMPT_SENTINEL"),
            ):
                observed = ScamMarkPipeline(
                    model_client=observed_client,
                    experiment_state=observed_state,
                    stop_controller=controller,
                    event_callback=lambda event, payload: callback_calls.append(event),
                ).predict(sample)
            checkpoint = observed_state.load_stages(sample["sample_id"])

        expected_boundaries = ["before_sample"]
        for stage in responses:
            expected_boundaries.extend(
                [
                    f"before_stage:{stage}",
                    f"before_request:{stage}",
                    f"after_stage:{stage}",
                ]
            )
        self.assertEqual(observed, baseline)
        self.assertEqual(observed_client.calls, list(responses))
        self.assertEqual(
            [stage for stage, value in checkpoint.items() if value["status"] == "complete"],
            list(responses),
        )
        self.assertEqual(controller.boundaries, expected_boundaries)
        self.assertEqual(callback_calls, [])

    def test_hostile_stage_event_clock_cannot_replace_successful_request(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline

        class ClockFailure(BaseException):
            pass

        class HostileClockValue:
            def __sub__(self, other):
                return self

            def __float__(self):
                raise ClockFailure("PROMPT_SENTINEL")

        responses = {
            "url": {"risk_level": "low", "evidence_summary": "url"},
            "text": {"evidence_summary": "text"},
            "visual": {"evidence_summary": "visual"},
            "cross": {"risk_level": "low"},
            "binary": {"binary": "benign", "reasoning": "binary"},
        }
        calls = []
        events = []

        class Client:
            def complete_json_with_metadata(self, stage, **kwargs):
                calls.append(stage)
                return responses[stage], {"attempt_count": 1, "usage": {}}

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"contract": "hostile-clock"})
            with patch(
                "scammark.pipeline.time.perf_counter",
                return_value=HostileClockValue(),
            ):
                prediction = ScamMarkPipeline(
                    model_client=Client(),
                    experiment_state=state,
                    event_callback=lambda event, payload: events.append(
                        (event, dict(payload))
                    ),
                ).predict({"sample_id": "hostile-clock", "title": "demo"})
            checkpoint = state.load_stages("hostile-clock")

        self.assertEqual(prediction["binary"], "benign")
        self.assertEqual(calls, list(responses))
        self.assertTrue(all(value["status"] == "complete" for value in checkpoint.values()))
        completed = [payload for name, payload in events if name == "stage_completed"]
        self.assertEqual(len(completed), len(responses))
        self.assertTrue(all(payload["elapsed_seconds"] == 0.0 for payload in completed))

    def test_stage_callback_baseexception_cannot_replace_provider_failure_or_repeat(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        class ProjectionFailure(BaseException):
            pass

        class HostileMetadataError(TransientRequestError):
            @property
            def request_metadata(self):
                raise ProjectionFailure("PROMPT_SENTINEL")

        provider_error = HostileMetadataError("HTTP 503: temporary")
        calls = []
        callback_events = []

        class Client:
            def complete_json_with_metadata(self, stage, **kwargs):
                calls.append(stage)
                raise provider_error

        def broken_callback(event, payload):
            callback_events.append(event)
            raise ProjectionFailure("RAW_RESPONSE_SENTINEL")

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"contract": "callback-failure"})
            pipeline = ScamMarkPipeline(
                model_client=Client(),
                experiment_state=state,
                event_callback=broken_callback,
            )
            with self.assertRaises(HostileMetadataError) as raised:
                pipeline.predict({"sample_id": "callback-failure", "title": "demo"})
            checkpoint = state.load_stages("callback-failure")

        self.assertIs(raised.exception, provider_error)
        self.assertEqual(calls, ["url"])
        self.assertEqual(checkpoint["url"]["status"], "error")
        self.assertEqual(callback_events, ["stage_started", "stage_failed"])

    def test_no_callback_skips_event_only_hostile_exception_properties(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        class ProjectionFailure(BaseException):
            pass

        accesses = []

        class HostileEventFailure(TransientRequestError):
            @property
            def response(self):
                accesses.append("response")
                raise ProjectionFailure("RAW_RESPONSE_SENTINEL")

            def __str__(self):
                accesses.append("str")
                raise ProjectionFailure("PROMPT_SENTINEL")

        provider_error = HostileEventFailure("private")
        provider_error.request_metadata = {"attempt_count": 1}
        calls = []

        class Client:
            def complete_json_with_metadata(self, stage, **kwargs):
                calls.append(stage)
                raise provider_error

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"contract": "no-callback"})
            pipeline = ScamMarkPipeline(
                model_client=Client(),
                experiment_state=state,
                event_callback=None,
            )
            with self.assertRaises(HostileEventFailure) as raised:
                pipeline.predict({"sample_id": "no-callback", "title": "demo"})
            checkpoint = state.load_stages("no-callback")

        self.assertIs(raised.exception, provider_error)
        self.assertEqual(calls, ["url"])
        self.assertEqual(checkpoint["url"]["status"], "error")
        self.assertEqual(accesses, [])

    def test_request_control_exceptions_are_not_checkpointed_as_stage_failures(self):
        import signal

        from scammark.experiment_state import ExperimentState
        from scammark.observability import (
            ApiBudgetStop,
            AuthBillingStop,
            BreakerAcknowledgementRequired,
            CircuitBreakerStop,
            ContractDriftStop,
            GracefulStopRequested,
        )
        from scammark.pipeline import ScamMarkPipeline

        errors = (
            AuthBillingStop("authentication stopped the run"),
            ApiBudgetStop("API budget stopped the run"),
            CircuitBreakerStop("circuit breaker stopped the run"),
            ContractDriftStop("contract drift stopped the run"),
            BreakerAcknowledgementRequired("breaker acknowledgement required"),
            GracefulStopRequested(signal.SIGINT, "request"),
            KeyboardInterrupt(),
            SystemExit(9),
        )
        for index, control_error in enumerate(errors):
            with self.subTest(error_type=type(control_error).__name__), tempfile.TemporaryDirectory() as tmp:
                calls = []
                events = []

                class Client:
                    def complete_json_with_metadata(self, stage, **kwargs):
                        calls.append(stage)
                        raise control_error

                state = ExperimentState(Path(tmp), {"contract": f"control-{index}"})
                pipeline = ScamMarkPipeline(
                    model_client=Client(),
                    experiment_state=state,
                    event_callback=lambda event, payload: events.append(
                        (event, dict(payload))
                    ),
                )
                with self.assertRaises(type(control_error)) as raised:
                    pipeline.predict({"sample_id": f"control-{index}", "title": "demo"})

                self.assertIs(raised.exception, control_error)
                self.assertEqual(calls, ["url"])
                self.assertEqual(state.load_stages(f"control-{index}"), {})
                self.assertFalse(any(name == "stage_failed" for name, _ in events))

    def test_pipeline_resumes_stages_and_keeps_request_metadata_separate(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline

        responses = {
            "url": {"risk_level": "low", "evidence_summary": "url"},
            "text": {"evidence_summary": "text"},
            "visual": {"evidence_summary": "visual"},
            "cross": {"risk_level": "high"},
            "binary": {"binary": "scam", "reasoning": "binary"},
            "type": {"scam_type": "Phishing", "reasoning": "type"},
            "grounding": {"description": "grounded", "reasoning": "grounding", "boxes": []},
        }

        class MetadataClient:
            def __init__(self):
                self.calls = []

            def complete_json_with_metadata(
                self,
                stage,
                prompt,
                sample,
                include_image=False,
                max_output_tokens=None,
                image_assets=None,
                max_attempts=None,
                max_total_calls=None,
            ):
                self.calls.append(stage)
                return responses[stage], {
                    "model": "qwen3-vl-flash" if include_image else "gpt-4o-mini",
                    "attempt_count": 1,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            state = ExperimentState(Path(tmp_dir), {"model_split": "qwen3", "forced_diagnostics": True})
            first_client = MetadataClient()
            first = ScamMarkPipeline(
                model_client=first_client,
                experiment_state=state,
                input_token_limits={stage: 4096 for stage in responses},
                output_token_limits={stage: 256 for stage in responses},
            ).predict({"sample_id": "resume-1", "title": "Sign in", "url": "https://example.test"})

            second_client = MetadataClient()
            second = ScamMarkPipeline(
                model_client=second_client,
                experiment_state=state,
                input_token_limits={stage: 4096 for stage in responses},
                output_token_limits={stage: 256 for stage in responses},
            ).predict({"sample_id": "resume-1", "title": "Sign in", "url": "https://example.test"})

        self.assertEqual(first_client.calls, list(responses))
        self.assertEqual(second_client.calls, [])
        self.assertEqual(second["evidence_used"]["prompt_stages"]["type"]["scam_type"], "Phishing")
        self.assertEqual(first["evidence_used"]["request_metadata"]["url"]["attempt_count"], 1)
        self.assertNotIn("request_metadata", first["evidence_used"]["prompt_stages"]["url"])

    def test_pipeline_projects_metadata_before_return_and_prediction_persistence(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.run_experiment import _atomic_write_jsonl, _prediction_record

        secret = "sk-SENTINEL-12345678"
        responses = {
            "url": {"risk_level": "low", "evidence_summary": "url"},
            "text": {"evidence_summary": "text"},
            "visual": {"evidence_summary": "visual"},
            "cross": {"risk_level": "low"},
            "binary": {"binary": "benign", "reasoning": "done"},
        }

        class UnsafeMetadataClient:
            def complete_json_with_metadata(self, stage, **kwargs):
                return responses[stage], {
                    "model": "gpt-4o-mini",
                    "provider": "Fake",
                    "attempt_count": 1,
                    "api_call_count": 1,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "unknown": "RAW_RESPONSE_SENTINEL",
                    },
                    "attempt_errors": [
                        {
                            "attempt": 1,
                            "error_type": "TransientRequestError",
                            "message": f"Bearer {secret} PROMPT_SENTINEL",
                        }
                    ],
                    "request_prompt": "PROMPT_SENTINEL",
                    "access_token": secret,
                    "raw_response": "RAW_RESPONSE_SENTINEL",
                    "endpoint": "https://private.example/path?token=secret",
                }

        sample = {
            "sample_id": "safe-prediction",
            "binary": "benign",
            "url": "https://example.test",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ExperimentState(root / "state", {"config": "projection"})
            prediction = ScamMarkPipeline(
                model_client=UnsafeMetadataClient(),
                experiment_state=state,
            ).predict(sample)
            predictions_path = root / "predictions.jsonl"
            _atomic_write_jsonl(
                predictions_path,
                [_prediction_record(sample, prediction)],
            )
            checkpoint_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in state.checkpoints.glob("*.json")
            )
            returned_text = json.dumps(prediction, ensure_ascii=False)
            durable_text = checkpoint_text + predictions_path.read_text(encoding="utf-8")

        metadata = prediction["evidence_used"]["request_metadata"]["url"]
        self.assertEqual(metadata["usage"], {"prompt_tokens": 10, "completion_tokens": 2})
        self.assertEqual(
            metadata["attempt_errors"],
            [
                {
                    "attempt": 1,
                    "error_type": "TransientRequestError",
                    "message": "<redacted-field>",
                }
            ],
        )
        self.assertEqual(metadata["attempt_count"], 1)
        self.assertEqual(metadata["cumulative_api_call_count"], 1)
        self.assertIn("prompt", metadata)
        for forbidden in (
            secret,
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "token=secret",
            "request_prompt",
            "access_token",
            '"endpoint"',
        ):
            self.assertNotIn(forbidden, returned_text)
            self.assertNotIn(forbidden, durable_text)

    def test_checkpoint_rejects_changed_stage_request_fingerprint(self):
        from scammark.experiment_state import CheckpointContractError, ExperimentState
        from scammark.pipeline import ScamMarkPipeline

        class Client:
            def complete_json(self, stage, prompt, sample, include_image=False):
                if stage == "binary":
                    return {"binary": "benign", "reasoning": "done"}
                return {"evidence_summary": "done"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            state = ExperimentState(Path(tmp_dir), {"contract": "same"})
            pipeline = ScamMarkPipeline(model_client=Client(), experiment_state=state)
            pipeline.predict({"sample_id": "one", "url": "https://example.test", "title": "First"})
            with self.assertRaises(CheckpointContractError):
                pipeline.predict({"sample_id": "one", "url": "https://example.test", "title": "Changed"})

    def test_failed_stage_resume_uses_only_remaining_attempt_budget(self):
        from scammark.experiment_state import ExperimentState
        from scammark.pipeline import ScamMarkPipeline
        from scammark.reliability import TransientRequestError

        class FailingClient:
            def complete_json_with_metadata(self, **kwargs):
                error = TransientRequestError("temporary outage")
                error.request_metadata = {"attempt_count": 4, "retry_count": 3}
                raise error

        class RecoveringClient:
            def __init__(self):
                self.attempt_limits = []

            def complete_json_with_metadata(self, stage, max_attempts=None, **kwargs):
                self.attempt_limits.append((stage, max_attempts))
                if stage == "binary":
                    output = {"binary": "benign", "reasoning": "recovered"}
                else:
                    output = {"evidence_summary": "recovered"}
                return output, {"attempt_count": 1, "retry_count": 0}

        with tempfile.TemporaryDirectory() as tmp_dir:
            state = ExperimentState(Path(tmp_dir), {"resume": "budget"})
            failing = ScamMarkPipeline(model_client=FailingClient(), experiment_state=state)
            with self.assertRaises(TransientRequestError):
                failing.predict({"sample_id": "retry-1", "title": "Example"})

            recovering_client = RecoveringClient()
            prediction = ScamMarkPipeline(
                model_client=recovering_client,
                experiment_state=state,
                max_stage_attempts_across_resume=6,
            ).predict({"sample_id": "retry-1", "title": "Example"})

        self.assertEqual(recovering_client.attempt_limits[0], ("url", 2))
        self.assertEqual(prediction["binary"], "benign")

    def test_api_messages_attach_all_selected_image_assets(self):
        from scammark.model_client import _build_api_messages

        assets = [
            {"index": 0, "kind": "overview", "data_url": "data:image/png;base64,AAAA"},
            {"index": 1, "kind": "tile", "data_url": "data:image/jpeg;base64,BBBB"},
        ]
        with patch.dict(os.environ, {"SCAMMARK_INCLUDE_IMAGE": "0"}, clear=False):
            messages = _build_api_messages(
                "qwen3-vl-flash",
                {},
                "Inspect image_index 0 and 1",
                include_image=True,
                image_assets=assets,
            )

        content = messages[1]["content"]
        self.assertEqual([item["type"] for item in content], ["text", "image_url", "image_url"])
        self.assertEqual(content[2]["image_url"]["url"], "data:image/jpeg;base64,BBBB")

    def test_formal_client_disables_environment_image_override(self):
        from scammark.model_client import _build_api_messages

        with patch.dict(os.environ, {"SCAMMARK_INCLUDE_IMAGE": "1"}, clear=False):
            messages = _build_api_messages(
                "gpt-4o-mini",
                {"image_path": "missing.png"},
                "text only",
                include_image=False,
                allow_env_image_override=False,
            )

        self.assertIsInstance(messages[1]["content"], str)

    def test_direct_formal_client_api_config_loads_local_env(self):
        from scammark.model_client import _select_api_config

        def load_test_env():
            os.environ["SCAMMARK_LLM_API_KEY"] = "test-key"

        with patch.dict(os.environ, {}, clear=True), patch(
            "scammark.model_client._load_local_env",
            side_effect=load_test_env,
        ) as loader:
            key, base_url = _select_api_config(
                "gpt-4o-mini",
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
                "OPENAI_BASE_URL",
            )

        loader.assert_called_once()
        self.assertEqual(key, "test-key")
        self.assertEqual(base_url, "")

    def test_provider_metadata_names_alibaba_model_studio_endpoint(self):
        from scammark.model_client import _provider_name

        self.assertEqual(
            _provider_name(
                "https://workspace-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            ),
            "Alibaba Cloud Model Studio",
        )

    def test_project_root_env_precedes_legacy_src_env(self):
        from scammark.model_client import _load_local_env

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "src").mkdir()
            (root / ".env").write_text("SCAMMARK_TEST_ENV_SOURCE=root\n", encoding="utf-8")
            (root / "src" / ".env").write_text(
                "SCAMMARK_TEST_ENV_SOURCE=legacy\n",
                encoding="utf-8",
            )
            try:
                os.chdir(root)
                with patch.dict(os.environ, {}, clear=True):
                    _load_local_env()
                    self.assertEqual(os.environ["SCAMMARK_TEST_ENV_SOURCE"], "root")
            finally:
                os.chdir(original_cwd)

    def test_json_schema_repair_is_attempted_once(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        replies = iter(
            [
                ("not json", {"usage": {"prompt_tokens": 10}, "finish_reason": "stop"}),
                (
                    '{"binary":"benign","reasoning":"repaired"}',
                    {"usage": {"prompt_tokens": 5, "completion_tokens": 2}, "finish_reason": "stop"},
                ),
            ]
        )
        calls = []

        def transport(**request):
            calls.append(request)
            return next(replies)

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
            transport=transport,
        )
        result, metadata = client.complete_json_with_metadata(
            "binary",
            "Return binary JSON",
            {},
            max_output_tokens=256,
        )

        self.assertEqual(result["binary"], "benign")
        self.assertEqual(len(calls), 2)
        self.assertTrue(metadata["schema_repair_attempted"])
        self.assertEqual(metadata["api_call_count"], 2)
        self.assertEqual(metadata["usage"]["prompt_tokens"], 15)

    def test_type_schema_repair_prompt_includes_every_allowed_taxonomy_label(self):
        from scammark.model_client import _build_schema_repair_prompt
        from scammark.taxonomy import CANONICAL_LABELS

        prompt = _build_schema_repair_prompt(
            "type",
            '{"scam_type":"Brand Impersonation Scam","reasoning":"brand mismatch"}',
            "Stage type returned a non-canonical scam type",
        )

        self.assertIn("Allowed scam_type values", prompt)
        for label in CANONICAL_LABELS:
            self.assertIn(label, prompt)

    def test_only_explicitly_temporary_provider_400_is_retryable(self):
        from scammark.model_client import _is_retryable_provider_http_error

        temporary = (
            '{"error":{"message":"Service temporarily unavailable, please try again later",'
            '"code":"data_inspection_failed"}}'
        )
        permanent = '{"error":{"message":"Invalid request","code":"bad_request"}}'

        self.assertTrue(_is_retryable_provider_http_error(400, temporary))
        self.assertFalse(_is_retryable_provider_http_error(400, permanent))
        self.assertFalse(_is_retryable_provider_http_error(401, temporary))

    def test_temporary_provider_400_uses_a_real_cooldown(self):
        from scammark.model_client import _http_chat_completion_once
        from scammark.reliability import TransientRequestError

        class Response:
            status_code = 400
            text = (
                '{"error":{"message":"Service temporarily unavailable, please try again later",'
                '"code":"data_inspection_failed"}}'
            )
            headers = {}

        class Client:
            def post(self, *args, **kwargs):
                return Response()

        with self.assertRaises(TransientRequestError) as raised:
            _http_chat_completion_once(
                api_key="test-key",
                base_url="https://cn.crazyrouter.com/v1",
                model="qwen3-vl-flash",
                messages=[{"role": "user", "content": "test"}],
                max_output_tokens=256,
                connect_timeout_seconds=20,
                read_timeout_seconds=300,
                enable_thinking=False,
                http_client=Client(),
            )

        self.assertEqual(raised.exception.retry_after_seconds, 30.0)

    def test_explicit_quota_429_is_permanent_but_generic_429_retries(self):
        from scammark.model_client import _http_chat_completion_once
        from scammark.reliability import PermanentRequestError, TransientRequestError

        class Response:
            status_code = 429
            headers = {}

            def __init__(self, text):
                self.text = text

        class Client:
            def __init__(self, text):
                self.text = text

            def post(self, *args, **kwargs):
                return Response(self.text)

        request = {
            "api_key": "test-key",
            "base_url": "https://cn.crazyrouter.com/v1",
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "test"}],
            "max_output_tokens": 10,
            "connect_timeout_seconds": 20,
            "read_timeout_seconds": 120,
            "enable_thinking": False,
        }
        with self.assertRaises(PermanentRequestError):
            _http_chat_completion_once(
                http_client=Client("quota exhausted for account"),
                **request,
            )
        with self.assertRaises(TransientRequestError):
            _http_chat_completion_once(
                http_client=Client("rate limit; retry later"),
                **request,
            )

    def test_json_underscore_quota_balance_credit_and_account_codes_are_explicit(self):
        from scammark.model_client import _is_explicit_exhausted_quota_429

        for detail in (
            '{"error":{"code":"insufficient_quota"}}',
            '{"error":{"code":"account_balance_not_enough"}}',
            '{"error_code":"credit_exhausted"}',
            '{"error":{"code":"account_deactivated"}}',
        ):
            with self.subTest(detail=detail):
                self.assertTrue(_is_explicit_exhausted_quota_429(429, detail))
        self.assertFalse(
            _is_explicit_exhausted_quota_429(
                429,
                '{"error":{"code":"rate_limit_exceeded"}}',
            )
        )
        self.assertFalse(
            _is_explicit_exhausted_quota_429(
                400,
                '{"error":{"code":"insufficient_quota"}}',
            )
        )

    def test_time_window_rate_quota_429_keeps_retry_after_and_four_attempts(self):
        from scammark.model_client import _http_chat_completion_once
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        class Response:
            status_code = 429
            headers = {"Retry-After": "1"}

            def __init__(self, text):
                self.text = text

        class Client:
            def __init__(self, text):
                self.text = text
                self.calls = 0

            def post(self, *args, **kwargs):
                self.calls += 1
                return Response(self.text)

        request = {
            "api_key": "test-key",
            "base_url": "https://cn.crazyrouter.com/v1",
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "test"}],
            "max_output_tokens": 10,
            "connect_timeout_seconds": 20,
            "read_timeout_seconds": 120,
            "enable_thinking": False,
        }
        details = (
            '{"error":{"code":"rate_limit_exceeded","message":"Request quota exceeded for this minute; retry later"}}',
            "requests per minute quota exceeded",
            "rate limit quota exceeded; retry after 1 second",
        )
        for detail in details:
            with self.subTest(detail=detail):
                client = Client(detail)
                with self.assertRaises(TransientRequestError) as raised:
                    _http_chat_completion_once(http_client=client, **request)
                self.assertEqual(raised.exception.retry_after_seconds, 1.0)

        client = Client(details[0])
        sleeps = []
        with self.assertRaises(TransientRequestError):
            execute_with_retry(
                lambda: _http_chat_completion_once(http_client=client, **request),
                RequestPolicy(
                    max_attempts=4,
                    backoff_seconds=(0, 0, 0),
                    full_jitter=False,
                ),
                sleep_fn=sleeps.append,
            )
        self.assertEqual(client.calls, 4)
        self.assertEqual(sleeps, [1.0, 1.0, 1.0])

    def test_structured_429_code_precedes_conflicting_message_text(self):
        from scammark.model_client import _is_explicit_exhausted_quota_429

        self.assertFalse(
            _is_explicit_exhausted_quota_429(
                429,
                '{"error":{"code":"rate_limit_exceeded","message":"insufficient quota"}}',
            )
        )
        self.assertFalse(
            _is_explicit_exhausted_quota_429(
                429,
                '{"error":{"type":"rate_limit_error","message":"account balance exhausted"}}',
            )
        )
        self.assertFalse(
            _is_explicit_exhausted_quota_429(
                429,
                '{"code":"quota_exceeded","message":"requests per minute; retry after 1 second"}',
            )
        )
        self.assertTrue(
            _is_explicit_exhausted_quota_429(
                429,
                '{"error":{"code":"insufficient_quota","message":"rate limit; retry later"}}',
            )
        )

    def test_429_billing_and_time_window_priority_matrix(self):
        from scammark.model_client import _http_chat_completion_once
        from scammark.observability import classify_failure
        from scammark.reliability import PermanentRequestError, TransientRequestError

        class Response:
            status_code = 429
            headers = {"Retry-After": "1"}

            def __init__(self, text):
                self.text = text

        class Client:
            def __init__(self, text):
                self.text = text

            def post(self, *args, **kwargs):
                return Response(self.text)

        request = {
            "api_key": "test-key",
            "base_url": "https://cn.crazyrouter.com/v1",
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "test"}],
            "max_output_tokens": 10,
            "connect_timeout_seconds": 20,
            "read_timeout_seconds": 120,
            "enable_thinking": False,
        }
        permanent_details = (
            "quota exhausted; retry later",
            "account balance exhausted; retry after 1 second",
            "quota limit reached",
            "credit limit reached",
            '{"code":"quota_limit_reached","message":"retry later"}',
            '{"type":"credit_limit_reached","message":"requests per minute"}',
            '{"code":"insufficient_quota","message":"requests per minute; retry after 1 second"}',
        )
        for detail in permanent_details:
            with self.subTest(permanent=detail):
                with self.assertRaises(PermanentRequestError) as raised:
                    _http_chat_completion_once(
                        http_client=Client(detail),
                        **request,
                    )
                classification = classify_failure(raised.exception)
                self.assertEqual(classification.category, "auth_billing")
                self.assertTrue(classification.systemic)
                self.assertEqual(classification.exit_code, 20)

        transient_details = (
            "requests per minute quota limit reached",
            "RPM quota exhausted",
            "rate limit account balance exhausted; retry after 1 second",
            '{"code":"rate_limit_exceeded","message":"account balance exhausted"}',
        )
        for detail in transient_details:
            with self.subTest(transient=detail):
                with self.assertRaises(TransientRequestError) as raised:
                    _http_chat_completion_once(
                        http_client=Client(detail),
                        **request,
                    )
                self.assertEqual(raised.exception.retry_after_seconds, 1.0)
                classification = classify_failure(raised.exception)
                self.assertEqual(classification.category, "transient")
                self.assertFalse(classification.systemic)
                self.assertIsNone(classification.exit_code)

    def test_deep_429_json_parse_failure_preserves_transient_four_attempt_behavior(self):
        from scammark.model_client import _http_chat_completion_once
        from scammark.reliability import RequestPolicy, TransientRequestError, execute_with_retry

        deep_body = "[" * 1100 + "0" + "]" * 1100

        class DeepText:
            def __getitem__(self, key):
                return deep_body

        class Response:
            status_code = 429
            headers = {}
            text = DeepText()

        class Client:
            def __init__(self):
                self.calls = 0

            def post(self, *args, **kwargs):
                self.calls += 1
                return Response()

        request = {
            "api_key": "test-key",
            "base_url": "https://cn.crazyrouter.com/v1",
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "test"}],
            "max_output_tokens": 10,
            "connect_timeout_seconds": 20,
            "read_timeout_seconds": 120,
            "enable_thinking": False,
        }
        client = Client()
        sleeps = []
        with patch(
            "scammark.reliability.json.loads",
            side_effect=RecursionError("provider JSON nesting is too deep"),
        ):
            with self.assertRaises(TransientRequestError):
                execute_with_retry(
                    lambda: _http_chat_completion_once(
                        http_client=client,
                        **request,
                    ),
                    RequestPolicy(
                        max_attempts=4,
                        backoff_seconds=(0, 0, 0),
                        full_jitter=False,
                    ),
                    sleep_fn=sleeps.append,
                )

        self.assertEqual(client.calls, 4)
        self.assertEqual(sleeps, [0, 0, 0])

    def test_permanent_429_transport_ledger_event_and_run_control_categories_agree(self):
        from scammark.experiment_state import ExperimentState
        from scammark.model_client import OpenAIAPIModelClient, _http_chat_completion_once
        from scammark.observability import classify_failure
        from scammark.reliability import PermanentRequestError, RequestPolicy

        class Response:
            status_code = 429
            headers = {}

            def __init__(self, text):
                self.text = text

        class HTTPClient:
            def __init__(self, text):
                self.text = text

            def post(self, *args, **kwargs):
                return Response(self.text)

        details = (
            '{"error":{"code":"account_balance_not_enough"}}',
            "余额不足",
            '{"error":{"code":"hard_limit_reached"}}',
        )
        for index, detail in enumerate(details):
            with self.subTest(detail=detail), tempfile.TemporaryDirectory() as tmp:
                state = ExperimentState(Path(tmp), {"config": f"billing-{index}"})
                events = []

                def transport(**request):
                    return _http_chat_completion_once(
                        api_key="test-key",
                        http_client=HTTPClient(detail),
                        **request,
                    )

                client = OpenAIAPIModelClient(
                    model="gpt-4o-mini",
                    text_base_url="https://cn.crazyrouter.com/v1",
                    request_policy=RequestPolicy(max_attempts=4, backoff_seconds=(0, 0, 0)),
                    transport=transport,
                    call_ledger=state,
                    max_api_calls=10,
                    event_callback=lambda event, payload: events.append((event, dict(payload))),
                )
                with self.assertRaises(PermanentRequestError) as raised:
                    client.complete_json_with_metadata(
                        "binary",
                        "Return JSON",
                        {"sample_id": "1"},
                    )
                rows = [
                    json.loads(line)
                    for line in state.call_ledger.read_text(encoding="utf-8").splitlines()
                ]
                classification = classify_failure(raised.exception)

                self.assertEqual(client.api_call_count, 1)
                self.assertEqual(rows[1]["metadata"]["error_category"], "auth_billing")
                self.assertEqual(events[-1][1]["error_category"], "auth_billing")
                self.assertEqual(classification.category, "auth_billing")
                self.assertTrue(classification.systemic)
                self.assertEqual(classification.exit_code, 20)

    def test_model_client_enforces_physical_api_call_cap(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import PermanentRequestError, RequestPolicy, TransientRequestError

        calls = []

        def transport(**request):
            calls.append(request)
            raise TransientRequestError("disconnect")

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            request_policy=RequestPolicy(max_attempts=4, backoff_seconds=(0, 0, 0)),
            transport=transport,
            max_api_calls=2,
        )
        with self.assertRaises(PermanentRequestError):
            client.complete_json_with_metadata("binary", "Return JSON", {})
        self.assertEqual(len(calls), 2)
        self.assertEqual(client.api_call_count, 2)

    def test_model_client_reservation_context_matches_sample_checkpoint(self):
        from scammark.experiment_state import ExperimentState
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        def transport(**request):
            return '{"binary":"benign","reasoning":"safe"}', {
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "finish_reason": "stop",
                "response_model": "gpt-4o-mini-2024-07-18",
            }

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "client-context"})
            client = OpenAIAPIModelClient(
                model="gpt-4o-mini",
                text_base_url="https://cn.crazyrouter.com/v1",
                request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
                transport=transport,
                call_ledger=state,
                max_api_calls=10,
            )
            output, metadata = client.complete_json_with_metadata(
                "binary",
                "Return JSON",
                {"sample_id": "sample-1"},
            )
            before_checkpoint = state.api_call_audit()
            reservation = json.loads(
                state.call_ledger.read_text(encoding="utf-8").splitlines()[0]
            )
            state.save_stage("sample-1", "binary", output, metadata)
            after_checkpoint = state.api_call_audit()

        self.assertEqual(reservation["context"]["sample_id"], "sample-1")
        self.assertEqual(reservation["context"]["provider"], "Crazyrouter")
        self.assertFalse(before_checkpoint.corrupt)
        self.assertEqual(
            before_checkpoint.finished_without_checkpoint,
            {("sample-1", "binary")},
        )
        self.assertEqual(before_checkpoint.replay_required_by_stage, {"binary": 1})
        self.assertEqual(after_checkpoint.finished_without_checkpoint, frozenset())
        self.assertEqual(after_checkpoint.replay_required_by_stage, {})

    def test_model_events_follow_durable_reserve_and_finish(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        order = []

        class Ledger:
            def reserve_api_call(self, context, *, max_calls):
                order.append("durable_reserve")
                return "call-1"

            def finish_api_call(self, call_id, *, status, metadata=None):
                order.append("durable_finish")

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
            transport=lambda **request: (
                '{"binary":"benign","reasoning":"safe"}',
                {"usage": {}, "finish_reason": "stop"},
            ),
            call_ledger=Ledger(),
            max_api_calls=10,
            event_callback=lambda event, payload: order.append(event),
        )
        client.complete_json_with_metadata("binary", "Return JSON", {"sample_id": "1"})

        self.assertEqual(
            order,
            [
                "durable_reserve",
                "api_call_reserved",
                "durable_finish",
                "api_call_finished",
            ],
        )

    def test_event_callback_failure_does_not_repeat_or_orphan_a_successful_call(self):
        from scammark.experiment_state import ExperimentState
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "v2"})
            calls = []

            def transport(**request):
                calls.append(request)
                return (
                    '{"binary":"benign","reasoning":"safe"}',
                    {"usage": {}, "finish_reason": "stop"},
                )

            def broken_callback(event, payload):
                payload.clear()
                raise OSError("event disk unavailable")

            client = OpenAIAPIModelClient(
                model="gpt-4o-mini",
                request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
                transport=transport,
                call_ledger=state,
                max_api_calls=10,
                event_callback=broken_callback,
            )
            result, _ = client.complete_json_with_metadata(
                "binary",
                "Return JSON",
                {"sample_id": "1"},
            )
            audit = state.api_call_audit()

        self.assertEqual(result["binary"], "benign")
        self.assertEqual(len(calls), 1)
        self.assertEqual((audit.reserved, audit.finished, audit.outstanding), (1, 1, 0))

    def test_model_retry_event_adds_safe_sample_stage_and_provider_context(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy, TransientRequestError

        events = []
        calls = []

        class Ledger:
            def reserve_api_call(self, context, *, max_calls):
                return f"call-{len(calls) + 1}"

            def finish_api_call(self, call_id, *, status, metadata=None):
                return None

        def transport(**request):
            calls.append(request)
            if len(calls) == 1:
                raise TransientRequestError("HTTP 503: private provider body")
            return (
                '{"binary":"benign","reasoning":"safe"}',
                {"usage": {}, "finish_reason": "stop"},
            )

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            text_base_url="https://cn.crazyrouter.com/v1",
            request_policy=RequestPolicy(
                max_attempts=2,
                backoff_seconds=(0,),
                full_jitter=False,
            ),
            transport=transport,
            call_ledger=Ledger(),
            max_api_calls=10,
            event_callback=lambda event, payload: events.append((event, dict(payload))),
        )
        client.complete_json_with_metadata(
            "binary",
            "Return JSON",
            {"sample_id": "sample-7"},
        )

        self.assertEqual(
            [event for event, _ in events],
            [
                "api_call_reserved",
                "api_call_finished",
                "retry_scheduled",
                "api_call_reserved",
                "api_call_finished",
            ],
        )
        retry = events[2][1]
        self.assertEqual(retry["sample_id"], "sample-7")
        self.assertEqual(retry["stage"], "binary")
        self.assertEqual(retry["model"], "gpt-4o-mini")
        self.assertEqual(retry["provider"], "Crazyrouter")
        self.assertEqual(retry["endpoint_host"], "cn.crazyrouter.com")
        self.assertEqual(len(retry["endpoint_sha256"]), 64)
        self.assertEqual(
            (retry["attempt"], retry["next_attempt"], retry["delay_seconds"]),
            (1, 2, 0),
        )

    def test_model_attempt_context_survives_run_event_logger_jsonl_projection(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.observability import ObservabilitySettings, RunEventLogger
        from scammark.reliability import RequestPolicy, TransientRequestError

        calls = []

        class Ledger:
            def reserve_api_call(self, context, *, max_calls):
                return f"call-{len(calls) + 1}"

            def finish_api_call(self, call_id, *, status, metadata=None):
                return None

        def transport(**request):
            calls.append(request)
            if len(calls) == 1:
                raise TransientRequestError("HTTP 503: temporary")
            return (
                '{"binary":"benign","reasoning":"safe"}',
                {"usage": {}, "finish_reason": "stop"},
            )

        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.jsonl"
            logger = RunEventLogger(
                event_path,
                ObservabilitySettings(),
                stream=None,
                attempt_id="attempt-1",
            )
            client = OpenAIAPIModelClient(
                model="gpt-4o-mini",
                text_base_url="https://cn.crazyrouter.com/v1",
                request_policy=RequestPolicy(
                    max_attempts=2,
                    backoff_seconds=(0,),
                    full_jitter=False,
                ),
                transport=transport,
                call_ledger=Ledger(),
                max_api_calls=10,
                event_callback=lambda event, payload: logger.emit(event, **payload),
            )
            client.complete_json_with_metadata(
                "binary",
                "Return JSON",
                {"sample_id": "sample-7"},
            )
            rows = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["event"] for row in rows], [
            "api_call_reserved",
            "api_call_finished",
            "retry_scheduled",
            "api_call_reserved",
            "api_call_finished",
        ])
        self.assertEqual([rows[index]["attempt"] for index in (0, 1, 3, 4)], [1, 1, 2, 2])
        retry = rows[2]
        self.assertEqual(retry["model"], "gpt-4o-mini")
        self.assertEqual(retry["provider"], "Crazyrouter")
        self.assertEqual(retry["endpoint_host"], "cn.crazyrouter.com")
        self.assertEqual(len(retry["endpoint_sha256"]), 64)

    def test_invalid_event_integer_fields_are_dropped_without_losing_logger_rows(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.observability import ObservabilitySettings, RunEventLogger

        class IntSubclass(int):
            pass

        huge = 10**5000
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.jsonl"
            logger = RunEventLogger(
                event_path,
                ObservabilitySettings(),
                stream=None,
                attempt_id="attempt-1",
            )
            client = OpenAIAPIModelClient(
                event_callback=lambda event, payload: logger.emit(event, **payload)
            )
            common = {
                "sample_id": "1",
                "stage": "binary",
                "model": "gpt-4o-mini",
                "provider": "Crazyrouter",
                "endpoint_host": "cn.crazyrouter.com",
                "endpoint_sha256": "a" * 64,
            }
            client._emit(
                "api_call_reserved",
                {
                    **common,
                    "call_id": "call-1",
                    "attempt": IntSubclass(1),
                    "api_calls": 1,
                    "api_call_limit": 10,
                },
            )
            client._emit(
                "api_call_reserved",
                {
                    **common,
                    "call_id": "call-2",
                    "attempt": 2,
                    "api_calls": huge,
                    "api_call_limit": 10,
                },
            )
            client._emit(
                "api_call_reserved",
                {
                    **common,
                    "call_id": "call-3",
                    "attempt": 3,
                    "api_calls": 3,
                    "api_call_limit": huge,
                },
            )
            client._emit(
                "retry_scheduled",
                {
                    **common,
                    "attempt": 3,
                    "next_attempt": huge,
                    "delay_seconds": 0,
                    "error_type": "TransientRequestError",
                    "message": "transient provider or transport failure",
                },
            )
            logger.emit("heartbeat", health="waiting_api", wait_seconds=1)
            rows = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [row["event"] for row in rows],
            [
                "api_call_reserved",
                "api_call_reserved",
                "api_call_reserved",
                "retry_scheduled",
                "heartbeat",
            ],
        )
        self.assertNotIn("attempt", rows[0])
        self.assertEqual(rows[0]["api_calls"], 1)
        self.assertNotIn("api_calls", rows[1])
        self.assertEqual(rows[1]["attempt"], 2)
        self.assertNotIn("api_call_limit", rows[2])
        self.assertEqual(rows[2]["api_calls"], 3)
        self.assertNotIn("next_attempt", rows[3])
        self.assertEqual(rows[3]["attempt"], 3)

    def test_invalid_delay_and_elapsed_fields_do_not_drop_logger_rows(self):
        from scammark.model_client import (
            OpenAIAPIModelClient,
            _MAX_EVENT_ELAPSED_SECONDS,
        )
        from scammark.observability import ObservabilitySettings, RunEventLogger

        class IntSubclass(int):
            pass

        class FloatSubclass(float):
            pass

        invalid_numbers = (
            IntSubclass(1),
            FloatSubclass(1.5),
            10**5000,
            float("nan"),
            float("inf"),
            _MAX_EVENT_ELAPSED_SECONDS + 1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.jsonl"
            logger = RunEventLogger(
                event_path,
                ObservabilitySettings(),
                stream=None,
                attempt_id="attempt-1",
            )
            client = OpenAIAPIModelClient(
                event_callback=lambda event, payload: logger.emit(event, **payload)
            )
            common = {
                "sample_id": "1",
                "stage": "binary",
                "model": "gpt-4o-mini",
                "provider": "Crazyrouter",
                "endpoint_host": "cn.crazyrouter.com",
                "endpoint_sha256": "a" * 64,
            }
            for index, number in enumerate(invalid_numbers, start=1):
                client._emit(
                    "retry_scheduled",
                    {
                        **common,
                        "attempt": index,
                        "next_attempt": index + 1,
                        "delay_seconds": number,
                        "error_type": "TransientRequestError",
                        "message": "transient provider or transport failure",
                    },
                )
            for index, number in enumerate(invalid_numbers, start=1):
                client._emit(
                    "api_call_finished",
                    {
                        **common,
                        "call_id": f"call-{index}",
                        "attempt": index,
                        "status": "completed",
                        "elapsed_seconds": number,
                        "usage": {},
                    },
                )
            logger.emit("heartbeat", health="waiting_api", wait_seconds=1)
            rows = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 13)
        self.assertEqual([row["event"] for row in rows[:6]], ["retry_scheduled"] * 6)
        self.assertEqual([row["event"] for row in rows[6:12]], ["api_call_finished"] * 6)
        for row in rows[:6]:
            self.assertNotIn("delay_seconds", row)
        for row in rows[6:12]:
            self.assertNotIn("elapsed_seconds", row)
        self.assertEqual(rows[-1]["event"], "heartbeat")

    def test_model_event_and_finish_metadata_are_exact_safe_projections(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        secret = "sk-SENTINEL-12345678"
        events = []
        finishes = []

        class Ledger:
            def reserve_api_call(self, context, *, max_calls):
                return "call-1"

            def finish_api_call(self, call_id, *, status, metadata=None):
                finishes.append((call_id, status, dict(metadata or {})))

        def transport(**request):
            return (
                '{"binary":"benign","reasoning":"safe"}',
                {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": "2",
                        "private": f"Bearer {secret} RAW_RESPONSE_SENTINEL",
                    },
                    "finish_reason": "stop",
                    "response_model": "gpt-4o-mini-2024-07-18",
                    "response_id": f"provider-{secret}",
                    "raw_response": "RAW_RESPONSE_SENTINEL",
                },
            )

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            text_base_url="https://cn.crazyrouter.com/v1?workspace=private",
            request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
            transport=transport,
            call_ledger=Ledger(),
            max_api_calls=10,
            event_callback=lambda event, payload: events.append((event, dict(payload))),
        )
        client.complete_json_with_metadata(
            "binary",
            "PROMPT_SENTINEL",
            {"sample_id": "sample-1"},
        )

        reserved = events[0][1]
        finished = events[1][1]
        self.assertEqual(
            set(reserved),
            {
                "sample_id",
                "stage",
                "call_id",
                "model",
                "provider",
                "endpoint_host",
                "endpoint_sha256",
                "attempt",
                "api_calls",
                "api_call_limit",
            },
        )
        self.assertEqual(
            set(finished),
            {
                "sample_id",
                "stage",
                "call_id",
                "model",
                "provider",
                "endpoint_host",
                "endpoint_sha256",
                "attempt",
                "status",
                "elapsed_seconds",
                "usage",
            },
        )
        self.assertEqual(reserved["call_id"], "call-1")
        self.assertEqual(reserved["attempt"], 1)
        self.assertEqual(reserved["api_calls"], 1)
        self.assertEqual(reserved["api_call_limit"], 10)
        self.assertEqual(finished["status"], "completed")
        self.assertEqual(finished["attempt"], 1)
        self.assertEqual(finished["usage"], {"prompt_tokens": 10})
        self.assertGreaterEqual(finished["elapsed_seconds"], 0.0)
        self.assertTrue(finishes)
        self.assertNotIn("response_id", finishes[0][2])
        serialized = json.dumps({"events": events, "finishes": finishes})
        for forbidden in (
            secret,
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "workspace=private",
            "request_fingerprint",
            "response_id",
            "raw_response",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_adversarial_usage_projection_cannot_orphan_or_replace_provider_result(self):
        from scammark.experiment_state import ExperimentState
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        class HostileUsage(dict):
            def items(self):
                raise RuntimeError("PROMPT_SENTINEL")

        events = []
        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "hostile-usage"})
            client = OpenAIAPIModelClient(
                model="gpt-4o-mini",
                request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
                transport=lambda **request: (
                    '{"binary":"benign","reasoning":"safe"}',
                    {
                        "usage": HostileUsage(prompt_tokens=10),
                        "finish_reason": "stop",
                    },
                ),
                call_ledger=state,
                max_api_calls=10,
                event_callback=lambda event, payload: events.append((event, dict(payload))),
            )
            result, _ = client.complete_json_with_metadata(
                "binary",
                "Return JSON",
                {"sample_id": "1"},
            )
            audit = state.api_call_audit()
            persisted = state.call_ledger.read_text(encoding="utf-8") + json.dumps(events)

        self.assertEqual(result["binary"], "benign")
        self.assertEqual((audit.reserved, audit.finished, audit.outstanding), (1, 1, 0))
        self.assertNotIn("PROMPT_SENTINEL", persisted)
        self.assertEqual(events[-1][1]["usage"], {})

    def test_emit_projection_failures_and_extreme_values_are_isolated(self):
        from scammark.model_client import OpenAIAPIModelClient

        class Unprintable:
            def __bool__(self):
                raise RuntimeError("PROMPT_SENTINEL")

            def __str__(self):
                raise RuntimeError("RAW_RESPONSE_SENTINEL")

        events = []
        client = OpenAIAPIModelClient(
            event_callback=lambda event, payload: events.append((event, dict(payload)))
        )
        client._emit(
            "api_call_finished",
            {
                "sample_id": Unprintable(),
                "stage": "binary",
                "call_id": "call-1",
                "status": "completed",
                "elapsed_seconds": float("inf"),
                "usage": {
                    "prompt_tokens": 10**5000,
                    "completion_tokens": float("nan"),
                    "output_tokens": Unprintable(),
                },
            },
        )

        self.assertEqual(len(events), 1)
        payload = events[0][1]
        self.assertNotIn("sample_id", payload)
        self.assertNotIn("elapsed_seconds", payload)
        self.assertEqual(payload["usage"], {})
        self.assertNotIn("PROMPT_SENTINEL", json.dumps(events))
        self.assertNotIn("RAW_RESPONSE_SENTINEL", json.dumps(events))

    def test_provider_usage_projector_failure_still_durably_finishes_once(self):
        from scammark.experiment_state import ExperimentState
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "projector-failure"})

            def transport(**request):
                calls.append(request)
                return (
                    '{"binary":"benign","reasoning":"safe"}',
                    {"usage": {"prompt_tokens": 10}, "finish_reason": "stop"},
                )

            client = OpenAIAPIModelClient(
                model="gpt-4o-mini",
                request_policy=RequestPolicy(max_attempts=4, backoff_seconds=(0, 0, 0)),
                transport=transport,
                call_ledger=state,
                max_api_calls=10,
                event_callback=lambda event, payload: None,
            )
            with patch(
                "scammark.model_client._safe_event_usage",
                side_effect=RuntimeError("PROMPT_SENTINEL"),
            ):
                result, _ = client.complete_json_with_metadata(
                    "binary",
                    "Return JSON",
                    {"sample_id": "1"},
                )
            audit = state.api_call_audit()

        self.assertEqual(result["binary"], "benign")
        self.assertEqual(len(calls), 1)
        self.assertEqual((audit.reserved, audit.finished, audit.outstanding), (1, 1, 0))

    def test_hostile_retry_context_is_not_read_without_an_event_callback(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        class HostileContext(dict):
            def get(self, key, default=None):
                raise RuntimeError("PROMPT_SENTINEL")

        calls = []

        def operation():
            calls.append(1)
            return "provider-result", {"usage": {}}

        client = OpenAIAPIModelClient(event_callback=None)
        result, _ = client._request_with_retry(
            operation,
            RequestPolicy(max_attempts=1, backoff_seconds=()),
            call_context=HostileContext(sample_id="1"),
        )

        self.assertEqual(result, "provider-result")
        self.assertEqual(calls, [1])

    def test_hostile_retry_context_projection_fails_closed_with_a_callback(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy, TransientRequestError

        class ProjectionFailure(BaseException):
            pass

        class HostileContext(dict):
            def get(self, key, default=None):
                raise ProjectionFailure("PROMPT_SENTINEL")

        calls = []
        events = []

        def operation():
            calls.append(1)
            if len(calls) == 1:
                raise TransientRequestError("HTTP 503: temporary")
            return "provider-result", {"usage": {}}

        client = OpenAIAPIModelClient(
            event_callback=lambda event, payload: events.append((event, dict(payload)))
        )
        result, _ = client._request_with_retry(
            operation,
            RequestPolicy(
                max_attempts=2,
                backoff_seconds=(0,),
                full_jitter=False,
            ),
            call_context=HostileContext(sample_id="1"),
        )

        self.assertEqual(result, "provider-result")
        self.assertEqual(calls, [1, 1])
        self.assertEqual([event for event, _ in events], ["retry_scheduled"])
        self.assertNotIn("sample_id", events[0][1])

    def test_durable_finish_failure_after_transport_never_reissues_provider_request(self):
        from scammark.experiment_state import ExperimentState
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import PermanentRequestError, RequestPolicy, TransientRequestError

        calls = []
        events = []

        class FailingFinishState(ExperimentState):
            def finish_api_call(self, call_id, *, status, metadata=None):
                raise TransientRequestError("ledger storage temporarily unavailable")

        def transport(**request):
            calls.append(request)
            return (
                '{"binary":"benign","reasoning":"safe"}',
                {"usage": {}, "finish_reason": "stop"},
            )

        with tempfile.TemporaryDirectory() as tmp:
            state = FailingFinishState(Path(tmp), {"config": "finish-failure"})
            client = OpenAIAPIModelClient(
                model="gpt-4o-mini",
                request_policy=RequestPolicy(
                    max_attempts=4,
                    backoff_seconds=(0, 0, 0),
                    full_jitter=False,
                ),
                transport=transport,
                call_ledger=state,
                max_api_calls=10,
                event_callback=lambda event, payload: events.append((event, dict(payload))),
            )
            with self.assertRaises(PermanentRequestError):
                client.complete_json_with_metadata(
                    "binary",
                    "Return JSON",
                    {"sample_id": "1"},
                )
            audit = state.api_call_audit()

        self.assertEqual(len(calls), 1)
        self.assertEqual([event for event, _ in events], ["api_call_reserved"])
        self.assertEqual((audit.reserved, audit.finished, audit.outstanding), (1, 0, 1))

    def test_durable_reserve_failure_emits_nothing_and_never_calls_provider(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import PermanentRequestError, RequestPolicy

        events = []
        calls = []

        class Ledger:
            def reserve_api_call(self, context, *, max_calls):
                raise RuntimeError("reservation storage unavailable")

            def finish_api_call(self, call_id, *, status, metadata=None):
                raise AssertionError("finish must not run")

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            request_policy=RequestPolicy(max_attempts=4, backoff_seconds=(0, 0, 0)),
            transport=lambda **request: calls.append(request),
            call_ledger=Ledger(),
            max_api_calls=10,
            event_callback=lambda event, payload: events.append((event, dict(payload))),
        )
        with self.assertRaises(PermanentRequestError):
            client.complete_json_with_metadata(
                "binary",
                "Return JSON",
                {"sample_id": "1"},
            )

        self.assertEqual(calls, [])
        self.assertEqual(events, [])
        self.assertEqual(client.api_call_count, 0)

    def test_provider_failure_sentinels_do_not_reach_ledger_events_or_retry_metadata(self):
        from scammark.experiment_state import ExperimentState
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy, TransientRequestError

        events = []
        secret = "sk-SENTINEL-12345678"
        body = (
            f"Bearer {secret} PROMPT_SENTINEL RAW_RESPONSE_SENTINEL "
            "data:image/png;base64,AAAA "
            "https://ws.cn-beijing.maas.aliyuncs.com/p?token=x"
        )
        with tempfile.TemporaryDirectory() as tmp:
            state = ExperimentState(Path(tmp), {"config": "v2"})

            def transport(**request):
                raise TransientRequestError(body)

            client = OpenAIAPIModelClient(
                model="gpt-4o-mini",
                request_policy=RequestPolicy(
                    max_attempts=4,
                    backoff_seconds=(0, 0, 0),
                ),
                transport=transport,
                call_ledger=state,
                max_api_calls=10,
                event_callback=lambda event, payload: events.append((event, dict(payload))),
            )
            with self.assertRaises(TransientRequestError) as raised:
                client.complete_json_with_metadata(
                    "binary",
                    "PROMPT_SENTINEL",
                    {"sample_id": "1"},
                )
            persisted = (
                state.call_ledger.read_text(encoding="utf-8")
                + json.dumps(events, ensure_ascii=False)
                + json.dumps(raised.exception.request_metadata, ensure_ascii=False)
            )

        for sentinel in (
            secret,
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "AAAA",
            "token=x",
            "ws.cn-beijing",
            "/p",
        ):
            self.assertNotIn(sentinel, persisted)

    def test_schema_repair_is_one_non_retried_extra_call(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy, TransientRequestError

        calls = []

        def transport(**request):
            calls.append(request)
            if len(calls) == 1:
                return "not json", {"usage": {}, "finish_reason": "stop"}
            raise TransientRequestError("repair connection failed")

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            request_policy=RequestPolicy(max_attempts=4, backoff_seconds=(0, 0, 0)),
            transport=transport,
        )
        with self.assertRaises(TransientRequestError):
            client.complete_json_with_metadata(
                "binary",
                "Return JSON",
                {},
                max_attempts=4,
                max_total_calls=6,
            )
        self.assertEqual(len(calls), 2)

    def test_grounding_tile_coordinates_are_validated_before_checkpoint(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        replies = iter(
            [
                (
                    '{"description":"page","reasoning":"risk","boxes":[{"xyxy":[1,2,3,4],"reason":"x","evidence_type":"text"}]}',
                    {"usage": {}, "finish_reason": "stop"},
                ),
                (
                    '{"description":"page","reasoning":"risk","boxes":[{"image_index":1,"bbox_1000":[10,20,300,400],"reason":"x","evidence_type":"text"}]}',
                    {"usage": {}, "finish_reason": "stop"},
                ),
            ]
        )
        calls = []

        def transport(**request):
            calls.append(request)
            return next(replies)

        client = OpenAIAPIModelClient(
            request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
            transport=transport,
        )
        result, metadata = client.complete_json_with_metadata(
            "grounding",
            "Return grounding JSON",
            {},
            include_image=True,
            image_assets=[{"index": 1, "data_url": "data:image/png;base64,AAAA"}],
            max_total_calls=2,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result["boxes"][0]["image_index"], 1)
        self.assertTrue(metadata["schema_repair_attempted"])

    def test_invalid_grounding_geometry_is_preserved_for_metric_layer(self):
        from scammark.image_preprocessing import ImageAsset, ImageBundle, normalize_grounding_boxes
        from scammark.model_client import _parse_and_validate_stage_output

        result = _parse_and_validate_stage_output(
            "grounding",
            json.dumps(
                {
                    "description": "page",
                    "reasoning": "risk",
                    "boxes": [
                        {
                            "image_index": 0,
                            "bbox_1000": [-5, 20, -5, 400],
                            "reason": "degenerate after clipping",
                            "evidence_type": "text",
                        },
                        {
                            "image_index": 0,
                            "bbox_1000": ["bad", 20, 300, 400],
                            "reason": "non-numeric",
                            "evidence_type": "text",
                        },
                    ],
                }
            ),
            valid_image_indexes={0},
        )
        asset = ImageAsset(
            index=0,
            kind="overview",
            path="overview.png",
            width=100,
            height=100,
            crop_xyxy=[0, 0, 100, 100],
            source_size=[100, 100],
            mime_type="image/png",
            data_url_size=100,
        )
        normalized = normalize_grounding_boxes(
            result,
            ImageBundle("page.png", [100, 100], [asset], {}),
        )

        self.assertEqual(len(normalized["boxes"]), 2)
        self.assertIsNone(normalized["boxes"][0]["xyxy"])
        self.assertIsNone(normalized["boxes"][1]["xyxy"])

    def test_invalid_binary_value_triggers_schema_repair(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import RequestPolicy

        replies = iter(
            [
                ('{"binary":"fraudulent","reasoning":"invalid label"}', {"usage": {}, "finish_reason": "stop"}),
                ('{"binary":"benign","reasoning":"repaired"}', {"usage": {}, "finish_reason": "stop"}),
            ]
        )
        calls = []

        def transport(**request):
            calls.append(request)
            return next(replies)

        client = OpenAIAPIModelClient(
            request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
            transport=transport,
        )
        result, metadata = client.complete_json_with_metadata(
            "binary",
            "Return binary JSON",
            {},
            max_total_calls=2,
        )

        self.assertEqual(result["binary"], "benign")
        self.assertEqual(len(calls), 2)
        self.assertTrue(metadata["schema_repair_attempted"])

    def test_length_finish_reason_is_rejected_without_schema_repair(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import PermanentRequestError, RequestPolicy

        calls = []

        def transport(**request):
            calls.append(request)
            return '{"binary":"benign","reasoning":"truncated"}', {
                "usage": {"prompt_tokens": 10},
                "finish_reason": "length",
                "response_model": "gpt-4o-mini-2024-07-18",
            }

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
            transport=transport,
        )
        with self.assertRaises(PermanentRequestError):
            client.complete_json_with_metadata("binary", "Return JSON", {})
        self.assertEqual(len(calls), 1)

    def test_formal_response_model_mismatch_is_rejected(self):
        from scammark.model_client import OpenAIAPIModelClient
        from scammark.reliability import PermanentRequestError, RequestPolicy

        def transport(**request):
            return '{"binary":"benign","reasoning":"ok"}', {
                "usage": {"prompt_tokens": 10},
                "finish_reason": "stop",
                "response_model": "different-model",
            }

        client = OpenAIAPIModelClient(
            model="gpt-4o-mini",
            text_model="gpt-4o-mini",
            vision_model="qwen3-vl-flash",
            text_base_url="https://cn.crazyrouter.com/v1",
            vision_base_url="https://cn.crazyrouter.com/v1",
            strict_routing=True,
            request_policy=RequestPolicy(max_attempts=1, backoff_seconds=()),
            transport=transport,
        )
        with self.assertRaises(PermanentRequestError):
            client.complete_json_with_metadata("binary", "Return JSON", {})

    def test_oracle_diagnostic_failure_preserves_primary_prediction(self):
        from scammark.run_experiment import _run_primary_and_oracle_diagnostic

        class Pipeline:
            def predict(self, sample, force_downstream=None):
                if force_downstream:
                    raise RuntimeError("diagnostic failed")
                return {"binary": "benign", "scam_type": "none", "boxes": [], "evidence_used": {}}

        primary, diagnostic = _run_primary_and_oracle_diagnostic(
            Pipeline(),
            {"sample_id": "scam-1", "binary": "scam"},
            enabled=True,
        )

        self.assertEqual(primary["binary"], "benign")
        self.assertEqual(diagnostic["status"], "error")
        self.assertEqual(diagnostic["error"]["type"], "RuntimeError")
        self.assertEqual(diagnostic["error"]["message"], "<redacted-field>")

    def test_partial_prediction_is_a_serializable_backend_error_record(self):
        from scammark.experiment_state import ExperimentState
        from scammark.run_experiment import _partial_prediction

        with tempfile.TemporaryDirectory() as tmp_dir:
            state = ExperimentState(Path(tmp_dir), {"contract": "partial-error-test"})
            prediction = _partial_prediction(
                {"sample_id": "failed-1"},
                state,
                RuntimeError("provider unavailable"),
            )

        self.assertEqual(prediction["sample_id"], "failed-1")
        self.assertEqual(prediction["binary"], "__invalid__")
        self.assertTrue(prediction["backend_error"])
        self.assertEqual(
            prediction["reasoning"],
            "backend_error:RuntimeError: <redacted-field>",
        )

    def test_failure_prediction_artifact_redacts_partial_and_oracle_errors(self):
        from scammark.experiment_state import ExperimentState
        from scammark.run_experiment import (
            _atomic_write_jsonl,
            _partial_prediction,
            _prediction_record,
            _run_primary_and_oracle_diagnostic,
        )

        secret = "sk-SENTINEL-12345678"
        failure = (
            f"Bearer {secret} PROMPT_SENTINEL RAW_RESPONSE_SENTINEL "
            "data:image/png;base64,BASE64SENTINEL "
            "https://workspace-private.cn-beijing.maas.aliyuncs.com/path?token=secret"
        )

        class Pipeline:
            def predict(self, sample, force_downstream=None):
                if force_downstream:
                    raise RuntimeError(failure)
                return {
                    "binary": "benign",
                    "scam_type": "none",
                    "boxes": [],
                    "evidence_used": {},
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ExperimentState(root / "state", {"config": "failure-artifact"})
            partial_sample = {"sample_id": "partial", "binary": "benign"}
            partial = _partial_prediction(
                partial_sample,
                state,
                RuntimeError(failure),
            )
            oracle_sample = {"sample_id": "oracle", "binary": "scam"}
            primary, oracle = _run_primary_and_oracle_diagnostic(
                Pipeline(),
                oracle_sample,
                enabled=True,
            )
            predictions_path = root / "predictions.jsonl"
            _atomic_write_jsonl(
                predictions_path,
                [
                    _prediction_record(partial_sample, partial),
                    _prediction_record(oracle_sample, primary, oracle),
                ],
            )
            persisted = predictions_path.read_text(encoding="utf-8")

        self.assertGreaterEqual(persisted.count("RuntimeError"), 2)
        self.assertGreaterEqual(persisted.count("<redacted-field>"), 2)
        for forbidden in (
            secret,
            "PROMPT_SENTINEL",
            "RAW_RESPONSE_SENTINEL",
            "BASE64SENTINEL",
            "workspace-private",
            "token=secret",
        ):
            self.assertNotIn(forbidden, persisted)

    def test_pilot_distribution_covers_all_taxonomy_labels(self):
        from scammark.pilot_manifest import select_pilot_records
        from scammark.taxonomy import CANONICAL_LABELS

        scam = []
        for label in CANONICAL_LABELS:
            for index in range(3):
                scam.append(
                    {
                        "sample_id": f"{label}-{index}",
                        "binary": "scam",
                        "scam_type": label,
                        "is_gold": True,
                        "label_source": "gold_manual_json",
                        "boxes": [{"xyxy": [0, 0, 10, 10], "reason": "gold"}],
                        "gold_input_conflict": False,
                        "image_size": [1000, 1000 + index],
                        "mhtml_audit_status": "valid",
                    }
                )
        benign = [
            {
                "sample_id": f"benign-{index}",
                "binary": "benign",
                "scam_type": "none",
                "image_size": [1000, 1000 + index * 10],
                "mhtml_audit_status": "valid",
            }
            for index in range(50)
        ]

        selected = select_pilot_records(scam + benign, size=60, seed=2027)
        selected_scam = [record for record in selected if record["binary"] == "scam"]
        selected_benign = [record for record in selected if record["binary"] == "benign"]
        self.assertEqual(len(selected_scam), 30)
        self.assertEqual(len(selected_benign), 30)
        self.assertEqual({record["scam_type"] for record in selected_scam}, set(CANONICAL_LABELS))
        self.assertNotEqual([record["binary"] for record in selected], ["scam"] * 30 + ["benign"] * 30)

    def test_qwen3_experiment_config_freezes_core_limits_and_pricing(self):
        from scammark.run_experiment import load_experiment_config

        with patch.dict(
            os.environ,
            {"SCAMWEB_DATASET_ROOT": "/data/scamweb_dataset"},
            clear=False,
        ):
            config = load_experiment_config(
                ROOT / "configs" / "qwen3_crazyrouter.toml"
            )

        self.assertEqual(config["models"]["vision"], "qwen3-vl-flash")
        self.assertEqual(config["input_tokens"]["url"], 8192)
        self.assertEqual(config["input_tokens"]["visual"], 4096)
        self.assertEqual(config["output_tokens"]["grounding"], 2048)
        self.assertEqual(config["output_tokens"]["cross"], 1024)
        self.assertEqual(config["experiment"]["api_call_limit"], 600)
        self.assertEqual(config["pricing"]["vision_input_per_million"], 0.05)
        self.assertEqual(config["endpoints"]["text_base_url"], "https://cn.crazyrouter.com/v1")
        self.assertEqual(config["request_limits"]["max_visual_tokens"], 8192)
        self.assertEqual(config["images"]["contact_sheet_base64_bytes"], 2500000)

    def test_dashscope_experiment_config_routes_models_to_separate_providers(self):
        from scammark.run_experiment import load_experiment_config

        config_path = ROOT / "configs" / "qwen3_dashscope.toml"
        self.assertTrue(config_path.is_file())
        with patch.dict(
            os.environ,
            {
                "SCAMMARK_VLM_BASE_URL": (
                    "https://workspace-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                ),
                "SCAMWEB_DATASET_ROOT": "/data/scamweb_dataset",
            },
            clear=False,
        ):
            config = load_experiment_config(config_path)

        self.assertEqual(config["models"]["text"], "gpt-4o-mini")
        self.assertEqual(config["models"]["vision"], "qwen3-vl-flash")
        self.assertEqual(config["endpoints"]["text_provider"], "Crazyrouter")
        self.assertEqual(config["endpoints"]["vision_provider"], "Alibaba Cloud Model Studio")
        self.assertEqual(config["endpoints"]["text_base_url"], "https://cn.crazyrouter.com/v1")
        self.assertEqual(
            config["endpoints"]["vision_base_url"],
            "https://workspace-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(config["pricing"]["text"]["currency"], "USD")
        self.assertEqual(config["pricing"]["vision"]["currency"], "CNY")
        self.assertEqual(
            config["data"]["runtime_dataset_root"],
            "/data/scamweb_dataset",
        )

    def test_full_run_config_has_explicit_safety_contract(self):
        from scammark.run_experiment import load_experiment_config

        config_path = ROOT / "configs" / "qwen3_dashscope_full.toml"
        self.assertTrue(config_path.is_file())
        with patch.dict(
            os.environ,
            {
                "SCAMMARK_VLM_BASE_URL": (
                    "https://workspace-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                ),
                "SCAMWEB_DATASET_ROOT": "/data/scamweb_dataset",
            },
            clear=False,
        ):
            config = load_experiment_config(config_path)

        self.assertTrue(config["experiment"]["allow_full_run"])
        self.assertEqual(config["experiment"]["expected_full_sample_count"], 5658)
        self.assertEqual(config["experiment"]["api_call_limit"], 50000)
        self.assertEqual(config["experiment"]["prediction_flush_interval"], 25)
        self.assertEqual(
            config["paths"]["output_root"],
            "outputs/scammark_qwen3_dashscope_full",
        )

    def test_server_full_config_defers_bertscore_and_keeps_full_safety_limits(self):
        from scammark.run_experiment import load_experiment_config

        config_path = ROOT / "configs" / "qwen3_dashscope_server_full.toml"
        self.assertTrue(config_path.is_file())
        with patch.dict(
            os.environ,
            {
                "SCAMMARK_VLM_BASE_URL": (
                    "https://workspace-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                ),
                "SCAMWEB_DATASET_ROOT": "/data/scamweb_dataset",
            },
            clear=False,
        ):
            config = load_experiment_config(config_path)

        self.assertFalse(config["experiment"]["with_bertscore"])
        self.assertTrue(config["experiment"]["allow_full_run"])
        self.assertEqual(config["experiment"]["api_call_limit"], 50000)
        self.assertEqual(config["experiment"]["expected_full_sample_count"], 5658)
        self.assertEqual(
            config["paths"]["output_root"],
            "outputs/scammark_qwen3_dashscope_server_full",
        )

    def test_full_input_selection_preserves_order_and_requires_exact_evidence(self):
        from scammark.run_experiment import _select_full_inputs

        manifest = [
            {"sample_id": "scam-1", "binary": "scam"},
            {"sample_id": "benign-1", "binary": "benign"},
        ]
        evidence = [
            {"sample_id": "benign-1", "visible_text": "clean"},
            {"sample_id": "scam-1", "visible_text": "risk"},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.jsonl"
            evidence_path = root / "evidence.jsonl"
            output_dir = root / "full"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            evidence_path.write_text(
                "".join(json.dumps(row) + "\n" for row in evidence),
                encoding="utf-8",
            )

            selected_manifest, selected_evidence, selected = _select_full_inputs(
                manifest_path,
                evidence_path,
                output_dir,
                expected_size=2,
                path_mapper=None,
            )

            self.assertEqual([row["sample_id"] for row in selected], ["scam-1", "benign-1"])
            written_evidence = [json.loads(line) for line in selected_evidence.read_text().splitlines()]
            self.assertEqual(
                [row["sample_id"] for row in written_evidence],
                ["scam-1", "benign-1"],
            )
            self.assertTrue(selected_manifest.is_file())

            evidence_path.write_text(json.dumps(evidence[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly match"):
                _select_full_inputs(
                    manifest_path,
                    evidence_path,
                    output_dir,
                    expected_size=2,
                    path_mapper=None,
                )

    def test_prediction_flush_schedule_always_writes_the_final_record(self):
        from scammark.run_experiment import _should_flush_predictions

        self.assertFalse(_should_flush_predictions(1, total=60, interval=25))
        self.assertTrue(_should_flush_predictions(25, total=60, interval=25))
        self.assertTrue(_should_flush_predictions(50, total=60, interval=25))
        self.assertTrue(_should_flush_predictions(60, total=60, interval=25))
        self.assertTrue(_should_flush_predictions(1, total=1, interval=25))

    def test_full_acceptance_allows_deferred_bertscore_when_disabled(self):
        from scammark.run_experiment import _acceptance_gates
        from scammark.taxonomy import CANONICAL_LABELS

        records = [
            {
                "truth": {"binary": "scam", "scam_type": label},
                "prediction": {"binary": "scam"},
            }
            for label in CANONICAL_LABELS
        ]
        metrics = {
            "engineering": {
                "backend_error_count": 0,
                "schema_invalid_count": 0,
                "expected_stage_sets_complete": True,
                "usage_complete": True,
                "image_limits_ok": True,
                "request_limits_ok": True,
                "routing_ok": True,
                "truncated_completion_count": 0,
                "clean_metrics": True,
                "unattributed_api_call_count": 0,
                "oracle_diagnostics_complete": True,
                "api_calls_cumulative": 175,
                "first_attempt_failure_rate": 0.0,
                "average_attempt_count": 1.0,
                "sample_count": len(records),
            },
            "grounding": {"end_to_end": {"counts_at_0_5": {"tp": 1}}},
            "rationale": {"end_to_end": {"bertscore_status": "not_requested"}},
        }
        config = {
            "experiment": {
                "api_call_limit": 50000,
                "expected_full_sample_count": len(records),
                "with_bertscore": False,
            }
        }

        gates = _acceptance_gates(metrics, records, config, pilot_size=None)

        self.assertTrue(gates["passed"])
        self.assertTrue(gates["checks"]["bertscore_deferred_as_configured"])
        self.assertNotIn("bertscore_end_to_end_computed", gates["checks"])

    def test_mixed_provider_costs_are_reported_without_currency_conversion(self):
        import scammark.run_experiment as run_experiment

        self.assertTrue(hasattr(run_experiment, "_estimate_api_cost"))
        cost = run_experiment._estimate_api_cost(
            {
                "status": "published_rate_estimate",
                "access_date": "2026-07-11",
                "text": {
                    "currency": "USD",
                    "input_per_million": 0.10,
                    "output_per_million": 0.42,
                    "source": "text-source",
                },
                "vision": {
                    "currency": "CNY",
                    "input_per_million": 0.15,
                    "output_per_million": 1.50,
                    "source": "vision-source",
                },
            },
            text_usage={"prompt_tokens": 1_000_000, "completion_tokens": 100_000},
            vision_usage={"prompt_tokens": 2_000_000, "completion_tokens": 200_000},
            text_model="gpt-4o-mini",
            vision_model="qwen3-vl-flash",
        )

        self.assertEqual(cost["status"], "published_rate_estimate")
        self.assertAlmostEqual(cost["by_role"]["text"]["estimated_total"], 0.142)
        self.assertAlmostEqual(cost["by_role"]["vision"]["estimated_total"], 0.6)
        self.assertEqual(cost["totals_by_currency"], {"USD": 0.142, "CNY": 0.6})
        self.assertNotIn("estimated_total", cost)


if __name__ == "__main__":
    unittest.main()
