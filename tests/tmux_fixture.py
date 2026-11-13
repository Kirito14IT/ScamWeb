from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ManagedTmuxFixture:
    PILOT10_BASELINE_GATE_KEYS = (
        "backend_error_count_zero",
        "schema_error_count_zero",
        "stage_sets_complete",
        "usage_complete",
        "image_limits_ok",
        "request_limits_ok",
        "routing_ok",
        "no_truncated_completions",
        "metrics_inputs_clean",
        "all_api_calls_attributed",
        "oracle_diagnostics_complete",
        "api_call_cap_respected",
    )

    def __init__(self, repository_root: Path) -> None:
        self.source_root = Path(repository_root).resolve()
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name) / "repo"
        self.session = f"scammark-full-v2-test-{uuid.uuid4().hex[:12]}"
        self.socket_name = f"scammark-test-{uuid.uuid4().hex[:12]}"
        self.unrelated_sessions: set[str] = set()
        self._last_identity: dict[str, Any] | None = None
        self._cleaned = False
        self._prepare_repository()

    @property
    def output_root(self) -> Path:
        return self.root / "outputs" / "scammark_qwen3_dashscope_server_full_v2"

    @property
    def runner_path(self) -> Path:
        return self.root / "fake_runner.py"

    @property
    def runner_identity_path(self) -> Path:
        return self.output_root / "runner.json"

    @property
    def event_path(self) -> Path:
        return self.output_root / "full" / "run_events.jsonl"

    @property
    def run_log_path(self) -> Path:
        return self.output_root / "logs" / "full_run.log"

    @property
    def tmux_nudge_log_path(self) -> Path:
        return Path(self._temporary.name) / "tmux-nudge-pre-states.log"

    def _prepare_repository(self) -> None:
        self.root.mkdir(parents=True)
        shutil.copytree(
            self.source_root / "scripts" / "server",
            self.root / "scripts" / "server",
        )
        (self.root / "configs").mkdir()
        (self.root / "configs" / "qwen3_dashscope_server_full_v2.toml").write_text(
            "[experiment]\n"
            "expected_full_sample_count = 5658\n\n"
            "[paths]\n"
            'output_root = "outputs/scammark_qwen3_dashscope_server_full_v2"\n',
            encoding="utf-8",
        )
        (self.root / "src").symlink_to(self.source_root / "src", target_is_directory=True)

        pilot_dir = self.output_root / "pilot_10"
        full_dir = self.output_root / "full"
        pilot_dir.mkdir(parents=True)
        full_dir.mkdir(parents=True)
        (pilot_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "engineering": {
                        "sample_count": 10,
                        "acceptance_gates": {
                            "passed": True,
                            "checks": {
                                key: True
                                for key in self.PILOT10_BASELINE_GATE_KEYS
                            },
                        },
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (full_dir / "preflight.json").write_text(
            json.dumps(
                {
                    "status": "validated",
                    "run_mode": "full",
                    "selected_sample_count": 5658,
                    "expected_full_sample_count": 5658,
                    "contract_image_count": 5658,
                    "api_call_limit": 50_000,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        env_path = self.root / ".env"
        env_path.write_text(
            "\n".join(
                (
                    "SCAMMARK_LLM_API_KEY=fixture-key",
                    "SCAMMARK_VLM_API_KEY=fixture-key",
                    "SCAMMARK_VLM_BASE_URL=https://fixture.invalid/v1",
                    "SCAMWEB_DATASET_ROOT=/fixture/dataset",
                    "",
                )
            ),
            encoding="utf-8",
        )
        env_path.chmod(0o600)

        self.runner_path.write_text(
            """from __future__ import annotations
import json
import os
import signal
import sys
import time
from pathlib import Path

root = Path.cwd()
event_path = root / "outputs/scammark_qwen3_dashscope_server_full_v2/full/run_events.jsonl"
event_path.parent.mkdir(parents=True, exist_ok=True)
attempt_id = os.environ.get("SCAMMARK_ATTEMPT_ID", "")

def stop(signum, frame):
    del signum, frame
    row = {
        "event": "run_stopped",
        "attempt_id": attempt_id,
        "pid": os.getpid(),
        "reason": "graceful_pause",
        "exit_code": 130,
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"FAKE_TEE_DRAIN_MARKER pid={os.getpid()}", flush=True)
    raise SystemExit(130)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
print(f"FAKE_RUNNER_STARTED pid={os.getpid()}", flush=True)
immediate_exit_code = os.environ.get("SCAMMARK_FIXTURE_IMMEDIATE_EXIT_CODE", "")
if immediate_exit_code:
    print(f"FAKE_TEE_DRAIN_MARKER pid={os.getpid()}", flush=True)
    os._exit(int(immediate_exit_code))
while True:
    time.sleep(0.05)
""",
            encoding="utf-8",
        )

        venv_bin = self.root / ".venv-server" / "bin"
        venv_bin.mkdir(parents=True)
        python_shim = venv_bin / "python"
        python_shim.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${SCAMMARK_FIXTURE_IDENTITY_WRITE_FAIL:-}" == "1" \
    && "${1:-}" == "-" \
    && "${2:-}" == "$PWD/outputs/scammark_qwen3_dashscope_server_full_v2/runner.json" \
    && "$#" == "5" ]]; then
    exit 1
fi
if [[ "${SCAMMARK_FIXTURE_MODULE_DECOY:-}" == "1" ]]; then
    exec -a "$0" "$SCAMMARK_FIXTURE_REAL_PYTHON" "$@"
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "scammark.run_experiment" ]]; then
    exec "$SCAMMARK_FIXTURE_REAL_PYTHON" "$SCAMMARK_TEST_RUNNER"
fi
exec "$SCAMMARK_FIXTURE_REAL_PYTHON" "$@"
""",
            encoding="utf-8",
        )
        python_shim.chmod(0o755)

        bin_dir = self.root / ".fixture-bin"
        bin_dir.mkdir()
        tmux_wrapper = bin_dir / "tmux"
        tmux_wrapper.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
args=()
for value in "$@"; do
    case "$value" in
        scammark-full-v2) value="$SCAMMARK_FIXTURE_SESSION" ;;
        =scammark-full-v2) value="=$SCAMMARK_FIXTURE_SESSION" ;;
        =scammark-full-v2:0.0) value="=$SCAMMARK_FIXTURE_SESSION:0.0" ;;
    esac
    args+=("$value")
done
if [[ "${args[0]:-}" == "run-shell" ]]; then
    "$SCAMMARK_FIXTURE_REAL_TMUX" -L "$SCAMMARK_FIXTURE_SOCKET" \
        display-message -p \
        -t "=$SCAMMARK_FIXTURE_SESSION:0.0" \
        '#{pane_dead}|#{pane_dead_status}' \
        >>"$SCAMMARK_FIXTURE_TMUX_NUDGE_LOG"
fi
exec "$SCAMMARK_FIXTURE_REAL_TMUX" -L "$SCAMMARK_FIXTURE_SOCKET" "${args[@]}"
""",
            encoding="utf-8",
        )
        tmux_wrapper.chmod(0o755)

        (self.root / ".gitignore").write_text(
            """outputs/scammark_qwen3_dashscope_server_full_v2/runner.json
outputs/scammark_qwen3_dashscope_server_full_v2/logs/
outputs/scammark_qwen3_dashscope_server_full_v2/full/run_events.jsonl
""",
            encoding="utf-8",
        )
        self.tmux_tmpdir = Path(self._temporary.name) / "tmux"
        self.tmux_tmpdir.mkdir()

        self.env = os.environ.copy()
        for key in (
            "SCAMMARK_LLM_API_KEY",
            "SCAMMARK_LLM_BASE_URL",
            "SCAMMARK_VLM_API_KEY",
            "SCAMMARK_VLM_BASE_URL",
        ):
            self.env.pop(key, None)
        real_tmux = shutil.which("tmux")
        if not real_tmux:
            raise RuntimeError("tmux is unavailable")
        self.env.update(
            {
                "PATH": f"{bin_dir}{os.pathsep}{self.env.get('PATH', '')}",
                "TMUX_TMPDIR": str(self.tmux_tmpdir),
                "SCAMMARK_FIXTURE_REAL_TMUX": real_tmux,
                "SCAMMARK_FIXTURE_SOCKET": self.socket_name,
                "SCAMMARK_FIXTURE_SESSION": self.session,
                "SCAMMARK_FIXTURE_TMUX_NUDGE_LOG": str(
                    self.tmux_nudge_log_path
                ),
                "SCAMMARK_FIXTURE_REAL_PYTHON": sys.executable,
                "SCAMMARK_TEST_TMUX_SESSION": self.session,
                "SCAMMARK_TEST_PAUSE_POLL_SECONDS": "0.01",
                "SCAMMARK_TEST_PAUSE_MAX_SECONDS": "0.5",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )

        self._git("init", "-q")
        self._git("config", "user.name", "ScamMark Fixture")
        self._git("config", "user.email", "fixture@example.invalid")
        self._git("add", "-A")
        self._git("commit", "-qm", "fixture")
        self._git("tag", "experiment-qwen3-dashscope-full-v2")

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        )

    def _commit_fixture_contract(self, message: str) -> None:
        self._git("add", "-A")
        self._git("commit", "-qm", message)
        self._git("tag", "-f", "experiment-qwen3-dashscope-full-v2")

    def production_checkout_state(self) -> tuple[str, set[str]]:
        status = self._git("status", "--porcelain").stdout
        tags = set(
            self._git("tag", "--points-at", "HEAD").stdout.splitlines()
        )
        return status, tags

    def replace_pilot_checks(self, checks: dict[str, bool] | None) -> None:
        acceptance_gates: dict[str, Any] = {"passed": True}
        if checks is not None:
            acceptance_gates["checks"] = checks
        path = self.output_root / "pilot_10" / "metrics.json"
        path.write_text(
            json.dumps(
                {
                    "engineering": {
                        "sample_count": 10,
                        "acceptance_gates": acceptance_gates,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self._commit_fixture_contract("replace pilot gate contract")

    def replace_preflight(self, payload: object, *, raw: bool = False) -> None:
        path = self.output_root / "full" / "preflight.json"
        text = str(payload) if raw else json.dumps(payload) + "\n"
        path.write_text(text, encoding="utf-8")
        self._commit_fixture_contract("replace preflight contract")

    def _tmux(
        self, *args: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["tmux", *args],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            check=check,
        )

    def run_start(
        self,
        *,
        allow_test_overrides: str | None = "1",
        test_runner: str | None = "fake",
        ack_breaker: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        env = self.env.copy()
        if allow_test_overrides is None:
            env.pop("SCAMMARK_ALLOW_TEST_OVERRIDES", None)
        else:
            env["SCAMMARK_ALLOW_TEST_OVERRIDES"] = allow_test_overrides
        if test_runner is None:
            env.pop("SCAMMARK_TEST_RUNNER", None)
        else:
            env["SCAMMARK_TEST_RUNNER"] = str(self.runner_path)
        if allow_test_overrides is None and test_runner is None:
            for name in (
                "SCAMMARK_TEST_TMUX_SESSION",
                "SCAMMARK_TEST_PAUSE_POLL_SECONDS",
                "SCAMMARK_TEST_PAUSE_MAX_SECONDS",
                "SCAMMARK_FIXTURE_MODULE_DECOY",
            ):
                env.pop(name, None)
        command = ["bash", "scripts/server/start_full.sh"]
        if ack_breaker:
            command.append("--ack-breaker")
        return subprocess.run(
            command,
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def start_fake_runner(self, *, ack_breaker: bool = False) -> None:
        completed = self.run_start(ack_breaker=ack_breaker)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)

    def configure_immediate_fake_exit(self, exit_code: int) -> None:
        self.env["SCAMMARK_FIXTURE_IMMEDIATE_EXIT_CODE"] = str(exit_code)

    def install_v1_runtime_symlink(self, location: str) -> None:
        legacy_root = (
            self.root
            / "outputs"
            / "scammark_qwen3_dashscope_server_full"
        )
        for directory in (
            legacy_root,
            legacy_root / "full",
            legacy_root / "logs",
            legacy_root / "state_full",
            legacy_root / "pilot_10",
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"sentinel-{directory.name}.txt").write_text(
                f"untouched legacy sentinel: {directory.name}\n",
                encoding="utf-8",
            )
        (legacy_root / "runner.json").write_text(
            '{"sentinel":"untouched legacy runner"}\n',
            encoding="utf-8",
        )
        legacy_files = {
            legacy_root / "logs" / "full_run.log": "untouched legacy run log\n",
            legacy_root / "full" / "run_events.jsonl": "untouched legacy events\n",
            legacy_root / "full" / "preflight.json": '{"sentinel":"legacy preflight"}\n',
            legacy_root / "pilot_10" / "metrics.json": '{"sentinel":"legacy pilot metrics"}\n',
        }
        for path, content in legacy_files.items():
            path.write_text(content, encoding="utf-8")

        links = {
            "output-root": (self.output_root, legacy_root, True),
            "full": (self.output_root / "full", legacy_root / "full", True),
            "logs": (self.output_root / "logs", legacy_root / "logs", True),
            "state": (
                self.output_root / "state_full",
                legacy_root / "state_full",
                True,
            ),
            "runner": (
                self.runner_identity_path,
                legacy_root / "runner.json",
                False,
            ),
            "run-log": (
                self.run_log_path,
                legacy_root / "logs" / "full_run.log",
                False,
            ),
            "events": (
                self.event_path,
                legacy_root / "full" / "run_events.jsonl",
                False,
            ),
            "preflight": (
                self.output_root / "full" / "preflight.json",
                legacy_root / "full" / "preflight.json",
                False,
            ),
            "pilot": (
                self.output_root / "pilot_10",
                legacy_root / "pilot_10",
                True,
            ),
            "pilot-metrics": (
                self.output_root / "pilot_10" / "metrics.json",
                legacy_root / "pilot_10" / "metrics.json",
                False,
            ),
        }
        hardlink = location == "run-log-hardlink"
        lookup_location = "run-log" if hardlink else location
        link, target, target_is_directory = links[lookup_location]
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        link.parent.mkdir(parents=True, exist_ok=True)
        if hardlink:
            os.link(target, link)
        else:
            link.symlink_to(target, target_is_directory=target_is_directory)
        self._legacy_runtime_root = legacy_root
        self._commit_fixture_contract(f"install {location} legacy runtime alias")

    def install_controlled_outputs_parent_symlink(self) -> None:
        outputs = self.root / "outputs"
        controlled_outputs = self.root / "controlled-outputs"
        outputs.rename(controlled_outputs)
        outputs.symlink_to(controlled_outputs, target_is_directory=True)
        self._commit_fixture_contract("install controlled outputs parent symlink")

    def legacy_runtime_inventory(self) -> tuple[tuple[object, ...], ...]:
        root = self._legacy_runtime_root
        inventory: list[tuple[object, ...]] = []
        for path in sorted((root, *root.rglob("*"))):
            metadata = path.lstat()
            relative = str(path.relative_to(root)) or "."
            if path.is_symlink():
                content: object = ("symlink", os.readlink(path))
            elif path.is_file():
                content = ("file", path.read_bytes())
            else:
                content = ("directory", None)
            inventory.append(
                (
                    relative,
                    metadata.st_mode,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    content,
                )
            )
        return tuple(inventory)

    def invalidate_production_prerequisites(self) -> None:
        (self.root / ".env").unlink(missing_ok=True)
        (self.output_root / "pilot_10" / "metrics.json").unlink(missing_ok=True)
        (self.output_root / "full" / "preflight.json").unlink(missing_ok=True)
        marker = self.root / "invalid-production-prerequisites"
        marker.write_text("fixture\n", encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-qm", "invalidate production prerequisites")

    def force_identity_write_failure(self) -> None:
        self.env["SCAMMARK_FIXTURE_IDENTITY_WRITE_FAIL"] = "1"

    @staticmethod
    def _process_starttime(pid: int) -> str:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return text[text.rfind(")") + 2 :].split()[19]

    @classmethod
    def _process_identity_alive(cls, pid: int, starttime: str) -> bool:
        try:
            return cls._process_starttime(pid) == starttime
        except (FileNotFoundError, ProcessLookupError, OSError, ValueError):
            return False

    def start_module_token_decoy(self) -> subprocess.Popen[str]:
        decoy_path = self.root / "decoy.py"
        marker = self.root / "decoy-signal-marker"
        started = self.root / "decoy-started"
        decoy_path.write_text(
            """import signal
import time
from pathlib import Path

marker = Path("decoy-signal-marker")
started = Path("decoy-started")

def handle(signum, frame):
    del signum, frame
    marker.write_text("signal received\\n", encoding="utf-8")

signal.signal(signal.SIGINT, handle)
started.write_text("ready\\n", encoding="utf-8")
while True:
    time.sleep(0.05)
""",
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(decoy_path),
                "--unrelated",
                "-m",
                "scammark.run_experiment",
            ],
            cwd=self.root,
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._wait(lambda: started.is_file(), "decoy startup")
        payload = {
            "schema": "scammark-runner-identity-v1",
            "pid": process.pid,
            "starttime": self._process_starttime(process.pid),
            "attempt_id": "fixture-decoy-attempt",
            "module": "scammark.run_experiment",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.runner_identity_path.parent.mkdir(parents=True, exist_ok=True)
        self.runner_identity_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._decoy_marker = marker
        return process

    def start_wrong_config_module_decoy(self) -> subprocess.Popen[str]:
        module_root = self.root / "wrong-config-module"
        package = module_root / "scammark"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        marker = self.root / "wrong-config-signal-marker"
        started = self.root / "wrong-config-started"
        (package / "run_experiment.py").write_text(
            """import signal
import time
from pathlib import Path

marker = Path("wrong-config-signal-marker")
started = Path("wrong-config-started")

def handle(signum, frame):
    del signum, frame
    marker.write_text("signal received\\n", encoding="utf-8")

signal.signal(signal.SIGINT, handle)
started.write_text("ready\\n", encoding="utf-8")
while True:
    time.sleep(0.05)
""",
            encoding="utf-8",
        )
        wrong_config = self.root / "wrong-config.toml"
        wrong_config.write_text("[experiment]\n", encoding="utf-8")
        env = self.env.copy()
        env["SCAMMARK_FIXTURE_MODULE_DECOY"] = "1"
        env["PYTHONPATH"] = (
            f"{module_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        )
        process = subprocess.Popen(
            [
                str(self.root / ".venv-server" / "bin" / "python"),
                "-m",
                "scammark.run_experiment",
                "--config",
                str(wrong_config),
                "--full",
                "--resume",
            ],
            cwd=self.root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._wait(lambda: started.is_file(), "wrong-config decoy startup")
        payload = {
            "schema": "scammark-runner-identity-v1",
            "pid": process.pid,
            "starttime": self._process_starttime(process.pid),
            "attempt_id": "fixture-wrong-config-attempt",
            "module": "scammark.run_experiment",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.runner_identity_path.parent.mkdir(parents=True, exist_ok=True)
        self.runner_identity_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._wrong_config_decoy_marker = marker
        return process

    def start_exact_production_module_decoy(self) -> subprocess.Popen[str]:
        module_root = self.root / "exact-production-module"
        package = module_root / "scammark"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        marker = self.root / "production-decoy-signal-marker"
        started = self.root / "production-decoy-started"
        (package / "run_experiment.py").write_text(
            """import signal
import time
from pathlib import Path

marker = Path("production-decoy-signal-marker")
started = Path("production-decoy-started")

def handle(signum, frame):
    del signum, frame
    marker.write_text("signal received\\n", encoding="utf-8")
    raise SystemExit(130)

signal.signal(signal.SIGINT, handle)
started.write_text("ready\\n", encoding="utf-8")
while True:
    time.sleep(0.05)
""",
            encoding="utf-8",
        )
        env = self.env.copy()
        env["SCAMMARK_FIXTURE_MODULE_DECOY"] = "1"
        env["PYTHONPATH"] = (
            f"{module_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        )
        process = subprocess.Popen(
            [
                str(self.root / ".venv-server" / "bin" / "python"),
                "-m",
                "scammark.run_experiment",
                "--config",
                str(
                    self.root
                    / "configs"
                    / "qwen3_dashscope_server_full_v2.toml"
                ),
                "--full",
                "--resume",
            ],
            cwd=self.root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._wait(lambda: started.is_file(), "production decoy startup")
        payload = {
            "schema": "scammark-runner-identity-v1",
            "pid": process.pid,
            "starttime": self._process_starttime(process.pid),
            "attempt_id": "fixture-production-decoy-attempt",
            "module": "scammark.run_experiment",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.runner_identity_path.parent.mkdir(parents=True, exist_ok=True)
        self.runner_identity_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._production_decoy_marker = marker
        return process

    def decoy_received_signal(self) -> bool:
        return getattr(self, "_decoy_marker", Path("/nonexistent")).is_file()

    def wrong_config_decoy_received_signal(self) -> bool:
        return getattr(
            self, "_wrong_config_decoy_marker", Path("/nonexistent")
        ).is_file()

    def production_decoy_received_signal(self) -> bool:
        return getattr(
            self, "_production_decoy_marker", Path("/nonexistent")
        ).is_file()

    @staticmethod
    def terminate_decoy(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    @staticmethod
    def _wait(predicate: Any, description: str) -> Any:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {description}")

    def wait_for_pid_file(self) -> dict[str, Any]:
        def ready() -> dict[str, Any] | None:
            try:
                identity = json.loads(
                    self.runner_identity_path.read_text(encoding="utf-8")
                )
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return None
            if "FAKE_RUNNER_STARTED" not in self.run_log():
                return None
            expected = {
                "schema",
                "pid",
                "starttime",
                "attempt_id",
                "module",
                "created_at_utc",
            }
            if set(identity) != expected:
                raise AssertionError(f"unexpected runner identity: {identity}")
            self._last_identity = identity
            return identity

        return self._wait(ready, "verified runner identity")

    def wait_for_fake_runner_start(self) -> int:
        def started_pid() -> int | None:
            match = re.search(r"FAKE_RUNNER_STARTED pid=([0-9]+)", self.run_log())
            return int(match.group(1)) if match else None

        return self._wait(started_pid, "fake runner startup")

    def wait_for_fake_start_or_pane_dead(self) -> str:
        def outcome() -> str | None:
            if "FAKE_RUNNER_STARTED" in self.run_log():
                return "fake-started"
            state = self._pane_state()
            if state is not None and state[0] == "1":
                return "pane-dead"
            return None

        return self._wait(outcome, "fake runner startup or dead pane")

    def fake_runner_process_alive(self, pid: int) -> bool:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, OSError):
            return False
        expected = os.fsencode(str(self.runner_path.resolve()))
        return expected in cmdline

    def terminate_fake_runner_if_alive(self, pid: int) -> None:
        if not self.fake_runner_process_alive(pid):
            return
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not self.fake_runner_process_alive(pid):
                return
            time.sleep(0.02)
        if self.fake_runner_process_alive(pid):
            os.kill(pid, signal.SIGKILL)

    def _known_fake_runner_pids(self) -> set[int]:
        pids = {
            int(value)
            for value in re.findall(
                r"FAKE_RUNNER_STARTED pid=([0-9]+)", self.run_log()
            )
        }
        if self._last_identity is not None:
            pid = self._last_identity.get("pid")
            if type(pid) is int:
                pids.add(pid)
        try:
            if self.runner_identity_path.stat().st_size <= 16_384:
                payload = json.loads(
                    self.runner_identity_path.read_text(encoding="utf-8")
                )
                pid = payload.get("pid") if type(payload) is dict else None
                if type(pid) is int:
                    pids.add(pid)
        except (FileNotFoundError, IsADirectoryError, OSError, ValueError):
            pass
        return pids

    def run_pause(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "scripts/server/pause_full.sh"],
            cwd=self.root,
            env={
                **self.env,
                "SCAMMARK_ALLOW_TEST_OVERRIDES": "1",
                "SCAMMARK_TEST_RUNNER": str(self.runner_path),
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def run_pause_with_environment(
        self,
        *,
        allow_test_overrides: str | None,
        test_runner: str | None,
        poll_seconds: str,
        max_seconds: str,
    ) -> subprocess.CompletedProcess[str]:
        env = self.env.copy()
        if allow_test_overrides is None:
            env.pop("SCAMMARK_ALLOW_TEST_OVERRIDES", None)
        else:
            env["SCAMMARK_ALLOW_TEST_OVERRIDES"] = allow_test_overrides
        if test_runner is None:
            env.pop("SCAMMARK_TEST_RUNNER", None)
        else:
            env["SCAMMARK_TEST_RUNNER"] = str(self.runner_path)
        env["SCAMMARK_TEST_PAUSE_POLL_SECONDS"] = poll_seconds
        env["SCAMMARK_TEST_PAUSE_MAX_SECONDS"] = max_seconds
        return subprocess.run(
            ["bash", "scripts/server/pause_full.sh"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def _pane_state(self) -> tuple[str, str] | None:
        result = self._tmux(
            "display-message",
            "-p",
            "-t",
            f"={self.session}:0.0",
            "#{pane_dead}|#{pane_dead_status}",
        )
        if result.returncode != 0:
            return None
        dead, status = result.stdout.strip().split("|", 1)
        return dead, status

    def wait_for_dead_pane(self) -> None:
        def retained() -> tuple[str, str] | None:
            state = self._pane_state()
            if state is None or state[0] != "1" or not state[1].isdigit():
                return None
            return state

        try:
            self._wait(retained, "retained dead pane")
        except AssertionError as error:
            raise AssertionError(
                "timed out waiting for retained dead pane; "
                f"state={self._pane_state()!r}; "
                f"identity_exists={self.runner_identity_exists()!r}; "
                f"log_tail={self.run_log()[-500:]!r}"
            ) from error

    def wait_for_pane_dead(self) -> None:
        def dead() -> tuple[str, str] | None:
            state = self._pane_state()
            if state is None or state[0] != "1":
                return None
            return state

        self._wait(dead, "dead pane")

    def wait_for_dead_pane_status(self) -> int:
        self.wait_for_pane_dead()
        state = self._pane_state()
        if state is not None and state[0] == "1" and not state[1]:
            self._tmux(
                "run-shell",
                "-b",
                "-t",
                f"={self.session}:0.0",
                "true",
                check=True,
            )

        def retained_status() -> int | None:
            current = self._pane_state()
            if current is None or current[0] != "1" or not current[1].isdigit():
                return None
            return int(current[1])

        return self._wait(retained_status, "retained dead pane status")

    def pane_dead_status(self) -> int:
        state = self._pane_state()
        if state is None or state[0] != "1":
            raise AssertionError(f"pane is not dead: {state}")
        return int(state[1])

    def tmux_nudge_pre_states(self) -> list[tuple[str, str]]:
        try:
            lines = self.tmux_nudge_log_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [tuple(line.split("|", 1)) for line in lines]

    def session_id(self) -> str:
        result = self._tmux(
            "display-message",
            "-p",
            "-t",
            f"={self.session}:0.0",
            "#{session_id}",
            check=True,
        )
        return result.stdout.strip()

    def events(self) -> list[dict[str, Any]]:
        if not self.event_path.is_file():
            return []
        rows = [
            json.loads(line)
            for line in self.event_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if self._last_identity is None:
            return rows
        attempt_id = self._last_identity["attempt_id"]
        return [row for row in rows if row.get("attempt_id") == attempt_id]

    def run_log(self) -> str:
        try:
            return self.run_log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def tee_process_drained(self, exit_code: int = 130) -> bool:
        log = self.run_log()
        marker = log.find("FAKE_TEE_DRAIN_MARKER")
        footer = log.find(f"exit code {exit_code}")
        return marker >= 0 and footer > marker

    def wait_for_runner_and_tee_exit(self) -> None:
        runner_argument = os.fsencode(str(self.runner_path.resolve()))
        log_argument = os.fsencode(str(self.run_log_path.resolve()))

        def residual_pids() -> list[int]:
            residual: list[int] = []
            for proc_path in Path("/proc").glob("[0-9]*"):
                try:
                    arguments = (proc_path / "cmdline").read_bytes().split(b"\0")
                except (FileNotFoundError, ProcessLookupError, OSError):
                    continue
                if runner_argument in arguments:
                    residual.append(int(proc_path.name))
                    continue
                if (
                    log_argument in arguments
                    and any(Path(os.fsdecode(value)).name == "tee" for value in arguments if value)
                ):
                    residual.append(int(proc_path.name))
            return residual

        def drained() -> bool | None:
            return True if not residual_pids() else None

        try:
            self._wait(drained, "runner and tee process exit")
        except AssertionError as error:
            raise AssertionError(
                f"runner/tee processes remained: {residual_pids()}"
            ) from error

    def runner_identity_exists(self) -> bool:
        return self.runner_identity_path.exists()

    def session_exists(self, name: str | None = None) -> bool:
        target = name or self.session
        return self._tmux("has-session", "-t", f"={target}").returncode == 0

    def create_unrelated_session(self) -> str:
        name = f"unrelated-{uuid.uuid4().hex[:12]}"
        self._tmux(
            "new-session", "-d", "-s", name, "exec sleep infinity", check=True
        )
        self.unrelated_sessions.add(name)
        return name

    def create_unrelated_session_with_stale_test_environment(self) -> str:
        name = f"unrelated-{uuid.uuid4().hex[:12]}"
        stale_env = self.env.copy()
        stale_env.update(
            {
                "SCAMMARK_ALLOW_TEST_OVERRIDES": "1",
                "SCAMMARK_TEST_RUNNER": str(self.runner_path),
                "SCAMMARK_TEST_TMUX_SESSION": self.session,
                "SCAMMARK_TEST_PAUSE_POLL_SECONDS": "0.01",
                "SCAMMARK_TEST_PAUSE_MAX_SECONDS": "0.5",
                "SCAMMARK_FIXTURE_MODULE_DECOY": "",
            }
        )
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                name,
                "exec sleep infinity",
            ],
            cwd=self.root,
            env=stale_env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.unrelated_sessions.add(name)
        return name

    def kill_exact_session(self, name: str) -> None:
        if self.session_exists(name):
            self._tmux("kill-session", "-t", f"={name}", check=True)
        self.unrelated_sessions.discard(name)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        tmux_identities: set[tuple[int, str]] = set()
        pane_process = self._tmux(
            "display-message",
            "-p",
            "-t",
            f"={self.session}:0.0",
            "#{pane_pid}|#{pid}",
        )
        if pane_process.returncode == 0:
            for value in pane_process.stdout.strip().split("|", 1):
                try:
                    pid = int(value)
                    tmux_identities.add((pid, self._process_starttime(pid)))
                except (FileNotFoundError, ProcessLookupError, OSError, ValueError):
                    pass

        fake_runner_pids = self._known_fake_runner_pids()
        for pid in fake_runner_pids:
            self.terminate_fake_runner_if_alive(pid)
        if self.session_exists(self.session):
            self._tmux("kill-session", "-t", f"={self.session}")
        for name in tuple(self.unrelated_sessions):
            self.kill_exact_session(name)
        self._tmux("kill-server")
        for pid in fake_runner_pids:
            self.terminate_fake_runner_if_alive(pid)

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not any(
                self._process_identity_alive(pid, starttime)
                for pid, starttime in tmux_identities
            ):
                break
            time.sleep(0.02)
        remaining = [
            pid
            for pid, starttime in tmux_identities
            if self._process_identity_alive(pid, starttime)
        ]
        if remaining:
            raise AssertionError(
                f"fixture tmux processes did not exit cleanly: {remaining}"
            )
        self._temporary.cleanup()

    def __enter__(self) -> "ManagedTmuxFixture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.cleanup()
