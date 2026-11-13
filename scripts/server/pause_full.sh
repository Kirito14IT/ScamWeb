#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

[[ "$#" == "0" ]] || die "Usage: pause_full.sh"
require_safe_runtime_namespace
require_file "$RUNNER_PID_FILE"

identity=()
if ! mapfile -t identity < <(
    "$VENV_PYTHON" - \
        "$RUNNER_PID_FILE" \
        "$TEST_OVERRIDES_ENABLED" \
        "${SCAMMARK_TEST_RUNNER:-}" \
        "$CONFIG_PATH" \
        "$VENV_PYTHON" <<'PY'
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
test_overrides_enabled = sys.argv[2] == "1"
test_runner = sys.argv[3]
config_path = str(Path(sys.argv[4]).resolve(strict=True))
python_path = str(Path(sys.argv[5]).resolve(strict=True))
flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
descriptor = os.open(path, flags)
try:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > 16_384
    ):
        raise SystemExit("Runner identity is not a bounded private regular file")
    encoded_payload = os.read(descriptor, 16_385)
    if len(encoded_payload) > 16_384 or os.read(descriptor, 1):
        raise SystemExit("Runner identity is oversized")
finally:
    os.close(descriptor)
payload = json.loads(encoded_payload.decode("utf-8", errors="strict"))
expected_fields = {
    "schema",
    "pid",
    "starttime",
    "attempt_id",
    "module",
    "created_at_utc",
}
if type(payload) is not dict or set(payload) != expected_fields:
    raise SystemExit("Runner identity fields are invalid")
if payload.get("schema") != "scammark-runner-identity-v1":
    raise SystemExit("Runner identity schema is invalid")
pid = payload.get("pid")
if type(pid) is not int or pid < 2:
    raise SystemExit("Runner PID is invalid")
raw_starttime = payload.get("starttime")
if type(raw_starttime) not in (int, str) or isinstance(raw_starttime, bool):
    raise SystemExit("Runner starttime is invalid")
starttime = str(raw_starttime)
if not starttime.isdigit() or len(starttime) > 32:
    raise SystemExit("Runner starttime is invalid")
attempt_id = payload.get("attempt_id")
if (
    type(attempt_id) is not str
    or re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", attempt_id) is None
):
    raise SystemExit("Runner attempt identity is invalid")
if payload.get("module") != "scammark.run_experiment":
    raise SystemExit("Runner module identity is invalid")
created_at = payload.get("created_at_utc")
if type(created_at) is not str or len(created_at) > 64:
    raise SystemExit("Runner creation time is invalid")
try:
    parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
except ValueError as error:
    raise SystemExit("Runner creation time is invalid") from error
if (
    parsed_created_at.tzinfo is None
    or parsed_created_at.utcoffset() != timezone.utc.utcoffset(None)
):
    raise SystemExit("Runner creation time is not UTC")

stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
closing = stat_text.rfind(")")
fields_after_comm = stat_text[closing + 2:].split() if closing >= 0 else []
if len(fields_after_comm) <= 19 or fields_after_comm[19] != starttime:
    raise SystemExit("Runner PID starttime is stale")
cmdline_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
if len(cmdline_bytes) > 65_536:
    raise SystemExit("Runner command identity is oversized")
arguments = [
    item.decode("utf-8", errors="strict")
    for item in cmdline_bytes.split(b"\0")
    if item
]
try:
    normalized_python_argument = str(Path(arguments[0]).resolve(strict=True))
except (IndexError, OSError):
    normalized_python_argument = ""
expected_production_arguments = [
    python_path,
    "-m",
    "scammark.run_experiment",
    "--config",
    config_path,
    "--full",
    "--resume",
]
normalized_arguments = [normalized_python_argument, *arguments[1:]]
production_identity = (
    normalized_arguments == expected_production_arguments
    or normalized_arguments == [*expected_production_arguments, "--ack-breaker"]
)
test_identity = False
if test_overrides_enabled and test_runner:
    expected_runner = str(Path(test_runner).resolve(strict=True))
    test_identity = (
        len(arguments) == 2
        and str(Path(arguments[1]).resolve(strict=False)) == expected_runner
    )
if not production_identity and not test_identity:
    raise SystemExit("Runner command identity does not match")

print(pid)
print(starttime)
print(attempt_id)
PY
); then
    die "Runner identity validation failed; no signal was sent"
fi

[[ "${#identity[@]}" == "3" ]] ||
    die "Runner identity validation failed; no signal was sent"
runner_pid="${identity[0]}"
runner_starttime="${identity[1]}"
runner_attempt_id="${identity[2]}"
[[ -n "$runner_attempt_id" ]] ||
    die "Runner identity validation failed; no signal was sent"

current_starttime="$(process_starttime "$runner_pid" 2>/dev/null)" ||
    die "Runner PID disappeared before signalling"
[[ "$current_starttime" == "$runner_starttime" ]] ||
    die "Runner PID was reused before signalling"

poll_seconds="1"
max_seconds="330"
if test_overrides_enabled; then
    poll_seconds="${SCAMMARK_TEST_PAUSE_POLL_SECONDS:-$poll_seconds}"
    max_seconds="${SCAMMARK_TEST_PAUSE_MAX_SECONDS:-$max_seconds}"
fi
read -r poll_seconds max_seconds < <(
    "$VENV_PYTHON" - "$poll_seconds" "$max_seconds" <<'PY'
import math
import sys

poll_seconds = float(sys.argv[1])
max_seconds = float(sys.argv[2])
if (
    not math.isfinite(poll_seconds)
    or not math.isfinite(max_seconds)
    or poll_seconds < 0
    or max_seconds <= 0
    or poll_seconds > 60
    or max_seconds > 330
):
    raise SystemExit("Invalid pause polling interval")
print(poll_seconds, max_seconds)
PY
)

printf 'Sending a verified SIGINT to runner PID %s.\n' "$runner_pid"
kill -INT "$runner_pid"

deadline="$(
    "$VENV_PYTHON" -c "import time; print(time.monotonic() + float('$max_seconds'))"
)"
while kill -0 "$runner_pid" 2>/dev/null; do
    observed_starttime="$(process_starttime "$runner_pid" 2>/dev/null || true)"
    if [[ "$observed_starttime" != "$runner_starttime" ]]; then
        break
    fi
    timed_out="$(
        "$VENV_PYTHON" -c "import time; print(int(time.monotonic() >= float('$deadline')))"
    )"
    if [[ "$timed_out" == "1" ]]; then
        printf 'Runner PID %s is still approaching a safe boundary; check status before signalling again.\n' \
            "$runner_pid"
        exit 0
    fi
    sleep "$poll_seconds"
done

if tmux has-session -t "=$TMUX_SESSION" 2>/dev/null; then
    pane_state=""
    pane_dead=""
    pane_dead_status=""
    for ((pane_poll = 0; pane_poll < 100; pane_poll++)); do
        pane_state="$(tmux_pane_state 2>/dev/null || true)"
        IFS='|' read -r pane_dead pane_dead_status <<<"$pane_state"
        [[ "$pane_dead" == "1" ]] && break
        sleep 0.01
    done
    if [[ "$pane_dead" == "1" ]]; then
        pane_state="$(tmux_pane_state_with_reap_nudge)"
    fi
    printf 'Pause completed; retained pane state: %s\n' "${pane_state:-unknown}"
else
    printf 'Pause completed; the verified runner PID exited.\n'
fi
