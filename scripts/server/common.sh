#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"

readonly REPOSITORY="nlixsys/ScamWeb-V"
readonly RELEASE_TAG="experiment-qwen3-dashscope-full-v2"
readonly RESULTS_BRANCH="results/qwen3-dashscope-full-v2"
readonly FROZEN_ASSET="scamweb-v1.0-frozen-inputs.tar.gz"
readonly INFERENCE_ASSET="scammark-qwen3-dashscope-full-v2-inference.tar.gz"
readonly FINAL_ASSET="scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz"
readonly CONFIG_PATH="$ROOT_DIR/configs/qwen3_dashscope_server_full_v2.toml"
readonly DEFAULT_DATASET_ROOT="/data/scamweb_dataset"
readonly ENV_FILE="$ROOT_DIR/.env"
readonly FROZEN_DIR="$ROOT_DIR/outputs/dataset_frozen_v1"
readonly OUTPUT_ROOT="$ROOT_DIR/outputs/scammark_qwen3_dashscope_server_full_v2"
readonly FULL_DIR="$OUTPUT_ROOT/full"
readonly STATE_DIR="$OUTPUT_ROOT/state_full"
readonly PILOT_DIR="$OUTPUT_ROOT/pilot_10"
readonly DIST_DIR="$ROOT_DIR/dist/release-assets/full-v2"
readonly DOWNLOAD_DIR="$ROOT_DIR/dist/frozen-download/full-v2"
readonly LOG_DIR="$OUTPUT_ROOT/logs"
readonly RUN_LOG="$LOG_DIR/full_run.log"
readonly RUN_EVENTS_FILE="$FULL_DIR/run_events.jsonl"
readonly RUNNER_PID_FILE="$OUTPUT_ROOT/runner.json"
readonly PREFLIGHT_FILE="$FULL_DIR/preflight.json"
readonly PILOT_METRICS_FILE="$PILOT_DIR/metrics.json"
readonly DELIVERY_STATE="$FULL_DIR/delivery_state.json"
readonly FINALIZE_LOG="$LOG_DIR/bertscore_finalize.log"
readonly VENV_DIR="$ROOT_DIR/.venv-server"
readonly VENV_PYTHON="$VENV_DIR/bin/python"

TEST_OVERRIDES_ENABLED=0
if [[ "${SCAMMARK_ALLOW_TEST_OVERRIDES:-}" == "1" && -n "${SCAMMARK_TEST_RUNNER:-}" ]]; then
    TEST_OVERRIDES_ENABLED=1
fi
readonly TEST_OVERRIDES_ENABLED

TMUX_SESSION_VALUE="scammark-full-v2"
if [[ "$TEST_OVERRIDES_ENABLED" == "1" && -n "${SCAMMARK_TEST_TMUX_SESSION:-}" ]]; then
    if [[ ! "$SCAMMARK_TEST_TMUX_SESSION" =~ ^scammark-full-v2-test-[A-Za-z0-9_.-]{1,48}$ ]]; then
        printf 'ERROR: Invalid guarded test tmux session\n' >&2
        exit 1
    fi
    TMUX_SESSION_VALUE="$SCAMMARK_TEST_TMUX_SESSION"
fi
readonly TMUX_SESSION="$TMUX_SESSION_VALUE"
unset TMUX_SESSION_VALUE

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command is missing: $1"
}

require_file() {
    [[ -f "$1" ]] || die "Required file is missing: $1"
}

test_overrides_enabled() {
    [[ "$TEST_OVERRIDES_ENABLED" == "1" ]]
}

require_clean_worktree() {
    [[ -z "$(git -C "$ROOT_DIR" status --short)" ]] ||
        die "The frozen experiment checkout is dirty; commit/tag a new version before running"
}

require_safe_runtime_namespace() {
    "$VENV_PYTHON" - \
        "$OUTPUT_ROOT" \
        "$FULL_DIR" \
        "$STATE_DIR" \
        "$LOG_DIR" \
        "$PILOT_DIR" \
        "$RUNNER_PID_FILE" \
        "$RUN_LOG" \
        "$RUN_EVENTS_FILE" \
        "$PREFLIGHT_FILE" \
        "$PILOT_METRICS_FILE" <<'PY'
import os
import stat
import sys
from pathlib import Path

(
    output_root,
    full_dir,
    state_dir,
    log_dir,
    pilot_dir,
    runner_file,
    run_log,
    run_events,
    preflight_file,
    pilot_metrics,
) = map(Path, sys.argv[1:])
directory_paths = (output_root, full_dir, state_dir, log_dir, pilot_dir)
private_file_paths = (
    runner_file,
    run_log,
    run_events,
    preflight_file,
    pilot_metrics,
)
all_paths = (*directory_paths, *private_file_paths)
metadata_by_path = {}

for path in all_paths:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        continue
    metadata_by_path[path] = metadata
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or (
        reparse_flag and file_attributes & reparse_flag
    ):
        raise SystemExit(
            f"Unsafe v2 runtime namespace: {path} is a symlink or reparse point"
        )

for path in directory_paths:
    metadata = metadata_by_path.get(path)
    if metadata is not None and not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit(
            f"Unsafe v2 runtime namespace: {path} is not a directory"
        )

for path in private_file_paths:
    metadata = metadata_by_path.get(path)
    if metadata is not None and (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise SystemExit(
            f"Unsafe v2 runtime namespace: {path} is not a private regular file"
        )

output_root_metadata = metadata_by_path.get(output_root)
if output_root_metadata is not None:
    for path in all_paths:
        if path == output_root:
            continue
        metadata = metadata_by_path.get(path)
        if metadata is None:
            continue
        if stat.S_ISDIR(metadata.st_mode) and os.path.ismount(path):
            raise SystemExit(
                f"Unsafe v2 runtime namespace: {path} contains a child mount point"
            )
        if metadata.st_dev != output_root_metadata.st_dev:
            raise SystemExit(
                f"Unsafe v2 runtime namespace: {path} crosses the v2 output device boundary"
            )

legacy_namespace = "scammark_qwen3_dashscope_server_full"
for path in all_paths:
    resolved = path.resolve(strict=False)
    if legacy_namespace in resolved.parts:
        raise SystemExit(
            f"Unsafe v2 runtime namespace: {path} resolves into the v1 namespace"
        )

legacy_root = output_root.parent / legacy_namespace
legacy_directory_pairs = (
    (output_root, legacy_root),
    (full_dir, legacy_root / "full"),
    (state_dir, legacy_root / "state_full"),
    (log_dir, legacy_root / "logs"),
    (pilot_dir, legacy_root / "pilot_10"),
)
for v2_path, v1_path in legacy_directory_pairs:
    v2_metadata = metadata_by_path.get(v2_path)
    if v2_metadata is None:
        continue
    try:
        v1_metadata = v1_path.stat()
    except (FileNotFoundError, OSError):
        continue
    if (
        stat.S_ISDIR(v1_metadata.st_mode)
        and (v2_metadata.st_dev, v2_metadata.st_ino)
        == (v1_metadata.st_dev, v1_metadata.st_ino)
    ):
        raise SystemExit(
            f"Unsafe v2 runtime namespace: {v2_path} aliases the v1 namespace"
        )
PY
}

tmux_target() {
    printf '=%s:0.0' "$TMUX_SESSION"
}

tmux_pane_state() {
    tmux display-message -p -t "$(tmux_target)" \
        '#{pane_dead}|#{pane_dead_status}'
}

tmux_pane_state_with_reap_nudge() {
    local pane_state pane_dead pane_dead_status poll
    pane_state="$(tmux_pane_state)"
    IFS='|' read -r pane_dead pane_dead_status <<<"$pane_state"

    # tmux 3.4 can observe PTY EOF before its coalesced SIGCHLD callback
    # reaps the pane child. A harmless tmux-owned child supplies one fresh
    # SIGCHLD; tmux then records the real wait status itself.
    if [[ "$pane_dead" == "1" && -z "$pane_dead_status" ]]; then
        tmux run-shell -b -t "$(tmux_target)" true
        for ((poll = 0; poll < 100; poll++)); do
            pane_state="$(tmux_pane_state)"
            IFS='|' read -r pane_dead pane_dead_status <<<"$pane_state"
            if [[ "$pane_dead" != "1" || -n "$pane_dead_status" ]]; then
                break
            fi
            sleep 0.01
        done
    fi
    printf '%s\n' "$pane_state"
}

process_starttime() {
    "$VENV_PYTHON" - "$1" <<'PY'
import sys
from pathlib import Path

pid = int(sys.argv[1])
if pid < 1:
    raise SystemExit("PID must be positive")
text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
closing = text.rfind(")")
if closing < 0:
    raise SystemExit("Malformed proc stat")
fields_after_comm = text[closing + 2:].split()
if len(fields_after_comm) <= 19:
    raise SystemExit("Malformed proc stat")
starttime = fields_after_comm[19]  # proc(5) field 22
if not starttime.isdigit():
    raise SystemExit("Malformed proc starttime")
print(starttime)
PY
}

write_runner_identity_atomically() {
    "$VENV_PYTHON" - "$RUNNER_PID_FILE" "$1" "$2" "$3" <<'PY'
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
pid = int(sys.argv[2])
starttime = sys.argv[3]
attempt_id = sys.argv[4]
if pid < 1 or not starttime.isdigit():
    raise SystemExit("Invalid runner process identity")
if re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", attempt_id) is None:
    raise SystemExit("Invalid runner attempt identity")

payload = {
    "schema": "scammark-runner-identity-v1",
    "pid": pid,
    "starttime": starttime,
    "attempt_id": attempt_id,
    "module": "scammark.run_experiment",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
PY
}

remove_runner_identity_if_matches() {
    "$VENV_PYTHON" - "$RUNNER_PID_FILE" "$1" "$2" "$3" <<'PY'
import json
import os
import re
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_pid = int(sys.argv[2])
expected_starttime = sys.argv[3]
expected_attempt_id = sys.argv[4]
flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
try:
    descriptor = os.open(path, flags)
except OSError:
    raise SystemExit(0)
try:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > 16_384
    ):
        raise SystemExit(0)
    encoded_payload = os.read(descriptor, 16_385)
    if len(encoded_payload) > 16_384 or os.read(descriptor, 1):
        raise SystemExit(0)
    try:
        payload = json.loads(encoded_payload.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, TypeError):
        raise SystemExit(0)
    matches = (
        type(payload) is dict
        and set(payload) == {
            "schema", "pid", "starttime", "attempt_id", "module", "created_at_utc"
        }
        and payload.get("schema") == "scammark-runner-identity-v1"
        and type(payload.get("pid")) is int
        and payload.get("pid") == expected_pid
        and str(payload.get("starttime")) == expected_starttime
        and payload.get("attempt_id") == expected_attempt_id
        and re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", expected_attempt_id) is not None
        and payload.get("module") == "scammark.run_experiment"
    )
    if not matches:
        raise SystemExit(0)

    try:
        current = os.lstat(path)
    except OSError:
        raise SystemExit(0)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        raise SystemExit(0)
    try:
        path.unlink()
    except OSError:
        raise SystemExit(0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    os.close(descriptor)
PY
}

require_exact_release_tag() {
    git -C "$ROOT_DIR" tag --points-at HEAD | grep -Fxq "$RELEASE_TAG" ||
        die "Checkout must point exactly at tag $RELEASE_TAG"
}

require_github_auth() {
    gh auth status --hostname github.com >/dev/null 2>&1 ||
        die "GitHub CLI is not authenticated. Run: gh auth login --web --git-protocol https"
}

validate_env_file() {
    require_file "$ENV_FILE"
    python3.12 - "$ENV_FILE" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
values = {}
for raw in path.read_text(encoding="utf-8-sig").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip()] = value.strip().strip('"').strip("'")

required = (
    "SCAMMARK_LLM_API_KEY",
    "SCAMMARK_VLM_API_KEY",
    "SCAMMARK_VLM_BASE_URL",
    "SCAMWEB_DATASET_ROOT",
)
missing = [key for key in required if not values.get(key)]
placeholder = [
    key for key in required
    if values.get(key) and any(token in values[key].upper() for token in ("PASTE_", "YOUR_WORKSPACE"))
]
if missing or placeholder:
    raise SystemExit(f"Invalid .env; missing={missing}, placeholders={placeholder}")

mode = stat.S_IMODE(path.stat().st_mode)
if mode & 0o077:
    raise SystemExit(f".env permissions must be 0600, found {mode:04o}")
print("Validated .env keys and permissions without displaying secret values.")
PY
}
validate_dataset_counts() {
    local root="${1:-$DEFAULT_DATASET_ROOT}"
    [[ -d "$root" ]] || die "Dataset root does not exist: $root"
    local png_count mhtml_count json_count
    png_count="$(find "$root" -type f -iname '*.png' | wc -l | tr -d ' ')"
    mhtml_count="$(find "$root" -type f -iname '*.mhtml' | wc -l | tr -d ' ')"
    json_count="$(find "$root" -type f -iname '*.json' | wc -l | tr -d ' ')"
    [[ "$png_count" == "5658" ]] || die "Expected 5658 PNG files, found $png_count"
    [[ "$mhtml_count" == "5658" ]] || die "Expected 5658 MHTML files, found $mhtml_count"
    [[ "$json_count" == "3416" ]] || die "Expected 3416 JSON files, found $json_count"
    printf 'Dataset counts verified: PNG=%s MHTML=%s JSON=%s\n' "$png_count" "$mhtml_count" "$json_count"
}

metrics_gate_passed() {
    local metrics_path="$1"
    local expected_sample_count="${2:-}"
    require_file "$metrics_path"
    "$VENV_PYTHON" - "$metrics_path" "$expected_sample_count" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
if type(metrics) is not dict:
    raise SystemExit("Metrics contract is not an object")
engineering = metrics.get("engineering")
if type(engineering) is not dict:
    raise SystemExit("Metrics engineering contract is invalid")
gates = engineering.get("acceptance_gates")
if type(gates) is not dict:
    raise SystemExit("Acceptance gate contract is invalid")
checks = gates.get("checks")
if type(checks) is not dict:
    raise SystemExit("Acceptance gate checks are invalid")
required_checks = frozenset(
    {
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
    }
)
missing = required_checks.difference(checks)
failed_count = sum(value is not True for value in checks.values())
if gates.get("passed") is not True or missing or failed_count:
    raise SystemExit(
        "Acceptance gates are incomplete or failed: "
        f"missing={len(missing)}, failed={failed_count}"
    )
sample_count = engineering.get("sample_count")
if expected and (type(sample_count) is not int or sample_count != int(expected)):
    raise SystemExit(
        f"Expected {expected} metric samples, found {sample_count}"
    )
print("Acceptance gates passed.")
PY
}

preflight_passed() {
    require_file "$PREFLIGHT_FILE"
    "$VENV_PYTHON" - "$PREFLIGHT_FILE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if type(payload) is not dict:
    raise SystemExit("Full preflight contract is not an object")
integer_contract = {
    "selected_sample_count": 5658,
    "expected_full_sample_count": 5658,
    "contract_image_count": 5658,
    "api_call_limit": 50_000,
}
if (
    payload.get("status") != "validated"
    or payload.get("run_mode") != "full"
    or any(
        type(payload.get(key)) is not int or payload.get(key) != expected
        for key, expected in integer_contract.items()
    )
):
    raise SystemExit("Full preflight contract is invalid")
print("Zero-API full preflight is valid for exactly 5,658 samples.")
PY
}
