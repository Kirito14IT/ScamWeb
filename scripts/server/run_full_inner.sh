#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

ack_breaker=0
case "$#" in
    0) ;;
    1)
        [[ "$1" == "--ack-breaker" ]] ||
            die "Usage: run_full_inner.sh [--ack-breaker]"
        ack_breaker=1
        ;;
    *) die "Usage: run_full_inner.sh [--ack-breaker]" ;;
esac

cd "$ROOT_DIR"
require_safe_runtime_namespace
if test_overrides_enabled; then
    require_file "$SCAMMARK_TEST_RUNNER"
else
    require_exact_release_tag
    validate_env_file
fi
mkdir -p "$LOG_DIR" "$FULL_DIR" "$(dirname "$RUNNER_PID_FILE")"

SCAMMARK_ATTEMPT_ID="$(
    "$VENV_PYTHON" -c 'import uuid; print(uuid.uuid4().hex)'
)"
export SCAMMARK_ATTEMPT_ID

runner_pid=""
runner_starttime=""
runner_reaped=0

runner_job_is_running() {
    local expected_pid="$1"
    local active_pid
    while IFS= read -r active_pid; do
        [[ "$active_pid" == "$expected_pid" ]] && return 0
    done < <(jobs -pr)
    return 1
}

emit_shell_event() {
    local event="$1"
    shift
    # The observability CLI derives a unique shell-suffixed attempt ID while
    # retaining this unsuffixed runner attempt in runner_attempt_id.
    "$VENV_PYTHON" -m scammark.observability append \
        --path "$RUN_EVENTS_FILE" \
        --event "$event" \
        --attempt-id "$SCAMMARK_ATTEMPT_ID" \
        --field "runner_attempt_id=$SCAMMARK_ATTEMPT_ID" \
        "$@"
}

finish_shell_bridge() {
    local code=$?
    trap - EXIT
    set +e
    if [[ "$runner_reaped" != "1" && -n "$runner_pid" ]]; then
        if [[ -n "$runner_starttime" ]]; then
            current_starttime="$(process_starttime "$runner_pid" 2>/dev/null || true)"
            if [[ "$current_starttime" == "$runner_starttime" ]]; then
                kill -TERM "$runner_pid" 2>/dev/null || true
            fi
        elif runner_job_is_running "$runner_pid"; then
            kill -TERM "$runner_pid" 2>/dev/null || true
        fi
        wait "$runner_pid" 2>/dev/null || true
        runner_reaped=1
        wait 2>/dev/null || true
    fi
    printf '[%s] Full runner shell exited with exit code %s.\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" | tee -a "$RUN_LOG"
    if [[ -n "$runner_pid" ]]; then
        emit_shell_event shell_exited \
            --field "pid=$runner_pid" \
            --field "exit_code=$code" \
            --field "forced=false" \
            --field "message=runner shell exited"
    fi
    if [[ -n "$runner_pid" && -n "$runner_starttime" ]]; then
        remove_runner_identity_if_matches \
            "$runner_pid" "$runner_starttime" "$SCAMMARK_ATTEMPT_ID"
    fi
    # Returning from an EXIT trap preserves the status that entered the trap
    # without introducing a second, cleanup-local exit operation.
    return 0
}
trap finish_shell_bridge EXIT

runner_command=(
    "$VENV_PYTHON" -m scammark.run_experiment
    --config "$CONFIG_PATH"
    --full
    --resume
)
if [[ "$ack_breaker" == "1" ]]; then
    runner_command+=(--ack-breaker)
fi
if test_overrides_enabled; then
    runner_command=("$VENV_PYTHON" "$SCAMMARK_TEST_RUNNER")
fi

printf '\n[%s] Starting or resuming the full ScamMark experiment.\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"
"${runner_command[@]}" > >(tee -a "$RUN_LOG") 2>&1 &
runner_pid=$!
runner_starttime_status=0
runner_starttime="$(process_starttime "$runner_pid")" || runner_starttime_status=$?
if [[ "$runner_starttime_status" -ne 0 ]]; then
    runner_was_running=0
    if runner_job_is_running "$runner_pid"; then
        runner_was_running=1
    fi
    set +e
    if [[ "$runner_was_running" == "1" ]]; then
        kill -TERM "$runner_pid" 2>/dev/null
        termination_status=$?
    else
        termination_status=1
    fi
    wait "$runner_pid"
    run_status=$?
    runner_reaped=1
    wait
    set -e
    if [[ "$runner_was_running" == "1" && "$termination_status" == "0" ]]; then
        exit "$runner_starttime_status"
    fi
else
    write_runner_identity_atomically \
        "$runner_pid" "$runner_starttime" "$SCAMMARK_ATTEMPT_ID"
    emit_shell_event shell_started \
        --field "pid=$runner_pid" \
        --field "starttime=$runner_starttime" \
        --field "message=runner shell started"

    set +e
    wait "$runner_pid"
    run_status=$?
    runner_reaped=1
    wait
    set -e
fi

if [[ "$run_status" -ne 0 ]]; then
    exit "$run_status"
fi

delivery_status=0
env \
    -u SCAMMARK_LLM_API_KEY \
    -u SCAMMARK_LLM_BASE_URL \
    -u SCAMMARK_VLM_API_KEY \
    -u SCAMMARK_VLM_BASE_URL \
    bash "$SCRIPT_DIR/recover_delivery.sh" package || delivery_status=$?
[[ "$delivery_status" == "0" ]] || exit "$delivery_status"
printf '[%s] Inference results are verified and uploaded. Run finalize_bertscore.sh interactively next.\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"
