#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

ack_breaker=0
case "$#" in
    0) ;;
    1)
        [[ "$1" == "--ack-breaker" ]] ||
            die "Usage: start_full.sh [--ack-breaker]"
        ack_breaker=1
        ;;
    *) die "Usage: start_full.sh [--ack-breaker]" ;;
esac

cd "$ROOT_DIR"
require_command tmux
require_clean_worktree
require_safe_runtime_namespace
if test_overrides_enabled; then
    require_file "$SCAMMARK_TEST_RUNNER"
    printf 'Using guarded local fake-runner overrides.\n'
else
    require_exact_release_tag
    validate_env_file
    metrics_gate_passed "$PILOT_METRICS_FILE" 10
    preflight_passed
fi

resume_dead_session=0
if tmux has-session -t "=$TMUX_SESSION" 2>/dev/null; then
    pane_state="$(tmux_pane_state_with_reap_nudge)"
    IFS='|' read -r pane_dead pane_dead_status <<<"$pane_state"
    if [[ "$pane_dead" != "1" ]]; then
        printf 'tmux session %s is already running; no mutation performed.\n' \
            "$TMUX_SESSION"
        bash "$SCRIPT_DIR/status.sh" || true
        exit 0
    fi

    case "$pane_dead_status" in
        0)
            printf 'The retained runner completed successfully. Run status/finalization; inference will not resume.\n'
            bash "$SCRIPT_DIR/status.sh" || true
            exit 0
            ;;
        1)
            printf 'The retained runner exited unexpectedly. This explicit start will resume after validation.\n'
            resume_dead_session=1
            ;;
        2)
            printf 'Acceptance gates failed. Inspect status and metrics; refusing automatic resume.\n' >&2
            exit 2
            ;;
        20)
            printf 'Authentication or billing stopped the run. Confirm replenishment; this explicit start will retry.\n'
            resume_dead_session=1
            ;;
        21)
            printf 'The physical API-call budget is exhausted; refusing resume.\n' >&2
            exit 21
            ;;
        22)
            if [[ "$ack_breaker" != "1" ]]; then
                printf 'The circuit breaker is open. Re-run with --ack-breaker after inspection.\n' >&2
                exit 22
            fi
            printf 'Circuit-breaker acknowledgement supplied; this explicit start will resume.\n'
            resume_dead_session=1
            ;;
        23)
            printf 'Experiment contract drift was detected; refusing resume.\n' >&2
            exit 23
            ;;
        30)
            printf 'Inference is complete; recover delivery only with: bash scripts/server/recover_delivery.sh package\n'
            exit 30
            ;;
        31)
            printf 'Inference is complete; recover delivery only with: bash scripts/server/recover_delivery.sh upload\n'
            exit 31
            ;;
        32)
            printf 'Inference is complete; recover delivery only with: bash scripts/server/recover_delivery.sh publish\n'
            exit 32
            ;;
        130)
            printf 'The retained runner paused safely with exit 130; this explicit start will resume.\n'
            resume_dead_session=1
            ;;
        143)
            printf 'The retained runner paused safely with exit 143; this explicit start will resume.\n'
            resume_dead_session=1
            ;;
        *)
            die "Unknown retained pane exit status: $pane_dead_status"
            ;;
    esac
fi

if [[ "$resume_dead_session" == "1" ]]; then
    tmux kill-session -t "=$TMUX_SESSION"
fi

mkdir -p "$LOG_DIR"
created_session=0
cleanup_failed_launch() {
    local status=$?
    trap - EXIT
    if [[ "$status" -ne 0 && "$created_session" == "1" ]]; then
        tmux kill-session -t "=$TMUX_SESSION" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup_failed_launch EXIT

new_session_command=(
    tmux new-session -d -s "$TMUX_SESSION" -c "$ROOT_DIR"
)
session_test_environment=(
    SCAMMARK_ALLOW_TEST_OVERRIDES
    SCAMMARK_TEST_RUNNER
    SCAMMARK_TEST_TMUX_SESSION
    SCAMMARK_TEST_PAUSE_POLL_SECONDS
    SCAMMARK_TEST_PAUSE_MAX_SECONDS
    SCAMMARK_FIXTURE_MODULE_DECOY
    SCAMMARK_FIXTURE_IMMEDIATE_EXIT_CODE
    SCAMMARK_FIXTURE_IDENTITY_WRITE_FAIL
    SCAMMARK_FIXTURE_REAL_TMUX
    SCAMMARK_FIXTURE_SOCKET
    SCAMMARK_FIXTURE_SESSION
    SCAMMARK_FIXTURE_TMUX_NUDGE_LOG
    SCAMMARK_FIXTURE_REAL_PYTHON
)
for environment_name in "${session_test_environment[@]}"; do
    environment_value=""
    if test_overrides_enabled; then
        environment_value="${!environment_name-}"
    fi
    new_session_command+=(
        -e "$environment_name=$environment_value"
    )
done
new_session_command+=("exec sleep infinity")
"${new_session_command[@]}"
created_session=1
tmux set-option -p -t "$(tmux_target)" remain-on-exit on

printf -v runner_command 'exec bash %q' "$SCRIPT_DIR/run_full_inner.sh"
if [[ "$ack_breaker" == "1" ]]; then
    printf -v runner_command '%s %q' "$runner_command" "--ack-breaker"
fi
tmux respawn-pane -k -t "$(tmux_target)" "$runner_command"

created_session=0
trap - EXIT
printf 'Started tmux session %s. Monitor it with:\n  bash scripts/server/status.sh\n' \
    "$TMUX_SESSION"
