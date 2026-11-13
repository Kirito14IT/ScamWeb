#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

usage="Usage: recover_delivery.sh [--mode inference|final] package|upload|publish"
mode="inference"
if [[ "${1:-}" == "--mode" ]]; then
    [[ "$#" -ge 2 ]] || die "$usage"
    mode="$2"
    shift 2
fi
[[ "$mode" == "inference" || "$mode" == "final" ]] || die "$usage"
[[ "$#" == "1" ]] || die "$usage"
requested_stage="$1"
[[ "$requested_stage" == "package" || "$requested_stage" == "upload" ||
    "$requested_stage" == "publish" ]] || die "$usage"

clean_python=(
    env
    -u SCAMMARK_LLM_API_KEY
    -u SCAMMARK_LLM_BASE_URL
    -u SCAMMARK_VLM_API_KEY
    -u SCAMMARK_VLM_BASE_URL
    "$VENV_PYTHON"
)
unset \
    SCAMMARK_LLM_API_KEY \
    SCAMMARK_LLM_BASE_URL \
    SCAMMARK_VLM_API_KEY \
    SCAMMARK_VLM_BASE_URL

cd "$ROOT_DIR"
require_clean_worktree
require_exact_release_tag
require_safe_runtime_namespace
"${clean_python[@]}" -m scammark.result_contract validate-full \
    --full-dir "$FULL_DIR" \
    --mode "$mode" \
    --expected-count 5658 >/dev/null

delivery_attempt_id="${SCAMMARK_ATTEMPT_ID:-}"
if [[ -z "$delivery_attempt_id" ]]; then
    delivery_attempt_id="$("${clean_python[@]}" -c 'import uuid; print("delivery-recovery-" + uuid.uuid4().hex)')"
fi

refresh_delivery_state() {
    local state_json state_projection
    state_json="$(
        "${clean_python[@]}" -m scammark.result_contract load-delivery \
            --state-path "$DELIVERY_STATE" \
            --mode "$mode"
    )"
    state_projection="$(
        printf '%s' "$state_json" |
            "${clean_python[@]}" -c '
import json
import re
import sys

state = json.load(sys.stdin)
if type(state) is not dict:
    raise SystemExit("Invalid delivery state")
sha = state.get("archive_sha256")
if sha is not None and (
    type(sha) is not str or re.fullmatch(r"[0-9a-f]{64}", sha) is None
):
    raise SystemExit("Invalid delivery digest")
for name in ("packaged", "uploaded", "published"):
    value = state.get(name)
    if type(value) is not bool:
        raise SystemExit("Invalid delivery state")
    print("1" if value else "0")
print(sha or "-")
last_exit_code = state.get("last_exit_code")
print("-" if last_exit_code is None else last_exit_code)
'
    )"
    unset state_json
    mapfile -t delivery_fields <<<"$state_projection"
    unset state_projection
    [[ "${#delivery_fields[@]}" == "5" ]] ||
        die "Unable to read matching delivery state"
    packaged="${delivery_fields[0]}"
    uploaded="${delivery_fields[1]}"
    published="${delivery_fields[2]}"
    archive_sha256="${delivery_fields[3]}"
    last_exit_code="${delivery_fields[4]}"
    if [[ "$archive_sha256" == "-" ]]; then
        archive_sha256=""
    fi
    if [[ "$last_exit_code" == "-" ]]; then
        last_exit_code=""
    fi
}

verify_recorded_archive() {
    local archive checksum local_sha expected_line actual_line
    [[ "$packaged" == "1" && -n "$archive_sha256" ]] ||
        die "A verified packaged archive is required for this recovery stage"
    if [[ "$mode" == "inference" ]]; then
        archive="$DIST_DIR/$INFERENCE_ASSET"
    else
        archive="$DIST_DIR/$FINAL_ASSET"
    fi
    checksum="$archive.sha256"
    require_file "$archive"
    require_file "$checksum"
    local_sha="$(sha256sum "$archive" | awk '{print $1}')"
    [[ "$local_sha" == "$archive_sha256" ]] ||
        die "Recorded delivery archive SHA256 does not match local bytes"
    expected_line="$archive_sha256  $(basename "$archive")"
    actual_line="$(<"$checksum")"
    [[ "$actual_line" == "$expected_line" ]] ||
        die "Recorded delivery checksum file is inconsistent"
    (
        cd "$DIST_DIR"
        sha256sum -c "$(basename "$checksum")"
    )
}

record_failed_stage() {
    local stage="$1"
    local exit_code="$2"
    local arguments=(
        -m scammark.result_contract record-delivery
        --state-path "$DELIVERY_STATE"
        --mode "$mode"
        --stage "$stage"
        --status failed
        --exit-code "$exit_code"
        --event-path "$RUN_EVENTS_FILE"
        --runner-attempt-id "$delivery_attempt_id"
    )
    if [[ "$stage" != "package" && -n "$archive_sha256" ]]; then
        arguments+=(--archive-sha256 "$archive_sha256")
    fi
    if ! "${clean_python[@]}" "${arguments[@]}" >/dev/null; then
        printf 'Delivery failure state could not be recorded for stage %s.\n' \
            "$stage" >&2
    fi
}

run_delivery_stage() {
    local stage="$1"
    local exit_code="$2"
    local script
    case "$stage" in
        package) script="$SCRIPT_DIR/package_results.sh" ;;
        upload) script="$SCRIPT_DIR/upload_release_assets.sh" ;;
        publish) script="$SCRIPT_DIR/publish_results_branch.sh" ;;
    esac
    if [[ -z "$cleanup_exit_code" ]]; then
        cleanup_exit_code="$exit_code"
    fi
    if ! bash "$script" "$mode"; then
        record_failed_stage "$stage" "$exit_code"
        exit "$exit_code"
    fi
    refresh_delivery_state
}

refresh_delivery_state
cleanup_exit_code="$last_exit_code"
case "$requested_stage" in
    package)
        if [[ "$packaged" != "1" ]]; then
            run_delivery_stage package 30
        fi
        verify_recorded_archive
        if [[ "$uploaded" != "1" ]]; then
            run_delivery_stage upload 31
        fi
        if [[ "$published" != "1" ]]; then
            run_delivery_stage publish 32
        fi
        ;;
    upload)
        [[ "$packaged" == "1" ]] ||
            die "Upload recovery cannot package missing delivery bytes"
        verify_recorded_archive
        if [[ "$uploaded" != "1" ]]; then
            run_delivery_stage upload 31
        fi
        if [[ "$published" != "1" ]]; then
            run_delivery_stage publish 32
        fi
        ;;
    publish)
        [[ "$packaged" == "1" && "$uploaded" == "1" ]] ||
            die "Publication recovery cannot run earlier delivery stages"
        verify_recorded_archive
        if [[ "$published" != "1" ]]; then
            run_delivery_stage publish 32
        fi
        ;;
esac

if [[ "$mode" == "inference" ]] &&
    [[ -n "$cleanup_exit_code" ]] &&
    command -v tmux >/dev/null 2>&1 &&
    tmux has-session -t "=$TMUX_SESSION" 2>/dev/null; then
    pane_state="$(tmux_pane_state_with_reap_nudge 2>/dev/null || true)"
    IFS='|' read -r pane_dead pane_dead_status <<<"$pane_state"
    if [[ "$pane_dead" == "1" && "$pane_dead_status" == "$cleanup_exit_code" ]]; then
        tmux kill-session -t "=$TMUX_SESSION"
    fi
fi

printf 'Delivery recovery completed for %s mode from %s.\n' \
    "$mode" "$requested_stage"
