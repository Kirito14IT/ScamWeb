#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

mode="${1:-}"
[[ "$mode" == "inference" || "$mode" == "final" ]] ||
    die "Usage: package_results.sh inference|final"

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
unset secret_values_json

cd "$ROOT_DIR"
require_clean_worktree
require_exact_release_tag
require_safe_runtime_namespace
"${clean_python[@]}" -m scammark.result_contract validate-full \
    --full-dir "$FULL_DIR" \
    --mode "$mode" \
    --expected-count 5658 >/dev/null

case "$mode" in
    inference)
        archive="$DIST_DIR/$INFERENCE_ASSET"
        extra_build_arguments=(
            --log-path "$RUN_LOG"
            --config-path "$CONFIG_PATH"
            --frozen-dir "$FROZEN_DIR"
        )
        ;;
    final)
        archive="$DIST_DIR/$FINAL_ASSET"
        extra_build_arguments=(
            --bertscore-asset-status-path \
                "$ROOT_DIR/models/huggingface/bertscore_asset_status.json"
            --finalize-log-path "$FINALIZE_LOG"
        )
        ;;
esac

package_staging_parent="$(mktemp -d "${TMPDIR:-/tmp}/scammark-package-inputs.XXXXXX")"
chmod 700 "$package_staging_parent"
trap 'rm -rf -- "$package_staging_parent"' EXIT
packaged="$(
    "${clean_python[@]}" -m scammark.result_contract load-delivery \
        --state-path "$DELIVERY_STATE" --mode "$mode" |
        "${clean_python[@]}" -c '
import json
import sys

value = json.load(sys.stdin).get("packaged")
if type(value) is not bool:
    raise SystemExit("Invalid delivery package state")
print("1" if value else "0")
'
)"

if [[ "$packaged" == "1" ]]; then
    "${clean_python[@]}" - \
        "$mode" \
        "$DELIVERY_STATE" \
        "$FULL_DIR" \
        "$DIST_DIR/$INFERENCE_ASSET" \
        "$DIST_DIR/$FINAL_ASSET" \
        "$package_staging_parent/reuse" <<'PY'
import sys

from scammark.result_contract import stage_verified_delivery_inputs

mode = sys.argv[1]
stage_verified_delivery_inputs(
    mode=mode,
    delivery_state_path=sys.argv[2],
    runtime_full_dir=sys.argv[3],
    inference_archive_path=sys.argv[4],
    final_archive_path=sys.argv[5],
    destination=sys.argv[6],
    expected_count=5658,
    include_optional_final=False,
    require_selected_uploaded=False,
    require_inference_uploaded=mode != "inference",
)
PY
    printf 'Reused the recorded, verified %s archive without rebuilding.\n' "$mode"
    exit 0
fi

if [[ "$mode" == "final" ]]; then
    "${clean_python[@]}" - \
        "$DELIVERY_STATE" \
        "$FULL_DIR" \
        "$DIST_DIR/$INFERENCE_ASSET" \
        "$DIST_DIR/$FINAL_ASSET" \
        "$package_staging_parent/prebuild" <<'PY'
import sys

from scammark.result_contract import stage_verified_delivery_inputs

stage_verified_delivery_inputs(
    mode="inference",
    delivery_state_path=sys.argv[1],
    runtime_full_dir=sys.argv[2],
    inference_archive_path=sys.argv[3],
    final_archive_path=sys.argv[4],
    destination=sys.argv[5],
    expected_count=5658,
    include_optional_final=False,
    require_inference_published=True,
)
PY
fi

secret_values_json="$(
    "${clean_python[@]}" - "$ENV_FILE" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
descriptor = os.open(path, flags)
try:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > 65_536
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise SystemExit("Invalid private environment file")
    encoded = os.read(descriptor, 65_537)
    if len(encoded) > 65_536 or os.read(descriptor, 1):
        raise SystemExit("Invalid private environment file")
finally:
    os.close(descriptor)

required = (
    "SCAMMARK_LLM_API_KEY",
    "SCAMMARK_VLM_API_KEY",
    "SCAMMARK_VLM_BASE_URL",
)
values = {}
for raw in encoded.decode("utf-8-sig", errors="strict").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if key not in required:
        continue
    if key in values:
        raise SystemExit("Duplicate private packaging value")
    values[key] = value.strip().strip('"').strip("'")
if any(not values.get(name) for name in required):
    raise SystemExit("Required private packaging values are missing")
json.dump([values[name] for name in required], sys.stdout, ensure_ascii=False)
PY
)" || die "Unable to read private packaging values"
export -n secret_values_json

git_commit="$(git -C "$ROOT_DIR" rev-parse HEAD)"
build_output=""
if ! build_output="$(
    printf '%s' "$secret_values_json" |
        "${clean_python[@]}" -m scammark.result_contract build-archive \
                --mode "$mode" \
                --full-dir "$FULL_DIR" \
                --state-dir "$STATE_DIR" \
                --destination "$archive" \
                --release-tag "$RELEASE_TAG" \
                --repository "$REPOSITORY" \
                --git-commit "$git_commit" \
                --secret-values-stdin-json \
                "${extra_build_arguments[@]}"
)"; then
    unset secret_values_json
    die "Validated result archive creation failed"
fi
unset secret_values_json

archive_sha="$(
    printf '%s' "$build_output" |
        "${clean_python[@]}" -c '
import json
import sys

value = json.load(sys.stdin)
sha = value.get("sha256") if type(value) is dict else None
if type(sha) is not str or len(sha) != 64:
    raise SystemExit("Invalid archive result")
print(sha)
'
)"
verified_sha="$(
    "${clean_python[@]}" - "$archive" "$archive_sha" <<'PY'
import sys

from scammark.result_contract import (
    ResultContractError,
    verify_result_archive_checksum,
)

try:
    verified = verify_result_archive_checksum(
        sys.argv[1], expected_sha256=sys.argv[2]
    )
except ResultContractError:
    raise SystemExit("result archive verification failed") from None
print(verified)
PY
)" || die "Validated archive checksum binding failed"
[[ "$verified_sha" == "$archive_sha" ]] ||
    die "Validated archive checksum result is inconsistent"

if [[ "$mode" == "final" ]]; then
    candidate_sha="$(
        "${clean_python[@]}" - \
            "$DELIVERY_STATE" \
            "$FULL_DIR" \
            "$DIST_DIR/$INFERENCE_ASSET" \
            "$archive" \
            "$package_staging_parent/postbuild" <<'PY'
import sys

from scammark.result_contract import stage_verified_final_candidate

result = stage_verified_final_candidate(
    delivery_state_path=sys.argv[1],
    runtime_full_dir=sys.argv[2],
    inference_archive_path=sys.argv[3],
    final_archive_path=sys.argv[4],
    destination=sys.argv[5],
    expected_count=5658,
)
print(result.archive_sha256)
PY
    )"
    [[ "$candidate_sha" == "$archive_sha" ]] ||
        die "Final candidate digest differs from validated packaging"
    archive_sha="$candidate_sha"
fi

delivery_attempt_id="${SCAMMARK_ATTEMPT_ID:-}"
if [[ -z "$delivery_attempt_id" ]]; then
    delivery_attempt_id="$("${clean_python[@]}" -c 'import uuid; print("delivery-shell-" + uuid.uuid4().hex)')"
fi
"${clean_python[@]}" -m scammark.result_contract record-delivery \
    --state-path "$DELIVERY_STATE" \
    --mode "$mode" \
    --stage package \
    --status complete \
    --archive-sha256 "$archive_sha" \
    --event-path "$RUN_EVENTS_FILE" \
    --runner-attempt-id "$delivery_attempt_id" >/dev/null

printf 'Created validated %s archive with verified SHA256 %s.\n' \
    "$mode" "$archive_sha"
