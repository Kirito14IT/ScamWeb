#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

mode="${1:-}"
case "$mode" in
    inference) archive="$DIST_DIR/$INFERENCE_ASSET" ;;
    final) archive="$DIST_DIR/$FINAL_ASSET" ;;
    *) die "Usage: upload_release_assets.sh inference|final" ;;
esac
checksum="$archive.sha256"

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

staging_parent="$(mktemp -d "${TMPDIR:-/tmp}/scammark-upload-inputs.XXXXXX")"
chmod 700 "$staging_parent"
snapshot="$staging_parent/snapshot"
trap 'rm -rf -- "$staging_parent"' EXIT
snapshot_json="$(
    "${clean_python[@]}" - \
        "$mode" \
        "$DELIVERY_STATE" \
        "$archive" \
        "$snapshot" \
        "$FULL_DIR" \
        "$DIST_DIR/$INFERENCE_ASSET" <<'PY'
import json
import sys

from scammark.result_contract import (
    stage_verified_delivery_assets,
    stage_verified_delivery_inputs,
)

mode = sys.argv[1]
if mode == "final":
    result = stage_verified_delivery_inputs(
        mode="final",
        delivery_state_path=sys.argv[2],
        runtime_full_dir=sys.argv[5],
        inference_archive_path=sys.argv[6],
        final_archive_path=sys.argv[3],
        destination=sys.argv[4],
        expected_count=5658,
        include_optional_final=False,
        require_selected_uploaded=False,
    )
    archive_sha256 = result.selected_archive_sha256
    checksum_sha256 = result.selected_checksum_sha256
    archive_name = result.selected_archive_name
    checksum_name = result.selected_checksum_name
else:
    result = stage_verified_delivery_assets(
        mode="inference",
        delivery_state_path=sys.argv[2],
        archive_path=sys.argv[3],
        destination=sys.argv[4],
    )
    archive_sha256 = result.archive_sha256
    checksum_sha256 = result.checksum_sha256
    archive_name = result.archive_name
    checksum_name = result.checksum_name
print(json.dumps({
    "archive_sha256": archive_sha256,
    "checksum_sha256": checksum_sha256,
    "archive_name": archive_name,
    "checksum_name": checksum_name,
}, sort_keys=True, separators=(",", ":")))
PY
)"
readarray -t snapshot_fields < <(
    printf '%s' "$snapshot_json" |
        "${clean_python[@]}" -c '
import json
import re
import sys

payload = json.load(sys.stdin)
values = (
    payload.get("archive_sha256"),
    payload.get("checksum_sha256"),
    payload.get("archive_name"),
    payload.get("checksum_name"),
)
if (
    any(type(value) is not str for value in values)
    or re.fullmatch(r"[0-9a-f]{64}", values[0]) is None
    or re.fullmatch(r"[0-9a-f]{64}", values[1]) is None
):
    raise SystemExit("Invalid verified upload snapshot")
print(*values, sep="\n")
'
)
unset snapshot_json
[[ "${#snapshot_fields[@]}" == "4" ]] || die "Invalid verified upload snapshot"
recorded_sha="${snapshot_fields[0]}"
checksum_sha="${snapshot_fields[1]}"
staged_archive="$snapshot/.verified-assets/$mode/${snapshot_fields[2]}"
staged_checksum="$snapshot/.verified-assets/$mode/${snapshot_fields[3]}"

verify_local_delivery_bytes() {
    local local_archive_sha expected_checksum_line actual_checksum_line
    require_file "$archive"
    require_file "$checksum"
    local_archive_sha="$(sha256sum "$archive" | awk '{print $1}')"
    [[ "$local_archive_sha" == "$recorded_sha" ]] ||
        die "Local archive does not match the recorded delivery SHA256"
    expected_checksum_line="$recorded_sha  $(basename "$archive")"
    actual_checksum_line="$(<"$checksum")"
    [[ "$actual_checksum_line" == "$expected_checksum_line" ]] ||
        die "Checksum file does not match the recorded delivery SHA256"
    (
        cd "$DIST_DIR"
        sha256sum -c "$(basename "$checksum")"
    )
}

verify_snapshot_file() {
    local file="$1"
    local expected="$2"
    [[ "$(sha256sum "$file" | awk '{print $1}')" == "$expected" ]] ||
        die "Verified upload snapshot changed"
}

remote_digest_for() {
    local name="$1"
    local remote_json
    remote_json="$(gh api "repos/$REPOSITORY/releases/tags/$RELEASE_TAG" \
        --jq ".assets[] | select(.name == \"$name\") | {name: .name, digest: .digest}")"
    if [[ -z "$remote_json" ]]; then
        printf '%s' ""
        return
    fi
    printf '%s' "$remote_json" |
        "${clean_python[@]}" -c '
import json
import sys

payload = json.load(sys.stdin)
digest = payload.get("digest")
print(digest if type(digest) is str else "")
'
}

verify_local_delivery_bytes
verify_snapshot_file "$staged_archive" "$recorded_sha"
verify_snapshot_file "$staged_checksum" "$checksum_sha"

require_github_auth
gh release view "$RELEASE_TAG" --repo "$REPOSITORY" >/dev/null
for file in "$staged_archive" "$staged_checksum"; do
    name="$(basename "$file")"
    if [[ "$file" == "$staged_archive" ]]; then
        expected_hash="$recorded_sha"
    else
        expected_hash="$checksum_sha"
    fi
    verify_local_delivery_bytes
    verify_snapshot_file "$file" "$expected_hash"
    remote_digest="$(remote_digest_for "$name")"
    verify_local_delivery_bytes
    verify_snapshot_file "$file" "$expected_hash"
    if [[ -n "$remote_digest" ]]; then
        [[ "$remote_digest" == "sha256:$expected_hash" ]] ||
            die "Release asset already exists with a different or unavailable digest"
        printf 'Release asset already matches verified SHA256: %s\n' "$name"
    else
        gh release upload "$RELEASE_TAG" "$file" --repo "$REPOSITORY"
        verify_snapshot_file "$file" "$expected_hash"
        remote_digest="$(remote_digest_for "$name")"
        [[ "$remote_digest" == "sha256:$expected_hash" ]] ||
            die "Uploaded release asset digest is different or unavailable"
        printf 'Uploaded and remotely verified release asset: %s\n' "$name"
    fi
done

delivery_attempt_id="${SCAMMARK_ATTEMPT_ID:-}"
if [[ -z "$delivery_attempt_id" ]]; then
    delivery_attempt_id="$("${clean_python[@]}" -c 'import uuid; print("delivery-shell-" + uuid.uuid4().hex)')"
fi
"${clean_python[@]}" -m scammark.result_contract record-delivery \
    --state-path "$DELIVERY_STATE" \
    --mode "$mode" \
    --stage upload \
    --status complete \
    --archive-sha256 "$recorded_sha" \
    --event-path "$RUN_EVENTS_FILE" \
    --runner-attempt-id "$delivery_attempt_id" >/dev/null
