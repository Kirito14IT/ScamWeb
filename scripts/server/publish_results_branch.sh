#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

mode="${1:-inference}"
[[ "$mode" == "inference" || "$mode" == "final" ]] ||
    die "Usage: publish_results_branch.sh [inference|final]"

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

staging_parent="$(mktemp -d "${TMPDIR:-/tmp}/scammark-publish-inputs.XXXXXX")"
chmod 700 "$staging_parent"
snapshot="$staging_parent/snapshot"
worktree_parent=""
worktree=""
cleanup_publish() {
    if [[ -n "$worktree" && -d "$worktree" ]]; then
        git -C "$ROOT_DIR" worktree remove --force "$worktree" >/dev/null 2>&1 || true
    fi
    if [[ -n "$worktree_parent" && -d "$worktree_parent" ]]; then
        rmdir "$worktree_parent" >/dev/null 2>&1 || true
    fi
    rm -rf -- "$staging_parent"
}
trap cleanup_publish EXIT

snapshot_json="$(
    "${clean_python[@]}" - \
        "$mode" \
        "$DELIVERY_STATE" \
        "$FULL_DIR" \
        "$DIST_DIR/$INFERENCE_ASSET" \
        "$DIST_DIR/$FINAL_ASSET" \
        "$snapshot" <<'PY'
import json
import sys

from scammark.result_contract import stage_verified_delivery_inputs

result = stage_verified_delivery_inputs(
    mode=sys.argv[1],
    delivery_state_path=sys.argv[2],
    runtime_full_dir=sys.argv[3],
    inference_archive_path=sys.argv[4],
    final_archive_path=sys.argv[5],
    destination=sys.argv[6],
    expected_count=5658,
)
print(json.dumps({
    "selected_archive_sha256": result.selected_archive_sha256,
    "included_modes": list(result.included_modes),
}, sort_keys=True, separators=(",", ":")))
PY
)"
selected_recorded_sha="$(
    printf '%s' "$snapshot_json" |
        "${clean_python[@]}" -c '
import json
import re
import sys

value = json.load(sys.stdin).get("selected_archive_sha256")
if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
    raise SystemExit("Invalid verified publication snapshot")
print(value)
'
)"
unset snapshot_json

# Every branch input is now a private, descriptor-pinned archive snapshot.
# No runtime result or dist checksum is reopened after this boundary.
require_github_auth
git -C "$ROOT_DIR" fetch origin "$RESULTS_BRANCH" 2>/dev/null || true

worktree_parent="$(mktemp -d "${TMPDIR:-/tmp}/scammark-results.XXXXXX")"
worktree="$worktree_parent/repo"

if git -C "$ROOT_DIR" show-ref --verify --quiet "refs/remotes/origin/$RESULTS_BRANCH"; then
    git -C "$ROOT_DIR" worktree add --force -B "$RESULTS_BRANCH" "$worktree" "origin/$RESULTS_BRANCH"
else
    empty_tree="$(git -C "$ROOT_DIR" mktree </dev/null)"
    empty_commit="$(printf 'Initialize ScamMark results branch\n' | git -C "$ROOT_DIR" \
        -c user.name='ScamWeb-V Experiment' \
        -c user.email='scamweb-v@users.noreply.github.com' \
        commit-tree "$empty_tree")"
    git -C "$ROOT_DIR" branch -f "$RESULTS_BRANCH" "$empty_commit"
    git -C "$ROOT_DIR" worktree add "$worktree" "$RESULTS_BRANCH"
fi

find "$worktree" -mindepth 1 -maxdepth 1 ! -name .git \
    -exec rm -rf -- {} +
mkdir -p "$worktree/checksums"
cp "$snapshot/inference/full/metrics.json" "$worktree/metrics.json"
cp "$snapshot/inference/full/full_summary.md" "$worktree/full_summary.md"
include_final_branch=0
if [[ -f "$snapshot/final/full/metrics_final.json" && \
    -f "$snapshot/final/full/final_evaluation_summary.md" ]]; then
    include_final_branch=1
    cp "$snapshot/final/full/metrics_final.json" "$worktree/metrics_final.json"
    cp \
        "$snapshot/final/full/final_evaluation_summary.md" \
        "$worktree/final_evaluation_summary.md"
fi
for checksum in \
    "$snapshot/checksums/$INFERENCE_ASSET.sha256" \
    "$snapshot/checksums/$FINAL_ASSET.sha256"; do
    [[ -f "$checksum" ]] && cp "$checksum" "$worktree/checksums/"
done

"${clean_python[@]}" - "$worktree/README.md" "$REPOSITORY" "$RELEASE_TAG" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import sys
from pathlib import Path

path, repository, tag, commit = sys.argv[1:]
text = f"""# ScamMark Full Experiment Results

This branch contains small, reviewable metrics and summaries for the frozen full experiment.
Predictions, ledgers, and final evaluation bundles are stored as verified assets in the private release.

- Release: https://github.com/{repository}/releases/tag/{tag}
- Experiment commit: `{commit}`
- Dataset size: 5,658 pages

No API keys, raw webpages, screenshots, MHTML files, model weights, or caches are stored on this branch.
"""
Path(path).write_text(text, encoding="utf-8")
PY

staged_relative_paths=(metrics.json full_summary.md README.md)
if [[ "$include_final_branch" == "1" ]]; then
    staged_relative_paths+=(metrics_final.json final_evaluation_summary.md)
fi
for checksum in \
    "$worktree/checksums/$INFERENCE_ASSET.sha256" \
    "$worktree/checksums/$FINAL_ASSET.sha256"; do
    if [[ -f "$checksum" ]]; then
        staged_relative_paths+=("checksums/$(basename "$checksum")")
    fi
done
"${clean_python[@]}" - "$worktree" "${staged_relative_paths[@]}" <<'PY'
import sys
from pathlib import Path

from scammark.result_contract import scan_for_leaks

root = Path(sys.argv[1])
expected_files = set(sys.argv[2:])
actual_files = set()
actual_directories = set()
for entry in root.rglob("*"):
    relative = entry.relative_to(root)
    if relative.parts and relative.parts[0] == ".git":
        continue
    if entry.is_symlink():
        raise SystemExit("Unexpected results-branch inventory")
    if entry.is_dir():
        actual_directories.add(relative.as_posix())
    elif entry.is_file():
        actual_files.add(relative.as_posix())
    else:
        raise SystemExit("Unexpected results-branch inventory")
if actual_files != expected_files or actual_directories != {"checksums"}:
    raise SystemExit("Unexpected results-branch inventory")
scan_for_leaks(root, relative_paths=sorted(expected_files))
PY

git -C "$worktree" config user.name "ScamWeb-V Experiment"
git -C "$worktree" config user.email "scamweb-v@users.noreply.github.com"
git -C "$worktree" add -A
if git -C "$worktree" diff --cached --quiet; then
    printf 'Results branch already matches the verified snapshot.\n'
else
    git -C "$worktree" commit -m "results: publish qwen3 dashscope full evaluation"
    git -C "$worktree" push origin "HEAD:refs/heads/$RESULTS_BRANCH"
    printf 'Published small metrics to branch %s.\n' "$RESULTS_BRANCH"
fi

delivery_attempt_id="${SCAMMARK_ATTEMPT_ID:-}"
if [[ -z "$delivery_attempt_id" ]]; then
    delivery_attempt_id="$("${clean_python[@]}" -c 'import uuid; print("delivery-shell-" + uuid.uuid4().hex)')"
fi
"${clean_python[@]}" -m scammark.result_contract record-delivery \
    --state-path "$DELIVERY_STATE" \
    --mode "$mode" \
    --stage publish \
    --status complete \
    --archive-sha256 "$selected_recorded_sha" \
    --event-path "$RUN_EVENTS_FILE" \
    --runner-attempt-id "$delivery_attempt_id" >/dev/null
