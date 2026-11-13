#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

readonly BERTSCORE_IMAGE="nvcr.io/nvidia/pytorch@sha256:d1a026a2b28de6ef6404334ca78b7b761434f10d370ad2130b37aafb6e68eb17"
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

inference_state_json="$(
    "${clean_python[@]}" -m scammark.result_contract load-delivery \
        --state-path "$DELIVERY_STATE" \
        --mode inference
)"
inference_next_stage="$(
    printf '%s' "$inference_state_json" |
        "${clean_python[@]}" -c '
import json
import sys

state = json.load(sys.stdin)
if type(state) is not dict:
    raise SystemExit("Invalid inference delivery state")
packaged = state.get("packaged")
uploaded = state.get("uploaded")
published = state.get("published")
if any(type(value) is not bool for value in (packaged, uploaded, published)):
    raise SystemExit("Invalid inference delivery state")
if not packaged:
    print("package")
elif not uploaded:
    print("upload")
elif not published:
    print("publish")
else:
    print("complete")
'
)"
unset inference_state_json
if [[ "$inference_next_stage" != "complete" ]]; then
    printf 'Inference delivery is incomplete. Run: bash scripts/server/recover_delivery.sh %s\n' \
        "$inference_next_stage" >&2
    case "$inference_next_stage" in
        package) exit 30 ;;
        upload) exit 31 ;;
        publish) exit 32 ;;
    esac
fi
"${clean_python[@]}" -m scammark.result_contract validate-full \
    --full-dir "$FULL_DIR" \
    --mode inference \
    --expected-count 5658 >/dev/null

input_staging_parent="$(mktemp -d "${TMPDIR:-/tmp}/scammark-finalizer-inputs.XXXXXX")"
chmod 700 "$input_staging_parent"
input_snapshot="$input_staging_parent/snapshot"
compute_parent=""
cleanup_finalizer_temporary() {
    if [[ -n "$compute_parent" ]]; then
        rm -rf -- "$compute_parent"
    fi
    rm -rf -- "$input_staging_parent"
}
trap cleanup_finalizer_temporary EXIT

# Bind the published inference archive and all mutable compute inputs before
# inspecting any final-delivery shortcut. Failed/unrecorded final residue is
# deliberately outside this inference-only trust boundary.
"${clean_python[@]}" - \
    "$DELIVERY_STATE" \
    "$FULL_DIR" \
    "$DIST_DIR/$INFERENCE_ASSET" \
    "$DIST_DIR/$FINAL_ASSET" \
    "$input_snapshot" <<'PY'
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
)
PY

final_state_json="$(
    "${clean_python[@]}" -m scammark.result_contract load-delivery \
        --state-path "$DELIVERY_STATE" \
        --mode final
)"
final_next_stage="$(
    printf '%s' "$final_state_json" |
        "${clean_python[@]}" -c '
import json
import sys

state = json.load(sys.stdin)
if type(state) is not dict:
    raise SystemExit("Invalid final delivery state")
packaged = state.get("packaged")
uploaded = state.get("uploaded")
published = state.get("published")
if any(type(value) is not bool for value in (packaged, uploaded, published)):
    raise SystemExit("Invalid final delivery state")
if packaged and not uploaded:
    print("upload")
elif uploaded and not published:
    print("publish")
elif published:
    print("complete")
else:
    print("package")
'
)"
unset final_state_json

if [[ "$final_next_stage" != "package" ]]; then
    require_selected_uploaded=1
    if [[ "$final_next_stage" == "upload" ]]; then
        require_selected_uploaded=0
    fi
    "${clean_python[@]}" - \
        "$DELIVERY_STATE" \
        "$FULL_DIR" \
        "$DIST_DIR/$INFERENCE_ASSET" \
        "$DIST_DIR/$FINAL_ASSET" \
        "$input_staging_parent/final-recovery" \
        "$require_selected_uploaded" <<'PY'
import sys

from scammark.result_contract import stage_verified_delivery_inputs

stage_verified_delivery_inputs(
    mode="final",
    delivery_state_path=sys.argv[1],
    runtime_full_dir=sys.argv[2],
    inference_archive_path=sys.argv[3],
    final_archive_path=sys.argv[4],
    destination=sys.argv[5],
    expected_count=5658,
    include_optional_final=False,
    require_selected_uploaded=sys.argv[6] == "1",
)
PY
    if [[ "$final_next_stage" == "complete" ]]; then
        printf 'Final evaluation delivery is already complete.\n'
        exit 0
    fi
    bash "$SCRIPT_DIR/recover_delivery.sh" --mode final "$final_next_stage"
    printf 'Final evaluation delivery recovery completed.\n'
    exit 0
fi

final_artifacts_complete=1
for final_artifact in \
    "$FULL_DIR/metrics_final.json" \
    "$FULL_DIR/final_evaluation_summary.md" \
    "$FULL_DIR/bertscore_environment.txt" \
    "$ROOT_DIR/models/huggingface/bertscore_asset_status.json" \
    "$FINALIZE_LOG"; do
    if [[ ! -f "$final_artifact" ]]; then
        final_artifacts_complete=0
    fi
done
if [[ "$final_artifacts_complete" == "1" ]] &&
    "${clean_python[@]}" -m scammark.result_contract validate-full \
        --full-dir "$FULL_DIR" \
        --mode final \
        --expected-count 5658 >/dev/null 2>&1; then
    bash "$SCRIPT_DIR/recover_delivery.sh" --mode final package
    printf 'Existing final evaluation was validated and delivered.\n'
    exit 0
fi

"${clean_python[@]}" - \
    "$ROOT_DIR" \
    "$OUTPUT_ROOT" \
    "$FULL_DIR" \
    "$LOG_DIR" \
    "$ROOT_DIR/models" \
    "$ROOT_DIR/models/huggingface" \
    "$FULL_DIR/metrics_final.json" \
    "$FULL_DIR/final_evaluation_summary.md" \
    "$FULL_DIR/bertscore_environment.txt" \
    "$FINALIZE_LOG" \
    "$ROOT_DIR/models/huggingface/bertscore_asset_status.json" <<'PY'
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
directories = tuple(map(Path, sys.argv[2:7]))
files = tuple(map(Path, sys.argv[7:]))
root_metadata = root.lstat()
if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
    raise SystemExit("Unsafe finalizer output namespace")

legacy_namespace = "scammark_qwen3_dashscope_server_full"
reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
directory_flag = getattr(os, "O_DIRECTORY", 0)
nofollow = getattr(os, "O_NOFOLLOW", 0)
if not directory_flag or not nofollow:
    raise SystemExit("Unsafe finalizer output namespace")

def reject_metadata(path, metadata, *, expected_kind):
    attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or (
        reparse_flag and attributes & reparse_flag
    ):
        raise SystemExit("Unsafe finalizer output namespace")
    if expected_kind == "directory":
        if not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit("Unsafe finalizer output namespace")
    elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SystemExit("Unsafe finalizer output namespace")
    if metadata.st_dev != root_metadata.st_dev:
        raise SystemExit("Unsafe finalizer output namespace")
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        raise SystemExit("Unsafe finalizer output namespace")
    if legacy_namespace in resolved.parts:
        raise SystemExit("Unsafe finalizer output namespace")

for path, expected_kind in (
    *((path, "directory") for path in directories),
    *((path, "file") for path in files),
):
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        reject_metadata(path, metadata, expected_kind=expected_kind)
        if expected_kind == "directory" and path != root and os.path.ismount(path):
            raise SystemExit("Unsafe finalizer output namespace")

recursive_roots = (
    (directories[1], False),  # full output
    (directories[2], False),  # logs
    (directories[4], True),   # Hugging Face cache
)

def walk_directory(path, descriptor, *, cache_root, allow_internal_links):
    with os.scandir(descriptor) as entries:
        for entry in entries:
            child = path / entry.name
            metadata = entry.stat(follow_symlinks=False)
            attributes = getattr(metadata, "st_file_attributes", 0)
            is_link = stat.S_ISLNK(metadata.st_mode) or (
                reparse_flag and attributes & reparse_flag
            )
            if is_link:
                if not allow_internal_links:
                    raise SystemExit("Unsafe finalizer writable subtree")
                try:
                    resolved = child.resolve(strict=True)
                    resolved.relative_to(cache_root)
                    target = resolved.stat()
                except (OSError, RuntimeError, ValueError):
                    raise SystemExit("Unsafe finalizer writable subtree")
                if (
                    legacy_namespace in resolved.parts
                    or target.st_dev != root_metadata.st_dev
                    or not stat.S_ISREG(target.st_mode)
                    or target.st_nlink != 1
                ):
                    raise SystemExit("Unsafe finalizer writable subtree")
                continue
            if metadata.st_dev != root_metadata.st_dev:
                raise SystemExit("Unsafe finalizer writable subtree")
            if stat.S_ISDIR(metadata.st_mode):
                if os.path.ismount(child):
                    raise SystemExit("Unsafe finalizer writable subtree")
                child_descriptor = os.open(
                    entry.name,
                    os.O_RDONLY | directory_flag | nofollow | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                try:
                    opened = os.fstat(child_descriptor)
                    if opened.st_dev != metadata.st_dev or opened.st_ino != metadata.st_ino:
                        raise SystemExit("Unsafe finalizer writable subtree")
                    walk_directory(
                        child,
                        child_descriptor,
                        cache_root=cache_root,
                        allow_internal_links=allow_internal_links,
                    )
                finally:
                    os.close(child_descriptor)
            elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise SystemExit("Unsafe finalizer writable subtree")

for subtree, allow_internal_links in recursive_roots:
    try:
        metadata = subtree.lstat()
    except FileNotFoundError:
        continue
    reject_metadata(subtree, metadata, expected_kind="directory")
    descriptor = os.open(
        subtree,
        os.O_RDONLY | directory_flag | nofollow | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        cache_root = subtree.resolve(strict=True)
        walk_directory(
            subtree,
            descriptor,
            cache_root=cache_root,
            allow_internal_links=allow_internal_links,
        )
    finally:
        os.close(descriptor)
PY

[[ "$(uname -m)" == "aarch64" ]] ||
    die "This finalizer compute path is pinned for ARM64/aarch64"
require_command docker
mkdir -p "$LOG_DIR" "$ROOT_DIR/models/huggingface"
compute_parent="$(mktemp -d "$OUTPUT_ROOT/.bertscore-compute.XXXXXX")"
chmod 700 "$compute_parent"
compute_venv="$compute_parent/venv"
compute_home="$compute_parent/home"
mkdir -m 700 "$compute_home"
container_compute_parent="/workspace/${compute_parent#"$ROOT_DIR"/}"

printf '[%s] Pulling the pinned ARM64 NVIDIA PyTorch image.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$FINALIZE_LOG"
sudo docker pull "$BERTSCORE_IMAGE" 2>&1 | tee -a "$FINALIZE_LOG"
sudo docker run --rm \
    --gpus all \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME="$container_compute_parent/home" \
    -e BERTSCORE_VENV="$container_compute_parent/venv" \
    -e HF_HOME=/workspace/models/huggingface \
    -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/workspace/src \
    -v "$input_snapshot/inference/full:/verified-inputs:ro" \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    "$BERTSCORE_IMAGE" \
    bash -lc '
set -Eeuo pipefail
unset PIP_CONSTRAINT
python -m venv --system-site-packages "$BERTSCORE_VENV"
"$BERTSCORE_VENV/bin/python" -m pip install \
    bert-score==0.3.13 \
    transformers==4.57.6 \
    huggingface-hub==0.36.0 \
    sentencepiece==0.2.1 \
    scipy==1.17.1
"$BERTSCORE_VENV/bin/python" -m scammark.prepare_eval_assets \
    --cache-dir models/huggingface \
    --device cuda
"$BERTSCORE_VENV/bin/python" -m pip freeze > \
    outputs/scammark_qwen3_dashscope_server_full_v2/full/bertscore_environment.txt
"$BERTSCORE_VENV/bin/python" -m scammark.evaluate \
    --predictions /verified-inputs/predictions.jsonl \
    --manifest /verified-inputs/manifest.jsonl \
    --engineering-metrics /verified-inputs/metrics.json \
    --with-bertscore \
    --bertscore-device cuda \
    --bertscore-batch-size 4 \
    --bertscore-model models/huggingface/FacebookAI_xlm-roberta-large_c23d21b0620b635a76227c604d44e43a9f0ee389 \
    --output outputs/scammark_qwen3_dashscope_server_full_v2/full/metrics_final.json
' 2>&1 | tee -a "$FINALIZE_LOG"

# Rebind mutable runtime inputs after compute before accepting final output.
"${clean_python[@]}" - \
    "$DELIVERY_STATE" \
    "$FULL_DIR" \
    "$DIST_DIR/$INFERENCE_ASSET" \
    "$DIST_DIR/$FINAL_ASSET" \
    "$input_staging_parent/postcompute" <<'PY'
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
)
PY

"${clean_python[@]}" - "$FULL_DIR/metrics_final.json" "$FULL_DIR/final_evaluation_summary.md" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
status = metrics.get("rationale", {}).get("end_to_end", {}).get("bertscore_status")
count = metrics.get("engineering", {}).get("prediction_sample_count")
if status != "computed" or count != 5658:
    raise SystemExit(f"Invalid final metrics: bertscore_status={status}, prediction_sample_count={count}")
binary = metrics["binary"]
subtype = metrics["subtype"]["end_to_end"]
grounding = metrics["grounding"]["end_to_end"]
rationale = metrics["rationale"]["end_to_end"]
summary = f"""# ScamMark Qwen3 Full Evaluation

- Samples: {count}
- Binary accuracy: {binary['accuracy']:.6f}
- Binary F1: {binary['f1']:.6f}
- End-to-end subtype accuracy: {subtype['accuracy']:.6f}
- End-to-end subtype macro F1: {subtype['macro_f1']:.6f}
- End-to-end grounding mIoU: {grounding['miou']:.6f}
- End-to-end grounding F1@0.5: {grounding['f1_at_0_5']:.6f}
- End-to-end multilingual BERTScore F1: {rationale['bertscore_f1']:.6f}
- Grounded rationale coverage@0.5: {rationale['grounded_rationale_coverage_at_0_5']:.6f}
- Human evaluation: {metrics['rationale']['human']['status']}
"""
summary_path.write_text(summary, encoding="utf-8")
print("Posthoc multilingual BERTScore is computed for the unchanged 5,658 predictions.")
PY

"${clean_python[@]}" -m scammark.result_contract validate-full \
    --full-dir "$FULL_DIR" \
    --mode final \
    --expected-count 5658 >/dev/null
# recover_delivery.sh owns package_results.sh, upload_release_assets.sh, and
# publish_results_branch.sh so successful stages are never repeated.
bash "$SCRIPT_DIR/recover_delivery.sh" --mode final package
printf 'Final evaluation is uploaded to the private release and results branch.\n'
