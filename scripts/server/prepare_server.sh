#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

readonly FROZEN_ASSET_SIZE=21802246
readonly FROZEN_ASSET_SHA256="8d0894a043913f6e049cfc4b212a310cb4fd79a810ff5651372c802f8ab2eb61"
readonly FROZEN_MANIFEST_SHA256="81904995efcae0bacdee1a2e31d0da9d92a81adb099c4a9d0b5b01da885e0d1c"
readonly FROZEN_EVIDENCE_SHA256="012ec051c0518900be2246ed5ef0a07185768bf8f451e378056f75289fa160ff"

run_without_provider_environment() (
    unset SCAMMARK_LLM_API_KEY
    unset SCAMMARK_LLM_BASE_URL
    unset SCAMMARK_VLM_API_KEY
    unset SCAMMARK_VLM_BASE_URL
    "$@"
)

require_safe_download_namespace() {
    "$VENV_PYTHON" - \
        "$ROOT_DIR" \
        "$(dirname "$DOWNLOAD_DIR")" \
        "$DOWNLOAD_DIR" \
        "$ROOT_DIR/outputs/scammark_qwen3_dashscope_server_full" <<'PY'
import os
import stat
import sys
from pathlib import Path

root, download_parent, download_target, legacy_root = map(
    lambda value: Path(os.path.abspath(value)),
    sys.argv[1:],
)

try:
    root_metadata = root.lstat()
except OSError as exc:
    raise SystemExit("Unsafe v2 download namespace: repository root is unavailable") from exc
if not stat.S_ISDIR(root_metadata.st_mode):
    raise SystemExit("Unsafe v2 download namespace: repository root is not a directory")

legacy_identities = set()
for legacy_path in (
    legacy_root,
    legacy_root / "full",
    legacy_root / "state_full",
    legacy_root / "logs",
    legacy_root / "pilot_10",
):
    try:
        legacy_metadata = legacy_path.stat()
    except (FileNotFoundError, OSError):
        continue
    if stat.S_ISDIR(legacy_metadata.st_mode):
        legacy_identities.add((legacy_metadata.st_dev, legacy_metadata.st_ino))

checked = set()
for guarded_path in (download_parent, download_target):
    try:
        relative = guarded_path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(
            "Unsafe v2 download namespace: path escapes the repository root"
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current in checked:
            continue
        checked.add(current)
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SystemExit(
                "Unsafe v2 download namespace: component metadata is unavailable"
            ) from exc
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        file_attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or (
            reparse_flag and file_attributes & reparse_flag
        ):
            raise SystemExit(
                "Unsafe v2 download namespace: component is a symlink or reparse point"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit(
                "Unsafe v2 download namespace: component is not a directory"
            )
        if metadata.st_dev != root_metadata.st_dev:
            raise SystemExit(
                "Unsafe v2 download namespace: component crosses the repository device"
            )
        if (metadata.st_dev, metadata.st_ino) in legacy_identities:
            raise SystemExit(
                "Unsafe v2 download namespace: component aliases the v1 namespace"
            )
        resolved = current.resolve(strict=False)
        if "scammark_qwen3_dashscope_server_full" in resolved.parts:
            raise SystemExit(
                "Unsafe v2 download namespace: component resolves into the v1 namespace"
            )
PY
}

directory_swap_with_rollback() {
    local staged_directory="$1"
    local destination="$2"
    local label="$3"
    local parent backup had_previous=0

    parent="$(dirname "$destination")"
    mkdir -p -- "$parent"
    backup="$(mktemp -d "$parent/.${label}.backup.XXXXXX")"
    rmdir -- "$backup"
    if [[ -e "$destination" || -L "$destination" ]]; then
        mv -- "$destination" "$backup"
        had_previous=1
    fi
    if mv -- "$staged_directory" "$destination"; then
        if [[ "$had_previous" == "1" ]]; then
            rm -rf -- "$backup"
        fi
        return 0
    fi

    if [[ "$had_previous" == "1" ]]; then
        mv -- "$backup" "$destination" ||
            die "Failed to restore the previous $label directory"
    fi
    die "Failed to install the validated $label directory"
}

cd "$ROOT_DIR"
[[ "$(uname -m)" == "aarch64" ]] || die "This server workflow is pinned for ARM64/aarch64"
for command_name in python3.12 git curl tar sha256sum tmux gh stat; do
    require_command "$command_name"
done
require_clean_worktree
require_exact_release_tag
head_commit="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if ! remote_tag_line="$(git -C "$ROOT_DIR" ls-remote --tags origin "refs/tags/$RELEASE_TAG^{}")"; then
    die "Unable to resolve the remote peeled tag for $RELEASE_TAG"
fi
[[ -n "$remote_tag_line" && "$remote_tag_line" != *$'\n'* ]] ||
    die "The remote peeled tag for $RELEASE_TAG is missing or ambiguous"
IFS=$'\t' read -r remote_tag_commit remote_tag_ref remote_tag_extra <<<"$remote_tag_line"
[[ -z "$remote_tag_extra" && "$remote_tag_ref" == "refs/tags/$RELEASE_TAG^{}" ]] ||
    die "The remote peeled tag response is not exact"
[[ "$head_commit" =~ ^[0-9a-f]{40,64}$ && "$remote_tag_commit" == "$head_commit" ]] ||
    die "HEAD does not match the remote peeled tag for $RELEASE_TAG"

require_file "$ENV_FILE"
dataset_root="$(python3.12 - "$ENV_FILE" "$DEFAULT_DATASET_ROOT" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
default_dataset_root = sys.argv[2]
lines = path.read_text(encoding="utf-8-sig").splitlines()
key = "SCAMWEB_DATASET_ROOT"
updated = []
found = False
selected_dataset_root = default_dataset_root
changed = False
for line in lines:
    stripped = line.strip()
    if "=" in stripped and stripped.split("=", 1)[0].strip() == key:
        current = stripped.split("=", 1)[1].strip().strip('"').strip("'")
        is_placeholder = any(
            token in current.upper() for token in ("PASTE_", "YOUR_", "<", ">")
        )
        if current and not is_placeholder:
            selected_dataset_root = current
            updated.append(line)
        else:
            updated.append(f"{key}={default_dataset_root}")
            changed = True
        found = True
    else:
        updated.append(line)
if not found:
    updated.extend(["", f"{key}={default_dataset_root}"])
    changed = True
if changed:
    temporary = path.with_suffix(".env.tmp")
    temporary.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
os.chmod(path, 0o600)
print(selected_dataset_root)
PY
)"
printf 'Validated the configured server dataset root without displaying provider secrets.\n'
validate_env_file
validate_dataset_counts "$dataset_root"

if [[ ! -x "$VENV_PYTHON" ]]; then
    if ! python3.12 -m venv "$VENV_DIR"; then
        printf 'python3.12-venv is unavailable; installing it with sudo.\n'
        sudo apt-get update
        sudo apt-get install -y python3.12-venv
        python3.12 -m venv "$VENV_DIR"
    fi
fi
"$VENV_PYTHON" -m pip install --require-hashes -r "$ROOT_DIR/requirements-server.lock.txt"
require_safe_runtime_namespace
require_safe_download_namespace
require_github_auth

mkdir -p -- "$(dirname "$DOWNLOAD_DIR")"
require_safe_download_namespace
mkdir -p -- "$ROOT_DIR/outputs" "$LOG_DIR"
download_staging="$(mktemp -d "$(dirname "$DOWNLOAD_DIR")/.full-v2-download.XXXXXX")"
extract_staging="$(mktemp -d "$ROOT_DIR/outputs/.frozen-extract.XXXXXX")"
cleanup_staging() {
    rm -rf -- "$download_staging" "$extract_staging"
}
trap cleanup_staging EXIT

gh release download "$RELEASE_TAG" \
    --repo "$REPOSITORY" \
    --dir "$download_staging" \
    --clobber \
    --pattern "$FROZEN_ASSET" \
    --pattern "$FROZEN_ASSET.sha256"

archive="$download_staging/$FROZEN_ASSET"
checksum="$archive.sha256"
[[ -f "$archive" && ! -L "$archive" ]] || die "Frozen archive download is missing or unsafe"
[[ -f "$checksum" && ! -L "$checksum" ]] || die "Frozen checksum download is missing or unsafe"
actual_size="$(stat -c %s "$archive")"
[[ "$actual_size" == "$FROZEN_ASSET_SIZE" ]] ||
    die "Frozen archive size mismatch: expected $FROZEN_ASSET_SIZE, found $actual_size"
actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
[[ "$actual_sha256" == "$FROZEN_ASSET_SHA256" ]] ||
    die "Frozen archive SHA256 mismatch"
python3.12 - "$checksum" "$FROZEN_ASSET_SHA256  $FROZEN_ASSET" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2].encode("ascii")
if path.read_bytes() != expected:
    raise SystemExit("Frozen checksum content is not exact")
PY
(
    cd "$download_staging"
    sha256sum -c "$FROZEN_ASSET.sha256"
)

python3.12 - "$archive" <<'PY'
import sys
import tarfile

expected = [
    "dataset_frozen_v1/boxes.jsonl",
    "dataset_frozen_v1/dataset_version.json",
    "dataset_frozen_v1/mhtml_audit.jsonl",
    "dataset_frozen_v1/mhtml_recapture.jsonl",
    "dataset_frozen_v1/sample_ids.txt",
    "dataset_frozen_v1/scamweb_v_manifest.jsonl",
    "dataset_frozen_v1/verification_report.json",
    "dataset_frozen_v1/web_evidence.jsonl",
]
try:
    with tarfile.open(sys.argv[1], mode="r:gz") as bundle:
        members = bundle.getmembers()
except (OSError, tarfile.TarError) as exc:
    raise SystemExit("Frozen archive could not be parsed safely") from exc
if [member.name for member in members] != expected:
    raise SystemExit("Frozen archive member allowlist mismatch")
if any(not member.isfile() for member in members):
    raise SystemExit("Frozen archive contains a non-regular member")
print("Frozen archive member allowlist verified.")
PY
tar -xzf "$archive" -C "$extract_staging"
staged_frozen="$extract_staging/dataset_frozen_v1"
[[ -d "$staged_frozen" && ! -L "$staged_frozen" ]] ||
    die "Frozen archive has an unexpected directory layout"
manifest="$staged_frozen/scamweb_v_manifest.jsonl"
evidence="$staged_frozen/web_evidence.jsonl"
[[ -f "$manifest" && ! -L "$manifest" ]] || die "Frozen manifest is missing or unsafe"
[[ -f "$evidence" && ! -L "$evidence" ]] || die "Frozen evidence is missing or unsafe"
manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
[[ "$manifest_sha256" == "$FROZEN_MANIFEST_SHA256" ]] ||
    die "Frozen manifest SHA256 mismatch"
evidence_sha256="$(sha256sum "$evidence" | awk '{print $1}')"
[[ "$evidence_sha256" == "$FROZEN_EVIDENCE_SHA256" ]] ||
    die "Frozen evidence SHA256 mismatch"

"$VENV_PYTHON" -m scammark.frozen_assets \
    --root "$staged_frozen" \
    --output "$staged_frozen/verification_report.json"
directory_swap_with_rollback "$staged_frozen" "$FROZEN_DIR" "dataset-frozen-v1"
require_safe_download_namespace
directory_swap_with_rollback "$download_staging" "$DOWNLOAD_DIR" "full-v2-download"

run_without_provider_environment \
    "$VENV_PYTHON" -m unittest discover -s tests -p 'test_*.py'
printf 'Fake-only test suite passed with provider variables removed.\n'
run_without_provider_environment \
    "$VENV_PYTHON" -m scammark.run_experiment \
    --config "$CONFIG_PATH" \
    --full \
    --validate-only

"$VENV_PYTHON" - "$FULL_DIR/preflight.json" <<'PY'
import json
import re
import sys
from pathlib import Path

def reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Preflight contains a duplicate key: {key}")
        payload[key] = value
    return payload


def reject_nonfinite_constant(value):
    raise ValueError(f"Preflight contains a non-finite constant: {value}")


try:
    payload = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite_constant,
    )
except (OSError, UnicodeError, ValueError) as exc:
    raise SystemExit(str(exc)) from exc
if type(payload) is not dict:
    raise SystemExit("Preflight must be a JSON object")
if payload.get("status") != "validated":
    raise SystemExit("Preflight status is not validated")
if payload.get("run_mode") != "full":
    raise SystemExit("Preflight run mode is not full")
if type(payload.get("selected_sample_count")) is not int or payload.get("selected_sample_count") != 5658:
    raise SystemExit("Preflight selected sample count is not 5658")
if type(payload.get("expected_full_sample_count")) is not int or payload.get("expected_full_sample_count") != 5658:
    raise SystemExit("Preflight expected full sample count is not 5658")
if type(payload.get("contract_image_count")) is not int or payload.get("contract_image_count") != 5658:
    raise SystemExit("Preflight contract image count is not 5658")
if type(payload.get("api_call_limit")) is not int or payload.get("api_call_limit") != 50000:
    raise SystemExit("Preflight physical-call limit is not 50000")
implementation = payload.get("implementation_sha256")
if type(implementation) is not dict:
    raise SystemExit("Preflight implementation hashes are missing")
for name in ("observability.py", "run_status.py", "result_contract.py"):
    value = implementation.get(name)
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit(f"Preflight implementation hash is invalid: {name}")
print("Zero-API full preflight passed for exactly 5,658 samples and 50,000 calls.")
PY

printf '\nServer preparation is complete. Paid smoke command (not executed):\n'
printf '  %q -m scammark.run_experiment --config %q --pilot 10 --resume\n' "$VENV_PYTHON" "$CONFIG_PATH"
printf 'After the paid smoke passes, start the full run with:\n  bash scripts/server/start_full.sh\n'
