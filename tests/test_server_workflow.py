from copy import deepcopy
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import tomllib
import unittest
from pathlib import Path

from tests.tmux_fixture import ManagedTmuxFixture


ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPTS = ROOT / "scripts" / "server"


class OfflineDeliveryFixture:
    """Execute delivery scripts with real local files and fake externals."""

    def __init__(self) -> None:
        from tests.result_fixture import CompleteResultFixture

        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name) / "repo"
        self.root.mkdir()
        shutil.copytree(SERVER_SCRIPTS, self.root / "scripts" / "server")
        (self.root / "src").symlink_to(ROOT / "src", target_is_directory=True)
        (self.root / "configs").mkdir()
        shutil.copy2(
            ROOT / "configs" / "qwen3_dashscope_server_full_v2.toml",
            self.root / "configs" / "qwen3_dashscope_server_full_v2.toml",
        )
        output = (
            self.root
            / "outputs"
            / "scammark_qwen3_dashscope_server_full_v2"
        )
        with CompleteResultFixture(sample_count=5658) as source:
            shutil.copytree(source.full, output / "full")
            shutil.copytree(source.state, output / "state_full")
            (output / "logs").mkdir()
            shutil.copy2(source.log, output / "logs" / "full_run.log")
            shutil.copy2(
                source.finalize_log,
                output / "logs" / "bertscore_finalize.log",
            )
            shutil.copytree(
                source.frozen,
                self.root / "outputs" / "dataset_frozen_v1",
            )
            status = (
                self.root
                / "models"
                / "huggingface"
                / "bertscore_asset_status.json"
            )
            status.parent.mkdir(parents=True)
            shutil.copy2(source.bertscore_asset_status, status)

        self.output = output
        self.full = output / "full"
        self.delivery_state = self.full / "delivery_state.json"
        self.dist = self.root / "dist" / "release-assets" / "full-v2"
        self.command_log = Path(self._temporary.name) / "commands.log"
        self.python_log = Path(self._temporary.name) / "python.log"
        self.exposure_marker = Path(self._temporary.name) / "secret-exposure"
        self.publish_capture = Path(self._temporary.name) / "publish-files.log"
        self.publish_tree = Path(self._temporary.name) / "publish-tree"
        self.remote_assets = Path(self._temporary.name) / "remote-assets"
        self.tmux_log = Path(self._temporary.name) / "tmux.log"
        self.docker_marker = Path(self._temporary.name) / "docker-called"

        env_path = self.root / ".env"
        env_path.write_text(
            "SCAMMARK_LLM_API_KEY=fixture-private-llm\n"
            "SCAMMARK_VLM_API_KEY=fixture-private-vlm\n"
            "SCAMMARK_VLM_BASE_URL=https://fixture-private.invalid/v1\n"
            "SCAMWEB_DATASET_ROOT=/must-not-enter-packaging\n",
            encoding="utf-8",
        )
        env_path.chmod(0o600)
        fake_runner = self.root / "fake_runner.py"
        fake_runner.write_text("raise SystemExit(0)\n", encoding="utf-8")

        venv_bin = self.root / ".venv-server" / "bin"
        venv_bin.mkdir(parents=True)
        python_wrapper = venv_bin / "python"
        python_wrapper.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
for argument in "$@"; do
    case "$argument" in
        *fixture-private*) : >"$FAKE_EXPOSURE_MARKER" ;;
    esac
done
inspect_provider_environment=1
if [[ "${FAKE_EXPOSURE_ONLY_RESULT_CONTRACT:-}" == "1" \
    && !( "${1:-}" == "-m" && "${2:-}" == "scammark.result_contract" ) ]]; then
    inspect_provider_environment=0
fi
if [[ "$inspect_provider_environment" == "1" ]]; then
    for name in SCAMMARK_LLM_API_KEY SCAMMARK_LLM_BASE_URL \
        SCAMMARK_VLM_API_KEY SCAMMARK_VLM_BASE_URL secret_values_json; do
        value="${!name-}"
        case "$value" in
            *fixture-private*) : >"$FAKE_EXPOSURE_MARKER" ;;
        esac
    done
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "scammark.result_contract" ]]; then
    printf 'result_contract:%s\n' "${3:-missing}" >>"$FAKE_PYTHON_LOG"
else
    printf 'python\n' >>"$FAKE_PYTHON_LOG"
fi
exec "$FAKE_REAL_PYTHON" "$@"
""",
            encoding="utf-8",
        )
        python_wrapper.chmod(0o755)

        fake_bin = self.root / ".fake-bin"
        fake_bin.mkdir()
        self._write_executable(
            fake_bin / "git",
            """#!/usr/bin/env bash
set -Eeuo pipefail
args=("$@")
command_name=""
command_index=-1
for ((index = 0; index < ${#args[@]}; index++)); do
    case "${args[index]}" in
        -C|-c) ((index += 1)) ;;
        -*) ;;
        *) command_name="${args[index]}"; command_index="$index"; break ;;
    esac
done
printf 'git:%s\n' "$command_name" >>"$FAKE_COMMAND_LOG"
case "$command_name" in
    status)
        if [[ "${FAKE_GIT_DIRTY:-}" == "1" ]]; then
            printf ' M fixture-dirty\n'
        fi
        exit 0
        ;;
    tag) printf 'experiment-qwen3-dashscope-full-v2\n' ;;
    rev-parse) printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n' ;;
    fetch)
        if [[ "${FAKE_SWAP_PUBLISH_SOURCE_ON_FETCH:-}" == "1" ]]; then
            rm -f -- "$FAKE_SWAP_PUBLISH_SOURCE"
            ln -s -- "$FAKE_SWAP_PUBLISH_TARGET" "$FAKE_SWAP_PUBLISH_SOURCE"
        fi
        exit 0
        ;;
    branch|config|add|commit|push) exit 0 ;;
    show-ref) exit 1 ;;
    mktree) printf 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n' ;;
    commit-tree) printf 'cccccccccccccccccccccccccccccccccccccccc\n' ;;
    diff) exit 1 ;;
    worktree)
        action="${args[command_index + 1]:-}"
        target=""
        for value in "${args[@]}"; do
            if [[ "$value" == /* && "$value" != "$FAKE_REPO_ROOT" ]]; then
                target="$value"
            fi
        done
        if [[ "$action" == "add" ]]; then
            mkdir -p "$target"
            if [[ "${FAKE_WORKTREE_SEED:-}" == "1" ]]; then
                printf 'raw_response=unexpected\n' >"$target/unexpected.txt"
            fi
        elif [[ "$action" == "remove" ]]; then
            if [[ -d "$target" ]]; then
                (cd "$target" && find . -type f -printf '%P\n' | sort) \
                    >"$FAKE_PUBLISH_CAPTURE"
                rm -rf -- "$FAKE_PUBLISH_TREE"
                mkdir -p "$FAKE_PUBLISH_TREE"
                while IFS= read -r relative; do
                    mkdir -p "$FAKE_PUBLISH_TREE/$(dirname "$relative")"
                    cp -- "$target/$relative" "$FAKE_PUBLISH_TREE/$relative"
                done <"$FAKE_PUBLISH_CAPTURE"
                rm -rf -- "$target"
            fi
        fi
        ;;
esac
""",
        )
        self._write_executable(
            fake_bin / "gh",
            """#!/usr/bin/env bash
set -Eeuo pipefail
printf 'gh:%s:%s\n' "${1:-}" "${2:-}" >>"$FAKE_COMMAND_LOG"
if [[ "${1:-}" == "api" ]]; then
    if [[ "${FAKE_TAMPER_ARCHIVE_ON_API:-}" == "1" \
        && ! -f "$FAKE_TAMPER_DONE" ]]; then
        printf 'tamper' >>"$FAKE_TAMPER_ARCHIVE"
        : >"$FAKE_TAMPER_DONE"
    fi
    requested=""
    for candidate in \
        scammark-qwen3-dashscope-full-v2-inference.tar.gz \
        scammark-qwen3-dashscope-full-v2-inference.tar.gz.sha256 \
        scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz \
        scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz.sha256; do
        if [[ "$*" == *"$candidate"* ]]; then
            requested="$candidate"
        fi
    done
    if [[ -n "$requested" && -f "$FAKE_REMOTE_ASSETS/$requested" ]]; then
        digest="$(<"$FAKE_REMOTE_ASSETS/$requested")"
        case "${FAKE_POST_UPLOAD_DIGEST_MODE:-}" in
            wrong) digest="sha256:$(printf '0%.0s' {1..64})" ;;
            unavailable) digest="" ;;
        esac
        printf '{"name":"%s","digest":"%s"}\n' "$requested" "$digest"
    fi
    exit 0
fi
if [[ "${1:-}" == "release" && "${2:-}" == "upload" ]]; then
    file="${4:-}"
    name="$(basename "$file")"
    mkdir -p "$FAKE_REMOTE_ASSETS"
    if [[ "${FAKE_SWAP_UPLOAD_PATH:-}" == "1" ]]; then
        backup="$FAKE_REMOTE_ASSETS/.upload-backup"
        cp -- "$file" "$backup"
        printf 'wrong-upload-bytes' >"$file"
        digest="$(sha256sum "$file" | awk '{print $1}')"
        mv -- "$backup" "$file"
    else
        digest="$(sha256sum "$file" | awk '{print $1}')"
    fi
    printf 'sha256:%s\n' "$digest" >"$FAKE_REMOTE_ASSETS/$name"
    exit 0
fi
exit 0
""",
        )
        self._write_executable(
            fake_bin / "tmux",
            """#!/usr/bin/env bash
set -Eeuo pipefail
case "${1:-}" in
    has-session) [[ -n "${FAKE_TMUX_STATE:-}" ]] ;;
    display-message) printf '%s\n' "$FAKE_TMUX_STATE" ;;
    run-shell) exit 0 ;;
    kill-session) printf 'kill:%s\n' "${3:-}" >>"$FAKE_TMUX_LOG" ;;
    *) exit 1 ;;
esac
""",
        )
        self._write_executable(
            fake_bin / "docker",
            """#!/usr/bin/env bash
set -Eeuo pipefail
: >"$FAKE_DOCKER_MARKER"
exit 99
""",
        )
        self._write_executable(
            fake_bin / "uname",
            """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -n "${FAKE_UNAME_ARCH:-}" ]]; then
    printf '%s\n' "$FAKE_UNAME_ARCH"
else
    exec /usr/bin/uname "$@"
fi
""",
        )
        self._write_executable(
            fake_bin / "sudo",
            """#!/usr/bin/env bash
set -Eeuo pipefail
exec "$@"
""",
        )

        self.env = os.environ.copy()
        for key in (
            "SCAMMARK_LLM_API_KEY",
            "SCAMMARK_LLM_BASE_URL",
            "SCAMMARK_VLM_API_KEY",
            "SCAMMARK_VLM_BASE_URL",
            "secret_values_json",
        ):
            self.env.pop(key, None)
        self.env.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}/usr/bin:/bin",
                "HOME": str(Path(self._temporary.name) / "home"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "FAKE_REAL_PYTHON": sys.executable,
                "FAKE_PYTHON_LOG": str(self.python_log),
                "FAKE_EXPOSURE_MARKER": str(self.exposure_marker),
                "FAKE_COMMAND_LOG": str(self.command_log),
                "FAKE_REPO_ROOT": str(self.root),
                "FAKE_PUBLISH_CAPTURE": str(self.publish_capture),
                "FAKE_PUBLISH_TREE": str(self.publish_tree),
                "FAKE_REMOTE_ASSETS": str(self.remote_assets),
                "FAKE_TMUX_LOG": str(self.tmux_log),
                "FAKE_DOCKER_MARKER": str(self.docker_marker),
                "FAKE_TAMPER_DONE": str(Path(self._temporary.name) / "tampered"),
                "SCAMMARK_ALLOW_TEST_OVERRIDES": "1",
                "SCAMMARK_TEST_RUNNER": str(fake_runner),
                "SCAMMARK_TEST_TMUX_SESSION": "scammark-full-v2-test-delivery",
            }
        )
        Path(self.env["HOME"]).mkdir()

    @staticmethod
    def _write_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def run(
        self,
        script: str,
        *arguments: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self.env.copy()
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", f"scripts/server/{script}", *arguments],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def state(self, mode: str) -> object:
        from scammark.result_contract import load_delivery_state

        return load_delivery_state(self.delivery_state, mode=mode)

    def log_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def clear_logs(self) -> None:
        for path in (
            self.command_log,
            self.python_log,
            self.publish_capture,
            self.tmux_log,
        ):
            path.unlink(missing_ok=True)

    def cleanup(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> "OfflineDeliveryFixture":
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self.cleanup()


class OfflinePrepareFixture:
    """Run prepare_server.sh against local fakes and a tiny frozen tree."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        temporary = Path(self._temporary.name)
        self.root = temporary / "repo"
        self.root.mkdir()
        shutil.copytree(SERVER_SCRIPTS, self.root / "scripts" / "server")
        (self.root / "configs").mkdir()
        shutil.copy2(
            ROOT / "configs" / "qwen3_dashscope_server_full_v2.toml",
            self.root / "configs" / "qwen3_dashscope_server_full_v2.toml",
        )
        shutil.copy2(
            ROOT / "requirements-server.lock.txt",
            self.root / "requirements-server.lock.txt",
        )
        self.frozen = self.root / "outputs" / "dataset_frozen_v1"
        self.frozen.mkdir(parents=True)
        (self.frozen / "old-sentinel.txt").write_text(
            "must survive failed preparation\n",
            encoding="utf-8",
        )
        self.v1_output = (
            self.root
            / "outputs"
            / "scammark_qwen3_dashscope_server_full"
        )
        (self.v1_output / "full").mkdir(parents=True)
        (self.v1_output / "full" / "v1-sentinel.bin").write_bytes(
            b"immutable-v1-runtime\x00sentinel\n"
        )
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.frozen_archive = temporary / "frozen-fixture.tar.gz"
        frozen_members = (
            "dataset_frozen_v1/boxes.jsonl",
            "dataset_frozen_v1/dataset_version.json",
            "dataset_frozen_v1/mhtml_audit.jsonl",
            "dataset_frozen_v1/mhtml_recapture.jsonl",
            "dataset_frozen_v1/sample_ids.txt",
            "dataset_frozen_v1/scamweb_v_manifest.jsonl",
            "dataset_frozen_v1/verification_report.json",
            "dataset_frozen_v1/web_evidence.jsonl",
        )
        with tarfile.open(self.frozen_archive, "w:gz") as bundle:
            for name in frozen_members:
                payload = b"{}\n"
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mtime = 0
                member.mode = 0o644
                bundle.addfile(member, io.BytesIO(payload))
        self.command_log = temporary / "commands.log"
        self.python_log = temporary / "python.log"
        self.provider_leak = temporary / "provider-leak"
        (self.root / ".env").write_text(
            "SCAMMARK_LLM_API_KEY=file-llm-secret\n"
            "SCAMMARK_LLM_BASE_URL=https://file-llm.invalid/v1\n"
            "SCAMMARK_VLM_API_KEY=file-vlm-secret\n"
            "SCAMMARK_VLM_BASE_URL=https://file-vlm.invalid/v1\n"
            f"SCAMWEB_DATASET_ROOT={self.dataset}\n",
            encoding="utf-8",
        )
        (self.root / ".env").chmod(0o600)

        fake_bin = self.root / ".fake-bin"
        fake_bin.mkdir()
        self._write_executable(
            fake_bin / "uname",
            """#!/usr/bin/env bash
printf 'aarch64\n'
""",
        )
        self._write_executable(
            fake_bin / "git",
            """#!/usr/bin/env bash
set -Eeuo pipefail
printf 'git:%s\n' "$*" >>"$FAKE_COMMAND_LOG"
case "$*" in
    *" status --short"*)
        [[ "${FAKE_GIT_DIRTY:-}" == "1" ]] && printf ' M dirty\n'
        ;;
    *" tag --points-at HEAD"*)
        printf '%s\n' "${FAKE_GIT_TAG:-experiment-qwen3-dashscope-full-v2}"
        ;;
    *" rev-parse HEAD"*)
        printf '%s\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        ;;
    *" ls-remote --tags origin refs/tags/experiment-qwen3-dashscope-full-v2^{}"*)
        printf '%s\t%s\n' \
            "${FAKE_REMOTE_TAG_COMMIT:-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}" \
            'refs/tags/experiment-qwen3-dashscope-full-v2^{}'
        ;;
esac
""",
        )
        self._write_executable(
            fake_bin / "gh",
            """#!/usr/bin/env bash
set -Eeuo pipefail
printf 'gh:%s\n' "$*" >>"$FAKE_COMMAND_LOG"
if [[ "${1:-}" == "auth" ]]; then
    exit 0
fi
destination=""
while (($#)); do
    if [[ "$1" == "--dir" ]]; then
        destination="$2"
        shift 2
    else
        shift
    fi
done
[[ -n "$destination" ]]
mkdir -p -- "$destination"
cp -- "$FAKE_FROZEN_ARCHIVE" \
    "$destination/scamweb-v1.0-frozen-inputs.tar.gz"
printf '%s  %s' \
    8d0894a043913f6e049cfc4b212a310cb4fd79a810ff5651372c802f8ab2eb61 \
    scamweb-v1.0-frozen-inputs.tar.gz \
    >"$destination/scamweb-v1.0-frozen-inputs.tar.gz.sha256"
""",
        )
        self._write_executable(
            fake_bin / "find",
            """#!/usr/bin/env bash
set -Eeuo pipefail
case "$*" in
    *"*.png"*) count=5658 ;;
    *"*.mhtml"*) count=5658 ;;
    *"*.json"*) count=3416 ;;
    *) exec /usr/bin/find "$@" ;;
esac
for ((index = 0; index < count; index++)); do
    printf 'fixture\n'
done
""",
        )
        self._write_executable(
            fake_bin / "stat",
            """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "-c" && "${2:-}" == "%s" ]]; then
    printf '21802246\n'
else
    exec /usr/bin/stat "$@"
fi
""",
        )
        self._write_executable(
            fake_bin / "sha256sum",
            """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "-c" ]]; then
    printf 'scamweb-v1.0-frozen-inputs.tar.gz: OK\n'
    exit 0
fi
path="${1:-}"
case "$path" in
    *scamweb-v1.0-frozen-inputs.tar.gz)
        digest=8d0894a043913f6e049cfc4b212a310cb4fd79a810ff5651372c802f8ab2eb61
        ;;
    *scamweb_v_manifest.jsonl)
        [[ -f "$path" ]] || exit 1
        if [[ "${FAKE_BAD_INTERNAL_HASH:-}" == "1" ]]; then
            digest=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        else
            digest=81904995efcae0bacdee1a2e31d0da9d92a81adb099c4a9d0b5b01da885e0d1c
        fi
        ;;
    *web_evidence.jsonl)
        [[ -f "$path" ]] || exit 1
        digest=012ec051c0518900be2246ed5ef0a07185768bf8f451e378056f75289fa160ff
        ;;
    *) exec /usr/bin/sha256sum "$@" ;;
esac
printf '%s  %s\n' "$digest" "$path"
""",
        )
        self._write_executable(
            fake_bin / "tar",
            """#!/usr/bin/env bash
set -Eeuo pipefail
printf 'tar:%s\n' "$*" >>"$FAKE_COMMAND_LOG"
exec /usr/bin/tar "$@"
""",
        )

        venv_bin = self.root / ".venv-server" / "bin"
        venv_bin.mkdir(parents=True)
        self._write_executable(
            venv_bin / "python",
            """#!/usr/bin/env bash
set -Eeuo pipefail
check_provider_environment() {
    for name in SCAMMARK_LLM_API_KEY SCAMMARK_LLM_BASE_URL \
        SCAMMARK_VLM_API_KEY SCAMMARK_VLM_BASE_URL; do
        if [[ -n "${!name+x}" ]]; then
            printf '%s\n' "$name" >"$FAKE_PROVIDER_LEAK"
            exit 97
        fi
    done
}
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
    printf 'pip\n' >>"$FAKE_PYTHON_LOG"
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "scammark.frozen_assets" ]]; then
    output=""
    while (($#)); do
        if [[ "$1" == "--output" ]]; then
            output="$2"
            shift 2
        else
            shift
        fi
    done
    mkdir -p -- "$(dirname "$output")"
    printf '{"verified":true}\n' >"$output"
    printf 'frozen-assets\n' >>"$FAKE_PYTHON_LOG"
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "unittest" ]]; then
    check_provider_environment
    printf 'unittest:providers-cleared\n' >>"$FAKE_PYTHON_LOG"
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "scammark.run_experiment" ]]; then
    check_provider_environment
    mkdir -p -- "$FAKE_REPO_ROOT/outputs/scammark_qwen3_dashscope_server_full_v2/full"
    if [[ "${FAKE_BAD_PREFLIGHT:-}" == "1" ]]; then
        status=invalid
    else
        status=validated
    fi
    if [[ "${FAKE_DUPLICATE_PREFLIGHT_KEY:-}" == "1" ]]; then
        status_fields='"status":"validated","status":"validated"'
    else
        status_fields='"status":"'"$status"'"'
    fi
    printf '{%s,"run_mode":"full","selected_sample_count":5658,"expected_full_sample_count":5658,"contract_image_count":5658,"api_call_limit":50000,"implementation_sha256":{"observability.py":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","run_status.py":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","result_contract.py":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}}\n' \
        "$status_fields" \
        >"$FAKE_REPO_ROOT/outputs/scammark_qwen3_dashscope_server_full_v2/full/preflight.json"
    printf 'validate-only:providers-cleared\n' >>"$FAKE_PYTHON_LOG"
    exit 0
fi
exec "$FAKE_REAL_PYTHON" "$@"
""",
        )

        self.env = os.environ.copy()
        self.env.update(
            {
                "PATH": f"{fake_bin}:{self.env['PATH']}",
                "HOME": str(temporary / "home"),
                "FAKE_COMMAND_LOG": str(self.command_log),
                "FAKE_PYTHON_LOG": str(self.python_log),
                "FAKE_PROVIDER_LEAK": str(self.provider_leak),
                "FAKE_FROZEN_ARCHIVE": str(self.frozen_archive),
                "FAKE_REAL_PYTHON": sys.executable,
                "FAKE_REPO_ROOT": str(self.root),
                "SCAMMARK_LLM_API_KEY": "inherited-llm-secret",
                "SCAMMARK_LLM_BASE_URL": "https://inherited-llm.invalid/v1",
                "SCAMMARK_VLM_API_KEY": "inherited-vlm-secret",
                "SCAMMARK_VLM_BASE_URL": "https://inherited-vlm.invalid/v1",
            }
        )
        Path(self.env["HOME"]).mkdir()

    @staticmethod
    def _write_executable(path: Path, source: str) -> None:
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    def run(
        self,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self.env.copy()
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", "scripts/server/prepare_server.sh"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def log(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def __enter__(self) -> "OfflinePrepareFixture":
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self._temporary.cleanup()


class ServerWorkflowContractTests(unittest.TestCase):
    @staticmethod
    def _exact_tree_inventory(root: Path) -> tuple[tuple[object, ...], ...]:
        rows = []
        for path in sorted((root, *root.rglob("*"))):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                kind = "symlink"
                content: bytes | str | None = os.readlink(path)
            elif stat.S_ISREG(metadata.st_mode):
                kind = "file"
                content = path.read_bytes()
            elif stat.S_ISDIR(metadata.st_mode):
                kind = "directory"
                content = None
            else:
                kind = "special"
                content = None
            rows.append(
                (
                    path.relative_to(root).as_posix(),
                    kind,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_mtime_ns,
                    content,
                )
            )
        return tuple(rows)

    @staticmethod
    def _rewrite_result_archive(archive: Path, mutator: object) -> None:
        entries: list[list[object]] = []
        with tarfile.open(archive, mode="r:gz") as source:
            for member in source.getmembers():
                payload = None
                if member.isfile():
                    extracted = source.extractfile(member)
                    if extracted is None:
                        raise AssertionError(f"missing payload for {member.name}")
                    payload = extracted.read()
                entries.append([deepcopy(member), payload])
        mutator(entries)
        replacement = archive.with_name(f".{archive.name}.replacement")
        with tarfile.open(replacement, mode="w:gz") as target:
            for member, payload in entries:
                if member.isfile():
                    member.size = len(payload)
                    target.addfile(member, io.BytesIO(payload))
                else:
                    target.addfile(member)
        os.replace(replacement, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_name(f"{archive.name}.sha256").write_text(
            f"{digest}  {archive.name}\n",
            encoding="ascii",
        )

    @staticmethod
    def _copy_windows_downloader(root: Path) -> tuple[Path, Path]:
        client_root = root / "windows-client"
        script_directory = client_root / "scripts" / "windows"
        script_directory.mkdir(parents=True)
        downloader = script_directory / "download_full_results_v2.ps1"
        shutil.copy2(
            ROOT / "scripts" / "windows" / "download_full_results_v2.ps1",
            downloader,
        )
        return client_root, downloader

    @staticmethod
    def _archive_race_environment(
        fixture: OfflineDeliveryFixture,
        *,
        race: str,
        sentinel: Path,
        replacement: Path,
        marker: Path,
        restore_marker: Path | None = None,
    ) -> dict[str, str]:
        hook = fixture.root.parent / f"sitecustomize-{race}"
        hook.mkdir()
        (hook / "sitecustomize.py").write_text(
            """import os
from pathlib import Path
import shutil
import tempfile

race = os.environ.get("FAKE_ARCHIVE_RACE")
asset = os.environ.get("FAKE_ARCHIVE_RACE_ASSET", "")
marker = Path(os.environ["FAKE_ARCHIVE_RACE_MARKER"])
restore_marker = Path(os.environ["FAKE_ARCHIVE_RESTORE_MARKER"])
original_mkstemp = tempfile.mkstemp
original_replace = os.replace

def injected_mkstemp(*args, **kwargs):
    descriptor, name = original_mkstemp(*args, **kwargs)
    path = Path(name)
    if race == "temporary-symlink" and path.name.startswith(f".{asset}."):
        marker.touch()
        path.unlink()
        path.symlink_to(os.environ["FAKE_ARCHIVE_RACE_SENTINEL"])
    return descriptor, name

def injected_replace(source, destination, *args, **kwargs):
    source_path = Path(source)
    destination_path = Path(destination)
    if (
        race in ("pre-rename-swap", "restore-failure")
        and source_path.name.startswith(f".{asset}.")
        and source_path.name.endswith(".tmp")
        and destination_path.name == asset
    ):
        marker.touch()
        source_path.unlink()
        shutil.copyfile(os.environ["FAKE_ARCHIVE_RACE_REPLACEMENT"], source_path)
    if (
        race == "restore-failure"
        and destination_path.name == asset
        and (
            source_path.name == ".previous-archive"
            or (
                source_path.name.startswith(f".{asset}.recovery-")
                and source_path.name.endswith(".bak")
            )
        )
    ):
        restore_marker.touch()
        raise OSError("injected archive restore failure")
    if (
        race == "checksum-commit-failure"
        and source_path.name.startswith(f".{asset}.sha256.")
        and source_path.name.endswith(".tmp")
        and destination_path.name == f"{asset}.sha256"
    ):
        marker.touch()
        raise OSError("injected checksum commit failure")
    return original_replace(source, destination, *args, **kwargs)

tempfile.mkstemp = injected_mkstemp
os.replace = injected_replace
""",
            encoding="utf-8",
        )
        return {
            "PYTHONPATH": os.pathsep.join((str(hook), str(fixture.root / "src"))),
            "FAKE_ARCHIVE_RACE": race,
            "FAKE_ARCHIVE_RACE_ASSET": (
                "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
            ),
            "FAKE_ARCHIVE_RACE_SENTINEL": str(sentinel),
            "FAKE_ARCHIVE_RACE_REPLACEMENT": str(replacement),
            "FAKE_ARCHIVE_RACE_MARKER": str(marker),
            "FAKE_ARCHIVE_RESTORE_MARKER": str(restore_marker or marker),
        }

    @staticmethod
    def _enable_post_build_dist_swap(fixture: OfflineDeliveryFixture) -> None:
        """Swap the logical dist parent only after build-archive returns."""

        wrapper = fixture.root / ".venv-server" / "bin" / "python"
        text = wrapper.read_text(encoding="utf-8")
        old = 'exec "$FAKE_REAL_PYTHON" "$@"\n'
        replacement = r'''status=0
"$FAKE_REAL_PYTHON" "$@" || status=$?
if [[ "$status" -eq 0 \
    && "${FAKE_POST_BUILD_DIST_SWAP:-}" == "1" \
    && "${1:-}" == "-m" \
    && "${2:-}" == "scammark.result_contract" \
    && "${3:-}" == "build-archive" \
    && ! -e "$FAKE_POST_BUILD_SWAP_MARKER" ]]; then
    safe_dist="$FAKE_REPO_ROOT/dist/release-assets/full-v2"
    legacy_dist="$FAKE_POST_BUILD_LEGACY_ROOT/release-assets/full-v2"
    asset="scammark-qwen3-dashscope-full-v2-inference.tar.gz"
    mkdir -p -- "$legacy_dist"
    cp -- "$safe_dist/$asset" "$legacy_dist/$asset"
    cp -- "$safe_dist/$asset.sha256" "$legacy_dist/$asset.sha256"
    printf '%s  %s\n' \
        "$FAKE_POST_BUILD_SENTINEL_SHA" \
        "$FAKE_POST_BUILD_SENTINEL_NAME" \
        >>"$legacy_dist/$asset.sha256"
    mv -- "$FAKE_REPO_ROOT/dist" "$FAKE_REPO_ROOT/dist.safe-after-build"
    ln -s -- "$FAKE_POST_BUILD_LEGACY_ROOT" "$FAKE_REPO_ROOT/dist"
    : >"$FAKE_POST_BUILD_SWAP_MARKER"
fi
exit "$status"
'''
        if old not in text:
            raise AssertionError("fixture Python wrapper shape changed")
        wrapper.write_text(text.replace(old, replacement), encoding="utf-8")

    def _assert_archive_race_rejected_without_touch(
        self,
        *,
        race: str,
    ) -> None:
        with OfflineDeliveryFixture() as fixture:
            legacy = (
                fixture.root.parent
                / "scammark_qwen3_dashscope_server_full"
            )
            legacy.mkdir()
            sentinel = legacy / "v1-sentinel.tar.gz"
            sentinel.write_bytes(b"untouched v1 archive sentinel\n")
            replacement = fixture.root.parent / "replacement.tar.gz"
            with tarfile.open(replacement, mode="w:gz") as handle:
                payload = b"wrong archive payload\n"
                member = tarfile.TarInfo("wrong-root/wrong.txt")
                member.size = len(payload)
                handle.addfile(member, io.BytesIO(payload))

            archive = (
                fixture.dist
                / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
            )
            checksum = Path(f"{archive}.sha256")
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(b"existing validated archive bytes\n")
            checksum.write_bytes(b"existing validated checksum bytes\n")
            before = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (sentinel, archive, checksum)
            }
            marker = fixture.root.parent / f"{race}-reached"

            rejected = fixture.run(
                "package_results.sh",
                "inference",
                extra_env=self._archive_race_environment(
                    fixture,
                    race=race,
                    sentinel=sentinel,
                    replacement=replacement,
                    marker=marker,
                ),
            )

            self.assertTrue(marker.exists(), rejected.stdout + rejected.stderr)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(
                {
                    path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in (sentinel, archive, checksum)
                },
                before,
            )
            self.assertFalse(fixture.delivery_state.exists())

    def test_package_rejects_archive_temporary_path_symlink_swap_without_touch(self):
        self._assert_archive_race_rejected_without_touch(
            race="temporary-symlink",
        )

    def test_package_rejects_archive_swap_after_validation_before_rename(self):
        self._assert_archive_race_rejected_without_touch(
            race="pre-rename-swap",
        )

    def _verified_archive_rollback_fixture(
        self,
        fixture: OfflineDeliveryFixture,
    ) -> tuple[Path, Path, Path]:
        packaged = fixture.run("package_results.sh", "inference")
        self.assertEqual(
            packaged.returncode,
            0,
            packaged.stdout + packaged.stderr,
        )
        archive = (
            fixture.dist
            / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
        )
        checksum = Path(f"{archive}.sha256")
        archive.chmod(0o644)
        fixture.delivery_state.unlink()
        replacement = fixture.root.parent / "rollback-replacement.tar.gz"
        with tarfile.open(replacement, mode="w:gz") as handle:
            payload = b"wrong postinstall archive payload\n"
            member = tarfile.TarInfo("wrong-root/wrong.txt")
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))
        fixture.clear_logs()
        return archive, checksum, replacement

    def test_package_normal_postinstall_rejection_restores_verified_archive(self):
        with OfflineDeliveryFixture() as fixture:
            archive, checksum, replacement = self._verified_archive_rollback_fixture(
                fixture
            )
            before = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (archive, checksum)
            }
            marker = fixture.root.parent / "normal-restore-postinstall-reached"

            rejected = fixture.run(
                "package_results.sh",
                "inference",
                extra_env=self._archive_race_environment(
                    fixture,
                    race="pre-rename-swap",
                    sentinel=archive,
                    replacement=replacement,
                    marker=marker,
                ),
            )

            self.assertTrue(marker.exists(), rejected.stdout + rejected.stderr)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(
                {
                    path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in (archive, checksum)
                },
                before,
            )
            self.assertEqual(
                list(fixture.dist.glob(f".{archive.name}.recovery-*.bak")),
                [],
            )
            self.assertFalse(fixture.delivery_state.exists())

    def test_package_preserves_stable_old_archive_when_restore_replace_fails(self):
        with OfflineDeliveryFixture() as fixture:
            archive, checksum, replacement = self._verified_archive_rollback_fixture(
                fixture
            )
            old_archive = archive.read_bytes()
            old_checksum = (checksum.read_bytes(), checksum.stat().st_mtime_ns)
            postinstall_marker = (
                fixture.root.parent / "restore-failure-postinstall-reached"
            )
            restore_marker = fixture.root.parent / "restore-replace-reached"

            rejected = fixture.run(
                "package_results.sh",
                "inference",
                extra_env=self._archive_race_environment(
                    fixture,
                    race="restore-failure",
                    sentinel=archive,
                    replacement=replacement,
                    marker=postinstall_marker,
                    restore_marker=restore_marker,
                ),
            )

            self.assertTrue(
                postinstall_marker.exists(),
                rejected.stdout + rejected.stderr,
            )
            self.assertTrue(restore_marker.exists(), rejected.stdout + rejected.stderr)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(archive.exists())
            backups = list(
                fixture.dist.glob(f".{archive.name}.recovery-*.bak")
            )
            self.assertEqual(len(backups), 1)
            backup = backups[0]
            metadata = backup.lstat()
            self.assertTrue(stat.S_ISREG(metadata.st_mode))
            self.assertEqual(metadata.st_nlink, 1)
            self.assertEqual(stat.S_IMODE(metadata.st_mode) & 0o077, 0)
            self.assertEqual(backup.read_bytes(), old_archive)
            matching_old_bytes = [
                path
                for path in fixture.dist.iterdir()
                if path.is_file() and path.read_bytes() == old_archive
            ]
            self.assertEqual(matching_old_bytes, [backup])
            self.assertEqual(
                (checksum.read_bytes(), checksum.stat().st_mtime_ns),
                old_checksum,
            )
            self.assertFalse(fixture.delivery_state.exists())
            self.assertFalse(
                any(
                    path.name.startswith(".result-stage-")
                    for path in fixture.dist.iterdir()
                )
            )
            expected_action = (
                "result archive recovery required: verify and rename "
                f"{backup.name} to {archive.name}, then verify or regenerate "
                f"{checksum.name}"
            )
            self.assertIn(expected_action, rejected.stdout + rejected.stderr)

    def test_package_reports_stable_backup_when_checksum_commit_fails(self):
        with OfflineDeliveryFixture() as fixture:
            archive, checksum, replacement = self._verified_archive_rollback_fixture(
                fixture
            )
            old_archive = archive.read_bytes()
            old_checksum = (checksum.read_bytes(), checksum.stat().st_mtime_ns)
            marker = fixture.root.parent / "checksum-commit-reached"

            rejected = fixture.run(
                "package_results.sh",
                "inference",
                extra_env=self._archive_race_environment(
                    fixture,
                    race="checksum-commit-failure",
                    sentinel=archive,
                    replacement=replacement,
                    marker=marker,
                ),
            )

            self.assertTrue(marker.exists(), rejected.stdout + rejected.stderr)
            self.assertNotEqual(rejected.returncode, 0)
            backups = list(
                fixture.dist.glob(f".{archive.name}.recovery-*.bak")
            )
            self.assertEqual(len(backups), 1)
            backup = backups[0]
            metadata = backup.lstat()
            self.assertTrue(stat.S_ISREG(metadata.st_mode))
            self.assertEqual(metadata.st_nlink, 1)
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(backup.read_bytes(), old_archive)
            self.assertTrue(archive.is_file())
            self.assertEqual(
                (checksum.read_bytes(), checksum.stat().st_mtime_ns),
                old_checksum,
            )
            self.assertFalse(fixture.delivery_state.exists())
            expected_action = (
                "result archive recovery required: verify and rename "
                f"{backup.name} to {archive.name}, then verify or regenerate "
                f"{checksum.name}"
            )
            self.assertIn(expected_action, rejected.stdout + rejected.stderr)

    def test_package_rejects_dist_link_into_v1_before_target_touch(self):
        with OfflineDeliveryFixture() as fixture:
            legacy = (
                fixture.root.parent
                / "scammark_qwen3_dashscope_server_full"
            )
            legacy.mkdir()
            (legacy / "sentinel.bin").write_bytes(b"untouched v1 dist sentinel\n")
            (legacy / "sentinel-link").symlink_to("sentinel.bin")
            (legacy / "existing-directory").mkdir()
            before = self._exact_tree_inventory(legacy)
            (fixture.root / "dist").symlink_to(
                legacy,
                target_is_directory=True,
            )

            rejected = fixture.run("package_results.sh", "inference")

            self.assertNotEqual(rejected.returncode, 0)
            after = self._exact_tree_inventory(legacy)
            if after != before:
                self.fail("package touched the physical v1 dist target")
            self.assertFalse(
                (
                    legacy
                    / "release-assets"
                    / "full-v2"
                    / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
                ).exists()
            )
            self.assertFalse(fixture.delivery_state.exists())

    def test_package_accepts_the_normal_safe_dist_target(self):
        with OfflineDeliveryFixture() as fixture:
            completed = fixture.run("package_results.sh", "inference")

            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertTrue(fixture.state("inference").packaged)

    def test_package_rejects_dist_binding_change_after_archive_build(self):
        with OfflineDeliveryFixture() as fixture:
            legacy_root = (
                fixture.root.parent
                / "scammark_qwen3_dashscope_server_full"
            )
            legacy_dist = legacy_root / "release-assets" / "full-v2"
            legacy_dist.mkdir(parents=True)
            sentinel = legacy_dist / "legacy-sentinel.bin"
            sentinel.write_bytes(b"legacy sentinel must not be read\n")
            sentinel_sha = hashlib.sha256(sentinel.read_bytes()).hexdigest()
            os.utime(
                sentinel,
                ns=(1_000_000_000, 2_000_000_000),
            )
            before = (
                sentinel.read_bytes(),
                sentinel.lstat().st_mode,
                sentinel.lstat().st_nlink,
                sentinel.lstat().st_mtime_ns,
            )
            os.utime(
                sentinel,
                ns=(1_000_000_000, before[3]),
            )
            before_atime = sentinel.stat().st_atime_ns
            marker = fixture.root.parent / "post-build-dist-swap-reached"
            self._enable_post_build_dist_swap(fixture)

            rejected = fixture.run(
                "package_results.sh",
                "inference",
                extra_env={
                    "FAKE_POST_BUILD_DIST_SWAP": "1",
                    "FAKE_POST_BUILD_LEGACY_ROOT": str(legacy_root),
                    "FAKE_POST_BUILD_SENTINEL_NAME": sentinel.name,
                    "FAKE_POST_BUILD_SENTINEL_SHA": sentinel_sha,
                    "FAKE_POST_BUILD_SWAP_MARKER": str(marker),
                },
            )

            self.assertTrue(marker.exists(), rejected.stdout + rejected.stderr)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(fixture.delivery_state.exists())
            after_metadata = sentinel.lstat()
            self.assertEqual(after_metadata.st_atime_ns, before_atime)
            self.assertEqual(
                (
                    sentinel.read_bytes(),
                    after_metadata.st_mode,
                    after_metadata.st_nlink,
                    after_metadata.st_mtime_ns,
                ),
                before,
            )

    def test_v2_server_scripts_only_retain_the_v1_frozen_asset(self):
        forbidden = (
            "outputs/scammark_qwen3_dashscope_server_full/",
            "scammark-qwen3-dashscope-full-v1-inference",
            "scammark-qwen3-dashscope-full-v1-final-evaluation",
        )
        for script in sorted(SERVER_SCRIPTS.glob("*.sh")):
            text = script.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(script=script.name, value=value):
                    self.assertNotIn(value, text)

        common = (SERVER_SCRIPTS / "common.sh").read_text(encoding="utf-8")
        self.assertIn("scamweb-v1.0-frozen-inputs.tar.gz", common)

    def test_v2_namespace_isolated_and_method_is_deep_equal(self):
        with (ROOT / "configs/qwen3_dashscope_server_full.toml").open("rb") as handle:
            v1 = tomllib.load(handle)
        v2_path = ROOT / "configs/qwen3_dashscope_server_full_v2.toml"
        self.assertTrue(v2_path.is_file())
        with v2_path.open("rb") as handle:
            v2 = tomllib.load(handle)

        expected = deepcopy(v1)
        expected["experiment"]["name"] = "scammark-qwen3-dashscope-server-full-v2"
        expected["paths"]["output_root"] = "outputs/scammark_qwen3_dashscope_server_full_v2"
        expected["paths"]["image_cache"] = "outputs/scammark_qwen3_dashscope_server_full_v2/image_cache"
        expected["observability"] = {
            "event_schema": "scammark-run-event-v1",
            "timezone": "Asia/Shanghai",
            "heartbeat_poll_seconds": 30,
            "heartbeat_stall_seconds": 60,
            "summary_sample_interval": 25,
            "summary_seconds": 300,
            "eta_warmup_samples": 5,
            "eta_history_size": 50,
            "cost_full_update_samples": 25,
            "breaker_consecutive_backend_errors": 5,
        }
        self.assertEqual(v2, expected)

        common = (SERVER_SCRIPTS / "common.sh").read_text(encoding="utf-8")
        for value in (
            "experiment-qwen3-dashscope-full-v2",
            "results/qwen3-dashscope-full-v2",
            "scammark-qwen3-dashscope-full-v2-inference.tar.gz",
            "scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz",
            "qwen3_dashscope_server_full_v2.toml",
            "scammark_qwen3_dashscope_server_full_v2",
            "scammark-full-v2",
            "dist/release-assets/full-v2",
            "dist/frozen-download/full-v2",
        ):
            self.assertIn(value, common)

    def test_required_server_scripts_exist(self):
        expected = {
            "common.sh",
            "prepare_server.sh",
            "start_full.sh",
            "run_full_inner.sh",
            "pause_full.sh",
            "recover_delivery.sh",
            "status.sh",
            "package_results.sh",
            "upload_release_assets.sh",
            "finalize_bertscore.sh",
            "publish_results_branch.sh",
        }

        self.assertEqual(
            {path.name for path in SERVER_SCRIPTS.glob("*.sh")},
            expected,
        )

    def test_prepare_script_verifies_release_asset_and_zero_api_preflight(self):
        text = (SERVER_SCRIPTS / "prepare_server.sh").read_text(encoding="utf-8")

        self.assertIn("gh release download", text)
        self.assertIn("sha256sum -c", text)
        self.assertIn("requirements-server.lock.txt", text)
        self.assertIn("scammark.frozen_assets", text)
        self.assertIn("--validate-only", text)
        self.assertIn("SCAMWEB_DATASET_ROOT", text)
        self.assertNotIn("source \"$ENV_FILE\"", text)

    def test_prepare_script_enforces_exact_frozen_and_preflight_contract(self):
        text = (SERVER_SCRIPTS / "prepare_server.sh").read_text(encoding="utf-8")

        for value in (
            "21802246",
            "8d0894a043913f6e049cfc4b212a310cb4fd79a810ff5651372c802f8ab2eb61",
            "81904995efcae0bacdee1a2e31d0da9d92a81adb099c4a9d0b5b01da885e0d1c",
            "012ec051c0518900be2246ed5ef0a07185768bf8f451e378056f75289fa160ff",
            'payload.get("run_mode") != "full"',
            'payload.get("selected_sample_count") != 5658',
            'payload.get("expected_full_sample_count") != 5658',
            'payload.get("contract_image_count") != 5658',
            'payload.get("api_call_limit") != 50000',
            '"observability.py"',
            '"run_status.py"',
            '"result_contract.py"',
            "object_pairs_hook",
            "parse_constant",
            "tarfile.open",
            '"dataset_frozen_v1/boxes.jsonl"',
            '"dataset_frozen_v1/dataset_version.json"',
            '"dataset_frozen_v1/mhtml_audit.jsonl"',
            '"dataset_frozen_v1/mhtml_recapture.jsonl"',
            '"dataset_frozen_v1/sample_ids.txt"',
            '"dataset_frozen_v1/scamweb_v_manifest.jsonl"',
            '"dataset_frozen_v1/verification_report.json"',
            '"dataset_frozen_v1/web_evidence.jsonl"',
            "Frozen archive member allowlist mismatch",
        ):
            with self.subTest(required=value):
                self.assertIn(value, text)

        for variable in (
            "SCAMMARK_LLM_API_KEY",
            "SCAMMARK_LLM_BASE_URL",
            "SCAMMARK_VLM_API_KEY",
            "SCAMMARK_VLM_BASE_URL",
        ):
            with self.subTest(provider_variable=variable):
                self.assertRegex(text, rf"(?:env\s+-u\s+|unset\s+){variable}\b")

        self.assertLess(text.index("require_clean_worktree"), text.index("gh release download"))
        self.assertLess(text.index("require_exact_release_tag"), text.index("gh release download"))
        self.assertIn(
            'git -C "$ROOT_DIR" ls-remote --tags origin "refs/tags/$RELEASE_TAG^{}"',
            text,
        )
        self.assertIn('git -C "$ROOT_DIR" rev-parse HEAD', text)
        self.assertLess(
            text.index('git -C "$ROOT_DIR" ls-remote --tags origin'),
            text.index('require_file "$ENV_FILE"'),
        )
        self.assertIn("Fake-only test suite passed", text)
        self.assertIn("Paid smoke command (not executed)", text)

    def test_prepare_script_fake_run_scrubs_providers_and_installs_only_valid_frozen(self):
        with OfflinePrepareFixture() as fixture:
            completed = fixture.run()

            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertFalse(fixture.provider_leak.exists())
            self.assertEqual(
                fixture.log(fixture.python_log).splitlines(),
                [
                    "pip",
                    "frozen-assets",
                    "unittest:providers-cleared",
                    "validate-only:providers-cleared",
                ],
            )
            self.assertTrue(
                (fixture.frozen / "scamweb_v_manifest.jsonl").is_file()
            )
            self.assertTrue((fixture.frozen / "web_evidence.jsonl").is_file())
            self.assertFalse((fixture.frozen / "old-sentinel.txt").exists())
            self.assertIn(
                "Paid smoke command (not executed)",
                completed.stdout,
            )
            self.assertEqual(
                fixture.log(fixture.command_log).count("gh:release download"),
                1,
            )

    def test_prepare_script_rejects_internal_hash_before_replacing_existing_frozen(self):
        with OfflinePrepareFixture() as fixture:
            before = self._exact_tree_inventory(fixture.frozen)

            completed = fixture.run(extra_env={"FAKE_BAD_INTERNAL_HASH": "1"})

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("manifest SHA256", completed.stdout + completed.stderr)
            self.assertEqual(self._exact_tree_inventory(fixture.frozen), before)
            self.assertNotIn("unittest", fixture.log(fixture.python_log))
            self.assertNotIn("validate-only", fixture.log(fixture.python_log))

    def test_prepare_script_rejects_dirty_or_wrong_tag_before_download(self):
        variants = (
            ({"FAKE_GIT_DIRTY": "1"}, "dirty"),
            ({"FAKE_GIT_TAG": "wrong-tag"}, "tag"),
        )
        for environment, label in variants:
            with self.subTest(label=label), OfflinePrepareFixture() as fixture:
                before = self._exact_tree_inventory(fixture.frozen)

                completed = fixture.run(extra_env=environment)

                self.assertNotEqual(completed.returncode, 0)
                self.assertNotIn("gh:release download", fixture.log(fixture.command_log))
                self.assertEqual(self._exact_tree_inventory(fixture.frozen), before)

    def test_prepare_script_rejects_remote_peeled_tag_mismatch_before_local_mutation(self):
        with OfflinePrepareFixture() as fixture:
            frozen_before = self._exact_tree_inventory(fixture.frozen)
            env_path = fixture.root / ".env"
            env_before = (
                env_path.read_bytes(),
                env_path.stat().st_mode,
                env_path.stat().st_mtime_ns,
            )

            completed = fixture.run(
                extra_env={
                    "FAKE_REMOTE_TAG_COMMIT": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                }
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("remote peeled tag", completed.stdout + completed.stderr)
            self.assertNotIn("gh:release download", fixture.log(fixture.command_log))
            self.assertEqual(self._exact_tree_inventory(fixture.frozen), frozen_before)
            self.assertEqual(
                (
                    env_path.read_bytes(),
                    env_path.stat().st_mode,
                    env_path.stat().st_mtime_ns,
                ),
                env_before,
            )

    def test_prepare_script_rejects_duplicate_preflight_keys(self):
        with OfflinePrepareFixture() as fixture:
            completed = fixture.run(
                extra_env={"FAKE_DUPLICATE_PREFLIGHT_KEY": "1"}
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate key", completed.stdout + completed.stderr)
            self.assertNotIn("Paid smoke command", completed.stdout)

    def test_prepare_rejects_v2_runtime_symlink_before_v1_mutation(self):
        with OfflinePrepareFixture() as fixture:
            v2_output = (
                fixture.root
                / "outputs"
                / "scammark_qwen3_dashscope_server_full_v2"
            )
            v2_output.symlink_to(fixture.v1_output, target_is_directory=True)
            before = self._exact_tree_inventory(fixture.v1_output)

            completed = fixture.run()

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "Unsafe v2 runtime namespace",
                completed.stdout + completed.stderr,
            )
            self.assertEqual(
                self._exact_tree_inventory(fixture.v1_output),
                before,
            )
            self.assertNotIn("gh:", fixture.log(fixture.command_log))

    def test_prepare_rejects_download_parent_symlink_before_v1_mutation(self):
        with OfflinePrepareFixture() as fixture:
            dist = fixture.root / "dist"
            dist.symlink_to(fixture.v1_output, target_is_directory=True)
            before = self._exact_tree_inventory(fixture.v1_output)

            completed = fixture.run()

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "Unsafe v2 download namespace",
                completed.stdout + completed.stderr,
            )
            self.assertEqual(
                self._exact_tree_inventory(fixture.v1_output),
                before,
            )
            self.assertNotIn("gh:", fixture.log(fixture.command_log))

    def test_windows_v2_downloader_has_exact_offline_strict_contract(self):
        path = ROOT / "scripts" / "windows" / "download_full_results_v2.ps1"
        text = path.read_text(encoding="utf-8-sig")

        expected_parameters = (
            "Repository",
            "Tag",
            "Destination",
            "ArchiveDirectory",
            "VerifyOnly",
            "SkipResultsBranchFetch",
        )
        param_match = re.search(r"(?s)^param\((.*?)\)\s*\$ErrorActionPreference", text)
        self.assertIsNotNone(param_match)
        self.assertEqual(
            re.findall(r"\$(\w+)\b", param_match.group(1)),
            list(expected_parameters),
        )
        for value in (
            '"nlixsys/ScamWeb-V"',
            '"experiment-qwen3-dashscope-full-v2"',
            '"outputs/server_results/qwen3-dashscope-full-v2"',
            '"results/qwen3-dashscope-full-v2"',
            '"scammark-qwen3-dashscope-full-v2-inference.tar.gz"',
            '"scammark-qwen3-dashscope-full-v2-inference.tar.gz.sha256"',
            '"scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz"',
            '"scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz.sha256"',
            "Get-FileHash",
            "run_events.jsonl",
            "full_run.log",
            "prediction_sample_count",
            "acceptance_gates",
            "bertscore_status",
            "provenance.json",
            "LASTEXITCODE",
            "System.Text.Json.JsonDocument",
            "duplicate JSON property",
            "post-extraction archive SHA256",
            "private-assets",
            "[System.IO.File]::Copy",
            "full_sample_count_exact",
            "[System.StringComparison]::Ordinal",
            "OrderedIds",
            "Manifest and prediction sample ID order differs",
            "result installation recovery required",
            "[System.IO.Path]::GetFileName($Backup)",
            "$AllowedDestinationRoot",
            "outputs/server_results",
            "Assert-SafeDestinationPath",
            "GetRelativePath",
            "GetPathRoot",
            "Destination must be a non-root child",
            "Destination ancestor is a reparse point",
            "ArchiveDirectory",
            "VerifyOnly",
            "SkipResultsBranchFetch",
        ):
            with self.subTest(required=value):
                self.assertIn(value, text)

        for relative in (
            "full/predictions.jsonl",
            "full/metrics.json",
            "full/manifest.jsonl",
            "full/preflight.json",
            "full/full_summary.md",
            "full/human_evaluation_candidates.csv",
            "audit/run_config.sanitized.json",
            "audit/api_call_ledger.sanitized.jsonl",
            "logs/run_events.jsonl",
            "logs/full_run.log",
            "config/qwen3_dashscope_server_full_v2.toml",
            "frozen/dataset_version.json",
            "frozen/verification_report.json",
            "full/metrics_final.json",
            "full/final_evaluation_summary.md",
            "environment/bertscore_environment.txt",
            "environment/bertscore_asset_status.json",
            "logs/bertscore_finalize.log",
        ):
            with self.subTest(allowlisted_member=relative):
                self.assertIn(relative, text)

        lines = text.splitlines()
        native_calls = [
            index
            for index, line in enumerate(lines)
            if re.search(r"&\s+(?:gh|tar|git)\b", line)
        ]
        self.assertGreaterEqual(len(native_calls), 6)
        for index in native_calls:
            next_nonempty = next(
                line.strip() for line in lines[index + 1 :] if line.strip()
            )
            with self.subTest(native_call=lines[index].strip()):
                self.assertEqual(next_nonempty, "if ($LASTEXITCODE -ne 0) {")

        self.assertNotIn('Get-ChildItem -LiteralPath $DownloadDir -Filter "*.tar.gz"', text)
        self.assertGreaterEqual(text.count("Assert-ExactChecksum"), 5)
        destination_block = text.index(
            "$DestinationPath = Get-AbsolutePath -Value $Destination"
        )
        destination_policy_call = text.index(
            "Assert-SafeDestinationPath -DestinationPath $DestinationPath",
            destination_block,
        )
        self.assertGreater(destination_policy_call, 0)
        self.assertGreaterEqual(
            text.count(
                "Assert-SafeDestinationPath -DestinationPath $DestinationPath"
            ),
            4,
        )
        self.assertLess(
            destination_policy_call,
            text.index("if ($VerifyOnly)", destination_block),
        )
        self.assertLess(
            destination_policy_call,
            text.index('New-Item -ItemType Directory -Force -Path $DestinationParent'),
        )
        self.assertLess(
            text.index("[System.IO.File]::Copy"),
            text.index("$InferenceArchive = Join-Path $AssetDirectory"),
        )
        install_function = re.search(
            r"(?s)function Install-StagedDestination \{(.*?)\n\}",
            text,
        )
        self.assertIsNotNone(install_function)
        self.assertIn(
            "Assert-SafeDestinationPath -DestinationPath $DestinationPath",
            install_function.group(1),
        )
        self.assertLess(
            install_function.group(1).index(
                "Assert-SafeDestinationPath -DestinationPath $DestinationPath"
            ),
            install_function.group(1).index("$Parent ="),
        )
        self.assertIn(".rollback-", install_function.group(1))
        self.assertIn("previous destination restored", install_function.group(1))
        self.assertNotIn(
            "$Installed -and",
            install_function.group(1),
            "cleanup failure after a successful install must not delete both copies",
        )
        self.assertIn(
            '$null = New-Item -ItemType Directory -Force -Path $DestinationParent\n'
            "    Assert-SafeDestinationPath -DestinationPath $DestinationPath",
            text,
        )
        self.assertIn(
            "if (-not $VerifyOnly) {\n"
            "        Assert-SafeDestinationPath -DestinationPath $DestinationPath\n"
            "        Install-StagedDestination",
            text,
        )

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is unavailable")
    def test_windows_v2_downloader_rejects_destination_outside_controlled_root(self):
        from tests.result_fixture import CompleteResultFixture

        with CompleteResultFixture(sample_count=5658) as fixture:
            archive_directory = fixture.build_archives()
            client_root, downloader = self._copy_windows_downloader(fixture.root)
            allowed_root = client_root / "outputs" / "server_results"
            outside_destination = fixture.root / "outside-destination"

            completed = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(downloader),
                    "-ArchiveDirectory",
                    str(archive_directory),
                    "-Destination",
                    str(outside_destination),
                    "-VerifyOnly",
                    "-SkipResultsBranchFetch",
                ],
                cwd=client_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "Destination must be a non-root child",
                completed.stdout + completed.stderr,
            )
            self.assertFalse(outside_destination.exists())
            self.assertFalse(allowed_root.exists())

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is unavailable")
    def test_windows_v2_downloader_rejects_reparse_destination_ancestor(self):
        from tests.result_fixture import CompleteResultFixture

        with CompleteResultFixture(sample_count=5658) as fixture:
            archive_directory = fixture.build_archives()
            client_root, downloader = self._copy_windows_downloader(fixture.root)
            outputs = client_root / "outputs"
            outputs.mkdir()
            external = fixture.root / "external-results"
            external.mkdir()
            (external / "sentinel.bin").write_bytes(b"outside-must-not-change\x00")
            allowed_root = outputs / "server_results"
            try:
                allowed_root.symlink_to(external, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            before = self._exact_tree_inventory(external)

            completed = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(downloader),
                    "-ArchiveDirectory",
                    str(archive_directory),
                    "-Destination",
                    str(allowed_root / "verified-results"),
                    "-VerifyOnly",
                    "-SkipResultsBranchFetch",
                ],
                cwd=client_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "Destination ancestor is a reparse point",
                completed.stdout + completed.stderr,
            )
            self.assertEqual(self._exact_tree_inventory(external), before)

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is unavailable")
    def test_windows_v2_downloader_rejects_nondirectory_destination_ancestor(self):
        from tests.result_fixture import CompleteResultFixture

        with CompleteResultFixture(sample_count=5658) as fixture:
            archive_directory = fixture.build_archives()
            client_root, downloader = self._copy_windows_downloader(fixture.root)
            outputs = client_root / "outputs"
            outputs.mkdir()
            blocked_root = outputs / "server_results"
            blocked_root.write_bytes(b"not-a-directory\x00")
            before = self._exact_tree_inventory(outputs)

            completed = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(downloader),
                    "-ArchiveDirectory",
                    str(archive_directory),
                    "-Destination",
                    str(blocked_root / "verified-results"),
                    "-VerifyOnly",
                    "-SkipResultsBranchFetch",
                ],
                cwd=client_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "Destination ancestor is not a directory",
                completed.stdout + completed.stderr,
            )
            self.assertEqual(self._exact_tree_inventory(outputs), before)

    @unittest.skipUnless(
        shutil.which("pwsh") and os.name != "nt",
        "PowerShell 7 on POSIX is unavailable",
    )
    def test_windows_v2_downloader_rechecks_destination_after_results_fetch(self):
        from tests.result_fixture import CompleteResultFixture

        with CompleteResultFixture(sample_count=5658) as fixture:
            archive_directory = fixture.build_archives()
            client_root, downloader = self._copy_windows_downloader(fixture.root)
            allowed_root = client_root / "outputs" / "server_results"
            displaced_root = fixture.root / "displaced-server-results"
            external = fixture.root / "external-results"
            external.mkdir()
            (external / "sentinel.bin").write_bytes(b"outside-must-not-change\x00")
            before = self._exact_tree_inventory(external)
            fake_bin = fixture.root / "fake-git-bin"
            fake_bin.mkdir()
            git_wrapper = fake_bin / "git"
            git_wrapper.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
mv -- "$FAKE_ALLOWED_DESTINATION_ROOT" "$FAKE_DISPLACED_DESTINATION_ROOT"
ln -s -- "$FAKE_EXTERNAL_DESTINATION_ROOT" "$FAKE_ALLOWED_DESTINATION_ROOT"
""",
                encoding="utf-8",
            )
            git_wrapper.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_ALLOWED_DESTINATION_ROOT": str(allowed_root),
                    "FAKE_DISPLACED_DESTINATION_ROOT": str(displaced_root),
                    "FAKE_EXTERNAL_DESTINATION_ROOT": str(external),
                }
            )

            completed = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(downloader),
                    "-ArchiveDirectory",
                    str(archive_directory),
                    "-Destination",
                    str(allowed_root / "installed-results"),
                ],
                cwd=client_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "Destination ancestor is a reparse point",
                completed.stdout + completed.stderr,
            )
            self.assertEqual(self._exact_tree_inventory(external), before)

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is unavailable")
    def test_windows_v2_downloader_verify_only_accepts_fixture_without_creating_destination_root(self):
        from tests.result_fixture import CompleteResultFixture

        with CompleteResultFixture(sample_count=5658) as fixture:
            archive_directory = fixture.build_archives()
            client_root, downloader = self._copy_windows_downloader(fixture.root)
            allowed_root = client_root / "outputs" / "server_results"
            destination = allowed_root / "qwen3-dashscope-full-v2"
            self.assertFalse(allowed_root.exists())

            completed = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(downloader),
                    "-ArchiveDirectory",
                    str(archive_directory),
                    "-VerifyOnly",
                    "-SkipResultsBranchFetch",
                ],
                cwd=client_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertFalse(allowed_root.exists())
            self.assertFalse(destination.exists())

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is unavailable")
    def test_windows_v2_downloader_installs_valid_fixture_from_same_volume_stage(self):
        from tests.result_fixture import CompleteResultFixture

        with CompleteResultFixture(sample_count=5658) as fixture:
            archive_directory = fixture.build_archives()
            client_root, downloader = self._copy_windows_downloader(fixture.root)
            destination = (
                client_root
                / "outputs"
                / "server_results"
                / "installed-results"
            )
            destination.mkdir(parents=True)
            (destination / "old-sentinel.txt").write_text(
                "replace me\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(downloader),
                    "-ArchiveDirectory",
                    str(archive_directory),
                    "-Destination",
                    str(destination),
                    "-SkipResultsBranchFetch",
                ],
                cwd=client_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertFalse((destination / "old-sentinel.txt").exists())
            self.assertTrue(
                (
                    destination
                    / "scammark-qwen3-dashscope-full-v2-inference"
                    / "full"
                    / "predictions.jsonl"
                ).is_file()
            )
            self.assertTrue(
                (
                    destination
                    / "scammark-qwen3-dashscope-full-v2-final-evaluation"
                    / "full"
                    / "metrics_final.json"
                ).is_file()
            )

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is unavailable")
    def test_windows_v2_downloader_rejects_each_strict_contract_tamper_without_destination_change(self):
        from tests.result_fixture import (
            CompleteResultFixture,
            FINAL_ASSET_NAME,
            INFERENCE_ASSET_NAME,
        )

        def checksum_tamper(directory: Path) -> None:
            path = directory / INFERENCE_ASSET_NAME
            path.write_bytes(path.read_bytes() + b"checksum-tamper")

        def traversal_tamper(directory: Path) -> None:
            path = directory / INFERENCE_ASSET_NAME

            def mutate(entries: list[list[object]]) -> None:
                member = tarfile.TarInfo(
                    "scammark-qwen3-dashscope-full-v2-inference/../escape.txt"
                )
                member.type = tarfile.REGTYPE
                entries.append([member, b"escape\n"])

            self._rewrite_result_archive(path, mutate)

        def link_type_tamper(directory: Path) -> None:
            path = directory / INFERENCE_ASSET_NAME

            def mutate(entries: list[list[object]]) -> None:
                for member, _payload in entries:
                    if member.name.endswith("/provenance.json"):
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../../outside"
                        member.size = 0
                        return
                raise AssertionError("provenance member not found")

            self._rewrite_result_archive(path, mutate)

        def content_tamper(directory: Path) -> None:
            path = directory / INFERENCE_ASSET_NAME

            def mutate(entries: list[list[object]]) -> None:
                for entry in entries:
                    member, payload = entry
                    if member.name.endswith("/full/predictions.jsonl"):
                        entry[1] = payload.replace(
                            b'"sample-0"', b'"sample-1"', 1
                        )
                        return
                raise AssertionError("predictions member not found")

            self._rewrite_result_archive(path, mutate)

        def duplicate_json_tamper(directory: Path) -> None:
            path = directory / INFERENCE_ASSET_NAME

            def mutate(entries: list[list[object]]) -> None:
                for entry in entries:
                    member, payload = entry
                    if member.name.endswith("/provenance.json"):
                        entry[1] = payload.replace(
                            b"{",
                            b'{"release_tag":"experiment-qwen3-dashscope-full-v2",',
                            1,
                        )
                        return
                raise AssertionError("provenance member not found")

            self._rewrite_result_archive(path, mutate)

        def whitespace_sample_id_tamper(directory: Path) -> None:
            path = directory / INFERENCE_ASSET_NAME

            def mutate(entries: list[list[object]]) -> None:
                changed = 0
                for entry in entries:
                    member, payload = entry
                    if member.name.endswith(("/full/manifest.jsonl", "/full/predictions.jsonl")):
                        entry[1] = payload.replace(
                            b'"sample-0"', b'" sample-0 "', 1
                        )
                        changed += 1
                if changed != 2:
                    raise AssertionError("manifest/predictions members not found")

            self._rewrite_result_archive(path, mutate)

        def sample_id_order_tamper(directory: Path) -> None:
            path = directory / INFERENCE_ASSET_NAME

            def mutate(entries: list[list[object]]) -> None:
                for entry in entries:
                    member, payload = entry
                    if member.name.endswith("/full/predictions.jsonl"):
                        rows = payload.splitlines(keepends=True)
                        if len(rows) < 2:
                            raise AssertionError("prediction rows are missing")
                        rows[0], rows[1] = rows[1], rows[0]
                        entry[1] = b"".join(rows)
                        return
                raise AssertionError("predictions member not found")

            self._rewrite_result_archive(path, mutate)

        def required_gate_name_tamper(directory: Path) -> None:
            for asset_name in (INFERENCE_ASSET_NAME, FINAL_ASSET_NAME):
                path = directory / asset_name

                def mutate(entries: list[list[object]]) -> None:
                    for entry in entries:
                        member, payload = entry
                        if member.name.endswith("/full/metrics.json"):
                            entry[1] = payload.replace(
                                b'"full_sample_count_exact"',
                                b'"renamed_count_gate"',
                                1,
                            )
                            return
                    raise AssertionError("metrics member not found")

                self._rewrite_result_archive(path, mutate)

        def provenance_tamper(directory: Path) -> None:
            path = directory / INFERENCE_ASSET_NAME

            def mutate(entries: list[list[object]]) -> None:
                for entry in entries:
                    member, payload = entry
                    if member.name.endswith("/provenance.json"):
                        provenance = json.loads(payload)
                        provenance["release_tag"] = "wrong-tag"
                        entry[1] = json.dumps(provenance).encode("utf-8") + b"\n"
                        return
                raise AssertionError("provenance member not found")

            self._rewrite_result_archive(path, mutate)

        def cross_package_tamper(directory: Path) -> None:
            path = directory / FINAL_ASSET_NAME

            def mutate(entries: list[list[object]]) -> None:
                for entry in entries:
                    member, payload = entry
                    if member.name.endswith("/full/metrics.json"):
                        entry[1] = payload.rstrip() + b" \n"
                        return
                raise AssertionError("final metrics member not found")

            self._rewrite_result_archive(path, mutate)

        variants = (
            ("checksum", checksum_tamper),
            ("traversal-member", traversal_tamper),
            ("link-type", link_type_tamper),
            ("content", content_tamper),
            ("duplicate-json-property", duplicate_json_tamper),
            ("sample-id-whitespace", whitespace_sample_id_tamper),
            ("sample-id-order", sample_id_order_tamper),
            ("required-gate-name", required_gate_name_tamper),
            ("provenance", provenance_tamper),
            ("cross-package", cross_package_tamper),
        )
        for label, tamper in variants:
            with self.subTest(tamper=label), CompleteResultFixture(
                sample_count=5658
            ) as fixture:
                archive_directory = fixture.build_archives()
                tamper(archive_directory)
                client_root, downloader = self._copy_windows_downloader(
                    fixture.root
                )
                destination = (
                    client_root
                    / "outputs"
                    / "server_results"
                    / "protected-destination"
                )
                destination.mkdir(parents=True)
                (destination / "sentinel.bin").write_bytes(
                    b"must remain unchanged\x00"
                )
                before = self._exact_tree_inventory(destination)

                completed = subprocess.run(
                    [
                        "pwsh",
                        "-NoProfile",
                        "-File",
                        str(downloader),
                        "-ArchiveDirectory",
                        str(archive_directory),
                        "-Destination",
                        str(destination),
                        "-VerifyOnly",
                        "-SkipResultsBranchFetch",
                    ],
                    cwd=client_root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )

                self.assertNotEqual(
                    completed.returncode,
                    0,
                    completed.stdout + completed.stderr,
                )
                self.assertEqual(self._exact_tree_inventory(destination), before)

    @unittest.skipUnless(
        shutil.which("pwsh") and os.name != "nt",
        "PowerShell 7 on POSIX is unavailable",
    )
    def test_windows_v2_downloader_rehashes_archives_after_extraction(self):
        from tests.result_fixture import CompleteResultFixture

        with CompleteResultFixture(sample_count=5658) as fixture:
            archive_directory = fixture.build_archives()
            client_root, downloader = self._copy_windows_downloader(fixture.root)
            destination = (
                client_root
                / "outputs"
                / "server_results"
                / "protected-destination"
            )
            destination.mkdir(parents=True)
            (destination / "sentinel.bin").write_bytes(b"unchanged\x00")
            before = self._exact_tree_inventory(destination)
            fake_bin = fixture.root / "fake-bin"
            fake_bin.mkdir()
            tar_wrapper = fake_bin / "tar"
            tar_wrapper.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
/usr/bin/tar "$@"
status=$?
if [[ "$status" -eq 0 && "${1:-}" == "-xzf" ]]; then
    printf 'post-extraction-tamper' >>"$2"
fi
exit "$status"
""",
                encoding="utf-8",
            )
            tar_wrapper.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

            completed = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(downloader),
                    "-ArchiveDirectory",
                    str(archive_directory),
                    "-Destination",
                    str(destination),
                    "-VerifyOnly",
                    "-SkipResultsBranchFetch",
                ],
                cwd=client_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "post-extraction archive SHA256",
                completed.stdout + completed.stderr,
            )
            self.assertEqual(self._exact_tree_inventory(destination), before)

    def test_full_runner_is_resumable_and_uploads_inference_results(self):
        start = (SERVER_SCRIPTS / "start_full.sh").read_text(encoding="utf-8")
        runner = (SERVER_SCRIPTS / "run_full_inner.sh").read_text(encoding="utf-8")

        self.assertIn("tmux new-session -d", start)
        self.assertIn("--full", runner)
        self.assertIn("--resume", runner)
        self.assertIn('recover_delivery.sh" package', runner)

    def test_finalizer_uses_pinned_arm64_container_and_posthoc_metrics(self):
        text = (SERVER_SCRIPTS / "finalize_bertscore.sh").read_text(encoding="utf-8")

        self.assertIn(
            "sha256:d1a026a2b28de6ef6404334ca78b7b761434f10d370ad2130b37aafb6e68eb17",
            text,
        )
        self.assertIn("scammark.prepare_eval_assets", text)
        self.assertIn("--with-bertscore", text)
        self.assertIn("--engineering-metrics", text)
        self.assertIn("metrics_final.json", text)
        self.assertIn("final_evaluation_summary.md", text)
        self.assertIn("publish_results_branch.sh", text)
        self.assertIn("-e PYTHONDONTWRITEBYTECODE=1", text)

    def test_result_contract_uses_release_assets_and_small_results_branch(self):
        common = (SERVER_SCRIPTS / "common.sh").read_text(encoding="utf-8")
        package = (SERVER_SCRIPTS / "package_results.sh").read_text(encoding="utf-8")
        publish = (SERVER_SCRIPTS / "publish_results_branch.sh").read_text(encoding="utf-8")

        self.assertIn("experiment-qwen3-dashscope-full-v2", common)
        self.assertIn("scammark-qwen3-dashscope-full-v2-inference.tar.gz", common)
        self.assertIn("scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz", common)
        self.assertIn("scammark.result_contract", package)
        self.assertIn("build-archive", package)
        self.assertIn("--secret-values-stdin-json", package)
        self.assertNotIn("--exclude=image_cache", package)
        self.assertNotIn("--exclude=checkpoints", package)
        self.assertIn("results/qwen3-dashscope-full-v2", common)
        self.assertIn("$RESULTS_BRANCH", publish)
        self.assertIn("worktree add", publish)
        self.assertIn("worktree remove", publish)
        self.assertNotIn("datetime.now", publish)

    def test_delivery_shell_wiring_is_validated_mode_aware_and_exact(self):
        texts = {
            name: (SERVER_SCRIPTS / name).read_text(encoding="utf-8")
            for name in (
                "prepare_server.sh",
                "run_full_inner.sh",
                "package_results.sh",
                "upload_release_assets.sh",
                "publish_results_branch.sh",
                "finalize_bertscore.sh",
                "recover_delivery.sh",
            )
        }

        for name in (
            "prepare_server.sh",
            "package_results.sh",
            "upload_release_assets.sh",
            "publish_results_branch.sh",
            "finalize_bertscore.sh",
        ):
            with self.subTest(script=name, guard="clean-worktree"):
                self.assertIn("require_clean_worktree", texts[name])
        for name in (
            "package_results.sh",
            "upload_release_assets.sh",
            "publish_results_branch.sh",
            "recover_delivery.sh",
        ):
            with self.subTest(script=name, guard="release-tag"):
                self.assertIn("require_exact_release_tag", texts[name])

        package = texts["package_results.sh"]
        self.assertNotIn('mkdir -p "$DIST_DIR"', package)
        self.assertLess(
            package.index("require_clean_worktree"),
            package.index("build-archive"),
        )
        self.assertLess(package.index("validate-full"), package.index("build-archive"))
        for value in (
            "SCAMMARK_LLM_API_KEY",
            "SCAMMARK_VLM_API_KEY",
            "SCAMMARK_VLM_BASE_URL",
            "--secret-values-stdin-json",
            "verify_result_archive_checksum",
            "record-delivery",
            'unset secret_values_json',
            'export -n secret_values_json',
        ):
            self.assertIn(value, package)
        self.assertNotIn('cd "$DIST_DIR"', package)
        self.assertNotIn("sha256sum", package)
        self.assertNotIn('source "$ENV_FILE"', package)

        upload = texts["upload_release_assets.sh"]
        self.assertLess(upload.index("validate-full"), upload.index("require_github_auth"))
        self.assertLess(
            upload.index("stage_verified_delivery_assets"),
            upload.index("require_github_auth"),
        )
        self.assertLess(upload.index("require_github_auth"), upload.index("gh release"))
        self.assertIn("archive_sha256", upload)
        self.assertIn("record-delivery", upload)
        self.assertGreaterEqual(upload.count("verify_local_delivery_bytes"), 3)

        publish = texts["publish_results_branch.sh"]
        self.assertLess(
            publish.index("stage_verified_delivery_inputs"),
            publish.index("require_github_auth"),
        )
        self.assertLess(
            publish.index("require_github_auth"),
            publish.index('git -C "$ROOT_DIR" fetch'),
        )
        self.assertIn('"$snapshot/inference/full/metrics.json"', publish)
        self.assertIn('"$snapshot/final/full/metrics_final.json"', publish)
        self.assertIn("scan_for_leaks", publish)
        self.assertIn("Unexpected results-branch inventory", publish)
        self.assertIn('"$snapshot/checksums/$INFERENCE_ASSET.sha256"', publish)
        self.assertIn('"$snapshot/checksums/$FINAL_ASSET.sha256"', publish)
        self.assertNotIn('"$DIST_DIR"/*.sha256', publish)

        runner = texts["run_full_inner.sh"]
        self.assertIn('recover_delivery.sh" package', runner)
        self.assertNotIn("30|31|32", runner)
        self.assertIn(
            '[[ "$delivery_status" == "0" ]] || exit "$delivery_status"',
            runner,
        )
        for provider_name in (
            "SCAMMARK_LLM_API_KEY",
            "SCAMMARK_LLM_BASE_URL",
            "SCAMMARK_VLM_API_KEY",
            "SCAMMARK_VLM_BASE_URL",
        ):
            self.assertIn(f"-u {provider_name}", runner)

        finalizer = texts["finalize_bertscore.sh"]
        self.assertIn("load-delivery", finalizer)
        self.assertIn('recover_delivery.sh" --mode final', finalizer)
        self.assertIn("uname -m", finalizer)
        self.assertLess(finalizer.index("load-delivery"), finalizer.index("uname -m"))
        self.assertNotIn("scammark_qwen3_dashscope_server_full/", finalizer)

        recovery = texts["recover_delivery.sh"]
        self.assertIn("[--mode inference|final] package|upload|publish", recovery)
        self.assertIn("load-delivery", recovery)
        self.assertIn("validate-full", recovery)
        self.assertIn("sha256sum", recovery)
        self.assertIn('kill-session -t "=$TMUX_SESSION"', recovery)
        self.assertIn('[[ "$mode" == "inference" ]]', recovery)
        self.assertIn("cleanup_exit_code", recovery)
        self.assertNotIn("scammark.run_experiment", recovery)
        self.assertNotIn("start_full.sh", recovery)

    def test_recover_default_state_and_post_stage_refresh_are_not_truncated(self):
        with OfflineDeliveryFixture() as fixture:
            completed = fixture.run("recover_delivery.sh", "package")
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            state = fixture.state("inference")
            self.assertTrue(state.packaged)
            self.assertTrue(state.uploaded)
            self.assertTrue(state.published)
            python_calls = fixture.log_text(fixture.python_log)
            self.assertEqual(
                python_calls.count("result_contract:build-archive"),
                1,
                python_calls,
            )
            self.assertNotIn("run_experiment", python_calls)

    def test_package_validates_before_mutation_and_keeps_secrets_stdin_only(self):
        hostile = {
            "SCAMMARK_LLM_API_KEY": "fixture-private-hostile-llm",
            "SCAMMARK_LLM_BASE_URL": "https://fixture-private-hostile.invalid",
            "SCAMMARK_VLM_API_KEY": "fixture-private-hostile-vlm",
            "SCAMMARK_VLM_BASE_URL": "https://fixture-private-hostile.invalid/v1",
            "secret_values_json": "fixture-private-hostile-export",
        }
        with OfflineDeliveryFixture() as fixture:
            completed = fixture.run(
                "package_results.sh",
                "inference",
                extra_env=hostile,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertTrue(fixture.state("inference").packaged)
            self.assertFalse(fixture.exposure_marker.exists())
            self.assertNotIn("fixture-private", completed.stdout + completed.stderr)
            archive = fixture.dist / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
            self.assertTrue(archive.is_file())
            self.assertTrue(Path(f"{archive}.sha256").is_file())

        with OfflineDeliveryFixture() as fixture:
            metrics_path = fixture.full / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["engineering"]["sample_count"] = 5657
            metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            rejected = fixture.run("package_results.sh", "inference")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(fixture.dist.exists())
            self.assertFalse(fixture.delivery_state.exists())

    def test_delivery_and_status_allow_only_the_controlled_outputs_parent_link(self):
        def legacy_inventory(root):
            rows = []
            for path in sorted((root, *root.rglob("*"))):
                metadata = path.lstat()
                if path.is_symlink():
                    content = ("symlink", os.readlink(path))
                elif path.is_file():
                    content = ("file", path.read_bytes())
                else:
                    content = ("directory", None)
                rows.append(
                    (
                        path.relative_to(root).as_posix(),
                        metadata.st_mode,
                        metadata.st_nlink,
                        metadata.st_mtime_ns,
                        content,
                    )
                )
            return tuple(rows)

        with OfflineDeliveryFixture() as fixture:
            legacy = (
                fixture.root
                / "outputs"
                / "scammark_qwen3_dashscope_server_full"
            )
            legacy.mkdir()
            sentinel = legacy / "untouched.txt"
            sentinel.write_bytes(b"untouched v1 controlled-parent sentinel\n")
            (legacy / "untouched-link").symlink_to("untouched.txt")
            before = legacy_inventory(legacy)
            outputs = fixture.root / "outputs"
            controlled = fixture.root / "controlled-outputs"
            outputs.rename(controlled)
            outputs.symlink_to(controlled, target_is_directory=True)

            packaged = fixture.run("package_results.sh", "inference")
            recovered = fixture.run("recover_delivery.sh", "package")
            status = fixture.run("status.sh")
            from scammark.run_status import collect_status
            from tests.test_scammark_observability import fake_absent_tmux

            original_cwd = Path.cwd()
            try:
                os.chdir(fixture.root)
                status_projection = collect_status(
                    fixture.root / "configs/qwen3_dashscope_server_full_v2.toml",
                    tmux_session="scammark-full-v2",
                    tmux_runner=fake_absent_tmux,
                )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(packaged.returncode, 0, packaged.stdout + packaged.stderr)
            self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
            self.assertEqual(status.returncode, 0, status.stdout + status.stderr)
            self.assertFalse(status_projection["delivery_corrupt"])
            self.assertEqual(
                status_projection["delivery_modes"]["inference"],
                {"stage": "publish", "status": "complete"},
            )
            self.assertTrue(outputs.is_symlink())
            self.assertEqual(outputs.resolve(), controlled.resolve())
            self.assertEqual(legacy_inventory(controlled / legacy.name), before)

    def test_final_package_binds_published_inference_before_build_or_recovery(self):
        for invocation in (
            ("package_results.sh", "final"),
            ("recover_delivery.sh", "--mode", "final", "package"),
        ):
            with self.subTest(invocation=invocation), OfflineDeliveryFixture() as fixture:
                self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
                metrics_path = fixture.full / "metrics.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics["engineering"]["alternate_valid_result"] = True
                metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
                fixture.clear_logs()

                rejected = fixture.run(*invocation)

                self.assertNotEqual(rejected.returncode, 0)
                self.assertFalse(fixture.state("final").packaged)
                self.assertIsNone(fixture.state("final").archive_sha256)
                self.assertNotIn(
                    "result_contract:build-archive",
                    fixture.log_text(fixture.python_log),
                )
                self.assertNotIn(
                    "gh:release:upload",
                    fixture.log_text(fixture.command_log),
                )

    def test_direct_package_reuses_recorded_archive_without_overwrite(self):
        assets = {
            "inference": "scammark-qwen3-dashscope-full-v2-inference.tar.gz",
            "final": "scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz",
        }
        for mode, asset in assets.items():
            with self.subTest(mode=mode), OfflineDeliveryFixture() as fixture:
                if mode == "final":
                    self.assertEqual(
                        fixture.run("recover_delivery.sh", "package").returncode,
                        0,
                    )
                self.assertEqual(
                    fixture.run("package_results.sh", mode).returncode,
                    0,
                )
                archive = fixture.dist / asset
                checksum = Path(f"{archive}.sha256")
                before = {
                    path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in (archive, checksum, fixture.delivery_state)
                }
                fixture.clear_logs()

                repeated = fixture.run("package_results.sh", mode)

                self.assertEqual(
                    repeated.returncode,
                    0,
                    repeated.stdout + repeated.stderr,
                )
                self.assertEqual(
                    {
                        path: (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in before
                    },
                    before,
                )
                self.assertNotIn(
                    "result_contract:build-archive",
                    fixture.log_text(fixture.python_log),
                )

    def test_run_full_reentry_skips_every_completed_delivery_stage(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
            archive = (
                fixture.dist
                / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
            )
            checksum = Path(f"{archive}.sha256")
            before = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (archive, checksum, fixture.delivery_state)
            }
            fixture.clear_logs()

            rerun = fixture.run("run_full_inner.sh")

            self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
            self.assertEqual(
                {
                    path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in before
                },
                before,
            )
            self.assertNotIn(
                "result_contract:build-archive",
                fixture.log_text(fixture.python_log),
            )
            commands = fixture.log_text(fixture.command_log)
            self.assertNotIn("gh:", commands)
            self.assertNotIn("git:fetch", commands)
            self.assertNotIn("git:push", commands)

    def test_run_full_preserves_generic_recover_failures_without_delivery_write(self):
        for case in ("contract-drift", "dirty-checkout"):
            with self.subTest(case=case), OfflineDeliveryFixture() as fixture:
                extra_env = {}
                if case == "contract-drift":
                    metrics_path = fixture.full / "metrics.json"
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    metrics["engineering"]["sample_count"] = 5657
                    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
                else:
                    extra_env["FAKE_GIT_DIRTY"] = "1"

                failed = fixture.run(
                    "run_full_inner.sh",
                    extra_env=extra_env,
                )

                self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
                self.assertFalse(fixture.delivery_state.exists())
                event_rows = [
                    json.loads(line)
                    for line in (fixture.full / "run_events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertFalse(
                    any(row.get("event") == "delivery_stage" for row in event_rows)
                )

    def test_run_full_scrubs_provider_environment_at_recover_exec_boundary(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
            recover = fixture.root / "scripts" / "server" / "recover_delivery.sh"
            real_recover = recover.with_name("recover_delivery.real.sh")
            recover.rename(real_recover)
            recover.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
for name in SCAMMARK_LLM_API_KEY SCAMMARK_LLM_BASE_URL \\
    SCAMMARK_VLM_API_KEY SCAMMARK_VLM_BASE_URL; do
    [[ -z "${!name-}" ]] || : >"$FAKE_EXPOSURE_MARKER"
done
exec bash scripts/server/recover_delivery.real.sh "$@"
""",
                encoding="utf-8",
            )
            recover.chmod(0o755)
            fixture.exposure_marker.unlink(missing_ok=True)
            hostile = {
                "SCAMMARK_LLM_API_KEY": "fixture-private-hostile-llm",
                "SCAMMARK_LLM_BASE_URL": "https://fixture-private-hostile.invalid",
                "SCAMMARK_VLM_API_KEY": "fixture-private-hostile-vlm",
                "SCAMMARK_VLM_BASE_URL": "https://fixture-private-hostile.invalid/v1",
                "FAKE_EXPOSURE_ONLY_RESULT_CONTRACT": "1",
            }

            rerun = fixture.run("run_full_inner.sh", extra_env=hostile)

            self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
            self.assertFalse(fixture.exposure_marker.exists())

    def test_final_upload_cross_binds_runtime_and_inference_before_remote_calls(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
            self.assertEqual(fixture.run("package_results.sh", "final").returncode, 0)
            metrics_path = fixture.full / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["engineering"]["alternate_valid_result"] = True
            metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            fixture.clear_logs()

            rejected = fixture.run("upload_release_assets.sh", "final")

            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(fixture.state("final").uploaded)
            self.assertNotIn("gh:", fixture.log_text(fixture.command_log))

    def test_upload_rechecks_recorded_bytes_after_remote_lookup_before_upload(self):
        with OfflineDeliveryFixture() as fixture:
            packaged = fixture.run("package_results.sh", "inference")
            self.assertEqual(packaged.returncode, 0, packaged.stderr)
            archive = fixture.dist / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
            archive.write_bytes(archive.read_bytes() + b"preflight-tamper")
            fixture.clear_logs()
            rejected = fixture.run("upload_release_assets.sh", "inference")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn("gh:", fixture.log_text(fixture.command_log))

        with OfflineDeliveryFixture() as fixture:
            packaged = fixture.run("package_results.sh", "inference")
            self.assertEqual(packaged.returncode, 0, packaged.stderr)
            archive = fixture.dist / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
            fixture.clear_logs()
            completed = fixture.run(
                "upload_release_assets.sh",
                "inference",
                extra_env={
                    "FAKE_TAMPER_ARCHIVE_ON_API": "1",
                    "FAKE_TAMPER_ARCHIVE": str(archive),
                },
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(fixture.state("inference").uploaded)
            calls = fixture.log_text(fixture.command_log)
            self.assertIn("gh:api:", calls)
            self.assertNotIn("gh:release:upload", calls)

    def test_publish_rehashes_and_copies_only_verified_explicit_checksums(self):
        with OfflineDeliveryFixture() as fixture:
            packaged = fixture.run("package_results.sh", "inference")
            self.assertEqual(packaged.returncode, 0, packaged.stderr)
            uploaded = fixture.run("upload_release_assets.sh", "inference")
            self.assertEqual(uploaded.returncode, 0, uploaded.stderr)
            (fixture.dist / "unrelated.sha256").write_text(
                "0" * 64 + "  unrelated\n",
                encoding="ascii",
            )
            fixture.clear_logs()
            published = fixture.run(
                "publish_results_branch.sh",
                "inference",
                extra_env={"FAKE_WORKTREE_SEED": "1"},
            )
            self.assertEqual(
                published.returncode,
                0,
                published.stdout + published.stderr,
            )
            self.assertTrue(fixture.state("inference").published)
            staged = set(fixture.log_text(fixture.publish_capture).splitlines())
            self.assertEqual(
                staged,
                {
                    "README.md",
                    "checksums/scammark-qwen3-dashscope-full-v2-inference.tar.gz.sha256",
                    "full_summary.md",
                    "metrics.json",
                },
            )
            self.assertFalse(any("unrelated" in name for name in staged))
            self.assertFalse(any("metrics_final" in name for name in staged))

        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(
                fixture.run("package_results.sh", "inference").returncode,
                0,
            )
            self.assertEqual(
                fixture.run("upload_release_assets.sh", "inference").returncode,
                0,
            )
            self.assertEqual(
                fixture.run("publish_results_branch.sh", "inference").returncode,
                0,
            )
            self.assertEqual(
                fixture.run("package_results.sh", "final").returncode,
                0,
            )
            fixture.clear_logs()
            rejected = fixture.run("publish_results_branch.sh", "inference")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn("git:fetch", fixture.log_text(fixture.command_log))

    def test_publish_stages_branch_inputs_before_git_and_rejects_unsafe_sources(self):
        for link_kind in ("symlink", "hardlink"):
            with self.subTest(link_kind=link_kind), OfflineDeliveryFixture() as fixture:
                self.assertEqual(
                    fixture.run("package_results.sh", "inference").returncode,
                    0,
                )
                self.assertEqual(
                    fixture.run("upload_release_assets.sh", "inference").returncode,
                    0,
                )
                source = fixture.full / "full_summary.md"
                target = (
                    fixture.root
                    / "outputs"
                    / "scammark_qwen3_dashscope_server_full"
                    / f"{link_kind}-summary.md"
                )
                target.parent.mkdir(parents=True)
                target.write_bytes(source.read_bytes())
                source.unlink()
                if link_kind == "symlink":
                    source.symlink_to(target)
                else:
                    os.link(target, source)
                before = (target.read_bytes(), target.stat().st_mtime_ns)
                fixture.clear_logs()

                rejected = fixture.run(
                    "publish_results_branch.sh",
                    "inference",
                )

                self.assertNotEqual(rejected.returncode, 0)
                self.assertNotIn("git:fetch", fixture.log_text(fixture.command_log))
                self.assertNotIn("git:push", fixture.log_text(fixture.command_log))
                self.assertEqual(
                    (target.read_bytes(), target.stat().st_mtime_ns),
                    before,
                )
                self.assertFalse(fixture.state("inference").published)

        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
            self.assertEqual(fixture.run("package_results.sh", "final").returncode, 0)
            self.assertEqual(fixture.run("upload_release_assets.sh", "final").returncode, 0)
            source = fixture.full / "final_evaluation_summary.md"
            target = (
                fixture.root
                / "outputs"
                / "scammark_qwen3_dashscope_server_full"
                / "final-summary.md"
            )
            target.parent.mkdir(parents=True)
            target.write_bytes(source.read_bytes())
            source.unlink()
            source.symlink_to(target)
            before = (target.read_bytes(), target.stat().st_mtime_ns)
            fixture.clear_logs()

            rejected = fixture.run("publish_results_branch.sh", "final")

            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn("git:fetch", fixture.log_text(fixture.command_log))
            self.assertEqual((target.read_bytes(), target.stat().st_mtime_ns), before)
            self.assertFalse(fixture.state("final").published)

    def test_publish_uses_pinned_snapshot_and_rejects_runtime_archive_mismatch(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("package_results.sh", "inference").returncode, 0)
            self.assertEqual(fixture.run("upload_release_assets.sh", "inference").returncode, 0)
            source = fixture.full / "full_summary.md"
            original = source.read_bytes()
            target = (
                fixture.root
                / "outputs"
                / "scammark_qwen3_dashscope_server_full"
                / "swapped-summary.md"
            )
            target.parent.mkdir(parents=True)
            target.write_bytes(b"# untouched v1 swapped summary\n")
            before = (target.read_bytes(), target.stat().st_mtime_ns)
            fixture.clear_logs()

            published = fixture.run(
                "publish_results_branch.sh",
                "inference",
                extra_env={
                    "FAKE_SWAP_PUBLISH_SOURCE_ON_FETCH": "1",
                    "FAKE_SWAP_PUBLISH_SOURCE": str(source),
                    "FAKE_SWAP_PUBLISH_TARGET": str(target),
                },
            )

            self.assertEqual(published.returncode, 0, published.stdout + published.stderr)
            self.assertEqual(
                (fixture.publish_tree / "full_summary.md").read_bytes(),
                original,
            )
            self.assertEqual((target.read_bytes(), target.stat().st_mtime_ns), before)

        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("package_results.sh", "inference").returncode, 0)
            self.assertEqual(fixture.run("upload_release_assets.sh", "inference").returncode, 0)
            metrics_path = fixture.full / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["engineering"]["alternate_valid_result"] = True
            metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            fixture.clear_logs()

            rejected = fixture.run("publish_results_branch.sh", "inference")

            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn("git:fetch", fixture.log_text(fixture.command_log))
            self.assertFalse(fixture.state("inference").published)

    def test_inference_republish_preserves_verified_final_branch_inventory(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
            self.assertEqual(
                fixture.run(
                    "recover_delivery.sh",
                    "--mode",
                    "final",
                    "package",
                ).returncode,
                0,
            )
            expected_final = {
                "metrics_final.json": (fixture.full / "metrics_final.json").read_bytes(),
                "final_evaluation_summary.md": (
                    fixture.full / "final_evaluation_summary.md"
                ).read_bytes(),
            }
            fixture.clear_logs()

            republished = fixture.run("publish_results_branch.sh", "inference")

            self.assertEqual(
                republished.returncode,
                0,
                republished.stdout + republished.stderr,
            )
            self.assertEqual(
                set(fixture.publish_capture.read_text(encoding="utf-8").splitlines()),
                {
                    "README.md",
                    "metrics.json",
                    "full_summary.md",
                    "metrics_final.json",
                    "final_evaluation_summary.md",
                    "checksums/scammark-qwen3-dashscope-full-v2-inference.tar.gz.sha256",
                    "checksums/scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz.sha256",
                },
            )
            for name, expected in expected_final.items():
                self.assertEqual((fixture.publish_tree / name).read_bytes(), expected)

    def test_inference_republish_rejects_uploaded_unpublished_final_before_git_mutation(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(
                fixture.run("recover_delivery.sh", "package").returncode,
                0,
            )
            self.assertEqual(
                fixture.run("package_results.sh", "final").returncode,
                0,
            )
            self.assertEqual(
                fixture.run("upload_release_assets.sh", "final").returncode,
                0,
            )
            self.assertTrue(fixture.state("inference").published)
            self.assertTrue(fixture.state("final").uploaded)
            self.assertFalse(fixture.state("final").published)
            fixture.clear_logs()
            shutil.rmtree(fixture.publish_tree, ignore_errors=True)

            rejected = fixture.run("publish_results_branch.sh", "inference")

            self.assertNotEqual(rejected.returncode, 0)
            commands = fixture.log_text(fixture.command_log)
            for mutation in (
                "git:fetch",
                "git:branch",
                "git:worktree",
                "git:config",
                "git:add",
                "git:commit",
                "git:push",
            ):
                self.assertNotIn(mutation, commands)
            self.assertFalse(fixture.publish_capture.exists())
            self.assertFalse(fixture.publish_tree.exists())
            self.assertTrue(fixture.state("inference").published)
            self.assertFalse(fixture.state("final").published)

    def test_upload_requires_post_upload_remote_digest_and_never_records_mismatch(self):
        for case, extra_env in (
            ("wrong", {"FAKE_POST_UPLOAD_DIGEST_MODE": "wrong"}),
            ("unavailable", {"FAKE_POST_UPLOAD_DIGEST_MODE": "unavailable"}),
            ("pathname-swap", {"FAKE_SWAP_UPLOAD_PATH": "1"}),
        ):
            with self.subTest(case=case), OfflineDeliveryFixture() as fixture:
                self.assertEqual(
                    fixture.run("package_results.sh", "inference").returncode,
                    0,
                )
                fixture.clear_logs()

                rejected = fixture.run(
                    "upload_release_assets.sh",
                    "inference",
                    extra_env=extra_env,
                )

                self.assertNotEqual(rejected.returncode, 0)
                self.assertFalse(fixture.state("inference").uploaded)
                calls = fixture.log_text(fixture.command_log)
                self.assertIn("gh:release:upload", calls)
                self.assertGreaterEqual(calls.count("gh:api:"), 2)

        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(
                fixture.run("package_results.sh", "inference").returncode,
                0,
            )
            self.assertEqual(
                fixture.run("upload_release_assets.sh", "inference").returncode,
                0,
            )
            archive = fixture.dist / "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
            archive.write_bytes(archive.read_bytes() + b"tamper")
            fixture.clear_logs()
            rejected = fixture.run("publish_results_branch.sh", "inference")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn("git:fetch", fixture.log_text(fixture.command_log))

    def test_recovery_uses_durable_failed_stage_and_never_kills_for_final_mode(self):
        from scammark.result_contract import record_delivery_stage

        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(
                fixture.run("package_results.sh", "inference").returncode,
                0,
            )
            state = fixture.state("inference")
            record_delivery_stage(
                fixture.delivery_state,
                mode="inference",
                stage="upload",
                status="failed",
                archive_sha256=state.archive_sha256,
                exit_code=31,
            )
            fixture.clear_logs()
            recovered = fixture.run(
                "recover_delivery.sh",
                "package",
                extra_env={"FAKE_TMUX_STATE": "1|31"},
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertNotIn(
                "result_contract:build-archive",
                fixture.log_text(fixture.python_log),
            )
            self.assertIn(
                "kill:=scammark-full-v2-test-delivery",
                fixture.log_text(fixture.tmux_log),
            )

        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(
                fixture.run("recover_delivery.sh", "package").returncode,
                0,
            )
            self.assertEqual(
                fixture.run("package_results.sh", "final").returncode,
                0,
            )
            state = fixture.state("final")
            record_delivery_stage(
                fixture.delivery_state,
                mode="final",
                stage="upload",
                status="failed",
                archive_sha256=state.archive_sha256,
                exit_code=31,
            )
            fixture.clear_logs()
            recovered = fixture.run(
                "recover_delivery.sh",
                "--mode",
                "final",
                "upload",
                extra_env={"FAKE_TMUX_STATE": "1|31"},
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertNotIn("kill:", fixture.log_text(fixture.tmux_log))
            self.assertNotIn(
                "result_contract:build-archive",
                fixture.log_text(fixture.python_log),
            )

    def test_publish_recovery_never_packages_or_uploads_and_kills_only_matching_dead_session(self):
        from scammark.result_contract import record_delivery_stage

        for pane_state, expect_kill in (("1|32", True), ("1|31", False), ("0|32", False)):
            with self.subTest(pane_state=pane_state), OfflineDeliveryFixture() as fixture:
                self.assertEqual(
                    fixture.run("package_results.sh", "inference").returncode,
                    0,
                )
                self.assertEqual(
                    fixture.run("upload_release_assets.sh", "inference").returncode,
                    0,
                )
                state = fixture.state("inference")
                record_delivery_stage(
                    fixture.delivery_state,
                    mode="inference",
                    stage="publish",
                    status="failed",
                    archive_sha256=state.archive_sha256,
                    exit_code=32,
                )
                fixture.clear_logs()
                recovered = fixture.run(
                    "recover_delivery.sh",
                    "publish",
                    extra_env={"FAKE_TMUX_STATE": pane_state},
                )
                self.assertEqual(recovered.returncode, 0, recovered.stderr)
                python_calls = fixture.log_text(fixture.python_log)
                commands = fixture.log_text(fixture.command_log)
                self.assertNotIn("result_contract:build-archive", python_calls)
                self.assertNotIn("gh:release:upload", commands)
                killed = "kill:" in fixture.log_text(fixture.tmux_log)
                self.assertEqual(killed, expect_kill)

    def test_finalizer_reuses_packaged_final_archive_without_docker_or_repackage(self):
        with OfflineDeliveryFixture() as fixture:
            inference = fixture.run("recover_delivery.sh", "package")
            self.assertEqual(inference.returncode, 0, inference.stderr)
            packaged = fixture.run("package_results.sh", "final")
            self.assertEqual(packaged.returncode, 0, packaged.stderr)
            fixture.clear_logs()
            completed = fixture.run("finalize_bertscore.sh")
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertTrue(fixture.state("final").published)
            self.assertFalse(fixture.docker_marker.exists())
            self.assertNotIn(
                "result_contract:build-archive",
                fixture.log_text(fixture.python_log),
            )

    def test_finalizer_requires_published_inference_before_compute_and_retries_partial_final(self):
        with OfflineDeliveryFixture() as fixture:
            (fixture.full / "final_evaluation_summary.md").unlink()
            blocked = fixture.run(
                "finalize_bertscore.sh",
                extra_env={"FAKE_UNAME_ARCH": "aarch64"},
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn(
                "bash scripts/server/recover_delivery.sh package",
                blocked.stdout + blocked.stderr,
            )
            self.assertFalse(fixture.docker_marker.exists())

        with OfflineDeliveryFixture() as fixture:
            delivered = fixture.run("recover_delivery.sh", "package")
            self.assertEqual(delivered.returncode, 0, delivered.stderr)
            (fixture.full / "final_evaluation_summary.md").unlink()
            fixture.docker_marker.unlink(missing_ok=True)
            retried = fixture.run(
                "finalize_bertscore.sh",
                extra_env={"FAKE_UNAME_ARCH": "aarch64"},
            )
            self.assertNotEqual(retried.returncode, 0)
            self.assertTrue(fixture.docker_marker.exists())

    def test_finalizer_compute_rejects_partial_symlink_to_v1_without_target_touch(self):
        with OfflineDeliveryFixture() as fixture:
            delivered = fixture.run("recover_delivery.sh", "package")
            self.assertEqual(delivered.returncode, 0, delivered.stderr)
            legacy_target = (
                fixture.root
                / "outputs"
                / "scammark_qwen3_dashscope_server_full"
                / "full"
                / "metrics_final.json"
            )
            legacy_target.parent.mkdir(parents=True)
            legacy_target.write_bytes(b"untouched-v1-final-marker\n")
            hostile = fixture.full / "metrics_final.json"
            hostile.unlink()
            hostile.symlink_to(legacy_target)
            (fixture.full / "final_evaluation_summary.md").unlink()
            before = legacy_target.read_bytes()
            fixture.docker_marker.unlink(missing_ok=True)

            rejected = fixture.run(
                "finalize_bertscore.sh",
                extra_env={"FAKE_UNAME_ARCH": "aarch64"},
            )

            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(fixture.docker_marker.exists())
            self.assertTrue(hostile.is_symlink())
            self.assertEqual(legacy_target.read_bytes(), before)

    def test_finalizer_ignores_legacy_fixed_venv_alias_and_never_references_it(self):
        with OfflineDeliveryFixture() as fixture:
            delivered = fixture.run("recover_delivery.sh", "package")
            self.assertEqual(delivered.returncode, 0, delivered.stderr)
            legacy_target = (
                fixture.root
                / "outputs"
                / "scammark_qwen3_dashscope_server_full"
                / "bertscore-venv"
            )
            legacy_target.mkdir(parents=True)
            marker = legacy_target / "untouched.marker"
            marker.write_bytes(b"untouched-v1-venv-marker\n")
            hostile = fixture.root / ".venv-bertscore-dgx"
            hostile.symlink_to(legacy_target, target_is_directory=True)
            (fixture.full / "final_evaluation_summary.md").unlink()
            before = (marker.read_bytes(), marker.stat().st_mtime_ns)
            fixture.docker_marker.unlink(missing_ok=True)

            rejected = fixture.run(
                "finalize_bertscore.sh",
                extra_env={"FAKE_UNAME_ARCH": "aarch64"},
            )

            self.assertNotEqual(rejected.returncode, 0)
            self.assertTrue(fixture.docker_marker.exists())
            self.assertTrue(hostile.is_symlink())
            self.assertEqual(
                (marker.read_bytes(), marker.stat().st_mtime_ns),
                before,
            )
            script = (SERVER_SCRIPTS / "finalize_bertscore.sh").read_text(
                encoding="utf-8"
            )
            self.assertNotIn(".venv-bertscore-dgx", script)
            self.assertNotIn(".container-home", script)

    def test_finalizer_recursively_rejects_hostile_writable_subtrees(self):
        cases = [
            "nested-symlink",
            "nested-hardlink",
            "hf-directory-symlink",
            "hf-multilink-symlink",
        ]
        if hasattr(os, "mkfifo"):
            cases.append("nested-fifo")
        for case in cases:
            with self.subTest(case=case), OfflineDeliveryFixture() as fixture:
                self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
                (fixture.full / "final_evaluation_summary.md").unlink()
                legacy = (
                    fixture.root
                    / "outputs"
                    / "scammark_qwen3_dashscope_server_full"
                    / case
                )
                legacy.parent.mkdir(parents=True, exist_ok=True)
                subtree = fixture.root / "models" / "huggingface"
                if case == "nested-symlink":
                    legacy.write_bytes(b"untouched nested symlink target\n")
                    hostile = subtree / "pyvenv.cfg"
                    hostile.symlink_to(legacy)
                    before = (legacy.read_bytes(), legacy.stat().st_mtime_ns)
                elif case == "nested-hardlink":
                    legacy.write_bytes(b"untouched nested hardlink target\n")
                    hostile = subtree / "state"
                    os.link(legacy, hostile)
                    before = (legacy.read_bytes(), legacy.stat().st_mtime_ns)
                elif case == "nested-fifo":
                    hostile = subtree / "hostile.fifo"
                    os.mkfifo(hostile)
                    before = None
                elif case == "hf-directory-symlink":
                    target_directory = subtree / "internal-directory"
                    target_directory.mkdir()
                    marker = target_directory / "untouched.marker"
                    marker.write_bytes(b"untouched internal directory\n")
                    hostile = subtree / "directory-link"
                    hostile.symlink_to(target_directory, target_is_directory=True)
                    legacy = marker
                    before = (marker.read_bytes(), marker.stat().st_mtime_ns)
                else:
                    target_file = subtree / "internal-target"
                    target_file.write_bytes(b"untouched internal multilink\n")
                    os.link(target_file, subtree / "second-hardlink")
                    hostile = subtree / "file-link"
                    hostile.symlink_to(target_file)
                    legacy = target_file
                    before = (target_file.read_bytes(), target_file.stat().st_mtime_ns)
                fixture.docker_marker.unlink(missing_ok=True)

                rejected = fixture.run(
                    "finalize_bertscore.sh",
                    extra_env={"FAKE_UNAME_ARCH": "aarch64"},
                )

                self.assertNotEqual(rejected.returncode, 0)
                self.assertFalse(fixture.docker_marker.exists())
                self.assertTrue(hostile.exists() or hostile.is_symlink())
                if before is not None:
                    self.assertEqual(
                        (legacy.read_bytes(), legacy.stat().st_mtime_ns),
                        before,
                    )

    def test_finalizer_allows_singly_linked_internal_hf_blob_symlink(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
            cache = fixture.root / "models" / "huggingface"
            blob = cache / "blobs" / "model-blob"
            blob.parent.mkdir()
            blob.write_bytes(b"verified internal model blob\n")
            snapshot = cache / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            link = snapshot / "model.bin"
            link.symlink_to("../../blobs/model-blob")
            before = (blob.read_bytes(), blob.stat().st_mtime_ns, blob.stat().st_nlink)
            (fixture.full / "final_evaluation_summary.md").unlink()
            fixture.docker_marker.unlink(missing_ok=True)

            attempted = fixture.run(
                "finalize_bertscore.sh",
                extra_env={"FAKE_UNAME_ARCH": "aarch64"},
            )

            self.assertNotEqual(attempted.returncode, 0)
            self.assertTrue(fixture.docker_marker.exists())
            self.assertTrue(link.is_symlink())
            self.assertEqual(
                (blob.read_bytes(), blob.stat().st_mtime_ns, blob.stat().st_nlink),
                before,
            )

    def test_finalizer_binds_compute_inputs_to_published_inference_archive(self):
        for changed_file in ("predictions.jsonl", "metrics.json"):
            with self.subTest(changed_file=changed_file), OfflineDeliveryFixture() as fixture:
                self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
                path = fixture.full / changed_file
                if changed_file == "predictions.jsonl":
                    rows = path.read_text(encoding="utf-8").splitlines()
                    first = json.loads(rows[0])
                    first["label"] = "alternate-valid-label"
                    rows[0] = json.dumps(first)
                    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
                else:
                    metrics = json.loads(path.read_text(encoding="utf-8"))
                    metrics["engineering"]["alternate_valid_result"] = True
                    path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
                (fixture.full / "final_evaluation_summary.md").unlink()
                fixture.docker_marker.unlink(missing_ok=True)

                rejected = fixture.run(
                    "finalize_bertscore.sh",
                    extra_env={"FAKE_UNAME_ARCH": "aarch64"},
                )

                self.assertNotEqual(rejected.returncode, 0)
                self.assertFalse(fixture.docker_marker.exists())

    def test_finalizer_binds_inference_before_any_final_recovery_shortcut(self):
        scenarios = ("complete-artifacts", "packaged-final")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), OfflineDeliveryFixture() as fixture:
                self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
                if scenario == "packaged-final":
                    self.assertEqual(
                        fixture.run("package_results.sh", "final").returncode,
                        0,
                    )
                predictions = fixture.full / "predictions.jsonl"
                rows = predictions.read_text(encoding="utf-8").splitlines()
                first = json.loads(rows[0])
                first["label"] = "alternate-valid-label"
                rows[0] = json.dumps(first)
                predictions.write_text("\n".join(rows) + "\n", encoding="utf-8")
                fixture.clear_logs()
                fixture.docker_marker.unlink(missing_ok=True)

                rejected = fixture.run("finalize_bertscore.sh")

                self.assertNotEqual(rejected.returncode, 0)
                self.assertFalse(fixture.docker_marker.exists())
                self.assertNotIn(
                    "result_contract:build-archive",
                    fixture.log_text(fixture.python_log),
                )
                commands = fixture.log_text(fixture.command_log)
                self.assertNotIn("gh:release:upload", commands)
                self.assertNotIn("git:fetch", commands)

    def test_finalizer_ignores_unrecorded_final_residue_for_inference_binding(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
            fixture.dist.mkdir(parents=True, exist_ok=True)
            residue = (
                fixture.dist
                / "scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz"
            )
            residue.write_bytes(b"unrecorded failed final package residue\n")
            Path(f"{residue}.sha256").write_text(
                "0" * 64 + f"  {residue.name}\n",
                encoding="ascii",
            )
            (fixture.full / "final_evaluation_summary.md").unlink()
            fixture.docker_marker.unlink(missing_ok=True)

            attempted = fixture.run(
                "finalize_bertscore.sh",
                extra_env={"FAKE_UNAME_ARCH": "aarch64"},
            )

            self.assertNotEqual(attempted.returncode, 0)
            self.assertTrue(fixture.docker_marker.exists())

    def test_finalizer_revalidates_packaged_final_bytes_before_upload_recovery(self):
        with OfflineDeliveryFixture() as fixture:
            self.assertEqual(fixture.run("recover_delivery.sh", "package").returncode, 0)
            self.assertEqual(fixture.run("package_results.sh", "final").returncode, 0)
            final_metrics = fixture.full / "metrics_final.json"
            metrics = json.loads(final_metrics.read_text(encoding="utf-8"))
            metrics["engineering"]["alternate_valid_result"] = True
            final_metrics.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            fixture.clear_logs()

            rejected = fixture.run("finalize_bertscore.sh")

            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(fixture.state("final").uploaded)
            self.assertNotIn(
                "gh:release:upload",
                fixture.log_text(fixture.command_log),
            )

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_full_runner_maps_package_failure_to_durable_30_without_lifecycle_regression(self):
        from scammark.result_contract import load_delivery_state
        from tests.result_fixture import CompleteResultFixture

        fixture = ManagedTmuxFixture(ROOT)
        try:
            with CompleteResultFixture(sample_count=5658) as source:
                shutil.copytree(
                    source.full,
                    fixture.output_root / "full",
                    dirs_exist_ok=True,
                )
            (fixture.root / ".env").write_text(
                "SCAMMARK_LLM_API_KEY=fixture-key\n"
                "SCAMMARK_VLM_API_KEY=fixture-key\n",
                encoding="utf-8",
            )
            (fixture.root / ".env").chmod(0o600)
            fixture._commit_fixture_contract("valid full with package-stage failure")
            fixture.configure_immediate_fake_exit(0)
            fixture.start_fake_runner()
            self.assertEqual(fixture.wait_for_dead_pane_status(), 30)
            fixture.wait_for_runner_and_tee_exit()
            state = load_delivery_state(
                fixture.output_root / "full" / "delivery_state.json",
                mode="inference",
            )
            self.assertFalse(state.packaged)
            self.assertEqual(state.last_exit_code, 30)
            shell_exited = [
                row for row in fixture.events() if row.get("event") == "shell_exited"
            ]
            self.assertEqual(len(shell_exited), 1, fixture.events())
            self.assertEqual(shell_exited[0].get("exit_code"), 30)
            self.assertFalse(fixture.runner_identity_exists())
        finally:
            fixture.cleanup()

    def test_bash_scripts_are_syntax_valid(self):
        bash = shutil.which("bash")
        git_bash = Path(r"D:\Program Files\Git\bin\bash.exe")
        if git_bash.exists():
            bash = str(git_bash)
        if not bash:
            self.skipTest("bash is unavailable")

        scripts = sorted(SERVER_SCRIPTS.glob("*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            completed = subprocess.run(
                [bash, "-n", str(script)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"{script}: {completed.stderr}",
            )

    def test_v2_tmux_uses_verified_runner_pid_not_process_group_ctrl_c(self):
        paths = {
            name: SERVER_SCRIPTS / name
            for name in (
                "common.sh",
                "start_full.sh",
                "run_full_inner.sh",
                "pause_full.sh",
            )
        }
        texts = {
            name: path.read_text(encoding="utf-8") if path.is_file() else ""
            for name, path in paths.items()
        }

        required = {
            "common.sh": (
                "require_clean_worktree",
                "tmux_target",
                "process_starttime",
            ),
            "start_full.sh": (
                "remain-on-exit on",
                "respawn-pane",
                '"$(tmux_target)"',
                'kill-session -t "=$TMUX_SESSION"',
                "require_safe_runtime_namespace",
            ),
            "run_full_inner.sh": (
                "RUNNER_PID_FILE",
                "SCAMMARK_ATTEMPT_ID",
                "write_runner_identity_atomically",
                "require_safe_runtime_namespace",
            ),
            "pause_full.sh": (
                "/proc/",
                "starttime",
                'kill -INT "$runner_pid"',
                "os.O_NOFOLLOW",
                "os.fstat",
                "os.read",
                "expected_production_arguments",
                "normalized_arguments == expected_production_arguments",
                "arguments[1]",
                "require_safe_runtime_namespace",
            ),
        }
        for name, needles in required.items():
            for needle in needles:
                with self.subTest(script=name, required=needle):
                    self.assertIn(needle, texts[name])

        start = texts["start_full.sh"]
        with self.subTest(order="remain-before-respawn"):
            self.assertTrue(
                "remain-on-exit on" in start and "respawn-pane" in start
            )
            self.assertLess(
                start.index("remain-on-exit on"),
                start.index("respawn-pane"),
            )

        pause = texts["pause_full.sh"]
        for forbidden in ("send-keys", "C-c", "pkill", "killall", "kill -- -"):
            with self.subTest(script="pause_full.sh", forbidden=forbidden):
                self.assertNotIn(forbidden, pause)

        common = texts["common.sh"]
        self.assertIn(
            '[[ "$pane_dead" == "1" && -z "$pane_dead_status" ]]', common
        )
        self.assertEqual(
            common.count('tmux run-shell -b -t "$(tmux_target)" true'), 1
        )
        self.assertIn("tmux_pane_state_with_reap_nudge", start)
        self.assertIn("tmux_pane_state_with_reap_nudge", pause)

    def test_dead_pane_exit_routing_does_not_blindly_resume(self):
        start = (SERVER_SCRIPTS / "start_full.sh").read_text(encoding="utf-8")

        for code in ("0", "1", "2", "20", "21", "22", "23", "30", "31", "32", "130", "143"):
            with self.subTest(exit_code=code):
                self.assertIn(f"{code})", start)
        for required in (
            "--ack-breaker",
            "recover_delivery.sh package",
            "recover_delivery.sh upload",
            "recover_delivery.sh publish",
            "pane_dead_status",
        ):
            with self.subTest(required=required):
                self.assertIn(required, start)

    def test_runtime_namespace_guard_rejects_child_mount_and_device_boundary(self):
        common = (SERVER_SCRIPTS / "common.sh").read_text(encoding="utf-8")

        self.assertIn("os.path.ismount(path)", common)
        self.assertIn(
            "metadata.st_dev != output_root_metadata.st_dev",
            common,
        )
        self.assertIn("contains a child mount point", common)
        self.assertIn("crosses the v2 output device boundary", common)

    def test_status_script_is_exact_read_only_delegate(self):
        status = (SERVER_SCRIPTS / "status.sh").read_text(encoding="utf-8")
        expected = """#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
exec "$VENV_PYTHON" -m scammark.run_status \\
  --config "$CONFIG_PATH" --tmux-session "$TMUX_SESSION"
"""
        self.assertEqual(status, expected)

    def test_public_server_runbook_contains_the_complete_handoff(self):
        text = (ROOT / "docs" / "SERVER_FULL_RUN_V2.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("gh repo clone nlixsys/ScamWeb-V", text)
        self.assertIn("git checkout experiment-qwen3-dashscope-full-v2", text)
        self.assertIn("/data/scamweb_dataset", text)
        self.assertIn("prepare_server.sh", text)
        self.assertIn("--pilot 10 --resume", text)
        self.assertIn("start_full.sh", text)
        self.assertIn("status.sh", text)
        self.assertIn("finalize_bertscore.sh", text)
        self.assertIn("download_full_results_v2.ps1", text)

    def test_v2_release_contract_records_exact_frozen_asset_and_result_names(self):
        text = (ROOT / "docs" / "EXPERIMENT_RELEASE_V2.md").read_text(
            encoding="utf-8"
        )

        for value in (
            "experiment-qwen3-dashscope-full-v2",
            "21,802,246` bytes",
            "8d0894a043913f6e049cfc4b212a310cb4fd79a810ff5651372c802f8ab2eb61",
            "81904995efcae0bacdee1a2e31d0da9d92a81adb099c4a9d0b5b01da885e0d1c",
            "012ec051c0518900be2246ed5ef0a07185768bf8f451e378056f75289fa160ff",
            "scammark-qwen3-dashscope-full-v2-inference.tar.gz",
            "scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz",
            "results/qwen3-dashscope-full-v2",
        ):
            with self.subTest(required=value):
                self.assertIn(value, text)

    def test_windows_frozen_packager_is_deterministic_and_raw_free(self):
        text = (
            ROOT / "scripts" / "windows" / "package_frozen_assets.ps1"
        ).read_text(encoding="utf-8-sig")

        for value in (
            "tarfile.USTAR_FORMAT", "gzip.GzipFile", "mtime=0",
            "info.uid = 0", "info.gid = 0", "info.mode = 0o644",
            "scamweb_v_manifest.jsonl",
            "web_evidence.jsonl",
        ):
            with self.subTest(required=value):
                self.assertIn(value, text)
        for forbidden in ("recaptured_mhtml", "audit_image_cache", "*.mhtml", "*.png"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_v2_runbook_contains_paid_boundary_pause_resume_and_delivery_actions(self):
        text = (ROOT / "docs" / "SERVER_FULL_RUN_V2.md").read_text(
            encoding="utf-8"
        )

        required = (
            "git checkout experiment-qwen3-dashscope-full-v2",
            "prepare_server.sh",
            "--pilot 10 --resume",
            "start_full.sh",
            "status.sh",
            "pause_full.sh",
            "tmux attach -t scammark-full-v2",
            "start_full.sh --ack-breaker",
            "recover_delivery.sh package",
            "recover_delivery.sh upload",
            "recover_delivery.sh publish",
            "recover_delivery.sh --mode final upload",
            "finalize_bertscore.sh",
            "download_full_results_v2.ps1",
            "provider invoice",
            "remaining",
            "USD",
            "CNY",
        )
        for value in required:
            with self.subTest(required=value):
                self.assertIn(value, text)
        for code in (0, 1, 2, 20, 21, 22, 23, 30, 31, 32, 130, 143):
            with self.subTest(exit_code=code):
                self.assertIn(f"`{code}`", text)

    def test_v2_runbook_requires_windows_parser_and_offline_fixture_gate(self):
        text = (ROOT / "docs" / "SERVER_FULL_RUN_V2.md").read_text(
            encoding="utf-8"
        )

        for value in (
            "A missing `pwsh` is a failed validation gate",
            "Before paid smoke",
            "pwsh -NoProfile -Command",
            "[System.Management.Automation.Language.Parser]::ParseFile",
            "python -m unittest -v -k windows_v2_downloader tests.test_server_workflow",
            "CompleteResultFixture(sample_count=5658)",
            "build_archives()",
            "$fixtureDirectory",
            "Do not assume a `D:` path",
            "-ArchiveDirectory",
            "-VerifyOnly",
            "-SkipResultsBranchFetch",
        ):
            with self.subTest(required=value):
                self.assertIn(value, text)
        ordered_markers = (
            "[System.Management.Automation.Language.Parser]::ParseFile",
            "python -m unittest -v -k windows_v2_downloader tests.test_server_workflow",
            "CompleteResultFixture(sample_count=5658)",
        )
        if all(marker in text for marker in ordered_markers):
            positions = tuple(text.index(marker) for marker in ordered_markers)
            archive_position = text.index("-ArchiveDirectory", positions[-1])
            self.assertEqual(
                (*positions, archive_position),
                tuple(sorted((*positions, archive_position))),
            )

    def test_windows_download_verifies_final_metric_contract(self):
        text = (ROOT / "scripts" / "windows" / "download_full_results.ps1").read_text(
            encoding="utf-8-sig"
        )

        self.assertIn("metrics_final.json", text)
        self.assertIn("prediction_sample_count", text)
        self.assertIn("bertscore_status", text)

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_pause_targets_python_retains_130_and_exactly_resumes_dead_session(self):
        fixture = ManagedTmuxFixture(ROOT)
        unrelated_session = fixture.create_unrelated_session()
        try:
            fixture.start_fake_runner()
            first_identity = fixture.wait_for_pid_file()
            first_session_id = fixture.session_id()

            paused = fixture.run_pause()
            self.assertEqual(paused.returncode, 0, paused.stderr)
            fixture.wait_for_dead_pane()

            self.assertEqual(fixture.pane_dead_status(), 130)
            nudge_pre_states = fixture.tmux_nudge_pre_states()
            self.assertLessEqual(len(nudge_pre_states), 1)
            self.assertTrue(
                all(state == ("1", "") for state in nudge_pre_states),
                nudge_pre_states,
            )
            self.assertEqual(fixture.events()[-1]["event"], "run_stopped")
            self.assertEqual(fixture.events()[-1]["pid"], first_identity["pid"])
            self.assertIn("exit code 130", fixture.run_log())
            self.assertTrue(fixture.tee_process_drained())
            self.assertFalse(fixture.runner_identity_exists())
            self.assertTrue(fixture.session_exists(unrelated_session))

            fixture.start_fake_runner()
            second_identity = fixture.wait_for_pid_file()
            self.assertNotEqual(second_identity["pid"], first_identity["pid"])
            self.assertNotEqual(
                second_identity["attempt_id"], first_identity["attempt_id"]
            )
            self.assertNotEqual(fixture.session_id(), first_session_id)
            self.assertTrue(fixture.session_exists(unrelated_session))
        finally:
            fixture.kill_exact_session(unrelated_session)
            fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_both_test_override_guards_are_required_to_bypass_production_gates(self):
        for allow, runner, expected_success in (
            ("1", None, False),
            (None, "fake", False),
            ("1", "fake", True),
        ):
            with self.subTest(allow=allow, runner=runner):
                fixture = ManagedTmuxFixture(ROOT)
                try:
                    fixture.invalidate_production_prerequisites()
                    completed = fixture.run_start(
                        allow_test_overrides=allow,
                        test_runner=runner,
                    )
                    self.assertEqual(
                        completed.returncode == 0,
                        expected_success,
                        completed.stdout + completed.stderr,
                    )
                    self.assertEqual(
                        fixture.session_exists(), expected_success
                    )
                    if expected_success:
                        fixture.wait_for_pid_file()
                finally:
                    fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_start_rejects_incomplete_pilot_gate_contracts(self):
        for label, checks in (
            ("empty", {}),
            ("missing", None),
            ("fake-only", {"fixture": True}),
        ):
            with self.subTest(label=label):
                fixture = ManagedTmuxFixture(ROOT)
                try:
                    fixture.replace_pilot_checks(checks)
                    checkout_status, checkout_tags = fixture.production_checkout_state()
                    self.assertEqual(checkout_status, "")
                    self.assertIn(
                        "experiment-qwen3-dashscope-full-v2",
                        checkout_tags,
                    )
                    completed = fixture.run_start(
                        allow_test_overrides=None,
                        test_runner=None,
                    )
                    self.assertNotEqual(
                        completed.returncode,
                        0,
                        completed.stdout + completed.stderr,
                    )
                    self.assertIn(
                        "Acceptance gate",
                        completed.stdout + completed.stderr,
                    )
                    self.assertFalse(fixture.session_exists())
                finally:
                    fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_start_rejects_incomplete_or_malformed_full_preflight(self):
        valid = {
            "status": "validated",
            "run_mode": "full",
            "selected_sample_count": 5658,
            "expected_full_sample_count": 5658,
            "contract_image_count": 5658,
            "api_call_limit": 50_000,
        }
        variants: tuple[tuple[str, object, bool], ...] = (
            (
                "minimal",
                {"status": "validated", "selected_sample_count": 5658},
                False,
            ),
            ("missing-run-mode", {k: v for k, v in valid.items() if k != "run_mode"}, False),
            ("wrong-run-mode", {**valid, "run_mode": "pilot"}, False),
            (
                "wrong-selected-count",
                {**valid, "selected_sample_count": 5657},
                False,
            ),
            (
                "float-selected-count",
                {**valid, "selected_sample_count": 5658.0},
                False,
            ),
            (
                "wrong-expected-count",
                {**valid, "expected_full_sample_count": 5657},
                False,
            ),
            (
                "wrong-contract-image-count",
                {**valid, "contract_image_count": 5657},
                False,
            ),
            ("wrong-api-limit", {**valid, "api_call_limit": 49_999}, False),
            ("float-api-limit", {**valid, "api_call_limit": 50_000.0}, False),
            ("malformed", "{not-json", True),
        )
        for label, payload, raw in variants:
            with self.subTest(label=label):
                fixture = ManagedTmuxFixture(ROOT)
                try:
                    fixture.replace_preflight(payload, raw=raw)
                    checkout_status, checkout_tags = fixture.production_checkout_state()
                    self.assertEqual(checkout_status, "")
                    self.assertIn(
                        "experiment-qwen3-dashscope-full-v2",
                        checkout_tags,
                    )
                    completed = fixture.run_start(
                        allow_test_overrides=None,
                        test_runner=None,
                    )
                    self.assertNotEqual(
                        completed.returncode,
                        0,
                        completed.stdout + completed.stderr,
                    )
                    expected_error = (
                        "JSONDecodeError"
                        if raw
                        else "Full preflight contract is invalid"
                    )
                    self.assertIn(
                        expected_error,
                        completed.stdout + completed.stderr,
                    )
                    self.assertFalse(fixture.session_exists())
                finally:
                    fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_production_start_clears_stale_tmux_test_environment(self):
        fixture = ManagedTmuxFixture(ROOT)
        unrelated_session = fixture.create_unrelated_session_with_stale_test_environment()
        try:
            fixture.replace_pilot_checks({})
            rejected = fixture.run_start(
                allow_test_overrides=None,
                test_runner=None,
            )
            self.assertNotEqual(
                rejected.returncode,
                0,
                rejected.stdout + rejected.stderr,
            )
            self.assertIn(
                "Acceptance gates are incomplete or failed",
                rejected.stdout + rejected.stderr,
            )
            self.assertFalse(fixture.session_exists())
            self.assertTrue(fixture.session_exists(unrelated_session))

            fixture.replace_pilot_checks(
                {
                    key: True
                    for key in fixture.PILOT10_BASELINE_GATE_KEYS
                }
            )
            launched = fixture.run_start(
                allow_test_overrides=None,
                test_runner=None,
            )
            self.assertEqual(
                launched.returncode,
                0,
                launched.stdout + launched.stderr,
            )
            self.assertEqual(
                fixture.wait_for_fake_start_or_pane_dead(),
                "pane-dead",
            )
            self.assertNotIn("FAKE_RUNNER_STARTED", fixture.run_log())
            self.assertTrue(fixture.session_exists(unrelated_session))
        finally:
            fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_immediate_runner_exit_preserves_exact_status_and_shell_audit(self):
        for exit_code in (2, 20, 21, 22, 23, 130, 143):
            with self.subTest(exit_code=exit_code):
                fixture = ManagedTmuxFixture(ROOT)
                try:
                    fixture.configure_immediate_fake_exit(exit_code)
                    fixture.start_fake_runner()
                    self.assertEqual(
                        fixture.wait_for_dead_pane_status(),
                        exit_code,
                    )
                    fixture.wait_for_runner_and_tee_exit()
                    self.assertTrue(
                        fixture.tee_process_drained(exit_code),
                        fixture.run_log(),
                    )

                    started_match = re.search(
                        r"FAKE_RUNNER_STARTED pid=([0-9]+)",
                        fixture.run_log(),
                    )
                    self.assertIsNotNone(started_match)
                    runner_pid = int(started_match.group(1))
                    exited_events = [
                        row
                        for row in fixture.events()
                        if row.get("event") == "shell_exited"
                    ]
                    self.assertEqual(len(exited_events), 1, fixture.events())
                    shell_exited = exited_events[0]
                    self.assertEqual(shell_exited["pid"], runner_pid)
                    self.assertEqual(shell_exited["exit_code"], exit_code)
                    self.assertEqual(shell_exited["forced"], False)
                    runner_attempt_id = shell_exited.get("runner_attempt_id")
                    self.assertIsInstance(runner_attempt_id, str)
                    self.assertTrue(runner_attempt_id)
                    for row in fixture.events():
                        if row.get("event") == "shell_started":
                            self.assertEqual(
                                row.get("runner_attempt_id"),
                                runner_attempt_id,
                            )
                    self.assertFalse(fixture.runner_identity_exists())
                    self.assertFalse(fixture.fake_runner_process_alive(runner_pid))
                finally:
                    fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_runtime_namespace_guard_rejects_legacy_symlinks_without_target_touch(self):
        for location in (
            "output-root",
            "full",
            "logs",
            "state",
            "runner",
            "run-log",
            "run-log-hardlink",
            "events",
            "preflight",
            "pilot",
            "pilot-metrics",
        ):
            with self.subTest(location=location):
                fixture = ManagedTmuxFixture(ROOT)
                try:
                    fixture.install_v1_runtime_symlink(location)
                    before = fixture.legacy_runtime_inventory()
                    completed = fixture.run_start()
                    self.assertNotEqual(
                        completed.returncode,
                        0,
                        completed.stdout + completed.stderr,
                    )
                    self.assertIn(
                        "Unsafe v2 runtime namespace",
                        completed.stdout + completed.stderr,
                    )
                    self.assertFalse(fixture.session_exists())
                    self.assertNotIn("FAKE_RUNNER_STARTED", fixture.run_log())
                    fixture.wait_for_runner_and_tee_exit()
                    self.assertEqual(
                        fixture.legacy_runtime_inventory(),
                        before,
                    )
                finally:
                    fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_runtime_namespace_guard_allows_controlled_outputs_parent_symlink(self):
        fixture = ManagedTmuxFixture(ROOT)
        try:
            fixture.install_controlled_outputs_parent_symlink()
            fixture.start_fake_runner()
            identity = fixture.wait_for_pid_file()
            self.assertTrue(fixture.fake_runner_process_alive(identity["pid"]))
        finally:
            fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_runner_identity_initialization_failure_reaps_started_child(self):
        fixture = ManagedTmuxFixture(ROOT)
        runner_pid = None
        try:
            fixture.force_identity_write_failure()
            fixture.start_fake_runner()
            runner_pid = fixture.wait_for_fake_runner_start()
            fixture.wait_for_pane_dead()
            self.assertFalse(fixture.fake_runner_process_alive(runner_pid))
        finally:
            if runner_pid is not None:
                fixture.terminate_fake_runner_if_alive(runner_pid)
            fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_pause_rejects_decoy_with_module_tokens_in_later_arguments(self):
        fixture = ManagedTmuxFixture(ROOT)
        decoy = None
        try:
            decoy = fixture.start_module_token_decoy()
            completed = fixture.run_pause()
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(fixture.decoy_received_signal())
        finally:
            if decoy is not None:
                fixture.terminate_decoy(decoy)
            fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_pause_rejects_same_module_runner_with_wrong_config(self):
        fixture = ManagedTmuxFixture(ROOT)
        decoy = None
        try:
            decoy = fixture.start_wrong_config_module_decoy()
            completed = fixture.run_pause()
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(fixture.wrong_config_decoy_received_signal())
        finally:
            if decoy is not None:
                fixture.terminate_decoy(decoy)
            fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_pause_timing_overrides_require_both_test_guards(self):
        fixture = ManagedTmuxFixture(ROOT)
        decoy = None
        reaper = None
        try:
            decoy = fixture.start_exact_production_module_decoy()
            reaper = threading.Thread(target=decoy.wait, daemon=True)
            reaper.start()
            completed = fixture.run_pause_with_environment(
                allow_test_overrides="1",
                test_runner=None,
                poll_seconds="not-a-number",
                max_seconds="not-a-number",
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertTrue(fixture.production_decoy_received_signal())
        finally:
            if decoy is not None:
                fixture.terminate_decoy(decoy)
            if reaper is not None:
                reaper.join(timeout=2)
            fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_tmux_fixture_cleanup_reaps_live_fake_runner(self):
        fixture = ManagedTmuxFixture(ROOT)
        runner_pid = None
        try:
            fixture.start_fake_runner()
            runner_pid = fixture.wait_for_pid_file()["pid"]
            fixture.cleanup()
            self.assertFalse(fixture.fake_runner_process_alive(runner_pid))
        finally:
            if runner_pid is not None:
                fixture.terminate_fake_runner_if_alive(runner_pid)
            fixture.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux is unavailable")
    def test_runner_identity_cleanup_is_same_fd_and_match_only(self):
        fixture = ManagedTmuxFixture(ROOT)
        identity_path = fixture.runner_identity_path
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        expected_pid = 4242
        expected_starttime = "123456"
        expected_attempt = "fixture-cleanup-attempt"

        def payload(*, attempt_id: str = expected_attempt) -> dict[str, object]:
            return {
                "schema": "scammark-runner-identity-v1",
                "pid": expected_pid,
                "starttime": expected_starttime,
                "attempt_id": attempt_id,
                "module": "scammark.run_experiment",
                "created_at_utc": "2026-07-13T00:00:00+00:00",
            }

        def invoke_cleanup() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    "bash",
                    "-c",
                    'source scripts/server/common.sh; '
                    'remove_runner_identity_if_matches "$1" "$2" "$3"',
                    "bash",
                    str(expected_pid),
                    expected_starttime,
                    expected_attempt,
                ],
                cwd=fixture.root,
                env=fixture.env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        try:
            symlink_target = fixture.root / "identity-symlink-target.json"
            symlink_target.write_text(
                json.dumps(payload()) + "\n", encoding="utf-8"
            )
            identity_path.symlink_to(symlink_target)
            self.assertEqual(invoke_cleanup().returncode, 0)
            self.assertTrue(identity_path.is_symlink())
            self.assertTrue(symlink_target.is_file())
            identity_path.unlink()

            identity_path.write_bytes(b"x" * 16_385)
            self.assertEqual(invoke_cleanup().returncode, 0)
            self.assertTrue(identity_path.is_file())
            identity_path.unlink()

            identity_path.write_text(
                json.dumps(payload(attempt_id="different-attempt")) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(invoke_cleanup().returncode, 0)
            self.assertTrue(identity_path.is_file())

            identity_path.write_text(
                json.dumps(payload()) + "\n", encoding="utf-8"
            )
            self.assertEqual(invoke_cleanup().returncode, 0)
            self.assertFalse(identity_path.exists())
        finally:
            fixture.cleanup()


if __name__ == "__main__":
    unittest.main()
