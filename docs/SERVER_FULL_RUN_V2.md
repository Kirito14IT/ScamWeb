# Full Experiment Runbook for ARM64 DGX

This runbook launches the immutable 5,658-page ScamMark v2 experiment on an
ARM64 NVIDIA DGX-class host. It assumes the raw dataset is already present on
the server and that the tagged repository and sanitized frozen-input asset are
available to the authenticated GitHub account.

## 1. Prerequisites

- Ubuntu 24.04 on `aarch64`
- Python 3.12
- Git, curl, tar, `sha256sum`, and `tmux`
- NVIDIA driver and CUDA-compatible GPU
- GitHub CLI authenticated for the private repository
- At least 250 GB free for environments, caches, checkpoints, and results
- Stable outbound access to GitHub, Crazyrouter, Alibaba Cloud Model Studio,
  PyPI, and the pinned Hugging Face model files

The raw dataset must remain outside the repository. This document uses
`/data/scamweb_dataset`; another absolute path is valid through `.env`.

## 2. Clone the Tagged Repository

```bash
cd /opt
gh repo clone nlixsys/ScamWeb-V /opt/scamweb-v
cd /opt/scamweb-v
git fetch --tags origin
git checkout experiment-qwen3-dashscope-full-v2
git status --short
```

`git status --short` must be empty. If a large clone is interrupted, remove only
the incomplete target after checking its absolute path, then retry with:

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/nlixsys/ScamWeb-V.git /opt/scamweb-v
cd /opt/scamweb-v
git fetch --tags origin
git checkout experiment-qwen3-dashscope-full-v2
```

Do not copy a working tree from another machine or continue from a same-named
local tag whose commit does not match the remote peeled tag.

## 3. Configure Secrets and Data

Transfer `.env` through a secure channel, then restrict it:

```bash
cd /opt/scamweb-v
chmod 600 .env
git check-ignore -v .env
```

Minimum variables:

```dotenv
SCAMMARK_LLM_API_KEY=...
SCAMMARK_LLM_BASE_URL=https://cn.crazyrouter.com/v1
SCAMMARK_LLM_MODEL=gpt-4o-mini

SCAMMARK_VLM_API_KEY=...
SCAMMARK_VLM_BASE_URL=https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
SCAMMARK_VLM_MODEL=qwen3-vl-flash

SCAMWEB_DATASET_ROOT=/data/scamweb_dataset
HF_HOME=/opt/scamweb-v/models/huggingface
```

Never print `.env`. Verify only file counts and paths:

```bash
find /data/scamweb_dataset -type f -iname '*.png' | wc -l
find /data/scamweb_dataset -type f -iname '*.mhtml' | wc -l
find /data/scamweb_dataset -type f -iname '*.json' | wc -l
```

Expected counts are 5,658 PNG, 5,658 MHTML, and 3,416 Gold JSON files.

## 4. Private Frozen-Input Release Asset

The release attached to tag `experiment-qwen3-dashscope-full-v2` must contain:

```text
scamweb-v1.0-frozen-inputs.tar.gz
scamweb-v1.0-frozen-inputs.tar.gz.sha256
```

The package is sanitized and contains derived JSON/JSONL/TXT only. Raw PNG,
MHTML, Excel, recapture files, and image caches must not be uploaded. The server
preparation script validates archive size, SHA256, member allowlist, internal
hashes, row counts, and source dataset counts before installation.

## 5. Prepare and Validate

```bash
cd /opt/scamweb-v
scripts/server/prepare_server.sh
```

Preparation performs the following without paid inference:

1. Verifies the exact remote tag and a clean worktree.
2. Creates `.venv-server` and installs the hash-locked ARM64 dependencies.
3. Downloads and verifies the sanitized frozen-input asset.
4. Verifies the external raw dataset inventory.
5. Runs the complete fake-client unit suite with provider variables removed.
6. Runs `--full --validate-only` for exactly 5,658 inputs.

Do not start paid inference if any gate fails.

Run the mandatory paid 10-page smoke before the full job:

```bash
.venv-server/bin/python -m scammark.run_experiment \
  --config configs/qwen3_dashscope_server_full_v2.toml \
  --pilot 10 --resume
```

Inspect `pilot_10/metrics.json` and require every acceptance gate to pass. Do
not treat a partially completed smoke, provider refusal, missing usage record,
or schema repair failure as permission to start the full run.

## 6. Start and Monitor

```bash
scripts/server/start_full.sh
scripts/server/status.sh
tmux attach -t scammark-full-v2
```

Detach from `tmux` with `Ctrl-b`, then `d`. Closing the remote desktop session
does not stop the run while the server and `tmux` remain active.

Status combines the tmux pane, runner process identity, checkpoint state,
durable API ledger, sanitized events, and metric gates. Do not infer safety from
the pane state alone.

Useful read-only checks:

```bash
scripts/server/status.sh
tail -n 80 outputs/scammark_qwen3_dashscope_server_full_v2/logs/full_run.log
```

Logs are sanitized but should still remain in the private experiment
environment until reviewed.

## 7. Pause and Resume

Request a checkpoint-aware stop:

```bash
scripts/server/pause_full.sh
scripts/server/status.sh
```

Resume only after status confirms that no provider request remains orphaned:

```bash
scripts/server/start_full.sh
```

If the five-sample backend-error breaker opens, diagnose the provider or
configuration first, then acknowledge explicitly:

```bash
scripts/server/start_full.sh --ack-breaker
```

Never delete checkpoints, edit the configuration, or bypass hash validation to
force a resume. A method or input change requires a new experiment version.

## 8. Completion and BERTScore

After inference and initial delivery succeed, compute the deferred rationale
metric:

```bash
scripts/server/finalize_bertscore.sh
scripts/server/status.sh
```

The finalizer downloads the pinned XLM-R revision if it is not already cached,
computes BERTScore on IoU-at-least-0.5 matched explanations, validates the final
metric contract, and creates the final evaluation package. It does not train or
fine-tune a model.

## 9. Delivery and Recovery

The normal runner packages inference results, uploads private Release assets,
and publishes the sanitized results branch. If inference is complete but a
delivery step fails, recover only that step:

```bash
scripts/server/recover_delivery.sh package
scripts/server/recover_delivery.sh upload
scripts/server/recover_delivery.sh publish
```

For the post-BERTScore package:

```bash
scripts/server/recover_delivery.sh --mode final package
scripts/server/recover_delivery.sh --mode final upload
scripts/server/recover_delivery.sh --mode final publish
```

Recovery scripts validate the same tag, hashes, package allowlists, metrics,
and provenance. They do not rerun successful inference stages.

## 10. Retrieve on Windows

From a clean Windows checkout of the same tag:

```powershell
gh auth status
.\scripts\windows\download_full_results_v2.ps1 `
  -Repository 'nlixsys/ScamWeb-V' `
  -Tag 'experiment-qwen3-dashscope-full-v2' `
  -Destination 'outputs/server_results/qwen3-dashscope-full-v2'
```

The downloader verifies checksums, exact tar allowlists, provenance, 5,658
unique matching IDs, acceptance gates, and final BERTScore before atomically
replacing the destination. Generated results remain ignored by Git.

## 11. Windows Offline Gate

Before paid smoke, validate the Windows downloader without downloading formal
results. A missing `pwsh` is a failed validation gate, not a pass.

First parse the script:

```powershell
pwsh -NoProfile -Command `
  '$errors=$null; [System.Management.Automation.Language.Parser]::ParseFile(
    "scripts/windows/download_full_results_v2.ps1", [ref]$null, [ref]$errors
  ) | Out-Null; if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }'
```

Run its fixture-backed tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Join-Path $PWD 'src')
python -m unittest -v -k windows_v2_downloader tests.test_server_workflow
if ($LASTEXITCODE -ne 0) { throw 'Windows downloader tests failed' }
```

Create an explicit 5,658-row offline fixture. Do not assume a `D:` path or a
previously downloaded asset exists:

```powershell
$fixtureDirectory = Join-Path $env:TEMP `
  ("scammark-full-v2-fixture-" + [guid]::NewGuid().ToString('N'))
$env:SCAMMARK_WINDOWS_FIXTURE_DIRECTORY = $fixtureDirectory
@'
import os
import shutil
from pathlib import Path
from tests.result_fixture import CompleteResultFixture

destination = Path(os.environ["SCAMMARK_WINDOWS_FIXTURE_DIRECTORY"])
with CompleteResultFixture(sample_count=5658) as fixture:
    source = fixture.build_archives()
    shutil.copytree(source, destination)
print(destination.resolve())
'@ | python -

.\scripts\windows\download_full_results_v2.ps1 `
  -ArchiveDirectory $fixtureDirectory `
  -VerifyOnly `
  -SkipResultsBranchFetch
```

All three gates must succeed before paid smoke.

## 12. Exit Codes

The retained runner and recovery scripts use stable exit codes:

| Code | Meaning | Required action |
| ---: | --- | --- |
| `0` | Current stage completed | Check `status.sh`; finalize BERTScore if pending. |
| `1` | Unexpected runner or shell failure | Inspect sanitized status and logs before resuming. |
| `2` | Acceptance gate failed | Diagnose metrics; do not auto-resume or deliver. |
| `20` | Authentication, billing, or explicit quota stop | Repair account state, then restart explicitly. |
| `21` | Durable physical-call cap exhausted | Do not resume this experiment version. |
| `22` | Backend-error breaker opened | Diagnose, then use `--ack-breaker`. |
| `23` | Input/configuration/checkpoint drift | Create a new version; do not force resume. |
| `30` | Packaging failed | Run the matching recovery `package` action. |
| `31` | Release upload failed | Run the matching recovery `upload` action. |
| `32` | Results-branch publication failed | Run the matching recovery `publish` action. |
| `130` | SIGINT | Resume only after status confirms a graceful checkpointed stop. |
| `143` | SIGTERM | Resume only after status confirms a graceful checkpointed stop. |

## 13. Cost and ETA Boundary

`status.sh` may show approximate `remaining` cost in provider-native `USD` and
`CNY` units. It does not convert currencies and does not claim accrued spend or
a final invoice total. Estimates are based on completed stages and are unstable
during the first five full samples. The provider invoice is authoritative.

## 14. Stop Conditions

Stop and investigate when any of the following occurs:

- HTTP 401/403, invalid workspace endpoint, disabled model, or exhausted credit;
- physical call cap reached;
- five consecutive samples end with backend errors;
- input, prompt, configuration, model-routing, or checkpoint hash drift;
- missing or duplicated prediction IDs;
- failed acceptance gate or unsafe archive member;
- repeated refusal or provider data-inspection errors on explicit pages.

Do not silently substitute GPT-4o for failed Qwen3 samples in the formal run.
At most three isolated GPT-4o calls may be used for diagnosis, and those results
must not enter the Qwen3 benchmark output.

## 15. Data and Secret Boundary

Never upload or commit:

- the approximately 40 GB raw dataset;
- raw PNG, MHTML, Excel, or recaptured pages;
- `.env` or API keys;
- image/base64 request bodies or raw model responses;
- checkpoints, provider caches, or model weights;
- unsanitized ledgers or endpoints containing workspace identifiers.

Only the audited source snapshot, sanitized frozen metadata package, sanitized
result packages, checksums, and result-contract outputs may be published.
