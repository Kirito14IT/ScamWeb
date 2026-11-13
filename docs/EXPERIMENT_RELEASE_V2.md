# ScamMark Qwen3 DashScope Full Experiment v2

This document defines the immutable contract for tag
`experiment-qwen3-dashscope-full-v2`. Version 2 preserves the v1 model routing,
prompts, cascade, image policy, and metrics. It adds observability, bounded
recovery, safe pausing, result validation, and resumable delivery.

## Identifiers

| Item | Value |
| --- | --- |
| Repository | `nlixsys/ScamWeb-V` |
| Tag | `experiment-qwen3-dashscope-full-v2` |
| Experiment | `scammark-qwen3-dashscope-server-full-v2` |
| Configuration | `configs/qwen3_dashscope_server_full_v2.toml` |
| Output root | `outputs/scammark_qwen3_dashscope_server_full_v2` |
| Tmux session | `scammark-full-v2` |
| Results branch | `results/qwen3-dashscope-full-v2` |
| Inference asset | `scammark-qwen3-dashscope-full-v2-inference.tar.gz` |
| Final asset | `scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz` |

The tag fixes code and configuration. Do not move the tag, modify a tagged
checkout, or combine checkpoints from another experiment namespace.

## Frozen Benchmark Contract

- 5,658 total pages
- 3,416 scam pages and 2,242 benign pages
- 25 canonical, single-label scam subtypes
- 5,489 Gold evidence regions
- English staged prompts
- Seed `2027`
- Sequential inference
- Maximum 50,000 physical API calls, including durable reservations

Gold labels, descriptions, and boxes are never included in model requests.
They are available only to offline evaluation and explicitly marked forced
diagnostics.

## Stages and Routing

| Stage | Provider and model | Image input | Output limit |
| --- | --- | --- | ---: |
| `url` | Crazyrouter, `gpt-4o-mini` | None | 512 |
| `text` | Crazyrouter, `gpt-4o-mini` | None | 768 |
| `visual` | Alibaba Cloud Model Studio, `qwen3-vl-flash` | Overview and tiles | 1,024 |
| `cross` | Alibaba Cloud Model Studio, `qwen3-vl-flash` | Overview | 1,024 |
| `binary` | Alibaba Cloud Model Studio, `qwen3-vl-flash` | Overview | 256 |
| `type` | Alibaba Cloud Model Studio, `qwen3-vl-flash` | None | 256 |
| `grounding` | Alibaba Cloud Model Studio, `qwen3-vl-flash` | Overview and tiles | 2,048 |

`url` and `text` have 8,192-token input limits. Multimodal stages have
4,096-token textual limits, an 8,192 visual-token limit, and a 12,288 combined
prompt-token limit. The experiment uses `temperature=0`, disables model
thinking, and requires schema-constrained JSON.

The normal cascade stops after a benign binary prediction and emits
`scam_type=none` and no boxes. Gold scam pages rejected by the binary stage are
run downstream only in the separate forced-diagnostic path. Those calls do not
replace the end-to-end prediction.

## Image Policy

- 1,048,576-pixel overview
- At most seven vertical tiles with 10 percent overlap
- 8,388,608 total source pixels
- 7,680-pixel maximum long side
- 10,000,000-byte maximum Base64 payload per image
- 2,500,000-byte maximum encoded contact sheet
- 50,000,000-byte maximum request body

The model returns a tile identifier and normalized `bbox_1000` coordinates.
The evaluator maps them back to the original screenshot's pixel-space `xyxy`
coordinates before scoring.

## Reliability Contract

| Setting | Value |
| --- | ---: |
| Connection timeout | 20 seconds |
| Text read timeout | 120 seconds |
| Vision read timeout | 300 seconds |
| Immediate attempts per stage | 4 |
| Backoff | 2, 4, and 8 seconds with full jitter |
| Cross-resume attempt ceiling | 6 |
| Consecutive backend-error breaker | 5 samples |

TLS EOF, connection loss, HTTP 408/429, HTTP 500/502/503/504, and empty
responses are retryable. HTTP 400/401/403/404 fail immediately. Invalid JSON
receives one schema-repair attempt. `Retry-After` is respected.

Each completed stage is written atomically. Resume validates data, prompt,
configuration, and model-routing hashes before filling only missing stages.
Every physical call records sanitized routing, latency, attempt, token usage,
finish reason, and error class. API keys, prompts, images, raw responses, full
endpoints, and checkpoints are never published.

## Frozen Input Asset

The private tagged release may provide
`scamweb-v1.0-frozen-inputs.tar.gz` with:

- Size: `21,802,246` bytes
- SHA256: `8d0894a043913f6e049cfc4b212a310cb4fd79a810ff5651372c802f8ab2eb61`
- Manifest SHA256: `81904995efcae0bacdee1a2e31d0da9d92a81adb099c4a9d0b5b01da885e0d1c`
- Evidence SHA256: `012ec051c0518900be2246ed5ef0a07185768bf8f451e378056f75289fa160ff`

The archive has an exact eight-file allowlist:

```text
dataset_frozen_v1/boxes.jsonl
dataset_frozen_v1/dataset_version.json
dataset_frozen_v1/mhtml_audit.jsonl
dataset_frozen_v1/mhtml_recapture.jsonl
dataset_frozen_v1/sample_ids.txt
dataset_frozen_v1/scamweb_v_manifest.jsonl
dataset_frozen_v1/verification_report.json
dataset_frozen_v1/web_evidence.jsonl
```

It contains derived text and metadata only. It excludes raw PNG, MHTML, Excel,
recaptured pages, and image caches. The complete raw dataset remains external
and must be mounted at `SCAMWEB_DATASET_ROOT`.

## Evaluation Contract

Binary reporting includes accuracy, scam-positive precision/recall/F1,
TP/FP/TN/FN, and bootstrap confidence intervals. The 25-way subtype task uses
accuracy and macro precision/recall/F1; `none` is an end-to-end error and is not
part of the 25-class macro average.

Grounding uses class-agnostic maximum-total-IoU one-to-one matching. It reports
page-averaged mIoU, global precision/recall/F1 at IoU 0.5, and page recall at
IoU 0.3 and 0.5. Missed Gold boxes contribute zero IoU; benign predicted boxes
are false positives. mAP is excluded because generated boxes have no calibrated
ranking confidence.

Rationale metrics are computed only for IoU-at-least-0.5 matched regions:
ROUGE-L F1, multilingual BERTScore F1, and grounded-rationale coverage. Human
ratings remain `not_provided` unless actual blinded annotations are supplied.

BERTScore is a post-inference metric. It uses `bert-score==0.3.13`,
`transformers==4.57.6`, and `FacebookAI/xlm-roberta-large` revision
`c23d21b0620b635a76227c604d44e43a9f0ee389`, layer 17, batch size 4, no IDF,
and no baseline rescaling. The inference package records `not_requested`; the
final evaluation must record `computed`.

## Acceptance Gates

Paid pilot and full runs require zero final backend errors, zero schema errors,
complete stage sets, complete usage attribution, valid image/request budgets,
correct routing, no truncated completion, clean metric inputs, complete forced
diagnostics, and compliance with the physical call cap.

The 5,658-page run additionally requires:

- first-attempt failure rate at most 5 percent;
- average attempts at most 1.25;
- support for all 25 Gold subtypes;
- at least one end-to-end IoU-at-least-0.5 match;
- exact 5,658-row manifest and prediction ID alignment;
- deferred BERTScore during inference and computed BERTScore in the final package.

No numerical paper result is hard-coded in this repository. Generated metrics
must pass the result contract before publication.
