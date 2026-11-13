# ScamWeb-V Dataset Contract

## Scope

ScamWeb-V contains 5,658 persistently captured webpages: 3,416 scam pages and
2,242 benign pages. The scam subset has 25 canonical, single-label subtypes and
5,489 human-annotated evidence regions. Every benchmark page has a screenshot;
the raw collection also retains an MHTML capture.

The raw collection is approximately 40 GB and is not stored in this Git
repository or its GitHub Release assets. The small files under `metadata/`
contain aggregate statistics, schema mappings, class support, and fixed split
membership only.

## Public Dataset

Download and schema documentation: [https://huggingface.co/datasets/kirco567/ScamWeb-Dataset](https://huggingface.co/datasets/kirco567/ScamWeb-Dataset). The `samples` and `artifacts`
configurations use the `full` split and store original payloads as bytes in
Parquet. The native tools described below expect the restored raw directory
layout, so a Parquet snapshot directory is not directly usable as
`SCAMWEB_DATASET_ROOT`. Pin the dataset revision for reproducible experiments.

## Source Authority

| Field | Authoritative source | Rule |
| --- | --- | --- |
| Binary label | Dataset partition | Gold JSON presence defines scam; the benign partition defines benign. |
| Scam subtype | Gold JSON `label` | Only deterministic English alias normalization is applied. |
| Page rationale | Gold JSON `description` | Workbook descriptions never override Gold JSON. |
| Language | Gold JSON `language` | Retained as annotation metadata. |
| Regions and reasons | Gold JSON `annotations` | Center/size coordinates are converted to pixel-space `xyxy`. |
| URL and title | Workbook Gold/benign sheets | Used for indexing and model input, not annotation truth. |
| MHTML availability | Frozen audit | Only `valid` and `valid_redirect` evidence is exposed to ScamMark. |

The original JSON, PNG, MHTML, and workbook files are read-only inputs. Dataset
builders write derived artifacts under `outputs/` and never modify source data.

## Canonicalization

The 28 observed raw label strings map deterministically to 25 canonical English
labels. The normalization covers formatting aliases for Education, Government,
Job, and Lottery labels, plus the confirmed malformed phishing string. Unknown
model predictions are invalid; they are not mapped by keyword similarity.

The complete taxonomy and support are available in
[`../metadata/taxonomy_support.csv`](../metadata/taxonomy_support.csv), and the
normalization implementation is in
[`../src/scammark/taxonomy.py`](../src/scammark/taxonomy.py).

## MHTML Audit Policy

The frozen input policy records one of four states:

| Status | Count | Model input policy |
| --- | ---: | --- |
| `valid` | 5,298 | Structured MHTML evidence is available. |
| `valid_redirect` | 184 | Evidence is available; original and final URLs are retained. |
| `mhtml_mismatch` | 161 | MHTML is withheld. URL, title, and screenshot remain available. |
| `missing` | 15 | MHTML is withheld. URL, title, and screenshot remain available. |

No page is removed solely because MHTML is unavailable. Nineteen frozen records
carry `gold_input_conflict=true`; they remain in the benchmark and must be
reported as a data-quality limitation. Every model must use the same status and
input policy.

## Expected Raw Layout

```text
scamweb_dataset/
|-- 总表_0325.xlsx
|-- dataset_scam_gold/
|   |-- 人工标注结果_scam/*.json
|   |-- 截图_scam/*.png
|   `-- mhtml_scam/*.mhtml
`-- dataset_benign/
    |-- 截图_benign/*.png
    `-- mhtml_benign/*.mhtml
```

Set the root through `SCAMWEB_DATASET_ROOT`; do not commit a machine-specific
path file.

## Build and Validate

```bash
export SCAMWEB_DATASET_ROOT=/data/scamweb_dataset
python -m data.scamweb_dataset \
  --output-dir outputs/dataset \
  --reports-dir reports
```

The builder fails if the authoritative contract drifts from 3,416 Gold JSON
records, 2,242 benign pages, 25 canonical subtypes, or 5,489 Gold boxes.

The formal run uses a separately audited `outputs/dataset_frozen_v1` directory.
Verify it with:

```bash
python -m scammark.frozen_assets \
  --root outputs/dataset_frozen_v1 \
  --output outputs/dataset_frozen_v1/verification_report.json
```

## Fixed Supervised Split

`metadata/fixed_split_v1.json` fixes a seed-2027 page split with 3,961 training,
849 validation, and 848 test records. Use this exact file for supervised
baselines. ScamMark is also evaluated zero-shot over all 5,658 pages; results
from the fixed test set and the all-page zero-shot evaluation must be labeled
separately.

## Safety and Distribution

Collected webpages may include phishing forms, malware delivery prompts,
financial deception, gambling, explicit imagery, and other harmful content.
MHTML can contain active markup and remote references. Keep raw files outside
the source tree, disable automatic browser execution, and process them in an
isolated environment.

Code licensing does not imply permission to redistribute third-party webpage
content. Raw-data access and publication require a separate legal, ethical, and
safety review.
