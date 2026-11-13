# Contributing

Thank you for helping improve ScamWeb and ScamMark.

## Development Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install optional evaluation or baseline dependencies only when needed:

```bash
python -m pip install -e ".[evaluation,baselines]"
```

## Change Guidelines

- Keep raw data, API keys, outputs, checkpoints, model weights, and provider
  responses out of Git.
- Preserve the frozen task contract unless the change explicitly introduces a
  new version.
- Keep prompts in English and preserve strict JSON schemas.
- Add focused tests for behavioral changes.
- Do not move an existing experiment tag or rewrite its provenance.
- Use Conventional Commits, for example `fix(evaluation): reject degenerate boxes`.

## Verification

Run the offline core suite before opening a pull request:

```bash
python -m unittest \
  tests.test_scammark \
  tests.test_scamweb_dataset \
  tests.test_scammark_evaluation \
  tests.test_paper_baselines
python -m compileall -q src tests
git diff --check
```

Changes to server orchestration or result archives must also run:

```bash
python -m unittest tests.test_server_workflow tests.test_result_contract
```

Paid API tests are never required for an ordinary pull request. If a provider
smoke test is necessary, use a separate private run, cap its sample count, and
publish only sanitized metrics.
