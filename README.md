# model-training-pipeline

## Description
ML foundations project that demonstrates:

- clean Python project structure (Poetry, typing, tests, linting)
- reproducible data splitting with a fixed seed
- leakage-safe train/validation/test workflow
- reproducible training CLI (`train.py`, `eval.py`) that logs metrics, saves model artifacts, and seeds randomness
- unit tests for data split + metric calculations

## Setup

```bash
poetry install
```

## Training

Run training with debug output:

```bash
poetry run train --debug
```

## Eval

Run evaluation for a saved run directory:

```bash
poetry run eval --run-dir runs/<YOUR_RUN_ID>
```

Run evaluation and save results to `eval_metrics.json`:

```bash
poetry run eval --run-dir runs/<YOUR_RUN_ID> --save
```

Run evaluation with a custom classification threshold:

```bash
poetry run eval --run-dir runs/<YOUR_RUN_ID> --threshold 0.7
```

Run evaluation with debug output:

```bash
poetry run eval --run-dir runs/<YOUR_RUN_ID> --debug
```

## Useful commands

Run tests:

```bash
poetry run pytest
```

Run type checks:

```bash
poetry run mypy
```

Run lint:

```bash
poetry run ruff check .
```
