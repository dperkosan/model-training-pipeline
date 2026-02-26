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
