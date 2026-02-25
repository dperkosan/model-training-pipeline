# model-training-pipeline

## Description
ML foundations project that demonstrates:

- clean Python project structure (Poetry, typing, tests, linting)
- reproducible data splitting with a fixed seed
- leakage-safe train/validation/test workflow
- clear, testable pipeline components that can evolve into full training/evaluation CLI commands

## Setup

```bash
poetry install
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
