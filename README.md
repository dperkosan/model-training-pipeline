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

Run training:

```bash
poetry run train --log-level INFO
```

Use `--log-level DEBUG` to include detailed probability debug logs.

Optional OpenTelemetry tracing for training:

Console exporter (local debug):

```bash
OTEL_TRACES_EXPORTER=console poetry run train --log-level DEBUG
```

OTLP exporter (collector at localhost:4318):

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 poetry run train --log-level DEBUG
```

## Eval

Run evaluation for a saved run directory:

```bash
poetry run eval --run-dir runs/<YOUR_RUN_ID> --log-level INFO
```

Run evaluation and save results to `eval_metrics.json`:

```bash
poetry run eval --run-dir runs/<YOUR_RUN_ID> --save --log-level INFO
```

Run evaluation with a custom classification threshold:

```bash
poetry run eval --run-dir runs/<YOUR_RUN_ID> --threshold 0.7 --log-level INFO
```

Run evaluation with detailed debug logs:

```bash
poetry run eval --run-dir runs/<YOUR_RUN_ID> --log-level DEBUG
```

Optional OpenTelemetry tracing for eval:

Console exporter (local debug):

```bash
OTEL_TRACES_EXPORTER=console poetry run eval --run-dir runs/<YOUR_RUN_ID> --log-level DEBUG
```

OTLP exporter (collector at localhost:4318):

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 poetry run eval --run-dir runs/<YOUR_RUN_ID> --log-level DEBUG
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
