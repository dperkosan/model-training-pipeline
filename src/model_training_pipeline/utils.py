from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from types import TracebackType
from typing import Any, Literal


def setup_logging(
    *, level: str = "INFO", log_file: Path | None = None
) -> logging.Logger:
    """
    Configure logging for the app:
    - Always logs to console
    - Optionally logs to a file (e.g., runs/<run_id>/train.log)

    Returns the "app" logger you should use in modules.
    """
    logger_name = "model_training_pipeline"
    logger = logging.getLogger(logger_name)

    # Convert "INFO"/"DEBUG" -> numeric level
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logger.setLevel(numeric_level)

    # Important: avoid duplicate handlers if you run setup twice
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "trace_id=%(trace_id)s span_id=%(span_id)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    trace_filter = TraceContextFilter()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    ch.addFilter(trace_filter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        fh.addFilter(trace_filter)
        logger.addHandler(fh)

    return logger


_TRACE_API: Any | None = None
_TRACING_INITIALIZED = False


class _NoopSpan:
    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False

    def set_attribute(self, _name: str, _value: object) -> None:
        return None


class _NoopTracer:
    def start_as_current_span(self, _name: str) -> _NoopSpan:
        return _NoopSpan()


_NOOP_TRACER = _NoopTracer()


class TraceContextFilter(logging.Filter):
    """
    Adds trace/span IDs to every log record when an active OTEL span exists.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        setattr(record, "trace_id", "-")
        setattr(record, "span_id", "-")

        if _TRACE_API is None:
            return True

        span = _TRACE_API.get_current_span()
        ctx = span.get_span_context()
        if getattr(ctx, "is_valid", False):
            setattr(record, "trace_id", f"{ctx.trace_id:032x}")
            setattr(record, "span_id", f"{ctx.span_id:016x}")
        return True


def setup_tracing(
    *,
    service_name: str = "model-training-pipeline",
    logger: logging.Logger | None = None,
) -> bool:
    """
    Initialize OpenTelemetry tracing if OTEL packages are available.

    Export behavior:
    - `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`:
      configure OTLP HTTP exporter
    - `OTEL_TRACES_EXPORTER=console`: export spans to console
    """

    global _TRACE_API
    global _TRACING_INITIALIZED

    if _TRACING_INITIALIZED:
        return True

    try:
        trace_api = importlib.import_module("opentelemetry.trace")
        resources_module = importlib.import_module("opentelemetry.sdk.resources")
        trace_sdk_module = importlib.import_module("opentelemetry.sdk.trace")
        trace_export_module = importlib.import_module("opentelemetry.sdk.trace.export")
    except ModuleNotFoundError:
        if logger is not None:
            logger.info("OpenTelemetry packages not installed; tracing is disabled.")
        return False

    resource = resources_module.Resource.create({"service.name": service_name})
    provider = trace_sdk_module.TracerProvider(resource=resource)

    exporter_configured = False

    use_otlp = bool(
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if use_otlp:
        try:
            otlp_module = importlib.import_module(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter"
            )
            otlp_exporter = otlp_module.OTLPSpanExporter()
            provider.add_span_processor(
                trace_export_module.BatchSpanProcessor(otlp_exporter)
            )
            exporter_configured = True
        except ModuleNotFoundError:
            if logger is not None:
                logger.warning(
                    "OTLP endpoint is configured but OTLP exporter package is missing."
                )

    if os.getenv("OTEL_TRACES_EXPORTER", "").lower() == "console":
        provider.add_span_processor(
            trace_export_module.SimpleSpanProcessor(
                trace_export_module.ConsoleSpanExporter()
            )
        )
        exporter_configured = True

    trace_api.set_tracer_provider(provider)
    _TRACE_API = trace_api
    _TRACING_INITIALIZED = True

    if logger is not None:
        if exporter_configured:
            logger.info("OpenTelemetry tracing enabled.")
        else:
            logger.info(
                "OpenTelemetry tracing enabled (no exporter configured; spans stay local)."
            )

    return True


def get_tracer(name: str = "model_training_pipeline") -> Any:
    """
    Return an OTEL tracer when available, otherwise a no-op tracer.
    """
    if _TRACE_API is None:
        return _NOOP_TRACER
    return _TRACE_API.get_tracer(name)
