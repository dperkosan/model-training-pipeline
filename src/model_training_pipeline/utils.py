from __future__ import annotations

import logging
from pathlib import Path


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
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
