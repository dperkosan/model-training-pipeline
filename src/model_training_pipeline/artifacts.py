from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    model_path: Path
    metrics_path: Path
    config_path: Path
    splits_path: Path


def make_run_dir(*, root: str = "runs", seed: int) -> RunPaths:
    """
    Creates a unique run directory and returns standard file paths for artifacts.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(root) / f"{timestamp}_seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        run_dir=run_dir,
        model_path=run_dir / "model.joblib",
        metrics_path=run_dir / "metrics.json",
        config_path=run_dir / "config.json",
        splits_path=run_dir / "splits.json",
    )


def save_json(*, path: Path, obj: dict[str, Any]) -> None:
    """
    Save a dict as pretty JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def save_split_indices(
    *,
    path: Path,
    idx_train: IntArray,
    idx_val: IntArray,
    idx_test: IntArray,
) -> None:
    """
    Save split indices to JSON (as plain Python lists).
    """
    obj = {
        "idx_train": idx_train.tolist(),
        "idx_val": idx_val.tolist(),
        "idx_test": idx_test.tolist(),
    }
    save_json(path=path, obj=obj)


def save_model(*, path: Path, model: Pipeline) -> None:
    """
    Save a scikit-learn model to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
