from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias, cast
import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

IntArray = NDArray[np.int64]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


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


def save_json(*, path: Path, obj: JsonValue) -> None:
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


def load_json(*, path: Path) -> JsonValue:
    """
    Load JSON from disk and return it as Python data.
    """
    with path.open("r", encoding="utf-8") as f:
        return cast(JsonValue, json.load(f))


def load_model(*, path: Path) -> Pipeline:
    """
    Load a scikit-learn model from disk.
    """
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected Pipeline, got {type(model)}")
    return model
