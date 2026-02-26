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


def load_split_indices(*, path: Path) -> tuple[IntArray, IntArray, IntArray]:
    """
    Load split indices from JSON and return them as NumPy int64 arrays.
    """
    obj = load_json(path=path)
    if not isinstance(obj, dict):
        raise TypeError("splits.json must be a JSON object")

    def _get_list(name: str) -> list[int]:
        value = obj.get(name)
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a list[int]")

        out: list[int] = []
        for x in value:
            if not isinstance(x, int):
                raise TypeError(f"{name} must be a list[int]")
            out.append(x)
        return out

    idx_train = np.asarray(_get_list("idx_train"), dtype=np.int64)
    idx_val = np.asarray(_get_list("idx_val"), dtype=np.int64)
    idx_test = np.asarray(_get_list("idx_test"), dtype=np.int64)

    return idx_train, idx_val, idx_test
