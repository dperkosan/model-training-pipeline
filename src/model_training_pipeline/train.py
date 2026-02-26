from __future__ import annotations
import argparse

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from model_training_pipeline.artifacts import (
    make_run_dir,
    save_json,
    save_model,
    save_split_indices,
)
from model_training_pipeline.data import load_dataset, split_dataset, split_indices
from model_training_pipeline.metrics import evaluate_binary_classifier

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def train_baseline_model(
    *, X_train: FloatArray, y_train: IntArray, seed: int
) -> Pipeline:
    """
    Fit a baseline classifier and return the trained pipeline.
    """
    # First step: scale each feature to a comparable range (mean 0, std 1).
    # This usually makes logistic regression train faster and avoids convergence issues.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            # Let the optimizer take more steps so training can finish cleanly.
            max_iter=1000,
            # Keep results reproducible across runs.
            random_state=seed,
            # Stable default solver for this type of dataset.
            solver="lbfgs",
        ),
    )
    model.fit(X_train, y_train)
    return model


def predict_probabilities(
    *, model: Pipeline, X: FloatArray, debug: bool = False
) -> FloatArray:
    """
    Return the probability of class 1 for each row in `X`.

    In binary classification, `predict_proba` returns two columns:
    column 0 is P(class 0), column 1 is P(class 1).
    This helper returns only column 1 as a 1D array with shape `(n_samples,)`.
    """
    # One row per sample: [P(class=0), P(class=1)].
    proba_2d = model.predict_proba(X)
    # Keep only P(class=1), which is the usual "score" for binary tasks.
    proba_class1 = proba_2d[:, 1]

    if debug:
        print("\n--- DEBUG: predict_proba output (first 5 rows) ---")
        print("Each row is: [P(class0), P(class1)]")
        print(proba_2d[:5])
        print("\n--- DEBUG: class1 scores (first 5) ---")
        print(proba_class1[:5])
        print("min/max:", float(proba_class1.min()), float(proba_class1.max()))

    return np.asarray(proba_class1, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a baseline model (demo entry point)."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--debug", action="store_true", help="Print predict_proba details.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Load full dataset (X matrix, y labels)
    X, y = load_dataset()
    print("Loaded dataset:", X.shape, y.shape)

    # 2) Split row indices deterministically
    idx_train, idx_val, idx_test = split_indices(
        n_samples=len(y),
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    # 3) Slice into train/val/test arrays
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X=X,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    print("Splits:", X_train.shape, X_val.shape, X_test.shape)

    # 4) Train model
    model = train_baseline_model(X_train=X_train, y_train=y_train, seed=args.seed)
    print("Model trained.")

    # 5) Scores + metrics
    val_scores = predict_probabilities(model=model, X=X_val, debug=args.debug)
    test_scores = predict_probabilities(model=model, X=X_test, debug=False)

    val_metrics = evaluate_binary_classifier(
        y_true=y_val, y_score=val_scores, threshold=0.5
    )
    test_metrics = evaluate_binary_classifier(
        y_true=y_test, y_score=test_scores, threshold=0.5
    )

    print("VAL metrics:", val_metrics)
    print("TEST metrics:", test_metrics)

    # 6) Create run dir + save artifacts
    paths = make_run_dir(seed=args.seed)

    # Save model
    save_model(path=paths.model_path, model=model)

    # Save metrics
    save_json(
        path=paths.metrics_path,
        obj={
            "val": val_metrics,
            "test": test_metrics,
        },
    )

    # Save split indices (so eval can reconstruct test set exactly)
    save_split_indices(
        path=paths.splits_path,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )

    # Save config used (so the run is reproducible)
    save_json(
        path=paths.config_path,
        obj={
            "seed": args.seed,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "threshold": 0.5,
            "model": {
                "type": "Pipeline",
                "steps": [
                    {
                        "name": "standardscaler",
                        "type": "StandardScaler",
                    },
                    {
                        "name": "logisticregression",
                        "type": "LogisticRegression",
                        "solver": "lbfgs",
                        "max_iter": 1000,
                        "random_state": args.seed,
                    },
                ],
            },
        },
    )

    print("Saved run artifacts to:", str(paths.run_dir))


if __name__ == "__main__":
    main()
