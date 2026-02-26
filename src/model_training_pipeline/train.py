from __future__ import annotations
import argparse

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from model_training_pipeline.data import load_dataset, split_dataset, split_indices

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def train_baseline_model(
    *, X_train: FloatArray, y_train: IntArray, seed: int
) -> LogisticRegression:
    """
    Trains a simple baseline classifier and returns the fitted model.
    """
    model = LogisticRegression(
        # Upper bound on optimizer steps; higher value reduces "did not converge" warnings.
        max_iter=1000,
        # Fixed seed for solver internals so repeated runs with same data stay consistent.
        random_state=seed,
        # Single-thread execution keeps behavior stable across different machines/setups.
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def predict_probabilities(
    *, model: LogisticRegression, X: FloatArray, debug: bool = False
) -> FloatArray:
    """
    Predict positive-class probability for each row in `X`.

    In binary logistic regression, `predict_proba` returns two columns:
    - column 0: P(class = 0)
    - column 1: P(class = 1)
    This helper returns only column 1 as a 1D array with shape `(n_samples,)`.
    """
    # Each row contains [P(class=0), P(class=1)] and sums to 1.0.
    proba_2d = model.predict_proba(X)
    # Keep only the positive-class probability, commonly used for thresholds/ROC-AUC.
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

    # 5) Get probability scores (and optionally print debug)
    test_scores = predict_probabilities(model=model, X=X_test, debug=args.debug)

    # 6) Show a simple metric just to confirm it works end-to-end
    test_acc = model.score(X_test, y_test)
    print("Test accuracy:", float(test_acc))
    print("Example test scores (first 5):", test_scores[:5])


if __name__ == "__main__":
    main()
