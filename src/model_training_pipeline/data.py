from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer

# Explicit aliases keep function signatures readable and consistent.
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def load_dataset() -> tuple[FloatArray, IntArray]:
    """
    Load the built-in breast cancer classification dataset from scikit-learn.

    This helper returns `X` and `y` directly (same as `return_X_y=True`) so the
    training pipeline can use plain NumPy arrays without dataset wrapper objects.

    Returns:
      X: Feature matrix with shape (n_samples, n_features), dtype float64.
      y: Binary labels with shape (n_samples,), dtype int64 (0 or 1).
    """
    # `return_X_y=True` gives a typed tuple and avoids union-attribute issues in mypy.
    X_raw, y_raw = load_breast_cancer(return_X_y=True)

    # Convert to plain NumPy arrays with stable dtypes (helps typing + consistency)
    X = np.asarray(X_raw, dtype=np.float64)
    y = np.asarray(y_raw, dtype=np.int64)

    # Returning NumPy arrays keeps the training pipeline framework-agnostic.
    return X, y


def split_indices(
    *, n_samples: int, seed: int, test_size: float, val_size: float
) -> tuple[IntArray, IntArray, IntArray]:
    """
    Create a reproducible (seed-based) split of dataset row indices into
    train, validation, and test sets.

    It returns row indices (not data slices), so callers can apply the same
    non-overlapping split to both X and y and avoid leakage between splits.
    `test_size` and `val_size` are fractions of the full dataset.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be between 0 and 1")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    # Seeded RNG ensures the same split order is produced for the same seed.
    rng = np.random.default_rng(seed)

    # Full list of row positions: [0, 1, ..., n_samples-1].
    idx_all = np.arange(n_samples, dtype=np.int64)
    # Shuffle indices in-place; same seed => same shuffled order.
    rng.shuffle(idx_all)

    n_test = int(round(n_samples * test_size))
    n_val = int(round(n_samples * val_size))

    # Ensure at least 1 in each split (for tiny datasets)
    n_test = max(1, n_test)
    n_val = max(1, n_val)

    # Ensure train is not empty
    if n_test + n_val >= n_samples:
        raise ValueError("Split sizes too large; train would be empty.")

    idx_test = idx_all[:n_test]
    idx_val = idx_all[n_test : n_test + n_val]
    idx_train = idx_all[n_test + n_val :]

    return idx_train, idx_val, idx_test
