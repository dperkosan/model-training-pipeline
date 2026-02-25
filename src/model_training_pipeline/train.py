from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def train_baseline_model(*, X_train: FloatArray, y_train: IntArray, seed: int) -> LogisticRegression:
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
