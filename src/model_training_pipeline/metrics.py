from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def threshold_predictions(*, y_score: FloatArray, threshold: float) -> IntArray:
    """
    Convert probability scores into 0/1 predictions using a threshold.

    Example: threshold=0.5 -> score >= 0.5 becomes 1 else 0
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    preds = (y_score >= threshold).astype(np.int64)
    return np.asarray(preds, dtype=np.int64)


def accuracy(*, y_true: IntArray, y_pred: IntArray) -> float:
    """
    Accuracy = (number of correct predictions) / (total predictions)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("y_true must not be empty")

    correct = (y_true == y_pred).mean()
    return float(correct)


def f1_score(*, y_true: IntArray, y_pred: IntArray) -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall)

    Assumes binary labels 0/1.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("y_true must not be empty")

    # True positives, false positives, false negatives
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    # precision = tp / (tp + fp)
    # recall    = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) == 0.0:
        return 0.0

    return 2.0 * (precision * recall) / (precision + recall)


def evaluate_binary_classifier(
    *, y_true: IntArray, y_score: FloatArray, threshold: float
) -> dict[str, float]:
    """
    Evaluate a binary classifier from probability scores.

    Returns a small metrics dict you can print/log/save.
    """
    y_pred = threshold_predictions(y_score=y_score, threshold=threshold)
    acc = accuracy(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "f1": float(f1),
    }
