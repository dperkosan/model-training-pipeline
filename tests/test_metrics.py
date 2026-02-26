import numpy as np

from model_training_pipeline.metrics import accuracy, f1_score


def test_accuracy_basic() -> None:
    y_true = np.asarray([0, 1, 1, 0, 1], dtype=np.int64)
    y_pred = np.asarray([0, 1, 0, 0, 1], dtype=np.int64)

    # correct positions: 0,1,3,4 => 4 out of 5
    assert accuracy(y_true=y_true, y_pred=y_pred) == 0.8


def test_f1_score_basic() -> None:
    y_true = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
    y_pred = np.asarray([1, 1, 0, 1, 0, 0], dtype=np.int64)

    # tp=2 (positions 0,1)
    # fp=1 (position 3)
    # fn=1 (position 2)
    # precision = 2/3, recall = 2/3, f1 = 2/3
    assert f1_score(y_true=y_true, y_pred=y_pred) == (2.0 / 3.0)
