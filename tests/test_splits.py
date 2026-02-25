import numpy as np

from model_training_pipeline.data import split_indices


def test_split_indices_no_overlap_and_covers_all_rows() -> None:
    n = 100
    idx_train, idx_val, idx_test = split_indices(
        n_samples=n,
        seed=42,
        test_size=0.2,
        val_size=0.1,
    )

    train_set = set(idx_train.tolist())
    val_set = set(idx_val.tolist())
    test_set = set(idx_test.tolist())

    # No overlap (no leakage)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # All rows are accounted for exactly once
    assert len(train_set) + len(val_set) + len(test_set) == n
    assert train_set | val_set | test_set == set(range(n))


def test_split_indices_deterministic_for_same_seed() -> None:
    a = split_indices(n_samples=100, seed=123, test_size=0.2, val_size=0.1)
    b = split_indices(n_samples=100, seed=123, test_size=0.2, val_size=0.1)

    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])
