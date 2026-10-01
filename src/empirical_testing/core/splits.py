"""Respondent-clustered split helpers."""

from __future__ import annotations

import numpy as np

from empirical_testing.core.holdout import FixedHoldout, sorted_id_sha256


def _cluster_assignments(respondent_ids: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("at least two partitions are required")
    ids = np.asarray(respondent_ids).astype(str)
    unique, inverse = np.unique(ids, return_inverse=True)
    if len(unique) < count:
        raise ValueError("fewer unique respondents than requested partitions")
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(unique), dtype=np.int64)
    order = rng.permutation(len(unique))
    assignment[order] = np.arange(len(unique), dtype=np.int64) % count
    return assignment[inverse]


def respondent_fixed_holdout(
    respondent_ids: np.ndarray,
    count: int,
    seed: int,
    test_partition: int,
) -> FixedHoldout:
    """Build one fixed respondent-clustered train/test holdout.

    Assignment is intentionally the legacy algorithm: NumPy sorts distinct
    string IDs, ``default_rng(seed)`` permutes their positions, and positions
    in that permutation cycle through partitions.  ``test_partition`` selects
    one partition; it is never a request to run cross-validation.
    """

    ids = np.asarray(respondent_ids)
    if ids.ndim != 1:
        raise ValueError("respondent IDs must be a one-dimensional array")
    if not 0 <= test_partition < count:
        raise ValueError("test_partition is outside the configured partitions")
    assignment = _cluster_assignments(ids, count, seed)
    test_indices = np.flatnonzero(assignment == test_partition)
    train_indices = np.flatnonzero(assignment != test_partition)
    string_ids = ids.astype(str)
    return FixedHoldout(
        train_indices=train_indices,
        test_indices=test_indices,
        n_rows=ids.size,
        partition_count=count,
        seed=seed,
        test_partition=test_partition,
        respondent_ids_sha256=sorted_id_sha256(string_ids),
        train_ids_sha256=sorted_id_sha256(string_ids[train_indices]),
        test_ids_sha256=sorted_id_sha256(string_ids[test_indices]),
    )

