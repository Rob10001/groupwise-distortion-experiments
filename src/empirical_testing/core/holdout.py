"""Validated, respondent-level fixed holdout contracts.

The primary Coffee protocol has one mechanically fixed test partition.  This
module deliberately represents that boundary with explicit row indices and
hashes of the respondent identifiers, rather than with a fold number that a
runner could accidentally iterate over.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from empirical_testing.core.types import DensePreferenceDataset


def sorted_id_sha256(respondent_ids: np.ndarray) -> str:
    """Return the SHA-256 of sorted *distinct* IDs, newline terminated.

    The protocol assigns distinct respondents, so duplicate records must not
    change this identifier-set fingerprint.  IDs are normalized exactly as the
    legacy splitting helper did: ``astype(str)`` followed by NumPy sorting.
    """

    ids = np.asarray(respondent_ids)
    if ids.ndim != 1:
        raise ValueError("respondent IDs must be a one-dimensional array")
    unique_ids = np.unique(ids.astype(str))
    payload = "".join(f"{respondent_id}\n" for respondent_id in unique_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _readonly_index_array(indices: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integer row indices")
    array = array.astype(np.int64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FixedHoldout:
    """An exhaustive, disjoint respondent-level train/test partition.

    ``respondent_ids_sha256`` fingerprints the full distinct ID population;
    the two remaining hashes fingerprint its train and test subsets.  A caller
    that needs the raw source-file hash keeps it in its run provenance, where
    file bytes (rather than this in-memory row partition) are available.
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    n_rows: int
    partition_count: int
    seed: int
    test_partition: int
    respondent_ids_sha256: str
    train_ids_sha256: str
    test_ids_sha256: str

    def __post_init__(self) -> None:
        train_indices = _readonly_index_array(self.train_indices, name="train_indices")
        test_indices = _readonly_index_array(self.test_indices, name="test_indices")
        object.__setattr__(self, "train_indices", train_indices)
        object.__setattr__(self, "test_indices", test_indices)
        self.validate()

    def validate(self) -> "FixedHoldout":
        """Check index coverage and hash formatting, returning ``self``."""

        if self.n_rows <= 0:
            raise ValueError("n_rows must be positive")
        if self.partition_count < 2:
            raise ValueError("partition_count must be at least two")
        if not 0 <= self.test_partition < self.partition_count:
            raise ValueError("test_partition is outside the configured partitions")
        combined = np.concatenate((self.train_indices, self.test_indices))
        if combined.size != self.n_rows:
            raise ValueError("train and test indices must exhaust all rows exactly once")
        if np.any(combined < 0) or np.any(combined >= self.n_rows):
            raise ValueError("holdout contains an out-of-range row index")
        if np.unique(combined).size != self.n_rows:
            raise ValueError("train and test indices must be disjoint and unique")
        if not np.array_equal(np.sort(combined), np.arange(self.n_rows, dtype=np.int64)):
            raise ValueError("train and test indices must cover every row")
        for name in (
            "respondent_ids_sha256",
            "train_ids_sha256",
            "test_ids_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{name} must be a SHA-256 hexadecimal digest") from error
        return self


def subset_dense_dataset(
    dataset: DensePreferenceDataset, indices: np.ndarray,
) -> DensePreferenceDataset:
    """Return a validated, non-overlapping respondent subset of ``dataset``.

    This is intentionally not a bootstrap helper: a dense preference dataset
    requires unique respondent IDs.  Bootstrap code should carry multiplicity
    weights separately instead of duplicating dense rows.
    """

    dataset.validate()
    rows = _readonly_index_array(indices, name="indices")
    n_rows = dataset.utilities.shape[0]
    if np.any(rows < 0) or np.any(rows >= n_rows):
        raise ValueError("subset indices contain an out-of-range row index")
    if np.unique(rows).size != rows.size:
        raise ValueError("subset indices must be unique")
    return DensePreferenceDataset(
        name=dataset.name,
        utilities=np.asarray(dataset.utilities)[rows].copy(),
        groups=np.asarray(dataset.groups)[rows].copy(),
        group_names=dataset.group_names,
        action_names=dataset.action_names,
        respondent_ids=np.asarray(dataset.respondent_ids)[rows].copy(),
        source=dataset.source,
        notes=dataset.notes,
        auxiliary={key: np.asarray(values)[rows].copy() for key, values in dataset.auxiliary.items()},
    ).validate()
