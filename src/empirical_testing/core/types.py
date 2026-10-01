"""Validated, dataset-independent data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DensePreferenceDataset:
    """Complete respondent-by-alternative preference profiles."""

    name: str
    utilities: np.ndarray
    groups: np.ndarray
    group_names: tuple[str, ...]
    action_names: tuple[str, ...]
    respondent_ids: np.ndarray
    source: str
    notes: tuple[str, ...] = ()
    auxiliary: Mapping[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> "DensePreferenceDataset":
        utilities = np.asarray(self.utilities)
        groups = np.asarray(self.groups)
        respondent_ids = np.asarray(self.respondent_ids)
        if utilities.ndim != 2 or utilities.shape[1] < 2:
            raise ValueError(f"{self.name}: utilities must be an n-by-m matrix with m >= 2")
        n, m = utilities.shape
        if groups.ndim != 2 or groups.shape[0] != n:
            raise ValueError(f"{self.name}: group rows must align with utilities")
        if respondent_ids.shape != (n,):
            raise ValueError(f"{self.name}: respondent IDs must have shape ({n},)")
        if groups.shape[1] != len(self.group_names):
            raise ValueError(f"{self.name}: group names disagree with group columns")
        if m != len(self.action_names):
            raise ValueError(f"{self.name}: action names disagree with utility columns")
        if len(np.unique(respondent_ids.astype(str))) != n:
            raise ValueError(f"{self.name}: dense-profile respondent IDs must be unique")
        if not np.all(np.isfinite(utilities)):
            raise ValueError(f"{self.name}: utilities must be complete and finite")
        if np.any(utilities < 0.0) or np.any(utilities > 1.0):
            raise ValueError(f"{self.name}: utilities must lie in [0,1]")
        if not np.all((groups == 0) | (groups == 1)):
            raise ValueError(f"{self.name}: groups must be binary indicators")
        if not self.group_names or self.group_names[0] != "universal":
            raise ValueError(f"{self.name}: the first group must be named universal")
        if not np.all(groups[:, 0] == 1):
            raise ValueError(f"{self.name}: every respondent must activate universal")
        for key, values in self.auxiliary.items():
            if np.asarray(values).shape[0] != n:
                raise ValueError(f"{self.name}: auxiliary field {key!r} has the wrong length")
        return self


@dataclass(frozen=True)
class ComparisonLog:
    """A materialized pairwise-feedback stream for a dense dataset."""

    respondent_indices: np.ndarray
    pairs: np.ndarray
    signs: np.ndarray
    propensities: np.ndarray

    def validate(self, n_rows: int, n_alternatives: int) -> "ComparisonLog":
        rows = np.asarray(self.respondent_indices)
        pairs = np.asarray(self.pairs)
        signs = np.asarray(self.signs)
        propensities = np.asarray(self.propensities)
        n = len(rows)
        if pairs.shape != (n, 2) or signs.shape != (n,) or propensities.shape != (n,):
            raise ValueError("comparison-log columns have inconsistent shapes")
        if np.any(rows < 0) or np.any(rows >= n_rows):
            raise ValueError("comparison log contains an invalid respondent index")
        if np.any(pairs < 0) or np.any(pairs >= n_alternatives):
            raise ValueError("comparison log contains an invalid alternative")
        if np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError("pairs must be stored once with left < right")
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("comparison labels must be -1 or +1")
        if np.any(propensities <= 0.0) or np.any(propensities > 1.0):
            raise ValueError("comparison propensities must lie in (0,1]")
        return self


@dataclass(frozen=True)
class LoggedComparisonDataset:
    """Actual logged comparisons for datasets without complete utilities.

    This contract is intentionally separate from `DensePreferenceDataset`.
    A future Rapidata runner should split respondents or defensible clusters,
    train directly from these records, and evaluate with logged estimators.
    """

    name: str
    groups: np.ndarray
    group_names: tuple[str, ...]
    action_names: tuple[str, ...]
    pairs: np.ndarray
    signs: np.ndarray
    propensities: np.ndarray
    respondent_ids: np.ndarray | None
    instance_ids: np.ndarray | None
    source: str
    notes: tuple[str, ...] = ()

    def validate(self) -> "LoggedComparisonDataset":
        groups = np.asarray(self.groups)
        if groups.ndim != 2:
            raise ValueError(f"{self.name}: groups must be a matrix")
        n = groups.shape[0]
        ComparisonLog(
            respondent_indices=np.arange(n),
            pairs=self.pairs,
            signs=self.signs,
            propensities=self.propensities,
        ).validate(n, len(self.action_names))
        if groups.shape[1] != len(self.group_names):
            raise ValueError(f"{self.name}: group names disagree with group columns")
        if not self.group_names or self.group_names[0] != "universal":
            raise ValueError(f"{self.name}: first group must be universal")
        if not np.all(groups[:, 0] == 1):
            raise ValueError(f"{self.name}: every record must activate universal")
        for label, values in (("respondent_ids", self.respondent_ids), ("instance_ids", self.instance_ids)):
            if values is not None and np.asarray(values).shape != (n,):
                raise ValueError(f"{self.name}: {label} must align with comparison rows")
        return self

