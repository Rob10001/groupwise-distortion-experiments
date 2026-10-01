"""Reconstruct deployment policies along one full-horizon span-adaptive fit.

Round ``t`` uses the first ``t`` pre-round snapshot maps from the *same* fit.
It does not refit the learner on a shortened training stream.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from empirical_testing.algorithms.one_expert import _as_group_matrix
from empirical_testing.algorithms.span_adaptive import GroupwiseSpanAdaptivePolicy


def checkpoint_rounds(horizon: int, fractions: Iterable[float]) -> NDArray[np.int64]:
    """Convert horizon fractions to unique, ascending integer round counts."""
    if isinstance(horizon, (bool, np.bool_)) or int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    values = np.asarray(list(fractions), dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("fractions must be finite values in [0, 1]")
    return np.unique(np.rint(values * int(horizon)).astype(np.int64))


def predict_prefixes(
    model: GroupwiseSpanAdaptivePolicy,
    test_groups: ArrayLike,
    checkpoints: Iterable[int],
) -> NDArray[np.float64]:
    """Evaluate prefix means of a fitted model's pre-round deployment maps.

    The returned checkpoint axis follows the caller's order. A requested zero
    checkpoint gives a uniform lottery, independently of group membership.
    The model must contain every pre-round snapshot of its original fit.
    """
    groups = _as_group_matrix(test_groups)
    outer = np.asarray(model._outer_log_mass_snapshots)
    inner = np.asarray(model._inner_policy_snapshots)
    horizon = outer.shape[0]
    if (
        outer.ndim != 2
        or inner.shape != (horizon, outer.shape[1], model.n_alternatives)
        or not np.array_equal(model.snapshot_rounds, np.arange(horizon))
        or groups.shape[1] != outer.shape[1]
    ):
        raise ValueError("model needs consecutive full-horizon snapshots and matching groups")
    raw = list(checkpoints)
    if any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) for t in raw):
        raise ValueError("checkpoints must be integer round counts")
    requested = np.asarray(raw, dtype=np.int64)
    if np.any((requested < 0) | (requested > horizon)):
        raise ValueError("checkpoints must lie in [0, horizon]")

    result = np.empty((len(requested), groups.shape[0], model.n_alternatives), dtype=np.float64)
    if not len(requested):
        return result
    zero = requested == 0
    result[zero] = 1.0 / model.n_alternatives
    positive = np.unique(requested[~zero])
    if not len(positive):
        return result

    patterns, inverse = np.unique(groups, axis=0, return_inverse=True)
    for start in range(0, len(patterns), 128):
        stop = min(start + 128, len(patterns))
        membership = patterns[start:stop].astype(bool)
        active = np.any(membership, axis=1)
        sums = np.zeros((stop - start, model.n_alternatives), dtype=np.float64)
        by_round = np.empty((len(positive), stop - start, model.n_alternatives), dtype=np.float64)
        for block_start in range(0, int(positive[-1]), 256):
            block_stop = min(block_start + 256, int(positive[-1]))
            n_block = block_stop - block_start
            block = np.empty((n_block, stop - start, model.n_alternatives), dtype=np.float64)
            if np.any(active):
                selected = np.where(
                    membership[None, active, :],
                    outer[block_start:block_stop, None, :],
                    -np.inf,
                )
                mass = np.exp(selected - np.max(selected, axis=2, keepdims=True))
                block[:, active] = np.einsum(
                    "tpg,tga->tpa", mass, inner[block_start:block_stop], optimize=True
                ) / mass.sum(axis=2, keepdims=True)
            if np.any(~active):
                block[:, ~active] = model.reference
            cumulative = np.cumsum(block, axis=0) + sums[None]
            in_block = (positive > block_start) & (positive <= block_stop)
            for checkpoint_index in np.flatnonzero(in_block):
                count = int(positive[checkpoint_index])
                prediction = cumulative[count - block_start - 1] / count
                prediction = np.maximum(prediction, 0.0)
                prediction /= prediction.sum(axis=1, keepdims=True)
                by_round[checkpoint_index] = prediction
            sums = cumulative[-1]
        selected_rows = (inverse >= start) & (inverse < stop)
        result[np.ix_(~zero, selected_rows)] = by_round[np.searchsorted(positive, requested[~zero])][:, inverse[selected_rows] - start]
    return result
