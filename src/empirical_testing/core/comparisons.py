"""Pairwise-feedback construction for complete preference profiles."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from empirical_testing.core.types import ComparisonLog, DensePreferenceDataset


def bradley_terry_signs(
    difference: np.ndarray, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw +-1 labels from the Bradley-Terry link at inverse temperature ``beta``.

    ``Pr(a > b) = sigmoid(beta * (u_a - u_b))``. Exact utility ties need no
    special case: ``sigmoid(0) = 1/2`` *is* the fair coin.

    This is the vectorized twin of the scalar rule in
    `algorithms.one_expert._draw_comparison`; `tests/test_beta_labels.py`
    cross-checks them so the two cannot drift.
    """

    if beta <= 0.0 or (not np.isfinite(beta) and not np.isposinf(beta)):
        raise ValueError("beta must be positive or +infinity")
    if np.isposinf(beta):
        signs = np.sign(difference).astype(np.float64)
        tied = signs == 0.0
        signs[tied] = np.where(rng.random(int(tied.sum())) < 0.5, 1.0, -1.0)
        return signs
    win_probability = expit(float(beta) * difference)
    return np.where(rng.random(difference.size) < win_probability, 1.0, -1.0)


def make_uniform_passive_log(
    dataset: DensePreferenceDataset,
    respondent_indices: np.ndarray,
    seed: int,
    comparisons_per_respondent: int = 1,
    beta: float = np.inf,
) -> ComparisonLog:
    """Reveal uniformly sampled unordered comparisons in a saved row order.

    ``beta`` is the Bradley-Terry inverse temperature of the *label* model. The
    default ``+inf`` is the deterministic score-order oracle: the strict
    ordering, with a seeded fair coin on exact score ties. A finite ``beta``
    draws each label from ``sigmoid(beta * (u_a - u_b))``, which is the
    manuscript's Assumption ``bradley-terry`` and the only regime in which the
    distortion factor ``beta/(4*(sigmoid(beta)-1/2))`` is finite.

    Pair *selection* does not depend on ``beta``, so the passive propensity is
    ``1 / C(m, 2)`` either way. The returned log should be built once and passed
    unchanged to every learning arm in a comparison.
    """

    if comparisons_per_respondent <= 0:
        raise ValueError("comparisons_per_respondent must be positive")
    base_rows = np.asarray(respondent_indices, dtype=np.int64)
    rows = np.repeat(base_rows, comparisons_per_respondent)
    m = dataset.utilities.shape[1]
    edges = np.asarray(
        [(left, right) for left in range(m) for right in range(left + 1, m)],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    pairs = edges[rng.integers(0, len(edges), size=len(rows))]
    difference = (
        dataset.utilities[rows, pairs[:, 0]]
        - dataset.utilities[rows, pairs[:, 1]]
    )
    # The beta = +inf branch consumes the RNG exactly as before (one `random`
    # call sized by the tie count), so an infinite-beta log is bit-identical to
    # every run produced before the label model existed.
    signs = bradley_terry_signs(difference, beta, rng)
    propensities = np.full(len(rows), 1.0 / len(edges), dtype=np.float64)

    order = rng.permutation(len(rows))
    return ComparisonLog(
        respondent_indices=rows[order],
        pairs=pairs[order],
        signs=signs[order],
        propensities=propensities[order],
    ).validate(len(dataset.utilities), m)

