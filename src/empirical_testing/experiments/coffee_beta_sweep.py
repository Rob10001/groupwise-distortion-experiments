"""Shared Bradley-Terry relabeling for the current comparison suite.

Passive protocol cells reuse the saved log's pairs, propensities, and row
order. Finite beta changes only its signs, using common random variates keyed
by the split's training seed. The beta=inf log remains unmodified.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit

# Keyed by the split's training seed, which encodes the CV seed, repeat,
# fold, and learner seed. Preserve this key to reproduce the frozen labels.
BETA_LABEL_STREAM_KEY = 91600317


def beta_label_uniforms(training_seed: int, n_rows: int) -> np.ndarray:
    """The common random numbers ``V_t`` shared by every finite ``beta``.

    One vector per (split, learner seed), indexed by saved log row, drawn from
    a stream keyed only by declared values -- never by task order, worker or
    the ``beta`` being evaluated.
    """

    generator = np.random.default_rng(
        np.random.SeedSequence([int(BETA_LABEL_STREAM_KEY), int(training_seed)])
    )
    return generator.random(int(n_rows))


def relabel_signs(
    respondent_indices: np.ndarray, pairs: np.ndarray, train_utilities: np.ndarray,
    beta: float, uniforms: np.ndarray,
) -> np.ndarray:
    """Bradley-Terry signs for one finite ``beta`` against shared variates.

    Training utilities only.  An exact utility tie needs no special case:
    ``expit(0) = 1/2`` *is* the fair coin. The rule is expressed directly
    against the saved comparison arrays and shared random variates.
    """

    if math.isinf(beta) or not beta > 0.0:
        raise ValueError(
            "relabeling is defined for finite positive beta only; beta=inf uses the saved log"
        )
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if uniforms.shape != (len(pairs),):
        raise ValueError("uniform variates must be indexed by saved log row")
    difference = (
        train_utilities[respondent_indices, pairs[:, 0]]
        - train_utilities[respondent_indices, pairs[:, 1]]
    )
    return np.where(uniforms < expit(float(beta) * difference), 1.0, -1.0)
