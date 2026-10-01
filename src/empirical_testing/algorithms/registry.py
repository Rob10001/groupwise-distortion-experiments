"""Dispatch between the two online learners by algorithm identifier.

``groupwise_self_play`` is the manuscript's fixed-rate Algorithm 1
(:mod:`one_expert`, unchanged).  ``groupwise_self_play_span_adaptive`` is the
span-adaptive sibling (:mod:`span_adaptive`). Shared helpers and numerical
regression tests use :func:`fit_online_learner`; the fixed-rate implementation
also supplies a reference for validating the adaptive learner.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from empirical_testing.algorithms.one_expert import fit_groupwise_self_play
from empirical_testing.algorithms.span_adaptive import (
    ALGORITHM_ID as SPAN_ADAPTIVE_ID,
    FrozenRates,
    fit_groupwise_span_adaptive,
)
from empirical_testing.reporting.metrics import theory_certificate_error

FIXED_ID = "groupwise_self_play"
ONLINE_ALGORITHMS = (FIXED_ID, SPAN_ADAPTIVE_ID)


def is_adaptive(algorithm: str) -> bool:
    if algorithm not in ONLINE_ALGORITHMS:
        raise ValueError(
            f"unknown online algorithm {algorithm!r}; known={list(ONLINE_ALGORITHMS)}"
        )
    return algorithm == SPAN_ADAPTIVE_ID


def fit_online_learner(
    algorithm: str,
    groups,
    utilities=None,
    beta: float = 1.0,
    reference=None,
    tau: float = 0.0,
    protocol: str = "passive",
    seed: int = 0,
    pair_indices=None,
    signs=None,
    propensities=None,
    *,
    group_masses=None,
    delta: float | None = None,
    frozen_rates: FrozenRates | None = None,
):
    """Fit the named online learner with the exact all-round output average.

    The fixed learner needs ``group_masses`` (and ``delta``, default 0.05); the
    adaptive learner takes neither and ignores them if supplied, so evaluation
    inputs can never leak into its training.
    """

    common = dict(
        groups=groups, utilities=utilities, beta=beta, reference=reference, tau=tau,
        protocol=protocol, seed=seed, n_snapshots=None, pair_indices=pair_indices,
        signs=signs, propensities=propensities,
    )
    if is_adaptive(algorithm):
        return fit_groupwise_span_adaptive(**common, frozen_rates=frozen_rates)
    if frozen_rates is not None:
        raise ValueError("frozen_rates is a hook of the adaptive learner only")
    if group_masses is None:
        raise ValueError("group_masses is required for the fixed-rate self-play learner")
    return fit_groupwise_self_play(
        **common, group_masses=group_masses, delta=0.05 if delta is None else float(delta)
    )


def rate_diagnostics(algorithm: str, model: Any) -> tuple[str, dict]:
    """The per-arm diagnostics block and its key suffix (``rates`` or ``adaptive``)."""

    if is_adaptive(algorithm):
        block = model.diagnostics.as_json()
        # Names shared with the fixed block so generic readers find Q and V.
        block["outer_second_moment"] = model.outer_second_moments.tolist()
        block["realized_inner_second_moment"] = model.inner_second_moments.tolist()
        block["centered_inner_second_moment"] = model.centered_inner_second_moments.tolist()
        return "adaptive", block
    rates = model.rates
    realized = model.normalized_inner_second_moments
    return "rates", {
        "group_masses": rates.group_masses.tolist(),
        "L0": rates.outer_complexity,
        "L_delta": rates.confidence_complexity,
        "tau_bar": rates.tau_bar,
        "H": rates.complexity,
        "second_moment_bound": rates.second_moment_bound.tolist(),
        "eta": rates.outer.tolist(),
        "theta": rates.inner.tolist(),
        "cap_binds": rates.cap_binds,
        "realized_inner_second_moment": realized.tolist(),
        "concentration_holds": bool(np.all(realized <= rates.second_moment_bound)),
        "outer_second_moment": model.outer_second_moments.tolist(),
        "max_outer_mass_error": model.max_outer_mass_error,
        "theory_certificate_error": theory_certificate_error(
            rates.horizon, rates.group_masses, rates.score_bound, rates.complexity
        ).tolist(),
    }
