"""Independent NumPy/SciPy reference for the span-adaptive groupwise learner.

This module independently implements the learner's updates to catch bugs in
the optimized implementation. Its numerical dependencies are only ``numpy``
and ``scipy``; it shares no update routines with ``span_adaptive``,
``one_expert``, or ``numerics``.

Clarity over speed: Python loops and explicit per-coordinate formulas are
intentional.  A slow oracle that is independently derived is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Learning-rate caps and accumulator offsets used by GLHF.
_OUTER_RATE_CAP = 0.5
_INNER_RATE_CAP = 1.0
_OUTER_DENOMINATOR_OFFSET = 1.0
_INNER_DENOMINATOR_OFFSET = 0.25
# Numerical threshold for treating the KL ball as the singleton reference.
_PROX_ZERO_RADIUS = 1e-10
_TINY = np.finfo(np.float64).tiny


@dataclass
class SpanAdaptiveReferenceResult:
    """Scientific output of the independent span-adaptive reference.

    Snapshots are the *pre-round* policy maps: one row per training
    round, uniform weight ``1/T``.  ``averaged_policy`` is ``predict`` on the
    training membership matrix.
    """

    averaged_policy: FloatArray
    outer_log_mass_snapshots: FloatArray  # (T, K)
    inner_policy_snapshots: FloatArray  # (T, K, m)
    snapshot_weights: FloatArray  # (T,)
    Q: FloatArray
    C: FloatArray
    V: FloatArray
    reference: FloatArray
    tau: float
    drawn_comparisons: FloatArray

    def predict(self, groups: ArrayLike) -> FloatArray:
        return predict_span_adaptive_reference(self, groups)


def _as_group_matrix(groups: ArrayLike) -> FloatArray:
    matrix = np.asarray(groups, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("groups must have shape (n_rows, n_groups) with n_groups >= 1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("groups contains a non-finite value")
    if not np.all((matrix == 0.0) | (matrix == 1.0)):
        raise ValueError("groups must contain only binary 0/1 memberships")
    return matrix


def _validate_reference(reference: ArrayLike) -> FloatArray:
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 1 or ref.size < 2:
        raise ValueError("reference must be a one-dimensional lottery of length at least 2")
    if not np.all(np.isfinite(ref)) or np.any(ref <= 0.0):
        raise ValueError("reference must be finite and have full support")
    total = float(ref.sum())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"reference must sum to one, got {total}")
    return ref / total


def _kl_divergence(q: ArrayLike, reference: ArrayLike) -> float:
    """``KL(q || reference)`` with the convention ``0 log 0 = 0``."""

    q_array = np.asarray(q, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if q_array.shape != ref.shape:
        raise ValueError("q and reference must have the same shape")
    if np.all(q_array > 0.0):
        return float(np.sum(q_array * (np.log(q_array) - np.log(ref))))
    positive = q_array > 0.0
    return float(
        np.sum(q_array[positive] * (np.log(q_array[positive]) - np.log(ref[positive])))
    )


def _log_sum_exp(log_weights: FloatArray) -> float:
    """SciPy's ``logsumexp``, including its operation order.

    Non-finite maxima are delegated to SciPy as well.  This is the independent
    definition the optimized kernel is required to match, not a reimplementation
    of that kernel.
    """

    return float(logsumexp(np.asarray(log_weights, dtype=np.float64)))


def _softmax_from_log(log_weights: FloatArray) -> FloatArray:
    """Exponentiate, floor at ``tiny``, then renormalize."""

    values = np.exp(np.asarray(log_weights, dtype=np.float64) - _log_sum_exp(log_weights))
    values = np.maximum(values, _TINY)
    return values / values.sum()


def _complexity_term(n_groups: int, horizon: int) -> float:
    """Logarithmic factor ``L = max(1, log(K (1 + 0.5 log(T + 1))))``."""

    return float(max(1.0, np.log(n_groups * (1.0 + 0.5 * np.log(horizon + 1.0)))))


def _effective_radius(tau: float, reference: FloatArray) -> float:
    """Effective radius ``tau_bar = min(tau, log(1 / min pi_ref))``."""

    return float(min(tau, np.log(1.0 / float(np.min(reference)))))


def _outer_rate(complexity: float, outer_moment: FloatArray) -> FloatArray:
    """Outer rate ``eta(Q) = min(1/2, sqrt(L / (2 (1 + Q))))``."""

    q = np.asarray(outer_moment, dtype=np.float64)
    return np.minimum(
        _OUTER_RATE_CAP,
        np.sqrt(complexity / (2.0 * (_OUTER_DENOMINATOR_OFFSET + q))),
    )


def _inner_rate(tau_bar: float, centered_moment: FloatArray) -> FloatArray:
    """Inner rate ``theta(C) = min(1, sqrt(tau_bar / (2 (1/4 + C))))``."""

    c = np.asarray(centered_moment, dtype=np.float64)
    return np.minimum(
        _INNER_RATE_CAP,
        np.sqrt(tau_bar / (2.0 * (_INNER_DENOMINATOR_OFFSET + c))),
    )


def _score_bound(
    protocol: str,
    n_alternatives: int,
    propensities: Optional[FloatArray],
) -> float:
    """Protocol-specific score scale ``R``, bounded below by one."""

    if protocol == "active":
        bound = float(n_alternatives - 1)
    elif protocol == "logged":
        if propensities is None:
            raise ValueError("logged feedback requires propensities")
        bound = float(np.max(1.0 / propensities))
    elif propensities is None:
        bound = float(n_alternatives * (n_alternatives - 1) // 2)
    else:
        bound = float(np.max(1.0 / propensities))
    return max(1.0, bound)


def _two_sparse_score(
    policy: FloatArray,
    i: int,
    j: int,
    sign: float,
    propensity: float,
) -> FloatArray:
    """Two-sparse inverse-propensity score."""

    score = np.zeros(policy.shape, dtype=np.float64)
    scale = sign / propensity
    score[j] += scale * policy[i]
    score[i] -= scale * policy[j]
    cancellation = float(np.dot(policy, score))
    if abs(cancellation) > 1e-9 * max(1.0, float(np.max(np.abs(score)))):
        raise AssertionError("policy-weighted score failed exact cancellation")
    return score


def _draw_comparison(
    round_index: int,
    policy: FloatArray,
    utilities: Optional[FloatArray],
    beta: float,
    protocol: str,
    pairs: Optional[IntArray],
    signs: Optional[FloatArray],
    propensities: Optional[FloatArray],
    all_pairs: IntArray,
    rng: np.random.Generator,
) -> tuple[int, int, float, float]:
    """One comparison, consuming RNG only when the protocol draws."""

    n_alternatives = int(policy.size)
    if protocol == "active":
        first = int(rng.choice(n_alternatives, p=policy))
        second = int(rng.integers(n_alternatives - 1))
        if second >= first:
            second += 1
        i, j = first, second
        propensity = float((policy[i] + policy[j]) / (n_alternatives - 1))
    elif pairs is None:
        edge = int(rng.integers(all_pairs.shape[0]))
        i, j = int(all_pairs[edge, 0]), int(all_pairs[edge, 1])
        propensity = 1.0 / float(all_pairs.shape[0])
    else:
        i, j = int(pairs[round_index, 0]), int(pairs[round_index, 1])
        if propensities is None:
            propensity = 1.0 / float(all_pairs.shape[0])
        else:
            propensity = float(propensities[round_index])

    if signs is None:
        if utilities is None:
            raise ValueError("utilities is needed to generate a preference label")
        difference = float(utilities[round_index, i] - utilities[round_index, j])
        if np.isposinf(beta):
            if difference > 0.0:
                sign = 1.0
            elif difference < 0.0:
                sign = -1.0
            else:
                sign = 1.0 if rng.random() < 0.5 else -1.0
        else:
            win_probability = float(expit(beta * difference))
            sign = 1.0 if rng.random() < win_probability else -1.0
    else:
        sign = float(signs[round_index])

    if propensity <= 0.0:
        raise ValueError("the observed pair has zero propensity")
    return i, j, sign, propensity


def kl_prox_project(
    q: FloatArray,
    loss: FloatArray,
    rate: float,
    reference: FloatArray,
    tau: float,
) -> tuple[FloatArray, Optional[FloatArray]]:
    """Exponentiated update followed by KL projection onto ``B_tau``.

    Solves ``min_p rate <loss, p> + KL(p || q)`` subject to ``KL(p || pi_ref) <= tau``.
    Returns ``(projected, unconstrained)``.  The unconstrained point is ``None``
    on the singleton-radius branch (step 1).
    """

    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    if tau <= _PROX_ZERO_RADIUS:
        return reference.copy(), None
    if rate < 0.0 or not np.isfinite(rate):
        raise ValueError("learning_rate must be finite and nonnegative")

    log_q = np.log(np.maximum(q, _TINY))
    log_ref = np.log(reference)
    unconstrained = _softmax_from_log(log_q - rate * loss)

    maximum_kl = float(np.log(1.0 / np.min(reference)))
    if tau >= maximum_kl - _PROX_ZERO_RADIUS:
        return unconstrained, unconstrained
    if _kl_divergence(unconstrained, reference) <= tau + _PROX_ZERO_RADIUS:
        return unconstrained, unconstrained

    def candidate(multiplier: float) -> FloatArray:
        scale = 1.0 / (1.0 + multiplier)
        candidate_log = scale * (log_q - rate * loss) + (1.0 - scale) * log_ref
        return _softmax_from_log(candidate_log)

    lower = 0.0
    upper = 1.0
    while _kl_divergence(candidate(upper), reference) > tau:
        upper *= 2.0
        if upper > 1e16:
            return reference.copy(), unconstrained

    def residual(multiplier: float) -> float:
        return _kl_divergence(candidate(multiplier), reference) - tau

    root = float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12))
    result = candidate(root)
    if _kl_divergence(result, reference) > tau:
        result = candidate(float(np.nextafter(root, np.inf)))
    if _kl_divergence(result, reference) > tau + 1e-8:
        raise AssertionError("KL-prox update failed to return a feasible point")
    return result, unconstrained


def predict_span_adaptive_reference(
    fit: SpanAdaptiveReferenceResult,
    groups: ArrayLike,
) -> FloatArray:
    """Uniform average of the pre-round policy maps, per membership pattern."""

    membership = _as_group_matrix(groups)
    n_groups = int(fit.inner_policy_snapshots.shape[1])
    n_alternatives = int(fit.reference.size)
    if membership.shape[1] != n_groups:
        raise ValueError(
            f"group_matrix has {membership.shape[1]} columns, expected {n_groups}"
        )
    unique_patterns, inverse = np.unique(membership, axis=0, return_inverse=True)
    n_patterns = int(unique_patterns.shape[0])
    accumulated = np.zeros((n_patterns, n_alternatives), dtype=np.float64)
    horizon = int(fit.outer_log_mass_snapshots.shape[0])

    for t in range(horizon):
        weight = float(fit.snapshot_weights[t])
        outer_log_mass = fit.outer_log_mass_snapshots[t]
        inner_policies = fit.inner_policy_snapshots[t]
        for p in range(n_patterns):
            active = unique_patterns[p] == 1.0
            if not np.any(active):
                accumulated[p] += weight * fit.reference
                continue
            log_mass = np.where(active, outer_log_mass, -np.inf)
            shifted = np.exp(log_mass - np.max(log_mass))
            mixed = shifted @ inner_policies
            accumulated[p] += weight * (mixed / shifted.sum())

    accumulated = np.maximum(accumulated, 0.0)
    accumulated /= accumulated.sum(axis=1, keepdims=True)
    for row in accumulated:
        if _kl_divergence(row, fit.reference) > fit.tau + 2e-7:
            raise AssertionError("snapshot-averaged alignment prediction is infeasible")
    return accumulated[inverse]


def _prepare_inputs(
    groups: ArrayLike,
    utilities: Optional[ArrayLike],
    beta: float,
    protocol: str,
    pair_indices: Optional[ArrayLike],
    signs: Optional[ArrayLike],
    propensities: Optional[ArrayLike],
    reference: FloatArray,
) -> tuple[
    FloatArray,
    Optional[FloatArray],
    str,
    Optional[IntArray],
    Optional[FloatArray],
    Optional[FloatArray],
    int,
    float,
]:
    """Validate inputs and form the protocol's score bound."""

    group_matrix = _as_group_matrix(groups)
    horizon = int(group_matrix.shape[0])
    if horizon == 0:
        raise ValueError("at least one training row is required")
    if beta <= 0.0 or (not np.isfinite(beta) and not np.isposinf(beta)):
        raise ValueError("beta must be positive or +infinity")

    if utilities is None:
        utility_matrix = None
    else:
        utility_matrix = np.asarray(utilities, dtype=np.float64)
        if utility_matrix.ndim != 2 or utility_matrix.shape[0] != horizon:
            raise ValueError("utilities must have shape (n_rows, n_alternatives)")
        if not np.all(np.isfinite(utility_matrix)):
            raise ValueError("utilities contains a non-finite value")
        if np.any(utility_matrix < -1e-12) or np.any(utility_matrix > 1.0 + 1e-12):
            raise ValueError("utilities must lie in [0, 1]")

    normalized = protocol.lower().replace("_", "-")
    if normalized == "general":
        normalized = "logged"
    if normalized not in {"passive", "active", "logged"}:
        raise ValueError("protocol must be 'passive', 'active', or 'logged'")

    if pair_indices is None:
        pairs = None
    else:
        pairs = np.asarray(pair_indices, dtype=np.int64)
        if pairs.shape != (horizon, 2):
            raise ValueError("pair_indices must have shape (n_rows, 2)")

    if signs is None:
        label_signs = None
    else:
        label_signs = np.asarray(signs, dtype=np.float64)
        if label_signs.shape != (horizon,):
            raise ValueError("signs must have shape (n_rows,)")
        if not np.all(np.isfinite(label_signs)) or not np.all(
            np.isin(label_signs, (-1.0, 1.0))
        ):
            raise ValueError("signs must encode a strict winner as -1 or +1")
        if pairs is None:
            raise ValueError("pair_indices is required when signs is supplied")

    if propensities is None:
        propensity_array = None
    else:
        propensity_array = np.asarray(propensities, dtype=np.float64)
        if propensity_array.shape != (horizon,):
            raise ValueError("propensities must have shape (n_rows,)")
        if (
            not np.all(np.isfinite(propensity_array))
            or np.any(propensity_array <= 0.0)
            or np.any(propensity_array > 1.0 + 1e-12)
        ):
            raise ValueError("propensities must be finite probabilities in (0, 1]")

    alternatives = int(reference.size)
    if utility_matrix is not None and utility_matrix.shape[1] != alternatives:
        raise ValueError("utilities has the wrong number of alternatives")
    if pairs is not None:
        if np.any(pairs < 0) or np.any(pairs >= alternatives):
            raise ValueError("pair_indices contains an out-of-range alternative")
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("a pair must contain two distinct alternatives")

    if normalized == "active":
        if pairs is not None or label_signs is not None or propensity_array is not None:
            raise ValueError(
                "active comparisons are generated after seeing the current policy; "
                "do not pass precomputed pairs, signs, or propensities"
            )
        if utility_matrix is None:
            raise ValueError("active simulation requires utilities")
    elif normalized == "logged":
        if pairs is None or label_signs is None or propensity_array is None:
            raise ValueError(
                "logged feedback requires pair_indices, signs, and propensities"
            )
    else:
        if pairs is None and propensity_array is not None:
            raise ValueError("propensities cannot be supplied without pair_indices")
        if pairs is None and utility_matrix is None:
            raise ValueError("passive simulation requires utilities")
        if label_signs is None and utility_matrix is None:
            raise ValueError("utilities is required to generate missing signs")

    bound = _score_bound(normalized, alternatives, propensity_array)
    return (
        group_matrix,
        utility_matrix,
        normalized,
        pairs,
        label_signs,
        propensity_array,
        alternatives,
        bound,
    )


def fit_span_adaptive_reference(
    groups: ArrayLike,
    utilities: Optional[ArrayLike] = None,
    beta: float = 1.0,
    reference: Optional[ArrayLike] = None,
    tau: float = 0.0,
    protocol: str = "passive",
    seed: int = 0,
    pair_indices: Optional[ArrayLike] = None,
    signs: Optional[ArrayLike] = None,
    propensities: Optional[ArrayLike] = None,
    n_snapshots: Optional[int] = None,
) -> SpanAdaptiveReferenceResult:
    """Fit GLHF with an independent implementation of its adaptive updates.

    Scientific inputs match the production fitter.  ``n_snapshots`` is accepted
    only so a non-``None`` value can be rejected: every pre-round map is saved.
    """

    if n_snapshots is not None:
        raise ValueError("the span-adaptive learner returns the exact all-round average only")
    if reference is None:
        if utilities is None:
            raise ValueError("reference is required when utilities is absent")
        utility_array = np.asarray(utilities)
        if utility_array.ndim != 2 or utility_array.shape[1] < 2:
            raise ValueError("cannot infer a uniform reference from utilities")
        reference_policy = np.full(utility_array.shape[1], 1.0 / utility_array.shape[1])
    else:
        reference_policy = _validate_reference(reference)
    if tau < 0.0 or not np.isfinite(tau):
        raise ValueError("tau must be finite and nonnegative")

    (
        group_matrix,
        utility_matrix,
        normalized_protocol,
        pairs,
        label_signs,
        propensity_array,
        alternatives,
        score_bound,
    ) = _prepare_inputs(
        groups,
        utilities,
        beta,
        protocol,
        pair_indices,
        signs,
        propensities,
        reference_policy,
    )
    horizon, n_groups = group_matrix.shape
    if horizon < 2:
        raise ValueError("the algorithm requires at least two rounds")

    complexity = _complexity_term(int(n_groups), int(horizon))
    tau_bar = _effective_radius(float(tau), reference_policy)

    # Shared random-stream derivation. The snapshot stream is unused
    # because the schedule is the exact all-round average.
    seed_sequence = np.random.SeedSequence(seed)
    training_seed, snapshot_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(training_seed)
    _snapshot_rng = np.random.default_rng(snapshot_seed)

    log_initial_mass = -np.log(float(n_groups))
    log_outer_mass = np.full(n_groups, log_initial_mass, dtype=np.float64)
    policies = np.tile(reference_policy, (n_groups, 1))
    outer_moment = np.zeros(n_groups, dtype=np.float64)
    centered_moment = np.zeros(n_groups, dtype=np.float64)
    raw_moment = np.zeros(n_groups, dtype=np.float64)

    eta0 = _outer_rate(complexity, outer_moment)
    if tau_bar <= _PROX_ZERO_RADIUS:
        # Singleton KL ball: return the reference; consume no comparisons.
        effective_log_mass = log_outer_mass + np.log(eta0)
        outer_snapshots = np.tile(effective_log_mass, (horizon, 1))
        inner_snapshots = np.tile(policies, (horizon, 1, 1))
        weights = np.full(horizon, 1.0 / horizon, dtype=np.float64)
        result = SpanAdaptiveReferenceResult(
            averaged_policy=np.empty((0, alternatives), dtype=np.float64),
            outer_log_mass_snapshots=outer_snapshots,
            inner_policy_snapshots=inner_snapshots,
            snapshot_weights=weights,
            Q=outer_moment.copy(),
            C=centered_moment.copy(),
            V=raw_moment.copy(),
            reference=reference_policy,
            tau=float(tau),
            drawn_comparisons=np.zeros((0, 4), dtype=np.float64),
        )
        result.averaged_policy = predict_span_adaptive_reference(result, group_matrix)
        return result

    all_pairs = np.asarray(
        [(i, j) for i in range(alternatives) for j in range(i + 1, alternatives)],
        dtype=np.int64,
    )
    n_unordered = int(all_pairs.shape[0])
    # span(g) <= 1 is asserted only for these protocols.
    span_checked = normalized_protocol == "active" or (
        normalized_protocol == "passive" and propensity_array is None
    ) or (
        propensity_array is not None
        and np.allclose(propensity_array, 1.0 / n_unordered, atol=1e-15, rtol=0.0)
    )
    log_reference = np.log(reference_policy)

    outer_snapshots = np.empty((horizon, n_groups), dtype=np.float64)
    inner_snapshots = np.empty((horizon, n_groups, alternatives), dtype=np.float64)
    drawn = np.full((horizon, 4), np.nan, dtype=np.float64)

    for round_index in range(horizon):
        # §3.1 pre-round rates from the accumulators as they stand.
        eta = _outer_rate(complexity, outer_moment)
        theta = _inner_rate(tau_bar, centered_moment)
        # §3.2 snapshot unconditionally, including rounds with no active group.
        effective_log_mass = log_outer_mass + np.log(eta)
        outer_snapshots[round_index] = effective_log_mass
        inner_snapshots[round_index] = policies

        active_groups = np.flatnonzero(group_matrix[round_index]).astype(np.int64)
        if active_groups.size == 0:
            continue

        # §3.4 deployment: softmax of active outer masses, mix inner policies.
        active_probabilities = _softmax_from_log(effective_log_mass[active_groups])
        proposals = policies[active_groups]
        policy = active_probabilities @ proposals
        policy = np.maximum(policy, 0.0)
        policy /= policy.sum()
        deployed_kl = _kl_divergence(policy, reference_policy)
        if deployed_kl > tau + 2e-8:
            raise AssertionError("deployed mixture is outside the KL ball")

        # §3.5 draw one comparison; form g = -score / R.
        i, j, sign, propensity = _draw_comparison(
            round_index,
            policy,
            utility_matrix,
            beta,
            normalized_protocol,
            pairs,
            label_signs,
            propensity_array,
            all_pairs,
            rng,
        )
        drawn[round_index] = (i, j, sign, propensity)
        score = _two_sparse_score(policy, i, j, sign, propensity)
        reward = -score / score_bound
        cancellation = abs(float(np.dot(policy, reward)))
        if cancellation > 2e-9:
            raise AssertionError("deployed lottery does not cancel the reward")
        if float(np.max(np.abs(reward))) > 1.0 + 1e-9:
            raise AssertionError("normalized reward left [-1, 1]")
        span = float(np.max(reward) - np.min(reward))
        if span_checked and span > 1.0 + 1e-9:
            raise AssertionError("reward span exceeded one under a span-bounded protocol")

        # §3.6 increments from the pre-update inner policies.
        # Compute the centered moment directly to avoid cancellation in v - r^2.
        n_active = int(active_groups.size)
        r = np.empty(n_active, dtype=np.float64)
        c = np.empty(n_active, dtype=np.float64)
        v = np.empty(n_active, dtype=np.float64)
        for position in range(n_active):
            pi_k = proposals[position]
            r[position] = float(np.dot(pi_k, reward))
            deviation = reward - r[position]
            c[position] = float(np.dot(pi_k, deviation * deviation))
            v[position] = float(np.dot(pi_k, reward * reward))

        # §3.7 next accumulators formed but not yet committed.
        q_next = outer_moment[active_groups] + r * r
        c_next = centered_moment[active_groups] + c
        v_next = raw_moment[active_groups] + v
        residual = float(np.max(np.abs(q_next + c_next - v_next)))
        if residual > 1e-9 * max(1.0, float(np.max(v_next))):
            raise AssertionError("Q + C = V failed beyond numerical tolerance")

        eta_active = eta[active_groups]
        theta_active = theta[active_groups]
        eta_next = _outer_rate(complexity, q_next)
        theta_next = _inner_rate(tau_bar, c_next)
        gamma_eta = eta_next / eta_active
        gamma_theta = theta_next / theta_active

        # §4 outer step at the *current* eta; no renormalization.
        factors = eta_active * r
        if np.any(factors <= -1.0):
            raise AssertionError("Prod update produced a nonpositive weight")
        log_plus = log_outer_mass[active_groups] + np.log1p(factors)
        log_outer_mass[active_groups] = (1.0 - gamma_eta) * (-np.log(float(n_groups))) + (
            gamma_eta * log_plus
        )

        # §5 inner: exact-float gamma==1 branch, then prox at the *next* theta.
        for position, group in enumerate(active_groups):
            k = int(group)
            g_theta = float(gamma_theta[position])
            if g_theta == 1.0:
                anchor = policies[k]
            else:
                log_pi = np.log(np.maximum(policies[k], _TINY))
                anchor = _softmax_from_log(
                    g_theta * log_pi + (1.0 - g_theta) * log_reference
                )
            projected, _unconstrained = kl_prox_project(
                anchor, -reward, float(theta_next[position]), reference_policy, float(tau)
            )
            policies[k] = projected

        # §3.9 commit active groups only.
        outer_moment[active_groups] = q_next
        centered_moment[active_groups] = c_next
        raw_moment[active_groups] = v_next

    weights = np.full(horizon, 1.0 / horizon, dtype=np.float64)
    result = SpanAdaptiveReferenceResult(
        averaged_policy=np.empty((0, alternatives), dtype=np.float64),
        outer_log_mass_snapshots=outer_snapshots,
        inner_policy_snapshots=inner_snapshots,
        snapshot_weights=weights,
        Q=outer_moment.copy(),
        C=centered_moment.copy(),
        V=raw_moment.copy(),
        reference=reference_policy,
        tau=float(tau),
        drawn_comparisons=drawn,
    )
    result.averaged_policy = predict_span_adaptive_reference(result, group_matrix)
    return result
