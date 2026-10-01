"""Groupwise self-play from one comparison per user.

This module implements the paper's fixed-rate groupwise algorithm:
:func:`fit_groupwise_self_play`.  Each group ``k`` maintains one outer Prod
weight ``w_k`` and one reverse-KL-constrained inner challenger ``mu_k``; the
deployed policy mixes the challengers of the groups containing the current
user.

The learner is told the group masses ``p_k``, the horizon ``T`` and the
confidence level ``delta``, so both learning rates are **fixed constants
computed before the first round** (``eq:rates``)::

    V_bar_k = 2 T p_k / R + L_delta
    eta_k   = min(1/2, sqrt(L_0 / V_bar_k))      # outer
    theta_k = min(1/2, sqrt(tau_bar / V_bar_k))  # inner

with ``L_0 = log(eK)``, ``L_delta = log(2K/delta)`` and ``tau_bar`` the capped
radius of ``eq:radius-cap``.  Because the rates never change, the outer update
is a plain Prod step and the inner update is an ordinary entropic step followed
by a KL projection: no anchoring toward the initialization and no geometric
stabilization toward the reference are required. The empirical study uses
the separate adaptive-rate learner in :mod:`.span_adaptive`.

The realized second moments ``Q`` and ``V`` are still accumulated, but only as
diagnostics. They let a run check the theoretical concentration bound
``V_{T,k} <= V_bar_k``.

The fitter accepts either simulated Bradley--Terry feedback or externally logged
pairwise comparisons.  For logged feedback, ``propensities[t]`` must be the
probability with which the *unordered* observed pair was selected, conditional
on the information used by the logging policy.  It is not the probability of the
observed preference label.  The inverse-propensity score is

    sign / propensity * (pi[i] e_j - pi[j] e_i),

and has exactly zero inner product with the deployed policy on every round.

The theorem returns the uniform average of all ``T`` pre-round policy *maps*, so
``fit_groupwise_self_play`` stores all ``T`` states by default.  Passing an
integer ``n_snapshots < T`` requests a stratified approximation; that output
is not covered by the guarantee, and the dense experiment
runner rejects it.

Only NumPy and SciPy are required.  Run this file directly to execute the
lightweight self-tests at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit

from empirical_testing.numerics import TINY as _TINY, log_sum_exp as _log_sum_exp


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


_PREDICTION_CHUNK_SIZE = 1024
#: Memory budget, per (block, active-rows, max(K, m)) float64 array, for
#: ``predict``'s snapshot-axis (``T``) blocking. Coffee's <=32 unique patterns never
#: approach this; datasets with ~1000+ unique membership patterns and
#: T in the thousands do, and without blocking predict would materialize
#: several full-``T`` arrays of that shape at once.
_PREDICT_T_BLOCK_BYTES_BUDGET = 32 * 1024 * 1024
_NUMERICAL_TOLERANCE = 1e-10


def _as_group_matrix(groups: ArrayLike, *, expected_rows: Optional[int] = None) -> FloatArray:
    """Validate and return a two-dimensional binary group-membership matrix."""

    matrix = np.asarray(groups, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("groups must have shape (n_rows, n_groups) with n_groups >= 1")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"groups has {matrix.shape[0]} rows, expected {expected_rows}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("groups contains a non-finite value")
    if not np.all((matrix == 0.0) | (matrix == 1.0)):
        raise ValueError("groups must contain only binary 0/1 memberships")
    return matrix


def _validate_reference(reference: ArrayLike) -> FloatArray:
    """Return a normalized, full-support reference lottery."""

    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 1 or ref.size < 2:
        raise ValueError("reference must be a one-dimensional lottery of length at least 2")
    if not np.all(np.isfinite(ref)) or np.any(ref <= 0.0):
        raise ValueError("reference must be finite and have full support")
    total = float(ref.sum())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"reference must sum to one, got {total}")
    return ref / total


def kl_divergence(q: ArrayLike, reference: ArrayLike) -> float:
    """Compute ``KL(q || reference)`` using the convention ``0 log 0 = 0``."""

    q_array = np.asarray(q, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if q_array.shape != ref.shape:
        raise ValueError("q and reference must have the same shape")
    positive = q_array > 0.0
    if positive.all():
        # The iterates this module produces are strictly positive, so the
        # ``0 log 0 = 0`` masking would select every coordinate.  Summing
        # the whole array is then a float64 reduction over the same values
        # in the same order as summing the (full-length) masked copy --
        # bit-identical, without materializing three boolean-indexed
        # temporaries on every call.
        return float(np.sum(q_array * (np.log(q_array) - np.log(ref))))
    return float(np.sum(q_array[positive] * (np.log(q_array[positive]) - np.log(ref[positive]))))


def _softmax_from_log(log_weights: FloatArray) -> FloatArray:
    """Normalize log weights without overflow and retain strict positivity."""

    values = np.exp(log_weights - _log_sum_exp(log_weights))
    # Exponentiated-gradient iterates are mathematically positive.  Protect
    # future logarithms if a negligible coordinate underflows in float64.
    values = np.maximum(values, _TINY)
    return values / values.sum()


def _kl_prox_update(
    q: FloatArray,
    loss: FloatArray,
    learning_rate: float,
    reference: FloatArray,
    tau: float,
) -> FloatArray:
    r"""Perform one entropic proximal update over a reverse-KL ball.

    This solves

    .. math::

        \arg\min_{p:\,KL(p\Vert\pi_{\rm ref})\le\tau}
        \eta\langle loss,p\rangle + KL(p\Vert q).

    If the unconstrained exponentiated-gradient update is infeasible, the
    Lagrange multiplier on the KL constraint is found by bisection using the
    one-dimensional form derived in the paper.
    """

    return _kl_prox_update_with_candidate(q, loss, learning_rate, reference, tau)[0]


def _kl_prox_update_with_candidate(
    q: FloatArray,
    loss: FloatArray,
    learning_rate: float,
    reference: FloatArray,
    tau: float,
) -> Tuple[FloatArray, Optional[FloatArray]]:
    """:func:`_kl_prox_update`, also returning the unconstrained candidate.

    The projected result is exactly what :func:`_kl_prox_update` returns --
    it is literally the same code path -- and the second element is the
    unconstrained exponentiated-gradient step
    ``softmax(log q - learning_rate * loss)`` that this routine forms and
    tests for feasibility on its way there.

    The span-adaptive inner loop needs that candidate for its projection
    diagnostic and used to recompute it from an inlined copy of the same
    expression, doubling the softmax evaluations in the hottest loop of the
    suite.  Handing back the value already computed here is the same
    arithmetic, not an approximation of it: the caller's
    ``softmax(log(max(anchor, tiny)) + rate * reward)`` and this function's
    ``softmax(log(max(q, tiny)) - rate * (-reward))`` differ only by
    IEEE-exact negations, so the two agree bit for bit.

    ``None`` is returned for the candidate in the singleton-trust-region
    branch (``tau <= _NUMERICAL_TOLERANCE``), which returns the reference
    without taking a step and so never forms one.  That branch is
    unreachable from the adaptive learner, which returns earlier when
    ``tau_bar`` is that small, but a caller must handle it rather than
    assume a value.
    """

    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    if tau <= _NUMERICAL_TOLERANCE:
        return reference.copy(), None
    if learning_rate < 0.0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and nonnegative")

    log_q = np.log(np.maximum(q, _TINY))
    log_ref = np.log(reference)
    unconstrained = _softmax_from_log(log_q - learning_rate * loss)

    maximum_kl = float(np.log(1.0 / np.min(reference)))
    if tau >= maximum_kl - _NUMERICAL_TOLERANCE:
        return unconstrained, unconstrained
    if kl_divergence(unconstrained, reference) <= tau + _NUMERICAL_TOLERANCE:
        return unconstrained, unconstrained

    def candidate(multiplier: float) -> FloatArray:
        inverse_scale = 1.0 / (1.0 + multiplier)
        candidate_log = (
            inverse_scale * (log_q - learning_rate * loss)
            + (1.0 - inverse_scale) * log_ref
        )
        return _softmax_from_log(candidate_log)

    lower = 0.0  # infeasible
    upper = 1.0
    while kl_divergence(candidate(upper), reference) > tau:
        upper *= 2.0
        if upper > 1e16:
            # This can occur only for a radius indistinguishable from zero at
            # machine precision; returning the reference is feasible.
            return reference.copy(), unconstrained

    def residual(multiplier: float) -> float:
        return kl_divergence(candidate(multiplier), reference) - tau

    # Brent's safeguarded interpolation usually needs far fewer projection
    # evaluations than a fixed 40-step bisection in the empirical sweeps.
    root = float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12))
    result = candidate(root)
    if kl_divergence(result, reference) > tau:
        # Move one representable step toward the feasible side of the root.
        result = candidate(float(np.nextafter(root, np.inf)))
    if kl_divergence(result, reference) > tau + 1e-8:
        raise AssertionError("KL-prox update failed to return a feasible point")
    return result, unconstrained


def _snapshot_schedule(
    horizon: int, n_snapshots: Optional[int], rng: np.random.Generator
) -> Tuple[dict[int, float], IntArray]:
    """Choose pre-round states and their online-to-batch weights.

    ``None`` means the theorem's exact uniform average over all ``T`` states.
    An integer smaller than ``T`` requests a stratified approximation that is
    not covered by the theorem's guarantee.
    """

    if n_snapshots is None:
        rounds = np.arange(horizon, dtype=np.int64)
        return {int(index): 1.0 / horizon for index in rounds}, rounds
    if n_snapshots <= 0:
        raise ValueError("n_snapshots must be positive")
    count = min(horizon, int(n_snapshots))
    selected: list[int] = []
    weights: list[float] = []
    for block in range(count):
        start = (block * horizon) // count
        stop = ((block + 1) * horizon) // count
        index = int(rng.integers(start, stop))
        selected.append(index)
        weights.append((stop - start) / float(horizon))
    schedule = {index: weight for index, weight in zip(selected, weights)}
    return schedule, np.asarray(selected, dtype=np.int64)


def _prepare_training_inputs(
    groups: ArrayLike,
    utilities: Optional[ArrayLike],
    beta: float,
    protocol: str,
    pair_indices: Optional[ArrayLike],
    signs: Optional[ArrayLike],
    propensities: Optional[ArrayLike],
    n_alternatives: Optional[int],
    reference: Optional[FloatArray] = None,
) -> Tuple[
    FloatArray,
    Optional[FloatArray],
    str,
    Optional[IntArray],
    Optional[FloatArray],
    Optional[FloatArray],
    int,
    float,
]:
    """Validate common fit inputs and determine the score's range bound."""

    group_matrix = _as_group_matrix(groups)
    horizon = group_matrix.shape[0]
    if horizon == 0:
        raise ValueError("at least one training row is required")
    if beta <= 0.0 or (not np.isfinite(beta) and not np.isposinf(beta)):
        raise ValueError("beta must be positive or +infinity")

    utility_matrix: Optional[FloatArray]
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

    normalized_protocol = protocol.lower().replace("_", "-")
    if normalized_protocol == "general":
        normalized_protocol = "logged"
    if normalized_protocol not in {"passive", "active", "logged"}:
        raise ValueError("protocol must be 'passive', 'active', or 'logged'")

    pairs: Optional[IntArray]
    if pair_indices is None:
        pairs = None
    else:
        pairs = np.asarray(pair_indices, dtype=np.int64)
        if pairs.shape != (horizon, 2):
            raise ValueError("pair_indices must have shape (n_rows, 2)")

    label_signs: Optional[FloatArray]
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

    propensity_array: Optional[FloatArray]
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

    if reference is not None:
        alternatives = int(reference.size)
    elif utility_matrix is not None:
        alternatives = int(utility_matrix.shape[1])
    elif n_alternatives is not None:
        alternatives = int(n_alternatives)
    elif pairs is not None:
        alternatives = int(np.max(pairs)) + 1
    else:
        raise ValueError(
            "cannot infer the number of alternatives; provide utilities or n_alternatives"
        )
    if alternatives < 2:
        raise ValueError("at least two alternatives are required")
    if n_alternatives is not None and int(n_alternatives) != alternatives:
        raise ValueError("n_alternatives is inconsistent with utilities/reference")
    if utility_matrix is not None and utility_matrix.shape[1] != alternatives:
        raise ValueError("utilities has the wrong number of alternatives")
    if pairs is not None:
        if np.any(pairs < 0) or np.any(pairs >= alternatives):
            raise ValueError("pair_indices contains an out-of-range alternative")
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("a pair must contain two distinct alternatives")

    if normalized_protocol == "active":
        if pairs is not None or label_signs is not None or propensity_array is not None:
            raise ValueError(
                "active comparisons are generated after seeing the current policy; "
                "do not pass precomputed pairs, signs, or propensities"
            )
        if utility_matrix is None:
            raise ValueError("active simulation requires utilities")
        reward_bound = float(alternatives - 1)
    elif normalized_protocol == "logged":
        if pairs is None or label_signs is None or propensity_array is None:
            raise ValueError(
                "logged feedback requires pair_indices, signs, and propensities"
            )
        reward_bound = float(np.max(1.0 / propensity_array))
    else:
        if pairs is None and propensity_array is not None:
            raise ValueError("propensities cannot be supplied without pair_indices")
        if pairs is None and utility_matrix is None:
            raise ValueError("passive simulation requires utilities")
        if label_signs is None and utility_matrix is None:
            raise ValueError("utilities is required to generate missing signs")
        if propensity_array is None:
            reward_bound = float(alternatives * (alternatives - 1) // 2)
        else:
            reward_bound = float(np.max(1.0 / propensity_array))

    return (
        group_matrix,
        utility_matrix,
        normalized_protocol,
        pairs,
        label_signs,
        propensity_array,
        alternatives,
        max(1.0, reward_bound),
    )


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
) -> Tuple[int, int, float, float]:
    """Return ``(i, j, sign, unordered-pair propensity)`` for one round."""

    alternatives = policy.size
    if protocol == "active":
        first = int(rng.choice(alternatives, p=policy))
        second = int(rng.integers(alternatives - 1))
        if second >= first:
            second += 1
        i, j = first, second
        propensity = float((policy[i] + policy[j]) / (alternatives - 1))
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
            raise AssertionError("utilities is needed to generate a preference label")
        difference = float(
            utilities[round_index, i] - utilities[round_index, j]
        )
        if np.isposinf(beta):
            # A score-order oracle returns the strict ordering.  Exact score
            # ties carry zero expected margin but still produce one comparison
            # bit, so break them with a seeded fair coin.
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


def _policy_weighted_score(
    policy: FloatArray,
    i: int,
    j: int,
    sign: float,
    propensity: float,
) -> FloatArray:
    """Construct the two-sparse score and verify exact self-play cancellation."""

    score = np.zeros_like(policy)
    scale = sign / propensity
    score[j] += scale * policy[i]
    score[i] -= scale * policy[j]
    cancellation = float(np.dot(policy, score))
    if abs(cancellation) > 1e-9 * max(1.0, float(np.max(np.abs(score)))):
        raise AssertionError("policy-weighted score failed exact cancellation")
    return score


@dataclass(frozen=True)
class FixedRates:
    """The constants of ``eq:rates`` and ``eq:complexity``, set before round 1.

    Every field is determined by quantities the learner is given -- the group
    masses, the horizon, the confidence level, the protocol scale and the
    reference lottery -- and none by observed data.  Storing them together lets
    a run record exactly which numbers governed it.
    """

    group_masses: FloatArray
    delta: float
    horizon: int
    score_bound: float
    tau_bar: float
    outer_complexity: float  # L_0 = log(eK)
    confidence_complexity: float  # L_delta = log(2K/delta)
    complexity: float  # H = tau_bar + L_0 + L_delta
    second_moment_bound: FloatArray  # V_bar_k
    outer: FloatArray  # eta_k
    inner: FloatArray  # theta_k

    @property
    def n_groups(self) -> int:
        return int(self.group_masses.size)

    @property
    def cap_binds(self) -> bool:
        """Whether the ``1/2`` cap in ``eq:rates`` is active for any group."""

        return bool(np.any(self.outer >= 0.5) or np.any(self.inner >= 0.5))


def fixed_rates(
    group_masses: ArrayLike,
    horizon: int,
    score_bound: float,
    tau: float,
    reference: ArrayLike,
    delta: float,
) -> FixedRates:
    """Evaluate ``eq:rates`` from the parameters the algorithm is given.

    ``group_masses`` are the known population masses ``p_k = P(G_k(X) = 1)``
    of ``sections/02-model.tex:7``.  They are a property of the deployment
    population and of the declared indicators, never estimated from the
    comparisons the rates then govern.
    """

    masses = np.asarray(group_masses, dtype=np.float64)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("group_masses must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(masses)):
        raise ValueError("group_masses contains a non-finite value")
    if np.any(masses <= 0.0) or np.any(masses > 1.0):
        raise ValueError("every group mass must lie in (0, 1]")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if score_bound < 1.0 or not np.isfinite(score_bound):
        raise ValueError("score_bound must be finite and at least one")

    reference_policy = _validate_reference(reference)
    if tau < 0.0 or not np.isfinite(tau):
        raise ValueError("tau must be finite and nonnegative")

    n_groups = int(masses.size)
    outer_complexity = float(np.log(np.e * n_groups))
    confidence_complexity = float(np.log(2.0 * n_groups / float(delta)))
    tau_bar = min(
        float(tau), float(np.log(1.0 / float(np.min(reference_policy))))
    )
    second_moment_bound = (
        2.0 * float(horizon) * masses / float(score_bound) + confidence_complexity
    )
    outer = np.minimum(0.5, np.sqrt(outer_complexity / second_moment_bound))
    if tau_bar <= _NUMERICAL_TOLERANCE:
        # The singleton trust region: the inner learner cannot move, and the
        # fitter returns the reference without consuming feedback.
        inner = np.zeros(n_groups, dtype=np.float64)
    else:
        inner = np.minimum(0.5, np.sqrt(tau_bar / second_moment_bound))
    return FixedRates(
        group_masses=masses,
        delta=float(delta),
        horizon=int(horizon),
        score_bound=float(score_bound),
        tau_bar=tau_bar,
        outer_complexity=outer_complexity,
        confidence_complexity=confidence_complexity,
        complexity=tau_bar + outer_complexity + confidence_complexity,
        second_moment_bound=second_moment_bound,
        outer=outer,
        inner=inner,
    )


@dataclass(frozen=True)
class AlignmentLotteryPolicy:
    """Snapshot-averaged output of the KL-constrained two-level learner.

    ``_outer_log_mass_snapshots``/``_inner_policy_snapshots`` are contiguous
    ``(S, K)``/``(S, K, m)`` arrays (``S`` = number of captured snapshots,
    row ``t`` = the pre-round state at ``snapshot_rounds[t]``), not a tuple
    of separate arrays (C5/D-303). Every observed access pattern -- ``len``,
    positive/negative indexing, iteration (yields axis-0 rows), ``zip``,
    ``np.stack`` -- is unchanged for a caller.
    """

    _outer_log_mass_snapshots: FloatArray
    _inner_policy_snapshots: FloatArray
    snapshot_weights: FloatArray
    snapshot_rounds: IntArray
    reference: FloatArray
    tau: float

    @property
    def n_groups(self) -> int:
        return int(self._inner_policy_snapshots[0].shape[0])

    @property
    def n_alternatives(self) -> int:
        return int(self.reference.size)

    def predict(
        self, group_matrix: ArrayLike, *, _t_block_override: Optional[int] = None
    ) -> FloatArray:
        """Evaluate the averaged policy; every returned row is KL-feasible.

        C5/D-303: vectorized over the snapshot axis ``T`` instead of a
        Python loop. Every step except the final fold is elementwise
        (masking, row-max, ``exp``, the small-``K``-axis row-sum) and is
        broadcast over ``T`` freely -- order does not matter for any of
        them. The fold over ``T`` is the one order-sensitive step and stays
        a strictly sequential accumulation (``np.cumsum(..., axis=0)[-1]``),
        matching the old Python ``+=`` loop bit-for-bit.

        The ``"tng,tgm->tnm"`` einsum below batches, over ``T``, the per-``t``
        ``"ng,gm->nm"`` contraction the old loop used. It is mathematically
        identical but NOT guaranteed bit-identical: NumPy may dispatch the
        batched and per-``t`` shapes to different BLAS kernels. It was bitwise
        equal in earlier environments, but under the
        2026-09-15 runtime (NumPy 2.3.5) it differs in the last bit for K > 1
        (worst observed 1.67e-16; K = 1 is exact). The regression suite
        therefore compares this path with the frozen per-``t`` reference to
        an absolute 2e-15 tolerance, while fitted snapshots, weights, learner
        state, repeat determinism and T-blocking are checked for exact equality.
        Rows whose pattern activates no group are folded once, since their
        per-``t`` value (the reference lottery) does not depend on the
        pattern at all; this is the same sequential fold, just shared across
        every always-inactive row instead of recomputed per chunk.

        To limit peak memory, the active branch below also blocks over ``T`` (in addition to the
        existing pattern-row chunking), so it never materializes a full-``T``
        intermediate. Each block's own fold is a plain
        ``np.cumsum(block, axis=0)[-1]``; blocks are then chained by
        *prepending* the running total as a phantom first row and reapplying
        ``cumsum`` (``np.cumsum(np.concatenate([acc[None], block]), axis=0)[-1]``)
        rather than pre-summing each block and adding the two sums. The two
        are not interchangeable: prepending replays the exact same sequence
        of pairwise additions a monolithic whole-``T`` cumsum would perform
        (each step's left operand is the previous step's exact result), so it
        is bit-identical to the unblocked fold by induction; independently
        summing each block and adding the block sums is not (double
        rounding: ``(acc + x1) + x2`` need not equal ``acc + (x1 + x2)``),
        which is why that alternative was rejected during the original
        design pass. ``_t_block_override`` is a private test-only hook to
        force a specific block size (e.g. 1, or an odd number) regardless of
        ``_PREDICT_T_BLOCK_BYTES_BUDGET``; production callers never pass it.
        """

        groups = _as_group_matrix(group_matrix)
        if groups.shape[1] != self.n_groups:
            raise ValueError(
                f"group_matrix has {groups.shape[1]} columns, expected {self.n_groups}"
            )
        # The policy depends on a context only through its binary membership
        # pattern.  Collapsing repeated patterns is exact and keeps the required
        # all-T checkpoint average practical on real datasets.
        unique_groups, inverse = np.unique(groups, axis=0, return_inverse=True)
        output = np.zeros(
            (unique_groups.shape[0], self.n_alternatives), dtype=np.float64
        )

        outer_snapshots = self._outer_log_mass_snapshots  # (S, K)
        inner_snapshots = self._inner_policy_snapshots  # (S, K, m)
        weights = self.snapshot_weights  # (S,)
        n_snapshots = outer_snapshots.shape[0]
        max_dim = max(self.n_groups, self.n_alternatives)
        # Shared fold for every row that never activates a group: identical
        # for all such rows and independent of the pattern chunk. (S, m) is
        # small even at large S/m, so this one is never T-blocked.
        reference_fold = np.cumsum(
            weights[:, None] * self.reference[None, :], axis=0
        )[-1]

        for start in range(0, unique_groups.shape[0], _PREDICTION_CHUNK_SIZE):
            stop = min(unique_groups.shape[0], start + _PREDICTION_CHUNK_SIZE)
            membership = unique_groups[start:stop].astype(bool)
            active = np.any(membership, axis=1)
            chunk_output = np.tile(reference_fold, (stop - start, 1))
            if np.any(active):
                active_membership = membership[active]
                n_active = active_membership.shape[0]
                if _t_block_override is not None:
                    t_block = max(1, int(_t_block_override))
                else:
                    per_t_bytes = 8 * max(1, n_active) * max(1, max_dim)
                    t_block = max(1, _PREDICT_T_BLOCK_BYTES_BUDGET // per_t_bytes)
                acc: Optional[FloatArray] = None
                for t_start in range(0, n_snapshots, t_block):
                    t_stop = min(n_snapshots, t_start + t_block)
                    outer_block = outer_snapshots[t_start:t_stop]
                    inner_block = inner_snapshots[t_start:t_stop]
                    weight_block = weights[t_start:t_stop]
                    selected_log_mass = np.where(
                        active_membership[None, :, :],
                        outer_block[:, None, :],
                        -np.inf,
                    )
                    row_maximum = np.max(selected_log_mass, axis=2)
                    group_mass = np.exp(selected_log_mass - row_maximum[:, :, None])
                    weighted_policies = np.einsum(
                        "tng,tgm->tnm",
                        group_mass,
                        inner_block,
                        optimize=True,
                    )
                    block_prediction = weighted_policies / group_mass.sum(
                        axis=2, keepdims=True
                    )
                    weighted_block = weight_block[:, None, None] * block_prediction
                    if acc is None:
                        acc = np.cumsum(weighted_block, axis=0)[-1]
                    else:
                        acc = np.cumsum(
                            np.concatenate([acc[None, :, :], weighted_block], axis=0),
                            axis=0,
                        )[-1]
                chunk_output[active] = acc
            output[start:stop] = chunk_output

        output = np.maximum(output, 0.0)
        output /= output.sum(axis=1, keepdims=True)
        for row in output:
            if kl_divergence(row, self.reference) > self.tau + 2e-7:
                raise AssertionError("snapshot-averaged alignment prediction is infeasible")
        return output[inverse]


@dataclass(frozen=True)
class GroupwiseSelfPlayPolicy(AlignmentLotteryPolicy):
    """Output and auditable terminal state of Algorithm 1.

    The snapshot masses already include the fixed outer learning rate, i.e.
    they store ``log(eta_k w_k)``.  The inherited :meth:`predict` therefore
    reconstructs exactly the active-group mixture of ``eq:deployment``.

    ``outer_second_moments`` and ``inner_second_moments`` are diagnostics only.
    The rates were fixed before round 1 and never consulted them.
    """

    final_outer_log_masses: FloatArray
    outer_second_moments: FloatArray
    inner_second_moments: FloatArray
    group_activation_counts: IntArray
    rates: FixedRates
    score_bound: float
    max_outer_mass_error: float
    #: (T, 4) array of the comparisons consumed, one row per round:
    #: ``[i, j, sign, propensity]``. Under the logged protocol this echoes the
    #: input; under the active protocol it is the only record of what was
    #: queried, so it must be saved for pair-level reproducibility.
    drawn_comparisons: FloatArray

    @property
    def normalized_inner_second_moments(self) -> FloatArray:
        """``V_{T,k}``: the inner moment on the scale of ``eq:inner-regret``.

        The accumulator sums ``sum_b mu_b ghat_b^2``; the regret bound is
        stated for losses ``ghat / R``, so the comparable quantity -- and the
        one that ``second_moment_bound`` bounds -- divides by ``R^2``.
        """

        return self.inner_second_moments / self.score_bound**2


def fit_groupwise_self_play(
    groups: ArrayLike,
    utilities: Optional[ArrayLike] = None,
    beta: float = 1.0,
    reference: Optional[ArrayLike] = None,
    tau: float = 0.0,
    protocol: str = "passive",
    seed: int = 0,
    n_snapshots: Optional[int] = None,
    pair_indices: Optional[ArrayLike] = None,
    signs: Optional[ArrayLike] = None,
    propensities: Optional[ArrayLike] = None,
    *,
    group_masses: ArrayLike,
    delta: float = 0.05,
) -> GroupwiseSelfPlayPolicy:
    """Fit Algorithm 1 of ``sections/04-algorithm.tex``.

    ``group_masses`` and ``delta`` are keyword-only and ``group_masses`` has no
    default: the rates of ``eq:rates`` are meaningless without them, and a
    silent fallback would let a caller run a different algorithm than it
    believes.  With the default ``n_snapshots=None`` the returned policy is the
    exact uniform average of all ``T`` pre-round policy maps.
    """

    if reference is None:
        if utilities is None:
            raise ValueError("reference is required when utilities is absent")
        utility_array = np.asarray(utilities)
        if utility_array.ndim != 2 or utility_array.shape[1] < 2:
            raise ValueError("cannot infer a uniform reference from utilities")
        reference_policy = np.full(
            utility_array.shape[1], 1.0 / utility_array.shape[1]
        )
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
    ) = _prepare_training_inputs(
        groups,
        utilities,
        beta,
        protocol,
        pair_indices,
        signs,
        propensities,
        reference_policy.size,
        reference=reference_policy,
    )
    horizon, n_groups = group_matrix.shape
    if horizon < 2:
        raise ValueError("the algorithm requires at least two rounds")

    rates = fixed_rates(
        group_masses, horizon, score_bound, tau, reference_policy, delta
    )
    if rates.n_groups != n_groups:
        raise ValueError(
            f"group_masses has {rates.n_groups} entries, expected {n_groups}"
        )

    seed_sequence = np.random.SeedSequence(seed)
    training_seed, snapshot_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(training_seed)
    snapshot_rng = np.random.default_rng(snapshot_seed)
    schedule, snapshot_rounds = _snapshot_schedule(
        horizon, n_snapshots, snapshot_rng
    )

    log_initial_mass = -np.log(float(n_groups))
    log_outer_mass = np.full(n_groups, log_initial_mass, dtype=np.float64)
    # Constant for the whole run, so the deployment offset is hoisted here.
    log_outer_rate = np.log(rates.outer)
    outer_moment = np.zeros(n_groups, dtype=np.float64)
    inner_moment = np.zeros(n_groups, dtype=np.float64)
    policies = np.tile(reference_policy, (n_groups, 1))
    activation_counts = np.zeros(n_groups, dtype=np.int64)
    max_outer_mass_error = 0.0

    if rates.tau_bar <= _NUMERICAL_TOLERANCE:
        # The singleton trust region is handled separately by the manuscript:
        # return the reference without consuming feedback or updating Q/V.  We
        # still materialize the checkpoint representation so the exact-output
        # runner can audit all T equal pre-round policies.
        effective_log_mass = log_outer_mass + log_outer_rate
        n_snap = snapshot_rounds.size
        outer_snapshots = np.tile(effective_log_mass, (n_snap, 1))
        inner_snapshots = np.tile(policies, (n_snap, 1, 1))
        weights = np.asarray(
            [schedule[int(round_index)] for round_index in snapshot_rounds],
            dtype=np.float64,
        )
        weights /= weights.sum()
        return GroupwiseSelfPlayPolicy(
            _outer_log_mass_snapshots=outer_snapshots,
            _inner_policy_snapshots=inner_snapshots,
            snapshot_weights=weights,
            snapshot_rounds=snapshot_rounds,
            reference=reference_policy,
            tau=float(tau),
            final_outer_log_masses=log_outer_mass.copy(),
            outer_second_moments=outer_moment.copy(),
            inner_second_moments=inner_moment.copy(),
            group_activation_counts=activation_counts.copy(),
            rates=rates,
            score_bound=float(score_bound),
            max_outer_mass_error=0.0,
            drawn_comparisons=np.zeros((0, 4), dtype=np.float64),
        )

    all_pairs = np.asarray(
        [(i, j) for i in range(alternatives) for j in range(i + 1, alternatives)],
        dtype=np.int64,
    )
    n_snapshots = snapshot_rounds.size
    outer_snapshots_buf = np.empty((n_snapshots, n_groups), dtype=np.float64)
    inner_snapshots_buf = np.empty((n_snapshots, n_groups, alternatives), dtype=np.float64)
    snapshot_weights: list[float] = []
    snapshot_position = 0
    drawn = np.full((horizon, 4), np.nan, dtype=np.float64)

    for round_index in range(horizon):
        effective_log_mass = log_outer_mass + log_outer_rate

        if round_index in schedule:
            # Assignment into a buffer row already copies; no .copy() needed.
            outer_snapshots_buf[snapshot_position] = effective_log_mass
            inner_snapshots_buf[snapshot_position] = policies
            snapshot_position += 1
            snapshot_weights.append(schedule[round_index])

        active_groups = np.flatnonzero(group_matrix[round_index]).astype(np.int64)
        if active_groups.size == 0:
            continue
        activation_counts[active_groups] += 1
        active_probabilities = _softmax_from_log(
            effective_log_mass[active_groups]
        )
        proposals = policies[active_groups]
        policy = active_probabilities @ proposals
        policy = np.maximum(policy, 0.0)
        policy /= policy.sum()
        if kl_divergence(policy, reference_policy) > tau + 2e-8:
            raise AssertionError("deployed mixture is outside the KL ball")

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
        score = _policy_weighted_score(policy, i, j, sign, propensity)
        rewards = -(proposals @ score)
        # eq:mass-cancellation, in its normalized form.
        if abs(float(np.dot(active_probabilities, rewards))) > (
            2e-9 * max(1.0, score_bound)
        ):
            raise AssertionError("outer reward should cancel to zero")

        normalized_reward = rewards / score_bound
        if np.any(np.abs(normalized_reward) > 1.0 + 1e-9):
            raise AssertionError("normalized outer reward left [-1, 1]")

        # eq:outer-update.  A plain Prod step: with a fixed rate there is
        # nothing to anchor toward the initialization.  Since eta <= 1/2 and
        # |r| <= 1 the factor is at least 1/2, so weights stay positive.
        factors = rates.outer[active_groups] * normalized_reward
        if np.any(factors <= -1.0):
            raise AssertionError("Prod update produced a nonpositive weight")
        log_outer_mass[active_groups] += np.log1p(factors)
        # eq:mass-cancellation implies the total outer mass is conserved
        # exactly.  This holds only because the rates are fixed.
        max_outer_mass_error = max(
            max_outer_mass_error, abs(float(_log_sum_exp(log_outer_mass)))
        )

        # Diagnostics only: neither moment feeds a rate.
        outer_moment[active_groups] += normalized_reward * normalized_reward
        inner_moment[active_groups] += np.einsum(
            "gm,m->g", proposals, score * score, optimize=True
        )

        # eq:inner-update: an entropic step from mu_{t,k} itself followed by
        # the KL projection.  No stabilization toward the reference.
        loss = score / score_bound
        for group in active_groups:
            index = int(group)
            policies[index] = _kl_prox_update(
                policies[index],
                loss,
                float(rates.inner[index]),
                reference_policy,
                tau,
            )

    weights = np.asarray(snapshot_weights, dtype=np.float64)
    weights /= weights.sum()
    if snapshot_position != snapshot_rounds.size:
        raise AssertionError("failed to capture every snapshot")
    if max_outer_mass_error > 1e-9:
        raise AssertionError(
            "total outer mass drifted from one by "
            f"{max_outer_mass_error:.3e}; eq:mass-cancellation is violated"
        )
    return GroupwiseSelfPlayPolicy(
        _outer_log_mass_snapshots=outer_snapshots_buf,
        _inner_policy_snapshots=inner_snapshots_buf,
        snapshot_weights=weights,
        snapshot_rounds=snapshot_rounds,
        reference=reference_policy,
        tau=float(tau),
        final_outer_log_masses=log_outer_mass.copy(),
        outer_second_moments=outer_moment.copy(),
        inner_second_moments=inner_moment.copy(),
        group_activation_counts=activation_counts.copy(),
        rates=rates,
        score_bound=float(score_bound),
        max_outer_mass_error=max_outer_mass_error,
        drawn_comparisons=drawn,
    )


def _run_self_tests() -> None:
    """Exercise the score, the projection, the fixed rates, and the fitter."""

    rng = np.random.default_rng(7)
    policy = rng.dirichlet(np.ones(4))
    score = _policy_weighted_score(policy, 0, 3, 1.0, 1.0 / 6.0)
    assert abs(float(np.dot(policy, score))) < 1e-12

    reference = np.full(4, 0.25)
    prox = _kl_prox_update(
        reference,
        np.asarray([-2.0, 0.3, 0.8, 1.1]),
        0.4,
        reference,
        0.08,
    )
    assert np.isclose(prox.sum(), 1.0)
    assert kl_divergence(prox, reference) <= 0.08 + 1e-8
    assert prox[0] > reference[0]

    # eq:rates against hand-evaluated constants for the Coffee configuration.
    coffee = fixed_rates(
        [1.0, 0.144498, 0.157585, 0.545139, 0.577991],
        horizon=2995,
        score_bound=6.0,
        tau=0.5,
        reference=np.full(4, 0.25),
        delta=0.05,
    )
    assert np.isclose(coffee.outer_complexity, np.log(np.e * 5))
    assert np.isclose(coffee.confidence_complexity, np.log(200.0))
    assert np.isclose(coffee.tau_bar, 0.5)
    assert np.isclose(coffee.outer[0], 0.050990, atol=1e-6)
    assert np.isclose(coffee.inner[0], 0.022320, atol=1e-6)
    # A rare group takes a larger step: the rate scales as 1/sqrt(T p_k).
    assert coffee.inner[1] > 2.5 * coffee.inner[0]
    assert not coffee.cap_binds

    # The fitter must stay feasible, update only active groups, move toward the
    # consistently preferred action, and conserve total outer mass exactly.
    utilities = np.tile(np.asarray([[1.0, 0.2, 0.0]]), (80, 1))
    groups = np.column_stack(
        [np.ones(80, dtype=np.int8), np.arange(80) % 2 == 0]
    )
    fitted = fit_groupwise_self_play(
        groups,
        utilities,
        beta=np.inf,
        reference=np.full(3, 1.0 / 3.0),
        tau=0.25,
        protocol="passive",
        seed=11,
        n_snapshots=80,
        group_masses=[1.0, 0.5],
    )
    predicted = fitted.predict(groups[:4])
    assert np.allclose(predicted.sum(axis=1), 1.0)
    assert all(
        kl_divergence(row, fitted.reference) <= fitted.tau + 1e-8
        for row in predicted
    )
    assert float(predicted[:, 0].mean()) > float(predicted[:, 2].mean())
    assert fitted.group_activation_counts.tolist() == [80, 40]
    assert fitted.max_outer_mass_error < 1e-12
    assert np.all(
        fitted.normalized_inner_second_moments <= fitted.rates.second_moment_bound
    )

    # Deleting the geometric anchor means a group facing a constant loss is no
    # longer dragged back toward the reference between rounds.
    horizon = 60
    logged_groups = np.ones((horizon, 1), dtype=np.float64)
    logged_pairs = np.tile(np.asarray([[0, 1]], dtype=np.int64), (horizon, 1))
    logged_signs = np.ones(horizon, dtype=np.float64)
    logged_propensities = np.full(horizon, 1.0 / 3.0)
    drift = fit_groupwise_self_play(
        logged_groups,
        utilities=None,
        reference=np.full(3, 1.0 / 3.0),
        tau=0.2,
        protocol="logged",
        seed=5,
        n_snapshots=None,
        pair_indices=logged_pairs,
        signs=logged_signs,
        propensities=logged_propensities,
        group_masses=[1.0],
    )
    iterates = np.asarray(
        [snapshot[0] for snapshot in drift._inner_policy_snapshots]
    )
    divergences = np.asarray(
        [kl_divergence(row, drift.reference) for row in iterates]
    )
    # Only while the ball is slack does the step reduce to plain exponentiated
    # gradient.  Once the constraint binds the projection legitimately trades
    # mass into the zero-loss coordinate instead, so monotonicity in the
    # preferred coordinate is asserted on the interior only.
    interior = int(np.argmax(divergences >= drift.tau - 1e-6))
    assert interior > 20, "the ball bound too early to test the interior"
    preferred = iterates[:interior, 0]
    assert np.all(np.diff(preferred) > 0.0), "challenger was dragged back"
    assert preferred[-1] > 1.5 * preferred[0]

    # The active path must generate its pair after seeing the current policy.
    active_utilities = np.zeros((120, 3), dtype=np.float64)
    active_utilities[:, 0] = 1.0
    active = fit_groupwise_self_play(
        np.ones((120, 1)),
        active_utilities,
        beta=3.0,
        reference=np.full(3, 1.0 / 3.0),
        tau=0.2,
        protocol="active",
        seed=17,
        n_snapshots=12,
        group_masses=[1.0],
    )
    active_prediction = active.predict(np.ones((2, 1)))
    assert np.all(np.isfinite(active_prediction))
    assert np.allclose(active_prediction.sum(axis=1), 1.0)


if __name__ == "__main__":
    _run_self_tests()
    print("one_expert.py self-tests passed")
