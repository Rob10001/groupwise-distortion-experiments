"""Groupwise self-play with span-adaptive learning rates.

This module is the adaptive-rate sibling of :mod:`one_expert`.  It keeps every
structural element of Algorithm 1 -- one outer Prod weight ``w_k`` and one
reverse-KL-constrained inner challenger ``pi_k`` per group, the deployed lottery
mixing the challengers of the active groups, one comparison per round, the exact
uniform average of all ``T`` pre-round policy maps -- but sets both learning
rates from trajectory statistics instead of from the a-priori bound ``V_bar_k``::

    L        = max(1, log(K (1 + 0.5 log(T + 1))))
    eta_k    = min(1/2, sqrt(L / (2 (1 + Q_k))))          # outer, Q_k = sum r_k^2
    theta_k  = min(1,   sqrt(tau_bar / (2 (1/4 + C_k))))  # inner, C_k = sum Var_{pi_k}(g)

with ``tau_bar = min(tau, log(1 / min pi_ref))`` the effective radius.  The
reward vector is the normalized, positive-signed version of the score used by
the fixed learner, ``g = -score / R``; it satisfies ``pi_hat . g = 0`` pathwise,
``|g_a| <= 1`` and ``span(g) <= 1`` under the passive uniform and the active
protocols.  Because a rate can decrease, the outer update anchors toward the
uniform initialization and the inner update toward the reference, each with the
ratio ``gamma = rate_next / rate`` of consecutive rates::

    log w_next  = (1 - gamma_eta) (-log K) + gamma_eta (log w + log1p(eta r))
    pi_next     = argmin_{q in B_tau} KL(q || softmax(gamma_theta log pi
                                          + (1 - gamma_theta) log pi_ref + theta_next g))

The exponential uses the *next* inner rate; deployment and the tentative Prod
step use the *current* outer rate.  Outer weights are not renormalized; their
total mass is recorded as a diagnostic.  The fitter takes no group masses and no
confidence level: those are evaluation inputs.

``frozen_rates`` supports regression tests: constant rates set both ``gamma``
values to one, recovering the fixed learner's recurrence. Score, comparison,
KL-projection, and policy-averaging helpers come from :mod:`one_expert`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from empirical_testing.numerics import log_sum_exp as _log_sum_exp
from empirical_testing.algorithms.one_expert import (
    _NUMERICAL_TOLERANCE,
    AlignmentLotteryPolicy,
    FloatArray,
    IntArray,
    _draw_comparison,
    _kl_prox_update_with_candidate,
    _policy_weighted_score,
    _prepare_training_inputs,
    _snapshot_schedule,
    _softmax_from_log,
    _validate_reference,
    kl_divergence,
)

ALGORITHM_ID = "groupwise_self_play_span_adaptive"
OUTER_RATE_CAP = 0.5
INNER_RATE_CAP = 1.0
OUTER_DENOMINATOR_OFFSET = 1.0
INNER_DENOMINATOR_OFFSET = 0.25
_TRACE_POINTS = 40
_VALID_KERNEL_BACKENDS = ("python", "numba")


def _resolve_kernel_backend(kernel_backend: Optional[str]):
    """Resolve the per-fit kernel backend once, before round 1.

    Precedence: the explicit ``kernel_backend`` argument, else the
    ``GWD_CPU_KERNELS`` environment variable, else the default, which is
    ``"numba"``. The trusted Numba backend is the frozen CPU production
    default and ``numba``/``llvmlite`` are declared dependencies of this
    package. ``"python"`` remains available explicitly, by
    argument or ``GWD_CPU_KERNELS=python``, as the reference implementation
    and fallback. Unknown names raise :class:`ValueError`.

    If the resolved name is ``"numba"`` -- whether by explicit request or by
    falling through to the default -- and numba is not importable in this
    environment, this raises :class:`RuntimeError` immediately, naming both
    remedies (install the declared ``numba`` dependency in this venv, or
    select the ``"python"`` backend explicitly). There is no silent
    fallback to Python: since the dependency is declared, a resolution that
    can't be honoured is an environment defect to surface, not a numerics
    choice to paper over, and a run must never silently get slower (or
    faster) numerics than the caller asked for.

    Returns ``(name, compiled_kernels_module_or_None)``: the module is
    imported here, lazily, only when a backend other than ``"python"`` is
    actually resolved, so an environment without numba installed (the
    ``"python"`` reference/baseline venv) never needs to import it as long
    as it only ever selects ``"python"`` explicitly.
    """

    name = kernel_backend if kernel_backend is not None else os.environ.get("GWD_CPU_KERNELS", "numba")
    if name not in _VALID_KERNEL_BACKENDS:
        raise ValueError(
            f"unknown kernel_backend {name!r}; expected one of {_VALID_KERNEL_BACKENDS}"
        )
    if name == "python":
        return name, None
    from empirical_testing import compiled_kernels  # local: keeps numba optional

    if not compiled_kernels.is_numba_available():
        raise RuntimeError(
            "kernel_backend='numba' was resolved (explicitly, via "
            "GWD_CPU_KERNELS, or as the default -- see D-502), but numba is "
            "not importable in this environment. Remedies: (1) install "
            "numba, the declared dependency of this package (see "
            "pyproject.toml), in "
            "this venv; or (2) select the 'python' backend explicitly, by "
            "the kernel_backend argument or GWD_CPU_KERNELS=python."
        )
    return name, compiled_kernels


def complexity_term(n_groups: int, horizon: int) -> float:
    """``L = max(1, log(K (1 + 0.5 log(T + 1))))``."""

    return float(max(1.0, np.log(n_groups * (1.0 + 0.5 * np.log(horizon + 1.0)))))


def effective_radius(tau: float, reference: FloatArray) -> float:
    """``tau_bar = min(tau, log(1 / min pi_ref))`` (eq:radius-cap)."""

    return float(min(tau, np.log(1.0 / float(np.min(reference)))))


@dataclass(frozen=True)
class SpanAdaptiveSchedule:
    """The two rate maps; both depend only on the accumulated trajectory statistic."""

    horizon: int
    n_groups: int
    score_bound: float
    tau_requested: float
    tau_bar: float
    complexity: float  # L

    def outer_rate(self, outer_moment: ArrayLike) -> FloatArray:
        q = np.asarray(outer_moment, dtype=np.float64)
        return np.minimum(
            OUTER_RATE_CAP, np.sqrt(self.complexity / (2.0 * (OUTER_DENOMINATOR_OFFSET + q)))
        )

    def inner_rate(self, centered_moment: ArrayLike) -> FloatArray:
        c = np.asarray(centered_moment, dtype=np.float64)
        return np.minimum(
            INNER_RATE_CAP, np.sqrt(self.tau_bar / (2.0 * (INNER_DENOMINATOR_OFFSET + c)))
        )


@dataclass(frozen=True)
class FrozenRates:
    """Test hook: constant rates make the adaptive kernels the fixed recurrence."""

    outer: FloatArray
    inner: FloatArray


@dataclass(frozen=True)
class SpanAdaptiveDiagnostics:
    """Terminal and trace state of one fit, all of it empirical."""

    algorithm: str
    L: float
    tau_requested: float
    tau_bar: float
    R: float
    T: int
    K: int
    eta_initial: float
    theta_initial: float
    eta_terminal: FloatArray
    theta_terminal: FloatArray
    eta_min: FloatArray
    theta_min: FloatArray
    Q: FloatArray
    C: FloatArray
    V: FloatArray
    identity_residual_max: float
    activation_counts: IntArray
    outer_mass_terminal: float
    outer_mass_min: float
    outer_mass_max: float
    projection_counts: IntArray
    update_counts: IntArray
    max_span: float
    max_abs_reward: float
    max_abs_cancellation: float
    kl_inner_max: float
    kl_deployed_max: float
    first_round_eta_below_cap: IntArray
    first_round_theta_below_one: IntArray
    frozen: bool
    rate_trace: dict = field(default_factory=dict)
    #: Which per-round inner-update implementation this fit resolved
    #: (``"python"`` or ``"numba"``, see ``fit_groupwise_span_adaptive``'s
    #: ``kernel_backend`` argument). Recorded in the diagnostics JSON.
    kernel_backend: str = "python"

    def as_json(self) -> dict:
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                out[key] = value.item()
            else:
                out[key] = value
        return out


@dataclass(frozen=True)
class GroupwiseSpanAdaptivePolicy(AlignmentLotteryPolicy):
    """Output of the span-adaptive learner: the exact all-round average plus its audit trail.

    The inherited :meth:`predict` reconstructs ``pi_bar_T(x) = (1/T) sum_t pi_hat_t(x)``
    from the stored pre-round maps ``(log(eta_t w_t), pi_t)``; the snapshot masses
    include the round's *current* outer rate, exactly as the fixed learner stores
    ``log(eta_k w_k)``.
    """

    final_outer_log_masses: FloatArray
    outer_second_moments: FloatArray  # Q
    centered_inner_second_moments: FloatArray  # C
    inner_second_moments: FloatArray  # V, already on the normalized scale of g
    group_activation_counts: IntArray
    schedule: SpanAdaptiveSchedule
    diagnostics: SpanAdaptiveDiagnostics
    score_bound: float
    drawn_comparisons: FloatArray
    algorithm: str = ALGORITHM_ID

    @property
    def normalized_inner_second_moments(self) -> FloatArray:
        """Same name as the fixed learner's property; here V is already normalized."""

        return self.inner_second_moments


def _trace_rounds(horizon: int) -> IntArray:
    if horizon <= _TRACE_POINTS:
        return np.arange(horizon, dtype=np.int64)
    grid = np.unique(np.round(np.geomspace(1, horizon, _TRACE_POINTS)).astype(np.int64) - 1)
    return np.unique(np.concatenate([[0], grid, [horizon - 1]])).astype(np.int64)


def fit_groupwise_span_adaptive(
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
    frozen_rates: Optional[FrozenRates] = None,
    kernel_backend: Optional[str] = None,
) -> GroupwiseSpanAdaptivePolicy:
    """Fit the span-adaptive groupwise self-play learner.

    The signature mirrors :func:`fit_groupwise_self_play` without ``group_masses``
    and ``delta``: the adaptive schedule uses neither.  ``n_snapshots`` must be
    ``None`` (the exact all-round average); it is accepted only for signature
    parity and rejected otherwise.

    ``kernel_backend`` selects, once for the whole fit, which implementation
    computes the per-group inner update (``one_expert.py``'s ``anchor`` /
    KL-prox-update / ``kl_divergence`` sequence). ``"python"`` is the plain,
    unmodified NumPy/SciPy reference path. ``"numba"`` is the default and
    dispatches feasible per-group
    updates to a compiled kernel (``compiled_kernels.
    span_adaptive_inner_round_numba``) that is bit-identical to the Python
    path on every input it accepts, falling back to the same Python code for
    the rare group whose KL projection binds or whose softmax hits a
    non-finite max. If neither this argument nor ``GWD_CPU_KERNELS`` is set,
    the resolved backend is ``"numba"``; requesting ``"numba"`` when numba
    is not importable raises :class:`RuntimeError` rather than silently
    running the Python path, and any other name raises :class:`ValueError`.
    """

    if n_snapshots is not None:
        raise ValueError("the span-adaptive learner returns the exact all-round average only")
    resolved_kernel_backend, _compiled_kernels = _resolve_kernel_backend(kernel_backend)
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

    tau_bar = effective_radius(float(tau), reference_policy)
    schedule = SpanAdaptiveSchedule(
        horizon=int(horizon),
        n_groups=int(n_groups),
        score_bound=float(score_bound),
        tau_requested=float(tau),
        tau_bar=tau_bar,
        complexity=complexity_term(n_groups, horizon),
    )
    frozen = frozen_rates is not None
    if frozen:
        frozen_outer = np.asarray(frozen_rates.outer, dtype=np.float64)
        frozen_inner = np.asarray(frozen_rates.inner, dtype=np.float64)
        if frozen_outer.shape != (n_groups,) or frozen_inner.shape != (n_groups,):
            raise ValueError("frozen_rates must supply one outer and one inner rate per group")
        if np.any(frozen_outer <= 0.0) or np.any(frozen_outer > 0.5):
            raise ValueError("frozen outer rates must satisfy 0 < eta <= 1/2")
        # A zero frozen inner rate is rejected rather than tolerated: the
        # anchored update forms ``gamma_theta = theta_next / theta_active``, and
        # with a constant zero rate that is ``0 / 0 = nan``, which would then
        # propagate silently through the geometric anchor into every subsequent
        # policy.  A caller that wants a frozen learner with no inner motion
        # should ask for ``tau = 0``, which the fitter handles explicitly by
        # returning the reference.
        if np.any(frozen_inner <= 0.0):
            raise ValueError(
                "frozen inner rates must be strictly positive; a zero rate makes "
                "gamma_theta = 0/0. Use tau = 0 for a learner that cannot move."
            )
        if np.any(frozen_inner > INNER_RATE_CAP):
            raise ValueError(f"frozen inner rates must satisfy theta <= {INNER_RATE_CAP}")

    # RNG: same derivation as the fixed learner so active draws coincide under
    # the frozen hook.  The snapshot generator is unused (exact schedule).
    seed_sequence = np.random.SeedSequence(seed)
    training_seed, snapshot_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(training_seed)
    snapshot_rng = np.random.default_rng(snapshot_seed)
    weights_schedule, snapshot_rounds = _snapshot_schedule(horizon, None, snapshot_rng)

    log_initial_mass = -np.log(float(n_groups))
    log_outer_mass = np.full(n_groups, log_initial_mass, dtype=np.float64)
    policies = np.tile(reference_policy, (n_groups, 1))
    outer_moment = np.zeros(n_groups)  # Q
    centered_moment = np.zeros(n_groups)  # C
    raw_moment = np.zeros(n_groups)  # V
    activation_counts = np.zeros(n_groups, dtype=np.int64)
    projection_counts = np.zeros(n_groups, dtype=np.int64)
    update_counts = np.zeros(n_groups, dtype=np.int64)
    first_eta_below = np.full(n_groups, -1, dtype=np.int64)
    first_theta_below = np.full(n_groups, -1, dtype=np.int64)

    def current_rates():
        if frozen:
            return frozen_outer.copy(), frozen_inner.copy()
        return schedule.outer_rate(outer_moment), schedule.inner_rate(centered_moment)

    eta0, theta0 = current_rates()
    eta_min = eta0.copy()
    theta_min = theta0.copy()

    if tau_bar <= _NUMERICAL_TOLERANCE:
        # Singleton trust region: return the reference, consume nothing.
        effective_log_mass = log_outer_mass + np.log(eta0)
        n_snap = snapshot_rounds.size
        outer_snapshots = np.tile(effective_log_mass, (n_snap, 1))
        inner_snapshots = np.tile(policies, (n_snap, 1, 1))
        weights = np.asarray([weights_schedule[int(r)] for r in snapshot_rounds])
        weights /= weights.sum()
        diagnostics = SpanAdaptiveDiagnostics(
            algorithm=ALGORITHM_ID, L=schedule.complexity, tau_requested=float(tau), tau_bar=tau_bar,
            R=float(score_bound), T=int(horizon), K=int(n_groups),
            eta_initial=float(eta0[0]), theta_initial=float(theta0[0]),
            eta_terminal=eta0.copy(), theta_terminal=theta0.copy(),
            eta_min=eta0.copy(), theta_min=theta0.copy(),
            Q=outer_moment.copy(), C=centered_moment.copy(), V=raw_moment.copy(),
            identity_residual_max=0.0, activation_counts=activation_counts.copy(),
            outer_mass_terminal=1.0, outer_mass_min=1.0, outer_mass_max=1.0,
            projection_counts=projection_counts.copy(), update_counts=update_counts.copy(),
            max_span=0.0, max_abs_reward=0.0, max_abs_cancellation=0.0,
            kl_inner_max=0.0, kl_deployed_max=0.0,
            first_round_eta_below_cap=first_eta_below.copy(),
            first_round_theta_below_one=first_theta_below.copy(),
            frozen=frozen, rate_trace={"rounds": [], "eta": [], "theta": [], "Q": [], "C": []},
            kernel_backend=resolved_kernel_backend,
        )
        return GroupwiseSpanAdaptivePolicy(
            _outer_log_mass_snapshots=outer_snapshots,
            _inner_policy_snapshots=inner_snapshots,
            snapshot_weights=weights,
            snapshot_rounds=snapshot_rounds,
            reference=reference_policy,
            tau=float(tau),
            final_outer_log_masses=log_outer_mass.copy(),
            outer_second_moments=outer_moment.copy(),
            centered_inner_second_moments=centered_moment.copy(),
            inner_second_moments=raw_moment.copy(),
            group_activation_counts=activation_counts.copy(),
            schedule=schedule,
            diagnostics=diagnostics,
            score_bound=float(score_bound),
            drawn_comparisons=np.zeros((0, 4), dtype=np.float64),
        )

    all_pairs = np.asarray(
        [(i, j) for i in range(alternatives) for j in range(i + 1, alternatives)], dtype=np.int64
    )
    # span(g) <= 1 is a property of the uniform passive and the active
    # protocols; it is asserted only there, never inferred for arbitrary logs.
    span_checked = normalized_protocol == "active" or (
        normalized_protocol == "passive" and propensity_array is None
    ) or (
        propensity_array is not None
        and np.allclose(propensity_array, 1.0 / all_pairs.shape[0], atol=1e-15, rtol=0.0)
    )
    log_reference = np.log(reference_policy)
    tiny = np.finfo(np.float64).tiny
    # Invariant for the whole fit (depends only on reference_policy), so
    # hoisted out of the per-group loop; one_expert._kl_prox_update_with_candidate
    # recomputes the same expression on every call, which gives the same
    # float64 value every time -- deterministic, so precomputing it once is
    # bit-identical, not an approximation.
    maximum_kl = float(np.log(1.0 / np.min(reference_policy)))
    if resolved_kernel_backend == "numba":
        # compiled_kernels.round_start_numba/round_rates_gamma_outer_numba take
        # concrete frozen-rate arrays unconditionally (nopython code cannot
        # branch on "the array doesn't exist"); these placeholders are never
        # read when ``frozen`` is False (see _rate_numba's own branch there).
        frozen_outer_for_kernel = frozen_outer if frozen else np.zeros(n_groups, dtype=np.float64)
        frozen_inner_for_kernel = frozen_inner if frozen else np.zeros(n_groups, dtype=np.float64)
    # Every round is captured (n_snapshots is forced None above), so the
    # snapshot buffer is indexed directly by round_index -- no separate
    # position counter needed (contrast one_expert.py's sparser schedule).
    outer_snapshots_buf = np.empty((horizon, n_groups), dtype=np.float64)
    inner_snapshots_buf = np.empty((horizon, n_groups, alternatives), dtype=np.float64)
    snapshot_position = 0
    drawn = np.full((horizon, 4), np.nan, dtype=np.float64)
    trace_rounds = set(int(r) for r in _trace_rounds(horizon))
    trace: dict[str, list] = {"rounds": [], "eta": [], "theta": [], "Q": [], "C": []}

    identity_residual_max = 0.0
    mass_min = mass_max = 1.0
    max_span = max_abs_reward = max_abs_cancellation = 0.0
    kl_inner_max = kl_deployed_max = 0.0

    for round_index in range(horizon):
        if resolved_kernel_backend == "numba":
            # Compiled fast path for the whole round: every step below is a
            # literal port of the ``else`` branch, except the five NumPy dot/matmul/einsum calls, which stay
            # exactly as written -- executed here, in Python, never inside a
            # kernel -- and ``_draw_comparison``/snapshot copies/trace
            # appends, also unchanged.
            active_groups = np.flatnonzero(group_matrix[round_index]).astype(np.int64)
            eta, theta, effective_log_mass, active_probabilities, mix_ok = (
                _compiled_kernels.round_start_numba(
                    active_groups, log_outer_mass, outer_moment, centered_moment,
                    frozen, frozen_outer_for_kernel, frozen_inner_for_kernel,
                    schedule.complexity, schedule.tau_bar,
                    OUTER_RATE_CAP, INNER_RATE_CAP,
                    OUTER_DENOMINATOR_OFFSET, INNER_DENOMINATOR_OFFSET,
                )
            )
            # The pre-round policy map, before this round's respondent is seen.
            # Assignment into a buffer row already copies; no .copy() needed.
            outer_snapshots_buf[round_index] = effective_log_mass
            inner_snapshots_buf[round_index] = policies
            snapshot_position += 1
            if round_index in trace_rounds:
                trace["rounds"].append(round_index)
                trace["eta"].append(eta.tolist())
                trace["theta"].append(theta.tolist())
                trace["Q"].append(outer_moment.tolist())
                trace["C"].append(centered_moment.tolist())

            if active_groups.size == 0:
                continue
            activation_counts[active_groups] += 1
            if not mix_ok:
                # The gather-softmax hit a non-finite max (see
                # compiled_kernels._softmax_or_flag_numba): redo with the
                # exact, unmodified Python softmax for this rare case.
                active_probabilities = _softmax_from_log(effective_log_mass[active_groups])
            proposals = policies[active_groups]
            policy_raw = active_probabilities @ proposals
            policy, deployed_kl, kl_ok = _compiled_kernels.round_mix_finish_numba(
                policy_raw, reference_policy, tau
            )
            if not kl_ok:
                raise AssertionError("deployed mixture is outside the KL ball")
            kl_deployed_max = max(kl_deployed_max, deployed_kl)

            i, j, sign, propensity = _draw_comparison(
                round_index, policy, utility_matrix, beta, normalized_protocol,
                pairs, label_signs, propensity_array, all_pairs, rng,
            )
            drawn[round_index] = (i, j, sign, propensity)
            score = _policy_weighted_score(policy, i, j, sign, propensity)
            reward, abs_reward, span = _compiled_kernels.round_reward_numba(score, score_bound)
            cancellation = abs(float(np.dot(policy, reward)))
            if cancellation > 2e-9:
                raise AssertionError("deployed lottery does not cancel the reward")
            max_abs_cancellation = max(max_abs_cancellation, cancellation)
            if abs_reward > 1.0 + 1e-9:
                raise AssertionError("normalized reward left [-1, 1]")
            max_abs_reward = max(max_abs_reward, abs_reward)
            if span_checked and span > 1.0 + 1e-9:
                raise AssertionError("reward span exceeded one under a span-bounded protocol")
            max_span = max(max_span, span)

            # Every increment from the same pre-update state.
            r = proposals @ reward
            centered, q_next = _compiled_kernels.round_centered_and_q_numba(
                proposals, reward, r, outer_moment, active_groups
            )
            c = np.einsum("gm,gm->g", proposals, centered * centered)
            v = proposals @ (reward * reward)
            c_next, v_next, residual_ok = _compiled_kernels.round_moments_finish_numba(
                c, v, centered_moment, raw_moment, active_groups, q_next
            )
            if not residual_ok:
                raise AssertionError("Q + C = V failed beyond numerical tolerance")
            identity_residual_max = max(
                identity_residual_max, float(np.max(np.abs(q_next + c_next - v_next)))
            )

            (
                eta_active, theta_active, eta_next, theta_next, gamma_eta, gamma_theta, factors_ok,
            ) = _compiled_kernels.round_rates_gamma_outer_numba(
                active_groups, eta, theta, q_next, c_next, r, frozen, log_outer_mass,
                log_initial_mass, schedule.complexity, schedule.tau_bar,
                OUTER_RATE_CAP, INNER_RATE_CAP, OUTER_DENOMINATOR_OFFSET, INNER_DENOMINATOR_OFFSET,
            )
            if not factors_ok:
                raise AssertionError("Prod update produced a nonpositive weight")

            # Inner: geometric anchor toward the reference, exponential step at
            # the NEXT rate, KL projection onto the common ball. Compiled fast
            # path for every active group at once; groups are independent
            # (nothing below carries state from one ``position`` to the
            # next), so batching them and resolving fallbacks in any order is
            # equivalent to the sequential Python loop it replaces.
            anchors, outputs, kl_values, status = _compiled_kernels.span_adaptive_inner_round_numba(
                policies[active_groups], gamma_theta, theta_next, reward,
                reference_policy, log_reference, tau, maximum_kl,
            )
            for position, group in enumerate(active_groups):
                k = int(group)
                update_counts[k] += 1
                st = status[position]
                if st == _compiled_kernels.STATUS_OK:
                    policies[k] = outputs[position]
                    kl_inner_max = max(kl_inner_max, float(kl_values[position]))
                    # Matches the python path's recomputed projection check
                    # exactly (see below): always false here by construction
                    # of STATUS_OK, kept for a literal parity with the
                    # Python branch rather than assumed.
                    if float(kl_values[position]) > tau + _NUMERICAL_TOLERANCE:
                        projection_counts[k] += 1
                    continue
                if st == _compiled_kernels.STATUS_BINDING:
                    # Compiled fast path for the projection itself (a literal
                    # port of the binding projection in one_expert.py): resolve here and
                    # move on, unless the kernel reports it cannot.
                    anchor = anchors[position]
                    rate = float(theta_next[position])
                    result, binding_status = _compiled_kernels.kl_prox_binding_numba(
                        anchor, rate, -reward, reference_policy, log_reference, tau
                    )
                    if binding_status == _compiled_kernels.BINDING_STATUS_OK:
                        # Every STATUS_BINDING group's unconstrained candidate
                        # already has KL > tau (that is what the status means),
                        # so this matches the Python branch's projection-count
                        # bookkeeping exactly without recomputing that KL.
                        projection_counts[k] += 1
                        policies[k] = result
                        kl_inner_max = max(
                            kl_inner_max, kl_divergence(policies[k], reference_policy)
                        )
                        continue
                    if binding_status == _compiled_kernels.BINDING_STATUS_INFEASIBLE:
                        raise AssertionError(
                            "KL-prox update failed to return a feasible point"
                        )
                    # BINDING_STATUS_NEEDS_PYTHON (maxiter exceeded, an
                    # invalid bracket, or a non-finite softmax mid-solve):
                    # fall through to the unmodified Python solver below with
                    # this same anchor/rate, exactly as it would run standalone.
                else:  # STATUS_NEEDS_PYTHON: even the anchor may be invalid, redo it too
                    g_theta = float(gamma_theta[position])
                    if g_theta == 1.0:
                        anchor = policies[k]
                    else:
                        log_pi = np.log(np.maximum(policies[k], tiny))
                        anchor = _softmax_from_log(g_theta * log_pi + (1.0 - g_theta) * log_reference)
                    rate = float(theta_next[position])
                projected, unconstrained = _kl_prox_update_with_candidate(
                    anchor, -reward, rate, reference_policy, tau
                )
                if unconstrained is None:  # pragma: no cover - tau_bar > 0 here
                    unconstrained = _softmax_from_log(
                        np.log(np.maximum(anchor, tiny)) + rate * reward
                    )
                if kl_divergence(unconstrained, reference_policy) > tau + _NUMERICAL_TOLERANCE:
                    projection_counts[k] += 1
                policies[k] = projected
                kl_inner_max = max(kl_inner_max, kl_divergence(policies[k], reference_policy))

            _compiled_kernels.round_finish_numba(
                active_groups, q_next, c_next, v_next, eta_next, theta_next,
                outer_moment, centered_moment, raw_moment, first_eta_below,
                first_theta_below, eta_min, theta_min, round_index,
                OUTER_RATE_CAP, INNER_RATE_CAP,
            )
            mass = float(np.exp(_compiled_kernels.log_sum_exp_numba(log_outer_mass)))
            mass_min = min(mass_min, mass)
            mass_max = max(mass_max, mass)
        else:
            eta, theta = current_rates()
            effective_log_mass = log_outer_mass + np.log(eta)
            # The pre-round policy map, before this round's respondent is seen.
            # Assignment into a buffer row already copies; no .copy() needed.
            outer_snapshots_buf[round_index] = effective_log_mass
            inner_snapshots_buf[round_index] = policies
            snapshot_position += 1
            if round_index in trace_rounds:
                trace["rounds"].append(round_index)
                trace["eta"].append(eta.tolist())
                trace["theta"].append(theta.tolist())
                trace["Q"].append(outer_moment.tolist())
                trace["C"].append(centered_moment.tolist())

            active_groups = np.flatnonzero(group_matrix[round_index]).astype(np.int64)
            if active_groups.size == 0:
                continue
            activation_counts[active_groups] += 1
            active_probabilities = _softmax_from_log(effective_log_mass[active_groups])
            proposals = policies[active_groups]
            policy = active_probabilities @ proposals
            policy = np.maximum(policy, 0.0)
            policy /= policy.sum()
            deployed_kl = kl_divergence(policy, reference_policy)
            if deployed_kl > tau + 2e-8:
                raise AssertionError("deployed mixture is outside the KL ball")
            kl_deployed_max = max(kl_deployed_max, deployed_kl)

            i, j, sign, propensity = _draw_comparison(
                round_index, policy, utility_matrix, beta, normalized_protocol,
                pairs, label_signs, propensity_array, all_pairs, rng,
            )
            drawn[round_index] = (i, j, sign, propensity)
            score = _policy_weighted_score(policy, i, j, sign, propensity)
            reward = -score / score_bound  # g: winner positive, |g_a| <= 1
            cancellation = abs(float(np.dot(policy, reward)))
            if cancellation > 2e-9:
                raise AssertionError("deployed lottery does not cancel the reward")
            max_abs_cancellation = max(max_abs_cancellation, cancellation)
            abs_reward = float(np.max(np.abs(reward)))
            if abs_reward > 1.0 + 1e-9:
                raise AssertionError("normalized reward left [-1, 1]")
            max_abs_reward = max(max_abs_reward, abs_reward)
            span = float(np.max(reward) - np.min(reward))
            if span_checked and span > 1.0 + 1e-9:
                raise AssertionError("reward span exceeded one under a span-bounded protocol")
            max_span = max(max_span, span)

            # Every increment from the same pre-update state.
            r = proposals @ reward
            centered = reward[None, :] - r[:, None]
            c = np.einsum("gm,gm->g", proposals, centered * centered)
            v = proposals @ (reward * reward)
            q_next = outer_moment[active_groups] + r * r
            c_next = centered_moment[active_groups] + c
            v_next = raw_moment[active_groups] + v
            residual = float(np.max(np.abs(q_next + c_next - v_next)))
            if residual > 1e-9 * max(1.0, float(np.max(v_next))):
                raise AssertionError("Q + C = V failed beyond numerical tolerance")
            identity_residual_max = max(identity_residual_max, residual)

            eta_active = eta[active_groups]
            theta_active = theta[active_groups]
            if frozen:
                eta_next, theta_next = eta_active, theta_active
            else:
                eta_next = schedule.outer_rate(q_next)
                theta_next = schedule.inner_rate(c_next)
            gamma_eta = eta_next / eta_active
            gamma_theta = theta_next / theta_active

            # Outer: tentative Prod step at the current rate, then the anchored
            # correction toward the uniform initialization.  No renormalization.
            factors = eta_active * r
            if np.any(factors <= -1.0):
                raise AssertionError("Prod update produced a nonpositive weight")
            log_plus = log_outer_mass[active_groups] + np.log1p(factors)
            log_outer_mass[active_groups] = (1.0 - gamma_eta) * log_initial_mass + gamma_eta * log_plus

            # Inner: geometric anchor toward the reference, exponential step at the
            # NEXT rate, KL projection onto the common ball.
            for position, group in enumerate(active_groups):
                k = int(group)
                g_theta = float(gamma_theta[position])
                if g_theta == 1.0:
                    anchor = policies[k]
                else:
                    log_pi = np.log(np.maximum(policies[k], tiny))
                    anchor = _softmax_from_log(g_theta * log_pi + (1.0 - g_theta) * log_reference)
                rate = float(theta_next[position])
                # The projection forms the unconstrained candidate
                # ``softmax(log(anchor) - rate * (-reward))`` itself; taking it
                # back is bit-identical to recomputing
                # ``softmax(log(anchor) + rate * reward)`` here (the two differ
                # only by IEEE-exact negations) and halves the softmax work in
                # this loop.  The feasibility predicate, and therefore the
                # projection and update counts, are unchanged.
                projected, unconstrained = _kl_prox_update_with_candidate(
                    anchor, -reward, rate, reference_policy, tau
                )
                if unconstrained is None:  # pragma: no cover - tau_bar > 0 here
                    unconstrained = _softmax_from_log(
                        np.log(np.maximum(anchor, tiny)) + rate * reward
                    )
                if kl_divergence(unconstrained, reference_policy) > tau + _NUMERICAL_TOLERANCE:
                    projection_counts[k] += 1
                update_counts[k] += 1
                policies[k] = projected
                kl_inner_max = max(kl_inner_max, kl_divergence(policies[k], reference_policy))

            outer_moment[active_groups] = q_next
            centered_moment[active_groups] = c_next
            raw_moment[active_groups] = v_next
            below_cap = (eta_next < OUTER_RATE_CAP) & (first_eta_below[active_groups] < 0)
            first_eta_below[active_groups[below_cap]] = round_index
            below_one = (theta_next < INNER_RATE_CAP) & (first_theta_below[active_groups] < 0)
            first_theta_below[active_groups[below_one]] = round_index
            eta_min[active_groups] = np.minimum(eta_min[active_groups], eta_next)
            theta_min[active_groups] = np.minimum(theta_min[active_groups], theta_next)
            mass = float(np.exp(_log_sum_exp(log_outer_mass)))
            mass_min = min(mass_min, mass)
            mass_max = max(mass_max, mass)

    if snapshot_position != snapshot_rounds.size:
        raise AssertionError("failed to capture every snapshot")
    weights = np.full(horizon, 1.0 / horizon, dtype=np.float64)
    eta_terminal, theta_terminal = current_rates()
    diagnostics = SpanAdaptiveDiagnostics(
        algorithm=ALGORITHM_ID, L=schedule.complexity, tau_requested=float(tau), tau_bar=tau_bar,
        R=float(score_bound), T=int(horizon), K=int(n_groups),
        eta_initial=float(eta0[0]), theta_initial=float(theta0[0]),
        eta_terminal=eta_terminal, theta_terminal=theta_terminal, eta_min=eta_min, theta_min=theta_min,
        Q=outer_moment.copy(), C=centered_moment.copy(), V=raw_moment.copy(),
        identity_residual_max=identity_residual_max, activation_counts=activation_counts.copy(),
        outer_mass_terminal=float(np.exp(_log_sum_exp(log_outer_mass))),
        outer_mass_min=mass_min, outer_mass_max=mass_max,
        projection_counts=projection_counts.copy(), update_counts=update_counts.copy(),
        max_span=max_span, max_abs_reward=max_abs_reward, max_abs_cancellation=max_abs_cancellation,
        kl_inner_max=kl_inner_max, kl_deployed_max=kl_deployed_max,
        first_round_eta_below_cap=first_eta_below.copy(),
        first_round_theta_below_one=first_theta_below.copy(),
        frozen=frozen, rate_trace=trace,
        kernel_backend=resolved_kernel_backend,
    )
    return GroupwiseSpanAdaptivePolicy(
        _outer_log_mass_snapshots=outer_snapshots_buf,
        _inner_policy_snapshots=inner_snapshots_buf,
        snapshot_weights=weights,
        snapshot_rounds=snapshot_rounds,
        reference=reference_policy,
        tau=float(tau),
        final_outer_log_masses=log_outer_mass.copy(),
        outer_second_moments=outer_moment.copy(),
        centered_inner_second_moments=centered_moment.copy(),
        inner_second_moments=raw_moment.copy(),
        group_activation_counts=activation_counts.copy(),
        schedule=schedule,
        diagnostics=diagnostics,
        score_bound=float(score_bound),
        drawn_comparisons=drawn,
    )
