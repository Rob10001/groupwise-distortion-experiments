"""Exact held-out metrics for complete preference profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from empirical_testing.numerics import log_sum_exp as _log_sum_exp

#: Tolerances of the KL tilt. All of them are applied to an objective that has
#: been centred and rescaled to unit range, so none of them depends on the units
#: the caller's values arrived in.
_TILT_ZERO_RADIUS = 1e-12
_TILT_TIE_TOLERANCE = 1e-12
_TILT_MIN_SCALE = 1e-14
_TILT_MAX_SCALE = 1e14
_TILT_ROOT_XTOL = 1e-15
_TILT_ROOT_RTOL = 8.9e-16
_TILT_FEASIBILITY_SLACK = 1e-9

#: Scale-aware near-tie
#: criterion, applied on the *raw* scale of `values` before it is centred and
#: rescaled to unit range. `near_tie_threshold(v, n) = c * eps * n * max(|v|)`,
#: the (deliberately coarse) worst-case absolute error of an `n`-term summation
#: that produced `v`. `c = 8` is the design's stated safety factor.
_TILT_NEAR_TIE_C = 8.0

#: Conservative default `n` ("tie_scale") when a caller cannot supply the
#: actual term count. Every in-repo caller can and does supply the real
#: respondent count; this default exists only so `_kl_tilt`/`batched_kl_tilt`
#: never silently fall back to the old, scale-blind `1e-12`-on-normalized
#: criterion for a caller outside this module. It is deliberately larger than
#: any respondent count this project's datasets reach (the bundled Coffee
#: survey is n~3744), so an unsupplied `n` errs toward treating an
#: accumulated-round-off-sized gap as a tie rather than missing it -- safe
#: because Sec 3's loss bound caps the cost of that at `near_tie_threshold`
#: itself, which stays tiny (`~1.8e-9` relative to `max|v|` at this default)
#: relative to any of this project's meaningful `values` differences.
_TILT_DEFAULT_TIE_SCALE = 1.0e6

#: Per-instance tilt status vocabulary.
TILT_STATUS_NONE = "none"
TILT_STATUS_EXACT_FLAT = "exact_flat"
TILT_STATUS_NEAR_FLAT = "near_flat"
TILT_STATUS_PARTIAL_TIE_CAP = "partial_tie_cap"
TILT_STATUS_NEAR_TIE_CAP = "near_tie_cap"

#: Status vocabulary for benchmark/achieved ratios. Every consumer of a
#: distortion number must carry the status with it: a ratio is a number only
#: when the denominator is positive, and 0/0 is not the same event as
#: "positive benchmark, zero achieved welfare".
RATIO_DEFINED = "defined"
RATIO_ZERO_OVER_ZERO = "undefined_zero_over_zero"
RATIO_POSITIVE_OVER_ZERO = "infinite_zero_denominator"
RATIO_EMPTY = "undefined_empty_group"
RATIO_NEGATIVE_DENOMINATOR = "undefined_negative_denominator"
RATIO_NONFINITE = "undefined_nonfinite_input"


def _centered_sigmoid(value: np.ndarray | float) -> np.ndarray:
    """Evaluate ``2*sigmoid(value) - 1`` stably, including near zero."""

    value_array = np.asarray(value, dtype=np.float64)
    magnitude = np.abs(value_array)
    return np.sign(value_array) * (
        -np.expm1(-magnitude) / (1.0 + np.exp(-magnitude))
    )


def margin_matrix(utilities: np.ndarray, beta: float = np.inf) -> np.ndarray:
    """Per-respondent oracle margin ``M[n,a,b] = E[2*1{a > b} - 1]``.

    Under the manuscript's Bradley-Terry assumption the comparison oracle has
    mean ``2*sigmoid(beta*d) - 1`` for ``d = u_a - u_b``. The
    default ``beta = +inf`` is the deterministic score-order oracle, whose mean
    margin is ``sign(d)``; the two agree in the limit, including at exact ties
    where both give zero. ``inf * 0`` is nan, so the infinite case is branched
    rather than evaluated.
    """

    difference = utilities[:, :, None] - utilities[:, None, :]
    if np.isposinf(beta):
        return np.sign(difference)
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be positive or +infinity")
    return _centered_sigmoid(float(beta) * difference)


def deterministic_h2h(
    policy_a: np.ndarray,
    policy_b: np.ndarray,
    utilities: np.ndarray,
    beta: float = np.inf,
) -> np.ndarray:
    """Exact probability that A beats B for each respondent; ties count 1/2."""

    payoff = 0.5 + 0.5 * margin_matrix(utilities, beta)
    return np.einsum("ni,nij,nj->n", policy_a, payoff, policy_b, optimize=True)


def welfare(policy: np.ndarray, utilities: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ni->n", policy, utilities, optimize=True)


def constrained_optimum(
    utilities: np.ndarray, reference: np.ndarray, tau: float
) -> float:
    """``OPT_k = max over q in B of W_k(q)``, the manuscript's benchmark.

    Welfare is linear in the lottery, so this is a linear objective over the KL
    ball -- exactly what `_kl_tilt` solves. The benchmark lives *inside* the
    trust region, so distortion measures the cost of learning from comparisons
    and does not fold in the welfare cost of the region itself.
    """

    utilities = np.asarray(utilities, dtype=np.float64)
    mean_utility = utilities.mean(axis=0)
    # D-510: `n` is this group's own respondent count. `constrained_optimum`
    # returns only the objective value, which is identical for every point of
    # the maximizing set (tie-invariant) -- passing
    # `tie_scale` here matters for `status_out`-style diagnostics downstream,
    # not for this return value.
    best = _kl_tilt(
        np.asarray(reference, dtype=np.float64),
        mean_utility,
        float(tau),
        tie_scale=float(utilities.shape[0]) if utilities.shape[0] > 0 else None,
    )
    return float(best @ mean_utility)


def ratio_with_status(
    numerator: float, denominator: float, count: int | None = None
) -> tuple[float, str]:
    """The single place a benchmark/achieved ratio is formed.

    ``count`` is the number of evaluation rows behind the ratio; zero means the
    group was empty on this sample, which is not the same event as a zero
    denominator. No epsilon is ever added to the denominator: that would change
    the estimand silently. The returned value is ``nan`` for every undefined
    status and ``+inf`` only for a genuinely positive numerator over a zero
    denominator.
    """

    if count is not None and int(count) <= 0:
        return float("nan"), RATIO_EMPTY
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan"), RATIO_NONFINITE
    if denominator < 0.0:
        return float("nan"), RATIO_NEGATIVE_DENOMINATOR
    if denominator == 0.0:
        if numerator == 0.0:
            return float("nan"), RATIO_ZERO_OVER_ZERO
        if numerator > 0.0:
            return float("inf"), RATIO_POSITIVE_OVER_ZERO
        return float("nan"), RATIO_NEGATIVE_DENOMINATOR
    return numerator / denominator, RATIO_DEFINED


def link_constants(beta: float) -> dict[str, float]:
    """``s_beta = 2*sigmoid(beta)-1``, ``c_beta = beta/2``, and factor c/s.

    The limiting worst-case constrained distortion factor is ``c_beta/s_beta``.
    It tends to 1 as beta -> 0 and grows like beta/2 as beta -> infinity, where
    comparisons retain little information about the magnitude of utility
    differences -- which is why the deterministic sign oracle admits no finite
    factor.
    """

    if np.isposinf(beta):
        return {"s_beta": 1.0, "c_beta": float("inf"), "factor": float("inf")}
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be positive or +infinity")
    s = float(_centered_sigmoid(beta))
    c = 0.5 * float(beta)
    return {"s_beta": s, "c_beta": c, "factor": c / s}


def theory_certificate_error(
    horizon: int,
    group_masses: np.ndarray,
    score_bound: float,
    complexity: float,
) -> np.ndarray:
    """``eps_{T,k} = 6 sqrt(RH/(T p_k)) + 6 RH/(T p_k)`` (``eq:main-error``).

    This is the *a priori* error of Theorems 1 and 2, computed from quantities
    the algorithm is given before it sees any data. It bounds the signed
    exploitability of the returned policy with probability ``1 - delta``,
    simultaneously across groups. It is vacuous for a group once it exceeds
    one, since margins lie in ``[-1, 1]``.
    """

    masses = np.asarray(group_masses, dtype=np.float64)
    if np.any(masses <= 0.0):
        # Every in-scope caller reaches this only after `one_expert.fixed_rates`
        # has already rejected a non-positive mass (that check stays in place);
        # this guard covers a direct call that skips it, so `p_k = 0` raises
        # here instead of dividing silently into inf/nan (D-304).
        raise ValueError("every group mass must be positive")
    ratio = float(score_bound) * float(complexity) / (float(horizon) * masses)
    return 6.0 * np.sqrt(ratio) + 6.0 * ratio


def distortion_bound(epsilon: float, optimum: float, beta: float) -> float:
    """``(s_beta/c_beta - epsilon/(c_beta*OPT))^-1``, from ``prop:bridge``.

    Vacuous (``+inf``) when ``epsilon >= s_beta * OPT``, which is exactly the
    regime the corollary excludes.

    ``epsilon`` must bound the advantage of the welfare-optimal challenger
    ``q_k^*``. A cross-fitted selected-challenger estimate can be smaller than
    that advantage and therefore cannot certify this bound.
    """

    constants = link_constants(beta)
    if np.isposinf(constants["c_beta"]):
        return float("inf")
    denominator = (
        constants["s_beta"] / constants["c_beta"]
        - epsilon / (constants["c_beta"] * optimum)
    )
    return float("inf") if denominator <= 0.0 else float(1.0 / denominator)


class KLTiltNonconvergenceError(RuntimeError):
    """Raised when the scalar KL tilt's bracket search exceeds ``_TILT_MAX_SCALE``.

    A blown-up bracket is not evidence that the reference itself is optimal --
    it means the residual never crossed zero within the scale range the solver
    trusts. Returning ``reference.copy()`` in that case (the previous behavior)
    was indistinguishable from a genuine zero-radius result, so scalar callers
    must see a hard failure instead (D-304). ``batched_kl_tilt`` already reports
    this per-instance via ``converged=False`` rather than raising; that remains
    the model for the batched path.
    """


def _near_tie_threshold(values: np.ndarray, tie_scale: float) -> float:
    """``c * eps * n * max(|v|)`` (D-510 design Sec 3), on the raw scale of ``v``."""

    magnitude = float(np.max(np.abs(values)))
    return _TILT_NEAR_TIE_C * np.finfo(np.float64).eps * float(tie_scale) * magnitude


def _near_tie_threshold_batch(values: np.ndarray, tie_scale: np.ndarray) -> np.ndarray:
    """Row-wise :func:`_near_tie_threshold`."""

    magnitude = np.max(np.abs(values), axis=1)
    return _TILT_NEAR_TIE_C * np.finfo(np.float64).eps * tie_scale * magnitude


def _kl_tilt(
    reference: np.ndarray,
    values: np.ndarray,
    tau: float,
    *,
    tie_scale: float | None = None,
    status_out: list[str] | None = None,
) -> np.ndarray:
    """Maximize ``values @ q`` subject to ``KL(q || reference) <= tau``.

    The maximizer is invariant under positive affine maps ``values -> a*values +
    b`` with ``a > 0``: adding ``b`` multiplies every candidate's unnormalized
    weight by the same factor, and scaling by ``a`` is absorbed by the tilt
    parameter. The objective is therefore centred and rescaled to unit range
    *before* the tilt is solved, so every tolerance below is applied on a
    scale-free quantity, with one deliberate exception: the near-tie criterion
    below, which is evaluated on the *raw* scale of ``values`` before that
    normalization, per D-510.

    The previous implementation short-circuited to the reference whenever
    ``np.allclose(values, values[0])`` held at NumPy's default tolerances. That
    treated a small but strictly nonconstant objective as a tie: with
    ``reference = [0.5, 0.5]``, ``values = [0.5, 0.500001]`` and ``tau = 0.1`` it
    returned the reference even though ``[0.3, 0.7]`` is feasible
    (``KL ~ 0.08228``) and strictly better. Only an exactly constant objective,
    or one within D-510's scale-aware near-tie threshold of constant, is
    treated as a tie now.

    ``tie_scale`` is ``n``, the number of per-respondent terms summed (or
    averaged) to produce ``values``, used by the scale-aware near-tie threshold.
    Every in-repo caller that
    selects a challenger passes its own respondent count explicitly; ``None``
    falls back to :data:`_TILT_DEFAULT_TIE_SCALE`, documented there.

    ``status_out``, if given, is a list this call appends exactly one status
    string to (:data:`TILT_STATUS_NONE` / ``EXACT_FLAT`` / ``NEAR_FLAT`` /
    ``PARTIAL_TIE_CAP`` / ``NEAR_TIE_CAP``, D-510 Sec 4) -- a diagnostic
    out-parameter that leaves this function's return type unchanged for every
    existing caller.
    """

    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if reference.shape != values.shape:
        raise ValueError("reference and values must have the same shape")
    if reference.ndim != 1 or reference.size < 2:
        raise ValueError("reference must be a one-dimensional lottery of length at least 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contains a non-finite entry")
    if np.any(reference <= 0.0):
        raise ValueError("reference must have full support")
    # Same check and tolerance as `nlhf._validate_reference`
    # (src/empirical_testing/algorithms/nlhf.py:130): atol=1e-12, rtol=0.0.
    if not np.isclose(reference.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("reference must sum to one")

    def _status(value: str) -> None:
        if status_out is not None:
            status_out.append(value)

    tau = float(tau)
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and nonnegative")
    if tau <= _TILT_ZERO_RADIUS:
        _status(TILT_STATUS_NONE)
        return reference.copy()

    scale = _TILT_DEFAULT_TIE_SCALE if tie_scale is None else float(tie_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("tie_scale must be finite and positive")

    spread = float(values.max() - values.min())
    if not np.isfinite(spread):
        # `values` is finite (checked above) but its range overflowed float64,
        # e.g. values=[-1e308, 1e308]. Dividing by an infinite spread below
        # would silently degenerate rather than tilt; refuse instead (D-304).
        raise ValueError("values has finite entries but their spread (max - min) overflowed to a non-finite value")
    if spread <= 0.0:
        # An exactly constant objective: every feasible lottery is optimal and
        # the reference is the canonical representative.
        _status(TILT_STATUS_EXACT_FLAT)
        return reference.copy()

    threshold = _near_tie_threshold(values, scale)
    if spread <= threshold:
        # D-510: a nonconstant objective whose spread is no bigger than the
        # accumulated round-off scale that could have produced `values` -- the
        # discontinuity of Sec 2 (dividing by a near-zero `spread` to build
        # `normalized`). Treated as a tie; the loss from doing so is bounded by
        # `spread <= threshold` (Sec 3's loss bound), verified in
        # `tests/test_challenger_ties.py`.
        _status(TILT_STATUS_NEAR_FLAT)
        return reference.copy()
    normalized = (values - float(values.mean())) / spread

    # D-510 Sec 3: near-max-set membership on the *raw* scale (`values`, with
    # the same absolute `threshold`), not on `normalized` with a fixed
    # `_TILT_TIE_TOLERANCE` -- that divided by `spread`, which is exactly the
    # quantity Sec 2 shows can be driven arbitrarily close to zero by an
    # unrelated last-bit perturbation.
    best_mask = values >= (values.max() - threshold)
    best_mass = float(reference[best_mask].sum())
    attainable_kl = float(np.log(1.0 / best_mass))

    def argmax_limit() -> np.ndarray:
        result = np.zeros_like(reference)
        result[best_mask] = reference[best_mask] / best_mass
        return result

    if tau >= attainable_kl - _TILT_ZERO_RADIUS:
        if int(np.count_nonzero(best_mask)) == 1:
            # A cap hit with a singleton maximizing set is unambiguous: the
            # smooth branch converges to the same pure action as scale ->
            # `_TILT_MIN_SCALE` (Sec 2), so there is no tie to report.
            _status(TILT_STATUS_NONE)
        elif abs(tau - attainable_kl) <= _TILT_ZERO_RADIUS:
            # The cap-branch-vs-interior-branch choice itself is noise
            # sensitive here: `tau` and `attainable_kl` agree to the same
            # `_TILT_ZERO_RADIUS` this comparison uses.
            _status(TILT_STATUS_NEAR_TIE_CAP)
        else:
            _status(TILT_STATUS_PARTIAL_TIE_CAP)
        return argmax_limit()

    log_reference = np.log(reference)

    def candidate(scale: float) -> np.ndarray:
        logits = log_reference + normalized / scale
        return np.exp(logits - _log_sum_exp(logits))

    def divergence(q: np.ndarray) -> float:
        positive = q > 0.0
        return float(
            np.sum(q[positive] * (np.log(q[positive]) - log_reference[positive]))
        )

    def residual(scale: float) -> float:
        return divergence(candidate(scale)) - tau

    # KL(candidate(scale) || reference) decreases in scale. `normalized` has unit
    # range, so a bracket that starts at one and moves geometrically encloses the
    # root whatever units `values` arrived in -- the fixed 1e-10 lower end of the
    # previous version was not small enough for near-tied maxima.
    upper = 1.0
    while residual(upper) > 0.0:
        upper *= 2.0
        if upper > _TILT_MAX_SCALE:
            raise KLTiltNonconvergenceError(
                f"KL tilt bracket exceeded scale {_TILT_MAX_SCALE:.1e} without "
                "finding a feasible upper end; refusing to silently return the "
                "reference (that outcome is indistinguishable from a genuine "
                "zero-radius result)"
            )
    lower = 1.0
    while residual(lower) < 0.0:
        lower *= 0.5
        if lower < _TILT_MIN_SCALE:
            # Numerical fallback, not a v-based tie decision: the interior
            # search underflowed before finding a feasible lower bound, so the
            # solution is taken to be at the near-max set found above.
            if int(np.count_nonzero(best_mask)) == 1:
                _status(TILT_STATUS_NONE)
            else:
                _status(TILT_STATUS_PARTIAL_TIE_CAP)
            return argmax_limit()
    root = float(brentq(residual, lower, upper, xtol=_TILT_ROOT_XTOL, rtol=_TILT_ROOT_RTOL))
    tilted = candidate(root)
    # Round the root toward the feasible side rather than returning a point a few
    # ulps outside the ball.
    guard = 0
    while divergence(tilted) > tau and guard < 64:
        root = float(np.nextafter(root, np.inf))
        tilted = candidate(root)
        guard += 1
    if divergence(tilted) > tau + _TILT_FEASIBILITY_SLACK:
        # D-515: for some near-degenerate `values` (e.g. two top components
        # tied to within ~1e-9 relative -- the D-508 tie fold T1), the true
        # KL(scale) == tau crossing falls strictly between two adjacent
        # *representable* `candidate(scale)` softmax outputs. At the scale
        # magnitude the root reaches, `normalized / scale` is a large ratio
        # (~1e7 on T1) whose own float64 resolution is coarser than the
        # logit gap needed to cross into feasibility, so `brentq`'s returned
        # root can land on the infeasible side, and stepping `scale` by a
        # single `nextafter` moves the *candidate* by an amount that does
        # not monotonically approach feasibility: consecutive guard steps
        # above can oscillate between two representable KL values, neither
        # feasible, rather than converging.
        #
        # `upper` is unconditionally feasible here by the bracket-search
        # invariant above (that while-loop only exits once
        # `residual(upper) <= 0`, i.e. `divergence(candidate(upper)) <=
        # tau`), and `root` is unconditionally infeasible here (this branch
        # only runs when `divergence(tilted) > tau`). Bisect *scale* between
        # them directly on `divergence`/`candidate` -- the same primitives
        # the rest of this function uses, so this changes no other branch --
        # instead of taking single-ulp steps that this magnitude regime
        # cannot resolve finely enough.
        lo, hi = root, upper
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if divergence(candidate(mid)) > tau:
                lo = mid
            else:
                hi = mid
        root = hi
        tilted = candidate(root)
    if divergence(tilted) > tau + _TILT_FEASIBILITY_SLACK:
        raise AssertionError(
            f"KL tilt returned an infeasible lottery: KL={divergence(tilted):.6e} > tau={tau:.6e}"
        )
    _status(TILT_STATUS_NONE)
    return tilted


@dataclass(frozen=True)
class BatchedKLTiltResult:
    """Per-instance output of :func:`batched_kl_tilt`.

    ``kl_residual`` is ``KL(q || reference) - tau``.  Nonconvergence (bracket
    blow-up past ``_TILT_MAX_SCALE``, or the safeguarded solver hitting its
    iteration cap) is reported with ``converged=False`` and a NaN lottery --
    never silently replaced by the reference (contract §10 / D-304).

    ``status`` is D-510's per-instance tie classification (an object array of
    :data:`TILT_STATUS_NONE` / ``EXACT_FLAT`` / ``NEAR_FLAT`` /
    ``PARTIAL_TIE_CAP`` / ``NEAR_TIE_CAP``), identical per instance to what the
    scalar :func:`_kl_tilt` would report via its ``status_out`` for the same
    row. A row that did not converge carries :data:`TILT_STATUS_NONE` as a
    placeholder -- ``converged`` is the authoritative failure signal for that
    row, not ``status``.
    """

    solution: np.ndarray
    kl_residual: np.ndarray
    feasible: np.ndarray
    converged: np.ndarray
    status: np.ndarray


def _tilt_kl(q: np.ndarray, log_reference: np.ndarray) -> np.ndarray:
    """Vectorized form of the scalar tilt's ``divergence``."""

    safe = np.where(q > 0.0, q, 1.0)
    terms = np.where(q > 0.0, q * (np.log(safe) - log_reference), 0.0)
    return terms.sum(axis=1)


def _batched_log_sum_exp(values: np.ndarray) -> np.ndarray:
    """Row-wise :func:`log_sum_exp`, SciPy operation order (contract §5.2)."""

    a_max = values.max(axis=1, keepdims=True)
    at_maximum = values == a_max
    multiplicity = at_maximum.sum(axis=1, keepdims=True).astype(np.float64)
    shifted = np.exp(np.where(at_maximum, -np.inf, values) - a_max)
    total = shifted.sum(axis=1, keepdims=True)
    nonzero = total[:, 0] != 0.0
    total[nonzero] = total[nonzero] / multiplicity[nonzero]
    return np.log1p(total) + np.log(multiplicity) + a_max


def _tilt_family(
    log_reference: np.ndarray, normalized: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``q(s) = softmax(log reference + normalized / s)`` and its KL to the reference."""

    logits = log_reference + normalized / scale[:, None]
    lse = _batched_log_sum_exp(logits)
    q = np.exp(logits - lse)
    return q, _tilt_kl(q, log_reference)


def _argmax_limit_batch(reference: np.ndarray, best_mask: np.ndarray) -> np.ndarray:
    """Near-max set ``best_mask``, split in proportion to the reference."""

    mass = np.where(best_mask, reference, 0.0)
    return mass / mass.sum(axis=1, keepdims=True)


def _solve_tilt_batch(
    normalized: np.ndarray,
    log_reference: np.ndarray,
    reference: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Safeguarded Newton-with-bisection on the scalar residual, all rows together.

    Rows are already past the zero-radius, exact-tie, and attainable-cap
    branches.  Returns ``(q, kl, converged, status)``.
    """

    n_rows, n_actions = normalized.shape
    q_out = np.full((n_rows, n_actions), np.nan, dtype=np.float64)
    kl_out = np.full(n_rows, np.nan, dtype=np.float64)
    converged = np.ones(n_rows, dtype=bool)
    status = np.full(n_rows, TILT_STATUS_NONE, dtype=object)
    if n_rows == 0:
        return q_out, kl_out, converged, status

    lower = np.ones(n_rows, dtype=np.float64)
    upper = np.ones(n_rows, dtype=np.float64)

    _q, kl = _tilt_family(log_reference, normalized, upper)
    need = kl > tau
    while np.any(need):
        upper[need] *= 2.0
        blown = need & (upper > _TILT_MAX_SCALE)
        if np.any(blown):
            converged[blown] = False
            need[blown] = False
        if not np.any(need):
            break
        _q, kl = _tilt_family(log_reference, normalized, upper)
        need &= kl > tau

    alive = converged.copy()
    _q, kl = _tilt_family(log_reference, normalized, lower)
    need = alive & (kl < tau)
    use_cap = np.zeros(n_rows, dtype=bool)
    while np.any(need):
        lower[need] *= 0.5
        floored = need & (lower < _TILT_MIN_SCALE)
        if np.any(floored):
            use_cap[floored] = True
            alive[floored] = False
            need[floored] = False
        if not np.any(need):
            break
        _q, kl = _tilt_family(log_reference, normalized, lower)
        need &= kl < tau

    if np.any(use_cap):
        # Numerical fallback, not a v-based tie decision (D-510 Sec 4 doesn't
        # cover this branch specifically): the interior search underflowed
        # `_TILT_MIN_SCALE` before finding a feasible lower bound, so the row
        # is taken to be at its own near-max set (same fixed-tolerance mask
        # the pre-D-510 cap branch used, since this is a solver safety net,
        # not the primary tie test). Classified the same way as a genuine cap
        # hit: a singleton set is unambiguous, a multi-member set is a tie.
        floor_mask = normalized >= (normalized.max(axis=1, keepdims=True) - _TILT_TIE_TOLERANCE)
        cap_q = _argmax_limit_batch(reference, floor_mask)
        q_out[use_cap] = cap_q[use_cap]
        kl_out[use_cap] = _tilt_kl(q_out[use_cap], log_reference)
        singleton = use_cap & (floor_mask.sum(axis=1) == 1)
        multi = use_cap & ~singleton
        status[singleton] = TILT_STATUS_NONE
        status[multi] = TILT_STATUS_PARTIAL_TIE_CAP

    pending = alive
    scale = np.empty(n_rows, dtype=np.float64)
    if np.any(pending):
        scale[pending], still = _vectorized_brentq(
            normalized, log_reference, tau, lower, upper, pending
        )
        if np.any(still):
            converged[still] = False

    solved = converged & ~use_cap
    if np.any(solved):
        tilted, tilt_kl = _tilt_family(log_reference, normalized, scale)
        guard_scale = scale.copy()
        for _ in range(64):
            bad = solved & (tilt_kl > tau)
            if not np.any(bad):
                break
            guard_scale[bad] = np.nextafter(guard_scale[bad], np.inf)
            q_step, kl_step = _tilt_family(log_reference, normalized, guard_scale)
            tilted[bad] = q_step[bad]
            tilt_kl[bad] = kl_step[bad]
        # D-515: per-instance equivalent of the scalar `_kl_tilt`'s bisection
        # fallback (see its comment) -- a near-degenerate row can put the
        # true KL(scale) == tau crossing strictly between two adjacent
        # representable `_tilt_family` outputs, so the ulp-sized guard steps
        # above oscillate rather than converge. Any row still outside the
        # feasibility slack is bisected between its own (infeasible)
        # `guard_scale` and its own `upper`, which is feasible by
        # construction (the bracket-search loop that built `upper` only
        # exits a row once its KL is `<= tau`, and `solved` excludes any row
        # that blew that loop's bracket without reaching feasibility).
        bad = solved & (tilt_kl > tau + _TILT_FEASIBILITY_SLACK)
        if np.any(bad):
            lo = guard_scale.copy()
            hi = upper.copy()
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                progress = bad & (mid > lo) & (mid < hi)
                if not np.any(progress):
                    break
                _q_mid, kl_mid = _tilt_family(log_reference, normalized, mid)
                infeasible_mid = progress & (kl_mid > tau)
                feasible_mid = progress & ~infeasible_mid
                lo[infeasible_mid] = mid[infeasible_mid]
                hi[feasible_mid] = mid[feasible_mid]
            q_hi, kl_hi = _tilt_family(log_reference, normalized, hi)
            tilted[bad] = q_hi[bad]
            tilt_kl[bad] = kl_hi[bad]
        q_out[solved] = tilted[solved]
        kl_out[solved] = tilt_kl[solved]
        # `status` for `solved` rows is already TILT_STATUS_NONE (the default).

    return q_out, kl_out, converged, status


def _vectorized_brentq(
    normalized: np.ndarray,
    log_reference: np.ndarray,
    tau: float,
    lower: np.ndarray,
    upper: np.ndarray,
    pending: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """SciPy ``brentq`` (Zeros/brentq.c), applied to every pending row together.

    The residual is the scalar tilt's ``KL(q(s) || reference) - tau``, so the
    evaluation sequence -- and therefore the root on a numerically flat
    residual -- matches the one-at-a-time solver.
    """

    n_rows = pending.shape[0]
    xpre = lower.copy()
    xcur = upper.copy()
    _q, kl_pre = _tilt_family(log_reference, normalized, xpre)
    _q, kl_cur = _tilt_family(log_reference, normalized, xcur)
    fpre = kl_pre - tau
    fcur = kl_cur - tau

    root = xcur.copy()
    at_pre = pending & (fpre == 0.0)
    at_cur = pending & (fcur == 0.0) & ~at_pre
    root[at_pre] = xpre[at_pre]
    root[at_cur] = xcur[at_cur]

    working = pending & ~at_pre & ~at_cur
    same_sign = working & (fpre != 0.0) & (fcur != 0.0) & (
        np.signbit(fpre) == np.signbit(fcur)
    )
    working &= ~same_sign

    xblk = xpre.copy()
    fblk = fpre.copy()
    spre = xcur - xpre
    scur = spre.copy()

    for _ in range(100):
        if not np.any(working):
            break

        opposite = (
            working
            & (fpre != 0.0)
            & (fcur != 0.0)
            & (np.signbit(fpre) != np.signbit(fcur))
        )
        if np.any(opposite):
            xblk[opposite] = xpre[opposite]
            fblk[opposite] = fpre[opposite]
            step = xcur[opposite] - xpre[opposite]
            spre[opposite] = step
            scur[opposite] = step

        swap = working & (np.abs(fblk) < np.abs(fcur))
        if np.any(swap):
            old_xcur = xcur[swap].copy()
            old_fcur = fcur[swap].copy()
            xpre[swap] = xcur[swap]
            xcur[swap] = xblk[swap]
            xblk[swap] = old_xcur
            fpre[swap] = fcur[swap]
            fcur[swap] = fblk[swap]
            fblk[swap] = old_fcur

        delta = 0.5 * (_TILT_ROOT_XTOL + _TILT_ROOT_RTOL * np.abs(xcur))
        sbis = 0.5 * (xblk - xcur)
        done = working & ((fcur == 0.0) | (np.abs(sbis) < delta))
        if np.any(done):
            root[done] = xcur[done]
            working[done] = False
        if not np.any(working):
            break

        stry = np.zeros(n_rows, dtype=np.float64)
        can_interp = (
            working
            & (np.abs(spre) > delta)
            & (np.abs(fcur) < np.abs(fpre))
        )
        linear = can_interp & (xpre == xblk)
        inverse = can_interp & ~linear
        if np.any(linear):
            denom = fcur[linear] - fpre[linear]
            stry[linear] = -fcur[linear] * (xcur[linear] - xpre[linear]) / denom
        if np.any(inverse):
            dpre = (fpre[inverse] - fcur[inverse]) / (xpre[inverse] - xcur[inverse])
            dblk = (fblk[inverse] - fcur[inverse]) / (xblk[inverse] - xcur[inverse])
            stry[inverse] = (
                -fcur[inverse]
                * (fblk[inverse] * dblk - fpre[inverse] * dpre)
                / (dblk * dpre * (fblk[inverse] - fpre[inverse]))
            )

        good = (
            can_interp
            & np.isfinite(stry)
            & (2.0 * np.abs(stry) < np.minimum(np.abs(spre), 3.0 * np.abs(sbis) - delta))
        )
        if np.any(good):
            spre[good] = scur[good]
            scur[good] = stry[good]
        bisect = working & ~good
        if np.any(bisect):
            spre[bisect] = sbis[bisect]
            scur[bisect] = sbis[bisect]

        xpre[working] = xcur[working]
        fpre[working] = fcur[working]
        use_scur = working & (np.abs(scur) > delta)
        xcur[use_scur] = xcur[use_scur] + scur[use_scur]
        tiny = working & ~use_scur
        if np.any(tiny):
            xcur[tiny] = xcur[tiny] + np.where(sbis[tiny] > 0.0, delta[tiny], -delta[tiny])

        _q, kl_new = _tilt_family(log_reference, normalized, xcur)
        fcur[working] = (kl_new - tau)[working]

    still = working | same_sign
    root[working] = xcur[working]
    return root, still


def batched_kl_tilt(
    values_batch: np.ndarray,
    reference: np.ndarray,
    tau: float,
    *,
    tie_scale: float | np.ndarray | None = None,
) -> BatchedKLTiltResult:
    """Maximize ``values @ q`` subject to ``KL(q || reference) <= tau``, many instances.

    This is the evaluator tilt of contract §10 -- a different problem from the
    learner's prox in ``one_expert.py``.  ``values_batch`` has shape ``(N, m)``;
    ``reference`` is a single length-``m`` lottery and ``tau`` is shared.  Branch
    conditions match :func:`_kl_tilt` exactly, per instance: zero-radius cut at
    ``1e-12``, an exactly constant objective returns the reference, D-510's
    scale-aware near-tie criterion (Sec 3), and the cap compares
    ``log(1 / best_mass)`` of the near-max set -- now on the *raw* scale of
    ``values`` with that same near-tie threshold, not the old fixed
    ``1e-12``-on-normalized test -- and splits that set in proportion to the
    reference.

    ``tie_scale`` is ``n`` per instance: a scalar (shared by every row) or an
    ``(N,)`` array (one respondent count per row, e.g. a different role size
    per (group, direction) job in `uncertainty.per_seed_certificates`).
    ``None`` falls back to :data:`_TILT_DEFAULT_TIE_SCALE` for every row, as
    :func:`_kl_tilt` does. See :func:`_kl_tilt` for the criterion and the
    status vocabulary; ``result.status`` here is that same classification, one
    entry per instance.
    """

    values = np.asarray(values_batch, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values_batch must have shape (N, m)")
    n_instances, n_actions = values.shape
    if reference.ndim != 1 or reference.size != n_actions:
        raise ValueError("reference must be a one-dimensional lottery of length m")
    if n_actions < 2:
        raise ValueError("reference must be a one-dimensional lottery of length at least 2")
    if n_instances > 0 and not np.all(np.isfinite(values)):
        raise ValueError("values contains a non-finite entry")
    if np.any(reference <= 0.0):
        raise ValueError("reference must have full support")
    # Same check and tolerance as `nlhf._validate_reference`
    # (src/empirical_testing/algorithms/nlhf.py:130): atol=1e-12, rtol=0.0.
    if not np.isclose(reference.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("reference must sum to one")

    tau = float(tau)
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and nonnegative")

    if tie_scale is None:
        scale = np.full(n_instances, _TILT_DEFAULT_TIE_SCALE, dtype=np.float64)
    else:
        scale = np.broadcast_to(
            np.asarray(tie_scale, dtype=np.float64), (n_instances,)
        ).copy()
    if n_instances > 0 and (not np.all(np.isfinite(scale)) or np.any(scale <= 0.0)):
        raise ValueError("tie_scale must be finite and positive")

    solution = np.empty((n_instances, n_actions), dtype=np.float64)
    kl_residual = np.empty(n_instances, dtype=np.float64)
    converged = np.ones(n_instances, dtype=bool)
    status = np.full(n_instances, TILT_STATUS_NONE, dtype=object)
    if n_instances == 0:
        return BatchedKLTiltResult(
            solution=solution,
            kl_residual=kl_residual,
            feasible=np.ones(0, dtype=bool),
            converged=converged,
            status=status,
        )

    if tau <= _TILT_ZERO_RADIUS:
        solution[:] = reference
        kl_residual[:] = -tau
        return BatchedKLTiltResult(
            solution=solution,
            kl_residual=kl_residual,
            feasible=np.ones(n_instances, dtype=bool),
            converged=converged,
            status=status,
        )

    log_reference = np.log(reference)
    spread = values.max(axis=1) - values.min(axis=1)
    exact_tie = spread <= 0.0
    bad_spread = ~np.isfinite(spread)
    solution[exact_tie] = reference
    kl_residual[exact_tie] = -tau
    status[exact_tie] = TILT_STATUS_EXACT_FLAT
    if np.any(bad_spread):
        solution[bad_spread] = np.nan
        kl_residual[bad_spread] = np.nan
        converged[bad_spread] = False
        # status stays TILT_STATUS_NONE: a refused/nonconvergent row, not a
        # tie classification -- `converged` is authoritative for it.

    pending = ~exact_tie & ~bad_spread
    if np.any(pending):
        # D-510 Sec 3: one threshold per row, on the raw scale, reused for
        # both the near-flat test below and the near-max-set membership test
        # further down -- exactly mirroring the scalar `_kl_tilt`.
        threshold = _near_tie_threshold_batch(values, scale)

        near_flat = pending & (spread <= threshold)
        if np.any(near_flat):
            solution[near_flat] = reference
            kl_residual[near_flat] = -tau
            status[near_flat] = TILT_STATUS_NEAR_FLAT
        pending = pending & ~near_flat

    if np.any(pending):
        pending_idx = np.nonzero(pending)[0]
        pending_values = values[pending]
        pending_spread = spread[pending]
        pending_threshold = threshold[pending]
        normalized = np.zeros_like(values)
        normalized[pending] = (
            pending_values - pending_values.mean(axis=1, keepdims=True)
        ) / pending_spread[:, None]

        # D-510 Sec 3: near-max-set membership on the raw scale, with the same
        # per-row threshold used for the near-flat test above -- not the old
        # fixed `_TILT_TIE_TOLERANCE` on `normalized`. Computed only over the
        # pending subset (rather than a full-width mask defaulting to False
        # elsewhere) so a non-pending row never divides by a zero `best_mass`.
        best_mask_local = pending_values >= (
            pending_values.max(axis=1, keepdims=True) - pending_threshold[:, None]
        )
        best_mass_local = (reference[None, :] * best_mask_local).sum(axis=1)
        attainable_local = np.log(1.0 / best_mass_local)
        singleton_local = best_mask_local.sum(axis=1) == 1
        near_cap_boundary_local = np.abs(tau - attainable_local) <= _TILT_ZERO_RADIUS
        cap_local = tau >= attainable_local - _TILT_ZERO_RADIUS

        if np.any(cap_local):
            cap_rows = pending_idx[cap_local]
            cap_q = _argmax_limit_batch(reference, best_mask_local[cap_local])
            solution[cap_rows] = cap_q
            kl_residual[cap_rows] = _tilt_kl(cap_q, log_reference) - tau
            status[pending_idx[cap_local & singleton_local]] = TILT_STATUS_NONE
            status[pending_idx[cap_local & ~singleton_local & near_cap_boundary_local]] = (
                TILT_STATUS_NEAR_TIE_CAP
            )
            status[pending_idx[cap_local & ~singleton_local & ~near_cap_boundary_local]] = (
                TILT_STATUS_PARTIAL_TIE_CAP
            )

        remaining_local = ~cap_local
        if np.any(remaining_local):
            remaining_rows = pending_idx[remaining_local]
            q_solved, kl_solved, conv, sub_status = _solve_tilt_batch(
                normalized[remaining_rows], log_reference, reference, tau
            )
            solution[remaining_rows] = q_solved
            kl_residual[remaining_rows] = kl_solved - tau
            converged[remaining_rows] = conv
            status[remaining_rows] = sub_status

    kl_value = kl_residual + tau
    feasible = np.isfinite(kl_residual) & (kl_value <= tau + _TILT_FEASIBILITY_SLACK)
    return BatchedKLTiltResult(
        solution=solution,
        kl_residual=kl_residual,
        feasible=feasible,
        converged=converged,
        status=status,
    )


# ---- the homogeneity gap of sections/06-homogeneous.tex -----------------------
#
# Neither suite measured `eq:homogeneity-gap` directly before this module gained
# the functions below. The sigma sweep of `scripts/synthetic_coverage.py` is not
# a substitute: zero label noise still leaves pattern-dependent disagreement, and
# a common favourite can still be excluded by the ball. With m = 4 actions and a
# uniform reference a pure action needs tau >= log(4) ~ 1.3863, so the suite's
# tau = 0.5 does not make one feasible -- Gamma has to be computed, not assumed
# from a parameter named "homogeneity".
#
# Every quantity here is evaluator-only: it reads held-out utilities and must
# never be exposed to training.
