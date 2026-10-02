"""Batch NLHF: a fixed lottery fitted to pooled passive comparisons.

The study fits this pooled baseline from passive comparisons, with the same
KL radius, comparison budget, and reference lottery as the corresponding GLHF
condition. Active GLHF uses its own policy-dependent pair sampler.

What this is, precisely
----------------------
It estimates pairwise win probabilities from the comparison log and numerically
solves the resulting zero-sum game, with both players restricted to the same
ball ``KL(policy || reference) <= tau``. The manuscript calls this baseline
"batch NLHF".

It is **not** an implementation of the Nash-MD iterative training procedure of
Munos et al., ICML 2024 (https://proceedings.mlr.press/v235/munos24a.html), and
must not be described as one. Constrained and regularized NLHF have equivalent
solutions only under a *solution-dependent* correspondence between the radius
and the regularization coefficient, so ``tau`` here is a **KL radius**, not the
Nash-MD regularization coefficient. See Golz, Haghtalab and Yang
(https://arxiv.org/html/2505.23749) for the constrained NLHF definition and that
equivalence; their result that Nash learning attains the minimax optimal
distortion is why NLHF -- and not RLHF or DPO -- is the baseline worth matching.

The baseline receives the whole comparison log at once, while the groupwise
arms make a single online pass. The numerical solver's returned policy is
checked against the exploitability tolerance; exact equilibrium attainment is
not assumed.

Structure of the problem
------------------------
Write ``A[a][b] = 2 P(a > b) - 1`` for the population preference margins, which
is skew-symmetric. For a constant policy ``pi`` the mean deployed margin is
``pi @ A``, so a fixed challenger ``q`` gains

    advantage(q; pi) = -(pi @ A) @ q = q @ (A @ pi),

which is exactly the quantity `reporting.metrics` certifies. Hence

    exploitability(pi) = max over q in B of q @ (A @ pi),

with ``B = {p : KL(p || reference) <= tau}`` the same ball the certificate uses.
Because ``A`` is skew-symmetric, ``q = pi`` gives zero, so exploitability is
**nonnegative for every policy** and vanishes exactly at a symmetric equilibrium
of the constrained game. The solver therefore has a known optimal value of zero,
which doubles as its correctness certificate.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize
from empirical_testing.numerics import log_sum_exp as _log_sum_exp

from empirical_testing.algorithms.one_expert import kl_divergence

FloatArray = np.ndarray

_LAMBDA_FLOOR = 1e-9

#: Maximum allowed exact exploitability of a returned policy. Acceptance uses
#: this measured gap, independently of the optimizer's success flag.
EXPLOITABILITY_TOLERANCE = 2e-5

#: Human-facing name. The internal arm key stays ``nlhf`` so result schemas and
#: archived tables keep working.
DISPLAY_NAME = "Batch NLHF"

#: Baseline identifiers recorded in diagnostics. Legacy selection keeps the
#: first restart that strictly improves the gap; minimum-KL selection chooses
#: the policy closest to the reference within the approximate equilibrium set.
NLHF_BASELINE_VERSION_LEGACY = "restart_legacy"
NLHF_BASELINE_VERSION_MIN_KL = "min_kl_v1"

#: The production baseline: **one** SLSQP call, started at the reference
#: policy, and nothing else.  No random restarts, no minimum-KL secondary
#: solve, no polishing pass and no automatic retry.
#:
#: Starting *from* the reference is not the same claim as selecting the
#: minimum-KL equilibrium: it fixes where the search begins, and makes no
#: assertion about which member of a non-unique equilibrium set is returned.
#: The earlier multi-start rules chose among near-equal optima in an order that
#: depended on the floating-point environment, which is why a single
#: deterministic start replaced them.
NLHF_BASELINE_VERSION_REFERENCE_START = "reference_start_v10"

#: Valid values of the ``selection`` argument/``GWD_NLHF_SELECTION`` env var,
#: mapped to the baseline version recorded in diagnostics.
_BASELINE_VERSION_BY_SELECTION = {
    NLHF_BASELINE_VERSION_LEGACY: NLHF_BASELINE_VERSION_LEGACY,
    "min_kl": NLHF_BASELINE_VERSION_MIN_KL,
    NLHF_BASELINE_VERSION_REFERENCE_START: NLHF_BASELINE_VERSION_REFERENCE_START,
}
_VALID_SELECTIONS = tuple(_BASELINE_VERSION_BY_SELECTION)

#: Selection rules a maintained production run may use.  The historical
#: rules stay reachable for reproduction tests that ask for them *by name*, but
#: neither a default nor an environment variable can put one into production.
_PRODUCTION_SELECTIONS = (NLHF_BASELINE_VERSION_REFERENCE_START,)

#: Allowable exploitability for the minimum-KL secondary solve. Historical
#: calibration selected 3e-6 across 18 study folds, exceeding ten times the
#: worst primary gap (2.953e-6). Restart/seed/gradient perturbations stayed
#: within 1e-8 in policy distance, but two thread-count comparisons exceeded
#: that threshold (1.55e-8 and 3.81e-7). This is not a guarantee of invariance
#: across BLAS thread settings; thread pinning remains disabled by default.
#:
#: This tolerance has a different purpose from EXPLOITABILITY_TOLERANCE and
#: must be specified explicitly if the module default is unset.
NLHF_SELECTION_EPS: Optional[float] = 3e-6


@dataclass(frozen=True, eq=False)
class NLHFPolicy:
    """A single population-level lottery, applied to every respondent."""

    policy: FloatArray
    reference: FloatArray
    tau: float
    exploitability: float
    preference_matrix: FloatArray
    observed_pairs: int
    unobserved_pairs: tuple[tuple[int, int], ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return DISPLAY_NAME

    @property
    def exploitability_display(self) -> float:
        """Exploitability clamped at zero, for display only.

        The raw value is kept in :attr:`exploitability` and is what the
        correctness gate is applied to; only presentation clamps roundoff.
        """

        return max(0.0, float(self.exploitability))

    def predict(self, group_matrix: np.ndarray) -> FloatArray:
        """Return the same lottery for every row; the arm is non-groupwise.

        ``group_matrix`` supplies the row count only. This baseline is
        population-level and never consumes group indicators.
        """

        rows = np.asarray(group_matrix).shape[0]
        return np.tile(self.policy, (rows, 1))


def _validate_reference(reference: np.ndarray) -> FloatArray:
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1 or reference.size < 2:
        raise ValueError("reference must be a 1-D lottery over at least 2 alternatives")
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference must be finite")
    if np.any(reference <= 0.0):
        raise ValueError("reference must have full support; a zero entry makes KL infinite")
    if not np.isclose(reference.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("reference must sum to one")
    return reference


def estimate_preference_matrix(
    pairs: np.ndarray, signs: np.ndarray, n_alternatives: int
) -> tuple[FloatArray, int, tuple[tuple[int, int], ...]]:
    """Estimate ``P(a > b)`` from a logged comparison stream.

    Pairs are drawn uniformly and independently of the respondent, so the
    empirical win rate among respondents who happened to receive a given pair is
    unbiased and the constant propensities cancel.

    Returns ``(matrix, n_observed_pairs, unobserved_pairs)``. Unobserved pairs
    are **imputed at 0.5 and reported**, never silently presented as evidence.
    """

    pairs = np.asarray(pairs, dtype=np.int64)
    signs = np.asarray(signs, dtype=np.float64)
    if int(n_alternatives) < 2:
        raise ValueError("n_alternatives must be at least 2")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_rows, 2)")
    if signs.shape != (pairs.shape[0],):
        raise ValueError("signs must have shape (n_rows,)")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("signs must encode a strict winner as -1 or +1")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= n_alternatives):
        raise ValueError("pair indices out of range")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("a comparison must involve two distinct alternatives")

    left, right = pairs[:, 0], pairs[:, 1]
    wins = np.zeros((n_alternatives, n_alternatives), dtype=np.float64)
    counts = np.zeros((n_alternatives, n_alternatives), dtype=np.float64)
    left_won = (signs > 0.0).astype(np.float64)
    np.add.at(wins, (left, right), left_won)
    np.add.at(wins, (right, left), 1.0 - left_won)
    np.add.at(counts, (left, right), 1.0)
    np.add.at(counts, (right, left), 1.0)

    matrix = np.full((n_alternatives, n_alternatives), 0.5, dtype=np.float64)
    seen = counts > 0
    matrix[seen] = wins[seen] / counts[seen]
    np.fill_diagonal(matrix, 0.5)

    upper = np.triu(counts, 1)
    observed = int((upper > 0).sum())
    missing = tuple(
        (int(a), int(b))
        for a in range(n_alternatives)
        for b in range(a + 1, n_alternatives)
        if counts[a, b] == 0
    )
    return matrix, observed, missing


def preference_margins(matrix: FloatArray) -> FloatArray:
    """Skew-symmetric margins ``A[a][b] = 2 P(a > b) - 1``.

    Orientation: ``A[a][b] > 0`` means alternative ``a`` is preferred to ``b``.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("preference matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("preference matrix must be finite")
    margins = 2.0 * matrix - 1.0
    np.fill_diagonal(margins, 0.0)
    # Enforce exact skew-symmetry against floating-point drift.
    return 0.5 * (margins - margins.T)


def _smoothed_exploitability(values: FloatArray, log_reference: FloatArray,
                             scale: float, tau: float) -> float:
    """Dual of ``max over q in B of q @ values``.

    ``max_{KL(q||u) <= tau} q @ v = min_{lambda > 0} lambda*tau
    + lambda*log(sum_j u_j exp(v_j / lambda))``. The right-hand side is the
    perspective of log-sum-exp, hence jointly convex in ``(v, lambda)``, and
    ``v`` is linear in the policy -- so minimizing over the policy is convex.
    """

    if scale <= _LAMBDA_FLOOR:
        return float(np.max(values))
    return float(scale * tau + scale * _log_sum_exp(values / scale + log_reference))


def exact_exploitability(
    policy: FloatArray, margins: FloatArray, reference: FloatArray, tau: float
) -> float:
    """``max over q in B of q @ (A @ policy)`` using the certificate's own tilt.

    Deliberately routed through `reporting.metrics._kl_tilt` so the solver is
    scored by exactly the primitive the certificate uses.
    """

    from empirical_testing.reporting.metrics import _kl_tilt

    values = margins @ np.asarray(policy, dtype=np.float64)
    challenger = _kl_tilt(np.asarray(reference, dtype=np.float64), values, float(tau))
    return float(challenger @ values)


#: Which derivative source `solve_constrained_nash`'s SLSQP call uses.
#: ``"finite_difference"`` is scipy's default `approx_derivative` behaviour
#: (unchanged); ``"analytic"`` is the opt-in closed-form gradient/Jacobian
#: below. Resolved once per solve, mirroring
#: `span_adaptive._resolve_kernel_backend`'s once-per-fit pattern.
_VALID_GRADIENT_MODES = ("analytic", "finite_difference")


def _resolve_gradient_mode(gradient: Optional[str]) -> str:
    """Resolve the SLSQP derivative source, once per solve.

    Precedence: the explicit ``gradient`` argument, else the
    ``GWD_NLHF_GRADIENT`` environment variable, else the default
    ``"finite_difference"`` -- the existing, unmodified scipy behaviour.
    The analytic path is opt-in only.
    Unknown names raise :class:`ValueError`.
    """

    name = gradient if gradient is not None else os.environ.get(
        "GWD_NLHF_GRADIENT", "finite_difference"
    )
    if name not in _VALID_GRADIENT_MODES:
        raise ValueError(
            f"unknown gradient {name!r}; expected one of {_VALID_GRADIENT_MODES}"
        )
    return name


def _resolve_selection(selection: Optional[str]) -> str:
    """Resolve the equilibrium-selection rule once per solve.

    The default is ``"reference_start_v10"``. Historical ``"min_kl"`` and
    ``"restart_legacy"`` rules require an explicit ``selection`` argument;
    ``GWD_NLHF_SELECTION`` may select only a production rule. Unknown names
    and attempts to select a historical rule through the environment raise
    :class:`ValueError`.
    """

    if selection is not None:
        name = selection
    else:
        # The environment may not promote a historical rule into a run that did
        # not ask for one: leakage of ``GWD_NLHF_SELECTION`` from a shell or a
        # parent process is exactly how a production run would silently stop
        # being one-start.  An explicit argument is the only way to select a
        # non-production rule.
        requested = os.environ.get("GWD_NLHF_SELECTION")
        if requested is not None and requested not in _PRODUCTION_SELECTIONS:
            raise ValueError(
                f"GWD_NLHF_SELECTION={requested!r} would select a non-production "
                f"equilibrium-selection rule; production runs use "
                f"{NLHF_BASELINE_VERSION_REFERENCE_START!r}. Pass "
                f"selection={requested!r} explicitly if this is a reproduction run."
            )
        name = requested if requested is not None else NLHF_BASELINE_VERSION_REFERENCE_START
    if name not in _VALID_SELECTIONS:
        raise ValueError(
            f"unknown selection {name!r}; expected one of {_VALID_SELECTIONS}"
        )
    return name


def _tilted_challenger(values: FloatArray, log_reference: FloatArray,
                       scale: float) -> FloatArray:
    """``softmax(values / scale + log_reference)``.

    This is both the maximizer of the smoothed dual
    (`_smoothed_exploitability`'s ``max_{q in B} q @ v`` surrogate) and,
    algebraically, ``d/dvalues`` of that same function. Uses the
    project's own bit-identical-to-SciPy `log_sum_exp` so the value this is
    built from matches `_smoothed_exploitability` exactly.
    """

    shifted = values / scale + log_reference
    return np.exp(shifted - _log_sum_exp(shifted))


def _objective_gradient(free: np.ndarray, margins: FloatArray,
                        log_reference: FloatArray, tau: float, m: int) -> FloatArray:
    """Analytic gradient of ``objective`` (the SLSQP ``jac=`` callback).

    Closed form:

    ``values = margins @ free[:m]``, ``scale = exp(free[m])``,
    ``challenger = softmax(values / scale + log_reference)``. Then, because
    ``margins`` is skew-symmetric (``margins.T == -margins``):

    ``d objective / d free[:m] = -(margins @ challenger)``
    ``d objective / d free[m]  = objective(free) - challenger @ values``

    The ``scale <= _LAMBDA_FLOOR`` branch of `_smoothed_exploitability`
    (plain ``max(values)``, non-differentiable at ties) is included for
    symmetry with the value function, but is unreachable under this solver's
    own bounds: the lower bound on ``free[m]`` is ``log(1e-6)``, so
    ``scale >= 1e-6 > _LAMBDA_FLOOR (1e-9)`` always holds (`nlhf.py:276`).
    Reached only by a tie-averaged subgradient of ``max``, which SLSQP's
    finite-difference path would itself approximate inconsistently right at
    the tie (an ordinary secant step lands on one side or the other), so no
    stronger claim than "a valid subgradient" is made for this branch.
    """

    values = margins @ free[:m]
    scale = float(np.exp(free[m]))
    grad = np.empty(m + 1, dtype=np.float64)
    if scale <= _LAMBDA_FLOOR:
        top = np.max(values)
        at_top = values == top
        grad_v = at_top.astype(np.float64)
        grad_v /= grad_v.sum()
        grad[:m] = -(margins @ grad_v)
        grad[m] = 0.0
        return grad
    challenger = _tilted_challenger(values, log_reference, scale)
    objective_value = scale * tau + scale * _log_sum_exp(values / scale + log_reference)
    grad[:m] = -(margins @ challenger)
    grad[m] = objective_value - float(challenger @ values)
    return grad


def _equality_constraint_gradient(m: int) -> FloatArray:
    """Constant gradient of ``sum(z[:m]) - 1``: ones on the policy block."""

    grad = np.zeros(m + 1, dtype=np.float64)
    grad[:m] = 1.0
    return grad


def _kl_constraint_gradient(z: np.ndarray, reference: FloatArray, m: int) -> FloatArray:
    """Analytic gradient of ``tau - KL(_safe(z[:m]) || reference)``.

    Writing
    ``raw = clip(z[:m], 0, None)``, ``S = sum(raw)``, ``p = raw / S`` (the
    same quantity `_safe` returns):

    ``d KL(p || reference) / d z_k = (log(p_k / reference_k) - KL(p || reference)) / S``

    so the constraint's gradient (``tau`` minus that) is
    ``(KL(p || reference) - log(p_k / reference_k)) / S``. Under this
    solver's own bounds ``z[:m] in [0, 1]^m``, the clip is a no-op except at
    the corner ``z[:m] == 0`` where `_safe` itself switches to its uniform
    fallback (``nlhf.py:378-385``) -- a genuine non-differentiable point of
    ``_safe``, not introduced here. That corner returns an all-zero gradient
    block (a valid subgradient choice; SLSQP cannot take a two-sided
    finite-difference step there either, since every coordinate already
    sits on its lower bound).
    """

    grad = np.zeros(m + 1, dtype=np.float64)
    raw = np.clip(z[:m], 0.0, None)
    total = float(raw.sum())
    if total <= 0.0:
        return grad
    p = raw / total
    positive = p > 0.0
    # p_k == 0 only when z_k was clipped from a negative SLSQP iterate (bounds
    # keep z[:m] >= 0 in practice, so this is a defensive corner, not the
    # normal case); log(p_k) diverges there, matching KL's genuine
    # singularity as a coordinate's mass -> 0, not an artifact of `_safe`.
    log_ratio = np.zeros(m, dtype=np.float64)
    log_ratio[positive] = np.log(p[positive] / reference[positive])
    kl = float(np.sum(np.where(positive, p * log_ratio, 0.0)))
    grad[:m] = np.where(positive, (kl - log_ratio) / total, 0.0)
    return grad


#: Numerical slack added when re-verifying the secondary solve's
#: *exact* exploitability against `eps_sel`, and when re-verifying its KL
#: against `tau`. `1e-9` matches the KL-ball feasibility slack already used
#: for the primary restart loop just below (`nlhf.py` `kl_divergence(...) >
#: tau + 1e-9`) and `project_to_kl_ball`'s own bisection target -- both
#: calibrated to SLSQP's own active-constraint residual, not to `eps_sel`'s
#: scale, so re-verification rejects genuine infeasibility without rejecting
#: the solver's last-step roundoff.
_MIN_KL_FEASIBILITY_SLACK = 1e-9

#: Threshold below which two verified secondary-
#: solve KL values are treated as a numerical tie rather than a genuine
#: ordering (the achieved KL
#: *value* across every diagnosed restart/tolerance/arithmetic-noise variant
#: on the tested tie folds agreed to well under this, while the *policy*
#: (argmin) varied by up to ~1e-5 Linf -- the value is well-conditioned, only
#: the argmin is not). Same order as `_MIN_KL_FEASIBILITY_SLACK`, which is
#: already calibrated to this solver's own re-verification noise floor.
_MIN_KL_TIE_MARGIN = 1e-9

#: Tighter SLSQP settings for the one-shot polish. Tightening ftol reduced
#: policy spread on near-tied folds in historical checks; 2000 iterations
#: leaves room to converge without changing the optimization problem.
_MIN_KL_POLISH_MAXITER = 2000
_MIN_KL_POLISH_FTOL = 1e-15


def _min_kl_objective_gradient(free: np.ndarray, reference: FloatArray, m: int) -> FloatArray:
    """Analytic gradient of ``KL(_safe(free[:m]) || reference)`` w.r.t. ``free``.

    `_kl_constraint_gradient` already derives ``d(tau - KL)/dz = -dKL/dz``
    (`nlhf.py` docstring above it); negating gives ``dKL/dz`` directly, with
    the same corner-case handling (the `_safe` clip/renormalize corner). The
    ``t = free[m]`` (log-scale) block is exactly zero: this objective does not
    depend on it.
    """

    return -_kl_constraint_gradient(free, reference, m)


def _solve_min_kl(
    margins: FloatArray,
    reference: FloatArray,
    tau: float,
    eps_sel: float,
    log_reference: FloatArray,
    m: int,
    warm_start_policy: FloatArray,
    warm_start_log_scale: float,
    *,
    maxiter: int = 500,
    ftol: float = 1e-12,
) -> tuple[Optional[FloatArray], Optional[float], str, int]:
    """Optional SLSQP secondary solve minimizing KL within an equilibrium set.

    ``maxiter``/``ftol`` control the SLSQP iteration limit and stopping
    tolerance. `_solve_min_kl_robust` uses tighter values for its polish pass,
    with the same objective, constraints, and gradients.

    ``min_{pi} KL(pi || reference)`` subject to the smoothed-exploitability
    upper bound ``smoothed(pi, scale) <= eps_sel`` (a safe over-approximation
    of the true constraint ``E(pi) <= eps_sel`` -- weak duality, see
    `_smoothed_exploitability`'s docstring), the same
    KL-ball (``KL(pi||reference) <= tau``) and simplex constraints as the
    primary solve, reusing its exact ``free = [pi, t]`` parameterization and
    bounds. Objective and the exploitability constraint swap roles relative to
    `solve_constrained_nash`'s primary objective/constraint, reusing its
    analytic-gradient machinery verbatim for the swapped-in constraint.
    Analytic derivatives are used here to reduce finite-difference error on
    degenerate folds with tied actions.

    Returns ``(policy_or_None, exact_gap_or_None, status, iterations)``.
    ``status`` is ``"ok"`` or ``"min_kl_failed:<reason>"``; the caller must
    fall back to the primary solution whenever status is not ``"ok"`` --
    acceptance here, as everywhere in this module, is by the recomputed exact
    objective, never by ``result.success``.
    """

    def kl_objective(free: np.ndarray) -> float:
        return float(kl_divergence(_safe(free[:m]), reference))

    def kl_objective_jac(free: np.ndarray) -> FloatArray:
        return _min_kl_objective_gradient(free, reference, m)

    def smoothed_constraint(free: np.ndarray) -> float:
        return float(
            eps_sel
            - _smoothed_exploitability(
                margins @ free[:m], log_reference, float(np.exp(free[m])), tau
            )
        )

    def smoothed_constraint_jac(free: np.ndarray) -> FloatArray:
        return -_objective_gradient(free, margins, log_reference, tau, m)

    equality_gradient = _equality_constraint_gradient(m)
    constraints = (
        {"type": "eq", "fun": lambda z: float(np.sum(z[:m]) - 1.0),
         "jac": lambda z: equality_gradient},
        {"type": "ineq",
         "fun": lambda z: float(tau - kl_divergence(_safe(z[:m]), reference)),
         "jac": lambda z: _kl_constraint_gradient(z, reference, m)},
        {"type": "ineq", "fun": smoothed_constraint, "jac": smoothed_constraint_jac},
    )
    bounds = [(0.0, 1.0)] * m + [(np.log(1e-6), np.log(1e3))]
    start = np.concatenate([_safe(warm_start_policy), [warm_start_log_scale]])

    try:
        result = minimize(
            kl_objective, start, method="SLSQP", jac=kl_objective_jac, bounds=bounds,
            constraints=constraints, options={"maxiter": maxiter, "ftol": ftol},
        )
    except Exception as error:  # solver robustness, mirroring the primary loop
        return None, None, f"min_kl_failed:solver_exception:{type(error).__name__}", 0

    iterations = int(getattr(result, "nit", 0))
    candidate = _safe(result.x[:m])
    if kl_divergence(candidate, reference) > tau + _MIN_KL_FEASIBILITY_SLACK:
        # Same pull-back as the primary loop: SLSQP can stop a hair outside
        # the ball when the KL constraint is active.
        candidate = project_to_kl_ball(candidate, reference, tau)
        if kl_divergence(candidate, reference) > tau + _MIN_KL_FEASIBILITY_SLACK:
            return None, None, "min_kl_failed:kl_ball_infeasible", iterations

    # Acceptance is by the recomputed *exact* objective, never by the smoothed
    # surrogate or `result.success`.
    try:
        exact_gap = exact_exploitability(candidate, margins, reference, tau)
    except AssertionError as error:
        # `exact_exploitability` routes through
        # `reporting.metrics._kl_tilt`, whose own root-polish loop
        # (metrics.py:556-568) can -- for some candidates this secondary
        # solve produces near a near-exact tie -- leave the returned
        # challenger a hair outside `tau + metrics._TILT_FEASIBILITY_SLACK`
        # and raise, rather than return a value. Catch the failure and report it as
        # a status, exactly as `min_kl_failed:solver_exception` already does
        # for a SLSQP exception, rather than letting it crash the pipeline.
        return None, None, f"min_kl_failed:kl_tilt_assertion:{error}", iterations
    if exact_gap > eps_sel + _MIN_KL_FEASIBILITY_SLACK:
        return None, float(exact_gap), "min_kl_failed:exact_exploitability_exceeds_eps_sel", iterations

    return candidate, float(exact_gap), "ok", iterations


def _solve_min_kl_robust(
    margins: FloatArray,
    reference: FloatArray,
    tau: float,
    eps_sel: float,
    log_reference: FloatArray,
    m: int,
    warm_start_policy: FloatArray,
    warm_start_log_scale: float,
) -> tuple[Optional[FloatArray], Optional[float], str, int, dict[str, Any]]:
    """Warm-start-robust wrapper around the optional `_solve_min_kl` step.

    On near-tied folds, the exact exploitability constraint is identically flat along
    the tied-action mass-split direction (an intrinsic property of the
    near-exact tie in ``margins``, not a solver artifact -- the equilibrium
    *segment*, not just its boundary, is degenerate there). Nothing pins the
    policy down in that direction except the KL objective's own curvature, so
    SLSQP's ``ftol``-based stopping rule (an objective-*value* tolerance)
    only bounds policy-space accuracy to roughly ``sqrt(ftol / curvature)``.
    The measured KL *value* at convergence was essentially identical (agreed
    to >=10 significant digits) across every restart seed, thread setting and
    gradient mode tested -- confirming the target value is found reliably --
    while the arg-min policy differs by up to ~1.3e-5 Linf depending on which
    point in that shallow valley the warm start happens to land SLSQP in.

    This applies two independent, bounded fixes to that mechanism, each
    keeping the objective/constraints/acceptance rule byte-identical to
    `_solve_min_kl` (only the *starting point* and *SLSQP options* vary):

    (i) **Canonical warm start.** In addition to the caller's own warm start
        (typically the primary restart-legacy solution -- itself
        seed/restart/thread dependent), also solve from a fixed, deterministic
        start independent of any of that: the reference policy itself with
        log-scale ``log(max(tau, 1e-3))``. Whichever attempt verifies
        feasible with the strictly lower KL (by more than
        `_MIN_KL_TIE_MARGIN`) wins; within that margin the two are a genuine
        numerical tie and the **canonical** result
        is preferred deterministically, since it alone does not depend on the
        primary restart's seed, thread count or gradient mode.
    (ii) **Polish.** The winning candidate is re-solved once more from its own
         converged point, with `_MIN_KL_POLISH_MAXITER`/`_MIN_KL_POLISH_FTOL`
         in place of the default solver settings. The polished point is accepted
         only if it independently
         verifies feasible *and* its KL does not increase beyond
         `_MIN_KL_TIE_MARGIN` -- it can only match or improve on the winner,
         never regress or violate `eps_sel`.

    Returns ``(policy_or_None, exact_gap_or_None, status, iterations, extra)``
    where ``extra`` carries the per-attempt breakdown for diagnostics
    (``min_kl_warm_start``, ``min_kl_primal_status``,
    ``min_kl_canonical_status``, ``min_kl_primal_kl``,
    ``min_kl_canonical_kl``, ``min_kl_polish_applied``,
    ``min_kl_polish_kl_before``, ``min_kl_polish_kl_after``). ``status`` and
    the first four return values have exactly `_solve_min_kl`'s contract, so
    `_finalize_selection` needs no branching on which path was taken.
    """

    canonical_log_scale = float(np.log(max(tau, 1e-3)))
    primal_result = _solve_min_kl(
        margins, reference, tau, eps_sel, log_reference, m,
        warm_start_policy, warm_start_log_scale,
    )
    canonical_result = _solve_min_kl(
        margins, reference, tau, eps_sel, log_reference, m,
        reference, canonical_log_scale,
    )

    attempts: list[tuple[str, tuple]] = [("primal", primal_result), ("canonical", canonical_result)]
    ok_attempts = [(name, res) for name, res in attempts if res[2] == "ok"]

    extra: dict[str, Any] = {
        "min_kl_primal_status": primal_result[2],
        "min_kl_canonical_status": canonical_result[2],
        "min_kl_primal_kl": (
            float(kl_divergence(primal_result[0], reference))
            if primal_result[0] is not None else None
        ),
        "min_kl_canonical_kl": (
            float(kl_divergence(canonical_result[0], reference))
            if canonical_result[0] is not None else None
        ),
        "min_kl_polish_applied": False,
        "min_kl_polish_kl_before": None,
        "min_kl_polish_kl_after": None,
    }

    if not ok_attempts:
        # Neither start produced a feasible secondary solution. Report the
        # primal attempt's failure (it is the one `_solve_min_kl` has always
        # reported, so unrecognized-status handling downstream is unchanged);
        # the canonical attempt's own status is still recorded in `extra`.
        extra["min_kl_warm_start"] = "primal"
        return (*primal_result, extra)

    # Deterministic winner selection: strictly-lower verified KL wins; a
    # near-tie (within `_MIN_KL_TIE_MARGIN`) is broken in favor of the
    # canonical start, since it alone is independent of the caller's seed,
    # restart count and thread setting.
    kls = {name: float(kl_divergence(res[0], reference)) for name, res in ok_attempts}
    if len(ok_attempts) == 1:
        winner_name, winner = ok_attempts[0]
    else:
        (primal_name, primal_res), (canon_name, canon_res) = ok_attempts
        if kls[primal_name] + _MIN_KL_TIE_MARGIN < kls[canon_name]:
            winner_name, winner = primal_name, primal_res
        elif kls[canon_name] + _MIN_KL_TIE_MARGIN < kls[primal_name]:
            winner_name, winner = canon_name, canon_res
        else:
            winner_name, winner = canon_name, canon_res  # deterministic tie-break
    extra["min_kl_warm_start"] = winner_name

    winner_policy, winner_gap, winner_status, winner_iterations = winner
    winner_kl = kls[winner_name]

    # (ii) Polish: one more SLSQP pass from the winner's own converged point,
    # with tightened tolerances. Never accepted if it regresses KL or fails
    # to verify feasible.
    polished = _solve_min_kl(
        margins, reference, tau, eps_sel, log_reference, m,
        winner_policy, canonical_log_scale,
        maxiter=_MIN_KL_POLISH_MAXITER, ftol=_MIN_KL_POLISH_FTOL,
    )
    polished_policy, polished_gap, polished_status, polished_iterations = polished
    if polished_status == "ok":
        polished_kl = float(kl_divergence(polished_policy, reference))
        extra["min_kl_polish_kl_before"] = winner_kl
        extra["min_kl_polish_kl_after"] = polished_kl
        if polished_kl <= winner_kl + _MIN_KL_TIE_MARGIN:
            extra["min_kl_polish_applied"] = True
            return polished_policy, polished_gap, polished_status, polished_iterations, extra

    return winner_policy, winner_gap, winner_status, winner_iterations, extra


def _finalize_selection(
    policy: FloatArray,
    gap: float,
    diagnostics: dict[str, Any],
    margins: FloatArray,
    reference: FloatArray,
    tau: float,
    m: int,
    log_reference: FloatArray,
    selection: Optional[str],
    eps_sel: Optional[float],
    warm_start_log_scale: float,
) -> tuple[FloatArray, float, dict[str, Any]]:
    """Apply the requested equilibrium-selection rule to a primary solve.

    Called from `solve_constrained_nash`'s two exit points (the ``tau <= 0``
    short-circuit and the end of the restart loop) so both inherit the same
    logic. `fit_pooled_nlhf` inherits this behavior.

    ``selection="reference_start_v10"`` (the default) returns the single
    primary solve unchanged: it performs no secondary optimization at all, so
    ``optimizer_calls`` stays at the one call the primary path made.

    ``selection="restart_legacy"`` (pass it explicitly for historical
    reproduction; the environment can no longer select it) returns
    ``(policy, gap, diagnostics)`` unchanged apart from the two additive
    diagnostics keys ``selection``/``baseline_version``, so callers pinned to
    it and archived results reproduced under it are unaffected byte-for-byte.
    """

    resolved_selection = _resolve_selection(selection)
    diagnostics["selection"] = resolved_selection
    diagnostics["baseline_version"] = _BASELINE_VERSION_BY_SELECTION[resolved_selection]
    if resolved_selection == NLHF_BASELINE_VERSION_LEGACY:
        return policy, gap, diagnostics
    if resolved_selection == NLHF_BASELINE_VERSION_REFERENCE_START:
        # The one primary solve *is* the answer: no secondary minimum-KL solve,
        # no polish, no second optimizer call of any kind.
        diagnostics.setdefault("optimizer_calls", 1)
        return policy, gap, diagnostics

    diagnostics["primary_exact_exploitability"] = float(gap)
    diagnostics["primary_kl"] = float(kl_divergence(policy, reference))

    resolved_eps_sel = eps_sel if eps_sel is not None else NLHF_SELECTION_EPS
    if resolved_eps_sel is None:
        raise ValueError(
            "selection='min_kl' requires eps_sel: pass it explicitly, or set "
            "nlhf.NLHF_SELECTION_EPS. The selection tolerance is separate "
            "from EXPLOITABILITY_TOLERANCE."
        )
    eps_sel = float(resolved_eps_sel)
    diagnostics["eps_sel"] = eps_sel

    if eps_sel < gap:
        # The selection tolerance must cover the primary solve's attained gap.
        diagnostics["min_kl_status"] = (
            f"min_kl_failed:eps_sel_below_primary_gap({eps_sel:.6e}<{gap:.6e})"
        )
        diagnostics["secondary_exact_exploitability"] = None
        diagnostics["selected_kl"] = diagnostics["primary_kl"]
        diagnostics["secondary_iterations"] = 0
        return policy, gap, diagnostics

    secondary_policy, secondary_gap, status, iterations, robust_diagnostics = _solve_min_kl_robust(
        margins, reference, tau, eps_sel, log_reference, m, policy, warm_start_log_scale,
    )
    diagnostics["secondary_iterations"] = iterations
    diagnostics["min_kl_status"] = status
    diagnostics.update(robust_diagnostics)
    if status != "ok":
        diagnostics["secondary_exact_exploitability"] = (
            None if secondary_gap is None else float(secondary_gap)
        )
        diagnostics["selected_kl"] = diagnostics["primary_kl"]
        return policy, gap, diagnostics

    diagnostics["secondary_exact_exploitability"] = float(secondary_gap)
    diagnostics["selected_kl"] = float(kl_divergence(secondary_policy, reference))
    return secondary_policy, float(secondary_gap), diagnostics


def solve_constrained_nash(
    margins: FloatArray,
    reference: FloatArray,
    tau: float,
    restarts: int = 4,
    seed: int = 0,
    gradient: Optional[str] = None,
    selection: Optional[str] = None,
    eps_sel: Optional[float] = None,
) -> tuple[FloatArray, float, dict[str, Any]]:
    """Minimize exploitability over the KL ball.

    Returns ``(policy, exact_exploitability, diagnostics)``. The optimum value is
    known to be zero, so the returned exploitability *is* the duality gap.

    The default ``selection="reference_start_v10"`` performs one SLSQP solve
    from the reference, with no random restarts or secondary optimization.
    The explicitly selected historical ``"min_kl"`` rule runs a secondary
    SLSQP solve minimizing
    ``KL(pi||reference)`` over the approximate equilibrium set
    ``{pi in B_tau : exact_exploitability(pi) <= eps_sel}``, using
    `NLHF_SELECTION_EPS` unless ``eps_sel`` is passed explicitly, independently
    re-verified, falling back to the primary (legacy) solution with a recorded
    ``diagnostics["min_kl_status"]`` on any failure -- never silently;
    ``"restart_legacy"`` explicitly selects the historical
    first-strict-improvement restart rule. Neither historical rule can be
    selected through ``GWD_NLHF_SELECTION``.
    """

    margins = np.asarray(margins, dtype=np.float64)
    reference = _validate_reference(reference)
    tau = float(tau)
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and nonnegative")
    if margins.shape != (reference.size, reference.size):
        raise ValueError("margins and reference disagree on the number of alternatives")
    if not np.allclose(margins, -margins.T, atol=1e-12, rtol=0.0):
        raise ValueError("margins must be skew-symmetric")

    m = reference.size
    log_reference = np.log(reference)
    resolved_gradient = _resolve_gradient_mode(gradient)
    resolved_selection = _resolve_selection(selection)
    one_start = resolved_selection == NLHF_BASELINE_VERSION_REFERENCE_START

    if tau <= 0.0:
        gap = exact_exploitability(reference, margins, reference, tau)
        diagnostics0: dict[str, Any] = {
            "restarts_used": 0, "solver_success": True,
            "final_kl": 0.0, "iterations": 0,
            "gradient_mode": resolved_gradient,
            # A zero-radius ball is the single point {reference}: the optimum
            # is known without solving, so no optimizer call is made.
            "optimizer_calls": 0,
            "initialization": "reference_policy",
            "exact_exploitability": float(gap),
            "solver_status": "trivial: B_tau = {reference}, no optimization required",
        }
        # B_tau is the single point {reference} here, so the optional secondary
        # solve (when selection="min_kl") is trivial; still routed through
        # `_finalize_selection` so eps_sel is validated and diagnostics are
        # recorded consistently with the restarts>0 path below.
        return _finalize_selection(
            reference.copy(), gap, diagnostics0, margins, reference, tau, m,
            log_reference, selection, eps_sel, np.log(1e-3),
        )

    def objective(free: np.ndarray) -> float:
        return _smoothed_exploitability(
            margins @ free[:m], log_reference, float(np.exp(free[m])), tau
        )

    constraints = (
        {"type": "eq", "fun": lambda z: float(np.sum(z[:m]) - 1.0)},
        {"type": "ineq", "fun": lambda z: float(tau - kl_divergence(_safe(z[:m]), reference))},
    )
    bounds = [(0.0, 1.0)] * m + [(np.log(1e-6), np.log(1e3))]

    # Opt-in analytic derivatives. `fun` entries above are untouched
    # either way (same closures, same values); only `jac=` is added, and only
    # under "analytic". The default uses finite differences.
    if resolved_gradient == "analytic":
        objective_jac = lambda free: _objective_gradient(  # noqa: E731
            free, margins, log_reference, tau, m
        )
        equality_gradient = _equality_constraint_gradient(m)
        constraints = (
            {**constraints[0], "jac": lambda z: equality_gradient},
            {**constraints[1], "jac": lambda z: _kl_constraint_gradient(z, reference, m)},
        )
    else:
        objective_jac = None

    rng = np.random.default_rng(seed)
    best_policy = reference.copy()
    best_gap = exact_exploitability(best_policy, margins, reference, tau)
    # Warm-start log-scale for the optional secondary (min-KL) solve --
    # the accepted restart's own converged `t`, matching the initial guess
    # below when no restart is ever accepted (best_policy stays the reference).
    best_log_scale = float(np.log(max(tau, 1e-3)))
    diagnostics: dict[str, Any] = {
        "restarts_used": 0, "solver_success": False, "iterations": 0,
        "accepted_from_restart": -1,
        # Additive, informational only: how many restarts raised out of the
        # solver call and why. Never consumed for control flow or acceptance.
        "restart_failures": 0,
        "restart_failure_reasons": [],
        # Which derivative source this solve resolved to.
        "gradient_mode": resolved_gradient,
    }

    if one_start:
        # ------------------------------------------------------------------
        # Production baseline: exactly one SLSQP call, started at the reference.
        #
        # Same objective, same KL constraint, same bounds, same optimizer
        # options and the same initial auxiliary scale as attempt 0 of the
        # historical loop -- only the restarts, the secondary solve and the
        # retries are gone.  The result is whatever that one call produced:
        # there is no candidate to compare against and therefore no silent
        # fallback to the reference.
        # ------------------------------------------------------------------
        start = np.concatenate([reference.copy(), [np.log(max(tau, 1e-3))]])
        diagnostics.update(
            restarts_used=1, optimizer_calls=1, initialization="reference_policy",
            accepted_from_restart=0,
        )
        try:
            result = minimize(
                objective, start, method="SLSQP", jac=objective_jac, bounds=bounds,
                constraints=constraints, options={"maxiter": 500, "ftol": 1e-12},
            )
        except Exception as error:
            # Reported, never retried and never replaced by the reference: a
            # fit that did not happen must be visible as a failure downstream.
            diagnostics.update(
                solver_success=False,
                solver_status=f"{type(error).__name__}: {error}",
                failure_reason="optimizer raised",
                final_kl=float("nan"),
                exact_exploitability=float("inf"),
            )
            diagnostics["restart_failures"] = 1
            diagnostics["restart_failure_reasons"] = [
                {"attempt": 0, "type": type(error).__name__, "message": str(error)}
            ]
            return _finalize_selection(
                reference.copy(), float("inf"), diagnostics, margins, reference, tau,
                m, log_reference, resolved_selection, eps_sel,
                float(np.log(max(tau, 1e-3))),
            )
        candidate = _safe(result.x[:m])
        projected = False
        if kl_divergence(candidate, reference) > tau + 1e-9:
            # The existing deterministic projection, for numerical feasibility
            # only: it moves the point back onto the ball along the geometric
            # path and optimizes nothing.
            candidate = project_to_kl_ball(candidate, reference, tau)
            projected = True
        candidate_kl = float(kl_divergence(candidate, reference))
        gap = float(exact_exploitability(candidate, margins, reference, tau))
        diagnostics.update(
            solver_success=bool(result.success),
            iterations=int(getattr(result, "nit", 0)),
            solver_status=str(getattr(result, "message", "")),
            kl_projected=projected,
            final_kl=candidate_kl,
            exact_exploitability=gap,
        )
        if candidate_kl > tau + 1e-9:
            # Even the projection could not make it feasible.  Say so; do not
            # quietly hand back the reference as though a fit had succeeded.
            diagnostics.update(
                solver_success=False,
                failure_reason=f"infeasible after projection: KL={candidate_kl:.6e} > tau={tau:.6e}",
            )
        return _finalize_selection(
            candidate, gap, diagnostics, margins, reference, tau, m,
            log_reference, resolved_selection, eps_sel, float(result.x[m]),
        )

    for attempt in range(max(1, restarts)):
        if attempt == 0:
            start_policy = reference.copy()
        else:
            start_policy = rng.dirichlet(np.full(m, 2.0))
            for _ in range(64):
                if kl_divergence(start_policy, reference) <= tau:
                    break
                start_policy = 0.5 * (start_policy + reference)
                start_policy /= start_policy.sum()
        start = np.concatenate([start_policy, [np.log(max(tau, 1e-3))]])
        diagnostics["restarts_used"] = attempt + 1
        try:
            result = minimize(
                objective, start, method="SLSQP", jac=objective_jac, bounds=bounds,
                constraints=constraints, options={"maxiter": 500, "ftol": 1e-12},
            )
        except Exception as error:  # solver robustness; see test_nlhf_restart_failures.py
            # Recorded for visibility only; does not affect which
            # restart is accepted or the reference-fallback behaviour below.
            diagnostics["restart_failures"] += 1
            diagnostics["restart_failure_reasons"].append(
                {
                    "attempt": attempt,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue
        candidate = _safe(result.x[:m])
        if kl_divergence(candidate, reference) > tau + 1e-9:
            # SLSQP can stop a hair outside the ball when the constraint is
            # active (seen at tau = 1 with m = 20: KL = 1.0000000x with an
            # exploitability of 1e-10). Discarding such a point returned the
            # reference instead of the equilibrium. Pull it back onto the ball
            # along the geometric path to the reference and score that.
            candidate = project_to_kl_ball(candidate, reference, tau)
            if kl_divergence(candidate, reference) > tau + 1e-9:
                continue
        # Acceptance is by exact verification, not by `result.success`: SLSQP
        # reports failure at perfectly good points and success at bad ones, and
        # the exploitability check below is strictly stronger. `success` is kept
        # only as a diagnostic.
        gap = exact_exploitability(candidate, margins, reference, tau)
        if gap < best_gap:
            best_policy, best_gap = candidate, gap
            best_log_scale = float(result.x[m])
            diagnostics.update(
                solver_success=bool(result.success),
                iterations=int(getattr(result, "nit", 0)),
                accepted_from_restart=attempt,
                solver_status=str(getattr(result, "message", "")),
            )

    diagnostics["final_kl"] = float(kl_divergence(best_policy, reference))
    diagnostics["exact_exploitability"] = float(best_gap)
    return _finalize_selection(
        best_policy, best_gap, diagnostics, margins, reference, tau, m,
        log_reference, selection, eps_sel, best_log_scale,
    )


def project_to_kl_ball(policy: FloatArray, reference: FloatArray, tau: float,
                       iterations: int = 60) -> FloatArray:
    """The point on the geometric path from ``reference`` to ``policy`` with KL <= tau.

    ``p_alpha = normalize(reference^(1-alpha) * policy^alpha)``; KL(p_alpha || reference)
    is increasing in alpha, so a bisection on alpha finds the feasible point
    closest to ``policy``. Returns ``policy`` itself when it is already feasible.
    """

    policy = _safe(policy)
    if kl_divergence(policy, reference) <= tau:
        return policy
    log_ref = np.log(reference)
    log_pol = np.log(np.clip(policy, 1e-300, None))

    def point(alpha: float) -> FloatArray:
        mixed = np.exp((1.0 - alpha) * log_ref + alpha * log_pol)
        return mixed / mixed.sum()

    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if kl_divergence(point(mid), reference) <= tau:
            lo = mid
        else:
            hi = mid
    return point(lo)


def _safe(vector: np.ndarray) -> FloatArray:
    """Clip to the simplex; SLSQP can leave tiny negative or unnormalized mass."""

    clipped = np.clip(np.asarray(vector, dtype=np.float64), 0.0, None)
    total = clipped.sum()
    if total <= 0.0:
        return np.full(clipped.size, 1.0 / clipped.size)
    return clipped / total


def fit_pooled_nlhf(
    pair_indices: np.ndarray,
    signs: np.ndarray,
    n_alternatives: int,
    reference: np.ndarray,
    tau: float,
    seed: int = 0,
    propensities: Optional[np.ndarray] = None,
    gradient: Optional[str] = None,
    selection: Optional[str] = None,
    eps_sel: Optional[float] = None,
) -> NLHFPolicy:
    """Fit batch NLHF from one uniformly sampled passive comparison log.

    ``propensities`` is accepted for interface symmetry with the online learners
    and is validated to be constant, which is what makes the plain empirical win
    rate unbiased; it is otherwise unused because constant propensities cancel.

    ``gradient`` selects the SLSQP solve's derivative source:
    ``"analytic"`` or ``"finite_difference"``. Resolved once per fit by
    `_resolve_gradient_mode` -- explicit argument, else ``GWD_NLHF_GRADIENT``,
    else ``"finite_difference"`` (unchanged default). The objective,
    constraints, restart loop, selection rule and this function's acceptance
    check are identical either way; only how SLSQP computes derivatives
    changes.

    ``selection``/``eps_sel`` choose the equilibrium-selection rule;
    see `solve_constrained_nash` and `_resolve_selection`. Only training
    inputs -- ``(margins derived from pair_indices/signs, reference, tau)`` --
    reach the secondary solve; no held-out data is accepted by this signature.

    Raises if the solved policy's exact Nash exploitability exceeds
    :data:`EXPLOITABILITY_TOLERANCE` -- the optimum is known to be zero, so a
    larger value means the equilibrium was not found and the arm must not be
    reported. This gate applies identically to the ``selection="min_kl"``
    return value (its own, generally tighter, ``eps_sel`` gate is separate and
    checked inside `solve_constrained_nash`).
    """

    if propensities is not None:
        propensity_array = np.asarray(propensities, dtype=np.float64)
        if propensity_array.size and not np.allclose(
            propensity_array, propensity_array[0], atol=1e-15, rtol=0.0
        ):
            raise ValueError(
                "the constrained NLHF estimator assumes a uniform pair logger; "
                "non-constant propensities require an inverse-propensity weighting"
            )

    reference = _validate_reference(reference)
    matrix, observed, missing = estimate_preference_matrix(
        pair_indices, signs, n_alternatives
    )
    if missing:
        warnings.warn(
            f"{len(missing)} unordered pair(s) never observed in this log "
            f"{missing}; their preference probabilities are imputed at 0.5 and "
            "are not evidence. Treat the resulting equilibrium as partly "
            "uninformed.",
            RuntimeWarning,
            stacklevel=2,
        )
    margins = preference_margins(matrix)
    policy, gap, diagnostics = solve_constrained_nash(
        margins, reference, tau, seed=seed, gradient=gradient,
        selection=selection, eps_sel=eps_sel,
    )
    if gap > EXPLOITABILITY_TOLERANCE:
        raise AssertionError(
            "constrained NLHF did not reach the equilibrium: exact exploitability "
            f"{gap:.3e} exceeds {EXPLOITABILITY_TOLERANCE:.1e} (the optimum is 0)"
        )
    diagnostics["unobserved_pairs"] = len(missing)
    return NLHFPolicy(
        policy=policy,
        reference=reference,
        tau=float(tau),
        exploitability=gap,
        preference_matrix=matrix,
        observed_pairs=observed,
        unobserved_pairs=missing,
        diagnostics=diagnostics,
    )
