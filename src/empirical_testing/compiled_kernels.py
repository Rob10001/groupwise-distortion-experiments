"""Opt-in Numba-compiled kernels for the adaptive learner's hot leaves.

This module is entirely optional: importing it never requires ``numba`` to be
installed, and importing it never changes any other module's behaviour --
there is no module-attribute rebinding here.  ``numerics.log_sum_exp`` and
``one_expert._softmax_from_log`` are plain functions, called by name at their
callers' own import time; nothing in this module patches them.  A caller that
wants the compiled kernels instead calls the functions in this module
directly, selected once per fit (see ``fit_groupwise_span_adaptive``'s
``kernel_backend`` argument in ``span_adaptive.py``).

Edge-case handling
-------------------
``log_sum_exp``'s contract (``numerics.py`` docstring) is: take the fast
finite-max path, or delegate to ``scipy.special.logsumexp`` whenever the max
is non-finite (any ``+inf``, any ``NaN``, or an all-``-inf`` vector).  The
delegated branch calls into SciPy, which cannot run under Numba's
``nopython`` mode.  ``log_sum_exp_numba`` below therefore performs the exact
same ``np.isfinite(a_max)`` check as the Python original in ordinary Python,
and only routes the finite case into the compiled kernel; the non-finite case
calls ``scipy.special.logsumexp`` exactly as the original does.  This is a
literal-port compiled fast path plus a Python fallback for the one branch
that cannot be compiled, not a silent approximation: on every input the
uncompiled function accepts, the two produce the identical sequence of
floating-point operations, checked by this module's test suite for both branches.

``_softmax_from_log`` is one line of arithmetic (``exp``, ``maximum``,
normalize) around a ``log_sum_exp`` call; ``softmax_from_log_numba`` calls
``log_sum_exp_numba`` for the same reason, then a compiled kernel for the
rest.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp as _scipy_logsumexp

# The span-adaptive inner-loop kernel is a literal port of code that uses
# ``one_expert._NUMERICAL_TOLERANCE``. Importing the single source of truth avoids
# silently drifting from it if it is ever tuned there.
from empirical_testing.algorithms.one_expert import _NUMERICAL_TOLERANCE

FloatArray = NDArray[np.float64]

__all__ = [
    "is_numba_available",
    "log_sum_exp_numba",
    "softmax_from_log_numba",
    "span_adaptive_inner_round_numba",
    "STATUS_OK",
    "STATUS_BINDING",
    "STATUS_NEEDS_PYTHON",
    "kl_prox_binding_numba",
    "BINDING_STATUS_OK",
    "BINDING_STATUS_NEEDS_PYTHON",
    "BINDING_STATUS_INFEASIBLE",
    "reset_binding_kernel_hits",
    "get_binding_kernel_hits",
    "round_start_numba",
    "round_mix_finish_numba",
    "round_reward_numba",
    "round_centered_and_q_numba",
    "round_moments_finish_numba",
    "round_rates_gamma_outer_numba",
    "round_finish_numba",
]

#: Same value as ``empirical_testing.numerics.TINY``, computed independently
#: so this module needs no import of ``numerics`` (that import existed only
#: to support the deleted rebinding mechanism -- see module docstring).
_TINY = np.finfo(np.float64).tiny

try:
    from numba import njit as _njit

    _NUMBA_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - exercised only without numba installed
    _njit = None
    _NUMBA_IMPORT_ERROR = _exc


def is_numba_available() -> bool:
    """Return whether ``numba`` imported successfully in this process."""

    return _njit is not None


# ---------------------------------------------------------------------------
# Compiled fast-path kernels (only defined when numba is importable).
# ---------------------------------------------------------------------------

#: NumPy's ``np.sum`` on a contiguous 1-D float64 array does not use a plain
#: running accumulator: below ``PW_BLOCKSIZE`` elements it uses 8 parallel
#: accumulators advanced 8-at-a-time and combined in a fixed tree order at the
#: end (``numpy/core/src/umath/loops_utils.h.src``, ``pairwise_sum``), and
#: above it, recursive halving. A naive
#: njit reduction (or numba's own ``np.sum``/``.sum()`` lowering) agrees with
#: NumPy for length < 8 and *disagrees* (in the last bit) from length 8
#: upward. Both compiled kernels below therefore reduce through this
#: hand-written replica instead of ``np.sum``, verified bit-identical to
#: ``np.sum`` for lengths 2-300 (a fresh, larger empirical check than the
#: probe report's).
_PW_BLOCKSIZE = 128


if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _pairwise_sum_numba(values: FloatArray) -> np.float64:
        n = values.shape[0]
        if n < 8:
            res = 0.0
            for i in range(n):
                res += values[i]
            return res
        elif n <= _PW_BLOCKSIZE:
            r0 = values[0]
            r1 = values[1]
            r2 = values[2]
            r3 = values[3]
            r4 = values[4]
            r5 = values[5]
            r6 = values[6]
            r7 = values[7]
            i = 8
            limit = n - (n % 8)
            while i < limit:
                r0 += values[i + 0]
                r1 += values[i + 1]
                r2 += values[i + 2]
                r3 += values[i + 3]
                r4 += values[i + 4]
                r5 += values[i + 5]
                r6 += values[i + 6]
                r7 += values[i + 7]
                i += 8
            res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n:
                res += values[i]
                i += 1
            return res
        else:
            n2 = n // 2
            n2 -= n2 % 8
            return _pairwise_sum_numba(values[:n2]) + _pairwise_sum_numba(values[n2:])

    @_njit(cache=True, fastmath=False)
    def _log_sum_exp_kernel_numba(values: FloatArray, a_max: np.float64) -> np.float64:
        """Literal port of the finite-max branch of ``numerics._log_sum_exp_python``."""

        at_maximum = values == a_max
        multiplicity = np.float64(np.count_nonzero(at_maximum))
        shifted = np.exp(np.where(at_maximum, -np.inf, values) - a_max)
        total = _pairwise_sum_numba(shifted)
        if total != 0.0:
            total = total / multiplicity
        return np.log1p(total) + np.log(multiplicity) + a_max

    @_njit(cache=True, fastmath=False)
    def _softmax_kernel_numba(log_weights: FloatArray, lse: np.float64) -> FloatArray:
        """Literal port of ``one_expert._softmax_from_log_python`` given ``lse``."""

        values = np.exp(log_weights - lse)
        values = np.maximum(values, _TINY)
        return values / _pairwise_sum_numba(values)

else:  # pragma: no cover - exercised only without numba installed
    _pairwise_sum_numba = None
    _log_sum_exp_kernel_numba = None
    _softmax_kernel_numba = None


def log_sum_exp_numba(values: FloatArray) -> np.float64:
    """Numba-backed ``log_sum_exp``, bit-identical to ``numerics._log_sum_exp_python``.

    Requires ``numba`` to be importable; raises :class:`RuntimeError` if it is
    not (call :func:`is_numba_available` first to check).
    """

    if not is_numba_available():
        raise RuntimeError(
            "compiled_kernels.log_sum_exp_numba requires numba, which is not "
            f"importable in this environment ({_NUMBA_IMPORT_ERROR!r})"
        )
    a_max = np.max(values)
    if not np.isfinite(a_max):
        # Same delegation as the Python original: cannot run inside nopython
        # mode, and there is no need to -- this branch is rare and SciPy's
        # behaviour, not a reimplementation of it, is the normative answer.
        return np.float64(_scipy_logsumexp(values))
    # Numba unboxes its nopython float64 return value back to a plain Python
    # ``float``, not ``np.float64``; the original always returns ``np.float64``
    # (see its own ``test_returns_float64``), so re-wrap. This changes only
    # the Python type, not the underlying 64-bit pattern -- still bit-exact.
    return np.float64(_log_sum_exp_kernel_numba(values, a_max))


def softmax_from_log_numba(log_weights: FloatArray) -> FloatArray:
    """Numba-backed ``_softmax_from_log``, bit-identical to the Python original."""

    if not is_numba_available():
        raise RuntimeError(
            "compiled_kernels.softmax_from_log_numba requires numba, which is "
            f"not importable in this environment ({_NUMBA_IMPORT_ERROR!r})"
        )
    lse = log_sum_exp_numba(log_weights)
    return _softmax_kernel_numba(log_weights, lse)


# ---------------------------------------------------------------------------
# Span-adaptive per-round inner-update kernel.
#
# A literal port of the per-group body of ``span_adaptive.py``'s inner loop
# (``fit_groupwise_span_adaptive``, the ``for position, group in
# enumerate(active_groups)`` block) together with
# the *feasible* path of ``one_expert._kl_prox_update_with_candidate``
# (everything up to but not including the ``brentq`` bisection) and
# ``one_expert.kl_divergence``.
#
# What stays in Python (``span_adaptive.py``, wired in M3), per active group:
#   * the ``brentq`` projection when the unconstrained candidate is
#     infeasible ("binding") -- SciPy cannot run under ``nopython``;
#   * the vanishingly rare case where a softmax inside this kernel would hit
#     a non-finite maximum, which needs ``scipy.special.logsumexp``'s
#     delegated behaviour (see ``log_sum_exp_numba`` above) instead of the
#     compiled fast path.
# Both are reported back per group via a status code so the caller can redo
# exactly (and only) that group's work with the existing, unchanged Python
# functions -- groups are independent (see the loop being ported: nothing
# carries state from one ``group`` to the next within a round, and
# ``policies``/``projection_counts``/``update_counts`` are written back one
# index at a time), so it makes no difference which groups the kernel
# resolves directly and which fall back, and no difference what order the
# fallbacks run in.
# ---------------------------------------------------------------------------

#: Per-group kernel outcome, part of :func:`span_adaptive_inner_round_numba`'s
#: public contract (used by ``span_adaptive.py``'s ``kernel_backend="numba"``
#: wiring to decide, per active group, whether the compiled result is final
#: or a Python fallback is required).
STATUS_OK = 0  #: feasible; ``outputs``/``kl_values`` at this index are final.
STATUS_BINDING = 1  #: infeasible; Python must run the ``brentq`` projection.
STATUS_NEEDS_PYTHON = 2  #: a softmax hit a non-finite max; redo in Python.
_STATUS_OK = STATUS_OK
_STATUS_BINDING = STATUS_BINDING
_STATUS_NEEDS_PYTHON = STATUS_NEEDS_PYTHON

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _softmax_or_flag_numba(log_weights: FloatArray):
        """``_softmax_kernel_numba``, but reporting a non-finite max instead
        of running the (nopython-incompatible) SciPy delegation itself."""

        a_max = np.max(log_weights)
        if not np.isfinite(a_max):
            return np.zeros_like(log_weights), False
        lse = _log_sum_exp_kernel_numba(log_weights, a_max)
        return _softmax_kernel_numba(log_weights, lse), True

    @_njit(cache=True, fastmath=False)
    def _kl_divergence_numba(q: FloatArray, reference: FloatArray) -> np.float64:
        """Literal port of ``one_expert.kl_divergence``.

        Reproduces the ``0 log 0 = 0`` masking exactly: when some entries of
        ``q`` are not strictly positive, only the positive ones are summed,
        *in their original index order*, via a compacted temporary array --
        the same effect as NumPy's boolean-array indexing, whose result is
        already index-ordered, but written out explicitly since Numba's own
        boolean-index lowering is not asserted bit-identical to NumPy's here.
        """

        n = q.shape[0]
        all_positive = True
        for a in range(n):
            if q[a] <= 0.0:
                all_positive = False
                break
        if all_positive:
            terms = np.empty(n, dtype=np.float64)
            for a in range(n):
                terms[a] = q[a] * (np.log(q[a]) - np.log(reference[a]))
            return _pairwise_sum_numba(terms)
        count = 0
        for a in range(n):
            if q[a] > 0.0:
                count += 1
        terms = np.empty(count, dtype=np.float64)
        idx = 0
        for a in range(n):
            if q[a] > 0.0:
                terms[idx] = q[a] * (np.log(q[a]) - np.log(reference[a]))
                idx += 1
        return _pairwise_sum_numba(terms)

    @_njit(cache=True, fastmath=False)
    def _span_adaptive_inner_round_kernel_numba(
        policies_active: FloatArray,
        gamma_theta: FloatArray,
        theta_next: FloatArray,
        reward: FloatArray,
        reference_policy: FloatArray,
        log_reference: FloatArray,
        tau: float,
        maximum_kl: float,
        tiny: float,
        numerical_tolerance: float,
    ):
        n_active = policies_active.shape[0]
        m = policies_active.shape[1]
        anchors = np.empty((n_active, m), dtype=np.float64)
        outputs = np.empty((n_active, m), dtype=np.float64)
        kl_values = np.zeros(n_active, dtype=np.float64)
        status = np.zeros(n_active, dtype=np.int64)

        # loss = -reward (the self-play learner's ``-reward`` argument), formed
        # once since it does not vary across groups within a round.
        loss = np.empty(m, dtype=np.float64)
        for a in range(m):
            loss[a] = -reward[a]

        for idx in range(n_active):
            g_theta = gamma_theta[idx]
            if g_theta == 1.0:
                anchor = policies_active[idx].copy()
            else:
                log_pi = np.empty(m, dtype=np.float64)
                for a in range(m):
                    v = policies_active[idx, a]
                    if v < tiny:
                        v = tiny
                    log_pi[a] = np.log(v)
                combined = np.empty(m, dtype=np.float64)
                for a in range(m):
                    combined[a] = g_theta * log_pi[a] + (1.0 - g_theta) * log_reference[a]
                anchor, ok = _softmax_or_flag_numba(combined)
                if not ok:
                    status[idx] = _STATUS_NEEDS_PYTHON
                    continue
            anchors[idx] = anchor

            rate = theta_next[idx]
            log_q = np.empty(m, dtype=np.float64)
            for a in range(m):
                v = anchor[a]
                if v < tiny:
                    v = tiny
                log_q[a] = np.log(v)
            arg = np.empty(m, dtype=np.float64)
            for a in range(m):
                arg[a] = log_q[a] - rate * loss[a]
            unconstrained, ok2 = _softmax_or_flag_numba(arg)
            if not ok2:
                status[idx] = _STATUS_NEEDS_PYTHON
                continue

            # The outer span-adaptive loop always recomputes this exact
            # ``kl_divergence(unconstrained, reference)`` after the prox
            # update returns, regardless of which internal branch produced
            # ``unconstrained`` (span_adaptive.py's projection-count check);
            # computing it once here and reusing it is bit-identical to
            # recomputing it later, since it is a pure function of the same
            # two arrays.
            kl = _kl_divergence_numba(unconstrained, reference_policy)
            kl_values[idx] = kl
            if tau >= maximum_kl - numerical_tolerance:
                outputs[idx] = unconstrained
                status[idx] = _STATUS_OK
                continue
            if kl <= tau + numerical_tolerance:
                outputs[idx] = unconstrained
                status[idx] = _STATUS_OK
                continue
            status[idx] = _STATUS_BINDING

        return anchors, outputs, kl_values, status

else:  # pragma: no cover - exercised only without numba installed
    _softmax_or_flag_numba = None
    _kl_divergence_numba = None
    _span_adaptive_inner_round_kernel_numba = None


def span_adaptive_inner_round_numba(
    policies_active: FloatArray,
    gamma_theta: FloatArray,
    theta_next: FloatArray,
    reward: FloatArray,
    reference_policy: FloatArray,
    log_reference: FloatArray,
    tau: float,
    maximum_kl: float,
):
    """Compiled fast path for one round's per-group inner update.

    Parameters mirror the per-round quantities already held by
    ``span_adaptive.fit_groupwise_span_adaptive``:
    ``policies_active`` is ``policies[active_groups]`` (shape
    ``(n_active, m)``); ``gamma_theta``/``theta_next`` are the same-named
    per-active-group arrays; ``reward`` is the round's shared ``m``-vector;
    ``reference_policy``/``log_reference`` are the fit's fixed reference and
    its log; ``tau`` is the fit's fixed radius; ``maximum_kl`` is
    ``log(1 / min(reference_policy))``, invariant for the fit and so hoisted
    out of the per-group loop (recomputing it per call, as
    ``one_expert._kl_prox_update_with_candidate`` does, would give the same
    float64 value every time -- deterministic, not a source of drift).

    Returns ``(anchors, outputs, kl_values, status)``, all length
    ``n_active`` (or ``(n_active, m)`` for the first two): for a group with
    ``status == 0`` (feasible), ``outputs[i]`` is the final updated policy
    and ``kl_values[i]`` is ``kl_divergence(outputs[i], reference_policy)``,
    ready to use directly. For ``status == 1`` (binding) or ``status == 2``
    (a softmax hit a non-finite max), the caller must redo that group with
    the existing Python code; ``anchors[i]`` is valid and reusable only when
    ``status == 1`` -- for ``status == 2`` the anchor computation itself may
    be the one that was invalid, so it must be recomputed in Python too. See
    ``span_adaptive.py``'s ``kernel_backend="numba"`` path for the exact
    reconstruction.

    Raises :class:`RuntimeError` if numba is not importable, and
    :class:`ValueError` for the same invalid inputs
    ``one_expert._kl_prox_update_with_candidate`` would reject (checked here
    in plain Python, once per round, rather than per group in ``nopython``
    code, since ``tau`` and the validity of every rate are round-invariant
    checks with nothing group-specific about them).
    """

    if not is_numba_available():
        raise RuntimeError(
            "compiled_kernels.span_adaptive_inner_round_numba requires numba, "
            f"which is not importable in this environment ({_NUMBA_IMPORT_ERROR!r})"
        )
    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    if np.any(theta_next < 0.0) or not np.all(np.isfinite(theta_next)):
        raise ValueError("learning_rate must be finite and nonnegative")
    return _span_adaptive_inner_round_kernel_numba(
        policies_active,
        gamma_theta,
        theta_next,
        reward,
        reference_policy,
        log_reference,
        float(tau),
        float(maximum_kl),
        _TINY,
        _NUMERICAL_TOLERANCE,
    )


# ---------------------------------------------------------------------------
# Per-round orchestration kernels.
#
# Six kernels that literally port everything numeric in
# ``fit_groupwise_span_adaptive``'s round loop *except* the five NumPy
# dot/matmul/einsum calls the brief requires stay exact, executed in Python,
# never replicated here (``active_probabilities @ proposals``, ``np.dot(
# policy, reward)``, ``proposals @ reward``, the ``np.einsum(...)``, and
# ``proposals @ (reward * reward)``), plus ``_draw_comparison`` (RNG),
# snapshot copies, and ``rate_trace`` appends (also Python, unchanged). Every
# kernel below sits in one of the gaps between those calls; see the M1 design
# note for which lines of ``span_adaptive.py`` each one replaces. Reductions
# use ``_pairwise_sum_numba`` wherever the ported code called ``.sum()``;
# ``np.max``/``np.min`` are reproduced with plain scalar ``np.maximum``/
# ``np.minimum`` folds (order-independent, no pairwise-blocking hazard --
# unlike ``sum``, ``max``/``min`` are exactly associative, including under
# NaN propagation, so a sequential scan matches NumPy's own reduction
# regardless of grouping). Every assertion is reported as a boolean "ok" flag
# for the caller to turn into the identical ``raise`` in Python.
# ---------------------------------------------------------------------------

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _rate_numba(moment: FloatArray, offset: float, scale: float, cap: float) -> FloatArray:
        """Literal port of ``SpanAdaptiveSchedule.outer_rate``/``inner_rate``:
        ``np.minimum(cap, np.sqrt(scale / (2 * (offset + moment))))``,
        elementwise (no reduction, so this is trivially bit-identical to the
        vectorized NumPy original regardless of loop order)."""

        n = moment.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = min(cap, np.sqrt(scale / (2.0 * (offset + moment[i]))))
        return out

    @_njit(cache=True, fastmath=False)
    def _round_start_kernel_numba(
        active_groups: NDArray[np.int64],
        log_outer_mass: FloatArray,
        outer_moment: FloatArray,
        centered_moment: FloatArray,
        frozen: bool,
        frozen_outer: FloatArray,
        frozen_inner: FloatArray,
        L: float,
        tau_bar: float,
        outer_cap: float,
        inner_cap: float,
        outer_offset: float,
        inner_offset: float,
    ):
        """Port of ``span_adaptive.py``'s ``current_rates()`` + effective log
        mass (458-459) + the active-groups softmax (474's mixing weights)."""

        n_groups = log_outer_mass.shape[0]
        if frozen:
            eta = np.empty(n_groups, dtype=np.float64)
            theta = np.empty(n_groups, dtype=np.float64)
            for g in range(n_groups):
                eta[g] = frozen_outer[g]
                theta[g] = frozen_inner[g]
        else:
            eta = _rate_numba(outer_moment, outer_offset, L, outer_cap)
            theta = _rate_numba(centered_moment, inner_offset, tau_bar, inner_cap)
        effective_log_mass = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            effective_log_mass[g] = log_outer_mass[g] + np.log(eta[g])

        n_active = active_groups.shape[0]
        if n_active == 0:
            return eta, theta, effective_log_mass, np.empty(0, dtype=np.float64), True
        sub = np.empty(n_active, dtype=np.float64)
        for idx in range(n_active):
            sub[idx] = effective_log_mass[active_groups[idx]]
        active_probabilities, ok = _softmax_or_flag_numba(sub)
        return eta, theta, effective_log_mass, active_probabilities, ok

    @_njit(cache=True, fastmath=False)
    def _round_mix_finish_kernel_numba(
        policy_raw: FloatArray, reference_policy: FloatArray, tau: float
    ):
        """Port of 477-479: clip+normalize the deployed mixture, and its KL
        divergence against the reference, with the 480 assertion as a flag."""

        m = policy_raw.shape[0]
        clipped = np.empty(m, dtype=np.float64)
        for a in range(m):
            v = policy_raw[a]
            if v < 0.0:
                v = 0.0
            clipped[a] = v
        total = _pairwise_sum_numba(clipped)
        policy = np.empty(m, dtype=np.float64)
        for a in range(m):
            policy[a] = clipped[a] / total
        kl = _kl_divergence_numba(policy, reference_policy)
        ok = kl <= tau + 2e-8
        return policy, kl, ok

    @_njit(cache=True, fastmath=False)
    def _round_reward_kernel_numba(score: FloatArray, score_bound: float):
        """Port of 490 (``reward = -score / score_bound``) and the
        ``abs_reward``/``span`` reductions of 495 and 499 (their assertions,
        496/500, stay scalar Python -- nothing array-shaped left to compile
        once these three are returned)."""

        m = score.shape[0]
        reward = np.empty(m, dtype=np.float64)
        for a in range(m):
            reward[a] = -score[a] / score_bound
        abs_reward = abs(reward[0])
        max_r = reward[0]
        min_r = reward[0]
        for a in range(1, m):
            v = reward[a]
            abs_reward = np.maximum(abs_reward, abs(v))
            max_r = np.maximum(max_r, v)
            min_r = np.minimum(min_r, v)
        span = max_r - min_r
        return reward, abs_reward, span

    @_njit(cache=True, fastmath=False)
    def _round_centered_and_q_kernel_numba(
        proposals: FloatArray,
        reward: FloatArray,
        r: FloatArray,
        outer_moment: FloatArray,
        active_groups: NDArray[np.int64],
    ):
        """Port of 506 (``centered = reward[None, :] - r[:, None]``) and the
        ``q_next`` half of 509, both needing only ``r``/``reward``/
        ``outer_moment`` -- computed together since nothing else sits between
        the ``proposals @ reward`` dot that produces ``r`` and the mandatory
        ``np.einsum`` call that consumes ``centered``."""

        n_active = proposals.shape[0]
        m = proposals.shape[1]
        centered = np.empty((n_active, m), dtype=np.float64)
        q_next = np.empty(n_active, dtype=np.float64)
        for idx in range(n_active):
            ri = r[idx]
            for a in range(m):
                centered[idx, a] = reward[a] - ri
            q_next[idx] = outer_moment[active_groups[idx]] + ri * ri
        return centered, q_next

    @_njit(cache=True, fastmath=False)
    def _round_moments_finish_kernel_numba(
        c: FloatArray,
        v: FloatArray,
        centered_moment: FloatArray,
        raw_moment: FloatArray,
        active_groups: NDArray[np.int64],
        q_next: FloatArray,
        tolerance: float,
    ):
        """Port of the ``c_next``/``v_next`` half of 510-511 and the Q+C=V
        residual check of 512-515 (assertion as a flag)."""

        n_active = active_groups.shape[0]
        c_next = np.empty(n_active, dtype=np.float64)
        v_next = np.empty(n_active, dtype=np.float64)
        for idx in range(n_active):
            g = active_groups[idx]
            c_next[idx] = centered_moment[g] + c[idx]
            v_next[idx] = raw_moment[g] + v[idx]
        max_abs_resid = 0.0
        max_v_next = v_next[0]
        for idx in range(n_active):
            resid = q_next[idx] + c_next[idx] - v_next[idx]
            max_abs_resid = np.maximum(max_abs_resid, abs(resid))
            max_v_next = np.maximum(max_v_next, v_next[idx])
        bound = np.maximum(1.0, max_v_next)
        ok = max_abs_resid <= tolerance * bound
        return c_next, v_next, ok

    @_njit(cache=True, fastmath=False)
    def _round_rates_gamma_outer_kernel_numba(
        active_groups: NDArray[np.int64],
        eta: FloatArray,
        theta: FloatArray,
        q_next: FloatArray,
        c_next: FloatArray,
        r: FloatArray,
        frozen: bool,
        log_outer_mass: FloatArray,
        log_initial_mass: float,
        L: float,
        tau_bar: float,
        outer_cap: float,
        inner_cap: float,
        outer_offset: float,
        inner_offset: float,
    ):
        """Port of 517-533: next rates, both gammas, the Prod ``factors``
        assertion (as a flag), and the anchored outer-mass update, which is
        applied (mutating ``log_outer_mass`` in place) only when the factors
        are all feasible -- exactly as the Python original never reaches that
        line once it would raise."""

        n_active = active_groups.shape[0]
        eta_active = np.empty(n_active, dtype=np.float64)
        theta_active = np.empty(n_active, dtype=np.float64)
        for idx in range(n_active):
            g = active_groups[idx]
            eta_active[idx] = eta[g]
            theta_active[idx] = theta[g]
        if frozen:
            eta_next = eta_active.copy()
            theta_next = theta_active.copy()
        else:
            eta_next = _rate_numba(q_next, outer_offset, L, outer_cap)
            theta_next = _rate_numba(c_next, inner_offset, tau_bar, inner_cap)

        gamma_eta = np.empty(n_active, dtype=np.float64)
        gamma_theta = np.empty(n_active, dtype=np.float64)
        factors = np.empty(n_active, dtype=np.float64)
        ok = True
        for idx in range(n_active):
            gamma_eta[idx] = eta_next[idx] / eta_active[idx]
            gamma_theta[idx] = theta_next[idx] / theta_active[idx]
            factors[idx] = eta_active[idx] * r[idx]
            if factors[idx] <= -1.0:
                ok = False
        if ok:
            for idx in range(n_active):
                g = active_groups[idx]
                log_plus = log_outer_mass[g] + np.log1p(factors[idx])
                log_outer_mass[g] = (
                    (1.0 - gamma_eta[idx]) * log_initial_mass + gamma_eta[idx] * log_plus
                )
        return eta_active, theta_active, eta_next, theta_next, gamma_eta, gamma_theta, ok

    @_njit(cache=True, fastmath=False)
    def _round_finish_kernel_numba(
        active_groups: NDArray[np.int64],
        q_next: FloatArray,
        c_next: FloatArray,
        v_next: FloatArray,
        eta_next: FloatArray,
        theta_next: FloatArray,
        outer_moment: FloatArray,
        centered_moment: FloatArray,
        raw_moment: FloatArray,
        first_eta_below: NDArray[np.int64],
        first_theta_below: NDArray[np.int64],
        eta_min: FloatArray,
        theta_min: FloatArray,
        round_index: int,
        outer_cap: float,
        inner_cap: float,
    ) -> None:
        """Port of 638-646: moment write-backs, first-below-cap indices, and
        the running eta/theta minima -- all in-place mutations of the fit's
        preallocated state arrays."""

        n_active = active_groups.shape[0]
        for idx in range(n_active):
            g = active_groups[idx]
            outer_moment[g] = q_next[idx]
            centered_moment[g] = c_next[idx]
            raw_moment[g] = v_next[idx]
            if eta_next[idx] < outer_cap and first_eta_below[g] < 0:
                first_eta_below[g] = round_index
            if theta_next[idx] < inner_cap and first_theta_below[g] < 0:
                first_theta_below[g] = round_index
            eta_min[g] = np.minimum(eta_min[g], eta_next[idx])
            theta_min[g] = np.minimum(theta_min[g], theta_next[idx])

else:  # pragma: no cover - exercised only without numba installed
    _rate_numba = None
    _round_start_kernel_numba = None
    _round_mix_finish_kernel_numba = None
    _round_reward_kernel_numba = None
    _round_centered_and_q_kernel_numba = None
    _round_moments_finish_kernel_numba = None
    _round_rates_gamma_outer_kernel_numba = None
    _round_finish_kernel_numba = None


def _require_numba(name: str) -> None:
    if not is_numba_available():
        raise RuntimeError(
            f"compiled_kernels.{name} requires numba, which is not importable "
            f"in this environment ({_NUMBA_IMPORT_ERROR!r})"
        )


def round_start_numba(
    active_groups: NDArray[np.int64],
    log_outer_mass: FloatArray,
    outer_moment: FloatArray,
    centered_moment: FloatArray,
    frozen: bool,
    frozen_outer: FloatArray,
    frozen_inner: FloatArray,
    L: float,
    tau_bar: float,
    outer_cap: float,
    inner_cap: float,
    outer_offset: float,
    inner_offset: float,
):
    """Public wrapper for :func:`_round_start_kernel_numba` (see there)."""

    _require_numba("round_start_numba")
    return _round_start_kernel_numba(
        active_groups, log_outer_mass, outer_moment, centered_moment, frozen,
        frozen_outer, frozen_inner, float(L), float(tau_bar), float(outer_cap),
        float(inner_cap), float(outer_offset), float(inner_offset),
    )


def round_mix_finish_numba(policy_raw: FloatArray, reference_policy: FloatArray, tau: float):
    """Public wrapper for :func:`_round_mix_finish_kernel_numba` (see there)."""

    _require_numba("round_mix_finish_numba")
    return _round_mix_finish_kernel_numba(policy_raw, reference_policy, float(tau))


def round_reward_numba(score: FloatArray, score_bound: float):
    """Public wrapper for :func:`_round_reward_kernel_numba` (see there)."""

    _require_numba("round_reward_numba")
    return _round_reward_kernel_numba(score, float(score_bound))


def round_centered_and_q_numba(
    proposals: FloatArray, reward: FloatArray, r: FloatArray,
    outer_moment: FloatArray, active_groups: NDArray[np.int64],
):
    """Public wrapper for :func:`_round_centered_and_q_kernel_numba` (see there)."""

    _require_numba("round_centered_and_q_numba")
    return _round_centered_and_q_kernel_numba(proposals, reward, r, outer_moment, active_groups)


def round_moments_finish_numba(
    c: FloatArray, v: FloatArray, centered_moment: FloatArray, raw_moment: FloatArray,
    active_groups: NDArray[np.int64], q_next: FloatArray, tolerance: float = 1e-9,
):
    """Public wrapper for :func:`_round_moments_finish_kernel_numba` (see there)."""

    _require_numba("round_moments_finish_numba")
    return _round_moments_finish_kernel_numba(
        c, v, centered_moment, raw_moment, active_groups, q_next, float(tolerance)
    )


def round_rates_gamma_outer_numba(
    active_groups: NDArray[np.int64], eta: FloatArray, theta: FloatArray,
    q_next: FloatArray, c_next: FloatArray, r: FloatArray, frozen: bool,
    log_outer_mass: FloatArray, log_initial_mass: float, L: float, tau_bar: float,
    outer_cap: float, inner_cap: float, outer_offset: float, inner_offset: float,
):
    """Public wrapper for :func:`_round_rates_gamma_outer_kernel_numba` (see there)."""

    _require_numba("round_rates_gamma_outer_numba")
    return _round_rates_gamma_outer_kernel_numba(
        active_groups, eta, theta, q_next, c_next, r, frozen, log_outer_mass,
        float(log_initial_mass), float(L), float(tau_bar), float(outer_cap),
        float(inner_cap), float(outer_offset), float(inner_offset),
    )


def round_finish_numba(
    active_groups: NDArray[np.int64], q_next: FloatArray, c_next: FloatArray,
    v_next: FloatArray, eta_next: FloatArray, theta_next: FloatArray,
    outer_moment: FloatArray, centered_moment: FloatArray, raw_moment: FloatArray,
    first_eta_below: NDArray[np.int64], first_theta_below: NDArray[np.int64],
    eta_min: FloatArray, theta_min: FloatArray, round_index: int,
    outer_cap: float, inner_cap: float,
) -> None:
    """Public wrapper for :func:`_round_finish_kernel_numba` (see there)."""

    _require_numba("round_finish_numba")
    _round_finish_kernel_numba(
        active_groups, q_next, c_next, v_next, eta_next, theta_next, outer_moment,
        centered_moment, raw_moment, first_eta_below, first_theta_below, eta_min,
        theta_min, int(round_index), float(outer_cap), float(inner_cap),
    )


# ---------------------------------------------------------------------------
# Compiled KL-projection ("binding") path -- a literal port of the bracket
# search, ``scipy.optimize.brentq`` call, and post-solve feasibility checks
# of ``one_expert._kl_prox_update_with_candidate`` (
# picking up exactly where the feasible-path kernel above leaves off: this
# module never re-derives ``log_q``/``unconstrained``, callers already have
# them).
#
# That helper calls ``brentq(residual, lower, upper, xtol=1e-12,
# rtol=1e-12)`` with ``maxiter`` left at SciPy's default (100 in the
# pinned environment). These three literal constants
# are duplicated here (not imported -- ``one_expert.py`` does not name them)
# so a change to the call site is a visible diff in both places, not a silent
# drift.
#
# The scalar root-finding loop below is a direct scalarization of
# ``reporting/metrics.py``'s ``_vectorized_brentq``, which reproduces SciPy's
# ``Zeros/brentq.c`` iteration. Every branch and update formula below is that function's per-row
# logic with the array masks turned into scalar ``if``s, in the same order:
#   * evaluate f(xa) then f(xb) (``brentq.c`` evaluates the left endpoint
#     first);
#   * an exact zero at either endpoint returns immediately;
#   * a non-bracketing sign (same sign at both endpoints, excluding the
#     exact-zero cases just handled) is reported rather than silently
#     iterated on -- real SciPy raises ``ValueError`` for this, which is
#     unreachable in this application by construction (the caller only
#     brackets a root after confirming ``residual(0) > 0`` and
#     ``residual(upper) <= 0`` in the doubling search below), but is still
#     reported as a status rather than assumed impossible;
#   * the same swap/interpolate/bisect update sequence as the vectorized
#     version, run for at most ``maxiter`` passes;
#   * a NaN residual (only possible here if a softmax inside ``candidate()``
#     hits a non-finite maximum -- the one branch ``log_sum_exp`` delegates
#     to SciPy for, which cannot run under ``nopython``) is reported as a
#     status instead of silently propagating through comparisons that treat
#     NaN inconsistently.
# ---------------------------------------------------------------------------

#: Literal duplication of ``_kl_prox_update_with_candidate``'s ``brentq`` call
#: arguments and SciPy's default ``maxiter`` (see module note above).
_BRENTQ_XTOL = 1e-12
_BRENTQ_RTOL = 1e-12
_BRENTQ_MAXITER = 100

#: Outcome of the generic scalar ``brentq`` port.
_BRENTQ_OK = 0  #: converged; ``root`` is final.
_BRENTQ_MAXITER_HIT = 1  #: ran ``maxiter`` passes without meeting tolerance.
_BRENTQ_INVALID_BRACKET = 2  #: endpoints did not bracket a sign change.
_BRENTQ_NONFINITE = 3  #: ``f`` reported a non-finite (NaN-flagged) residual.

#: Outcome of :func:`kl_prox_binding_numba`, part of its public contract.
BINDING_STATUS_OK = 0  #: ``result`` is the final, feasible projected policy.
BINDING_STATUS_NEEDS_PYTHON = 1  #: caller must redo with the unmodified
#: Python ``_kl_prox_update_with_candidate`` (maxiter exceeded, an invalid
#: bracket, or a softmax inside ``candidate()`` hit a non-finite maximum).
BINDING_STATUS_INFEASIBLE = 2  #: converged, but even the ``nextafter`` latch
#: stayed outside ``tau + 1e-8`` -- the caller must raise the identical
#: ``AssertionError`` from ``_kl_prox_update_with_candidate`` rather than use ``result``.

#: Module-level, test-visible counter of successful (``BINDING_STATUS_OK``)
#: compiled-binding resolutions. Not surfaced in any diagnostics/output --
#: existing solely so a test can assert the compiled path was actually
#: exercised (see ``tests/test_compiled_span_adaptive.py``), the same way one
#: would check a coverage flag.
_binding_kernel_hits = 0


def reset_binding_kernel_hits() -> None:
    """Zero the test-visible compiled-binding-resolution counter."""

    global _binding_kernel_hits
    _binding_kernel_hits = 0


def get_binding_kernel_hits() -> int:
    """Return how many groups the compiled binding path has resolved
    (``BINDING_STATUS_OK``) since the last :func:`reset_binding_kernel_hits`."""

    return _binding_kernel_hits


if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _kl_prox_candidate_numba(multiplier, log_q, rate, loss, log_ref):
        """Literal port of the ``candidate`` closure in ``_kl_prox_update_with_candidate``.

        Returns ``(values, ok)``; ``ok`` is ``False`` exactly when the
        underlying softmax hit a non-finite maximum (see
        ``_softmax_or_flag_numba``), in which case ``values`` is not the
        real candidate and must not be used.
        """

        m = log_q.shape[0]
        inverse_scale = 1.0 / (1.0 + multiplier)
        candidate_log = np.empty(m, dtype=np.float64)
        for a in range(m):
            candidate_log[a] = (
                inverse_scale * (log_q[a] - rate * loss[a])
                + (1.0 - inverse_scale) * log_ref[a]
            )
        return _softmax_or_flag_numba(candidate_log)

    @_njit(cache=True, fastmath=False)
    def _kl_prox_residual_numba(multiplier, args):
        """Literal port of the ``residual`` closure in ``_kl_prox_update_with_candidate``.

        ``args`` is ``(log_q, rate, loss, log_ref, reference, tau)``. Returns
        ``NaN`` in place of raising when the candidate softmax is non-finite,
        so the generic brentq port below can detect it via ``_BRENTQ_NONFINITE``
        without a bespoke residual signature per caller.
        """

        log_q, rate, loss, log_ref, reference, tau = args
        cand, ok = _kl_prox_candidate_numba(multiplier, log_q, rate, loss, log_ref)
        if not ok:
            return np.nan
        return _kl_divergence_numba(cand, reference) - tau

    @_njit(cache=False, fastmath=False)
    def _brentq_numba(f, xa, xb, xtol, rtol, maxiter, args):
        """Scalar port of SciPy's ``brentq`` (``Zeros/brentq.c``).

        ``cache=False``: ``f`` is passed as a first-class njit dispatcher
        value, which Numba cannot persist to an on-disk cache (it warns
        "cannot cache ... dynamic globals" under ``cache=True``); this
        changes only whether the compiled code is reused across processes,
        not what it computes -- still recompiled once per process either way.

        ``f`` must be an njit-compiled callable of ``(x, args) -> float64``.
        Returns ``(root, status, iterations)`` -- see the module-level
        ``_BRENTQ_*`` constants for ``status``, and SciPy's ``RootResults.
        iterations`` (``full_output=True``) for the quantity ``iterations``
        is checked against in the differential test.

        This is the scalarization described in the section banner above: a
        direct translation of ``reporting/metrics.py``'s ``_vectorized_
        brentq``, not an independent reimplementation.
        """

        xpre = xa
        xcur = xb
        fpre = f(xpre, args)
        fcur = f(xcur, args)
        if np.isnan(fpre) or np.isnan(fcur):
            return xcur, _BRENTQ_NONFINITE, 0
        if fpre == 0.0:
            return xpre, _BRENTQ_OK, 0
        if fcur == 0.0:
            return xcur, _BRENTQ_OK, 0
        if (fpre < 0.0) == (fcur < 0.0):
            return xcur, _BRENTQ_INVALID_BRACKET, 0

        xblk = xpre
        fblk = fpre
        spre = xcur - xpre
        scur = spre
        iterations = 0

        for _ in range(maxiter):
            iterations += 1
            if fpre != 0.0 and fcur != 0.0 and (fpre < 0.0) != (fcur < 0.0):
                xblk = xpre
                fblk = fpre
                step = xcur - xpre
                spre = step
                scur = step
            if abs(fblk) < abs(fcur):
                xpre = xcur
                fpre = fcur
                xcur = xblk
                fcur = fblk
                xblk = xpre
                fblk = fpre

            delta = 0.5 * (xtol + rtol * abs(xcur))
            sbis = 0.5 * (xblk - xcur)
            if fcur == 0.0 or abs(sbis) < delta:
                return xcur, _BRENTQ_OK, iterations

            if abs(spre) > delta and abs(fcur) < abs(fpre):
                if xpre == xblk:
                    stry = -fcur * (xcur - xpre) / (fcur - fpre)
                else:
                    dpre = (fpre - fcur) / (xpre - xcur)
                    dblk = (fblk - fcur) / (xblk - xcur)
                    stry = (
                        -fcur
                        * (fblk * dblk - fpre * dpre)
                        / (dblk * dpre * (fblk - fpre))
                    )
                if np.isfinite(stry) and 2.0 * abs(stry) < min(
                    abs(spre), 3.0 * abs(sbis) - delta
                ):
                    spre = scur
                    scur = stry
                else:
                    spre = sbis
                    scur = sbis
            else:
                spre = sbis
                scur = sbis

            xpre = xcur
            fpre = fcur
            if abs(scur) > delta:
                xcur = xcur + scur
            else:
                xcur = xcur + (delta if sbis > 0.0 else -delta)

            fcur = f(xcur, args)
            if np.isnan(fcur):
                return xcur, _BRENTQ_NONFINITE, iterations

        return xcur, _BRENTQ_MAXITER_HIT, iterations

    @_njit(cache=False, fastmath=False)
    def _kl_prox_binding_numba(log_q, rate, loss, log_ref, reference, tau):
        """Literal port of ``_kl_prox_update_with_candidate``'s bracket search, the
        ``brentq`` solve, and the post-solve feasibility latch, given the
        already-computed ``log_q``/``loss`` an infeasible unconstrained
        candidate the caller has already found binding.

        Returns ``(result, status)``; ``result`` is meaningful only when
        ``status == BINDING_STATUS_OK``.
        """

        lower = 0.0  # infeasible
        upper = 1.0  # initial upper bracket
        while True:
            cand, ok = _kl_prox_candidate_numba(upper, log_q, rate, loss, log_ref)
            if not ok:
                return reference, BINDING_STATUS_NEEDS_PYTHON
            if _kl_divergence_numba(cand, reference) <= tau:
                break
            upper *= 2.0
            if upper > 1e16:
                # Radius indistinguishable from zero.
                return reference.copy(), BINDING_STATUS_OK

        args = (log_q, rate, loss, log_ref, reference, tau)
        root, status, _iterations = _brentq_numba(
            _kl_prox_residual_numba, lower, upper, _BRENTQ_XTOL, _BRENTQ_RTOL,
            _BRENTQ_MAXITER, args,
        )
        if status != _BRENTQ_OK:
            # Maxiter exceeded, an invalid bracket, or a non-finite softmax
            # mid-solve: none of these may be silently handled (brief M2).
            # Punt the whole group back to the unmodified Python function,
            # which will converge, raise SciPy's own error, or raise the
            # same AssertionError this function raises below -- whichever
            # the real brentq call would do.
            return reference, BINDING_STATUS_NEEDS_PYTHON

        result, ok = _kl_prox_candidate_numba(root, log_q, rate, loss, log_ref)
        if not ok:
            return reference, BINDING_STATUS_NEEDS_PYTHON
        if _kl_divergence_numba(result, reference) > tau:
            # One representable step toward feasible.
            next_root = np.nextafter(root, np.inf)
            result, ok = _kl_prox_candidate_numba(next_root, log_q, rate, loss, log_ref)
            if not ok:
                return reference, BINDING_STATUS_NEEDS_PYTHON
        if _kl_divergence_numba(result, reference) > tau + 1e-8:
            # Match the Python helper's final feasibility check.
            return result, BINDING_STATUS_INFEASIBLE
        return result, BINDING_STATUS_OK

else:  # pragma: no cover - exercised only without numba installed
    _kl_prox_candidate_numba = None
    _kl_prox_residual_numba = None
    _brentq_numba = None
    _kl_prox_binding_numba = None


def kl_prox_binding_numba(
    anchor: FloatArray,
    rate: float,
    loss: FloatArray,
    reference_policy: FloatArray,
    log_reference: FloatArray,
    tau: float,
):
    """Compiled fast path for a single binding group's KL projection.

    Callers already know (from :func:`span_adaptive_inner_round_numba`
    reporting ``STATUS_BINDING`` for this group) that ``tau`` exceeds
    ``_NUMERICAL_TOLERANCE``, ``rate`` is finite and nonnegative, and the
    unconstrained candidate ``softmax(log(max(anchor, tiny)) - rate * loss)``
    is infeasible -- exactly the state ``one_expert._kl_prox_update_with_
    candidate`` is in before its bracket search. This function performs
    the remaining search, projection, and feasibility check.

    Returns ``(result, status)`` -- see ``BINDING_STATUS_OK``/
    ``BINDING_STATUS_NEEDS_PYTHON``/``BINDING_STATUS_INFEASIBLE``. Raises
    :class:`RuntimeError` if numba is not importable.
    """

    if not is_numba_available():
        raise RuntimeError(
            "compiled_kernels.kl_prox_binding_numba requires numba, which is "
            f"not importable in this environment ({_NUMBA_IMPORT_ERROR!r})"
        )
    log_q = np.log(np.maximum(anchor, _TINY))
    result, status = _kl_prox_binding_numba(
        log_q, float(rate), loss, log_reference, reference_policy, float(tau)
    )
    if status == BINDING_STATUS_OK:
        global _binding_kernel_hits
        _binding_kernel_hits += 1
    return result, status
