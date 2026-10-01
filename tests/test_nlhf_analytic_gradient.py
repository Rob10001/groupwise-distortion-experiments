"""Independent verification of the opt-in analytic NLHF gradient/Jacobian.

Checks:
  (a) analytic derivatives vs central finite differences with Richardson
      extrapolation, AND vs complex-step on an independently written,
      complex-analytic reimplementation of the interior-point math (the
      production `_smoothed_exploitability`/`log_sum_exp` path is not
      complex-step-safe: it branches on `np.max`/`==`/`np.where`);
  (b) a hand-derived, independently coded m=2 closed-form check;
  (c) the historical multi-start solver with the analytic option on, over
      Coffee-like and synthetic cases: acceptance, feasibility, and agreement
      with the finite-difference path; a representative default-fit check;
  (d) the default stays finite_difference and diagnostics record the option.

Analytic derivative correctness does not guarantee that one SLSQP call reaches
an accepted equilibrium. The current single-reference-start solver contract is
tested separately in ``test_nlhf_reference_start.py``.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from empirical_testing.algorithms.nlhf import (
    EXPLOITABILITY_TOLERANCE,
    _kl_constraint_gradient,
    _objective_gradient,
    _resolve_gradient_mode,
    _safe,
    fit_pooled_nlhf,
    preference_margins,
    solve_constrained_nash,
)
from empirical_testing.algorithms.one_expert import kl_divergence


def _uniform(m: int) -> np.ndarray:
    return np.full(m, 1.0 / m)


def _random_skew_symmetric(rng: np.random.Generator, m: int) -> np.ndarray:
    raw = rng.normal(size=(m, m))
    return 0.5 * (raw - raw.T)


def _coffee_like_margins() -> np.ndarray:
    """A structured, non-random m=4 preference matrix in Coffee's shape.

    Modelled on the qualitative pattern real Coffee margins show (one clearly
    dominant alternative, one clearly weak one, a near-tie in the middle) --
    not the real dataset, which this test file does not load.
    """

    matrix = np.array(
        [
            [0.5, 0.65, 0.80, 0.55],
            [0.35, 0.5, 0.58, 0.42],
            [0.20, 0.42, 0.5, 0.30],
            [0.45, 0.58, 0.70, 0.5],
        ]
    )
    return preference_margins(matrix)


# ---------------------------------------------------------------------------
# (a) Central finite differences + Richardson extrapolation, on the actual
#     production objective/constraint closures (bounds-respecting).
# ---------------------------------------------------------------------------

def _objective_value(free, margins, log_reference, tau, m):
    from empirical_testing.algorithms.nlhf import _smoothed_exploitability

    return _smoothed_exploitability(
        margins @ free[:m], log_reference, float(np.exp(free[m])), tau
    )


def _kl_constraint_value(free, reference, tau, m):
    return float(tau - kl_divergence(_safe(free[:m]), reference))


def _fd_gradient(func, x: np.ndarray, h: float) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (func(xp) - func(xm)) / (2.0 * h)
    return grad


def _richardson_gradient(func, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """O(h^4) central-difference gradient via Richardson extrapolation.

    Combines central differences at step h and h/2: for a centered
    second-order scheme the leading error term is O(h^2), so
    (4*g(h/2) - g(h)) / 3 cancels it, leaving O(h^4).
    """

    g_h = _fd_gradient(func, x, h)
    g_h2 = _fd_gradient(func, x, h / 2.0)
    return (4.0 * g_h2 - g_h) / 3.0


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b)) / denom


# ---------------------------------------------------------------------------
# Independent, complex-step-safe reimplementation for interior points.
# Deliberately NOT calling `numerics.log_sum_exp` (its max-shift trick uses
# `np.max`/`==`/`np.where`, none of which are complex-differentiable) or any
# `nlhf` helper -- this is a second, independent expression of the same math.
# ---------------------------------------------------------------------------

def _naive_objective(free, margins, log_reference, tau, m):
    values = margins @ free[:m]
    scale = np.exp(free[m])
    w = values / scale + log_reference
    lse = np.log(np.sum(np.exp(w)))
    return scale * tau + scale * lse


def _naive_kl_constraint_interior(free, reference, tau, m):
    """Same as the production KL constraint, valid only where `_safe` is the
    identity normalization (all of `free[:m]` strictly positive, no clip) --
    exactly the interior points this check is used on."""

    raw = free[:m]
    total = np.sum(raw)
    p = raw / total
    kl = np.sum(p * (np.log(p) - np.log(reference)))
    return tau - kl


def _complex_step_gradient(func, x: np.ndarray, h: float = 1e-30) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.astype(np.complex128).copy()
        xp[i] += 1j * h
        grad[i] = func(xp).imag / h
    return grad


class ObjectiveGradientVerificationTests(unittest.TestCase):
    """M3(a): objective jac vs Richardson FD and vs complex-step."""

    def test_matches_richardson_finite_differences_interior_and_near_boundary(self) -> None:
        rng = np.random.default_rng(20260913)
        m = 4
        log_reference = np.log(_uniform(m))
        tau = 0.5
        margins = _random_skew_symmetric(rng, m)
        errors = []
        for trial in range(40):
            pi = rng.dirichlet(np.full(m, 3.0))
            # Half the trials sit near the KL-ball boundary (tau just above
            # the drawn point's own KL to the uniform reference); the other
            # half are ordinary interior points.
            local_tau = tau
            if trial % 2 == 0:
                local_tau = float(kl_divergence(pi, _uniform(m))) + 1e-6
            t = np.log(rng.uniform(1e-3, 2.0))
            free = np.concatenate([pi, [t]])

            analytic = _objective_gradient(free, margins, log_reference, local_tau, m)
            richardson = _richardson_gradient(
                lambda z: _objective_value(z, margins, log_reference, local_tau, m), free
            )
            err = _relative_error(analytic, richardson)
            errors.append(err)
            self.assertLessEqual(err, 1e-6, f"trial {trial}: rel err {err:.3e}")
        print(f"[D-504 M3a] objective vs Richardson-FD: max rel err "
              f"{max(errors):.3e}, median {sorted(errors)[len(errors) // 2]:.3e}")

    def test_matches_complex_step_on_independent_reimplementation(self) -> None:
        rng = np.random.default_rng(7)
        m = 5
        log_reference = np.log(_uniform(m))
        tau = 0.7
        margins = _random_skew_symmetric(rng, m)
        errors = []
        for _ in range(25):
            pi = rng.dirichlet(np.full(m, 4.0))  # strictly positive, no clipping
            t = np.log(rng.uniform(1e-3, 2.0))
            free = np.concatenate([pi, [t]])

            analytic = _objective_gradient(free, margins, log_reference, tau, m)
            complex_step = _complex_step_gradient(
                lambda z: _naive_objective(z, margins, log_reference, tau, m), free
            )
            err = _relative_error(analytic, complex_step)
            errors.append(err)
            self.assertLessEqual(err, 1e-6, f"rel err {err:.3e}")
        print(f"[D-504 M3a] objective vs complex-step (independent reimpl): "
              f"max rel err {max(errors):.3e}")


class KLConstraintGradientVerificationTests(unittest.TestCase):
    """M3(a): KL-constraint jac vs Richardson FD and vs complex-step."""

    def test_matches_richardson_finite_differences_interior_and_near_boundary(self) -> None:
        rng = np.random.default_rng(11)
        m = 4
        reference = _uniform(m)
        tau = 0.5
        errors = []
        for trial in range(40):
            # Un-normalized z[:m] (SLSQP iterates need not sum to 1), always
            # strictly positive so `_safe`'s clip is a no-op (the interior
            # regime the solver actually explores under its own bounds).
            raw = rng.uniform(0.05, 1.0, size=m)
            t = 0.0  # unused by the KL constraint
            free = np.concatenate([raw, [t]])

            analytic = _kl_constraint_gradient(free, reference, m)
            richardson = _richardson_gradient(
                lambda z: _kl_constraint_value(z, reference, tau, m), free
            )
            err = _relative_error(analytic, richardson)
            errors.append(err)
            self.assertLessEqual(err, 1e-6, f"trial {trial}: rel err {err:.3e}")
        print(f"[D-504 M3a] KL constraint vs Richardson-FD: max rel err "
              f"{max(errors):.3e}, median {sorted(errors)[len(errors) // 2]:.3e}")

    def test_matches_complex_step_on_independent_reimplementation(self) -> None:
        rng = np.random.default_rng(23)
        m = 4
        reference = _uniform(m)
        tau = 0.5
        errors = []
        for _ in range(25):
            raw = rng.uniform(0.1, 1.0, size=m)
            free = np.concatenate([raw, [0.0]])
            analytic = _kl_constraint_gradient(free, reference, m)
            complex_step = _complex_step_gradient(
                lambda z: _naive_kl_constraint_interior(z, reference, tau, m), free
            )
            err = _relative_error(analytic, complex_step)
            errors.append(err)
            self.assertLessEqual(err, 1e-6, f"rel err {err:.3e}")
        print(f"[D-504 M3a] KL constraint vs complex-step (independent reimpl): "
              f"max rel err {max(errors):.3e}")


class HandDerivedM2Tests(unittest.TestCase):
    """M3(b): independent, from-scratch closed-form check on a tiny m=2 game."""

    def test_objective_gradient_m2_closed_form(self) -> None:
        # A[0,1] = a = 0.6 (skew-symmetric, m=2 has one free parameter).
        a = 0.6
        margins = np.array([[0.0, a], [-a, 0.0]])
        r0 = 0.3
        reference = np.array([r0, 1.0 - r0])
        log_reference = np.log(reference)
        tau = 0.4
        p0 = 0.55
        t = -0.2
        free = np.array([p0, 1.0 - p0, t])
        m = 2

        # Hand algebra, independent of `margins @ free[:m]`:
        #   values = [a*(1-p0), -a*p0]
        v0 = a * (1.0 - p0)
        v1 = -a * p0
        s = np.exp(t)
        w0 = v0 / s + np.log(r0)
        w1 = v1 / s + np.log(1.0 - r0)
        # softmax by hand (no shared helper)
        big = max(w0, w1)
        e0, e1 = np.exp(w0 - big), np.exp(w1 - big)
        z = e0 + e1
        c0, c1 = e0 / z, e1 / z  # challenger = [c0, c1]
        obj_value = s * tau + s * (np.log(z) + big)

        # d obj/d free[:m] = -(margins @ challenger); margins @ [c0,c1] = [a*c1, -a*c0]
        expected_grad_pi = np.array([-(a * c1), -(-a * c0)])
        # d obj/dt = obj_value - challenger @ values
        expected_grad_t = obj_value - (c0 * v0 + c1 * v1)
        expected = np.concatenate([expected_grad_pi, [expected_grad_t]])

        actual = _objective_gradient(free, margins, log_reference, tau, m)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_kl_constraint_gradient_m2_closed_form(self) -> None:
        r0 = 0.3
        reference = np.array([r0, 1.0 - r0])
        # Un-normalized iterate (S != 1) to exercise the general formula.
        z0, z1 = 0.4, 0.5
        free = np.array([z0, z1, 0.0])
        m = 2

        s_total = z0 + z1
        p0, p1 = z0 / s_total, z1 / s_total
        kl = p0 * (np.log(p0) - np.log(r0)) + p1 * (np.log(p1) - np.log(1.0 - r0))
        # d KL/d z_k = (log(p_k/ref_k) - KL) / S  =>  d constraint/d z_k = (KL - log(p_k/ref_k))/S
        expected_g0 = (kl - (np.log(p0) - np.log(r0))) / s_total
        expected_g1 = (kl - (np.log(p1) - np.log(1.0 - r0))) / s_total
        expected = np.array([expected_g0, expected_g1, 0.0])

        actual = _kl_constraint_gradient(free, reference, m)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class OrdinaryCaseAgreementTests(unittest.TestCase):
    """Compare derivatives under the original multi-start agreement protocol."""

    def _check_pair(self, margins, reference, tau, seed):
        policy_fd, gap_fd, diag_fd = solve_constrained_nash(
            margins, reference, tau, seed=seed, gradient="finite_difference",
            selection="restart_legacy",
        )
        policy_an, gap_an, diag_an = solve_constrained_nash(
            margins, reference, tau, seed=seed, gradient="analytic",
            selection="restart_legacy",
        )
        self.assertEqual(diag_fd["gradient_mode"], "finite_difference")
        self.assertEqual(diag_an["gradient_mode"], "analytic")
        self.assertEqual(diag_fd["selection"], "restart_legacy")
        self.assertEqual(diag_an["selection"], "restart_legacy")
        for policy, gap in ((policy_fd, gap_fd), (policy_an, gap_an)):
            self.assertLessEqual(gap, EXPLOITABILITY_TOLERANCE)
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=9)
            self.assertTrue((policy >= -1e-9).all())
            self.assertLessEqual(kl_divergence(policy, reference), tau + 1e-9)
        return float(np.max(np.abs(policy_fd - policy_an)))

    def test_coffee_like_tau_half(self) -> None:
        margins = _coffee_like_margins()
        reference = _uniform(4)
        diffs = [self._check_pair(margins, reference, 0.5, seed) for seed in range(8)]
        print(f"[D-504 M3c] Coffee-like tau=0.5: policy Linf diffs "
              f"min={min(diffs):.2e} median={sorted(diffs)[len(diffs)//2]:.2e} "
              f"max={max(diffs):.2e}")
        self.assertLessEqual(max(diffs), 1e-6)

    def test_synthetic_random_games(self) -> None:
        diffs = []
        for m in (2, 3, 4, 6):
            rng = np.random.default_rng(1000 + m)
            reference = _uniform(m)
            for seed in range(6):
                margins = _random_skew_symmetric(rng, m)
                diffs.append(self._check_pair(margins, reference, 0.5, seed))
        print(f"[D-504 M3c] synthetic tau=0.5: policy Linf diffs "
              f"min={min(diffs):.2e} median={sorted(diffs)[len(diffs)//2]:.2e} "
              f"max={max(diffs):.2e}")
        self.assertLessEqual(max(diffs), 1e-6)

    def test_fit_pooled_nlhf_with_analytic_option(self) -> None:
        rng = np.random.default_rng(99)
        m = 4
        edges = [(a, b) for a in range(m) for b in range(a + 1, m)]
        index = rng.integers(0, len(edges), size=3000)
        pairs = np.array([edges[i] for i in index], dtype=np.int64)
        truth = np.array(
            [[.5, .7, .8, .3], [.3, .5, .6, .2], [.2, .4, .5, .1], [.7, .8, .9, .5]]
        )
        p_left = truth[pairs[:, 0], pairs[:, 1]]
        signs = np.where(rng.random(3000) < p_left, 1.0, -1.0)
        reference = _uniform(m)

        fitted_fd = fit_pooled_nlhf(pairs, signs, m, reference, 0.5, gradient="finite_difference")
        fitted_an = fit_pooled_nlhf(pairs, signs, m, reference, 0.5, gradient="analytic")
        self.assertEqual(fitted_fd.diagnostics["gradient_mode"], "finite_difference")
        self.assertEqual(fitted_an.diagnostics["gradient_mode"], "analytic")
        self.assertLessEqual(fitted_fd.exploitability, EXPLOITABILITY_TOLERANCE)
        self.assertLessEqual(fitted_an.exploitability, EXPLOITABILITY_TOLERANCE)
        self.assertTrue(np.allclose(fitted_fd.policy, fitted_an.policy, atol=1e-6))


class DefaultAndEnvVarTests(unittest.TestCase):
    """M3(d): default is finite_difference; env var and diagnostics work."""

    def test_default_is_finite_difference(self) -> None:
        self.assertEqual(_resolve_gradient_mode(None), "finite_difference")

    def test_explicit_argument_takes_precedence_over_env(self) -> None:
        old = os.environ.get("GWD_NLHF_GRADIENT")
        try:
            os.environ["GWD_NLHF_GRADIENT"] = "analytic"
            self.assertEqual(_resolve_gradient_mode("finite_difference"), "finite_difference")
            self.assertEqual(_resolve_gradient_mode(None), "analytic")
        finally:
            if old is None:
                os.environ.pop("GWD_NLHF_GRADIENT", None)
            else:
                os.environ["GWD_NLHF_GRADIENT"] = old

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_gradient_mode("bogus")

    def test_default_fit_records_finite_difference_in_diagnostics(self) -> None:
        rng = np.random.default_rng(3)
        m = 3
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)
        _, _, diagnostics = solve_constrained_nash(margins, reference, 0.5, seed=1)
        self.assertEqual(diagnostics["gradient_mode"], "finite_difference")

    def test_env_var_selects_analytic_without_touching_callers(self) -> None:
        """Simulates GWD_NLHF_GRADIENT=analytic reaching an unmodified caller."""

        old = os.environ.get("GWD_NLHF_GRADIENT")
        try:
            os.environ["GWD_NLHF_GRADIENT"] = "analytic"
            rng = np.random.default_rng(4)
            m = 3
            margins = _random_skew_symmetric(rng, m)
            reference = _uniform(m)
            _, _, diagnostics = solve_constrained_nash(margins, reference, 0.5, seed=1)
            self.assertEqual(diagnostics["gradient_mode"], "analytic")
        finally:
            if old is None:
                os.environ.pop("GWD_NLHF_GRADIENT", None)
            else:
                os.environ["GWD_NLHF_GRADIENT"] = old

    def test_tau_zero_short_circuit_still_records_mode(self) -> None:
        rng = np.random.default_rng(5)
        m = 3
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)
        _, _, diagnostics = solve_constrained_nash(
            margins, reference, 0.0, seed=1, gradient="analytic"
        )
        self.assertEqual(diagnostics["gradient_mode"], "analytic")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
