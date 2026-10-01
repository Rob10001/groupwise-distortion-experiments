"""D-304: SLSQP restart failures must be counted and their reasons recorded.

The restart loop in ``solve_constrained_nash`` catches any exception raised
by ``scipy.optimize.minimize`` and moves on to the next restart so a single
flaky restart cannot crash the solve. This test forces exactly one restart to
raise and asserts the failure is recorded (count + reason), without changing
the accepted policy or the ``continue`` control flow.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from empirical_testing.algorithms.nlhf import preference_margins, solve_constrained_nash


def _toy_margins(m: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    matrix = np.full((m, m), 0.5)
    for a in range(m):
        for b in range(a + 1, m):
            p = 0.5 + 0.1 * rng.standard_normal()
            matrix[a, b] = np.clip(p, 0.05, 0.95)
            matrix[b, a] = 1.0 - matrix[a, b]
    return preference_margins(matrix)


class RestartFailureDiagnosticsTests(unittest.TestCase):
    def test_one_failed_restart_is_counted_with_its_reason(self) -> None:
        margins = _toy_margins()
        reference = np.full(3, 1.0 / 3.0)
        tau = 0.5

        real_minimize = __import__(
            "scipy.optimize", fromlist=["minimize"]
        ).minimize
        calls = {"n": 0}

        def flaky_minimize(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic solver blowup for D-304 test")
            return real_minimize(*args, **kwargs)

        with mock.patch(
            "empirical_testing.algorithms.nlhf.minimize", side_effect=flaky_minimize
        ):
            # Restart behaviour is a property of the historical multi-start
            # path, which must be asked for by name: the suite default makes one
            # call and never retries (see
            # ``test_the_one_start_default_reports_a_failure_instead_of_retrying``).
            policy, gap, diagnostics = solve_constrained_nash(
                margins, reference, tau, restarts=4, seed=0, selection="restart_legacy"
            )

        self.assertEqual(diagnostics["restart_failures"], 1)
        reasons = diagnostics["restart_failure_reasons"]
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0]["type"], "RuntimeError")
        self.assertIn("synthetic solver blowup", reasons[0]["message"])
        self.assertEqual(reasons[0]["attempt"], 0)

        # The failed restart must not have derailed the solve: a later
        # restart still ran and produced a valid, near-optimal policy.
        self.assertGreater(calls["n"], 1)
        self.assertLess(gap, 1e-4)
        self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=8)

    def test_no_failures_reports_empty(self) -> None:
        margins = _toy_margins()
        reference = np.full(3, 1.0 / 3.0)
        _, _, diagnostics = solve_constrained_nash(
            margins, reference, 0.5, restarts=3, seed=0, selection="restart_legacy"
        )
        self.assertEqual(diagnostics["restart_failures"], 0)
        self.assertEqual(diagnostics["restart_failure_reasons"], [])

    def test_the_one_start_default_reports_a_failure_instead_of_retrying(self) -> None:
        """A failed fit is visible, never papered over with another start."""

        margins = _toy_margins()
        reference = np.full(3, 1.0 / 3.0)
        real_minimize = __import__("scipy.optimize", fromlist=["minimize"]).minimize
        calls = {"n": 0}

        def always_failing(*args, **kwargs):
            calls["n"] += 1
            raise RuntimeError("synthetic solver blowup")

        with mock.patch(
            "empirical_testing.algorithms.nlhf.minimize", side_effect=always_failing
        ):
            # ``restarts`` is offered and deliberately ignored: the one-start
            # rule is not negotiable through an argument.
            policy, gap, diagnostics = solve_constrained_nash(
                margins, reference, 0.5, restarts=20, seed=0
            )

        self.assertEqual(calls["n"], 1, "the failed solve was retried")
        self.assertEqual(diagnostics["optimizer_calls"], 1)
        self.assertFalse(diagnostics["solver_success"])
        self.assertIn("synthetic solver blowup", diagnostics["solver_status"])
        self.assertEqual(diagnostics["failure_reason"], "optimizer raised")
        # The failure is propagated as an unusable exploitability rather than
        # dressed up as a successful uniform fit.
        self.assertEqual(gap, float("inf"))
        self.assertEqual(diagnostics["restart_failures"], 1)

    def test_a_failed_one_start_fit_is_rejected_by_the_exploitability_gate(self) -> None:
        """The failure must stop the pipeline, not flow into results."""

        from empirical_testing.algorithms.nlhf import fit_pooled_nlhf

        rng = np.random.default_rng(3)
        n, m = 40, 3
        pairs = np.array([[a, b] for a in range(m) for b in range(a + 1, m)] * 8)[:n]
        signs = rng.choice([-1.0, 1.0], size=len(pairs))

        def always_failing(*args, **kwargs):
            raise RuntimeError("synthetic solver blowup")

        with mock.patch(
            "empirical_testing.algorithms.nlhf.minimize", side_effect=always_failing
        ):
            with self.assertRaises(AssertionError) as caught:
                fit_pooled_nlhf(
                    pair_indices=pairs, signs=signs, n_alternatives=m,
                    reference=np.full(m, 1.0 / m), tau=0.5, seed=0,
                )
        self.assertIn("did not reach the equilibrium", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
