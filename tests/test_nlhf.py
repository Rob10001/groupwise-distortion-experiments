"""Regression tests for the constrained NLHF equilibrium oracle."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from empirical_testing.algorithms.nlhf import (
    EXPLOITABILITY_TOLERANCE,
    estimate_preference_matrix,
    exact_exploitability,
    fit_pooled_nlhf,
    preference_margins,
    solve_constrained_nash,
)
from empirical_testing.algorithms.one_expert import kl_divergence


def _uniform(m: int) -> np.ndarray:
    return np.full(m, 1.0 / m)


class PreferenceEstimationTests(unittest.TestCase):
    def test_manual_counts_reproduce_the_matrix(self) -> None:
        # 0 beats 1 twice and loses once; 0 beats 2 once; 1 vs 2 split.
        pairs = np.array([[0, 1], [0, 1], [0, 1], [0, 2], [1, 2], [1, 2]])
        signs = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
        matrix, observed, missing = estimate_preference_matrix(pairs, signs, 3)
        self.assertEqual(observed, 3)
        self.assertEqual(missing, ())
        self.assertAlmostEqual(matrix[0, 1], 2 / 3)
        self.assertAlmostEqual(matrix[1, 0], 1 / 3)
        self.assertAlmostEqual(matrix[0, 2], 1.0)
        self.assertAlmostEqual(matrix[2, 0], 0.0)
        self.assertAlmostEqual(matrix[1, 2], 0.5)
        self.assertTrue(np.allclose(np.diag(matrix), 0.5))

    def test_margin_orientation(self) -> None:
        """A[a][b] > 0 must mean a is preferred to b, matching the certificate."""
        pairs = np.array([[0, 1]] * 10)
        signs = np.ones(10)  # alternative 0 always wins
        matrix, _, _ = estimate_preference_matrix(pairs, signs, 2)
        margins = preference_margins(matrix)
        self.assertGreater(margins[0, 1], 0.0)
        self.assertLess(margins[1, 0], 0.0)
        self.assertTrue(np.allclose(margins, -margins.T))

    def test_unobserved_pairs_are_reported_not_hidden(self) -> None:
        pairs = np.array([[0, 1], [0, 1]])
        signs = np.array([1.0, -1.0])
        matrix, observed, missing = estimate_preference_matrix(pairs, signs, 3)
        self.assertEqual(observed, 1)
        self.assertEqual(set(missing), {(0, 2), (1, 2)})
        self.assertAlmostEqual(matrix[0, 2], 0.5)  # imputed, and declared above

    def test_missing_pairs_warn_through_the_public_entry_point(self) -> None:
        pairs = np.array([[0, 1], [0, 1]])
        signs = np.array([1.0, -1.0])
        with self.assertWarns(RuntimeWarning):
            fit_pooled_nlhf(pairs, signs, 3, _uniform(3), 0.3)

    def test_rejects_bad_input(self) -> None:
        good_pairs = np.array([[0, 1]])
        for pairs, signs, m in (
            (np.array([[0, 0]]), np.array([1.0]), 2),      # self-pair
            (np.array([[0, 5]]), np.array([1.0]), 2),      # out of range
            (good_pairs, np.array([0.0]), 2),              # tie sign
        ):
            with self.assertRaises(ValueError):
                estimate_preference_matrix(pairs, signs, m)


class EquilibriumTests(unittest.TestCase):
    def test_rock_paper_scissors_is_uniform(self) -> None:
        """The canonical cyclic game: the unique equilibrium is uniform."""
        matrix = np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
        margins = preference_margins(matrix)
        for tau in (0.05, 0.3, 1.0):
            policy, gap, _ = solve_constrained_nash(margins, _uniform(3), tau)
            self.assertTrue(
                np.allclose(policy, _uniform(3), atol=1e-6),
                f"tau={tau} gave {policy}",
            )
            self.assertLessEqual(gap, EXPLOITABILITY_TOLERANCE)

    def test_condorcet_winner_respects_the_ball(self) -> None:
        matrix = np.full((4, 4), 0.5)
        for j in range(3):
            matrix[3, j], matrix[j, 3] = 0.9, 0.1
        margins = preference_margins(matrix)
        reference, tau = _uniform(4), 0.5
        policy, gap, diagnostics = solve_constrained_nash(margins, reference, tau)
        self.assertEqual(int(np.argmax(policy)), 3)
        self.assertLessEqual(kl_divergence(policy, reference), tau + 1e-9)
        self.assertLessEqual(gap, EXPLOITABILITY_TOLERANCE)
        self.assertIn("final_kl", diagnostics)
        # tau above log(m) makes the ball vacuous, so the pure winner is feasible.
        loose, _, _ = solve_constrained_nash(margins, reference, np.log(4) + 0.01)
        self.assertGreater(loose[3], 0.99)

    def test_exploitability_is_nonnegative_and_minimized_at_the_solution(self) -> None:
        rng = np.random.default_rng(0)
        reference, tau = _uniform(4), 0.5
        for _ in range(20):
            raw = rng.random((4, 4))
            matrix = 0.5 * (raw + 1.0 - raw.T)
            np.fill_diagonal(matrix, 0.5)
            margins = preference_margins(matrix)
            policy, gap, _ = solve_constrained_nash(margins, reference, tau)
            self.assertLessEqual(gap, EXPLOITABILITY_TOLERANCE)
            self.assertAlmostEqual(policy.sum(), 1.0, places=9)
            self.assertTrue((policy >= -1e-12).all())
            self.assertLessEqual(kl_divergence(policy, reference), tau + 1e-9)
            # Any feasible competitor is at least as exploitable.
            other = 0.5 * (policy + reference)
            self.assertGreaterEqual(
                exact_exploitability(other, margins, reference, tau) + 1e-9, gap
            )


class FitTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        self.m = 4
        edges = [(a, b) for a in range(self.m) for b in range(a + 1, self.m)]
        index = rng.integers(0, len(edges), size=4000)
        self.pairs = np.array([edges[i] for i in index], dtype=np.int64)
        truth = np.array(
            [[.5, .7, .8, .3], [.3, .5, .6, .2], [.2, .4, .5, .1], [.7, .8, .9, .5]]
        )
        p_left = truth[self.pairs[:, 0], self.pairs[:, 1]]
        self.signs = np.where(rng.random(4000) < p_left, 1.0, -1.0)
        self.reference = _uniform(self.m)

    def test_fit_is_feasible_and_at_equilibrium(self) -> None:
        fitted = fit_pooled_nlhf(self.pairs, self.signs, self.m, self.reference, 0.5)
        self.assertAlmostEqual(fitted.policy.sum(), 1.0, places=9)
        self.assertLessEqual(kl_divergence(fitted.policy, self.reference), 0.5 + 1e-9)
        self.assertLessEqual(fitted.exploitability, EXPLOITABILITY_TOLERANCE)
        self.assertGreaterEqual(fitted.exploitability_display, 0.0)
        self.assertEqual(fitted.observed_pairs, 6)
        self.assertEqual(fitted.unobserved_pairs, ())

    def test_refitting_the_same_log_is_deterministic(self) -> None:
        a = fit_pooled_nlhf(self.pairs, self.signs, self.m, self.reference, 0.5, seed=3)
        b = fit_pooled_nlhf(self.pairs, self.signs, self.m, self.reference, 0.5, seed=3)
        self.assertTrue(np.allclose(a.policy, b.policy, atol=1e-12, rtol=0.0))

    def test_never_consumes_group_indicators(self) -> None:
        """The baseline is population level: predict uses only the row count."""
        fitted = fit_pooled_nlhf(self.pairs, self.signs, self.m, self.reference, 0.5)
        ones = fitted.predict(np.ones((7, 1)))
        demographic = fitted.predict(np.random.default_rng(0).integers(0, 2, (7, 5)))
        self.assertEqual(ones.shape, (7, self.m))
        self.assertTrue(np.array_equal(ones, demographic))
        self.assertTrue(np.allclose(ones, fitted.policy))

    def test_nonuniform_propensities_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_pooled_nlhf(
                self.pairs, self.signs, self.m, self.reference, 0.5,
                propensities=np.linspace(0.1, 0.2, self.pairs.shape[0]),
            )

    def test_reference_must_be_a_full_support_lottery(self) -> None:
        for bad in (np.array([0.5, 0.5, 0.0, 0.0]), np.array([0.3, 0.3, 0.3, 0.3])):
            with self.assertRaises(ValueError):
                fit_pooled_nlhf(self.pairs, self.signs, self.m, bad, 0.5)


if __name__ == "__main__":  # pragma: no cover
    warnings.simplefilter("always")
    unittest.main()
