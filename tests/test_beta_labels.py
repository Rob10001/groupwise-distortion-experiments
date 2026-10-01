"""Bradley-Terry label model, beta-aware margins, and the distortion bound."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import expit

from empirical_testing.algorithms.one_expert import _draw_comparison
from empirical_testing.core.comparisons import bradley_terry_signs
from empirical_testing.reporting.metrics import (
    constrained_optimum,
    distortion_bound,
    link_constants,
    margin_matrix,
    welfare,
)


class LabelModelTests(unittest.TestCase):
    def test_infinite_beta_is_the_sign_oracle(self) -> None:
        difference = np.array([0.5, -0.25, 0.0, 1.0, -1.0])
        signs = bradley_terry_signs(difference, np.inf, np.random.default_rng(0))
        self.assertTrue(np.array_equal(signs[[0, 1, 3, 4]], [1.0, -1.0, 1.0, -1.0]))
        self.assertIn(signs[2], (-1.0, 1.0))  # exact tie -> fair coin

    def test_infinite_beta_consumes_the_rng_exactly_as_before(self) -> None:
        """Only the tie count is drawn, which is what keeps archived runs
        byte-identical after the label model was introduced."""
        difference = np.array([0.5, 0.0, -0.5, 0.0])
        a = np.random.default_rng(7)
        bradley_terry_signs(difference, np.inf, a)
        b = np.random.default_rng(7)
        b.random(2)  # two ties
        self.assertEqual(a.integers(0, 10_000), b.integers(0, 10_000))

    def test_finite_beta_matches_the_bradley_terry_link(self) -> None:
        n = 200_000
        for beta, difference in ((1.0, 0.5), (4.0, 0.25), (0.5, 1.0)):
            signs = bradley_terry_signs(
                np.full(n, difference), beta, np.random.default_rng(3)
            )
            observed = float((signs > 0).mean())
            expected = float(expit(beta * difference))
            self.assertAlmostEqual(observed, expected, places=2,
                                   msg=f"beta={beta} d={difference}")

    def test_agrees_with_the_learners_scalar_rule(self) -> None:
        """The log builder and `_draw_comparison` must not drift apart."""
        beta, difference, n = 2.0, 0.75, 60_000
        utilities = np.tile(np.array([difference, 0.0]), (n, 1))
        pairs = np.tile(np.array([[0, 1]]), (n, 1))
        rng = np.random.default_rng(5)
        scalar = np.array([
            _draw_comparison(i, np.array([0.5, 0.5]), utilities, beta, "passive",
                             pairs, None, None, pairs, rng)[2]
            for i in range(n)
        ])
        vector = bradley_terry_signs(
            np.full(n, difference), beta, np.random.default_rng(6)
        )
        self.assertAlmostEqual(float((scalar > 0).mean()),
                               float((vector > 0).mean()), places=2)

    def test_rejects_bad_beta(self) -> None:
        for bad in (0.0, -1.0, np.nan, -np.inf):
            with self.assertRaises(ValueError):
                bradley_terry_signs(np.array([0.5]), bad, np.random.default_rng(0))


class MarginTests(unittest.TestCase):
    def test_centered_sigmoid_margin_tends_to_sign(self) -> None:
        utilities = np.array([[0.0, 0.25, 0.5, 1.0], [0.5, 0.5, 0.25, 0.75]])
        sign = margin_matrix(utilities, np.inf)
        self.assertTrue(np.array_equal(sign, np.sign(
            utilities[:, :, None] - utilities[:, None, :])))
        self.assertTrue(np.allclose(margin_matrix(utilities, 1e6), sign))

    def test_exact_ties_give_zero_margin_at_every_beta(self) -> None:
        utilities = np.full((3, 4), 0.5)
        for beta in (0.5, 2.0, np.inf):
            block = margin_matrix(utilities, beta)
            self.assertTrue(np.all(block == 0.0))
            self.assertFalse(np.any(np.isnan(block)))

    def test_margin_is_skew_symmetric_and_bounded(self) -> None:
        rng = np.random.default_rng(1)
        utilities = rng.random((6, 4))
        for beta in (0.25, 1.0, 8.0, np.inf):
            block = margin_matrix(utilities, beta)
            self.assertTrue(np.allclose(block, -np.transpose(block, (0, 2, 1))))
            self.assertLessEqual(np.abs(block).max(), 1.0 + 1e-12)

    def test_rejects_bad_beta(self) -> None:
        for bad in (0.0, -1.0, np.nan):
            with self.assertRaises(ValueError):
                margin_matrix(np.zeros((2, 2)), bad)


class LinkConstantTests(unittest.TestCase):
    def test_factor_matches_hand_values(self) -> None:
        for beta, expected in ((0.5, 1.0207), (2.0, 1.3130),
                               (4.0, 2.0746), (8.0, 4.0027)):
            self.assertAlmostEqual(link_constants(beta)["factor"], expected, places=4)

    def test_factor_tends_to_one_as_beta_vanishes(self) -> None:
        self.assertAlmostEqual(link_constants(1e-6)["factor"], 1.0, places=9)

    def test_infinite_beta_has_no_finite_factor(self) -> None:
        """Why the deterministic sign oracle cannot be compared to the bound."""
        self.assertTrue(np.isposinf(link_constants(np.inf)["factor"]))
        self.assertTrue(np.isposinf(distortion_bound(0.01, 0.5, np.inf)))

    def test_bound_is_vacuous_when_the_certificate_is_large(self) -> None:
        beta, optimum = 2.0, 0.5
        s = link_constants(beta)["s_beta"]
        self.assertTrue(np.isposinf(distortion_bound(s * optimum + 0.01, optimum, beta)))
        self.assertTrue(np.isfinite(distortion_bound(0.0, optimum, beta)))

    def test_bound_at_zero_certificate_is_the_limiting_factor(self) -> None:
        for beta in (0.5, 2.0, 8.0):
            self.assertAlmostEqual(
                distortion_bound(0.0, 0.5, beta), link_constants(beta)["factor"],
                places=10,
            )


class ConstrainedOptimumTests(unittest.TestCase):
    def test_dominates_any_fixed_feasible_lottery(self) -> None:
        rng = np.random.default_rng(2)
        utilities = rng.random((200, 4))
        reference, tau = np.full(4, 0.25), 0.5
        optimum = constrained_optimum(utilities, reference, tau)
        for _ in range(50):
            candidate = rng.dirichlet(np.full(4, 3.0))
            from empirical_testing.algorithms.one_expert import kl_divergence
            if kl_divergence(candidate, reference) > tau:
                continue
            achieved = float(welfare(np.tile(candidate, (200, 1)), utilities).mean())
            self.assertLessEqual(achieved, optimum + 1e-9)

    def test_never_exceeds_the_unconstrained_optimum(self) -> None:
        rng = np.random.default_rng(4)
        utilities = rng.random((50, 4))
        reference = np.full(4, 0.25)
        unconstrained = float(np.max(utilities.mean(axis=0)))
        for tau in (0.05, 0.5, 5.0):
            self.assertLessEqual(
                constrained_optimum(utilities, reference, tau), unconstrained + 1e-9
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
