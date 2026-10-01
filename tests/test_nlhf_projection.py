"""Near-feasible solver output is projected onto the KL ball, not thrown away."""

from __future__ import annotations

import unittest

import numpy as np

from empirical_testing.algorithms import kl_divergence
from empirical_testing.algorithms.nlhf import (
    EXPLOITABILITY_TOLERANCE,
    project_to_kl_ball,
    solve_constrained_nash,
)


class ProjectionTest(unittest.TestCase):
    def test_feasible_point_is_unchanged(self) -> None:
        u = np.full(6, 1 / 6)
        p = np.array([0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
        np.testing.assert_array_equal(project_to_kl_ball(p, u, 0.5), p)

    def test_infeasible_point_lands_on_the_boundary(self) -> None:
        u = np.full(6, 1 / 6)
        p = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
        self.assertGreater(kl_divergence(p, u), 0.3)
        q = project_to_kl_ball(p, u, 0.3)
        self.assertLessEqual(kl_divergence(q, u), 0.3)
        self.assertGreater(kl_divergence(q, u), 0.3 - 1e-9)
        self.assertAlmostEqual(q.sum(), 1.0, places=12)
        self.assertEqual(int(np.argmax(q)), 0)  # same ordering as the input

    def test_large_ball_many_actions_reaches_tolerance(self) -> None:
        # Random skew-symmetric margins, m = 20, tau = 1: the regime where SLSQP
        # stops a hair outside the ball. Every seed must reach the equilibrium.
        rng = np.random.default_rng(11)
        u = np.full(20, 0.05)
        for seed in range(3):
            raw = rng.uniform(-1.0, 1.0, size=(20, 20))
            margins = 0.5 * (raw - raw.T)
            policy, gap, _ = solve_constrained_nash(margins, u, 1.0, seed=seed)
            self.assertLessEqual(kl_divergence(policy, u), 1.0 + 1e-9)
            self.assertLessEqual(gap, EXPLOITABILITY_TOLERANCE, f"seed {seed}: gap {gap}")


if __name__ == "__main__":
    unittest.main()
