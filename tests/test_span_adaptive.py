"""Implementation checks for the span-adaptive learner (algorithms/span_adaptive.py).

These verify the update-to-code mapping of docs/ADAPTIVE_IMPLEMENTATION.md:
reward sign and normalization, pre-update rate timing, the next-rate inner
exponent, both gamma corrections, the nonnegative centered C with Q + C = V,
inactive-state preservation, feasibility, exact all-round averaging, the
frozen-rate regression against the fixed learner, and an actual outer-rate
decrease. They are correctness tests, not theory.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import brentq

from empirical_testing.algorithms.one_expert import (
    _kl_prox_update,
    _policy_weighted_score,
    fit_groupwise_self_play,
    kl_divergence,
)
from empirical_testing.algorithms.registry import fit_online_learner
from empirical_testing.algorithms.span_adaptive import (
    ALGORITHM_ID,
    FrozenRates,
    SpanAdaptiveSchedule,
    complexity_term,
    effective_radius,
    fit_groupwise_span_adaptive,
)


def _random_log(T=300, K=5, m=4, seed=0, uniform_propensity=True):
    rng = np.random.default_rng(seed)
    groups = np.ones((T, K), dtype=np.int8)
    if K > 1:
        groups[:, 1:] = rng.integers(0, 2, size=(T, K - 1))
    pairs = np.array([sorted(rng.choice(m, 2, replace=False)) for _ in range(T)], dtype=np.int64)
    signs = rng.choice([-1.0, 1.0], size=T)
    n_pairs = m * (m - 1) // 2
    props = np.full(T, 1.0 / n_pairs) if uniform_propensity else rng.uniform(0.05, 0.5, size=T)
    return groups, pairs, signs, props


def _fit(groups, pairs, signs, props, tau=0.5, ref=None, seed=3, **kw):
    m = int(pairs.max()) + 1
    ref = np.full(m, 1.0 / m) if ref is None else ref
    return fit_groupwise_span_adaptive(groups, reference=ref, tau=tau, protocol="logged", pair_indices=pairs,
                                       signs=signs, propensities=props, seed=seed, **kw)


def _conflicting_construction(T=400):
    """K = 2, m = 2, tau = log 2, rho = 1: the universal group is told 0 beats 1
    when alone and 1 beats 0 when group B is present, so its expert disagrees
    with the deployed mixture every other round and Q grows quickly."""

    groups = np.ones((T, 2), dtype=np.int8)
    groups[:, 1] = (np.arange(T) % 2 == 1).astype(np.int8)
    pairs = np.tile([[0, 1]], (T, 1)).astype(np.int64)
    signs = np.where(groups[:, 1] == 1, -1.0, 1.0)
    props = np.ones(T)
    return groups, pairs, signs, props


class RewardTests(unittest.TestCase):
    def test_reward_is_minus_score_over_R_with_unit_bounds(self) -> None:
        groups, pairs, signs, props = _random_log()
        model = _fit(groups, pairs, signs, props)
        d = model.diagnostics
        self.assertEqual(model.score_bound, 6.0)
        self.assertLessEqual(d.max_abs_reward, 1.0 + 1e-9)
        self.assertLessEqual(d.max_span, 1.0 + 1e-9)
        self.assertLess(d.max_abs_cancellation, 2e-9)
        # Direct check of the sign convention on one comparison: the winner's
        # coordinate of g = -score / R is positive.
        policy = np.array([0.4, 0.3, 0.2, 0.1])
        score = _policy_weighted_score(policy, 0, 1, +1.0, 1.0 / 6.0)
        g = -score / 6.0
        self.assertGreater(g[0], 0.0)
        self.assertLess(g[1], 0.0)
        np.testing.assert_allclose(g[0], policy[1], atol=1e-12)  # pi(b)/(R rho) with R rho = 1
        np.testing.assert_allclose(g[1], -policy[0], atol=1e-12)
        self.assertAlmostEqual(float(policy @ g), 0.0, places=14)

    def test_active_span_is_exactly_one_and_R_is_m_minus_one(self) -> None:
        rng = np.random.default_rng(1)
        T, K, m = 200, 3, 4
        groups = np.ones((T, K), dtype=np.int8)
        groups[:, 1:] = rng.integers(0, 2, size=(T, K - 1))
        U = rng.uniform(size=(T, m))
        model = fit_groupwise_span_adaptive(groups, utilities=U, beta=4.0, reference=np.full(m, 0.25), tau=0.5,
                                            protocol="active", seed=5)
        self.assertEqual(model.score_bound, 3.0)
        self.assertAlmostEqual(model.diagnostics.max_span, 1.0, places=9)
        # One comparison per round, recorded with its propensity.
        self.assertEqual(model.drawn_comparisons.shape, (T, 4))
        self.assertTrue(np.all(np.isfinite(model.drawn_comparisons)))
        self.assertTrue(np.all(model.drawn_comparisons[:, 3] > 0))

    def test_span_not_asserted_for_arbitrary_logged_propensities(self) -> None:
        groups, pairs, signs, props = _random_log(uniform_propensity=False, seed=4)
        model = _fit(groups, pairs, signs, props)  # must not raise even if span > 1 occurs
        self.assertGreater(model.score_bound, 1.0)

    def test_learning_direction_on_a_tiny_example(self) -> None:
        # Two actions, action 0 always wins: mass moves to action 0.
        T = 60
        groups = np.ones((T, 1), dtype=np.int8)
        pairs = np.tile([[0, 1]], (T, 1)).astype(np.int64)
        model = _fit(groups, pairs, np.ones(T), np.ones(T), tau=0.3)
        prediction = model.predict(np.ones((1, 1), dtype=np.int8))[0]
        self.assertGreater(prediction[0], 0.5)
        self.assertLessEqual(kl_divergence(prediction, np.full(2, 0.5)), 0.3 + 1e-7)


class ScheduleAndMomentTests(unittest.TestCase):
    def test_initial_rates_and_complexity(self) -> None:
        self.assertAlmostEqual(complexity_term(1, 3), max(1.0, np.log(1 * (1 + 0.5 * np.log(4.0)))))
        self.assertEqual(complexity_term(1, 3), 1.0)  # floor binds for a short pooled run
        self.assertAlmostEqual(complexity_term(5, 2995), np.log(5 * (1 + 0.5 * np.log(2996.0))))
        s = SpanAdaptiveSchedule(horizon=100, n_groups=5, score_bound=6.0, tau_requested=0.5, tau_bar=0.5,
                                 complexity=complexity_term(5, 100))
        np.testing.assert_allclose(s.outer_rate(np.zeros(5)), 0.5)  # cap binds at Q = 0 for every K
        np.testing.assert_allclose(s.inner_rate(np.zeros(5)), 1.0)  # sqrt(0.5 / 0.5) = 1 at tau = 0.5
        np.testing.assert_allclose(s.inner_rate([3.75]), [np.sqrt(0.5 / (2 * 4.0))])
        small = SpanAdaptiveSchedule(horizon=100, n_groups=1, score_bound=6.0, tau_requested=0.02, tau_bar=0.02, complexity=1.0)
        self.assertAlmostEqual(float(small.inner_rate([0.0])[0]), np.sqrt(0.02 / 0.5))  # 0.2

    def test_identity_Q_plus_C_equals_V_and_increment_ranges(self) -> None:
        groups, pairs, signs, props = _random_log(T=300, K=5)
        model = _fit(groups, pairs, signs, props)
        d = model.diagnostics
        np.testing.assert_allclose(d.Q + d.C, d.V, rtol=0, atol=1e-10)
        self.assertLess(d.identity_residual_max, 1e-12)
        self.assertTrue(np.all(d.C >= 0.0))
        # Q increments lie in [0, 1] and C increments in [0, 1/4]: the totals are
        # bounded by the number of updates.
        self.assertTrue(np.all(d.Q <= d.activation_counts + 1e-12))
        self.assertTrue(np.all(d.C <= 0.25 * d.activation_counts + 1e-12))

    def test_rates_are_pre_update_nonincreasing_and_theta_uses_next_rate(self) -> None:
        groups, pairs, signs, props = _random_log(T=200, K=3, seed=7)
        model = _fit(groups, pairs, signs, props)
        trace = model.diagnostics.rate_trace
        eta = np.asarray(trace["eta"])
        theta = np.asarray(trace["theta"])
        self.assertTrue(np.all(np.diff(eta, axis=0) <= 1e-12))
        self.assertTrue(np.all(np.diff(theta, axis=0) <= 1e-12))
        # Round 0 uses the initial rates (pre-update timing).
        self.assertEqual(trace["rounds"][0], 0)
        np.testing.assert_allclose(eta[0], 0.5)
        np.testing.assert_allclose(theta[0], 1.0)
        # Terminal theta equals inner_rate(C_T) exactly: the last exponent used C after the update.
        np.testing.assert_allclose(model.diagnostics.theta_terminal, model.schedule.inner_rate(model.diagnostics.C))

    def test_hand_calculated_two_round_example(self) -> None:
        # K = 2, m = 2, tau large enough that no projection binds in two rounds, rho = 1.
        groups = np.array([[1, 1], [1, 1]], dtype=np.int8)
        pairs = np.array([[0, 1], [0, 1]], dtype=np.int64)
        signs = np.array([1.0, 1.0])
        props = np.ones(2)
        ref = np.full(2, 0.5)
        tau = np.log(2.0) - 1e-9  # below the cap so the ball is a genuine constraint
        model = _fit(groups, pairs, signs, props, tau=tau, ref=ref)
        L = complexity_term(2, 2)
        sched = SpanAdaptiveSchedule(2, 2, 1.0, tau, effective_radius(tau, ref), L)
        # Round 1: both experts at the reference, deployment = reference, g = -score/R with
        # score = sign/rho * (pi[0] e_1 - pi[1] e_0) = 0.5 e_1 - 0.5 e_0  ->  g = (0.5, -0.5).
        g = np.array([0.5, -0.5])
        r = float(ref @ g)  # 0
        self.assertAlmostEqual(r, 0.0)
        c = float(ref @ (g - r) ** 2)  # 0.25
        Q1, C1 = r * r, c
        eta1 = sched.outer_rate([Q1])[0]
        theta1 = sched.inner_rate([C1])[0]
        gamma_theta = theta1 / 1.0
        # Inner: anchor = pi^gamma ref^(1-gamma) = ref (both equal ref), exponent uses theta_next = theta1.
        expected_pi = np.exp(np.log(ref) + theta1 * g)
        expected_pi /= expected_pi.sum()
        if kl_divergence(expected_pi, ref) > tau:
            expected_pi = _kl_prox_update(ref, -g, float(theta1), ref, tau)
        # Second snapshot holds the post-round-1 inner policies (pre-round-2 state).
        np.testing.assert_allclose(model._inner_policy_snapshots[1][0], expected_pi, atol=1e-12)
        np.testing.assert_allclose(model._inner_policy_snapshots[1][1], expected_pi, atol=1e-12)
        # Outer: w_plus = w (1 + eta r) with r = 0 -> w_plus = 1/2; anchored update leaves log w = -log 2.
        np.testing.assert_allclose(model._outer_log_mass_snapshots[1], np.log(0.5) + np.log(eta1), atol=1e-12)
        np.testing.assert_allclose(model.diagnostics.C, [C1 + float(expected_pi @ (g - expected_pi @ g) ** 2)] * 2, atol=1e-12)
        self.assertGreater(1.0, gamma_theta)  # the inner rate did decrease after round 1

    def test_gamma_corrections_with_an_actual_outer_rate_decrease(self) -> None:
        groups, pairs, signs, props = _conflicting_construction(T=400)
        ref = np.full(2, 0.5)
        tau = np.log(2.0)
        model = _fit(groups, pairs, signs, props, tau=tau, ref=ref)
        d = model.diagnostics
        L = complexity_term(2, 400)
        self.assertGreater(d.Q[0], 2 * L - 1)  # eta leaves the cap only once Q exceeds 2L - 1
        self.assertLess(d.eta_terminal[0], 0.5)
        self.assertGreaterEqual(d.first_round_eta_below_cap[0], 0)
        eta = np.asarray(d.rate_trace["eta"])
        self.assertTrue(np.all(np.diff(eta[:, 0]) <= 1e-12))
        np.testing.assert_allclose(d.eta_terminal, model.schedule.outer_rate(d.Q))
        # The anchored outer update does not conserve total mass once eta moves.
        self.assertGreater(abs(d.outer_mass_terminal - 1.0), 1e-6)
        self.assertNotAlmostEqual(d.outer_mass_max, d.outer_mass_min, places=6)
        # Replay the recurrence with the recorded rates to check the displayed formula.
        log_w = np.full(2, -np.log(2.0))
        Q = np.zeros(2)
        pi = np.tile(ref, (2, 1))
        sched = model.schedule
        for t in range(400):
            eta_t = sched.outer_rate(Q)
            active = np.flatnonzero(groups[t])
            wts = np.exp(log_w[active] + np.log(eta_t[active]))
            wts /= wts.sum()
            dep = wts @ pi[active]
            score = _policy_weighted_score(dep, 0, 1, signs[t], 1.0)
            g = -score / model.score_bound
            r = pi[active] @ g
            Qn = Q[active] + r * r
            gamma = sched.outer_rate(Qn) / eta_t[active]
            log_w[active] = (1 - gamma) * (-np.log(2.0)) + gamma * (log_w[active] + np.log1p(eta_t[active] * r))
            Q[active] = Qn
            pi[active] = model._inner_policy_snapshots[t + 1][active] if t + 1 < 400 else pi[active]
        np.testing.assert_allclose(log_w, model.final_outer_log_masses, atol=1e-9)

    def test_frozen_rates_reproduce_the_fixed_learner(self) -> None:
        groups, pairs, signs, props = _random_log(T=250, K=4, seed=11)
        masses = groups.mean(axis=0)
        fixed = fit_groupwise_self_play(groups, reference=np.full(4, 0.25), tau=0.5, protocol="logged",
                                        pair_indices=pairs, signs=signs, propensities=props, seed=3, group_masses=masses)
        frozen = _fit(groups, pairs, signs, props, frozen_rates=FrozenRates(fixed.rates.outer, fixed.rates.inner))
        for a, b in zip(frozen._inner_policy_snapshots, fixed._inner_policy_snapshots):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        for a, b in zip(frozen._outer_log_mass_snapshots, fixed._outer_log_mass_snapshots):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        np.testing.assert_allclose(frozen.predict(groups), fixed.predict(groups), atol=1e-12, rtol=0)
        np.testing.assert_allclose(frozen.outer_second_moments, fixed.outer_second_moments, atol=1e-12)
        np.testing.assert_allclose(frozen.inner_second_moments, fixed.normalized_inner_second_moments, atol=1e-10)
        self.assertLess(abs(frozen.diagnostics.outer_mass_max - 1.0), 1e-12)
        self.assertTrue(frozen.diagnostics.frozen)
        # Active protocol: same seed derivation, so the same pairs are drawn.
        rng = np.random.default_rng(2)
        U = rng.uniform(size=(250, 4))
        fixed_a = fit_groupwise_self_play(groups, utilities=U, beta=4.0, reference=np.full(4, 0.25), tau=0.5,
                                          protocol="active", seed=9, group_masses=masses)
        frozen_a = fit_groupwise_span_adaptive(groups, utilities=U, beta=4.0, reference=np.full(4, 0.25), tau=0.5,
                                               protocol="active", seed=9,
                                               frozen_rates=FrozenRates(fixed_a.rates.outer, fixed_a.rates.inner))
        np.testing.assert_allclose(frozen_a.drawn_comparisons[:, :3], fixed_a.drawn_comparisons[:, :3], atol=0)
        np.testing.assert_allclose(frozen_a.predict(groups), fixed_a.predict(groups), atol=1e-12)


class StateAndFeasibilityTests(unittest.TestCase):
    def test_inactive_groups_keep_their_entire_state(self) -> None:
        groups, pairs, signs, props = _random_log(T=120, K=3, seed=5)
        groups[:, 2] = 0  # group 2 never active
        model = _fit(groups, pairs, signs, props)
        d = model.diagnostics
        self.assertEqual(d.activation_counts[2], 0)
        self.assertEqual(d.Q[2], 0.0)
        self.assertEqual(d.C[2], 0.0)
        self.assertEqual(d.V[2], 0.0)
        self.assertAlmostEqual(model.final_outer_log_masses[2], -np.log(3.0))
        np.testing.assert_allclose(model._inner_policy_snapshots[-1][2], np.full(4, 0.25))
        self.assertEqual(d.eta_terminal[2], 0.5)
        self.assertEqual(d.theta_terminal[2], 1.0)
        # A group active only in the first half keeps its state through the second half.
        groups2 = groups.copy()
        groups2[60:, 1] = 0
        m2 = _fit(groups2, pairs, signs, props)
        np.testing.assert_allclose(m2._inner_policy_snapshots[60][1], m2._inner_policy_snapshots[-1][1])
        np.testing.assert_allclose(m2._outer_log_mass_snapshots[60][1], m2._outer_log_mass_snapshots[-1][1])

    def test_pooled_K1_has_trivial_outer_state_and_deployment_equals_inner_policy(self) -> None:
        groups, pairs, signs, props = _random_log(T=200, K=1, seed=6)
        model = _fit(groups, pairs, signs, props)
        d = model.diagnostics
        self.assertLess(d.Q[0], 1e-24)  # r = pi . g cancels pathwise; only rounding remains
        self.assertEqual(d.eta_terminal[0], 0.5)
        self.assertAlmostEqual(d.outer_mass_terminal, 1.0, places=12)
        self.assertAlmostEqual(model.final_outer_log_masses[0], 0.0, places=12)
        # The averaged output equals the mean of the inner-policy snapshots.
        mean_inner = np.mean([s[0] for s in model._inner_policy_snapshots], axis=0)
        np.testing.assert_allclose(model.predict(np.ones((1, 1), dtype=np.int8))[0], mean_inner, atol=1e-12)

    def test_tau_zero_returns_reference_and_consumes_nothing(self) -> None:
        groups, pairs, signs, props = _random_log(T=50, K=2, seed=8)
        model = _fit(groups, pairs, signs, props, tau=0.0)
        np.testing.assert_allclose(model.predict(groups), np.full((50, 4), 0.25))
        self.assertEqual(model.drawn_comparisons.shape, (0, 4))
        self.assertTrue(np.all(model.diagnostics.Q == 0) and np.all(model.diagnostics.C == 0))
        self.assertTrue(np.all(model.group_activation_counts == 0))
        self.assertEqual(model.snapshot_rounds.size, 50)

    def test_effective_radius_cap_and_nonuniform_reference(self) -> None:
        groups, pairs, signs, props = _random_log(T=100, K=2, seed=9)
        capped = _fit(groups, pairs, signs, props, tau=9.0)
        self.assertAlmostEqual(capped.diagnostics.tau_bar, np.log(4.0))
        self.assertEqual(capped.diagnostics.tau_requested, 9.0)
        self.assertEqual(int(np.sum(capped.diagnostics.projection_counts)), 0)  # whole simplex
        ref = np.array([0.4, 0.3, 0.2, 0.1])
        model = _fit(groups, pairs, signs, props, tau=0.3, ref=ref)
        self.assertAlmostEqual(model.diagnostics.tau_bar, 0.3)
        self.assertAlmostEqual(model.diagnostics.theta_initial, min(1.0, np.sqrt(0.3 / 0.5)))
        for row in model.predict(groups):
            self.assertLessEqual(kl_divergence(row, ref), 0.3 + 1e-7)
            self.assertAlmostEqual(row.sum(), 1.0, places=12)
            self.assertTrue(np.all(row > 0))
        # Small radius with a binding projection stays feasible and is counted:
        # a consistent signal drives the inner policy to the boundary of the ball.
        T = 100
        one = np.ones((T, 1), dtype=np.int8)
        tight = _fit(one, np.tile([[0, 1]], (T, 1)).astype(np.int64), np.ones(T), np.ones(T), tau=0.01)
        self.assertGreater(int(np.sum(tight.diagnostics.projection_counts)), 0)
        self.assertLessEqual(tight.diagnostics.kl_inner_max, 0.01 + 1e-8)
        self.assertLessEqual(tight.diagnostics.kl_deployed_max, 0.01 + 2e-8)
        self.assertGreater(tight.diagnostics.kl_inner_max, 0.009)

    def test_projection_is_the_KL_projection_onto_the_ball(self) -> None:
        # argmin_{q in B} KL(q || u) has the form q ~ u^{1/(1+nu)} ref^{nu/(1+nu)} with KL(q||ref) = tau.
        ref = np.full(4, 0.25)
        anchor = np.array([0.5, 0.3, 0.15, 0.05])
        g = np.array([0.8, -0.2, -0.3, -0.3])
        theta, tau = 1.0, 0.1
        u = anchor * np.exp(theta * g)
        u /= u.sum()
        self.assertGreater(kl_divergence(u, ref), tau)
        prox = _kl_prox_update(anchor, -g, theta, ref, tau)

        def candidate(nu):
            q = u ** (1 / (1 + nu)) * ref ** (nu / (1 + nu))
            return q / q.sum()
        nu = brentq(lambda x: kl_divergence(candidate(x), ref) - tau, 0.0, 1e6, xtol=1e-14)
        np.testing.assert_allclose(prox, candidate(nu), atol=1e-8)
        # Bregman optimality: KL(prox || u) is minimal among feasible points on the geodesic and below a blend.
        blend = 0.7 * u + 0.3 * ref
        if kl_divergence(blend, ref) <= tau:
            self.assertLessEqual(kl_divergence(prox, u), kl_divergence(blend, u) + 1e-12)

    def test_exact_all_round_average_over_membership_patterns(self) -> None:
        groups, pairs, signs, props = _random_log(T=80, K=3, seed=12)
        model = _fit(groups, pairs, signs, props)
        self.assertEqual(model.snapshot_rounds.size, 80)
        np.testing.assert_allclose(model.snapshot_weights, 1.0 / 80)
        # A pattern absent from training, and the empty pattern, are still averaged exactly.
        patterns = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=np.int8)
        prediction = model.predict(patterns)
        for p, row in zip(patterns, prediction):
            manual = np.zeros(4)
            for log_mass, policies in zip(model._outer_log_mass_snapshots, model._inner_policy_snapshots):
                active = np.flatnonzero(p)
                if active.size == 0:
                    manual += np.full(4, 0.25) / 80
                    continue
                w = np.exp(log_mass[active] - log_mass[active].max())
                manual += (w @ policies[active]) / w.sum() / 80
            np.testing.assert_allclose(row, manual, atol=1e-12)

    def test_determinism_and_registry(self) -> None:
        groups, pairs, signs, props = _random_log(T=60, K=2, seed=13)
        a = _fit(groups, pairs, signs, props)
        b = _fit(groups, pairs, signs, props)
        np.testing.assert_array_equal(a.predict(groups), b.predict(groups))
        c = fit_online_learner(ALGORITHM_ID, groups, reference=np.full(4, 0.25), tau=0.5, protocol="logged",
                               pair_indices=pairs, signs=signs, propensities=props, seed=3,
                               group_masses=np.array([1.0, 0.5]), delta=0.01)  # ignored by the adaptive learner
        np.testing.assert_array_equal(a.predict(groups), c.predict(groups))
        with self.assertRaises(ValueError):
            fit_online_learner("groupwise_inner_outer", groups, reference=np.full(4, 0.25), tau=0.5, protocol="logged",
                               pair_indices=pairs, signs=signs, propensities=props)
        with self.assertRaisesRegex(ValueError, "group_masses"):
            fit_online_learner("groupwise_self_play", groups, reference=np.full(4, 0.25), tau=0.5, protocol="logged",
                               pair_indices=pairs, signs=signs, propensities=props)
        with self.assertRaises(ValueError):
            _fit(groups, pairs, signs, props, n_snapshots=16)


if __name__ == "__main__":
    unittest.main()
