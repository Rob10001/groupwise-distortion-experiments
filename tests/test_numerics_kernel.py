"""The specialized float64 kernels, against the routines they replace.

These tests exist because the optimization they cover is only legitimate if it
changes nothing.  They therefore assert **bit equality**, not closeness: the
deployed policy feeds seeded discrete draws under the active protocol, and
archived runs are validated by file digests, so a one-ulp difference is a
scientific difference here, not a rounding detail.  Nothing in this file may be
relaxed to a tolerance without a stated reason for why exactness is no longer
required.

The reference side never touches the optimized code.  `log_sum_exp` is compared
against `scipy.special.logsumexp` directly, and the learner-level differential
tests rebuild a pre-optimization learner by substituting the general-purpose
routine back in (`_reference_kernels`), so the "before" side genuinely computes
what the tree computed before.
"""

from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from empirical_testing.algorithms import one_expert, span_adaptive  # noqa: E402
from empirical_testing.algorithms.one_expert import (  # noqa: E402
    _kl_prox_update,
    _kl_prox_update_with_candidate,
    _softmax_from_log,
    fit_groupwise_self_play,
    kl_divergence,
)
from empirical_testing.algorithms.span_adaptive import (  # noqa: E402
    fit_groupwise_span_adaptive,
)
from empirical_testing.numerics import TINY, log_sum_exp  # noqa: E402


def same_bits(left, right) -> bool:
    """Exact equality on float64, counting NaN as equal to NaN."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return left.shape == right.shape and bool(np.array_equal(left, right, equal_nan=True))


@contextmanager
def _reference_kernels():
    """Restore the pre-optimization numerical path for the duration of a block.

    Two substitutions, matching the two optimizations:

    * every module-level ``_log_sum_exp`` binding goes back to
      :func:`scipy.special.logsumexp`, the general-purpose routine the kernel
      specializes;
    * the span-adaptive inner loop's projection helper goes back to *computing
      the unconstrained candidate a second time* rather than receiving the one
      the projection already formed.

    Anything fitted inside this block is therefore fitted by the arithmetic the
    tree used before, with no path back into the optimized helper.
    """

    from empirical_testing.algorithms import nlhf
    from empirical_testing.reporting import metrics

    modules = (one_expert, span_adaptive, nlhf, metrics)
    saved = [(module, getattr(module, "_log_sum_exp")) for module in modules]
    saved_helper = span_adaptive._kl_prox_update_with_candidate

    def recomputing_helper(q, loss, learning_rate, reference, tau):
        projected = _reference_prox(q, loss, learning_rate, reference, tau)
        candidate = _softmax_from_log(
            np.log(np.maximum(q, TINY)) - learning_rate * np.asarray(loss)
        )
        return projected, candidate

    def _reference_prox(q, loss, learning_rate, reference, tau):
        return _kl_prox_update(q, loss, learning_rate, reference, tau)

    try:
        for module, _previous in saved:
            setattr(module, "_log_sum_exp", logsumexp)
        span_adaptive._kl_prox_update_with_candidate = recomputing_helper
        yield
    finally:
        for module, previous in saved:
            setattr(module, "_log_sum_exp", previous)
        span_adaptive._kl_prox_update_with_candidate = saved_helper


class LogSumExpKernelTests(unittest.TestCase):
    """`log_sum_exp` must equal `scipy.special.logsumexp` bit for bit."""

    def assert_matches(self, values) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.assertTrue(
            same_bits(log_sum_exp(values), np.asarray(logsumexp(values))),
            msg=f"log_sum_exp disagreed with scipy on {values!r}: "
                f"{log_sum_exp(values)!r} vs {logsumexp(values)!r}",
        )

    def test_singleton_and_small_vectors(self) -> None:
        for values in ([0.0], [3.25], [-7.5], [1.0, 2.0], [0.0, 0.0, 0.0]):
            self.assert_matches(values)

    def test_supported_action_and_group_counts(self) -> None:
        # Nothing may assume Coffee's four actions: the framework's learners run
        # over any m >= 2 and any K >= 1.
        rng = np.random.default_rng(11)
        for size in (2, 3, 4, 5, 7, 12, 33, 128):
            for _ in range(200):
                self.assert_matches(rng.normal(size=size))

    def test_exact_ties_and_near_ties(self) -> None:
        rng = np.random.default_rng(12)
        for size in (2, 4, 9):
            for _ in range(300):
                values = rng.normal(size=size)
                values[rng.integers(size)] = values[0]  # an exact tie
                self.assert_matches(values)
                self.assert_matches(np.full(size, values[0]))  # all tied
                near = values.copy()
                near[0] = np.nextafter(near[1], np.inf)  # one ulp apart
                self.assert_matches(near)

    def test_small_and_large_objective_scales(self) -> None:
        rng = np.random.default_rng(13)
        for scale in (1e-300, 1e-12, 1e-6, 1.0, 1e3, 1e6, 1e300):
            for _ in range(200):
                self.assert_matches(rng.normal(size=5) * scale)

    def test_extreme_finite_logits(self) -> None:
        # Values where a naive exp() would overflow or underflow to zero.
        for values in ([710.0, 0.0], [-745.0, -746.0], [1e308, 1e308],
                       [-1e308, -1e308], [800.0, 800.0, -800.0],
                       [np.finfo(np.float64).max, 1.0]):
            self.assert_matches(values)

    def test_masked_and_all_negative_infinity(self) -> None:
        # A -inf entry beside a finite maximum takes the fast path and must
        # contribute exactly zero; an all -inf vector is delegated.
        self.assert_matches([-np.inf, 3.0])
        self.assert_matches([-np.inf, -np.inf, 2.0])
        self.assert_matches([-np.inf, -np.inf])
        self.assert_matches([-np.inf])

    def test_nonfinite_inputs_are_delegated_not_approximated(self) -> None:
        for values in ([np.inf, 1.0], [np.nan, 1.0], [np.inf, np.nan],
                       [-np.inf, np.inf], [np.inf, np.inf]):
            self.assert_matches(values)

    def test_returns_float64(self) -> None:
        self.assertIsInstance(log_sum_exp(np.array([1.0, 2.0])), np.float64)
        self.assertEqual(np.asarray(log_sum_exp(np.array([1.0, 2.0]))).dtype, np.float64)

    def test_positive_affine_shift_invariance(self) -> None:
        # log_sum_exp(x + c) = log_sum_exp(x) + c, to the accuracy of the shift.
        rng = np.random.default_rng(14)
        values = rng.normal(size=6)
        for shift in (-500.0, -1.0, 0.0, 1.0, 500.0):
            self.assertAlmostEqual(
                float(log_sum_exp(values + shift)),
                float(log_sum_exp(values)) + shift,
                places=10,
            )


class KlDivergenceFastPathTests(unittest.TestCase):
    def test_fast_path_matches_the_masked_path_exactly(self) -> None:
        rng = np.random.default_rng(21)
        for size in (2, 4, 9):
            for _ in range(500):
                q = rng.dirichlet(np.ones(size))
                reference = rng.dirichlet(np.ones(size) * 3.0)
                masked = float(np.sum(q * (np.log(q) - np.log(reference))))
                self.assertTrue(same_bits(kl_divergence(q, reference), masked))

    def test_zero_coordinates_still_use_the_convention(self) -> None:
        q = np.array([0.5, 0.5, 0.0])
        reference = np.array([0.2, 0.3, 0.5])
        expected = 0.5 * np.log(0.5 / 0.2) + 0.5 * np.log(0.5 / 0.3)
        self.assertAlmostEqual(kl_divergence(q, reference), float(expected), places=15)
        self.assertTrue(np.isfinite(kl_divergence(q, reference)))


class KlProxCandidateTests(unittest.TestCase):
    """The split projection must be the same update, and hand back the same candidate."""

    def cases(self):
        rng = np.random.default_rng(31)
        references = [
            np.full(4, 0.25),
            np.array([0.7, 0.1, 0.1, 0.1]),
            np.array([0.97, 0.01, 0.01, 0.01]),   # highly skewed
            np.array([0.5, 0.5]),                  # two actions: closed form
            rng.dirichlet(np.ones(9)),
        ]
        for reference in references:
            size = reference.size
            cap = float(np.log(1.0 / reference.min()))
            radii = [0.0, 1e-12, 1e-6, 0.05, 0.5, cap * 0.99, cap, cap * 2.0]
            for tau in radii:
                for rate in (0.0, 1e-8, 0.1, 0.5, 1.0):
                    for _ in range(4):
                        q = rng.dirichlet(np.ones(size))
                        loss = rng.normal(size=size)
                        yield q, loss, rate, reference, float(tau)

    def test_projection_is_unchanged_by_the_split(self) -> None:
        seen_binding = seen_free = 0
        for q, loss, rate, reference, tau in self.cases():
            plain = _kl_prox_update(q, loss, rate, reference, tau)
            split, candidate = _kl_prox_update_with_candidate(q, loss, rate, reference, tau)
            self.assertTrue(same_bits(plain, split))
            if candidate is not None:
                if kl_divergence(candidate, reference) > tau + 1e-10:
                    seen_binding += 1
                else:
                    seen_free += 1
        # The sweep must actually exercise both sides of the constraint, or it
        # would be testing one branch and reporting on two.
        self.assertGreater(seen_binding, 0, "no binding-constraint case was exercised")
        self.assertGreater(seen_free, 0, "no slack-constraint case was exercised")

    def test_candidate_equals_the_inlined_expression_bit_for_bit(self) -> None:
        # This is the identity the span-adaptive loop now relies on: the caller
        # used to form softmax(log(anchor) + rate * reward) itself, and now
        # receives softmax(log(q) - rate * (-reward)). The two differ only by
        # IEEE-exact negations.
        for q, loss, rate, reference, tau in self.cases():
            _projected, candidate = _kl_prox_update_with_candidate(q, loss, rate, reference, tau)
            if candidate is None:
                self.assertLessEqual(tau, 1e-10)
                continue
            reward = -np.asarray(loss)
            inlined = _softmax_from_log(np.log(np.maximum(q, TINY)) + rate * reward)
            self.assertTrue(same_bits(candidate, inlined))

    def test_zero_radius_returns_the_reference_and_no_candidate(self) -> None:
        reference = np.full(3, 1.0 / 3.0)
        projected, candidate = _kl_prox_update_with_candidate(
            reference.copy(), np.array([1.0, -1.0, 0.0]), 0.3, reference, 0.0
        )
        self.assertTrue(same_bits(projected, reference))
        self.assertIsNone(candidate)

    def test_every_returned_point_is_feasible(self) -> None:
        for q, loss, rate, reference, tau in self.cases():
            projected, _candidate = _kl_prox_update_with_candidate(q, loss, rate, reference, tau)
            self.assertLessEqual(kl_divergence(projected, reference), tau + 1e-8)
            self.assertAlmostEqual(float(projected.sum()), 1.0, places=12)
            self.assertTrue(np.all(projected > 0.0))


class LearnerDifferentialTests(unittest.TestCase):
    """Whole fits, optimized versus the restored pre-optimization arithmetic."""

    def _inputs(self, n_rows=260, n_groups=3, m=4, seed=5):
        rng = np.random.default_rng(seed)
        groups = np.zeros((n_rows, n_groups), dtype=np.float64)
        groups[:, 0] = 1.0
        groups[:, 1:] = (rng.random((n_rows, n_groups - 1)) < 0.4).astype(np.float64)
        utilities = rng.random((n_rows, m))
        return groups, utilities

    def _compare(self, fit, **kwargs):
        optimized = fit(**kwargs)
        with _reference_kernels():
            reference = fit(**kwargs)
        for name in ("final_outer_log_masses", "outer_second_moments",
                     "inner_second_moments", "drawn_comparisons", "snapshot_weights"):
            self.assertTrue(
                same_bits(getattr(optimized, name), getattr(reference, name)),
                msg=f"{name} differs between the optimized and reference fits",
            )
        self.assertTrue(same_bits(
            np.stack(optimized._inner_policy_snapshots),
            np.stack(reference._inner_policy_snapshots)))
        self.assertTrue(same_bits(
            np.stack(optimized._outer_log_mass_snapshots),
            np.stack(reference._outer_log_mass_snapshots)))
        groups = kwargs["groups"]
        self.assertTrue(same_bits(optimized.predict(groups), reference.predict(groups)))
        return optimized, reference

    def test_fixed_rate_passive_matches_the_reference_arithmetic(self) -> None:
        groups, utilities = self._inputs()
        self._compare(
            fit_groupwise_self_play, groups=groups, utilities=utilities, beta=4.0,
            reference=np.full(4, 0.25), tau=0.5, protocol="passive", seed=3,
            group_masses=groups.mean(axis=0), delta=0.05,
        )

    def test_active_protocol_draws_the_same_comparisons(self) -> None:
        # The strongest of these: under the active protocol the deployed policy
        # feeds `rng.choice`, so a one-ulp difference in a probability could
        # select a different pair. `drawn_comparisons` is the record of every
        # pair and label actually queried, and it must match exactly.
        groups, utilities = self._inputs(seed=9)
        optimized, reference = self._compare(
            fit_groupwise_self_play, groups=groups, utilities=utilities, beta=6.0,
            reference=np.full(4, 0.25), tau=0.5, protocol="active", seed=17,
            group_masses=groups.mean(axis=0), delta=0.05,
        )
        self.assertEqual(optimized.drawn_comparisons.shape[0], groups.shape[0])
        self.assertTrue(np.all(np.isfinite(optimized.drawn_comparisons)))

    def test_span_adaptive_matches_including_projection_counts(self) -> None:
        groups, utilities = self._inputs(seed=23)
        optimized = fit_groupwise_span_adaptive(
            groups=groups, utilities=utilities, beta=4.0, reference=np.full(4, 0.25),
            tau=0.5, protocol="passive", seed=31,
        )
        with _reference_kernels():
            reference = fit_groupwise_span_adaptive(
                groups=groups, utilities=utilities, beta=4.0, reference=np.full(4, 0.25),
                tau=0.5, protocol="passive", seed=31,
            )
        # The projection diagnostic is the thing the candidate reuse touches.
        np.testing.assert_array_equal(optimized.diagnostics.projection_counts,
                                      reference.diagnostics.projection_counts)
        np.testing.assert_array_equal(optimized.diagnostics.update_counts,
                                      reference.diagnostics.update_counts)
        for name in ("Q", "C", "V", "eta_terminal", "theta_terminal"):
            self.assertTrue(same_bits(getattr(optimized.diagnostics, name),
                                      getattr(reference.diagnostics, name)))
        self.assertTrue(same_bits(optimized.predict(groups), reference.predict(groups)))
        # Q + C = V is the adaptive learner's own invariant; the reuse must not
        # have disturbed the state the increments are taken from.
        self.assertLess(float(np.max(np.abs(
            optimized.diagnostics.Q + optimized.diagnostics.C - optimized.diagnostics.V
        ))), 1e-9 * max(1.0, float(np.max(optimized.diagnostics.V))))

    def test_span_adaptive_active_matches(self) -> None:
        groups, utilities = self._inputs(seed=41)
        optimized = fit_groupwise_span_adaptive(
            groups=groups, utilities=utilities, beta=8.0, reference=np.full(4, 0.25),
            tau=0.4, protocol="active", seed=43,
        )
        with _reference_kernels():
            reference = fit_groupwise_span_adaptive(
                groups=groups, utilities=utilities, beta=8.0, reference=np.full(4, 0.25),
                tau=0.4, protocol="active", seed=43,
            )
        self.assertTrue(same_bits(optimized.drawn_comparisons, reference.drawn_comparisons))

    def test_nonuniform_reference_and_more_actions(self) -> None:
        groups, utilities = self._inputs(n_rows=180, n_groups=4, m=6, seed=55)
        reference = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
        self._compare(
            fit_groupwise_self_play, groups=groups, utilities=utilities, beta=np.inf,
            reference=reference, tau=0.3, protocol="passive", seed=61,
            group_masses=groups.mean(axis=0), delta=0.05,
        )

    def test_inactive_and_overlapping_groups(self) -> None:
        rng = np.random.default_rng(71)
        n_rows = 200
        groups = np.zeros((n_rows, 4), dtype=np.float64)
        groups[:, 0] = 1.0
        groups[:, 1] = (rng.random(n_rows) < 0.5).astype(np.float64)
        groups[:, 2] = groups[:, 1]           # fully overlapping with group 1
        groups[:, 3] = 0.0                    # never active
        groups[0, 3] = 1.0                    # active exactly once
        utilities = rng.random((n_rows, 3))
        optimized, _reference = self._compare(
            fit_groupwise_self_play, groups=groups, utilities=utilities, beta=2.0,
            reference=np.full(3, 1.0 / 3.0), tau=0.25, protocol="passive", seed=73,
            group_masses=np.maximum(groups.mean(axis=0), 1e-3), delta=0.05,
        )
        self.assertEqual(int(optimized.group_activation_counts[3]), 1)


if __name__ == "__main__":
    unittest.main()
