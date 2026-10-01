"""Tests for minimum-KL-to-reference NLHF equilibrium selection.

Covers explicit legacy-selection fixtures, minimum-KL tie breaking, feasibility,
input validation, training-only inputs, and stability across solver restarts.
The historical "min_kl" rule is selected explicitly with
NLHF_SELECTION_EPS = 3e-6; the production default is "reference_start_v10".
"""

from __future__ import annotations

import inspect
import re
import unittest
from unittest import mock

import numpy as np

from empirical_testing.algorithms import nlhf as nlhf_module
from empirical_testing.algorithms.nlhf import (
    EXPLOITABILITY_TOLERANCE,
    NLHF_BASELINE_VERSION_LEGACY,
    NLHF_BASELINE_VERSION_MIN_KL,
    NLHF_SELECTION_EPS,
    _MIN_KL_TIE_MARGIN,
    _safe,
    _solve_min_kl_robust,
    exact_exploitability,
    fit_pooled_nlhf,
    solve_constrained_nash,
)
from empirical_testing.algorithms.one_expert import kl_divergence

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _uniform(m: int) -> np.ndarray:
    return np.full(m, 1.0 / m)


def _random_skew_symmetric(rng: np.random.Generator, m: int) -> np.ndarray:
    raw = rng.normal(size=(m, m))
    return 0.5 * (raw - raw.T)


#: In this constructed degenerate game, actions 1 and 2 are
#: exact payoff clones of each other (A[0,1]=A[0,2], A[1,3]=A[2,3], A[1,2]=0),
#: so the game reduces to a generic, non-degenerate 3-action zero-sum game on
#: {0, class{1,2}, 3}. Solving ``A @ p = 0`` for that reduced game by hand
#: gives a UNIQUE reduced
#: equilibrium (p0, pY, p3) = (4/13, 3/13, 6/13); any split of pY between
#: actions 1 and 2 is then also an exact equilibrium of the full 4-action
#: game (exploitability identically 0 on the whole segment, since A @ p = 0
#: regardless of the 1/2 split), giving a genuine positive-dimensional
#: equilibrium segment that is analytically tractable.
TIE_MARGINS = np.array(
    [
        [0.0, 0.6, 0.6, -0.3],
        [-0.6, 0.0, 0.0, 0.4],
        [-0.6, 0.0, 0.0, 0.4],
        [0.3, -0.4, -0.4, 0.0],
    ]
)
TIE_REFERENCE = _uniform(4)
TIE_TAU = 1.0
#: With a uniform reference, the KL-minimizer over the segment
#: {(4/13, p1, 3/13 - p1, 6/13) : 0 <= p1 <= 3/13} is the balanced split
#: p1 = p2 = 3/26 (Lagrange stationarity on p1 + p2 = 3/13 with equal
#: reference mass forces p1 = p2; see report).
TIE_ANALYTIC_MIN_KL = np.array([4.0 / 13.0, 1.5 / 13.0, 1.5 / 13.0, 6.0 / 13.0])

_SLACK = 1e-6  # generic "tiny" tolerance used throughout, well above solver ftol


def _assert_feasible(test: unittest.TestCase, policy: np.ndarray,
                     margins: np.ndarray, reference: np.ndarray, tau: float,
                     eps_sel: float, slack: float = 1e-7) -> float:
    """Independently re-check simplex + KL-ball + exact-exploitability feasibility.

    Returns the recomputed exact exploitability.
    """

    test.assertAlmostEqual(float(policy.sum()), 1.0, places=9)
    test.assertTrue((policy >= -1e-9).all())
    kl = kl_divergence(policy, reference)
    test.assertLessEqual(kl, tau + slack)
    gap = exact_exploitability(policy, margins, reference, tau)
    test.assertLessEqual(gap, eps_sel + slack)
    return gap


# ---------------------------------------------------------------------------
# (1) Explicit legacy-selection regression against frozen numerical fixtures.
# ---------------------------------------------------------------------------

#: Values produced by the historical restart-based `solve_constrained_nash`
#: implementation on these exact fixtures.

_FROZEN_FIXTURES = [
    # (m, tau, seed, margins, reference, expected_policy, expected_gap)
    (
        2, 0.5, 0,
        [[0.0, -1.476751278525685], [1.476751278525685, 0.0]],
        [0.5, 0.5],
        [0.04818874584360389, 0.9518112541563961],
        -9.71445146547012e-17,
    ),
    (
        3, 0.5, 3,
        [[0.0, 0.13839330790322985, -1.4554553275524236],
         [-0.13839330790322985, 0.0, 0.4603668357042715],
         [1.4554553275524236, -0.4603668357042715, 0.0]],
        [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
        [0.22410720020982544, 0.7085218503269226, 0.06737094946325199],
        1.3971783962891776e-07,
    ),
    (
        4, 1.0, 0,
        [[0.0, -0.6226900005979931, -0.10087518550481081, -0.2272726454128911],
         [0.6226900005979931, 0.0, 0.5331485870502407, 0.03763397808518515],
         [0.10087518550481081, -0.5331485870502407, 0.0, -0.7487291400923599],
         [0.2272726454128911, -0.03763397808518515, 0.7487291400923599, 0.0]],
        [0.25, 0.25, 0.25, 0.25],
        [3.810970636286548e-17, 0.8700485727198841, 0.0, 0.1299514272801159],
        1.0456913113188193e-14,
    ),
    (
        5, 0.3, 3,
        [[0.0, 0.22321319838867482, -0.7030492803430294, -0.6506949030311799, 0.6057811572746616],
         [-0.22321319838867482, 0.0, 1.2403711848427803, -1.8293710045026408, -0.44612316415583253],
         [0.7030492803430294, -1.2403711848427803, 0.0, -1.0216628945583892, -0.007459315618562945],
         [0.6506949030311799, 1.8293710045026408, 1.0216628945583892, 0.0, -0.9995825186226419],
         [-0.6057811572746616, 0.44612316415583253, 0.007459315618562945, 0.9995825186226419, 0.0]],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.19124472126087147, 0.0029534058582230444, 0.26048592483747257,
         0.12602023649126015, 0.4192957115521728],
        -7.721254191572768e-15,
    ),
    (
        "tie", 1.0, 0,
        TIE_MARGINS.tolist(), TIE_REFERENCE.tolist(),
        [0.3076933730780211, 0.09275049682631487, 0.13801842182884425, 0.4615377082668198],
        3.922211777439833e-07,
    ),
    (
        "tie", 1.0, 2,
        TIE_MARGINS.tolist(), TIE_REFERENCE.tolist(),
        [0.30769337369966254, 0.12563122354648196, 0.10513769410627154, 0.461537708647584],
        3.9264845929157397e-07,
    ),
]


class FrozenLegacyRegressionTests(unittest.TestCase):
    """Explicit legacy selection reproduces the frozen numerical fixtures.

    These tests request "restart_legacy" to preserve coverage of that
    historical rule independently of the production default."""

    #: Fixtures whose equilibrium is not unique, so *which* optimum a multi-start
    #: search returns is not a property of the code.  ``TIE_MARGINS`` is
    #: deliberately degenerate: several policies reach the same near-zero
    #: exploitability, the restarts land on them in an order that depends on the
    #: platform's floating-point arithmetic, and the legacy rule then keeps
    #: whichever came first.  Their *gap* is still pinned below, and that is the
    #: quantity the selection rule is actually specified in terms of.
    #:
    #: This non-uniqueness is precisely the fragility the single reference-start
    #: default removes; it is retained here only to keep the historical path
    #: covered.
    _NON_UNIQUE = {"tie"}

    #: The frozen fixtures were recorded on the reference platform (CPython
    #: 3.12, x86-64).  SLSQP's iterates are not portable to the last bit, so the
    #: reproduction is asserted at a stated tolerance rather than exactly.
    _GAP_RTOL = 1e-2
    _POLICY_ATOL = 1e-6

    def test_explicit_restart_legacy_matches_frozen_pre_change_output(self) -> None:
        for m, tau, seed, margins, reference, expected_policy, expected_gap in _FROZEN_FIXTURES:
            margins = np.array(margins)
            reference = np.array(reference)
            restarts = 20 if m == "tie" else 4
            with self.subTest(m=m, tau=tau, seed=seed):
                policy, gap, diagnostics = solve_constrained_nash(
                    margins, reference, tau, seed=seed, restarts=restarts,
                    selection="restart_legacy",
                )
                # The solution is an equilibrium to the accuracy the suite
                # gates on, whichever optimum was selected.
                self.assertLess(abs(gap), EXPLOITABILITY_TOLERANCE)
                self.assertLess(abs(expected_gap), EXPLOITABILITY_TOLERANCE)
                # It is the *same* equilibrium value as the frozen one.  Both
                # are compared on an absolute floor too, because several
                # fixtures sit at machine zero where a relative test is
                # meaningless.
                self.assertLess(
                    abs(gap - expected_gap),
                    self._GAP_RTOL * abs(expected_gap) + 1e-12,
                    f"gap moved: frozen={expected_gap!r} here={gap!r}",
                )
                self.assertTrue(np.all(policy >= 0.0))
                self.assertAlmostEqual(float(policy.sum()), 1.0, places=9)
                if m not in self._NON_UNIQUE:
                    np.testing.assert_allclose(
                        policy, np.array(expected_policy), atol=self._POLICY_ATOL, rtol=0.0
                    )
                self.assertEqual(diagnostics["selection"], NLHF_BASELINE_VERSION_LEGACY)
                self.assertEqual(diagnostics["baseline_version"], NLHF_BASELINE_VERSION_LEGACY)

    def test_explicit_restart_legacy_is_deterministic(self) -> None:
        # Identical explicit legacy calls must return identical output,
        # independently of the production default's selection rule.
        rng = np.random.default_rng(2026)
        m = 4
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)
        p_first, g_first, d_first = solve_constrained_nash(
            margins, reference, 0.5, seed=1, selection="restart_legacy"
        )
        p_second, g_second, d_second = solve_constrained_nash(
            margins, reference, 0.5, seed=1, selection="restart_legacy"
        )
        np.testing.assert_array_equal(p_first, p_second)
        self.assertEqual(g_first, g_second)
        for key in d_first:
            self.assertEqual(d_first[key], d_second[key], key)

    def test_fit_pooled_nlhf_explicit_restart_legacy_selection(self) -> None:
        rng = np.random.default_rng(77)
        m = 3
        edges = [(a, b) for a in range(m) for b in range(a + 1, m)]
        index = rng.integers(0, len(edges), size=600)
        pairs = np.array([edges[i] for i in index], dtype=np.int64)
        signs = np.where(rng.random(600) < 0.5, 1.0, -1.0)
        fitted = fit_pooled_nlhf(
            pairs, signs, m, _uniform(m), 0.5, seed=1, selection="restart_legacy"
        )
        self.assertEqual(fitted.diagnostics["selection"], NLHF_BASELINE_VERSION_LEGACY)
        self.assertEqual(fitted.diagnostics["baseline_version"], NLHF_BASELINE_VERSION_LEGACY)


# ---------------------------------------------------------------------------
# (2) min_kl: feasibility, exact-exploitability <= eps_sel, KL ordering.
# ---------------------------------------------------------------------------


class MinKLFeasibilityAndOrderingTests(unittest.TestCase):
    def _check_m(self, m: int, tau: float, seed: int) -> None:
        rng = np.random.default_rng(3000 + m)
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)

        # Use the historical restart-loop primal as an independent reference,
        # selected explicitly so the production default does not affect it.
        primal_policy, primal_gap, primal_diag = solve_constrained_nash(
            margins, reference, tau, seed=seed, selection="restart_legacy",
        )
        eps_sel = max(primal_gap * 10.0, 1e-6)

        selected, selected_gap, diagnostics = solve_constrained_nash(
            margins, reference, tau, seed=seed, selection="min_kl", eps_sel=eps_sel,
        )
        self.assertEqual(diagnostics["selection"], "min_kl")
        self.assertEqual(diagnostics["baseline_version"], NLHF_BASELINE_VERSION_MIN_KL)
        self.assertEqual(diagnostics["min_kl_status"], "ok",
                         f"m={m} tau={tau} seed={seed}: {diagnostics['min_kl_status']}")

        # Independent re-verification (not the solver's own success flag).
        recomputed_gap = _assert_feasible(self, selected, margins, reference, tau, eps_sel)
        self.assertAlmostEqual(recomputed_gap, selected_gap, places=9)
        self.assertLessEqual(diagnostics["secondary_exact_exploitability"], eps_sel + 1e-7)

        primal_kl = kl_divergence(primal_policy, reference)
        selected_kl = kl_divergence(selected, reference)
        self.assertLessEqual(selected_kl, primal_kl + _SLACK)
        self.assertAlmostEqual(diagnostics["primary_kl"], primal_kl, places=9)
        self.assertAlmostEqual(diagnostics["selected_kl"], selected_kl, places=9)
        self.assertGreaterEqual(diagnostics["secondary_iterations"], 0)

    def test_m2(self) -> None:
        self._check_m(2, 0.5, 0)

    def test_m3(self) -> None:
        self._check_m(3, 0.5, 1)

    def test_m4(self) -> None:
        self._check_m(4, 0.75, 2)

    def test_m5(self) -> None:
        self._check_m(5, 1.0, 3)


# ---------------------------------------------------------------------------
# (3) Independent accuracy check on m <= 3: dense sampling over S_eps finds
#     nothing materially better, plus a local no-feasible-descent check
#     (KKT-residual-style: no feasible direction improves the objective).
# ---------------------------------------------------------------------------


def _dense_feasible_min_kl(margins: np.ndarray, reference: np.ndarray, tau: float,
                           eps_sel: float, rng: np.random.Generator, n: int,
                           slack: float) -> float | None:
    """Best (lowest) KL found by dense random sampling of the simplex,
    restricted to the feasible set {KL<=tau, exact exploitability<=eps_sel}.

    An independent method from the SLSQP secondary solve: no solver, no
    gradients, just direct sampling + the same exact-verification primitives
    used everywhere else in this module.
    """

    m = reference.size
    best = None
    for policy in rng.dirichlet(np.full(m, 1.0), size=n):
        if kl_divergence(policy, reference) > tau + slack:
            continue
        if exact_exploitability(policy, margins, reference, tau) > eps_sel + slack:
            continue
        kl = kl_divergence(policy, reference)
        if best is None or kl < best:
            best = kl
    return best


def _local_no_feasible_descent(policy: np.ndarray, margins: np.ndarray,
                               reference: np.ndarray, tau: float, eps_sel: float,
                               rng: np.random.Generator, n: int, radius: float,
                               slack: float) -> float:
    """KKT-residual-style check: the largest KL *improvement* found among
    small feasible perturbations of ``policy``. Should be ~0 at a true
    constrained local minimum (no feasible descent direction).
    """

    m = policy.size
    base_kl = kl_divergence(policy, reference)
    best_improvement = 0.0
    for _ in range(n):
        direction = rng.normal(size=m)
        direction -= direction.mean()  # tangent to the simplex (sums to 0)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        candidate = _safe(policy + radius * direction / norm)
        if kl_divergence(candidate, reference) > tau + slack:
            continue
        if exact_exploitability(candidate, margins, reference, tau) > eps_sel + slack:
            continue
        improvement = base_kl - kl_divergence(candidate, reference)
        best_improvement = max(best_improvement, improvement)
    return best_improvement


class IndependentAccuracyVerificationTests(unittest.TestCase):
    def _check(self, m: int, tau: float, seed: int) -> None:
        rng = np.random.default_rng(4000 + m)
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)

        # D-509 brief 3: pinned to the raw restart-loop primal explicitly, as
        # in MinKLFeasibilityAndOrderingTests._check_m above.
        primal_policy, primal_gap, _ = solve_constrained_nash(
            margins, reference, tau, seed=seed, selection="restart_legacy",
        )
        eps_sel = max(primal_gap * 10.0, 1e-6)
        selected, selected_gap, diagnostics = solve_constrained_nash(
            margins, reference, tau, seed=seed, selection="min_kl", eps_sel=eps_sel,
        )
        self.assertEqual(diagnostics["min_kl_status"], "ok")
        selected_kl = kl_divergence(selected, reference)

        # (i) dense global sampling -- an independent method, no gradients.
        dense_rng = np.random.default_rng(90000 + m)
        best_dense = _dense_feasible_min_kl(
            margins, reference, tau, eps_sel, dense_rng, n=1500, slack=1e-6
        )
        if best_dense is not None:
            # The grid cannot beat the solver's KL by more than the stated
            # sampling resolution: with n Dirichlet draws on the simplex the
            # nearest sample to any point is O(n^{-1/(m-1)}) away in each
            # coordinate; 5e-2 is a generous bound for n=1500, m<=3.
            self.assertGreaterEqual(best_dense, selected_kl - 5e-2)

        # (ii) local no-feasible-descent (KKT-residual-style) check.
        local_rng = np.random.default_rng(91000 + m)
        improvement = _local_no_feasible_descent(
            selected, margins, reference, tau, eps_sel, local_rng,
            n=400, radius=0.02, slack=1e-6,
        )
        self.assertLessEqual(improvement, 5e-3,
                             f"m={m}: found a feasible local step improving KL by {improvement:.3e}")

    def test_m2(self) -> None:
        self._check(2, 0.5, 0)

    def test_m3(self) -> None:
        self._check(3, 0.5, 1)


# ---------------------------------------------------------------------------
# (4) Constructed degenerate tie game: analytic minimum-KL point of the
#     equilibrium segment, independent of the primary restart's landing point.
# ---------------------------------------------------------------------------


class DegenerateTieSegmentTests(unittest.TestCase):
    def test_primary_lands_at_different_points_on_the_segment(self) -> None:
        """Sanity check that the fixture is actually degenerate: several
        restarts/seeds land at different splits of the tied pair, all with
        (near-)zero exploitability -- the D-508 phenomenon, reproduced by
        construction."""

        splits = []
        for seed in range(4):
            # Explicit restarts expose the equilibrium degeneracy; the
            # production single-start default would not exercise it.
            policy, gap, _ = solve_constrained_nash(
                TIE_MARGINS, TIE_REFERENCE, TIE_TAU, seed=seed, restarts=20,
                selection="restart_legacy",
            )
            self.assertLessEqual(gap, 1e-5)
            splits.append(policy[1] - policy[2])
        self.assertGreater(max(splits) - min(splits), 0.05,
                           "fixture did not exercise the segment; splits too similar")

    def test_min_kl_recovers_analytic_point_independent_of_primary_start(self) -> None:
        for seed in range(6):
            with self.subTest(seed=seed):
                policy, gap, diagnostics = solve_constrained_nash(
                    TIE_MARGINS, TIE_REFERENCE, TIE_TAU, seed=seed, restarts=20,
                    selection="min_kl", eps_sel=1e-6,
                )
                self.assertEqual(diagnostics["min_kl_status"], "ok")
                linf = float(np.max(np.abs(policy - TIE_ANALYTIC_MIN_KL)))
                self.assertLessEqual(linf, 1e-4, f"seed={seed}: Linf {linf:.3e}")

    def test_min_kl_recovers_analytic_point_under_perturbed_starts(self) -> None:
        """Perturb which point on the segment the *primal* restart loop finds
        (by varying both seed and restart count) and confirm min_kl still
        converges to the same analytic point -- it must not just echo the
        primary's own landing spot."""

        for seed, restarts in ((0, 4), (1, 8), (2, 30), (5, 6)):
            with self.subTest(seed=seed, restarts=restarts):
                # Compare minimum-KL selection with the explicitly requested
                # historical restart-loop primal.
                primal_policy, _, _ = solve_constrained_nash(
                    TIE_MARGINS, TIE_REFERENCE, TIE_TAU, seed=seed, restarts=restarts,
                    selection="restart_legacy",
                )
                policy, _, diagnostics = solve_constrained_nash(
                    TIE_MARGINS, TIE_REFERENCE, TIE_TAU, seed=seed, restarts=restarts,
                    selection="min_kl", eps_sel=1e-6,
                )
                self.assertEqual(diagnostics["min_kl_status"], "ok")
                linf_to_analytic = float(np.max(np.abs(policy - TIE_ANALYTIC_MIN_KL)))
                self.assertLessEqual(linf_to_analytic, 1e-4)
                # It must actually have MOVED from an off-center primal split
                # (when the primal landed off-center) toward the balanced one.
                primal_off_center = abs(primal_policy[1] - primal_policy[2])
                selected_off_center = abs(policy[1] - policy[2])
                self.assertLessEqual(selected_off_center, primal_off_center + 1e-4)


# ---------------------------------------------------------------------------
# (5) eps_sel required; eps_sel below the primary gap is recorded, not silent.
# ---------------------------------------------------------------------------


class EpsSelValidationTests(unittest.TestCase):
    """The default selection tolerance is 3e-6.

    Tests temporarily unset it to verify that callers must provide a tolerance
    when no module default is available."""

    def test_module_default_is_brief3_eps_sel(self) -> None:
        # D-509 brief 3: NLHF_SELECTION_EPS is no longer the brief-1 `None`
        # placeholder. Pinned to the exact value the sensitivity/stability
        # studies recommend, citing both reports as nlhf.py's own comment does.
        self.assertEqual(NLHF_SELECTION_EPS, 3e-6)

    def test_raises_when_eps_sel_unset(self) -> None:
        rng = np.random.default_rng(9)
        m = 3
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)
        with mock.patch.object(nlhf_module, "NLHF_SELECTION_EPS", None):
            with self.assertRaises(ValueError):
                solve_constrained_nash(margins, reference, 0.5, seed=1, selection="min_kl")

    def test_raises_when_eps_sel_unset_via_fit_pooled_nlhf(self) -> None:
        rng = np.random.default_rng(10)
        m = 3
        edges = [(a, b) for a in range(m) for b in range(a + 1, m)]
        index = rng.integers(0, len(edges), size=400)
        pairs = np.array([edges[i] for i in index], dtype=np.int64)
        signs = np.where(rng.random(400) < 0.5, 1.0, -1.0)
        with mock.patch.object(nlhf_module, "NLHF_SELECTION_EPS", None):
            with self.assertRaises(ValueError):
                fit_pooled_nlhf(pairs, signs, m, _uniform(m), 0.5, seed=1, selection="min_kl")

    def test_eps_sel_below_primary_gap_is_recorded_and_primary_returned(self) -> None:
        rng = np.random.default_rng(11)
        m = 4
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)
        # D-509 brief 3: the primal restart loop itself does not depend on
        # `selection` (only `_finalize_selection`'s dispatch afterward does),
        # so an explicit "restart_legacy" call reproduces exactly the raw
        # primal that a too-tight eps_sel below falls back to -- unlike a bare
        # call, which now resolves to "min_kl" and would return a different
        # (secondary-refined) policy.
        primal_policy, primal_gap, _ = solve_constrained_nash(
            margins, reference, 0.5, seed=1, selection="restart_legacy",
        )

        too_tight = -1.0  # certainly below any achievable exact gap
        selected, gap, diagnostics = solve_constrained_nash(
            margins, reference, 0.5, seed=1, selection="min_kl", eps_sel=too_tight,
        )
        self.assertTrue(diagnostics["min_kl_status"].startswith(
            "min_kl_failed:eps_sel_below_primary_gap"
        ))
        np.testing.assert_array_equal(selected, primal_policy)
        self.assertEqual(gap, primal_gap)
        self.assertEqual(diagnostics["eps_sel"], too_tight)
        self.assertIsNone(diagnostics["secondary_exact_exploitability"])
        self.assertEqual(diagnostics["secondary_iterations"], 0)

    def test_never_silent_status_is_always_present_for_min_kl(self) -> None:
        rng = np.random.default_rng(12)
        m = 3
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)
        _, _, diagnostics = solve_constrained_nash(
            margins, reference, 0.5, seed=1, selection="min_kl", eps_sel=1e-4,
        )
        self.assertIn("min_kl_status", diagnostics)
        self.assertIn(diagnostics["min_kl_status"], ("ok",)) if diagnostics["min_kl_status"] == "ok"\
            else self.assertTrue(diagnostics["min_kl_status"].startswith("min_kl_failed:"))


# ---------------------------------------------------------------------------
# (6) Held-out isolation: fit_pooled_nlhf's signature and its call into
#     solve_constrained_nash carry only training inputs.
# ---------------------------------------------------------------------------


class HeldOutIsolationTests(unittest.TestCase):
    _ALLOWED_TRAINING_PARAMS = {
        "pair_indices", "signs", "n_alternatives", "reference", "tau", "seed",
        "propensities", "gradient", "selection", "eps_sel",
    }
    _FORBIDDEN_SUBSTRINGS = ("held", "heldout", "held_out", "test_pair", "eval_",
                             "holdout", "oos_", "out_of_sample")

    def test_fit_pooled_nlhf_signature_is_training_only(self) -> None:
        params = list(inspect.signature(fit_pooled_nlhf).parameters)
        self.assertEqual(set(params), self._ALLOWED_TRAINING_PARAMS)
        for name in params:
            lowered = name.lower()
            for forbidden in self._FORBIDDEN_SUBSTRINGS:
                self.assertNotIn(forbidden, lowered, f"parameter {name!r} looks held-out")

    def test_fit_pooled_nlhf_calls_solve_constrained_nash_with_training_args_only(self) -> None:
        source = inspect.getsource(fit_pooled_nlhf)
        match = re.search(r"solve_constrained_nash\(([^)]*)\)", source, re.DOTALL)
        self.assertIsNotNone(match, "could not locate the solve_constrained_nash call site")
        call_args = match.group(1)
        # Every identifier the call site passes must come from fit_pooled_nlhf's
        # own training-only locals/params (margins is derived from
        # pair_indices/signs; everything else is a direct parameter).
        allowed_call_identifiers = self._ALLOWED_TRAINING_PARAMS | {"margins"}
        identifiers = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=", call_args))
        positional_head = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)",
                                   call_args)
        self.assertIsNotNone(positional_head)
        identifiers.update(positional_head.groups())
        for forbidden in self._FORBIDDEN_SUBSTRINGS:
            self.assertNotIn(forbidden, call_args.lower())
        self.assertTrue(identifiers.issubset(allowed_call_identifiers), identifiers)

    def test_solve_constrained_nash_signature_is_training_only(self) -> None:
        params = list(inspect.signature(solve_constrained_nash).parameters)
        allowed = {"margins", "reference", "tau", "restarts", "seed", "gradient",
                   "selection", "eps_sel"}
        self.assertEqual(set(params), allowed)


# ---------------------------------------------------------------------------
# (7) D-509 brief 2b M4: the M2 stages -- canonical-start determinism,
#     polishing (never increases KL / never violates eps_sel), and a
#     regression pinning tie-fold-like stability across starts.
# ---------------------------------------------------------------------------


class CanonicalWarmStartDeterminismTests(unittest.TestCase):
    """M2(i): the canonical start (reference, log(max(tau,1e-3))) does not
    depend on the caller's seed/restart count, so calling
    `_solve_min_kl_robust` with different primal warm starts -- but the same
    (margins, reference, tau, eps_sel) -- must report the *same* canonical
    attempt (status and KL) every time."""

    def test_canonical_attempt_is_independent_of_primal_warm_start(self) -> None:
        rng = np.random.default_rng(500)
        m = 4
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)
        tau = 0.6
        log_reference = np.log(reference)
        primal_gap_seed, _, _ = solve_constrained_nash(margins, reference, tau, seed=0)
        eps_sel = 1e-4

        canonical_kls = []
        for fake_primal_seed in range(5):
            fake_primal = np.random.default_rng(1000 + fake_primal_seed).dirichlet(np.full(m, 2.0))
            fake_primal = _safe(fake_primal)
            _, _, _, _, extra = _solve_min_kl_robust(
                margins, reference, tau, eps_sel, log_reference, m,
                fake_primal, float(np.log(max(tau, 1e-3))),
            )
            canonical_kls.append(extra["min_kl_canonical_kl"])

        for kl in canonical_kls[1:]:
            if canonical_kls[0] is None:
                self.assertIsNone(kl)
            else:
                self.assertAlmostEqual(kl, canonical_kls[0], places=12)


class PolishStageTests(unittest.TestCase):
    """M2(ii): the polish pass never increases KL beyond `_MIN_KL_TIE_MARGIN`
    and, whenever applied, its result still verifies feasible."""

    def _check_m(self, m: int, tau: float, seed: int) -> None:
        rng = np.random.default_rng(6000 + m)
        margins = _random_skew_symmetric(rng, m)
        reference = _uniform(m)
        # D-509 brief 3: pinned to the raw restart-loop primal explicitly (see
        # MinKLFeasibilityAndOrderingTests._check_m) so `primal_gap` is the
        # actual primary gap the eps_sel margin below is computed from,
        # regardless of which selection is the unset default.
        primal_policy, primal_gap, _ = solve_constrained_nash(
            margins, reference, tau, seed=seed, selection="restart_legacy",
        )
        eps_sel = max(primal_gap * 10.0, 1e-6)

        selected, selected_gap, diagnostics = solve_constrained_nash(
            margins, reference, tau, seed=seed, selection="min_kl", eps_sel=eps_sel,
        )
        self.assertEqual(diagnostics["min_kl_status"], "ok")
        for key in ("min_kl_warm_start", "min_kl_primal_status", "min_kl_canonical_status",
                    "min_kl_polish_applied"):
            self.assertIn(key, diagnostics)

        if diagnostics["min_kl_polish_applied"]:
            before = diagnostics["min_kl_polish_kl_before"]
            after = diagnostics["min_kl_polish_kl_after"]
            self.assertIsNotNone(before)
            self.assertIsNotNone(after)
            self.assertLessEqual(after, before + _MIN_KL_TIE_MARGIN,
                                 "polish increased KL beyond the tie margin")
            # The polished (== final selected) point must still verify feasible.
            _assert_feasible(self, selected, margins, reference, tau, eps_sel)
            self.assertAlmostEqual(float(kl_divergence(selected, reference)), after, places=9)

    def test_m2(self) -> None:
        self._check_m(2, 0.5, 0)

    def test_m3(self) -> None:
        self._check_m(3, 0.5, 1)

    def test_m4(self) -> None:
        self._check_m(4, 0.75, 2)

    def test_m5(self) -> None:
        self._check_m(5, 1.0, 3)

    def test_tie_game_polish_never_regresses(self) -> None:
        for seed in range(6):
            with self.subTest(seed=seed):
                _, _, diagnostics = solve_constrained_nash(
                    TIE_MARGINS, TIE_REFERENCE, TIE_TAU, seed=seed, restarts=20,
                    selection="min_kl", eps_sel=1e-6,
                )
                self.assertEqual(diagnostics["min_kl_status"], "ok")
                if diagnostics["min_kl_polish_applied"]:
                    self.assertLessEqual(
                        diagnostics["min_kl_polish_kl_after"],
                        diagnostics["min_kl_polish_kl_before"] + _MIN_KL_TIE_MARGIN,
                    )


class TieFoldStabilityRegressionTests(unittest.TestCase):
    """Check stability on a constructed tie game across seeds and restart counts.

    The expected policy spread is at most 1e-8 on this synthetic fixture.
    This does not establish the same bound for real-data folds."""

    _CASES = [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (0, 8), (1, 20), (2, 30), (5, 6), (0, 50)]

    def test_stable_within_1e8_across_seeds_and_restarts(self) -> None:
        policies = []
        for seed, restarts in self._CASES:
            policy, _, diagnostics = solve_constrained_nash(
                TIE_MARGINS, TIE_REFERENCE, TIE_TAU, seed=seed, restarts=restarts,
                selection="min_kl", eps_sel=1e-6,
            )
            self.assertEqual(diagnostics["min_kl_status"], "ok")
            policies.append(policy)

        arr = np.array(policies)
        max_linf = 0.0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                max_linf = max(max_linf, float(np.max(np.abs(arr[i] - arr[j]))))
        self.assertLessEqual(max_linf, 1e-8,
                             f"tie-game stability regressed: max Linf {max_linf:.3e}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
