"""Differential tests: independent reference vs production, through backends.

Tolerances are 1e-12 absolute.  A larger discrepancy is a finding against the
fidelity contract, not a reason to loosen the gate.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from empirical_testing.backends import (  # noqa: E402
    BackendUnavailableError,
    list_backends,
    resolve_backend,
)

TOL = 1e-12
REFERENCE_SOURCE = (
    PROJECT_ROOT / "src" / "empirical_testing" / "reference" / "span_adaptive_reference.py"
)
FORBIDDEN_IMPORT_ROOTS = (
    "empirical_testing.algorithms.span_adaptive",
    "empirical_testing.algorithms.one_expert",
    "empirical_testing.numerics",
)


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _logged_data(T, K, m, seed, *, empty_rounds=(), uniform_propensity=True, ref=None):
    rng = np.random.default_rng(seed)
    groups = np.ones((T, K), dtype=np.int8)
    if K > 1:
        groups[:, 1:] = rng.integers(0, 2, size=(T, K - 1))
    for t in empty_rounds:
        groups[int(t)] = 0
    pairs = np.empty((T, 2), dtype=np.int64)
    for t in range(T):
        pairs[t] = np.sort(rng.choice(m, size=2, replace=False))
    signs = rng.choice([-1.0, 1.0], size=T)
    n_pairs = m * (m - 1) // 2
    props = np.full(T, 1.0 / n_pairs) if uniform_propensity else rng.uniform(0.05, 0.5, size=T)
    if ref is None:
        ref = np.full(m, 1.0 / m)
    return groups, pairs, signs, props, np.asarray(ref, dtype=np.float64)


def _fit_both(groups, *, utilities=None, beta=1.0, reference=None, tau=0.5, protocol="logged",
              seed=0, pair_indices=None, signs=None, propensities=None):
    kwargs = dict(
        groups=groups,
        utilities=utilities,
        beta=beta,
        reference=reference,
        tau=tau,
        protocol=protocol,
        seed=seed,
        pair_indices=pair_indices,
        signs=signs,
        propensities=propensities,
    )
    reference_fit = resolve_backend("cpu-reference").fit_span_adaptive(**kwargs)
    production_fit = resolve_backend("cpu-production").fit_span_adaptive(**kwargs)
    return reference_fit, production_fit


def _predict_patterns(K: int) -> np.ndarray:
    extra = [np.zeros(K, dtype=np.float64), np.ones(K, dtype=np.float64)]
    extra.extend(np.eye(K, dtype=np.float64))
    return np.vstack(extra)


class IndependenceTests(unittest.TestCase):
    def test_reference_source_does_not_import_implementations_under_test(self) -> None:
        tree = ast.parse(REFERENCE_SOURCE.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        offenders = [
            name
            for name in imported
            if name in FORBIDDEN_IMPORT_ROOTS
            or any(name.startswith(root + ".") for root in FORBIDDEN_IMPORT_ROOTS)
        ]
        self.assertEqual(offenders, [], f"reference imported implementations under test: {offenders}")
        self.assertTrue(any(name == "numpy" or name.startswith("numpy.") for name in imported))
        self.assertTrue(any(name == "scipy" or name.startswith("scipy.") for name in imported))


class BackendSelectionTests(unittest.TestCase):
    def test_unavailable_backend_raises_and_does_not_fall_back(self) -> None:
        available_before = list_backends()
        self.assertEqual(available_before, ("cpu-reference", "cpu-production"))
        with self.assertRaises(BackendUnavailableError) as unknown:
            resolve_backend("not-a-backend")
        self.assertIn("never falls back", str(unknown.exception).lower())
        available_after = list_backends()
        self.assertEqual(available_before, available_after)

    def test_provenance_exposes_backend_id_and_module_path(self) -> None:
        reference = resolve_backend("cpu-reference")
        production = resolve_backend("cpu-production")
        self.assertEqual(reference.provenance()["backend_id"], "cpu-reference")
        self.assertEqual(production.provenance()["backend_id"], "cpu-production")
        self.assertTrue(reference.loaded_module_path.endswith("span_adaptive_reference.py"))
        self.assertTrue(production.loaded_module_path.endswith("span_adaptive.py"))
        self.assertNotEqual(reference.loaded_module_path, production.loaded_module_path)


class _DifferentialMixin:
    discrepancies: dict[str, float]

    def _record_and_assert(self, label: str, left, right) -> float:
        gap = _max_abs(left, right)
        self.discrepancies[label] = gap
        self.assertLessEqual(
            gap,
            TOL,
            f"{label} disagreed by {gap:.3e} > {TOL} (finding: investigate against the contract)",
        )
        return gap

    def _compare_fits(self, name: str, groups, ref_fit, prod_fit, extra_groups=None) -> dict[str, float]:
        self.discrepancies = {}
        self._record_and_assert(f"{name}.outer_snapshots", ref_fit.outer_log_mass_snapshots, prod_fit.outer_log_mass_snapshots)
        self._record_and_assert(f"{name}.inner_snapshots", ref_fit.inner_policy_snapshots, prod_fit.inner_policy_snapshots)
        self._record_and_assert(f"{name}.Q", ref_fit.Q, prod_fit.Q)
        self._record_and_assert(f"{name}.C", ref_fit.C, prod_fit.C)
        self._record_and_assert(f"{name}.V", ref_fit.V, prod_fit.V)
        ref_backend = resolve_backend("cpu-reference")
        prod_backend = resolve_backend("cpu-production")
        self._record_and_assert(
            f"{name}.predict_train",
            ref_backend.predict(ref_fit, groups),
            prod_backend.predict(prod_fit, groups),
        )
        self._record_and_assert(f"{name}.averaged_policy", ref_fit.averaged_policy, prod_fit.averaged_policy)
        if extra_groups is not None:
            self._record_and_assert(
                f"{name}.predict_extra",
                ref_backend.predict(ref_fit, extra_groups),
                prod_backend.predict(prod_fit, extra_groups),
            )
        print(f"DISCREPANCIES {name}: {self.discrepancies}")
        return dict(self.discrepancies)


class LoggedProtocolTests(_DifferentialMixin, unittest.TestCase):
    def test_logged_small_case(self) -> None:
        groups, pairs, signs, props, ref = _logged_data(T=24, K=3, m=3, seed=1)
        ref_fit, prod_fit = _fit_both(
            groups, reference=ref, tau=0.4, protocol="logged", seed=3,
            pair_indices=pairs, signs=signs, propensities=props,
        )
        self._compare_fits("logged_small", groups, ref_fit, prod_fit, _predict_patterns(3))

    def test_logged_production_scale(self) -> None:
        groups, pairs, signs, props, ref = _logged_data(T=3000, K=5, m=4, seed=2)
        ref_fit, prod_fit = _fit_both(
            groups, reference=ref, tau=0.5, protocol="logged", seed=9,
            pair_indices=pairs, signs=signs, propensities=props,
        )
        self._compare_fits("logged_T3000", groups, ref_fit, prod_fit, _predict_patterns(5))


class ActiveProtocolTests(_DifferentialMixin, unittest.TestCase):
    def test_active_same_seed(self) -> None:
        rng = np.random.default_rng(4)
        T, K, m = 200, 4, 4
        groups = np.ones((T, K), dtype=np.int8)
        groups[:, 1:] = rng.integers(0, 2, size=(T, K - 1))
        utilities = rng.uniform(size=(T, m))
        ref = np.full(m, 1.0 / m)
        seed = 11
        ref_fit, prod_fit = _fit_both(
            groups, utilities=utilities, beta=4.0, reference=ref, tau=0.5,
            protocol="active", seed=seed,
        )
        if ref_fit.drawn_comparisons.shape != prod_fit.drawn_comparisons.shape:
            self.fail(
                "FINDING: active comparison-log shapes differ "
                f"{ref_fit.drawn_comparisons.shape} vs {prod_fit.drawn_comparisons.shape}; "
                "draws did not coincide under the shared seed derivation"
            )
        pair_gap = _max_abs(ref_fit.drawn_comparisons, prod_fit.drawn_comparisons)
        if not np.array_equal(
            np.isnan(ref_fit.drawn_comparisons), np.isnan(prod_fit.drawn_comparisons)
        ) or pair_gap > TOL:
            first = None
            for t in range(ref_fit.drawn_comparisons.shape[0]):
                if not np.allclose(
                    ref_fit.drawn_comparisons[t],
                    prod_fit.drawn_comparisons[t],
                    atol=TOL,
                    rtol=0.0,
                    equal_nan=True,
                ):
                    first = t
                    break
            self.fail(
                "FINDING: active draws did not coincide under contract §8.1 "
                f"(first differing round={first}, max abs={pair_gap:.3e}). "
                f"reference row={ref_fit.drawn_comparisons[first] if first is not None else None}; "
                f"production row={prod_fit.drawn_comparisons[first] if first is not None else None}"
            )
        self._compare_fits("active_seed11", groups, ref_fit, prod_fit, _predict_patterns(K))


class DegenerateCaseTests(_DifferentialMixin, unittest.TestCase):
    def test_tau_zero_returns_reference_and_consumes_nothing(self) -> None:
        groups, pairs, signs, props, ref = _logged_data(T=40, K=3, m=4, seed=5)
        ref_fit, prod_fit = _fit_both(
            groups, reference=ref, tau=0.0, protocol="logged", seed=0,
            pair_indices=pairs, signs=signs, propensities=props,
        )
        self.assertEqual(ref_fit.drawn_comparisons.shape, (0, 4))
        self.assertEqual(prod_fit.drawn_comparisons.shape, (0, 4))
        self.assertTrue(np.allclose(ref_fit.Q, 0.0) and np.allclose(ref_fit.C, 0.0) and np.allclose(ref_fit.V, 0.0))
        for t in range(ref_fit.inner_policy_snapshots.shape[0]):
            np.testing.assert_allclose(ref_fit.inner_policy_snapshots[t], ref_fit.inner_policy_snapshots[0])
        self._compare_fits("degenerate_tau0", groups, ref_fit, prod_fit, _predict_patterns(3))

    def test_round_with_no_active_group(self) -> None:
        groups, pairs, signs, props, ref = _logged_data(
            T=30, K=3, m=3, seed=6, empty_rounds=(0, 7, 29)
        )
        self.assertTrue(np.all(groups[0] == 0) and np.all(groups[7] == 0) and np.all(groups[29] == 0))
        ref_fit, prod_fit = _fit_both(
            groups, reference=ref, tau=0.35, protocol="logged", seed=1,
            pair_indices=pairs, signs=signs, propensities=props,
        )
        self.assertTrue(np.isnan(ref_fit.drawn_comparisons[0]).all())
        self.assertTrue(np.isnan(ref_fit.drawn_comparisons[7]).all())
        self._compare_fits("degenerate_inactive", groups, ref_fit, prod_fit, _predict_patterns(3))

    def test_k_equals_one(self) -> None:
        groups, pairs, signs, props, ref = _logged_data(T=50, K=1, m=3, seed=7)
        ref_fit, prod_fit = _fit_both(
            groups, reference=ref, tau=0.4, protocol="logged", seed=2,
            pair_indices=pairs, signs=signs, propensities=props,
        )
        self._compare_fits("degenerate_K1", groups, ref_fit, prod_fit, _predict_patterns(1))

    def test_nonuniform_reference(self) -> None:
        ref = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
        groups, pairs, signs, props, _ = _logged_data(T=80, K=3, m=4, seed=8, ref=ref)
        ref_fit, prod_fit = _fit_both(
            groups, reference=ref, tau=0.25, protocol="logged", seed=3,
            pair_indices=pairs, signs=signs, propensities=props,
        )
        self._compare_fits("degenerate_nonuniform_ref", groups, ref_fit, prod_fit, _predict_patterns(3))

    def test_tau_at_or_above_radius_cap(self) -> None:
        ref = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
        cap = float(np.log(1.0 / float(np.min(ref))))
        groups, pairs, signs, props, _ = _logged_data(T=60, K=3, m=4, seed=9, ref=ref)
        ref_fit, prod_fit = _fit_both(
            groups, reference=ref, tau=cap, protocol="logged", seed=4,
            pair_indices=pairs, signs=signs, propensities=props,
        )
        self._compare_fits("degenerate_tau_at_cap", groups, ref_fit, prod_fit, _predict_patterns(3))


if __name__ == "__main__":
    unittest.main()
