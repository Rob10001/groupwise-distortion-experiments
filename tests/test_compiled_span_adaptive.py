"""End-to-end differential test: ``fit_groupwise_span_adaptive`` must produce
bit-identical output whether ``kernel_backend="python"`` (the reference
implementation and explicit fallback) or ``"numba"`` (the compiled
per-group inner-update kernel of
``compiled_kernels.span_adaptive_inner_round_numba``, wired in
``span_adaptive.py``, and the CPU production **default** since D-502) is
selected.

Skips entirely (via ``pytest.importorskip``) in the baseline venv, which has
no numba installed and can therefore only ever select ``"python"``
explicitly -- ``TestBackendResolution`` covers what that venv's default
path does (raise, not silently fall back) in ``tests/test_compiled_span_adaptive.py``
by monkeypatching numba's absence rather than depending on the venv itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from empirical_testing import compiled_kernels  # noqa: E402
from empirical_testing.algorithms.span_adaptive import (  # noqa: E402
    FrozenRates,
    fit_groupwise_span_adaptive,
)


def same_bits(left, right) -> bool:
    """Exact ``uint64``-view equality (NaN-safe), as used throughout this suite."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _dense_active_config(T=400, K=4, m=4, seed=0):
    """K >= 3 groups, group 0 universal, others random membership so several
    rounds have more than one active group; utilities in [0, 1] for the
    "active" protocol."""

    rng = np.random.default_rng(seed)
    groups = np.ones((T, K), dtype=np.int8)
    groups[:, 1:] = rng.integers(0, 2, size=(T, K - 1))
    utilities = rng.uniform(size=(T, m))
    return groups, utilities


def _conflicting_active_config(T=350, K=3, m=4, seed=1):
    """K >= 3, group 0 universal, group 1 alternating, group 2 on a
    different period (so many rounds activate more than one group); the
    preference utilities invert sharply whenever group 1 is present, so the
    universal expert is pulled in opposite directions round to round -- like
    ``tests/test_span_adaptive.py``'s ``_conflicting_construction`` but for
    the utility/beta-driven "active" protocol.  This reliably drives the KL
    projection to bind on a large fraction of updates, generalized to
    ``beta`` finite or ``+inf``."""

    groups = np.ones((T, K), dtype=np.int8)
    groups[:, 1] = (np.arange(T) % 2 == 1).astype(np.int8)
    if K > 2:
        groups[:, 2] = (np.arange(T) % 3 == 0).astype(np.int8)
    utilities = np.zeros((T, m))
    utilities[:, 0] = 0.9
    utilities[:, 1] = 0.1
    mask = groups[:, 1] == 1
    utilities[mask, 0] = 0.1
    utilities[mask, 1] = 0.9
    return groups, utilities


def _single_active_group_config(T=280, K=4, m=6, seed=2):
    """Exactly one active group per round (round-robin) -- the ``G=1``
    vecmat/matvec shape where NumPy's BLAS dispatch could disagree with a naive replica
    (worst at ``m=20``). The kernels here never replicate those dot/matmul
    ops (they stay literal Python NumPy per the brief's hard rule); this
    checks the surrounding new kernels are still correct at that shape."""

    groups = np.zeros((T, K), dtype=np.int8)
    for t in range(T):
        groups[t, t % K] = 1
    rng = np.random.default_rng(seed)
    utilities = rng.uniform(size=(T, m))
    return groups, utilities


def _dense_logged_config(T=400, K=4, m=4, seed=0):
    """Same group structure, but the "logged" protocol with fixed uniform
    propensities -- exercises the ``span_checked`` branch differently and is
    cheap for the frozen-rate hook, which needs no ``utilities``."""

    rng = np.random.default_rng(seed)
    groups = np.ones((T, K), dtype=np.int8)
    groups[:, 1:] = rng.integers(0, 2, size=(T, K - 1))
    pairs = np.array([sorted(rng.choice(m, 2, replace=False)) for _ in range(T)], dtype=np.int64)
    signs = rng.choice([-1.0, 1.0], size=T)
    n_pairs = m * (m - 1) // 2
    props = np.full(T, 1.0 / n_pairs)
    return groups, pairs, signs, props


def _assert_fits_bit_identical(py_model, nb_model, sample_groups) -> None:
    assert len(py_model._outer_log_mass_snapshots) == len(nb_model._outer_log_mass_snapshots)
    for i, (a, b) in enumerate(zip(py_model._outer_log_mass_snapshots, nb_model._outer_log_mass_snapshots)):
        assert same_bits(a, b), f"outer snapshot {i} mismatch"
    for i, (a, b) in enumerate(zip(py_model._inner_policy_snapshots, nb_model._inner_policy_snapshots)):
        assert same_bits(a, b), f"inner snapshot {i} mismatch"

    assert same_bits(py_model.final_outer_log_masses, nb_model.final_outer_log_masses)
    assert same_bits(py_model.outer_second_moments, nb_model.outer_second_moments)
    assert same_bits(py_model.centered_inner_second_moments, nb_model.centered_inner_second_moments)
    assert same_bits(py_model.inner_second_moments, nb_model.inner_second_moments)
    assert np.array_equal(py_model.group_activation_counts, nb_model.group_activation_counts)
    assert np.array_equal(py_model.drawn_comparisons, nb_model.drawn_comparisons) or (
        np.isnan(py_model.drawn_comparisons) == np.isnan(nb_model.drawn_comparisons)
    ).all()

    py_pred = py_model.predict(sample_groups)
    nb_pred = nb_model.predict(sample_groups)
    assert same_bits(py_pred, nb_pred), "predict() mismatch"

    py_diag = py_model.diagnostics.as_json()
    nb_diag = nb_model.diagnostics.as_json()
    assert py_diag.keys() == nb_diag.keys()
    for key in py_diag:
        if key == "kernel_backend":
            assert py_diag[key] == "python"
            assert nb_diag[key] == "numba"
            continue
        pv, nv = py_diag[key], nb_diag[key]
        if isinstance(pv, float):
            assert same_bits(pv, nv), f"diagnostics field {key!r} mismatch: {pv!r} vs {nv!r}"
        elif isinstance(pv, list) and pv and isinstance(pv[0], float):
            assert same_bits(np.asarray(pv), np.asarray(nv)), f"diagnostics field {key!r} mismatch"
        elif isinstance(pv, dict):
            for subkey in pv:
                assert same_bits(np.asarray(pv[subkey], dtype=np.float64), np.asarray(nv[subkey], dtype=np.float64)), (
                    f"rate_trace field {subkey!r} mismatch"
                )
        else:
            assert pv == nv, f"diagnostics field {key!r} mismatch: {pv!r} vs {nv!r}"


class TestBindsOftenSmallTau:
    """Small tau: the KL projection binds on most updates (the brentq /
    STATUS_BINDING path is exercised heavily)."""

    def test_active_protocol_finite_beta(self) -> None:
        groups, utilities = _conflicting_active_config(T=350, K=3, m=4, seed=1)
        kwargs = dict(
            groups=groups, utilities=utilities, beta=3.0, reference=np.full(4, 0.25),
            tau=0.01, protocol="active", seed=11,
        )
        py_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="python")
        compiled_kernels.reset_binding_kernel_hits()
        nb_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="numba")
        assert np.any(nb_model.diagnostics.projection_counts > 0), "fixture produced no binding case"
        # M4: the compiled KL-projection path (compiled_kernels.
        # kl_prox_binding_numba), not just the feasible-path kernel, must
        # actually have run for this binds-often fixture.
        assert compiled_kernels.get_binding_kernel_hits() > 0, (
            "compiled binding path was not exercised by this fit"
        )
        _assert_fits_bit_identical(py_model, nb_model, groups[:6])

    def test_active_protocol_beta_infinite(self) -> None:
        groups, utilities = _conflicting_active_config(T=300, K=3, m=4, seed=1)
        kwargs = dict(
            groups=groups, utilities=utilities, beta=np.inf, reference=np.full(4, 0.25),
            tau=0.02, protocol="active", seed=23,
        )
        py_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="python")
        compiled_kernels.reset_binding_kernel_hits()
        nb_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="numba")
        assert np.any(nb_model.diagnostics.projection_counts > 0), "fixture produced no binding case"
        assert compiled_kernels.get_binding_kernel_hits() > 0, (
            "compiled binding path was not exercised by this fit"
        )
        _assert_fits_bit_identical(py_model, nb_model, groups[:6])


class TestNeverBindsLargeTau:
    """Large tau: the KL ball is loose enough that no update should bind
    (STATUS_OK covers essentially every group)."""

    def test_active_protocol_finite_beta(self) -> None:
        groups, utilities = _dense_active_config(T=350, K=4, m=4, seed=3)
        kwargs = dict(
            groups=groups, utilities=utilities, beta=2.0, reference=np.full(4, 0.25),
            tau=5.0, protocol="active", seed=31,
        )
        py_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="python")
        nb_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="numba")
        assert np.all(nb_model.diagnostics.projection_counts == 0), "fixture unexpectedly bound"
        _assert_fits_bit_identical(py_model, nb_model, groups[:6])

    def test_logged_protocol_multi_group_rounds(self) -> None:
        groups, pairs, signs, props = _dense_logged_config(T=400, K=5, m=4, seed=4)
        kwargs = dict(
            groups=groups, reference=np.full(4, 0.25), tau=3.0, protocol="logged",
            pair_indices=pairs, signs=signs, propensities=props, seed=41,
        )
        assert np.sum(groups.sum(axis=1) > 1) > 0, "fixture produced no multi-active-group round"
        py_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="python")
        nb_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="numba")
        _assert_fits_bit_identical(py_model, nb_model, groups[:6])


class TestFrozenRates:
    """Cheap regression hook: constant rates, still must match under numba."""

    def test_frozen_rates_match(self) -> None:
        groups, pairs, signs, props = _dense_logged_config(T=200, K=3, m=3, seed=5)
        frozen = FrozenRates(outer=np.full(3, 0.1), inner=np.full(3, 0.05))
        kwargs = dict(
            groups=groups, reference=np.full(3, 1.0 / 3.0), tau=0.05, protocol="logged",
            pair_indices=pairs, signs=signs, propensities=props, seed=51,
            frozen_rates=frozen,
        )
        py_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="python")
        nb_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="numba")
        _assert_fits_bit_identical(py_model, nb_model, groups[:6])


class TestLinalgProbeShapes:
    """M4 (c1d brief): m=5, m=20, and an exactly-one-active-group-per-round
    pattern -- shapes where NumPy's dot/matmul dispatch can differ from naive loops.  The
    new kernels never touch those ops (hard rule), so these check the
    orchestration around them is still bit-identical at these shapes."""

    def test_m5_dense_active(self) -> None:
        groups, utilities = _dense_active_config(T=300, K=4, m=5, seed=12)
        kwargs = dict(
            groups=groups, utilities=utilities, beta=2.5, reference=np.full(5, 0.2),
            tau=0.05, protocol="active", seed=121,
        )
        py_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="python")
        nb_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="numba")
        _assert_fits_bit_identical(py_model, nb_model, groups[:6])

    def test_m20_dense_active(self) -> None:
        groups, utilities = _dense_active_config(T=250, K=3, m=20, seed=13)
        kwargs = dict(
            groups=groups, utilities=utilities, beta=1.5, reference=np.full(20, 1.0 / 20),
            tau=0.3, protocol="active", seed=131,
        )
        py_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="python")
        nb_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="numba")
        _assert_fits_bit_identical(py_model, nb_model, groups[:6])

    def test_single_active_group_per_round_m20(self) -> None:
        groups, utilities = _single_active_group_config(T=280, K=4, m=20, seed=14)
        assert np.all(groups.sum(axis=1) == 1), "fixture must activate exactly one group per round"
        kwargs = dict(
            groups=groups, utilities=utilities, beta=2.0, reference=np.full(20, 1.0 / 20),
            tau=0.1, protocol="active", seed=141,
        )
        py_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="python")
        nb_model = fit_groupwise_span_adaptive(**kwargs, kernel_backend="numba")
        _assert_fits_bit_identical(py_model, nb_model, groups[:6])


class TestBackendResolution:
    def test_default_is_numba(self, monkeypatch) -> None:
        """D-502: with no explicit argument and no env var, the default is numba."""

        monkeypatch.delenv("GWD_CPU_KERNELS", raising=False)
        groups, pairs, signs, props = _dense_logged_config(T=50, K=2, m=3, seed=6)
        model = fit_groupwise_span_adaptive(
            groups, reference=np.full(3, 1.0 / 3.0), tau=0.1, protocol="logged",
            pair_indices=pairs, signs=signs, propensities=props, seed=61,
        )
        assert model.diagnostics.kernel_backend == "numba"

    def test_env_var_selects_numba(self, monkeypatch) -> None:
        monkeypatch.setenv("GWD_CPU_KERNELS", "numba")
        groups, pairs, signs, props = _dense_logged_config(T=50, K=2, m=3, seed=7)
        model = fit_groupwise_span_adaptive(
            groups, reference=np.full(3, 1.0 / 3.0), tau=0.1, protocol="logged",
            pair_indices=pairs, signs=signs, propensities=props, seed=71,
        )
        assert model.diagnostics.kernel_backend == "numba"

    def test_env_var_selects_python(self, monkeypatch) -> None:
        """D-502: "python" stays available explicitly via GWD_CPU_KERNELS."""

        monkeypatch.setenv("GWD_CPU_KERNELS", "python")
        groups, pairs, signs, props = _dense_logged_config(T=50, K=2, m=3, seed=12)
        model = fit_groupwise_span_adaptive(
            groups, reference=np.full(3, 1.0 / 3.0), tau=0.1, protocol="logged",
            pair_indices=pairs, signs=signs, propensities=props, seed=121,
        )
        assert model.diagnostics.kernel_backend == "python"

    def test_explicit_argument_python_overrides_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("GWD_CPU_KERNELS", "numba")
        groups, pairs, signs, props = _dense_logged_config(T=50, K=2, m=3, seed=8)
        model = fit_groupwise_span_adaptive(
            groups, reference=np.full(3, 1.0 / 3.0), tau=0.1, protocol="logged",
            pair_indices=pairs, signs=signs, propensities=props, seed=81,
            kernel_backend="python",
        )
        assert model.diagnostics.kernel_backend == "python"

    def test_explicit_argument_numba_overrides_env_var(self, monkeypatch) -> None:
        """D-502: an explicit "numba" argument wins even if the env var says python."""

        monkeypatch.setenv("GWD_CPU_KERNELS", "python")
        groups, pairs, signs, props = _dense_logged_config(T=50, K=2, m=3, seed=13)
        model = fit_groupwise_span_adaptive(
            groups, reference=np.full(3, 1.0 / 3.0), tau=0.1, protocol="logged",
            pair_indices=pairs, signs=signs, propensities=props, seed=131,
            kernel_backend="numba",
        )
        assert model.diagnostics.kernel_backend == "numba"

    def test_unknown_backend_raises_value_error(self) -> None:
        groups, pairs, signs, props = _dense_logged_config(T=20, K=1, m=2, seed=9)
        with pytest.raises(ValueError):
            fit_groupwise_span_adaptive(
                groups, reference=np.full(2, 0.5), tau=0.1, protocol="logged",
                pair_indices=pairs, signs=signs, propensities=props, seed=91,
                kernel_backend="not-a-real-backend",
            )

    def test_requesting_numba_without_numba_raises(self, monkeypatch) -> None:
        import empirical_testing.compiled_kernels as compiled_kernels

        monkeypatch.setattr(compiled_kernels, "_njit", None)
        groups, pairs, signs, props = _dense_logged_config(T=20, K=1, m=2, seed=10)
        with pytest.raises(RuntimeError):
            fit_groupwise_span_adaptive(
                groups, reference=np.full(2, 0.5), tau=0.1, protocol="logged",
                pair_indices=pairs, signs=signs, propensities=props, seed=101,
                kernel_backend="numba",
            )

    def test_missing_numba_raises_with_remedy_text_on_default_path(self, monkeypatch) -> None:
        """D-502: the default (no argument, no env var) must fail loudly, not

        silently fall back to python, when numba is unavailable -- and the
        error must name both remedies (install numba, or select "python").
        """

        import empirical_testing.compiled_kernels as compiled_kernels

        monkeypatch.setattr(compiled_kernels, "_njit", None)
        monkeypatch.delenv("GWD_CPU_KERNELS", raising=False)
        groups, pairs, signs, props = _dense_logged_config(T=20, K=1, m=2, seed=11)
        with pytest.raises(RuntimeError) as excinfo:
            fit_groupwise_span_adaptive(
                groups, reference=np.full(2, 0.5), tau=0.1, protocol="logged",
                pair_indices=pairs, signs=signs, propensities=props, seed=111,
            )
        message = str(excinfo.value)
        assert "numba" in message and "install" in message.lower()
        assert "python" in message.lower()
