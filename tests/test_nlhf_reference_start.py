"""Suite acceptance: NLHF fits make exactly one reference-start optimization.

The production baseline is ``reference_start_v10``: a single SLSQP call
initialized at the reference policy, with no random restarts, no minimum-KL
secondary selection, no polishing pass and no automatic retry.  The historical
rules remain reachable *by explicit name* for reproduction, and must not be
reachable any other way -- including through environment-variable leakage.
"""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

from empirical_testing.algorithms import nlhf
from empirical_testing.algorithms.nlhf import (
    EXPLOITABILITY_TOLERANCE,
    NLHF_BASELINE_VERSION_LEGACY,
    NLHF_BASELINE_VERSION_MIN_KL,
    NLHF_BASELINE_VERSION_REFERENCE_START,
    fit_pooled_nlhf,
    solve_constrained_nash,
)


def _problem(m: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, m))
    return a - a.T, np.full(m, 1.0 / m)


def _count_optimizer_calls(**kwargs):
    """Solve while counting real SLSQP invocations."""

    real = nlhf.minimize
    calls = []

    def counting(*args, **call_kwargs):
        calls.append(1)
        return real(*args, **call_kwargs)

    with mock.patch.object(nlhf, "minimize", side_effect=counting):
        policy, gap, diagnostics = solve_constrained_nash(**kwargs)
    return len(calls), policy, gap, diagnostics


# ------------------------------------------------------------------ the default

def test_the_default_is_the_suite_reference_start_baseline():
    margins, reference = _problem()
    _, _, _, diagnostics = _count_optimizer_calls(margins=margins, reference=reference, tau=0.5)
    assert diagnostics["selection"] == NLHF_BASELINE_VERSION_REFERENCE_START
    assert diagnostics["baseline_version"] == NLHF_BASELINE_VERSION_REFERENCE_START


@pytest.mark.parametrize("m", [3, 4, 5, 8])
@pytest.mark.parametrize("tau", [0.2, 0.5, 1.0])
def test_exactly_one_primary_optimization_and_no_secondary_solve(m, tau):
    margins, reference = _problem(m=m, seed=m)
    calls, policy, gap, diagnostics = _count_optimizer_calls(
        margins=margins, reference=reference, tau=tau
    )
    assert calls == 1, f"{calls} optimizer calls; the production baseline makes exactly one"
    assert diagnostics["optimizer_calls"] == 1
    assert diagnostics["restarts_used"] == 1
    # No trace of the secondary minimum-KL machinery.
    assert not [key for key in diagnostics if key.startswith("min_kl")]
    assert "eps_sel" not in diagnostics
    assert np.isclose(policy.sum(), 1.0)


def test_the_single_call_starts_at_the_reference_policy():
    margins, reference = _problem()
    seen = {}
    real = nlhf.minimize

    def capture(objective, start, *args, **kwargs):
        seen["start"] = np.asarray(start).copy()
        return real(objective, start, *args, **kwargs)

    with mock.patch.object(nlhf, "minimize", side_effect=capture):
        _, _, diagnostics = solve_constrained_nash(margins, reference, 0.5)
    m = reference.size
    assert np.array_equal(seen["start"][:m], reference)
    # The existing initial auxiliary-scale convention, unchanged.
    assert seen["start"][m] == pytest.approx(np.log(max(0.5, 1e-3)))
    assert diagnostics["initialization"] == "reference_policy"


def test_restarts_argument_cannot_reintroduce_multiple_starts():
    margins, reference = _problem()
    for restarts in (0, 1, 4, 50):
        calls, _, _, diagnostics = _count_optimizer_calls(
            margins=margins, reference=reference, tau=0.5, restarts=restarts
        )
        assert calls == 1, f"restarts={restarts} produced {calls} optimizer calls"
        assert diagnostics["optimizer_calls"] == 1


def test_a_zero_radius_ball_makes_no_optimizer_call_at_all():
    """B_tau is the single point {reference}; the optimum needs no solving."""

    margins, reference = _problem()
    calls, policy, gap, diagnostics = _count_optimizer_calls(
        margins=margins, reference=reference, tau=0.0
    )
    assert calls == 0
    assert diagnostics["optimizer_calls"] == 0
    assert np.array_equal(policy, reference)
    assert diagnostics["selection"] == NLHF_BASELINE_VERSION_REFERENCE_START


# ------------------------------------------------------------------ diagnostics

def test_the_recorded_diagnostics_expose_the_whole_resolved_configuration():
    margins, reference = _problem()
    _, _, gap, diagnostics = _count_optimizer_calls(margins=margins, reference=reference, tau=0.5)
    for key in ("selection", "baseline_version", "optimizer_calls", "initialization",
                "gradient_mode", "solver_success", "solver_status", "final_kl",
                "exact_exploitability"):
        assert key in diagnostics, f"diagnostics do not record {key!r}"
    # The derivative mode is explicitly selected and recorded, not implied.
    assert diagnostics["gradient_mode"] == "finite_difference"
    assert diagnostics["exact_exploitability"] == pytest.approx(gap)
    assert diagnostics["final_kl"] <= 0.5 + 1e-9


def test_verification_still_holds_finite_normalized_feasible_and_exact():
    margins, reference = _problem(m=6, seed=11)
    _, policy, gap, diagnostics = _count_optimizer_calls(
        margins=margins, reference=reference, tau=0.5
    )
    assert np.all(np.isfinite(policy)) and np.isfinite(gap)
    assert np.all(policy >= 0.0)
    assert policy.sum() == pytest.approx(1.0, abs=1e-9)
    assert diagnostics["final_kl"] <= 0.5 + 1e-9
    # The independent exact-exploitability check, against the existing gate.
    assert abs(gap) < EXPLOITABILITY_TOLERANCE


# ------------------------------------------------- historical modes stay walled off

@pytest.mark.parametrize("selection", [NLHF_BASELINE_VERSION_LEGACY, "min_kl"])
def test_a_historical_mode_runs_only_when_named_explicitly(selection):
    margins, reference = _problem()
    calls, _, _, diagnostics = _count_optimizer_calls(
        margins=margins, reference=reference, tau=0.5, selection=selection, restarts=4
    )
    assert calls > 1, "the historical multi-start paths should still multi-start"
    assert diagnostics["selection"] == selection


@pytest.mark.parametrize("leaked", ["min_kl", NLHF_BASELINE_VERSION_LEGACY])
def test_the_environment_cannot_select_a_non_production_rule(monkeypatch, leaked):
    """A stray GWD_NLHF_SELECTION must not quietly de-standardize a run."""

    monkeypatch.setenv("GWD_NLHF_SELECTION", leaked)
    margins, reference = _problem()
    with pytest.raises(ValueError, match="non-production"):
        solve_constrained_nash(margins, reference, 0.5)


def test_the_environment_may_still_name_the_production_rule(monkeypatch):
    monkeypatch.setenv("GWD_NLHF_SELECTION", NLHF_BASELINE_VERSION_REFERENCE_START)
    margins, reference = _problem()
    _, _, diagnostics = solve_constrained_nash(margins, reference, 0.5)
    assert diagnostics["selection"] == NLHF_BASELINE_VERSION_REFERENCE_START


def test_an_unknown_selection_is_refused():
    margins, reference = _problem()
    with pytest.raises(ValueError, match="unknown selection"):
        solve_constrained_nash(margins, reference, 0.5, selection="whatever")


# ------------------------------------------------------------- end-to-end fitter

def test_fit_pooled_nlhf_inherits_the_one_start_default():
    rng = np.random.default_rng(5)
    m = 4
    pairs = np.array([[a, b] for a in range(m) for b in range(a + 1, m)] * 10)
    signs = rng.choice([-1.0, 1.0], size=len(pairs))
    real = nlhf.minimize
    calls = []
    with mock.patch.object(nlhf, "minimize",
                           side_effect=lambda *a, **k: (calls.append(1), real(*a, **k))[1]):
        model = fit_pooled_nlhf(
            pair_indices=pairs, signs=signs, n_alternatives=m,
            reference=np.full(m, 1.0 / m), tau=0.5, seed=0,
        )
    assert len(calls) == 1
    assert model.diagnostics["selection"] == NLHF_BASELINE_VERSION_REFERENCE_START
    assert model.diagnostics["optimizer_calls"] == 1
    assert abs(model.exploitability) < EXPLOITABILITY_TOLERANCE
