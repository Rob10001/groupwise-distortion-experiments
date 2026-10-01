"""suite acceptance: the declared inventory, the shared cells and the reuse gate.

These pin the arithmetic and the policy boundaries of the suite comparison suite
without running any of it: what is fitted, what is imported, what is shared
between conditions, and what the execution policy forbids outright.
"""

from __future__ import annotations

import math

import pytest

from empirical_testing.experiments import suite as suite_module
from empirical_testing.experiments.suite import (
    ACTIVE,
    ACTIVE_BETAS,
    BETA_GRID,
    EVALUATION_TAU,
    PASSIVE,
    TAU_GRID,
    BetaFitGuard,
    ExecutionPolicyError,
    FitPlan,
    SUITES,
    beta_condition_cells,
    dry_run_inventory,
    tau_condition_cells,
)
from empirical_testing.methods import ALLOWED_METHODS, is_retired


# ------------------------------------------------------------------- identities

@pytest.mark.parametrize("dataset", ["coffee", "persona"])
def test_the_naming_contract_is_exactly_as_specified(dataset):
    suite = SUITES[dataset]
    assert suite.suite_id == f"{dataset}-comparison-suite-v10"
    assert suite.beta_protocol_id == f"{dataset}-protocol-beta-v10"
    assert suite.tau_protocol_id == f"{dataset}-active-tau-v10"
    assert suite.beta_directory == f"results/{dataset}/beta-source"
    assert suite.tau_directory == f"results/{dataset}/radius-rerun"
    assert suite.suite_directory == f"results/{dataset}/comparison-rerun"
    assert suite.figure_directory == f"figures/{dataset}"


@pytest.mark.parametrize("dataset", ["coffee", "persona"])
def test_historical_source_ids_are_recorded_as_historical(dataset):
    """Not rewritten to suggest a different design or fresh training."""

    suite = SUITES[dataset]
    historical = suite.as_mapping()["historical_ids"]
    assert "v10" not in historical["source_design_id"]
    assert "v10" not in historical["source_beta_run_id"]
    assert historical["source_beta_run_id"].endswith("protocol-beta-sweep")


def test_the_nlhf_baseline_is_the_suite_reference_start_rule():
    from empirical_testing.algorithms.nlhf import NLHF_BASELINE_VERSION_REFERENCE_START

    assert suite_module.NLHF_BASELINE == NLHF_BASELINE_VERSION_REFERENCE_START == "reference_start_v10"
    for suite in SUITES.values():
        assert suite.as_mapping()["nlhf_baseline"] == "reference_start_v10"


# ------------------------------------------------------------------- the grids

def test_only_the_four_permitted_families_appear_anywhere():
    methods = {cell.method for cell in suite_module.all_cells()}
    assert methods == set(ALLOWED_METHODS)
    assert not any(is_retired(m) for m in methods)


def test_the_beta_condition_runs_active_glhf_only_at_four_and_infinity():
    active_betas = {
        cell.beta for cell in beta_condition_cells()
        if cell.protocol == ACTIVE and cell.method in ("multigroup", "pooled")
    }
    assert active_betas == set(ACTIVE_BETAS)
    passive_betas = {
        cell.beta for cell in beta_condition_cells()
        if cell.protocol == PASSIVE and cell.method in ("multigroup", "pooled")
    }
    assert passive_betas == set(BETA_GRID)


def test_the_tau_ablation_is_active_glhf_at_beta_infinity():
    cells = tau_condition_cells()
    glhf = [c for c in cells if c.method in ("multigroup", "pooled")]
    assert glhf, "the tau ablation declared no GLHF arms"
    assert all(c.protocol == ACTIVE for c in glhf)
    assert all(math.isinf(c.beta) for c in cells)
    assert {c.learner_tau for c in glhf} == set(TAU_GRID)


def test_nlhf_is_passive_trained_in_every_condition_including_the_active_one():
    """NLHF is not an active learner in this study."""

    for cell in suite_module.all_cells():
        if cell.method == "pooled_nlhf":
            assert cell.protocol == PASSIVE, f"{cell.label} is not passive-trained"
    assert "never fitted on an active log" in suite_module.TRAINING_PROTOCOL["pooled_nlhf"]


def test_the_evaluation_radius_never_moves():
    for cell in suite_module.all_cells():
        assert cell.as_mapping()["evaluation_tau"] == EVALUATION_TAU == 0.5
    # ...while the learner radius does, which is the point of the ablation.
    assert len({c.learner_tau for c in tau_condition_cells()}) == len(TAU_GRID)


def test_every_cell_states_its_training_protocol_explicitly():
    for cell in suite_module.all_cells():
        payload = cell.as_mapping()
        assert payload["training_protocol"], f"{cell.label} does not state how it is trained"
        assert payload["display_name"] in ALLOWED_METHODS.values()


def test_uniform_is_evaluate_only_everywhere():
    for cell in suite_module.all_cells():
        if cell.method == "uniform":
            assert cell.provenance == "evaluate_only"
            assert not cell.is_fit


# --------------------------------------------------------------- shared results

def test_tau_half_glhf_is_the_imported_active_infinity_beta_cell():
    shared = [c for c in tau_condition_cells()
              if c.learner_tau == 0.5 and c.method in ("multigroup", "pooled")]
    assert len(shared) == 2
    for cell in shared:
        assert cell.provenance == "shared", "tau=0.5 GLHF must not be refitted"
        assert cell.shared_with == f"beta:active@inf:{cell.method}"


def test_the_tau_half_nlhf_benchmark_is_the_passive_infinity_beta_fit():
    shared = [c for c in tau_condition_cells()
              if c.learner_tau == 0.5 and c.method == "pooled_nlhf"]
    assert len(shared) == 1
    assert shared[0].provenance == "shared"
    assert shared[0].shared_with == "beta:passive@inf:pooled_nlhf"


def test_an_nlhf_benchmark_shown_beside_both_protocols_is_counted_once():
    """Displaying it twice must not schedule it twice."""

    beside_active = [c for c in beta_condition_cells()
                     if c.displayed_beside == ACTIVE and c.method == "pooled_nlhf"]
    assert len(beside_active) == len(ACTIVE_BETAS)
    for cell in beside_active:
        assert cell.provenance == "shared", "an active-protocol NLHF column was scheduled as a fit"
        assert cell.shared_with.startswith("beta:passive@")
        # Displayed beside the active protocol, but still recorded as the
        # passive fit it is.
        assert cell.protocol == PASSIVE
        assert "not an active fit" in cell.as_mapping()["display_note"]
    # And no cell claims to be an *active* NLHF fit.
    assert not any(c.method == "pooled_nlhf" and c.protocol == ACTIVE and c.is_fit
                   for c in suite_module.all_cells())


# ---------------------------------------------------------------- the arithmetic

def test_the_production_plan_matches_the_specified_fit_counts():
    plan = FitPlan(n_splits=100, n_seeds=50)
    counts = plan.fits_by_kind()
    assert plan.units == 5000
    # Zero new beta GLHF fits: the beta grid is imported, never retrained.
    assert counts["glhf_beta"] == 0
    # 50,000 new GLHF fits: five tau radii, the sixth being shared.
    assert counts["glhf_tau"] == 50_000
    # 50,000 unique NLHF fits: five betas plus five tau radii.
    assert counts["nlhf_beta"] == 25_000
    assert counts["nlhf_tau"] == 25_000
    assert counts["nlhf_beta"] + counts["nlhf_tau"] == 50_000
    assert plan.total_new_fits() == 100_000


def test_the_shared_tau_radius_is_the_one_that_is_not_refitted():
    plan = FitPlan(n_splits=100, n_seeds=50)
    # Five of the six radii are fitted; 6 x 2 x 5000 would be 60,000.
    assert plan.fits_by_kind()["glhf_tau"] == (len(TAU_GRID) - 1) * 2 * plan.units


# ------------------------------------------------------------ the execution gate

@pytest.mark.parametrize("dataset", ["coffee", "persona"])
def test_a_reuse_required_dry_run_schedules_zero_beta_glhf_fits(dataset):
    inventory = dry_run_inventory(SUITES[dataset], policy="reuse-required")
    assert inventory["plan"]["beta_glhf_fits"] == 0
    assert inventory["execution"] == "reuse-required"


def test_the_dry_run_inventory_contains_only_allowed_families():
    for suite in SUITES.values():
        inventory = dry_run_inventory(suite)
        assert inventory["method_inventory"] == sorted(ALLOWED_METHODS)
        assert not any(is_retired(m) for m in inventory["method_inventory"])


def test_an_unknown_execution_policy_is_refused():
    with pytest.raises(ExecutionPolicyError, match="unknown execution policy"):
        dry_run_inventory(SUITES["coffee"], policy="whatever-is-cheapest")


@pytest.mark.parametrize("method", ["multigroup", "pooled"])
def test_reuse_required_refuses_a_beta_glhf_fit_at_the_call_site(method):
    """Not an intention: an actual refusal, naming the cell."""

    guard = BetaFitGuard("reuse-required")
    with pytest.raises(ExecutionPolicyError, match="forbids beta GLHF fitting"):
        guard.record_fit("beta:passive@1:" + method, method=method, condition="beta")
    assert guard.calls == [], "the forbidden fit was recorded as if it had run"


def test_reuse_required_still_permits_the_nlhf_and_tau_fits_suite_must_do():
    guard = BetaFitGuard("reuse-required")
    guard.record_fit("beta:passive@1:pooled_nlhf", method="pooled_nlhf", condition="beta")
    guard.record_fit("tau:active@0.2:multigroup", method="multigroup", condition="tau")
    guard.record_fit("tau:passive@0.2:pooled_nlhf", method="pooled_nlhf", condition="tau")
    assert len(guard.calls) == 3


def test_the_fresh_policy_permits_what_reuse_required_forbids():
    guard = BetaFitGuard("fresh")
    guard.record_fit("beta:passive@1:multigroup", method="multigroup", condition="beta")
    assert guard.calls == ["beta:passive@1:multigroup"]


def test_no_policy_lets_anything_fit_the_uniform_reference():
    for policy in ("reuse-required", "fresh"):
        guard = BetaFitGuard(policy)
        with pytest.raises(ExecutionPolicyError, match="never fitted"):
            guard.record_fit("beta:passive@1:uniform", method="uniform", condition="beta")


def test_the_guard_refuses_a_retired_family_outright():
    from empirical_testing.methods import RetiredMethodError

    guard = BetaFitGuard("fresh")
    with pytest.raises(RetiredMethodError):
        guard.record_fit("whatever", method="intersection_nlhf", condition="tau")


def test_running_the_whole_plan_through_the_guard_makes_no_beta_glhf_call():
    """The end-to-end statement: reuse-required schedules zero beta GLHF fits."""

    plan = FitPlan(n_splits=2, n_seeds=2)
    guard = BetaFitGuard("reuse-required")
    for cell in plan.cells:
        if cell.is_fit:
            guard.record_fit(cell)
    assert guard.calls, "the plan scheduled no fits at all"
    assert not any(label.startswith("beta:") and "nlhf" not in label for label in guard.calls)


# ------------------------------------------- no fitter runs during validation

def test_anchor_validation_and_the_reuse_check_invoke_no_fitter(tmp_path, monkeypatch):
    """"Zero beta GLHF fitting calls" includes validating the anchor.

    The anchor is checked by comparing saved arrays on disk, so no fitter --
    online learner or NLHF solver -- may be entered while doing it.  This
    patches the fitters themselves rather than trusting the call graph: if any
    path reaches one, the test fails with the name of what was called.
    """

    from empirical_testing.algorithms import nlhf, registry
    from empirical_testing.experiments import provenance as migration
    from empirical_testing.experiments.suite import check_reuse_available

    entered: list[str] = []

    def forbidden(name):
        def _call(*_args, **_kwargs):
            entered.append(name)
            raise AssertionError(f"{name} was called during validation")
        return _call

    monkeypatch.setattr(registry, "fit_online_learner", forbidden("fit_online_learner"))
    monkeypatch.setattr(nlhf, "fit_pooled_nlhf", forbidden("fit_pooled_nlhf"))
    monkeypatch.setattr(nlhf, "solve_constrained_nash", forbidden("solve_constrained_nash"))
    monkeypatch.setattr(nlhf, "minimize", forbidden("scipy.optimize.minimize"))

    for suite in SUITES.values():
        # Against an empty root: the gate must decide, and report, without
        # fitting anything -- a blocked result is still a decision.
        status = check_reuse_available(tmp_path, suite)
        assert status["available"] is False

    # ...and the same for the direct array comparison, on a run that does not
    # exist: it reports rather than reaching for a fitter to fill the gap.
    result = migration.compare_with_frozen_arrays(tmp_path / "run", tmp_path / "frozen")
    assert result["agrees"] is False
    assert "nothing recomputed" in result["method"]

    assert entered == [], f"a fitter was entered during validation: {entered}"


@pytest.mark.parametrize("dataset", ["coffee", "persona"])
def test_source_resolver_requires_the_canonical_location(dataset, tmp_path):
    suite = SUITES[dataset]
    original = tmp_path / suite.beta_directory / "source"
    archive = tmp_path / "results" / "v10" / "runs" / suite.beta_protocol_id / "source"
    assert suite_module.resolve_beta_source(tmp_path, suite) == original
    archive.mkdir(parents=True)
    assert suite_module.resolve_beta_source(tmp_path, suite) == original
    original.mkdir(parents=True)
    assert suite_module.resolve_beta_source(tmp_path, suite) == original
    # An incomplete canonical run remains a blocker even if an older copy exists.
    status = suite_module.check_reuse_available(tmp_path, suite)
    assert status["available"] is False
    assert status["imported_source"] == str(original)
