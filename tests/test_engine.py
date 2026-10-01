"""The suite execution engine's written specification, pinned as tests.

``src/empirical_testing/experiments/engine.py`` makes claims that are
cheap to state and expensive to notice being violated: that every active fit
draws its own policy-dependent log, that the passive log is shared, that NLHF
is never trained actively, that held-out utilities never reach the fitter, and
above all that the second evaluation regime is a *re-scoring* of an already
fitted policy rather than a second fit.  Each test below quotes the clause it
is defending.

Everything here is synthetic: a 24-row training block, a 12-row test fold, four
alternatives and three groups, with two learner seeds.  That is enough to
exercise every cell of the plan (5 betas + 5 new radii x 2 protocols) in well
under a second, and none of it touches the real datasets or any run directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from empirical_testing.experiments import engine as engine
from empirical_testing.experiments.engine import (
    GLHF_ARMS,
    METHODS,
    NEW_TAUS,
    NLHF_ARM,
    NLHF_EXPLOITABILITY_GATE,
    UNIFORM_ARM,
    FitCell,
    all_fit_cells,
    fits_per_split,
    fit_split,
    compute_split,
    publish_split,
    saved_split,
)

#: What the shipped configs use; the engine is only ever run under this learner.
ALGORITHM = "groupwise_self_play_span_adaptive"
#: A fixed synthetic-fixture CV seed keeps these small fits reproducible.
CV_SEED = 20260917

N_TRAIN, N_TEST, N_ACTIONS = 24, 12, 4
GROUP_NAMES = ("g0", "g1", "g2")
LEARNER_SEEDS = (1, 2)


def _view(n_rows: int, seed: int) -> SimpleNamespace:
    """A minimal stand-in for a dense dataset view.

    ``fit_split`` and ``compute_split`` read exactly two attributes off a view
    -- ``utilities`` and ``groups`` -- so a namespace is a faithful double.
    Utilities are kept strictly inside ``(0, 1]``: the span-adaptive learner
    refuses anything outside ``[0, 1]``, and a strictly positive group mean
    keeps every welfare ratio finite, which is what makes a split usable.
    Every respondent is placed in two of the three groups so no group is empty
    on either side of the split.
    """

    rng = np.random.default_rng(seed)
    utilities = 0.2 + 0.7 * rng.random((n_rows, N_ACTIONS))
    groups = np.zeros((n_rows, len(GROUP_NAMES)), dtype=np.int8)
    for row in range(n_rows):
        groups[row, row % len(GROUP_NAMES)] = 1
        groups[row, (row + 1) % len(GROUP_NAMES)] = 1
    return SimpleNamespace(utilities=np.asarray(utilities, dtype=np.float64), groups=groups)


@pytest.fixture(scope="module")
def views() -> tuple[SimpleNamespace, SimpleNamespace]:
    """A train/test pair whose utilities genuinely differ between the folds.

    Different draws matter: several invariants below would be vacuous if the
    test fold happened to carry the training block's numbers.
    """

    return _view(N_TRAIN, seed=101), _view(N_TEST, seed=202)


def _fit(views, cells=None):
    train_view, test_view = views
    return fit_split(
        algorithm=ALGORITHM, cv_seed=CV_SEED, learner_seeds=LEARNER_SEEDS,
        comparisons_per_respondent=1,
        cells=all_fit_cells() if cells is None else cells,
        train_view=train_view, test_view=test_view, repeat=0, fold=0,
    )


def _compute(views):
    train_view, test_view = views
    return compute_split(
        algorithm=ALGORITHM, cv_seed=CV_SEED, learner_seeds=LEARNER_SEEDS,
        comparisons_per_respondent=1, group_names=GROUP_NAMES,
        train_view=train_view, test_view=test_view, repeat=0, fold=0,
    )


@pytest.fixture(scope="module")
def fitted(views):
    """One ``fit_split`` over the whole plan, reused by the sampling tests."""

    return _fit(views)


@pytest.fixture(scope="module")
def computed(views):
    """One ``compute_split``, reused by the evaluation and structure tests."""

    return _compute(views)


# --------------------------------------------------------- sampling invariants

def test_every_active_radius_draws_its_own_log(fitted):
    """"every active fit draws its **own** policy-dependent log at its **own**
    learner radius -- the ``tau = 0.5`` active log is never replayed at another
    radius, because no active log is ever carried between cells".

    Two things are proved: that the bookkeeping gives each (radius, arm) its
    own entry, and -- the part that would actually catch a replayed log -- that
    the drawn comparisons differ between radii for the same arm and seed.  A
    radius change moves the learner's policy, so it must move the queries it
    asks; identical logs across radii would mean one log was reused.
    """

    _, _, logs = fitted
    expected_keys = {
        f"tau@{engine.tau_label(tau)}|active|{arm}"
        for tau in NEW_TAUS for arm in GLHF_ARMS
    }
    assert expected_keys <= set(logs), (
        "missing active logs for "
        f"{sorted(expected_keys - set(logs))}; every new radius x arm needs its own"
    )
    assert len(expected_keys) == len(NEW_TAUS) * len(GLHF_ARMS) == 10

    for key in expected_keys:
        block = np.asarray(logs[key])
        assert block.shape == (len(LEARNER_SEEDS), N_TRAIN, 4), (
            f"{key} recorded {block.shape}, not one comparison per training respondent "
            "per seed"
        )

    # The replay check proper: same arm, same seed, different radius.
    for arm in GLHF_ARMS:
        for left, right in zip(NEW_TAUS, NEW_TAUS[1:]):
            a = np.asarray(logs[f"tau@{engine.tau_label(left)}|active|{arm}"])
            b = np.asarray(logs[f"tau@{engine.tau_label(right)}|active|{arm}"])
            for seed_index in range(len(LEARNER_SEEDS)):
                assert not np.array_equal(a[seed_index], b[seed_index]), (
                    f"{arm} drew an identical log at tau={left} and tau={right} for seed "
                    f"{LEARNER_SEEDS[seed_index]}; a policy-dependent log cannot be "
                    "radius-invariant, so one of them was replayed"
                )


def test_passive_respondent_order_is_shared_and_is_a_permutation(views, fitted):
    """"the source-compatible passive ``beta = inf`` log is built **once** per
    ``(split, seed)`` and its respondent order is reused by every cell, the
    active ones included, so a protocol or radius difference is never a row
    -order difference in disguise".

    The digest is the engine's own evidence that one order served the whole
    plan; it must be reproducible from an identical call, or "shared" would
    mean nothing across reruns.
    """

    records, logs = fitted[1], fitted[2]
    order = np.asarray(logs["respondent_order"])
    assert order.shape == (len(LEARNER_SEEDS), N_TRAIN)
    for row, seed in zip(order, LEARNER_SEEDS):
        assert np.array_equal(np.sort(row), np.arange(N_TRAIN)), (
            f"seed {seed}'s respondent order is not a permutation of the training rows"
        )
    # Different seeds get different orders -- that is the design, not a bug --
    # but the rule that produced them is one rule.
    assert not np.array_equal(order[0], order[1])

    again = _fit(views)
    assert again[1]["respondent_order_sha256"] == records["respondent_order_sha256"], (
        "an identical call produced a different respondent order; the passive log is "
        "not being derived deterministically from (split, seed)"
    )
    assert records["shared_passive_log_per_seed"] is True
    assert records["active_log_per_fit"] is True


def test_nlhf_is_passive_trained_in_every_cell():
    """"NLHF is fitted on the **passive** log in every cell, at the cell's
    learner radius, and is never handed an active log".

    This is a structural claim about the plan, so it is checked over the plan
    rather than over one run: no cell can both carry the NLHF arm and declare
    itself active, in either direction.
    """

    cells = all_fit_cells()
    assert cells, "the plan is empty"
    for cell in cells:
        if NLHF_ARM in cell.arms:
            assert cell.protocol == "passive", (
                f"{cell.label} trains {NLHF_ARM} under protocol {cell.protocol!r}; NLHF is "
                "passive everywhere in this study"
            )
        if cell.protocol == "active":
            assert NLHF_ARM not in cell.arms, (
                f"active cell {cell.label} declares an NLHF arm"
            )
            assert set(cell.arms) == set(GLHF_ARMS)
    assert any(c.protocol == "active" for c in cells), "no active cell to check"


def test_fitter_only_ever_sees_the_training_block(views, monkeypatch):
    """"held-out utilities are never read: the test fold contributes group
    memberships only".

    The spy checks the utilities handed to every learner call against the
    training block, by value, and separately against the test fold's block.
    """

    from empirical_testing.algorithms import registry

    train_view, test_view = views
    train_rows = {tuple(row) for row in np.asarray(train_view.utilities)}
    test_rows = {tuple(row) for row in np.asarray(test_view.utilities)}
    assert not (train_rows & test_rows), "the fixture folds must not share a row"

    real = registry.fit_online_learner
    seen: list[np.ndarray] = []

    def spy(*args, **kwargs):
        seen.append(np.asarray(kwargs["utilities"], dtype=np.float64))
        return real(*args, **kwargs)

    monkeypatch.setattr(registry, "fit_online_learner", spy)
    _fit(views, cells=[c for c in all_fit_cells() if c.protocol == "active"])

    assert seen, "no learner call was observed"
    for utilities in seen:
        assert utilities.shape == (N_TRAIN, N_ACTIONS)
        rows = {tuple(row) for row in utilities}
        assert rows <= train_rows, "the learner was handed a row that is not a training row"
        assert not (rows & test_rows), "a held-out row reached the fitter"


def test_training_only_guard_is_invoked_for_active_cells(views, monkeypatch):
    """The same clause, proved by making the guard fire.

    ``_check_training_only_utilities`` is the boundary check; a guard that is
    never called protects nothing.  It is counted here (once per active arm per
    seed) and then made to raise by handing the engine a split whose test fold
    *is* its training block.
    """

    from empirical_testing.experiments import coffee_protocol_beta_sweep as sweep

    real = sweep._check_training_only_utilities
    calls: list[int] = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sweep, "_check_training_only_utilities", counting)
    active_cells = [c for c in all_fit_cells() if c.protocol == "active"]
    _fit(views, cells=active_cells)
    assert len(calls) == len(active_cells) * len(LEARNER_SEEDS), (
        "the held-out guard did not run once per active cell per learner seed"
    )

    # Now the guard's job: identical folds must be refused, not fitted.  Every
    # row is the same vector, so any permutation of the training block is still
    # equal to the test block and the leak is unambiguous.
    leaked = np.tile(np.array([0.3, 0.5, 0.7, 0.9]), (8, 1))
    groups = np.tile(np.array([1, 1, 0], dtype=np.int8), (8, 1))
    same = SimpleNamespace(utilities=leaked, groups=groups)
    with pytest.raises(AssertionError, match="held-out"):
        fit_split(
            algorithm=ALGORITHM, cv_seed=CV_SEED, learner_seeds=(1,),
            comparisons_per_respondent=1,
            cells=(FitCell("tau", "active", math.inf, 0.05, GLHF_ARMS),),
            train_view=same, test_view=same, repeat=0, fold=0,
        )


def test_budget_other_than_one_comparison_is_refused(views):
    """The active protocol consumes one training row per round, so the engine
    accepts exactly one comparison per training respondent.
    """

    train_view, test_view = views
    with pytest.raises(ValueError, match="one comparison per training respondent"):
        fit_split(
            algorithm=ALGORITHM, cv_seed=CV_SEED, learner_seeds=(1,),
            comparisons_per_respondent=2, cells=all_fit_cells()[:1],
            train_view=train_view, test_view=test_view, repeat=0, fold=0,
        )


# --------------------------------------------------------- evaluation regimes

def test_fixed_regime_always_evaluates_at_half(computed):
    """"``fixed_0.5``: evaluation radius ``0.5`` at every learner radius."""

    _, arrays, _ = computed
    fixed = arrays["tau_fixed_evaluation_tau"]
    assert fixed.shape == (len(NEW_TAUS),)
    assert np.all(fixed == engine.FIXED_EVALUATION_TAU), (
        f"the fixed regime evaluated at {fixed}, not uniformly at 0.5"
    )
    beta = arrays["beta_evaluation_tau"]
    assert np.all(beta == engine.FIXED_EVALUATION_TAU), (
        f"the beta condition evaluated at {beta}, not uniformly at 0.5"
    )


def test_moving_regime_evaluates_at_the_learner_radius(computed):
    """"``moving``: evaluation radius equal to the learner radius."""

    _, arrays, _ = computed
    moving = arrays["tau_moving_evaluation_tau"]
    assert np.array_equal(moving, np.asarray(NEW_TAUS, dtype=np.float64)), (
        f"the moving regime evaluated at {moving}, not at the learner radii {NEW_TAUS}"
    )


def test_second_regime_adds_no_fits(views, monkeypatch):
    """"the policy cube is computed once and scored twice".

    The whole separation rests on the second regime being a re-scoring.  If it
    silently refitted, the numbers would still look plausible but the two
    regimes would no longer share a policy, so the count of fitting calls in a
    single ``compute_split`` is pinned to the scheduled count exactly -- the
    second regime must add zero.  The two regimes' arrays are then compared to
    confirm that the scoring really did differ, so the constant fit count is
    not just the absence of a second evaluation.
    """

    from empirical_testing.algorithms import registry
    from empirical_testing.algorithms import nlhf

    real_glhf, real_nlhf = registry.fit_online_learner, nlhf.fit_pooled_nlhf
    counts = {"glhf": 0, "nlhf": 0}

    def glhf_spy(*args, **kwargs):
        counts["glhf"] += 1
        return real_glhf(*args, **kwargs)

    def nlhf_spy(*args, **kwargs):
        counts["nlhf"] += 1
        return real_nlhf(*args, **kwargs)

    monkeypatch.setattr(registry, "fit_online_learner", glhf_spy)
    monkeypatch.setattr(nlhf, "fit_pooled_nlhf", nlhf_spy)
    summary, arrays, _ = _compute(views)

    scheduled = fits_per_split(len(LEARNER_SEEDS))
    assert summary["fits"] == scheduled
    assert counts["glhf"] == scheduled["glhf_beta"] + scheduled["glhf_tau"], (
        f"{counts['glhf']} GLHF fits against a schedule of "
        f"{scheduled['glhf_beta'] + scheduled['glhf_tau']}; the second regime refitted"
    )
    assert counts["nlhf"] == scheduled["nlhf_beta"] + scheduled["nlhf_tau"], (
        f"{counts['nlhf']} NLHF fits against a schedule of "
        f"{scheduled['nlhf_beta'] + scheduled['nlhf_tau']}; the second regime refitted"
    )
    assert counts["glhf"] + counts["nlhf"] == sum(scheduled.values())

    fixed = arrays["tau_fixed_worst_group_distortion"]
    moving = arrays["tau_moving_worst_group_distortion"]
    assert fixed.shape == moving.shape
    assert not np.array_equal(fixed, moving), (
        "the two regimes produced identical numbers; then they are not two evaluations"
    )


def test_beta_condition_is_evaluated_once_at_half(computed):
    """"For the beta condition the learner radius is ``0.5`` throughout, so the
    two regimes coincide there; it is computed once and labelled ``fixed_0.5``."

    The coincidence is what licenses storing a single beta block, so it is
    checked rather than assumed: every beta cell carries learner radius 0.5 and
    is evaluated at 0.5, and no ``tau_moving``-style beta array exists.
    """

    summary, arrays, _ = computed
    assert np.all(arrays["beta_evaluation_tau"] == engine.FIXED_EVALUATION_TAU)
    for cell in all_fit_cells():
        if cell.condition == "beta":
            assert float(cell.learner_tau) == engine.FIXED_EVALUATION_TAU
            assert cell.evaluation_taus == (
                (engine.REGIME_FIXED, engine.FIXED_EVALUATION_TAU),
            ), f"{cell.label} would be scored twice at one radius"
    assert not any(key.startswith("beta_moving") for key in arrays)
    assert summary["fixed_evaluation_tau"] == engine.FIXED_EVALUATION_TAU


def test_uniform_column_moves_only_under_the_moving_regime(computed):
    """"Under ``moving`` the welfare comparator moves too, so it is stored per
    radius rather than once per split."

    The uniform arm is untrained, so at a fixed evaluation radius its score
    cannot depend on the learner radius at all -- a varying uniform column
    would mean the evaluation leaked the learner's radius.  Under the moving
    regime the same untrained policy is scored against a different uncertainty
    set at every radius, so it *must* vary.  These two facts together are the
    reason the regimes are separate arrays rather than one.
    """

    _, arrays, _ = computed
    uniform = METHODS.index(UNIFORM_ARM)

    fixed = arrays["tau_fixed_worst_group_distortion"][:, :, uniform]
    assert np.all(np.isfinite(fixed))
    assert np.allclose(fixed, fixed[0], rtol=0.0, atol=0.0), (
        "the untrained uniform benchmark varied across learner radii at a fixed "
        f"evaluation radius: {fixed[:, 0]}"
    )
    fixed_optimum = arrays["tau_fixed_welfare_optimum"]
    assert np.allclose(fixed_optimum, fixed_optimum[0], rtol=0.0, atol=0.0), (
        "the welfare comparator moved across radii at a fixed evaluation radius"
    )

    moving = arrays["tau_moving_worst_group_distortion"][:, :, uniform]
    assert np.all(np.isfinite(moving))
    assert len({tuple(row) for row in moving}) == len(NEW_TAUS), (
        "the uniform benchmark did not move with the evaluation radius under the "
        f"moving regime: {moving[:, 0]}"
    )
    moving_optimum = arrays["tau_moving_welfare_optimum"]
    assert not np.allclose(moving_optimum, moving_optimum[0]), (
        "the comparator is supposed to move with the radius under the moving regime"
    )


# ----------------------------------------------------------- structure, safety

def test_methods_axis_is_the_four_reported_methods(computed):
    """Every condition carries all four methods, in reporting order."""

    assert METHODS == ("multigroup", "pooled", "pooled_nlhf", "uniform")

    summary, arrays, _ = computed
    assert summary["methods"] == list(METHODS)
    for key in ("beta_worst_group_distortion", "tau_fixed_worst_group_distortion",
                "tau_moving_worst_group_distortion"):
        assert arrays[key].shape[-1] == len(METHODS), (
            f"{key} has {arrays[key].shape[-1]} method columns, not {len(METHODS)}"
        )
    for key in ("beta_group_distortion", "tau_fixed_group_distortion",
                "tau_moving_group_distortion"):
        assert arrays[key].shape[-2] == len(METHODS)
    # Every reported point declares which of the four columns are real.
    for label, real in summary["real_columns"].items():
        assert set(real) <= set(METHODS), label
        assert UNIFORM_ARM in real and NLHF_ARM in real, label


def test_padded_columns_are_blank(computed):
    """"a column that stands for an imported policy must never carry a
    plausible-looking number".

    In the beta condition the GLHF arms are imported, not refitted; the engine
    fills those slots only to satisfy the evaluator's paired-arm contract and
    must blank them before saving.  A number there would silently be read as a
    beta-grid GLHF result.
    """

    summary, arrays, _ = computed
    for method in ("pooled", "multigroup"):
        column = METHODS.index(method)
        for key in ("beta_worst_group_distortion", "beta_group_distortion",
                    "beta_welfare"):
            block = arrays[key][:, :, column]
            assert np.all(np.isnan(block)), (
                f"{key} carries a non-NaN value in the imported {method!r} column"
            )
        index = arrays["beta_worst_group_index"][:, :, column]
        assert np.all(index == -1), "a padded column kept a worst-group index"
    for label, real in summary["real_columns"].items():
        if label.split("|", 1)[1].startswith("beta@"):
            assert set(real) == {NLHF_ARM, UNIFORM_ARM}, (
                f"{label} claims {real} as real columns; only NLHF is fitted at a beta"
            )
    # ... while the tau condition really does fit all three non-uniform arms.
    for method in METHODS:
        column = METHODS.index(method)
        for key in ("tau_fixed_worst_group_distortion", "tau_moving_worst_group_distortion"):
            assert np.all(np.isfinite(arrays[key][:, :, column])), (
                f"{key} left the {method!r} column blank; every tau column is fitted"
            )


def test_fit_schedule_has_no_beta_glhf_fits():
    """"suite **adopts** the completed GLHF beta grid rather than retraining it."

    The zero is structural, not a coincidence of the count: no beta cell may
    declare a GLHF arm at all.
    """

    assert fits_per_split(50) == {
        "glhf_beta": 0, "glhf_tau": 500, "nlhf_beta": 250, "nlhf_tau": 250,
    }
    for cell in all_fit_cells():
        if cell.condition == "beta":
            assert not set(cell.arms) & set(GLHF_ARMS), (
                f"{cell.label} would retrain GLHF at a beta the run imports"
            )
    # Linear in the seed count, so a pilot's schedule is the full run's scaled.
    assert fits_per_split(1) == {k: v // 50 for k, v in fits_per_split(50).items()}
    # The radius that is never refitted stays out of the plan entirely.
    assert engine.SHARED_TAU not in NEW_TAUS
    assert set(NEW_TAUS) | {engine.SHARED_TAU} == set(engine.TAU_GRID)


def test_nlhf_solver_diagnostics_are_recorded_per_fit(computed):
    """Each NLHF fit records how it was solved, not just what it returned.

    ``reference_start_v10`` with a single optimizer call is the suite selection
    rule; a second call would mean a restart crept back in.  The exploitability
    is the independent verification that makes the returned policy usable, so
    it is checked against the gate the fitter enforces.
    """

    summary, _, _ = computed
    solver = summary["solver"]
    nlhf_cells = [c.label for c in all_fit_cells() if NLHF_ARM in c.arms]
    assert set(solver) == {c.label for c in all_fit_cells()}
    # Active cells run no solver at all, so their slots stay empty rather than
    # borrowing a neighbouring cell's diagnostics.
    for cell in all_fit_cells():
        if NLHF_ARM not in cell.arms:
            assert solver[cell.label] == [], (
                f"active cell {cell.label} carries NLHF solver diagnostics"
            )
    total = 0
    for label in nlhf_cells:
        records = solver[label]
        assert len(records) == len(LEARNER_SEEDS), f"{label} recorded {len(records)} fits"
        for record in records:
            total += 1
            assert record["selection"] == "reference_start_v10", (
                f"{label} solved from {record['selection']!r}"
            )
            assert record["optimizer_calls"] == 1, (
                f"{label} made {record['optimizer_calls']} optimizer calls; suite starts once "
                "and never restarts"
            )
            assert abs(record["exploitability"]) < NLHF_EXPLOITABILITY_GATE, (
                f"{label} returned exploitability {record['exploitability']}, at or above "
                f"the gate {NLHF_EXPLOITABILITY_GATE}"
            )
    assert total == sum(
        v for k, v in fits_per_split(len(LEARNER_SEEDS)).items() if k.startswith("nlhf")
    )
    assert summary["max_abs_exploitability"] < NLHF_EXPLOITABILITY_GATE
    assert summary["exploitability_gate"] == NLHF_EXPLOITABILITY_GATE


def test_unconverged_solver_reports_are_recorded_not_failed(computed):
    """"These are **not** failed fits: ``fit_pooled_nlhf`` raises when
    exploitability exceeds the gate, so anything that returns has been verified
    independently and is usable ... Recorded so the margin is visible, never
    used for control flow."

    So the summary must always expose the count and the margin, including when
    the count is zero -- a reader has to be able to see that nothing was hidden
    rather than infer it from a missing key.
    """

    summary, _, _ = computed
    for key in ("n_solver_unconverged", "solver_unconverged",
                "max_abs_exploitability", "exploitability_gate"):
        assert key in summary, f"the summary hides {key}"
    assert summary["n_solver_unconverged"] == len(summary["solver_unconverged"])
    for record in summary["solver_unconverged"]:
        # An unconverged report still carries a verified policy: it is inside
        # the gate, which is strictly stronger than the optimizer's own status.
        assert record["solver_success"] is False
        assert abs(record["exploitability"]) < NLHF_EXPLOITABILITY_GATE
    # Whatever the count, every fit that returned is inside the gate.
    assert summary["max_abs_exploitability"] < summary["exploitability_gate"]


def test_an_unconverged_report_does_not_fail_the_fit(views, monkeypatch):
    """The same clause, exercised rather than merely inspected.

    On this fixture SLSQP happens to converge everywhere, so the branch is
    forced: the optimizer is made to report failure at a point the gate still
    accepts.  The engine must record it and carry on -- "never used for control
    flow, and never repaired by another start" -- so the policy cube comes out
    whole and the fit is not retried.
    """

    from empirical_testing.algorithms import nlhf

    real = nlhf.fit_pooled_nlhf
    calls = {"n": 0}

    class _Sulking:
        """The real result, wearing a failed optimizer status."""

        def __init__(self, inner):
            self._inner = inner
            self.exploitability = inner.exploitability
            self.diagnostics = dict(inner.diagnostics)
            self.diagnostics.update(
                solver_success=False, solver_status="Iteration limit reached")

        def predict(self, groups):
            return self._inner.predict(groups)

    def sulking(*args, **kwargs):
        calls["n"] += 1
        return _Sulking(real(*args, **kwargs))

    monkeypatch.setattr(nlhf, "fit_pooled_nlhf", sulking)
    beta_cells = [c for c in all_fit_cells() if c.condition == "beta"]
    policies, records, _ = _fit(views, cells=beta_cells)

    assert calls["n"] == len(beta_cells) * len(LEARNER_SEEDS), (
        "a non-convergence report triggered a retry or an extra start"
    )
    assert len(records["solver_unconverged"]) == calls["n"], (
        "an unconverged fit was not recorded"
    )
    assert {r["cell"] for r in records["solver_unconverged"]} == {c.label for c in beta_cells}
    # The policies are still there: nothing was dropped or replaced.
    for cell in beta_cells:
        assert policies[cell.label][NLHF_ARM].shape[0] == len(LEARNER_SEEDS)
    assert records["max_abs_exploitability"] < NLHF_EXPLOITABILITY_GATE


# --------------------------------------------------------------- atomic resume

@pytest.fixture
def published(tmp_path: Path, computed) -> tuple[Path, dict, dict]:
    """One published unit plus the identity a resume would check it against."""

    summary, arrays, active_logs = computed
    summary = dict(summary)
    summary.update({
        "protocol_id": "suite-test", "base_protocol_id": "base-test",
        "config_sha256": "c" * 8, "base_config_sha256": "c" * 8,
        "data_sha256": "d" * 8, "train_ids_sha256": "t" * 8,
        "test_ids_sha256": "s" * 8, "discovery_ids_sha256": "x" * 8,
        "groups": list(GROUP_NAMES),
    })
    directory = tmp_path / "splits" / "r00_f00"
    publish_split(directory, summary, arrays, active_logs)
    expected = {key: summary[key] for key in engine.REUSE_KEYS}
    return directory, expected, summary


def _saved(directory: Path, expected) -> dict | None:
    return saved_split(
        directory, expected, n_betas=len(engine.BETA_GRID), n_taus=len(NEW_TAUS),
        n_seeds=len(LEARNER_SEEDS), n_groups=len(GROUP_NAMES),
    )


def test_a_complete_unit_is_reused(published):
    """A unit that is whole and identical is adopted rather than recomputed."""

    directory, expected, summary = published
    assert (directory / "split.json").is_file()
    assert (directory / "split.npz").is_file()
    reused = _saved(directory, expected)
    assert reused is not None, "a complete, matching unit was not reused"
    assert reused["implementation_id"] == engine.IMPLEMENTATION_ID
    assert reused["regimes"] == list(engine.REGIMES)


def test_a_truncated_array_bundle_is_recomputed_not_raised(published):
    """"A half-written bundle is not an error: it is a unit that cannot be
    reused, so it is recomputed."

    An interruption mid-write must cost a recomputation, never a crash that
    strands the whole run.
    """

    directory, expected, _ = published
    arrays_path = directory / "split.npz"
    blob = arrays_path.read_bytes()
    arrays_path.write_bytes(blob[: len(blob) // 2])
    assert _saved(directory, expected) is None, (
        "a truncated split.npz was accepted for reuse"
    )
    # Not even an empty file may pass; it must simply decline.
    arrays_path.write_bytes(b"")
    assert _saved(directory, expected) is None


def test_the_completion_marker_is_what_makes_a_unit_reusable(published):
    """"``split.json`` is written last, so its presence is the completion
    marker."

    Deleting only the marker leaves a perfectly good ``split.npz`` behind; it
    must still be refused, because that is exactly the state an interruption
    between the two writes leaves on disk.
    """

    directory, expected, _ = published
    (directory / "split.json").unlink()
    assert (directory / "split.npz").is_file()
    assert _saved(directory, expected) is None, (
        "a unit with no completion marker was reused on the strength of its arrays"
    )


def test_leftover_staging_directory_is_not_a_unit(tmp_path: Path, published):
    """A ``.partial-*`` directory is the staging area, never a result.

    ``publish_split`` writes into ``.partial-<name>`` and then moves it into
    place, so a crashed run can leave one behind.  Reading it as though it were
    the published unit would resurrect a half-written split.
    """

    directory, expected, _ = published
    staging = directory.parent / f".partial-{directory.name}"
    staging.mkdir()
    for name in ("split.json", "split.npz"):
        (staging / name).write_bytes((directory / name).read_bytes())
    import shutil

    shutil.rmtree(directory)
    assert _saved(directory, expected) is None, (
        "a leftover staging directory was mistaken for a published unit"
    )
    assert staging.is_dir(), "the staging directory should be untouched by a failed resume"


@pytest.mark.parametrize(
    "key, replacement",
    [
        ("learner_seeds", [1, 2, 3]),
        ("regimes", ["fixed_0.5"]),
        ("learner_taus", [0.05, 0.2, 0.35, 0.75, 2.0]),
        ("fixed_evaluation_tau", 0.75),
        ("methods", ["multigroup", "pooled", "uniform"]),
        ("implementation_id", "engine-someone-elses"),
        ("data_sha256", "different"),
    ],
)
def test_a_unit_whose_identity_disagrees_is_refused(published, key, replacement):
    """"A unit computed under a different grid, regime set, seed list or
    implementation is never silently adopted."

    Each of these keys changes what the numbers *mean*, so a saved unit that
    disagrees on any one of them has to be recomputed rather than joined to the
    current run's arrays.
    """

    directory, expected, _ = published
    assert expected[key] != replacement, "the parametrization must actually change the key"
    disagreeing = dict(expected, **{key: replacement})
    assert _saved(directory, disagreeing) is None, (
        f"a unit whose {key} disagrees was accepted for reuse"
    )
    # ... and the untouched identity still resumes, so the refusal above is
    # about this key and not about the fixture being unusable.
    assert _saved(directory, expected) is not None


def test_published_summary_is_plain_json(published):
    """The unit on disk must be readable without unpickling anything."""

    directory, _, _ = published
    saved = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    assert saved["methods"] == list(METHODS)
    assert saved["learner_taus"] == [float(t) for t in NEW_TAUS]
    with np.load(directory / "split.npz", allow_pickle=False) as bundle:
        shapes = engine.expected_array_shapes(
            n_betas=len(engine.BETA_GRID), n_taus=len(NEW_TAUS),
            n_seeds=len(LEARNER_SEEDS), n_groups=len(GROUP_NAMES),
        )
        for name, shape in shapes.items():
            assert np.asarray(bundle[name]).shape == shape, name
