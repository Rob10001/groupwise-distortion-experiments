"""suite acceptance: the combined four-method reductions and the two regimes.

The fixed and moving evaluation regimes are **different estimands**.  These
tests pin that they stay apart, that every reported condition carries all four
methods with correct provenance, that all six pairwise contrasts are present,
and that the corrections are the ones the specification names.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from empirical_testing.experiments import engine as engine
from empirical_testing.experiments import reduce as reduce_module
from empirical_testing.experiments.reduce import (
    METHODS, PAIRWISE, REGIME_FIXED, REGIME_MOVING,
    assemble_beta_split, assemble_tau_split, beta_conditions, corrected,
    reduce_condition, reduce_protocol_contrasts, tau_conditions,
)

N_SEEDS, N_GROUPS, N_SPLITS = 4, 3, 12
GROUPS = [f"g{i}" for i in range(N_GROUPS)]


def _imported(rng) -> dict[str, np.ndarray]:
    """A stand-in for one split of the imported beta run.

    Shapes and axis order follow the real saved arrays: 7 cells x seeds x the
    three imported methods (pooled, multigroup, uniform) x groups.
    """

    n_cells, n_methods = len(reduce_module.IMPORTED_CELLS), len(reduce_module.IMPORTED_METHODS)
    return {
        "worst_group_distortion": 1.0 + rng.random((n_cells, N_SEEDS, n_methods)),
        "group_distortion": 1.0 + rng.random((n_cells, N_SEEDS, n_methods, N_GROUPS)),
        "welfare_optimum": np.tile(rng.random(N_GROUPS), (n_cells, 1)),
    }


def _new(rng) -> dict[str, np.ndarray]:
    """A stand-in for one split computed by the suite engine."""

    n_betas, n_taus, n_methods = len(engine.BETA_GRID), len(engine.NEW_TAUS), len(METHODS)
    arrays = {
        "beta_worst_group_distortion": np.full((n_betas, N_SEEDS, n_methods), np.nan),
        "beta_group_distortion": np.full((n_betas, N_SEEDS, n_methods, N_GROUPS), np.nan),
    }
    nlhf = METHODS.index("pooled_nlhf")
    arrays["beta_worst_group_distortion"][:, :, nlhf] = 1.0 + rng.random((n_betas, N_SEEDS))
    arrays["beta_group_distortion"][:, :, nlhf] = 1.0 + rng.random(
        (n_betas, N_SEEDS, N_GROUPS))
    for prefix in ("tau_fixed", "tau_moving"):
        arrays[f"{prefix}_worst_group_distortion"] = 1.0 + rng.random(
            (n_taus, N_SEEDS, n_methods))
        arrays[f"{prefix}_group_distortion"] = 1.0 + rng.random(
            (n_taus, N_SEEDS, n_methods, N_GROUPS))
    return arrays


@pytest.mark.parametrize("difference", [0.0, 1.2e-15, 1e-8])
def test_comparator_gate_accepts_platform_roundoff_and_rejects_changed_values(tmp_path, difference):
    """Exercise the production reduction gate with independently saved inputs."""

    from empirical_testing.experiments.run import _reduce_everything

    rng = np.random.default_rng(20260923)
    imported, new = _imported(rng), _new(rng)
    labels = list(reduce_module.IMPORTED_CELLS)
    comparator = imported["welfare_optimum"][labels.index("passive@inf")]
    new["beta_welfare_optimum"] = np.tile(comparator, (len(engine.BETA_GRID), 1))
    new["beta_welfare_optimum"][0, 0] += difference
    output, source = tmp_path / "new", tmp_path / "imported"
    for root, arrays, summary in (
        (output, new, {"n_train": 270, "n_test": 30}),
        (source, imported, {"groups": GROUPS, "cell_labels": labels}),
    ):
        directory = root / "splits" / "r00_f00"
        directory.mkdir(parents=True)
        np.savez(directory / "split.npz", **arrays)
        (directory / "split.json").write_text(json.dumps(summary), encoding="utf-8")
    new_bytes = (output / "splits/r00_f00/split.npz").read_bytes()
    imported_bytes = (source / "splits/r00_f00/split.npz").read_bytes()

    kwargs = {
        "plan": SimpleNamespace(groups=GROUPS, learner_seeds=range(N_SEEDS)),
        "splits": [(0, 0)], "output_dir": output, "imported_source": source,
        "confidence_level": 0.95, "log": lambda _: None,
    }
    if difference == 1e-8:
        with pytest.raises(RuntimeError, match="different welfare comparator"):
            _reduce_everything(**kwargs)
    else:
        result = _reduce_everything(**kwargs)
        validation = result["validation"]
        assert validation["comparator_matches_on_every_split"] is True
        assert validation["comparator_exact_on_every_split"] is (difference == 0.0)
        check = validation["comparator_checks"][0]
        assert check["max_abs_difference"] == float(abs(new["beta_welfare_optimum"][0] - comparator).max())
        assert check["comparator"] == comparator.tolist()
        saved = json.loads((output / "validation.json").read_text(encoding="utf-8"))
        assert saved["comparator_checks"] == validation["comparator_checks"]
    assert (output / "splits/r00_f00/split.npz").read_bytes() == new_bytes
    assert (source / "splits/r00_f00/split.npz").read_bytes() == imported_bytes


@pytest.mark.parametrize("invalid", [np.array([0.4]), np.array([np.nan, 0.5]), np.array([np.inf, 0.5])])
def test_comparator_agreement_rejects_shape_mismatch_and_nonfinite_values(invalid):
    result = reduce_module.check_comparator_matches(
        {"beta_welfare_optimum": np.array([invalid])},
        {"welfare_optimum": np.array([[0.4, 0.5]])},
        ["passive@inf"],
    )
    assert result["matches"] is False


@pytest.fixture(scope="module")
def cubes():
    rng = np.random.default_rng(20260921)
    imported = [_imported(rng) for _ in range(N_SPLITS)]
    new = [_new(rng) for _ in range(N_SPLITS)]
    labels = list(reduce_module.IMPORTED_CELLS)
    beta = [assemble_beta_split(imported=i, labels=labels, new_arrays=n,
                                n_seeds=N_SEEDS, n_groups=N_GROUPS)
            for i, n in zip(imported, new)]
    tau = {
        regime: [assemble_tau_split(regime=regime, imported=i, labels=labels,
                                    new_arrays=n, n_seeds=N_SEEDS, n_groups=N_GROUPS)
                 for i, n in zip(imported, new)]
        for regime in (REGIME_FIXED, REGIME_MOVING)
    }
    return {"beta": beta, "tau": tau,
            "test_sizes": [30] * N_SPLITS, "train_sizes": [270] * N_SPLITS}


def _reduce_tau(cubes, regime):
    cells = tau_conditions(regime)
    blocks = cubes["tau"][regime]
    return reduce_condition(
        cells=cells,
        worst_by_split=[b.worst for b in blocks],
        levels_by_split=[b.levels for b in blocks],
        provenance=blocks[0].provenance, groups=GROUPS,
        test_sizes=cubes["test_sizes"], train_sizes=cubes["train_sizes"],
        confidence_level=0.95,
    )


# ------------------------------------------------- the two regimes stay apart

def test_fixed_evaluation_always_uses_evaluation_tau_one_half(cubes):
    out = _reduce_tau(cubes, REGIME_FIXED)
    for table in out.values():
        for row in table:
            assert row["evaluation_tau"] == 0.5, row
            assert row["evaluation_regime"] == REGIME_FIXED


def test_moving_evaluation_always_uses_evaluation_tau_equal_to_learner_tau(cubes):
    out = _reduce_tau(cubes, REGIME_MOVING)
    for table in out.values():
        for row in table:
            assert row["evaluation_tau"] == row["learner_tau"], row
            assert row["evaluation_regime"] == REGIME_MOVING


def test_no_table_mixes_the_two_regimes(cubes):
    for regime in (REGIME_FIXED, REGIME_MOVING):
        out = _reduce_tau(cubes, regime)
        for name, table in out.items():
            regimes = {row["evaluation_regime"] for row in table}
            assert regimes == {regime}, f"{name} mixed regimes: {regimes}"


def test_the_moving_regime_covers_the_whole_learner_grid(cubes):
    out = _reduce_tau(cubes, REGIME_MOVING)
    taus = sorted({row["learner_tau"] for row in out["method_levels"]})
    assert taus == sorted(engine.TAU_GRID)
    evaluation = sorted({row["evaluation_tau"] for row in out["method_levels"]})
    assert evaluation == taus, "the moving regime's evaluation grid is not the learner grid"


# ------------------------------------------------------- four methods, always

@pytest.mark.parametrize("regime", [REGIME_FIXED, REGIME_MOVING])
def test_every_condition_contains_all_four_methods(cubes, regime):
    out = _reduce_tau(cubes, regime)
    for cell in {row["cell"] for row in out["method_levels"]}:
        methods = {row["method"] for row in out["method_levels"] if row["cell"] == cell}
        assert methods == set(METHODS), f"{cell} carries {methods}"


def test_the_beta_condition_also_carries_all_four_methods(cubes):
    blocks = cubes["beta"]
    out = reduce_condition(
        cells=beta_conditions(),
        worst_by_split=[b.worst for b in blocks],
        levels_by_split=[b.levels for b in blocks],
        provenance=blocks[0].provenance, groups=GROUPS,
        test_sizes=cubes["test_sizes"], train_sizes=cubes["train_sizes"],
        confidence_level=0.95,
    )
    for cell in {row["cell"] for row in out["method_levels"]}:
        methods = {row["method"] for row in out["method_levels"] if row["cell"] == cell}
        assert methods == set(METHODS), f"{cell} carries {methods}"


@pytest.mark.parametrize("regime", [REGIME_FIXED, REGIME_MOVING])
def test_all_six_pairwise_contrasts_are_present_in_every_condition(cubes, regime):
    out = _reduce_tau(cubes, regime)
    expected = {f"{a}-{b}" for a, b in PAIRWISE}
    assert len(expected) == 6
    for cell in {row["cell"] for row in out["method_contrasts"]}:
        got = {row["contrast"] for row in out["method_contrasts"] if row["cell"] == cell}
        assert got == expected, f"{cell} has {got}"


@pytest.mark.parametrize("regime", [REGIME_FIXED, REGIME_MOVING])
def test_every_contrast_has_a_per_group_counterpart(cubes, regime):
    out = _reduce_tau(cubes, regime)
    for cell in {row["cell"] for row in out["method_group_contrasts"]}:
        for contrast in {f"{a}-{b}" for a, b in PAIRWISE}:
            groups = {row["group"] for row in out["method_group_contrasts"]
                      if row["cell"] == cell and row["contrast"] == contrast}
            assert groups == set(GROUPS), f"{cell}/{contrast} covers {groups}"


# ------------------------------------------------------- labels and provenance

def test_nlhf_is_labelled_passive_trained_in_every_condition(cubes):
    """Including the active ones: there is no 'active NLHF' in this study."""

    for regime in (REGIME_FIXED, REGIME_MOVING):
        out = _reduce_tau(cubes, regime)
        for row in out["method_levels"]:
            if row["method"] == "pooled_nlhf":
                assert row["sampling_protocol_trained"] == "passive", row
                assert "never fitted on an active log" in row["training_protocol"]
                assert "active" not in row["display_name"].lower()


def test_uniform_is_labelled_as_not_fitted(cubes):
    out = _reduce_tau(cubes, REGIME_FIXED)
    for row in out["method_levels"]:
        if row["method"] == "uniform":
            assert row["training_protocol"] == "not fitted"
            assert row["provenance"] in ("evaluate_only", "shared")


def test_the_shared_tau_half_row_is_marked_shared_not_fitted(cubes):
    for regime in (REGIME_FIXED, REGIME_MOVING):
        out = _reduce_tau(cubes, regime)
        shared = [row for row in out["method_levels"] if row["learner_tau"] == 0.5]
        assert shared, "the shared radius produced no rows"
        for row in shared:
            assert row["provenance"] == "shared", row


def test_the_beta_glhf_columns_are_marked_imported(cubes):
    blocks = cubes["beta"]
    provenance = blocks[0].provenance
    for cell, methods in provenance.items():
        assert methods["pooled"] == "imported", cell
        assert methods["multigroup"] == "imported", cell
        # NLHF is fitted beside a passive cell and shared beside an active one.
        assert methods["pooled_nlhf"] in ("fit", "shared")
        if cell.startswith("active@"):
            assert methods["pooled_nlhf"] == "shared", cell


def test_display_names_are_the_four_specified_ones():
    assert set(reduce_module.DISPLAY_NAMES.values()) == {
        "Multigroup GLHF", "Pooled GLHF", "NLHF", "Uniform reference"}


# ----------------------------------------------------- corrections and degrees

def test_holm_ranges_over_the_six_worst_group_comparisons(cubes):
    out = _reduce_tau(cubes, REGIME_FIXED)
    for row in out["method_contrasts"]:
        assert row["holm_family_size"] == 6
        assert "six pairwise method comparisons" in row["holm_family"]
        assert 0.0 <= row["p_holm"] <= 1.0
        assert row["p_holm"] >= row["p_value_two_sided"] - 1e-12, (
            "a Holm-adjusted p-value cannot be smaller than its raw p-value")


def test_per_group_holm_ranges_over_the_selected_groups(cubes):
    out = _reduce_tau(cubes, REGIME_FIXED)
    for row in out["method_group_contrasts"]:
        assert row["holm_family_size"] == len(GROUPS)
        assert "selected groups" in row["holm_family"]


def test_degrees_of_freedom_is_one_less_than_the_number_of_splits(cubes):
    out = _reduce_tau(cubes, REGIME_FIXED)
    for row in out["method_contrasts"]:
        assert row["n_splits"] == N_SPLITS
        assert row["degrees_of_freedom"] == N_SPLITS - 1


def test_the_production_design_would_give_df_99():
    """100 split scores, df = 99 -- the specified correction."""

    scores = list(np.linspace(0.0, 1.0, 100))
    result = corrected(scores, [337] * 100, [3032] * 100, 0.95)
    assert result["n_splits"] == 100
    assert result["degrees_of_freedom"] == 99


def test_the_correction_widens_rather_than_narrows():
    """Nadeau-Bengio acknowledges the overlap; it never buys precision."""

    from empirical_testing.core.repeated_cv import nadeau_bengio

    scores = np.linspace(0.05, 0.15, 100)
    result = nadeau_bengio(scores, np.full(100, 337), np.full(100, 3032))
    assert result.corrected_se > result.naive_se


def test_split_scores_back_every_aggregate(cubes):
    out = _reduce_tau(cubes, REGIME_FIXED)
    cells = {row["cell"] for row in out["method_contrasts"]}
    for cell in cells:
        for contrast in {f"{a}-{b}" for a, b in PAIRWISE}:
            rows = [r for r in out["split_scores"]
                    if r["cell"] == cell and r["contrast"] == contrast]
            assert len(rows) == N_SPLITS, f"{cell}/{contrast} has {len(rows)} split scores"
            aggregate = next(r for r in out["method_contrasts"]
                             if r["cell"] == cell and r["contrast"] == contrast)
            assert aggregate["estimate"] == pytest.approx(
                float(np.mean([r["split_score"] for r in rows])), abs=1e-12)


# ---------------------------------------------- active minus passive contrasts

def test_active_minus_passive_exists_only_at_beta_four_and_infinity(cubes):
    blocks = cubes["beta"]
    out = reduce_protocol_contrasts(
        worst_by_split=[b.worst for b in blocks], cells=beta_conditions(),
        groups=GROUPS, levels_by_split=[b.levels for b in blocks],
        test_sizes=cubes["test_sizes"], train_sizes=cubes["train_sizes"],
        confidence_level=0.95,
    )
    betas = {row["beta"] for row in out["protocol_contrasts"]}
    assert betas == {"4", "inf"}


def test_no_active_nlhf_contrast_is_ever_formed(cubes):
    """NLHF has no active counterpart, so no such contrast may exist."""

    blocks = cubes["beta"]
    out = reduce_protocol_contrasts(
        worst_by_split=[b.worst for b in blocks], cells=beta_conditions(),
        groups=GROUPS, levels_by_split=[b.levels for b in blocks],
        test_sizes=cubes["test_sizes"], train_sizes=cubes["train_sizes"],
        confidence_level=0.95,
    )
    for row in out["protocol_contrasts"] + out["protocol_group_contrasts"]:
        assert row["method"] in ("pooled", "multigroup"), row
        assert "nlhf" not in row["contrast"].lower()


def test_a_degenerate_score_vector_reports_rather_than_inventing_a_number():
    """Too few splits must yield no p-value, not a fabricated one."""

    result = corrected([0.1], [30], [270], 0.95)
    assert result["status"] != "ok"
    assert math.isnan(result["p_value_two_sided"])
