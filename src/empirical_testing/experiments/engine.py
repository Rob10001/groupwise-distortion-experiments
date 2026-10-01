"""The execution engine: remaining fits under two evaluation regimes.

The suite reuses the completed GLHF beta grid, so this module computes the
remaining fits:

* the passive-trained NLHF benchmark at every beta (learner ``tau = 0.5``);
* active pooled and multigroup GLHF at the five learner radii other than
  ``0.5``, each from its **own** policy-dependent log;
* the passive-trained NLHF benchmark at those same five radii.

``tau = 0.5`` is never refitted in either family: the GLHF arms there *are* the
imported active/``beta = inf`` cell, and the NLHF benchmark there *is* the
passive ``beta = inf`` fit computed here once.

Two evaluation regimes, kept apart
----------------------------------

Every tau-sweep policy is scored twice, from the **same fitted policy object**:

``fixed_0.5``
    evaluation radius ``0.5`` at every learner radius.  The primary
    cross-radius comparison, because every policy is held to one robustness
    standard.

``moving``
    evaluation radius equal to the learner radius.  A separate estimand: what
    happens when the evaluation uncertainty set moves with the policy's own
    training radius.

They are different estimands and are never mixed.  The two evaluations differ
in exactly one argument -- the radius handed to ``evaluate_single_holdout`` --
and share the training data, the training radius, the seeds and the fitted
parameters by construction, because the policy cube is computed once and
scored twice.  Under ``moving`` the welfare comparator moves too, so it is
stored per radius rather than once per split.

For the beta condition the learner radius is ``0.5`` throughout, so the two
regimes coincide there; it is computed once and labelled ``fixed_0.5``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: Methods in every reported condition, in reporting order.
METHODS: tuple[str, ...] = ("multigroup", "pooled", "pooled_nlhf", "uniform")
GLHF_ARMS: tuple[str, ...] = ("pooled", "multigroup")
NLHF_ARM = "pooled_nlhf"
UNIFORM_ARM = "uniform"

#: The beta grid, learner radius 0.5 throughout.
BETA_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, math.inf)

#: The full learner-radius grid of the active tau ablation.
TAU_GRID: tuple[float, ...] = (0.05, 0.2, 0.35, 0.5, 0.75, 1.0)

#: The radius that is *not* fitted here: its GLHF arms are the imported
#: active/beta=inf cell and its NLHF benchmark is the passive beta=inf fit.
SHARED_TAU = 0.5

#: The radii this engine actually fits.
NEW_TAUS: tuple[float, ...] = tuple(t for t in TAU_GRID if t != SHARED_TAU)

#: The fixed evaluation radius, and the labels the regimes are recorded under.
FIXED_EVALUATION_TAU = 0.5
REGIME_FIXED = "fixed_0.5"
REGIME_MOVING = "moving"
REGIMES: tuple[str, ...] = (REGIME_FIXED, REGIME_MOVING)

#: Exploitability gate; ``fit_pooled_nlhf`` refuses anything above it.
from empirical_testing.algorithms.nlhf import (  # noqa: E402
    EXPLOITABILITY_TOLERANCE as NLHF_EXPLOITABILITY_GATE,
)

#: Derivative source for the NLHF solve, explicitly selected and recorded.
#:
#: **finite_difference**, as specified.  ``analytic`` was measured on
#: 2026-09-21 and is **not** a safe substitute: across one full split (500
#: fits) it exceeded the 1e-5 exploitability gate on 6 fits, worst
#: 1.7e-02, where finite-difference exceeded it on none (worst 9.5e-08).
#: The two modes are not ordered -- each beats the other by more than 10x on
#: roughly 90 of those 500 fits -- so neither is uniformly more accurate and
#: the exact derivative is not a drop-in improvement.
#:
#: This is recorded because the opposite conclusion looks obvious from a small
#: sample: on ~50 fits the two modes agree to ~1e-7 and ``analytic`` cleanly
#: fixes the one finite-difference stall that motivated looking at it at all.
NLHF_GRADIENT = "finite_difference"

IMPLEMENTATION_ID = "v10-engine-2026-09-21.4-gate2e-5"


def tau_label(tau: float) -> str:
    return f"{float(tau):g}"


def beta_label(beta: float) -> str:
    return "inf" if math.isinf(beta) else f"{float(beta):g}"


# --------------------------------------------------------------------- cells

@dataclass(frozen=True)
class FitCell:
    """One block of fitting work inside a split.

    ``protocol`` is the protocol the arms here are **trained** under.  NLHF is
    passive in every cell of this study, including the cells of the active tau
    ablation, and nothing in this module can label it otherwise.
    """

    condition: str          # "beta" | "tau"
    protocol: str           # "passive" | "active"
    beta: float
    learner_tau: float
    arms: tuple[str, ...]

    @property
    def label(self) -> str:
        if self.condition == "beta":
            return f"beta@{beta_label(self.beta)}"
        return f"tau@{tau_label(self.learner_tau)}|{self.protocol}"

    @property
    def evaluation_taus(self) -> tuple[tuple[str, float], ...]:
        """``(regime, evaluation tau)`` pairs this cell is scored under."""

        if self.condition == "beta":
            # Learner radius is 0.5 here, so the moving regime would be the
            # fixed one; it is computed once and reported as fixed.
            return ((REGIME_FIXED, FIXED_EVALUATION_TAU),)
        return (
            (REGIME_FIXED, FIXED_EVALUATION_TAU),
            (REGIME_MOVING, float(self.learner_tau)),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "cell": self.label, "condition": self.condition,
            "training_protocol": self.protocol, "beta": beta_label(self.beta),
            "learner_tau": float(self.learner_tau), "arms": list(self.arms),
        }


def beta_cells() -> tuple[FitCell, ...]:
    """The NLHF benchmark at every beta.  GLHF at these betas is imported."""

    return tuple(
        FitCell("beta", "passive", beta, FIXED_EVALUATION_TAU, (NLHF_ARM,))
        for beta in BETA_GRID
    )


def tau_cells() -> tuple[FitCell, ...]:
    """Active GLHF and passive NLHF at the five radii other than 0.5."""

    cells: list[FitCell] = []
    for tau in NEW_TAUS:
        cells.append(FitCell("tau", "active", math.inf, tau, GLHF_ARMS))
        cells.append(FitCell("tau", "passive", math.inf, tau, (NLHF_ARM,)))
    return tuple(cells)


def all_fit_cells() -> tuple[FitCell, ...]:
    return beta_cells() + tau_cells()


def fits_per_split(n_seeds: int) -> dict[str, int]:
    """New fits one split schedules, by family.  Beta GLHF is structurally 0."""

    counts = {"glhf_beta": 0, "glhf_tau": 0, "nlhf_beta": 0, "nlhf_tau": 0}
    for cell in all_fit_cells():
        for arm in cell.arms:
            family = "nlhf" if arm == NLHF_ARM else "glhf"
            counts[f"{family}_{cell.condition}"] += n_seeds
    return counts


# ------------------------------------------------------------------- fitting

def fit_split(
    *,
    algorithm: str,
    cv_seed: int,
    learner_seeds: Sequence[int],
    comparisons_per_respondent: int,
    cells: Sequence[FitCell],
    train_view: Any,
    test_view: Any,
    repeat: int,
    fold: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, np.ndarray]]:
    """Fit every arm of every cell, once per learner seed, on one split.

    Returns ``(policies, records, active_logs)`` where
    ``policies[cell_label][arm]`` is ``(n_seeds, n_test, n_actions)``.

    Sampling invariants enforced here, not merely intended:

    * the source-compatible passive ``beta = inf`` log is built **once** per
      ``(split, seed)`` and its respondent order is reused by every cell, the
      active ones included, so a protocol or radius difference is never a row
      -order difference in disguise;
    * every active fit draws its **own** policy-dependent log at its **own**
      learner radius -- the ``tau = 0.5`` active log is never replayed at
      another radius, because no active log is ever carried between cells;
    * NLHF is fitted on the **passive** log in every cell, at the cell's
      learner radius, and is never handed an active log;
    * held-out utilities are never read: the test fold contributes group
      memberships only.
    """

    from empirical_testing.algorithms.nlhf import fit_pooled_nlhf
    from empirical_testing.algorithms.registry import fit_online_learner
    from empirical_testing.experiments.coffee_crossfit import (
        make_uniform_passive_log, split_learner_seed,
    )
    from empirical_testing.experiments.coffee_protocol_beta_sweep import (
        _check_training_only_utilities, beta_label_uniforms, relabel_signs,
    )

    if int(comparisons_per_respondent) != 1:
        raise ValueError(
            "the suite engine supports exactly one comparison per training respondent; the "
            f"active protocol consumes one training row per round, so "
            f"{comparisons_per_respondent} is not a supported budget"
        )

    train_utilities = np.asarray(train_view.utilities, dtype=np.float64)
    train_groups = np.asarray(train_view.groups)
    n_train, n_actions = int(train_utilities.shape[0]), int(train_utilities.shape[1])
    test_groups = np.asarray(test_view.groups)
    n_test = int(test_groups.shape[0])
    reference = np.full(n_actions, 1.0 / n_actions)
    pooled_train_groups = np.ones((n_train, 1), dtype=np.int8)
    pooled_test_groups = np.ones((n_test, 1), dtype=np.int8)

    stacked: dict[str, dict[str, list[np.ndarray]]] = {
        cell.label: {arm: [] for arm in cell.arms} for cell in cells
    }
    active_logs: dict[str, list[np.ndarray]] = {}
    orders: list[np.ndarray] = []
    solver: dict[str, list[dict[str, Any]]] = {cell.label: [] for cell in cells}
    #: Fits whose optimizer reported non-convergence.  These are **not** failed
    #: fits: ``fit_pooled_nlhf`` raises when exploitability exceeds the gate, so
    #: anything that returns has been verified independently and is usable.
    #: SLSQP routinely reports failure at perfectly good points -- typically
    #: "Iteration limit reached" on a point already at the KL boundary -- and
    #: the exact-exploitability check is strictly stronger than its own status.
    #: Recorded so the margin is visible, never used for control flow, and
    #: never repaired by another start.
    unconverged: list[dict[str, Any]] = []

    for learner_seed in learner_seeds:
        training_seed = split_learner_seed(cv_seed, repeat, fold, learner_seed)
        # One source-compatible passive log per (split, seed).  It fixes the
        # respondent order for the whole plan, active cells included.
        log_rows = make_uniform_passive_log(
            train_view, np.arange(n_train, dtype=np.int64), seed=training_seed,
            comparisons_per_respondent=int(comparisons_per_respondent), beta=np.inf,
        )
        order = np.asarray(log_rows.respondent_indices, dtype=np.int64)
        if order.shape != (n_train,) or not np.array_equal(np.sort(order), np.arange(n_train)):
            raise AssertionError(
                "the passive log's respondent order is not a permutation of this split's "
                "training rows; the plan cannot share one order"
            )
        orders.append(order)
        ordered_utilities = train_utilities[order]
        uniforms = beta_label_uniforms(training_seed, len(log_rows.signs))

        for cell in cells:
            if cell.protocol == "active":
                # The expectation is re-read from the view rather than reusing
                # the local array, so the guard compares two independently
                # derived values instead of an expression against itself.
                _check_training_only_utilities(
                    ordered_utilities,
                    np.asarray(train_view.utilities, dtype=np.float64)[order],
                    test_view,
                )
                aligned_train_groups = train_groups[order]
                for arm, groups, prediction_groups in (
                    ("pooled", pooled_train_groups, pooled_test_groups),
                    ("multigroup", aligned_train_groups, test_groups),
                ):
                    if arm not in cell.arms:
                        continue
                    # This fit's own log, drawn by the learner from its own
                    # policy at *this* radius.  Nothing is carried in.
                    model = fit_online_learner(
                        algorithm, groups=groups, utilities=ordered_utilities,
                        beta=float(cell.beta), reference=reference,
                        tau=float(cell.learner_tau), protocol="active", seed=training_seed,
                    )
                    if not np.array_equal(
                        model.snapshot_rounds, np.arange(n_train, dtype=np.int64)
                    ):
                        raise AssertionError(
                            "the fitted policy is not the all-round pre-round average"
                        )
                    drawn = np.asarray(model.drawn_comparisons, dtype=np.float64)
                    if drawn.shape != (n_train, 4) or not np.all(np.isfinite(drawn)):
                        raise AssertionError(
                            f"the active learner recorded {drawn.shape} comparisons, not one "
                            f"per each of the {n_train} training respondents"
                        )
                    if not np.all(np.isin(drawn[:, 2], (-1.0, 1.0))):
                        raise AssertionError("a drawn label is not a strict +-1 winner")
                    if not np.all((drawn[:, 3] > 0.0) & (drawn[:, 3] <= 1.0)):
                        raise AssertionError("a drawn pair has an out-of-range propensity")
                    active_logs.setdefault(f"{cell.label}|{arm}", []).append(drawn)
                    stacked[cell.label][arm].append(
                        np.asarray(model.predict(prediction_groups), dtype=np.float64)
                    )
                continue

            # Passive cell: the shared log, relabelled when beta is finite.
            if NLHF_ARM not in cell.arms:
                raise AssertionError(f"passive suite cell {cell.label} declares no NLHF arm")
            signs = (
                log_rows.signs if math.isinf(cell.beta)
                else relabel_signs(
                    order, log_rows.pairs, train_utilities, float(cell.beta), uniforms
                )
            )
            nlhf = fit_pooled_nlhf(
                pair_indices=log_rows.pairs, signs=signs, n_alternatives=n_actions,
                reference=reference, tau=float(cell.learner_tau), seed=training_seed,
                propensities=log_rows.propensities, gradient=NLHF_GRADIENT,
            )
            diagnostics = dict(nlhf.diagnostics)
            record = {
                "learner_seed": int(learner_seed),
                "exploitability": float(nlhf.exploitability),
                "solver_success": bool(diagnostics.get("solver_success", False)),
                "solver_status": str(diagnostics.get("solver_status", "")),
                "optimizer_calls": int(diagnostics.get("optimizer_calls", -1)),
                "selection": str(diagnostics.get("selection", "")),
                "initialization": str(diagnostics.get("initialization", "")),
                "gradient_mode": str(diagnostics.get("gradient_mode", "")),
                "final_kl": float(diagnostics.get("final_kl", np.nan)),
                "unobserved_pairs": int(diagnostics.get("unobserved_pairs", 0)),
            }
            solver[cell.label].append(record)
            if not record["solver_success"]:
                # Recorded, never retried and never replaced.
                unconverged.append({"cell": cell.label, **record})
            stacked[cell.label][NLHF_ARM].append(
                np.asarray(nlhf.predict(pooled_test_groups), dtype=np.float64)
            )

    # Every seed must have seen the same respondent order rule; the orders
    # themselves differ by seed, which is the design.
    policies = {
        label: {arm: np.stack(items, axis=0) for arm, items in arms.items()}
        for label, arms in stacked.items()
    }
    for cell in cells:
        for arm in cell.arms:
            got = policies[cell.label][arm].shape
            if got != (len(learner_seeds), n_test, n_actions):
                raise AssertionError(
                    f"{cell.label}|{arm} produced policies of shape {got}, expected "
                    f"{(len(learner_seeds), n_test, n_actions)}"
                )
    records = {
        "n_train": n_train, "n_test": n_test, "n_actions": n_actions,
        "respondent_order_sha256": _order_digest(orders),
        "solver": solver, "solver_unconverged": unconverged,
        "max_abs_exploitability": max(
            (abs(r["exploitability"]) for rs in solver.values() for r in rs), default=0.0
        ),
        "exploitability_gate": NLHF_EXPLOITABILITY_GATE,
        "shared_passive_log_per_seed": True,
        "active_log_per_fit": True,
    }
    stacked_logs = {
        key: np.stack(items, axis=0) for key, items in active_logs.items()
    }
    if orders:
        stacked_logs["respondent_order"] = np.stack(orders, axis=0)
    return policies, records, stacked_logs


def _order_digest(orders: Sequence[np.ndarray]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for order in orders:
        digest.update(np.asarray(order, dtype=np.int64).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- evaluation

#: Methods whose policies this engine actually fits, per condition.  The beta
#: condition's GLHF arms are imported, so only NLHF is fitted there.
FITTED_METHODS_BY_CONDITION: Mapping[str, tuple[str, ...]] = {
    "beta": (NLHF_ARM,),
    "tau": ("pooled", "multigroup", NLHF_ARM),
}


@dataclass(frozen=True)
class ReportCell:
    """One reportable point: a condition, a radius/beta, and the arms scored there.

    A report cell collects the arms of one or two :class:`FitCell` blocks.  In
    the tau condition that is the active GLHF pair plus the passive NLHF
    benchmark, which is why the two are scored in a single call and therefore
    share one welfare comparator.
    """

    condition: str
    beta: float
    learner_tau: float
    #: ``arm -> fit-cell label`` for the arms whose policies were fitted here.
    sources: Mapping[str, str]

    @property
    def label(self) -> str:
        if self.condition == "beta":
            return f"beta@{beta_label(self.beta)}"
        return f"tau@{tau_label(self.learner_tau)}"


def report_cells() -> tuple[ReportCell, ...]:
    """Group the fit cells into the points the reduction reports."""

    cells: list[ReportCell] = []
    for beta in BETA_GRID:
        source = FitCell("beta", "passive", beta, FIXED_EVALUATION_TAU, (NLHF_ARM,)).label
        cells.append(ReportCell("beta", beta, FIXED_EVALUATION_TAU, {NLHF_ARM: source}))
    for tau in NEW_TAUS:
        active = FitCell("tau", "active", math.inf, tau, GLHF_ARMS).label
        passive = FitCell("tau", "passive", math.inf, tau, (NLHF_ARM,)).label
        cells.append(ReportCell(
            "tau", math.inf, tau,
            {"pooled": active, "multigroup": active, NLHF_ARM: passive},
        ))
    return tuple(cells)


def _evaluate_cell(
    *,
    cell: ReportCell,
    policies: Mapping[str, Mapping[str, np.ndarray]],
    test_view: Any,
    group_names: Sequence[str],
    learner_seeds: Sequence[int],
    evaluation_tau: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Score one report cell's arms at one evaluation radius.

    Returns ``(arrays, real_methods)``.  ``real_methods`` names the columns
    that correspond to a policy this engine actually fitted; any other column
    is padding and is never read.

    Why padding can be necessary: :func:`evaluate_single_holdout` requires the
    two primary arms to be present in the cube, because its paired
    pooled-minus-multigroup diagnostic is part of its contract.  In the beta
    condition those arms are *imported*, not refitted, so the slots are filled
    with the NLHF policy purely to satisfy that contract.  Distortion for a
    method depends only on that method's own policy, so the padding cannot
    contaminate the columns that are read -- and the padded columns are dropped
    before anything is saved.
    """

    from empirical_testing.reporting import evaluate_single_holdout

    test_groups = np.asarray(test_view.groups)
    n_test = int(test_groups.shape[0])
    n_actions = int(test_view.utilities.shape[1])
    n_seeds = len(learner_seeds)

    fitted = {
        arm: np.asarray(policies[source][arm], dtype=np.float64)
        for arm, source in cell.sources.items()
    }
    uniform_policy = np.broadcast_to(
        np.full(n_actions, 1.0 / n_actions), (n_seeds, n_test, n_actions)
    ).copy()

    padding_donor = fitted.get("pooled", fitted[NLHF_ARM])
    arms: dict[str, np.ndarray] = {}
    real: list[str] = []
    for method in METHODS:
        if method == UNIFORM_ARM:
            arms[method] = uniform_policy
            real.append(method)
        elif method in fitted:
            arms[method] = fitted[method]
            real.append(method)
        else:
            arms[method] = padding_donor.copy()
    evaluation = evaluate_single_holdout(
        test_view.utilities, test_groups.astype(bool), arms,
        group_names=list(group_names), method_names=list(METHODS),
        seed_ids=list(learner_seeds), tau=float(evaluation_tau),
    )
    arrays = {
        "worst_group_distortion": np.asarray(evaluation.worst_group_distortion, np.float64),
        "group_distortion": np.asarray(evaluation.distortion, np.float64),
        "welfare": np.asarray(evaluation.welfare, np.float64),
        "welfare_optimum": np.asarray(evaluation.welfare_optimum, np.float64),
        "worst_group_index": np.asarray(evaluation.worst_group_index, np.int64),
        "distortion_status": np.asarray(evaluation.distortion_status),
    }
    return arrays, real


def compute_split(
    *,
    algorithm: str,
    cv_seed: int,
    learner_seeds: Sequence[int],
    comparisons_per_respondent: int,
    group_names: Sequence[str],
    train_view: Any,
    test_view: Any,
    repeat: int,
    fold: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit and score one split under both evaluation regimes.

    The policy cube is built once and scored twice.  Nothing between the two
    scorings touches the training data, the training radius, the seeds or the
    fitted parameters: the second call differs from the first in the single
    ``tau`` argument handed to the evaluator.
    """

    cells = all_fit_cells()
    started = time.perf_counter()
    policies, records, active_logs = fit_split(
        algorithm=algorithm, cv_seed=cv_seed, learner_seeds=learner_seeds,
        comparisons_per_respondent=comparisons_per_respondent, cells=cells,
        train_view=train_view, test_view=test_view, repeat=repeat, fold=fold,
    )
    fit_seconds = time.perf_counter() - started

    n_seeds, n_groups = len(learner_seeds), len(group_names)
    n_methods = len(METHODS)
    reports = report_cells()
    beta_reports = [c for c in reports if c.condition == "beta"]
    tau_reports = [c for c in reports if c.condition == "tau"]

    arrays: dict[str, np.ndarray] = {}
    real_columns: dict[str, list[str]] = {}

    started = time.perf_counter()

    def _block(cells_subset, regime, evaluation_tau_of):
        worst = np.full((len(cells_subset), n_seeds, n_methods), np.nan)
        levels = np.full((len(cells_subset), n_seeds, n_methods, n_groups), np.nan)
        welfare = np.full((len(cells_subset), n_seeds, n_methods, n_groups), np.nan)
        optimum = np.full((len(cells_subset), n_groups), np.nan)
        index = np.full((len(cells_subset), n_seeds, n_methods), -1, dtype=np.int64)
        taus = np.full(len(cells_subset), np.nan)
        for position, cell in enumerate(cells_subset):
            evaluation_tau = evaluation_tau_of(cell)
            block, real = _evaluate_cell(
                cell=cell, policies=policies, test_view=test_view,
                group_names=group_names, learner_seeds=learner_seeds,
                evaluation_tau=evaluation_tau,
            )
            real_columns[f"{regime}|{cell.label}"] = real
            # Padded columns are blanked here, before anything is saved: a
            # column that stands for an imported policy must never carry a
            # plausible-looking number.
            keep = np.array([method in real for method in METHODS])
            worst[position] = np.where(keep[None, :], block["worst_group_distortion"], np.nan)
            levels[position] = np.where(
                keep[None, :, None], block["group_distortion"], np.nan)
            welfare[position] = np.where(keep[None, :, None], block["welfare"], np.nan)
            index[position] = np.where(keep[None, :], block["worst_group_index"], -1)
            optimum[position] = block["welfare_optimum"]
            taus[position] = float(evaluation_tau)
        return worst, levels, welfare, optimum, index, taus

    # --- beta condition: learner radius 0.5, one regime -------------------
    worst, levels, welfare, optimum, index, taus = _block(
        beta_reports, REGIME_FIXED, lambda _c: FIXED_EVALUATION_TAU
    )
    arrays.update({
        "beta_worst_group_distortion": worst, "beta_group_distortion": levels,
        "beta_welfare": welfare, "beta_welfare_optimum": optimum,
        "beta_worst_group_index": index, "beta_evaluation_tau": taus,
    })
    # One evaluation radius across the beta axis, so the comparator cannot move.
    if not np.allclose(optimum, optimum[0], rtol=0.0, atol=0.0):
        raise AssertionError("the welfare comparator moved across the beta axis")

    # --- tau condition: both regimes, same policies -----------------------
    for regime in REGIMES:
        evaluation_tau_of = (
            (lambda _c: FIXED_EVALUATION_TAU) if regime == REGIME_FIXED
            else (lambda c: float(c.learner_tau))
        )
        worst, levels, welfare, optimum, index, taus = _block(
            tau_reports, regime, evaluation_tau_of
        )
        prefix = f"tau_{'fixed' if regime == REGIME_FIXED else 'moving'}"
        arrays.update({
            f"{prefix}_worst_group_distortion": worst,
            f"{prefix}_group_distortion": levels,
            f"{prefix}_welfare": welfare,
            f"{prefix}_welfare_optimum": optimum,
            f"{prefix}_worst_group_index": index,
            f"{prefix}_evaluation_tau": taus,
        })
        uniform_column = worst[:, :, METHODS.index(UNIFORM_ARM)]
        if regime == REGIME_FIXED:
            # One evaluation radius across the whole axis: the untrained
            # uniform arm and the comparator must both be constant.  Under the
            # moving regime they legitimately are not, which is exactly why the
            # two regimes are separate arrays and this check is scoped here.
            if not np.allclose(uniform_column, uniform_column[0], rtol=0.0, atol=0.0):
                raise AssertionError(
                    "the untrained uniform benchmark varied across radii at a fixed "
                    "evaluation radius"
                )
            if not np.allclose(optimum, optimum[0], rtol=0.0, atol=0.0):
                raise AssertionError(
                    "the welfare comparator moved across radii at a fixed evaluation radius"
                )
            if not np.allclose(taus, FIXED_EVALUATION_TAU):
                raise AssertionError("the fixed regime used a radius other than 0.5")
        else:
            if not np.allclose(taus, [c.learner_tau for c in tau_reports]):
                raise AssertionError(
                    "the moving regime's evaluation radius does not equal the learner radius"
                )
    evaluate_seconds = time.perf_counter() - started

    for key, value in arrays.items():
        if key.endswith("worst_group_index") or key.endswith("evaluation_tau"):
            continue
        finite_or_padded = np.isfinite(value) | np.isnan(value)
        if not np.all(finite_or_padded):
            raise AssertionError(f"split r{repeat} f{fold}: {key} holds a nonfinite value")

    train_counts = np.asarray(train_view.groups).astype(bool).sum(axis=0).astype(np.int64)
    test_counts = np.asarray(test_view.groups).astype(bool).sum(axis=0).astype(np.int64)
    arrays["train_group_counts"] = train_counts
    arrays["test_group_counts"] = test_counts

    summary = {
        "repeat": int(repeat), "fold": int(fold),
        "implementation_id": IMPLEMENTATION_ID,
        "methods": list(METHODS),
        "regimes": list(REGIMES),
        "beta_cells": [c.label for c in beta_reports],
        "tau_cells": [c.label for c in tau_reports],
        "learner_taus": [float(c.learner_tau) for c in tau_reports],
        "fixed_evaluation_tau": FIXED_EVALUATION_TAU,
        "nlhf_gradient": NLHF_GRADIENT,
        "real_columns": real_columns,
        "learner_seeds": [int(s) for s in learner_seeds],
        "groups": list(group_names),
        "n_train": records["n_train"], "n_test": records["n_test"],
        "respondent_order_sha256": records["respondent_order_sha256"],
        "solver": records["solver"],
        "solver_unconverged": records["solver_unconverged"],
        "n_solver_unconverged": len(records["solver_unconverged"]),
        "max_abs_exploitability": records["max_abs_exploitability"],
        "exploitability_gate": records["exploitability_gate"],
        "fit_seconds": fit_seconds, "evaluate_seconds": evaluate_seconds,
        "fits": fits_per_split(len(learner_seeds)),
    }
    return summary, arrays, active_logs


# ---------------------------------------------------- atomic units and resume

#: Everything that has to agree before a saved unit may be reused.  A unit
#: computed under a different grid, regime set, seed list or implementation is
#: never silently adopted.
REUSE_KEYS: tuple[str, ...] = (
    "protocol_id", "implementation_id", "base_protocol_id", "config_sha256",
    "base_config_sha256", "data_sha256", "repeat", "fold",
    "train_ids_sha256", "test_ids_sha256", "discovery_ids_sha256",
    "groups", "learner_seeds", "methods", "regimes",
    "beta_cells", "tau_cells", "learner_taus", "fixed_evaluation_tau",
    "nlhf_gradient",
)


def split_dir(output_dir: Path, repeat: int, fold: int) -> Path:
    """The repeated-CV layout, so a suite split pairs with its source by name."""

    return Path(output_dir) / "splits" / f"r{int(repeat):02d}_f{int(fold):02d}"


def expected_array_shapes(*, n_betas: int, n_taus: int, n_seeds: int,
                          n_groups: int) -> dict[str, tuple[int, ...]]:
    n_methods = len(METHODS)
    shapes: dict[str, tuple[int, ...]] = {
        "beta_worst_group_distortion": (n_betas, n_seeds, n_methods),
        "beta_group_distortion": (n_betas, n_seeds, n_methods, n_groups),
        "beta_welfare_optimum": (n_betas, n_groups),
        "beta_evaluation_tau": (n_betas,),
        "train_group_counts": (n_groups,),
        "test_group_counts": (n_groups,),
    }
    for prefix in ("tau_fixed", "tau_moving"):
        shapes.update({
            f"{prefix}_worst_group_distortion": (n_taus, n_seeds, n_methods),
            f"{prefix}_group_distortion": (n_taus, n_seeds, n_methods, n_groups),
            f"{prefix}_welfare_optimum": (n_taus, n_groups),
            f"{prefix}_evaluation_tau": (n_taus,),
        })
    return shapes


def saved_split(directory: Path, expected: Mapping[str, Any], *, n_betas: int,
                n_taus: int, n_seeds: int, n_groups: int) -> dict[str, Any] | None:
    """Reuse a saved unit only when its declared identity *and* arrays check out.

    A half-written bundle is not an error: it is a unit that cannot be reused,
    so it is recomputed.  ``split.json`` is written last, so its presence is
    the completion marker.
    """

    import zipfile

    summary_path, arrays_path = directory / "split.json", directory / "split.npz"
    if not (summary_path.is_file() and arrays_path.is_file()):
        return None
    try:
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key, value in expected.items():
        if saved.get(key) != value:
            return None
    shapes = expected_array_shapes(
        n_betas=n_betas, n_taus=n_taus, n_seeds=n_seeds, n_groups=n_groups)
    try:
        with np.load(arrays_path, allow_pickle=False) as bundle:
            for key, shape in shapes.items():
                if key not in bundle.files or np.asarray(bundle[key]).shape != shape:
                    return None
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return None
    return saved


def publish_split(directory: Path, summary: Mapping[str, Any],
                  arrays: Mapping[str, np.ndarray],
                  active_logs: Mapping[str, np.ndarray]) -> None:
    """Write one split atomically: arrays first, ``split.json`` last."""

    import shutil

    from empirical_testing.experiments.coffee_crossfit import _publish

    staging = directory.parent / f".partial-{directory.name}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    for path in (staging, directory):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir()
    np.savez_compressed(staging / "split.npz", **arrays)
    if active_logs:
        compact: dict[str, np.ndarray] = {}
        for key, value in active_logs.items():
            if key == "respondent_order":
                compact[key] = np.asarray(value, dtype=np.int32)
                continue
            block = np.asarray(value, dtype=np.float64)
            compact[f"{key}|first"] = block[..., 0].astype(np.int16)
            compact[f"{key}|second"] = block[..., 1].astype(np.int16)
            compact[f"{key}|sign"] = block[..., 2].astype(np.int8)
            compact[f"{key}|propensity"] = block[..., 3]
        np.savez_compressed(staging / "comparisons.npz", **compact)
    (staging / "split.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    _publish(staging, directory)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def split_arrays(output_dir: Path, repeat: int, fold: int) -> dict[str, np.ndarray]:
    with np.load(split_dir(output_dir, repeat, fold) / "split.npz",
                 allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


# ----------------------------------------------------------------- the runner

@dataclass
class SplitPlan:
    """Everything one worker needs to compute a split of one dataset."""

    dataset: str
    base_config: Any
    inputs: Any
    output_dir: Path
    protocol_id: str

    @property
    def learner_seeds(self) -> tuple[int, ...]:
        return tuple(self.base_config.learner_seeds)

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self.base_config.groups)


def _views(plan: SplitPlan, repeat: int, fold: int):
    from empirical_testing.core.holdout import subset_dense_dataset

    reservation, dataset = plan.inputs.reservation, plan.inputs.dataset
    train_pos, test_pos = plan.inputs.plan.split_positions(repeat, fold)
    train_rows = reservation.analysis_indices[train_pos]
    test_rows = reservation.analysis_indices[test_pos]
    if np.intersect1d(train_rows, test_rows).size:
        raise AssertionError("a respondent appears in both the training set and the test fold")
    if np.intersect1d(np.concatenate((train_rows, test_rows)),
                      reservation.discovery_indices).size:
        raise AssertionError("a reserved discovery respondent entered a suite split")
    return (subset_dense_dataset(dataset, train_rows),
            subset_dense_dataset(dataset, test_rows),
            train_rows, test_rows)


def expected_identity(plan: SplitPlan, repeat: int, fold: int) -> dict[str, Any]:
    from empirical_testing.core.holdout import sorted_id_sha256

    base = plan.base_config
    ids = np.asarray(plan.inputs.dataset.respondent_ids).astype(str)
    reservation = plan.inputs.reservation
    train_pos, test_pos = plan.inputs.plan.split_positions(repeat, fold)
    reports = report_cells()
    return {
        "protocol_id": plan.protocol_id,
        "implementation_id": IMPLEMENTATION_ID,
        "base_protocol_id": base.route.protocol_id,
        "config_sha256": base.digest(),
        "base_config_sha256": base.digest(),
        "data_sha256": plan.inputs.data_sha256,
        "repeat": int(repeat), "fold": int(fold),
        "train_ids_sha256": sorted_id_sha256(ids[reservation.analysis_indices[train_pos]]),
        "test_ids_sha256": sorted_id_sha256(ids[reservation.analysis_indices[test_pos]]),
        "discovery_ids_sha256": sorted_id_sha256(ids[reservation.discovery_indices]),
        "groups": list(base.groups),
        "learner_seeds": [int(s) for s in base.learner_seeds],
        "methods": list(METHODS),
        "regimes": list(REGIMES),
        "beta_cells": [c.label for c in reports if c.condition == "beta"],
        "tau_cells": [c.label for c in reports if c.condition == "tau"],
        "learner_taus": [float(c.learner_tau) for c in reports if c.condition == "tau"],
        "fixed_evaluation_tau": FIXED_EVALUATION_TAU,
        "nlhf_gradient": NLHF_GRADIENT,
    }


def run_split(plan: SplitPlan, repeat: int, fold: int) -> dict[str, Any]:
    """Compute one split unless an identical one is already saved; publish atomically."""

    directory = split_dir(plan.output_dir, repeat, fold)
    expected = expected_identity(plan, repeat, fold)
    reports = report_cells()
    reused = saved_split(
        directory, expected,
        n_betas=len([c for c in reports if c.condition == "beta"]),
        n_taus=len([c for c in reports if c.condition == "tau"]),
        n_seeds=len(plan.learner_seeds), n_groups=len(plan.groups),
    )
    if reused is not None:
        reused["reused"] = True
        return reused

    train_view, test_view, _, _ = _views(plan, repeat, fold)
    base = plan.base_config
    summary, arrays, active_logs = compute_split(
        algorithm=base.algorithm, cv_seed=base.cv_seed,
        learner_seeds=base.learner_seeds,
        comparisons_per_respondent=base.comparisons_per_respondent,
        group_names=base.groups, train_view=train_view, test_view=test_view,
        repeat=repeat, fold=fold,
    )
    summary.update(expected)
    publish_split(directory, summary, arrays, active_logs)
    summary["reused"] = False
    return summary


_WORKER_STATE: dict[str, Any] = {}


def _init_worker(dataset: str, config_mapping: Mapping[str, Any], project_root: str,
                 output_dir: str, protocol_id: str,
                 expected_data_sha256: str | None) -> None:
    if _WORKER_STATE.get("output_dir") == output_dir:
        return
    plan = build_plan(
        dataset, project_root=Path(project_root), output_dir=Path(output_dir),
        protocol_id=protocol_id, config_mapping=config_mapping,
        expected_data_sha256=expected_data_sha256,
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(plan=plan, output_dir=output_dir)


def _run_split_task(task: tuple[int, int]) -> dict[str, Any]:
    repeat, fold = task
    return run_split(_WORKER_STATE["plan"], repeat, fold)


def build_plan(dataset: str, *, project_root: Path, output_dir: Path, protocol_id: str,
               config_mapping: Mapping[str, Any] | None = None,
               config_path: Path | None = None,
               expected_data_sha256: str | None = None) -> SplitPlan:
    """Load one dataset's frozen design and bind it to a suite output directory."""

    if dataset == "coffee":
        from empirical_testing.experiments.coffee_crossfit import (
            CoffeeCrossfitConfig, load_crossfit_config, load_crossfit_inputs,
        )
        base = (CoffeeCrossfitConfig.from_mapping(config_mapping)
                if config_mapping is not None else load_crossfit_config(config_path))
        inputs = load_crossfit_inputs(
            base, project_root=Path(project_root),
            expected_data_sha256=expected_data_sha256,
        )
    elif dataset == "persona":
        from empirical_testing.experiments.persona_crossfit import (
            PersonaCrossfitConfig, load_persona_config, load_persona_inputs,
            select_groups_from_inputs,
        )
        base = (PersonaCrossfitConfig.from_mapping(config_mapping)
                if config_mapping is not None else load_persona_config(config_path))
        if not base.groups:
            # Persona's family is *discovery-selected*, not declared in the
            # configuration: it is re-derived here by the frozen rule, from the
            # reserved discovery sample only.  A worker receives the already
            # resolved family in ``config_mapping`` and skips this.
            discovery_inputs = load_persona_inputs(
                base, project_root=Path(project_root),
                group_names=base.candidate_groups,
            )
            selection = select_groups_from_inputs(base, discovery_inputs)
            base = base.with_groups(selection.selected)
        inputs = load_persona_inputs(
            base, project_root=Path(project_root), group_names=base.groups,
        )
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'coffee' or 'persona'")
    return SplitPlan(dataset=dataset, base_config=base, inputs=inputs,
                     output_dir=Path(output_dir), protocol_id=protocol_id)
