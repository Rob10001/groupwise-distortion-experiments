"""Combined four-method suite reductions, under two separated evaluation regimes.

The numbers suite reports come from two places and are joined here:

* **imported** — pooled and multigroup GLHF, and the uniform reference, from the
  completed beta run now living at ``results/*-protocol-beta-v10/source``;
* **newly fitted** — the passive-trained NLHF benchmark at every beta, and
  active GLHF plus passive NLHF at the five learner radii other than ``0.5``.

Two results are *shared* rather than recomputed, and are counted once: the
``tau = 0.5`` GLHF arms are the imported active/``beta = inf`` cell, and the
``tau = 0.5`` NLHF benchmark is the passive ``beta = inf`` fit.

The two evaluation regimes are different estimands and are reduced separately,
into separate files.  ``fixed_0.5`` scores every policy at evaluation radius
``0.5``; ``moving`` scores each policy at its own learner radius.  Nothing in
this module puts a row of one regime into a table of the other.

Reduction order, unchanged from the frozen routes: maximize over groups within
each ``(seed, method)``, take the paired difference for a contrast, average
over learner seeds, and only then reduce to **one number per split**.  The
Nadeau-Bengio correction is applied to those 100 split scores with the actual
fold sizes and ``df = 99``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from empirical_testing.experiments.engine import (
    BETA_GRID, FIXED_EVALUATION_TAU, METHODS, NEW_TAUS, REGIME_FIXED,
    REGIME_MOVING, REGIMES, SHARED_TAU, TAU_GRID, beta_label, tau_label,
)

#: The six pairwise method contrasts, in reporting order.  Holm ranges over
#: exactly these six within each condition.
PAIRWISE: tuple[tuple[str, str], ...] = (
    ("pooled", "multigroup"),
    ("pooled_nlhf", "multigroup"),
    ("uniform", "multigroup"),
    ("pooled_nlhf", "pooled"),
    ("uniform", "pooled"),
    ("uniform", "pooled_nlhf"),
)

#: The imported run's own method axis, in its saved order.
IMPORTED_METHODS: tuple[str, ...] = ("pooled", "multigroup", "uniform")

#: The imported run's cell axis, in its saved order.
IMPORTED_CELLS: tuple[str, ...] = (
    "passive@0.5", "passive@1", "passive@2", "passive@4", "passive@inf",
    "active@4", "active@inf",
)

SCHEMA_VERSION = "v10"

# Independently recomputing the same welfare optimum across NumPy/libm builds
# can differ in the last few float64 bits. This comparison-only tolerance is
# far below the reported distortion precision; it does not alter either solve,
# the saved comparators, or any optimizer acceptance gate.
COMPARATOR_RTOL = 1e-12
COMPARATOR_ATOL = 1e-12


@dataclass(frozen=True)
class Condition:
    """One reportable point of one condition, under one evaluation regime."""

    condition: str          # "beta" | "tau"
    regime: str             # "fixed_0.5" | "moving"
    protocol: str           # the protocol the GLHF arms here were trained under
    beta: float
    learner_tau: float
    evaluation_tau: float

    @property
    def label(self) -> str:
        if self.condition == "beta":
            return f"{self.protocol}@{beta_label(self.beta)}"
        return f"tau@{tau_label(self.learner_tau)}"

    def as_row(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "evaluation_regime": self.regime,
            "cell": self.label,
            "protocol": self.protocol,
            "beta": beta_label(self.beta),
            "learner_tau": float(self.learner_tau),
            "evaluation_tau": float(self.evaluation_tau),
        }


#: Display names and training descriptions, so every table says how a method
#: was trained rather than leaving it to the reader's memory.
DISPLAY_NAMES: Mapping[str, str] = {
    "multigroup": "Multigroup GLHF", "pooled": "Pooled GLHF",
    "pooled_nlhf": "NLHF", "uniform": "Uniform reference",
}


def training_protocol(method: str, cell_protocol: str) -> str:
    """How ``method`` was trained in a cell whose GLHF arms used ``cell_protocol``.

    NLHF is passive-trained in **every** condition of this study, so it never
    inherits the cell's protocol; the uniform reference is not trained at all.
    """

    if method == "uniform":
        return "not fitted"
    if method == "pooled_nlhf":
        return "passive-trained batch solver (never fitted on an active log)"
    return f"{cell_protocol}-trained span-adaptive learner"


def method_row(cell: Condition, method: str, provenance: str) -> dict[str, Any]:
    """The identity columns every reported row carries for one method."""

    displayed_beside = (
        "active" if (method in ("pooled_nlhf", "uniform") and cell.protocol == "active")
        else ""
    )
    return {
        **cell.as_row(),
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        # The *training* protocol, which for NLHF is passive everywhere.
        "training_protocol": training_protocol(method, cell.protocol),
        "sampling_protocol_trained": (
            "passive" if method in ("pooled_nlhf", "uniform") else cell.protocol
        ),
        "provenance": provenance,
        "displayed_beside_protocol": displayed_beside,
        "schema_version": SCHEMA_VERSION,
    }


def beta_conditions() -> tuple[Condition, ...]:
    """Beta/protocol condition: passive at every beta, active at 4 and inf."""

    cells = [
        Condition("beta", REGIME_FIXED, "passive", b, FIXED_EVALUATION_TAU,
                  FIXED_EVALUATION_TAU)
        for b in BETA_GRID
    ]
    cells += [
        Condition("beta", REGIME_FIXED, "active", b, FIXED_EVALUATION_TAU,
                  FIXED_EVALUATION_TAU)
        for b in (4.0, math.inf)
    ]
    return tuple(cells)


def tau_conditions(regime: str) -> tuple[Condition, ...]:
    """Active tau ablation, under one regime.  Evaluation radius depends on it."""

    if regime not in REGIMES:
        raise ValueError(f"unknown evaluation regime {regime!r}; expected one of {REGIMES}")
    return tuple(
        Condition(
            "tau", regime, "active", math.inf, t,
            FIXED_EVALUATION_TAU if regime == REGIME_FIXED else float(t),
        )
        for t in TAU_GRID
    )


# ------------------------------------------------------- assembling one split

@dataclass
class SplitCube:
    """One split's four-method values for one condition under one regime.

    ``worst`` is ``(n_cells, n_seeds, 4)`` and ``levels`` is
    ``(n_cells, n_seeds, 4, n_groups)``, with the method axis in
    :data:`empirical_testing.experiments.engine.METHODS` order.
    ``provenance[cell][method]`` records where each column came from.
    """

    cells: tuple[Condition, ...]
    worst: np.ndarray
    levels: np.ndarray
    provenance: dict[str, dict[str, str]]
    n_train: int
    n_test: int


def load_imported_split(source_dir: Path, repeat: int, fold: int) -> dict[str, np.ndarray]:
    """One split of the imported beta run, read from disk and never recomputed."""

    path = Path(source_dir) / "splits" / f"r{int(repeat):02d}_f{int(fold):02d}" / "split.npz"
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def imported_cell_labels(source_dir: Path, repeat: int, fold: int) -> list[str]:
    path = Path(source_dir) / "splits" / f"r{int(repeat):02d}_f{int(fold):02d}" / "split.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["cell_labels"])


def check_comparator_matches(new_arrays: Mapping[str, np.ndarray],
                             imported: Mapping[str, np.ndarray],
                             labels: Sequence[str]) -> dict[str, Any]:
    """The newly fitted columns must be scored against the imported comparator.

    Distortion is a ratio against a welfare comparator that is refit per test
    fold at the evaluation radius.  If the suite's own evaluation produced a
    different comparator from the imported run's, the imported GLHF numbers and
    the new NLHF numbers would not be commensurable and every contrast between
    them would be meaningless.  They are computed independently, so the
    agreement is checked rather than assumed. Both solve the same test fold at
    tau = 0.5, but platform roundoff may prevent bitwise identity. Numerical
    agreement uses the tight tolerances above; exact identity and the observed
    maximum difference are retained as separate diagnostics.
    """

    mine = np.asarray(new_arrays["beta_welfare_optimum"])
    theirs = np.asarray(imported["welfare_optimum"])[labels.index("passive@inf")]
    mine = mine[0]
    equal = bool(np.array_equal(mine, theirs))
    comparable = (
        mine.shape == theirs.shape and mine.size > 0
        and bool(np.isfinite(mine).all()) and bool(np.isfinite(theirs).all())
    )
    matches = comparable and bool(np.allclose(
        mine, theirs, rtol=COMPARATOR_RTOL, atol=COMPARATOR_ATOL, equal_nan=False,
    ))
    return {
        "exact": equal,
        "matches": matches,
        "rtol": COMPARATOR_RTOL,
        "atol": COMPARATOR_ATOL,
        "max_abs_difference": float(np.max(np.abs(mine - theirs))) if comparable else float("inf"),
        "comparator": theirs.tolist(),
    }


def _imported_column(imported: Mapping[str, np.ndarray], labels: Sequence[str],
                     cell: str, method: str, key: str) -> np.ndarray:
    """One (cell, method) slice of an imported array, selected by *name*.

    Positional indexing would silently return the wrong arm if the saved axis
    order ever differed from this module's assumption, so both axes are looked
    up by label.
    """

    if cell not in labels:
        raise KeyError(f"the imported run has no cell {cell!r}; it has {list(labels)}")
    if method not in IMPORTED_METHODS:
        raise KeyError(f"the imported run has no method {method!r}")
    return imported[key][labels.index(cell), :, IMPORTED_METHODS.index(method)]


def _require_seed_agreement(new_arrays: Mapping[str, np.ndarray],
                            imported: Mapping[str, np.ndarray]) -> int:
    """The two sources must describe the same learner seeds, in the same order.

    A mismatch here would pair a newly fitted seed against a different imported
    one, which no later check would catch.
    """

    mine = int(np.asarray(new_arrays["beta_worst_group_distortion"]).shape[1])
    theirs = int(np.asarray(imported["worst_group_distortion"]).shape[1])
    if mine != theirs:
        raise ValueError(
            f"the suite fits carry {mine} learner seeds but the imported run carries "
            f"{theirs}; they cannot be joined. A reduced-seed run is a pilot and is "
            f"not reportable against the full imported grid."
        )
    return mine


def assemble_beta_split(
    *, imported: Mapping[str, np.ndarray], labels: Sequence[str],
    new_arrays: Mapping[str, np.ndarray], n_seeds: int, n_groups: int,
) -> SplitCube:
    """The beta condition's four-method cube for one split.

    GLHF and uniform come from the imported run; NLHF is the newly fitted
    passive benchmark.  Beside an **active** cell the NLHF column is the *same*
    passive fit at that beta -- shared, not refitted, and never relabelled as
    an active fit.
    """

    _require_seed_agreement(new_arrays, imported)
    cells = beta_conditions()
    worst = np.full((len(cells), n_seeds, len(METHODS)), np.nan)
    levels = np.full((len(cells), n_seeds, len(METHODS), n_groups), np.nan)
    provenance: dict[str, dict[str, str]] = {}
    beta_axis = [beta_label(b) for b in BETA_GRID]

    for position, cell in enumerate(cells):
        provenance[cell.label] = {}
        nlhf_index = beta_axis.index(beta_label(cell.beta))
        for method_index, method in enumerate(METHODS):
            if method == "pooled_nlhf":
                worst[position, :, method_index] = (
                    new_arrays["beta_worst_group_distortion"][
                        nlhf_index, :, METHODS.index("pooled_nlhf")]
                )
                levels[position, :, method_index] = (
                    new_arrays["beta_group_distortion"][
                        nlhf_index, :, METHODS.index("pooled_nlhf")]
                )
                provenance[cell.label][method] = (
                    "shared" if cell.protocol == "active" else "fit"
                )
                continue
            worst[position, :, method_index] = _imported_column(
                imported, labels, cell.label, method, "worst_group_distortion")
            levels[position, :, method_index] = _imported_column(
                imported, labels, cell.label, method, "group_distortion")
            provenance[cell.label][method] = (
                "evaluate_only" if method == "uniform" else "imported"
            )
    return SplitCube(cells=cells, worst=worst, levels=levels, provenance=provenance,
                     n_train=0, n_test=0)


def assemble_tau_split(
    *, regime: str, imported: Mapping[str, np.ndarray], labels: Sequence[str],
    new_arrays: Mapping[str, np.ndarray], n_seeds: int, n_groups: int,
) -> SplitCube:
    """The active tau ablation's four-method cube for one split, under one regime.

    At ``tau = 0.5`` every column is shared rather than refitted: the GLHF arms
    and the uniform reference are the imported active/``beta = inf`` cell, and
    the NLHF benchmark is the passive ``beta = inf`` fit.  Because that radius
    is evaluated at ``0.5`` in either regime, the same imported numbers serve
    both -- which is what makes the two regimes agree there by construction.
    """

    if regime not in REGIMES:
        raise ValueError(f"unknown evaluation regime {regime!r}")
    prefix = "tau_fixed" if regime == REGIME_FIXED else "tau_moving"
    _require_seed_agreement(new_arrays, imported)
    cells = tau_conditions(regime)
    worst = np.full((len(cells), n_seeds, len(METHODS)), np.nan)
    levels = np.full((len(cells), n_seeds, len(METHODS), n_groups), np.nan)
    provenance: dict[str, dict[str, str]] = {}
    new_axis = [tau_label(t) for t in NEW_TAUS]
    beta_axis = [beta_label(b) for b in BETA_GRID]

    for position, cell in enumerate(cells):
        provenance[cell.label] = {}
        shared = float(cell.learner_tau) == SHARED_TAU
        for method_index, method in enumerate(METHODS):
            if shared:
                if method == "pooled_nlhf":
                    source_index = beta_axis.index("inf")
                    worst[position, :, method_index] = new_arrays[
                        "beta_worst_group_distortion"][
                            source_index, :, METHODS.index("pooled_nlhf")]
                    levels[position, :, method_index] = new_arrays[
                        "beta_group_distortion"][
                            source_index, :, METHODS.index("pooled_nlhf")]
                else:
                    worst[position, :, method_index] = _imported_column(
                        imported, labels, "active@inf", method, "worst_group_distortion")
                    levels[position, :, method_index] = _imported_column(
                        imported, labels, "active@inf", method, "group_distortion")
                provenance[cell.label][method] = "shared"
                continue
            source_index = new_axis.index(tau_label(cell.learner_tau))
            worst[position, :, method_index] = new_arrays[
                f"{prefix}_worst_group_distortion"][source_index, :, method_index]
            levels[position, :, method_index] = new_arrays[
                f"{prefix}_group_distortion"][source_index, :, method_index]
            provenance[cell.label][method] = (
                "evaluate_only" if method == "uniform" else "fit"
            )
    return SplitCube(cells=cells, worst=worst, levels=levels, provenance=provenance,
                     n_train=0, n_test=0)


# -------------------------------------------------------------- the reduction

#: Keys every interval row carries, so a reader never has to guess which
#: correction produced a number.
INTERVAL_KEYS: tuple[str, ...] = (
    "status", "n_splits", "estimate", "corrected_se", "degrees_of_freedom",
    "ci_lower", "ci_upper", "t_statistic", "p_value_two_sided",
)


def corrected(scores: Sequence[float], test_sizes: Sequence[int],
              train_sizes: Sequence[int], confidence_level: float) -> dict[str, Any]:
    """Nadeau-Bengio corrected interval and two-sided test over the split scores.

    The correction uses the **actual** fold sizes and ``df = n_splits - 1``,
    which is 99 in the production design.  It widens the naive interval: the
    same respondents are reused across splits, and that is not new data.
    """

    from empirical_testing.core.repeated_cv import nadeau_bengio

    result = nadeau_bengio(
        np.asarray(scores, dtype=np.float64),
        np.asarray(test_sizes, dtype=np.int64),
        np.asarray(train_sizes, dtype=np.int64),
        confidence_level=float(confidence_level),
    )
    def _optional(value: float | None) -> float:
        # A degenerate or insufficient score vector yields None rather than a
        # number; it is carried through as NaN with ``status`` explaining why,
        # never silently replaced by zero.
        return float("nan") if value is None else float(value)

    return {
        "status": result.status, "n_splits": int(result.n_scores),
        "estimate": float(result.estimate), "corrected_se": float(result.corrected_se),
        "degrees_of_freedom": int(result.degrees_of_freedom),
        "ci_lower": _optional(result.ci_lower), "ci_upper": _optional(result.ci_upper),
        "t_statistic": _optional(result.t_statistic),
        "p_value_two_sided": _optional(result.p_value_two_sided),
        "variance_factor": float(result.variance_factor),
        "test_train_ratio": float(result.test_train_ratio),
        "confidence_level": float(result.confidence_level),
    }


def _seed_mean_worst(cube: np.ndarray, cell_index: int, method_index: int) -> float:
    """Mean over learner seeds of the worst-group distortion, for one split.

    The maximum over groups is already taken inside the saved arrays, so the
    order is: max over groups within ``(seed, method)`` first, then the mean
    over seeds -- never the other way round.
    """

    return float(np.mean(cube[cell_index, :, method_index]))


def reduce_condition(
    *,
    cells: Sequence[Condition],
    worst_by_split: Sequence[np.ndarray],
    levels_by_split: Sequence[np.ndarray],
    provenance: Mapping[str, Mapping[str, str]],
    groups: Sequence[str],
    test_sizes: Sequence[int],
    train_sizes: Sequence[int],
    confidence_level: float,
    selected_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Levels, six pairwise contrasts, and their per-group counterparts.

    ``worst_by_split[s]`` is ``(n_cells, n_seeds, 4)`` and
    ``levels_by_split[s]`` is ``(n_cells, n_seeds, 4, n_groups)``.

    Holm adjusts the **decisions** of the six worst-group method comparisons
    within each cell; the intervals stay ordinary marginal corrected intervals.
    For the per-group comparisons the adjustment ranges over the selected
    groups, separately for each contrast and cell.
    """

    from empirical_testing.experiments.coffee_crossfit import holm_adjusted

    selected = list(selected_groups if selected_groups is not None else groups)
    n_splits = len(worst_by_split)
    levels_rows: list[dict[str, Any]] = []
    group_level_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    group_contrast_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for cell_index, cell in enumerate(cells):
        base_row = cell.as_row()

        # --- four-method levels, worst-group and per-group ----------------
        for method_index, method in enumerate(METHODS):
            scores = [
                _seed_mean_worst(worst_by_split[s], cell_index, method_index)
                for s in range(n_splits)
            ]
            interval = corrected(scores, test_sizes, train_sizes, confidence_level)
            levels_rows.append({
                **method_row(cell, method,
                             provenance.get(cell.label, {}).get(method, "unknown")),
                "quantity": "worst_group_distortion", **interval,
            })
            for group_index, group in enumerate(groups):
                group_scores = [
                    float(np.mean(levels_by_split[s][cell_index, :, method_index, group_index]))
                    for s in range(n_splits)
                ]
                group_level_rows.append({
                    **method_row(cell, method,
                                 provenance.get(cell.label, {}).get(method, "unknown")),
                    "group": group, "quantity": "group_distortion",
                    **corrected(group_scores, test_sizes, train_sizes, confidence_level),
                })

        # --- all six pairwise contrasts, worst-group ----------------------
        cell_contrasts: list[dict[str, Any]] = []
        for minuend, subtrahend in PAIRWISE:
            mi, si = METHODS.index(minuend), METHODS.index(subtrahend)
            scores = []
            for s in range(n_splits):
                # Paired within split and seed, then averaged over seeds.
                per_seed = (worst_by_split[s][cell_index, :, mi]
                            - worst_by_split[s][cell_index, :, si])
                scores.append(float(np.mean(per_seed)))
                split_rows.append({
                    **base_row, "contrast": f"{minuend}-{subtrahend}",
                    "split_index": s, "split_score": scores[-1],
                    "n_test": int(test_sizes[s]), "n_train": int(train_sizes[s]),
                })
            interval = corrected(scores, test_sizes, train_sizes, confidence_level)
            cell_contrasts.append({
                **base_row, "contrast": f"{minuend}-{subtrahend}",
                "minuend": minuend, "subtrahend": subtrahend,
                "minuend_display": DISPLAY_NAMES[minuend],
                "subtrahend_display": DISPLAY_NAMES[subtrahend],
                "minuend_training_protocol": training_protocol(minuend, cell.protocol),
                "subtrahend_training_protocol": training_protocol(subtrahend, cell.protocol),
                "positive_favors": subtrahend,
                "quantity": "worst_group_distortion",
                "schema_version": SCHEMA_VERSION, **interval,
            })
        adjusted = _holm(holm_adjusted, [row["p_value_two_sided"] for row in cell_contrasts])
        for row, value in zip(cell_contrasts, adjusted):
            row["p_holm"] = float(value)
            row["holm_family"] = (
                "the six pairwise method comparisons within this condition"
            )
            row["holm_family_size"] = len(PAIRWISE)
        contrast_rows.extend(cell_contrasts)

        # --- per-group counterparts --------------------------------------
        for minuend, subtrahend in PAIRWISE:
            mi, si = METHODS.index(minuend), METHODS.index(subtrahend)
            per_contrast: list[dict[str, Any]] = []
            for group_index, group in enumerate(groups):
                if group not in selected:
                    continue
                scores = [
                    float(np.mean(
                        levels_by_split[s][cell_index, :, mi, group_index]
                        - levels_by_split[s][cell_index, :, si, group_index]
                    ))
                    for s in range(n_splits)
                ]
                per_contrast.append({
                    **base_row, "contrast": f"{minuend}-{subtrahend}",
                    "minuend": minuend, "subtrahend": subtrahend, "group": group,
                    "positive_favors": subtrahend, "quantity": "group_distortion",
                    **corrected(scores, test_sizes, train_sizes, confidence_level),
                })
            group_adjusted = _holm(
                holm_adjusted, [row["p_value_two_sided"] for row in per_contrast])
            for row, value in zip(per_contrast, group_adjusted):
                row["p_holm"] = float(value)
                row["holm_family"] = (
                    "the selected groups, for this contrast and condition"
                )
                row["holm_family_size"] = len(per_contrast)
            group_contrast_rows.extend(per_contrast)

    return {
        "method_levels": levels_rows,
        "method_group_levels": group_level_rows,
        "method_contrasts": contrast_rows,
        "method_group_contrasts": group_contrast_rows,
        "split_scores": split_rows,
    }


def reduce_protocol_contrasts(
    *,
    worst_by_split: Sequence[np.ndarray],
    cells: Sequence[Condition],
    groups: Sequence[str],
    levels_by_split: Sequence[np.ndarray],
    test_sizes: Sequence[int],
    train_sizes: Sequence[int],
    confidence_level: float,
) -> dict[str, Any]:
    """Active minus passive GLHF, at beta 4 and infinity, paired at the split.

    Only the two GLHF methods have an active counterpart.  NLHF does not: it is
    passive-trained in every condition of this study, so there is no
    "active NLHF" and no such contrast is formed.
    """

    by_label = {cell.label: index for index, cell in enumerate(cells)}
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for beta in (4.0, math.inf):
        active, passive = f"active@{beta_label(beta)}", f"passive@{beta_label(beta)}"
        if active not in by_label or passive not in by_label:
            continue
        ai, pi = by_label[active], by_label[passive]
        for method in ("multigroup", "pooled"):
            mi = METHODS.index(method)
            scores = [
                float(np.mean(worst_by_split[s][ai, :, mi] - worst_by_split[s][pi, :, mi]))
                for s in range(len(worst_by_split))
            ]
            rows.append({
                "condition": "beta", "evaluation_regime": REGIME_FIXED,
                "beta": beta_label(beta), "learner_tau": FIXED_EVALUATION_TAU,
                "evaluation_tau": FIXED_EVALUATION_TAU,
                "method": method, "contrast": f"active-minus-passive:{method}",
                "minuend_training_protocol": "active",
                "subtrahend_training_protocol": "passive",
                "positive_favors": "passive",
                "quantity": "worst_group_distortion",
                **corrected(scores, test_sizes, train_sizes, confidence_level),
            })
            for group_index, group in enumerate(groups):
                group_scores = [
                    float(np.mean(
                        levels_by_split[s][ai, :, mi, group_index]
                        - levels_by_split[s][pi, :, mi, group_index]
                    ))
                    for s in range(len(levels_by_split))
                ]
                group_rows.append({
                    "condition": "beta", "evaluation_regime": REGIME_FIXED,
                    "beta": beta_label(beta), "learner_tau": FIXED_EVALUATION_TAU,
                    "evaluation_tau": FIXED_EVALUATION_TAU,
                    "method": method, "group": group,
                    "contrast": f"active-minus-passive:{method}",
                    "quantity": "group_distortion",
                    **corrected(group_scores, test_sizes, train_sizes, confidence_level),
                })
    return {"protocol_contrasts": rows, "protocol_group_contrasts": group_rows}


def coverage_rows(
    *, condition: str, regime: str, cells: Sequence[Condition],
    provenance: Mapping[str, Mapping[str, str]], n_splits: int, n_seeds: int,
    expected_splits: int,
) -> list[dict[str, Any]]:
    """One row per (cell, method): what was expected and what is present."""

    rows = []
    for cell in cells:
        for method in METHODS:
            rows.append({
                **cell.as_row(), "method": method,
                "provenance": provenance.get(cell.label, {}).get(method, "unknown"),
                "expected_splits": int(expected_splits),
                "observed_splits": int(n_splits),
                "learner_seeds": int(n_seeds),
                "complete": bool(n_splits == expected_splits),
            })
    return rows


def _holm(holm_adjusted, p_values: Sequence[float]) -> list[float]:
    """Holm over p-values that may be undefined.

    A degenerate score vector (too few splits, or zero variance) yields no
    p-value.  Those positions stay NaN rather than being treated as 1.0, which
    would silently claim a non-significant decision was made; the defined
    p-values are adjusted among themselves.
    """

    values = [float(v) for v in p_values]
    defined = [i for i, v in enumerate(values) if not math.isnan(v)]
    if not defined:
        return [float("nan")] * len(values)
    adjusted = holm_adjusted([values[i] for i in defined])
    out = [float("nan")] * len(values)
    for position, index in enumerate(defined):
        out[index] = float(adjusted[position])
    return out


#: Columns whose string form must survive a CSV round trip.  ``beta`` is
#: written as ``"inf"`` by convention, and pandas would otherwise read that
#: back as ``float('inf')`` and ``"4"`` as ``4.0``, quietly changing the label
#: a figure prints and breaking any comparison against the written form.
STRING_COLUMNS: tuple[str, ...] = ("beta", "cell", "condition", "evaluation_regime",
                                   "method", "contrast", "group", "provenance")


def read_table(path: Any) -> "pd.DataFrame":  # noqa: F821
    """Read a suite reduction CSV without losing the string columns.

    Always use this rather than a bare ``pd.read_csv`` on these tables.
    """

    import pandas as pd

    # The dtype must be given to the *reader*: converting afterwards is too
    # late, because pandas has already turned "4" into 4.0 and "inf" into
    # float infinity, and ``astype(str)`` would then produce "4.0".
    header = pd.read_csv(path, nrows=0)
    dtypes = {c: str for c in STRING_COLUMNS if c in header.columns}
    return pd.read_csv(path, dtype=dtypes)
