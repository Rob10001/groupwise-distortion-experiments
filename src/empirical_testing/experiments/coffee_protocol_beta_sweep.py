"""Shared GLHF protocol/beta fitting and reduction for Coffee and Persona.

The grid uses passive beta values 0.5, 1, 2, 4, and infinity, and active beta
values 4 and infinity, with learner and evaluation radii fixed at 0.5.
All cells share respondent order within each split and learner seed.
Passive cells share pairs and relabel them using common random variates;
active arms draw comparisons from their own policies and native RNG streams.
Protocol contrasts include each sampler's native score normalization.

experiments.fresh calls these fitters for raw-input reproduction. The lower-level
sweep runner also supports checking a saved passive-infinity reference."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import expit

from empirical_testing.core import make_uniform_passive_log, subset_dense_dataset
from empirical_testing.core.holdout import sorted_id_sha256
from empirical_testing.core.repeated_cv import NB_STATUS_OK, nadeau_bengio, split_learner_seed
from empirical_testing.experiments.coffee_beta_sweep import (
    BETA_LABEL_STREAM_KEY,
    beta_label_uniforms,
    relabel_signs,
)
from empirical_testing.experiments.coffee_crossfit import (
    ARMS,
    MEMBERSHIP_ALIGNMENT,
    ROUTE_V4_10X10,
    CoffeeCrossfitConfig,
    _now,
    _publish,
    _write_json,
    holm_adjusted,
    load_crossfit_config,
    load_crossfit_inputs,
)
from empirical_testing.parallel import ordered_map
from empirical_testing.reporting.metrics import RATIO_DEFINED

# --------------------------------------------------------------- the axes

PASSIVE = "passive"
ACTIVE = "active"
#: The two sampling protocols this sweep knows how to run.  A third name is a
#: refusal, never a silent passive fallback.
SUPPORTED_PROTOCOLS: tuple[str, ...] = (PASSIVE, ACTIVE)

#: The evaluation ball never moves.  ``beta`` changes the *training labels*;
#: it does not rescale utilities and it does not touch the held-out target.
EVALUATION_TAU = 0.5

#: The untrained uniform lottery, scored beside the two fitted arms.  It needs
#: no fit, ignores the comparison log entirely, and must therefore be identical
#: in every cell -- which is asserted per split, not assumed.
UNIFORM_ARM = "uniform"
SWEEP_METHODS: tuple[str, ...] = (*ARMS, UNIFORM_ARM)

#: The production grid, as specified: passive at five label models and active at
#: two.  It is deliberately **not** a full product: the active arm is run at the
#: deterministic oracle and at the one finite ``beta`` the contrast is wanted at.
DEFAULT_PASSIVE_BETAS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, math.inf)
DEFAULT_ACTIVE_BETAS: tuple[float, ...] = (4.0, math.inf)

#: The one cell that must reproduce the source route exactly.
ANCHOR_PROTOCOL = PASSIVE
ANCHOR_BETA = math.inf


def beta_label(beta: float) -> str:
    """The canonical string for one label model (``'inf'``, ``'0.5'``, ``'4'``).

    ``inf`` is serialized as the string ``"inf"`` everywhere, so no table or
    JSON document ever carries a nonstandard ``Infinity`` token.
    """

    return "inf" if math.isinf(beta) else f"{float(beta):g}"


def parse_beta(text: str) -> float:
    """Parse one ``--betas`` entry: finite and strictly positive, or ``inf``."""

    value = math.inf if str(text).strip().lower() in {"inf", "+inf", "infinity"} else float(text)
    if math.isnan(value) or value <= 0.0:
        raise ValueError(
            f"beta must be a finite positive number or 'inf', not {text!r}; "
            "NaN, zero and negative values are refused"
        )
    return value


@dataclass(frozen=True)
class SweepCell:
    """One ``(protocol, beta)`` cell of the grid."""

    protocol: str
    beta: float

    @property
    def label(self) -> str:
        return f"{self.protocol}@{beta_label(self.beta)}"

    @property
    def beta_text(self) -> str:
        return beta_label(self.beta)

    @property
    def is_active(self) -> bool:
        return self.protocol == ACTIVE

    @property
    def is_anchor(self) -> bool:
        return self.protocol == ANCHOR_PROTOCOL and math.isinf(self.beta)

    def as_mapping(self) -> dict[str, Any]:
        return {"cell": self.label, "protocol": self.protocol, "beta": self.beta_text}


def default_cells() -> tuple[SweepCell, ...]:
    """The production grid: five passive cells and two active ones."""

    return (
        *(SweepCell(PASSIVE, beta) for beta in DEFAULT_PASSIVE_BETAS),
        *(SweepCell(ACTIVE, beta) for beta in DEFAULT_ACTIVE_BETAS),
    )


def parse_cells(text: str) -> tuple[SweepCell, ...]:
    """Parse an explicit ``protocol@beta,...`` grid specification."""

    cells = []
    for item in str(text).split(","):
        piece = item.strip()
        if not piece:
            continue
        if "@" not in piece:
            raise ValueError(f"a cell must be written protocol@beta, not {piece!r}")
        protocol, _, beta_text = piece.partition("@")
        cells.append(SweepCell(protocol.strip().lower(), parse_beta(beta_text)))
    return validate_cells(cells)


def cells_from_product(
    protocols: Sequence[str], betas: Sequence[float]
) -> tuple[SweepCell, ...]:
    """Build the full ``protocols x betas`` product and add the anchor.

    ``--betas`` names *finite* label models: the anchor's ``beta = inf`` is not
    a sweep level and is added exactly once, as the ``passive`` anchor, however
    small a subset of finite cells was asked for.
    """

    protocols = [str(name).strip().lower() for name in protocols]
    if len(set(protocols)) != len(protocols):
        raise ValueError(f"--protocols must be distinct, got {protocols}")
    unknown = sorted(set(protocols) - set(SUPPORTED_PROTOCOLS))
    if unknown:
        raise ValueError(
            f"unsupported sampling protocols {unknown}; supported: {list(SUPPORTED_PROTOCOLS)}"
        )
    values = [float(value) for value in betas]
    if any(math.isinf(value) for value in values):
        raise ValueError(
            "--betas names the finite label models only; the passive beta=inf anchor is always "
            "added and is not a sweep level"
        )
    if len({beta_label(value) for value in values}) != len(values):
        raise ValueError(f"--betas must be distinct, got {values}")
    cells = [SweepCell(protocol, beta) for protocol in protocols for beta in values]
    cells.append(SweepCell(ANCHOR_PROTOCOL, ANCHOR_BETA))
    return validate_cells(cells)


def validate_cells(cells: Iterable[SweepCell]) -> tuple[SweepCell, ...]:
    """Refuse a grid that is empty, repeats a cell, or has no single anchor."""

    ordered = tuple(cells)
    if not ordered:
        raise ValueError("the sweep needs at least one cell")
    unknown = sorted({cell.protocol for cell in ordered} - set(SUPPORTED_PROTOCOLS))
    if unknown:
        raise ValueError(
            f"unsupported sampling protocols {unknown}; supported: {list(SUPPORTED_PROTOCOLS)}"
        )
    labels = [cell.label for cell in ordered]
    if len(set(labels)) != len(labels):
        raise ValueError(f"the grid repeats a cell: {sorted(labels)}")
    for cell in ordered:
        if math.isnan(cell.beta) or cell.beta <= 0.0:
            raise ValueError(f"cell {cell.label} has a non-positive or NaN beta")
    anchors = [cell for cell in ordered if cell.is_anchor]
    if len(anchors) != 1:
        raise ValueError(
            f"the grid must contain exactly one {ANCHOR_PROTOCOL} beta=inf anchor cell, found "
            f"{len(anchors)}; it is what ties the sweep to the frozen source run"
        )
    return ordered


def finite_protocol_contrast_betas(cells: Sequence[SweepCell]) -> tuple[str, ...]:
    """The ``beta`` labels at which both protocols were run, in grid order."""

    by_beta: dict[str, set[str]] = {}
    for cell in cells:
        by_beta.setdefault(cell.beta_text, set()).add(cell.protocol)
    seen: list[str] = []
    for cell in cells:
        if cell.beta_text not in seen and {PASSIVE, ACTIVE} <= by_beta[cell.beta_text]:
            seen.append(cell.beta_text)
    return tuple(seen)


# ----------------------------------------------------------- the one fitter


#: The score bound the learner normalizes rewards by, per protocol.  These are
#: the learner's existing, unmodified choices; they are recorded so that the
#: protocol contrast is read as "this protocol as implemented", normalization
#: included, rather than as "these queried pairs".
SCORE_BOUND_NOTE = {
    ACTIVE: "K - 1 (one_expert._prepare_training_inputs, protocol='active')",
    PASSIVE: "max(1 / propensity) = K (K - 1) / 2 for a uniform log (protocol='logged')",
}

#: How the active learner derives the generator that draws both its pairs and
#: its label coins.  Recorded verbatim rather than described loosely, because
#: the passive relabelling stream is a *different* stream.
ACTIVE_RNG_DERIVATION = (
    "np.random.default_rng(np.random.SeedSequence(training_seed).spawn(2)[0]); the same generator "
    "draws the first alternative from the current policy, the second uniformly from the other "
    "K - 1, and the Bradley-Terry label coin"
)
PASSIVE_RELABEL_RNG_DERIVATION = (
    f"np.random.default_rng(np.random.SeedSequence([{BETA_LABEL_STREAM_KEY}, training_seed]))"
)
RNG_COUPLING_NOTE = (
    "within a split and a learner seed every passive cell shares one log and one vector of common "
    "random numbers, so the passive beta curve is a pure label-noise effect. The active cells draw "
    "their pairs and their labels from the learner's own spawned generator, which is a different "
    "stream from the passive relabelling stream: the same outer seed therefore does NOT give the "
    "two protocols identical label coins, and no such alignment is claimed. What pairs the protocol "
    "contrast is the matched split and the matched learner seed."
)


def _check_training_only_utilities(
    ordered_utilities: np.ndarray, expected: np.ndarray, test_view: Any
) -> None:
    """Refuse to hand the learner anything but this split's *training* block.

    The active protocol is the only path here that passes utilities into the
    fitter at all, so the held-out invariant is asserted at the boundary rather
    than left implicit: the array must be exactly ``train_view.utilities[order]``
    and must not be the test view's block.  ``tests/`` proves the invariant the
    other way round too, by poisoning every test-fold utility with NaN and
    requiring the fitted policies to be unchanged.
    """

    if ordered_utilities.shape != expected.shape:
        raise AssertionError(
            f"the learner was handed utilities of shape {ordered_utilities.shape}, not this "
            f"split's {expected.shape} ordered training block"
        )
    if not np.array_equal(ordered_utilities, expected):
        raise AssertionError(
            "the learner was handed utilities that are not this split's ordered training block"
        )
    test_utilities = np.asarray(test_view.utilities, dtype=np.float64)
    if (ordered_utilities.shape == test_utilities.shape
            and np.array_equal(ordered_utilities, test_utilities)):
        raise AssertionError("held-out test-fold utilities reached the fitter")


def _label_diagnostics(
    difference: np.ndarray, signs: np.ndarray, beta: float
) -> dict[str, float]:
    """Flips against ``sign(u_i - u_j)`` on this cell's own queried pairs.

    The denominator is the **strict** utility differences only.  An exact tie is
    a fair coin even at ``beta = inf``, so disagreement between two tie labels is
    not a noise-induced flip and never enters the rate.  When every queried pair
    ties the rate is saved as ``nan`` beside its zero denominator rather than as
    a zero that would read as "no noise".
    """

    difference = np.asarray(difference, dtype=np.float64)
    untied = difference != 0.0
    n_untied = int(untied.sum())
    n_ties = int(difference.size - n_untied)
    if n_untied == 0:
        flips = 0
        rate = float("nan")
        expected_error = float("nan")
    else:
        flips = int((np.asarray(signs)[untied] != np.sign(difference[untied])).sum())
        rate = flips / n_untied
        expected_error = (
            0.0 if math.isinf(beta)
            else float(expit(-float(beta) * np.abs(difference[untied])).mean())
        )
    return {
        "flips": float(flips),
        "n_untied": float(n_untied),
        "n_ties": float(n_ties),
        "flip_rate": float(rate),
        "expected_error": float(expected_error),
    }


DIAGNOSTIC_KEYS = ("flips", "n_untied", "n_ties", "flip_rate", "expected_error")


def fit_split_cells(
    *,
    algorithm: str,
    cv_seed: int,
    learner_seeds: Sequence[int],
    comparisons_per_respondent: int,
    tau: float,
    cells: Sequence[SweepCell],
    train_view: Any,
    test_view: Any,
    repeat: int,
    fold: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]],
           dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit both arms in every cell, once per learner seed, on one split.

    Dataset-agnostic on purpose: the Coffee and persona sweeps are the same
    computation on different inputs, so there is one fitter, parameterised by
    ``(protocol, beta)``.  The frozen routes' own fitter is untouched.

    Returns ``(policies, seed_records, diagnostics, active_logs)``:

    ``policies[cell_label][arm]``
        ``(n_seeds, n_test, n_actions)`` predicted lotteries.
    ``diagnostics[key]``
        ``(n_cells, n_seeds, n_arms)`` label-noise arrays, one per
        :data:`DIAGNOSTIC_KEYS`.  Both arm slots carry the same numbers in a
        passive cell, where the two arms share one log.
    ``active_logs["<cell>|<arm>"]``
        ``(n_seeds, n_train, 4)`` -- first, second, sign, unordered-pair
        propensity -- in the orientation the learner returned, plus
        ``respondent_order`` ``(n_seeds, n_train)``.

    Held-out utilities are never read: the test fold contributes only group
    memberships, as prediction contexts.
    """

    from empirical_testing.algorithms.registry import fit_online_learner

    if int(comparisons_per_respondent) != 1:
        raise ValueError(
            "this sweep supports exactly one comparison per training respondent; the active "
            f"protocol consumes one training row per round, so {comparisons_per_respondent} is "
            "not a supported budget"
        )
    cells = validate_cells(cells)
    train_utilities = np.asarray(train_view.utilities, dtype=np.float64)
    train_groups = np.asarray(train_view.groups)
    n_train, n_actions = int(train_utilities.shape[0]), int(train_utilities.shape[1])
    test_groups = np.asarray(test_view.groups)
    n_test = int(test_groups.shape[0])
    reference = np.full(n_actions, 1.0 / n_actions)
    pooled_train_groups = np.ones((n_train, 1), dtype=np.int8)
    pooled_test_groups = np.ones((n_test, 1), dtype=np.int8)

    labels = [cell.label for cell in cells]
    stacked: dict[str, dict[str, list[np.ndarray]]] = {
        label: {arm: [] for arm in ARMS} for label in labels
    }
    drawn_logs: dict[str, list[np.ndarray]] = {}
    orders: list[np.ndarray] = []
    diagnostics = {
        key: np.full((len(cells), len(learner_seeds), len(ARMS)), np.nan) for key in DIAGNOSTIC_KEYS
    }
    records: list[dict[str, Any]] = []

    for seed_index, learner_seed in enumerate(learner_seeds):
        training_seed = split_learner_seed(cv_seed, repeat, fold, learner_seed)
        # One source-compatible passive log per (split, seed).  It fixes the
        # respondent order for the entire grid, active cells included.
        log_rows = make_uniform_passive_log(
            train_view, np.arange(n_train, dtype=np.int64), seed=training_seed,
            comparisons_per_respondent=int(comparisons_per_respondent), beta=np.inf,
        )
        order = np.asarray(log_rows.respondent_indices, dtype=np.int64)
        if order.shape != (n_train,) or not np.array_equal(np.sort(order), np.arange(n_train)):
            raise AssertionError(
                "the passive log's respondent order is not a permutation of this split's training "
                "rows; the grid cannot share one order"
            )
        orders.append(order)
        ordered_utilities = train_utilities[order]
        aligned_train_groups = train_groups[order]
        uniforms = beta_label_uniforms(training_seed, len(log_rows.signs))
        passive_difference = (
            train_utilities[order, log_rows.pairs[:, 0]]
            - train_utilities[order, log_rows.pairs[:, 1]]
        )

        for cell_index, cell in enumerate(cells):
            if cell.is_active:
                _check_training_only_utilities(ordered_utilities, train_utilities[order], test_view)
                for arm_index, (arm, groups, prediction_groups) in enumerate((
                    ("pooled", pooled_train_groups, pooled_test_groups),
                    ("multigroup", aligned_train_groups, test_groups),
                )):
                    model = fit_online_learner(
                        algorithm, groups=groups, utilities=ordered_utilities, beta=float(cell.beta),
                        reference=reference, tau=float(tau), protocol="active", seed=training_seed,
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
                            f"the active learner recorded {drawn.shape} comparisons, not one per "
                            f"each of the {n_train} training respondents"
                        )
                    if not np.all(np.isin(drawn[:, 2], (-1.0, 1.0))):
                        raise AssertionError("a drawn label is not a strict +-1 winner")
                    if not np.all((drawn[:, 3] > 0.0) & (drawn[:, 3] <= 1.0)):
                        raise AssertionError("a drawn pair has an out-of-range propensity")
                    first = drawn[:, 0].astype(np.int64)
                    second = drawn[:, 1].astype(np.int64)
                    # This arm's own queried pairs -- never the anchor's.
                    difference = (
                        ordered_utilities[np.arange(n_train), first]
                        - ordered_utilities[np.arange(n_train), second]
                    )
                    stats = _label_diagnostics(difference, drawn[:, 2], cell.beta)
                    for key in DIAGNOSTIC_KEYS:
                        diagnostics[key][cell_index, seed_index, arm_index] = stats[key]
                    drawn_logs.setdefault(f"{cell.label}|{arm}", []).append(drawn)
                    stacked[cell.label][arm].append(
                        np.asarray(model.predict(prediction_groups), dtype=np.float64)
                    )
                continue

            signs = (
                log_rows.signs if math.isinf(cell.beta)
                else relabel_signs(
                    order, log_rows.pairs, train_utilities, float(cell.beta), uniforms
                )
            )
            stats = _label_diagnostics(passive_difference, signs, cell.beta)
            for arm_index in range(len(ARMS)):
                for key in DIAGNOSTIC_KEYS:
                    diagnostics[key][cell_index, seed_index, arm_index] = stats[key]
            for arm, groups, prediction_groups in (
                ("pooled", pooled_train_groups, pooled_test_groups),
                ("multigroup", aligned_train_groups, test_groups),
            ):
                model = fit_online_learner(
                    algorithm, groups=groups, utilities=None, reference=reference,
                    tau=float(tau), protocol="logged", seed=training_seed,
                    pair_indices=log_rows.pairs, signs=signs, propensities=log_rows.propensities,
                )
                if not np.array_equal(
                    model.snapshot_rounds, np.arange(len(signs), dtype=np.int64)
                ):
                    raise AssertionError("the fitted policy is not the all-round pre-round average")
                stacked[cell.label][arm].append(
                    np.asarray(model.predict(prediction_groups), dtype=np.float64)
                )

        records.append({
            "learner_seed": int(learner_seed),
            "training_seed": int(training_seed),
            "n_comparisons": int(n_train),
            "respondent_order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
            "training_membership_alignment": MEMBERSHIP_ALIGNMENT,
            "shared_log_across_arms": {
                cell.label: (not cell.is_active) for cell in cells
            },
            "score_bound": {cell.label: SCORE_BOUND_NOTE[cell.protocol] for cell in cells},
        })

    policies = {
        label: {arm: np.stack(rows, axis=0) for arm, rows in arms.items()}
        for label, arms in stacked.items()
    }
    active_logs = {key: np.stack(rows, axis=0) for key, rows in drawn_logs.items()}
    if active_logs:
        active_logs["respondent_order"] = np.stack(orders, axis=0)
    return policies, records, diagnostics, active_logs


# ------------------------------------------------------------- one split


def _check_distortion_complete(
    groups: Sequence[str], evaluation: Any, label: str, repeat: int, fold: int
) -> None:
    status = np.asarray(evaluation.distortion_status, dtype=object)
    values = np.asarray(evaluation.distortion, dtype=np.float64)
    bad = (status != RATIO_DEFINED) | ~np.isfinite(values)
    if not np.any(bad):
        return
    offenders = [
        f"seed={evaluation.seed_ids[s]} arm={evaluation.method_names[m]} group={groups[k]} "
        f"status={status[s, m, k]}"
        for s, m, k in zip(*np.nonzero(bad))
    ]
    raise RuntimeError(
        f"split repeat={repeat} fold={fold} cell={label} has {len(offenders)} undefined or "
        f"nonfinite distortion values, e.g. {offenders[:5]}. The design requires a defined finite "
        "value for every selected group; values are not clipped and groups are not dropped."
    )


def compute_split_cells(
    *,
    algorithm: str,
    cv_seed: int,
    learner_seeds: Sequence[int],
    comparisons_per_respondent: int,
    tau: float,
    groups: Sequence[str],
    cells: Sequence[SweepCell],
    train_view: Any,
    test_view: Any,
    repeat: int,
    fold: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit and evaluate every cell of one split.  Shared by both datasets."""

    from empirical_testing.reporting import evaluate_single_holdout

    cells = validate_cells(cells)
    labels = [cell.label for cell in cells]
    n_seeds, n_groups = len(learner_seeds), len(groups)
    test_groups = np.asarray(test_view.groups)
    n_test = int(test_groups.shape[0])
    n_actions = int(test_view.utilities.shape[1])

    started = time.perf_counter()
    policies, seed_records, diagnostics, active_logs = fit_split_cells(
        algorithm=algorithm, cv_seed=cv_seed, learner_seeds=learner_seeds,
        comparisons_per_respondent=comparisons_per_respondent, tau=tau, cells=cells,
        train_view=train_view, test_view=test_view, repeat=repeat, fold=fold,
    )
    fit_seconds = time.perf_counter() - started

    uniform_policy = np.broadcast_to(
        np.full(n_actions, 1.0 / n_actions), (n_seeds, n_test, n_actions)
    ).copy()
    worst = np.full((len(cells), n_seeds, len(SWEEP_METHODS)), np.nan)
    levels = np.full((len(cells), n_seeds, len(SWEEP_METHODS), n_groups), np.nan)
    welfare = np.full((len(cells), n_seeds, len(SWEEP_METHODS), n_groups), np.nan)
    welfare_optimum = np.full((len(cells), n_groups), np.nan)
    worst_index = np.full((len(cells), n_seeds, len(SWEEP_METHODS)), -1, dtype=np.int64)
    seed_difference = np.full((len(cells), n_seeds), np.nan)
    status_array = np.empty((len(cells), n_seeds, len(SWEEP_METHODS), n_groups), dtype="<U7")

    started = time.perf_counter()
    # Held-out utilities are first read here, after every policy for this split
    # is fitted.  The comparator is refit on this test fold inside the fixed
    # evaluation ball and shared across cells, arms and learner seeds.
    for index, cell in enumerate(cells):
        arms = dict(policies[cell.label])
        arms[UNIFORM_ARM] = uniform_policy
        evaluation = evaluate_single_holdout(
            test_view.utilities, test_groups.astype(bool), arms,
            group_names=list(groups), method_names=SWEEP_METHODS,
            seed_ids=list(learner_seeds), tau=EVALUATION_TAU,
        )
        _check_distortion_complete(groups, evaluation, cell.label, repeat, fold)
        worst[index] = np.asarray(evaluation.worst_group_distortion, dtype=np.float64)
        levels[index] = np.asarray(evaluation.distortion, dtype=np.float64)
        welfare[index] = np.asarray(evaluation.welfare, dtype=np.float64)
        welfare_optimum[index] = np.asarray(evaluation.welfare_optimum, dtype=np.float64)
        worst_index[index] = np.asarray(evaluation.worst_group_index, dtype=np.int64)
        status_array[index] = np.asarray(evaluation.distortion_status)
        difference = np.asarray(evaluation.per_seed_pooled_minus_multigroup, dtype=np.float64)
        if difference.shape != (n_seeds,) or not np.all(np.isfinite(difference)):
            raise RuntimeError(
                f"split repeat={repeat} fold={fold} cell={cell.label} produced an incomplete "
                "per-seed contrast"
            )
        seed_difference[index] = difference
    evaluate_seconds = time.perf_counter() - started

    if not np.all(np.isfinite(worst)) or not np.all(np.isfinite(levels)):
        raise RuntimeError(f"split repeat={repeat} fold={fold} has nonfinite distortion")
    # The uniform lottery is not fitted and ignores the log, so it cannot depend
    # on the protocol or on beta.  A violation would mean a cell leaked its
    # training feedback into the evaluation.
    uniform_column = worst[:, :, SWEEP_METHODS.index(UNIFORM_ARM)]
    if not np.allclose(uniform_column, uniform_column[0], rtol=0.0, atol=0.0):
        raise AssertionError("the untrained uniform benchmark varied across the sweep cells")
    if not np.allclose(welfare_optimum, welfare_optimum[0], rtol=0.0, atol=0.0):
        raise AssertionError("the held-out welfare comparator varied across the sweep cells")

    train_counts = np.asarray(train_view.groups).sum(axis=0).astype(np.int64)
    test_counts = test_groups.sum(axis=0).astype(np.int64)
    summary: dict[str, Any] = {
        "repeat": int(repeat), "fold": int(fold),
        "cells": [cell.as_mapping() for cell in cells],
        "cell_labels": labels,
        "groups": list(groups),
        "learner_seeds": [int(value) for value in learner_seeds],
        "methods": list(SWEEP_METHODS),
        "evaluation_tau": float(EVALUATION_TAU),
        "learner_tau": float(tau),
        "split_scores": {
            label: float(seed_difference[index].mean()) for index, label in enumerate(labels)
        },
        "test_outcomes_used_for_training": False,
        "seed_records": seed_records,
        "fit_seconds": round(fit_seconds, 3),
        "evaluate_seconds": round(evaluate_seconds, 3),
    }
    arrays = {
        "worst_group_distortion": worst,
        "group_distortion": levels,
        "welfare": welfare,
        "welfare_optimum": welfare_optimum,
        "worst_group_index": worst_index,
        "distortion_status": status_array,
        "seed_difference": seed_difference,
        "train_group_counts": train_counts,
        "test_group_counts": test_counts,
        **{f"label_{key}": diagnostics[key] for key in DIAGNOSTIC_KEYS},
    }
    return summary, arrays, active_logs


#: Array keys the anchor cell must reproduce from the frozen source run, mapped
#: to the name the frozen run saved them under.  Benchmark-only axes (the
#: untrained uniform arm), timings and route metadata are excluded: the frozen
#: run never computed them.
ANCHOR_ARRAY_MAP: dict[str, str] = {
    "worst_group_distortion": "worst_group_distortion",
    "group_distortion": "distortion",
    "welfare": "welfare",
    "welfare_optimum": "welfare_optimum",
    "worst_group_index": "worst_group_index",
    "distortion_status": "distortion_status",
    "seed_difference": "seed_difference",
    "train_group_counts": "train_group_counts",
    "test_group_counts": "test_group_counts",
}


def anchor_arrays(arrays: Mapping[str, np.ndarray], anchor_index: int) -> dict[str, np.ndarray]:
    """The anchor cell's arrays, reduced to the frozen run's own axes."""

    n_arms = len(ARMS)
    out: dict[str, np.ndarray] = {}
    for key in ANCHOR_ARRAY_MAP:
        array = np.asarray(arrays[key])
        if key in ("train_group_counts", "test_group_counts"):
            out[key] = array
        elif key == "welfare_optimum":
            out[key] = array[anchor_index]
        elif key == "seed_difference":
            out[key] = array[anchor_index]
        elif key in ("worst_group_distortion", "worst_group_index"):
            out[key] = array[anchor_index][:, :n_arms]
        else:
            out[key] = array[anchor_index][:, :n_arms, :]
    return out


# --------------------------------------------------------- split persistence


def split_dir(output_dir: Path, repeat: int, fold: int) -> Path:
    """The repeated-CV routes' layout, so a split pairs with its source by name."""

    return Path(output_dir) / "splits" / f"r{int(repeat):02d}_f{int(fold):02d}"


def saved_split(
    directory: Path, expected: Mapping[str, Any], *, n_cells: int, n_seeds: int, n_groups: int
) -> dict[str, Any] | None:
    """Reuse a saved unit only when every declared design input *and* its arrays check out."""

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
    n_methods = len(SWEEP_METHODS)
    shapes = {
        "worst_group_distortion": (n_cells, n_seeds, n_methods),
        "group_distortion": (n_cells, n_seeds, n_methods, n_groups),
        "seed_difference": (n_cells, n_seeds),
        "welfare_optimum": (n_cells, n_groups),
        **{f"label_{key}": (n_cells, n_seeds, len(ARMS)) for key in DIAGNOSTIC_KEYS},
    }
    try:
        with np.load(arrays_path, allow_pickle=False) as bundle:
            for key, shape in shapes.items():
                if key not in bundle.files or np.asarray(bundle[key]).shape != shape:
                    return None
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        # A half-written bundle is not an error to raise: it is a unit that
        # cannot be reused, so it is recomputed.
        return None
    return saved


def publish_split(
    directory: Path, summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    active_logs: Mapping[str, np.ndarray],
) -> None:
    """Write one split atomically: arrays first, ``split.json`` last."""

    staging = directory.parent / f".partial-{directory.name}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    for path in (staging, directory):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir()
    np.savez_compressed(staging / "split.npz", **arrays)
    if active_logs:
        # The drawn logs live in their own bundle so the reduction never has to
        # load them.  Pairs and signs are small integers; the propensity keeps
        # full float64 precision because it is what the learner divided by.
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
    _write_json(staging / "split.json", summary)
    _publish(staging, directory)


def split_arrays(output_dir: Path, repeat: int, fold: int) -> dict[str, np.ndarray]:
    with np.load(split_dir(output_dir, repeat, fold) / "split.npz", allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


# ------------------------------------------------------------------ reduce


def corrected(scores, test_sizes, train_sizes, level: float) -> dict[str, Any]:
    """One marginal Nadeau-Bengio corrected repeated-CV interval."""

    result = nadeau_bengio(
        np.asarray(scores, dtype=np.float64), test_sizes, train_sizes, confidence_level=level
    )
    return {
        "status": result.status,
        "n_splits": int(result.n_scores),
        "estimate": float(result.estimate),
        "corrected_se": (None if result.corrected_se is None
                         or not np.isfinite(result.corrected_se) else float(result.corrected_se)),
        "degrees_of_freedom": int(result.degrees_of_freedom),
        "ci_lower": None if result.ci_lower is None else float(result.ci_lower),
        "ci_upper": None if result.ci_upper is None else float(result.ci_upper),
        "t_statistic": None if result.t_statistic is None else float(result.t_statistic),
        "p_value_two_sided": (None if result.p_value_two_sided is None
                              else float(result.p_value_two_sided)),
    }


INTERVAL_KEYS = ("status", "n_splits", "estimate", "corrected_se", "degrees_of_freedom",
                 "ci_lower", "ci_upper", "t_statistic", "p_value_two_sided")

INTERVAL_LABEL = (
    "every cell is a marginal Nadeau-Bengio corrected repeated-CV interval over the same splits: "
    "100 split scores, each the learner-seed mean, variance factor 1/100 + mean(n_test/n_train) and "
    "df = 99. The intervals are NOT simultaneous over the cells, the protocols, the arms or the "
    "groups, and they are NOT the manuscript's respondent bootstrap bands. Holm adjustment applies "
    "to the per-group p-values within one cell and nowhere else."
)


def reduce_cells(
    *,
    output_dir: Path,
    cells: Sequence[SweepCell],
    groups: Sequence[str],
    learner_seeds: Sequence[int],
    repeats: int,
    folds: int,
    confidence_level: float,
    split_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-cell corrected intervals, the paired gap, per-group gaps and contrasts.

    Every planned (cell, split) must be present: a missing unit is an error,
    never a smaller sample quietly analysed.
    """

    import pandas as pd

    output_dir = Path(output_dir)
    reduce_dir = output_dir / "reduce"
    reduce_dir.mkdir(parents=True, exist_ok=True)
    cells = validate_cells(cells)
    labels = [cell.label for cell in cells]
    planned = {(r, f) for r in range(repeats) for f in range(folds)}
    seen = {(int(row["repeat"]), int(row["fold"])): row for row in split_summaries}
    missing = sorted(planned - set(seen))
    if missing:
        raise RuntimeError(
            f"{len(missing)} planned splits have no result, e.g. {missing[:5]}; the corrected "
            "intervals require every planned split"
        )
    order = sorted(planned)
    for key in order:
        recorded = list(seen[key].get("cell_labels", ()))
        if recorded != labels:
            raise RuntimeError(
                f"split {key} covers cells {recorded}, not the planned grid {labels}; the reduction "
                "refuses a partially covered grid"
            )

    test_sizes = np.array([seen[key]["n_test"] for key in order], dtype=np.float64)
    train_sizes = np.array([seen[key]["n_train"] for key in order], dtype=np.float64)
    n_methods, n_groups = len(SWEEP_METHODS), len(groups)
    worst = np.empty((len(cells), len(order), n_methods))
    levels = np.empty((len(cells), len(order), n_methods, n_groups))
    gaps = np.empty((len(cells), len(order)))
    group_gaps = np.empty((len(cells), len(order), n_groups))
    noise = {key: np.empty((len(cells), len(order), len(ARMS))) for key in DIAGNOSTIC_KEYS}
    pooled_at, multigroup_at = SWEEP_METHODS.index("pooled"), SWEEP_METHODS.index("multigroup")
    for position, key in enumerate(order):
        arrays = split_arrays(output_dir, *key)
        # W = max_g D is formed per fitted policy before any averaging; the
        # split score is then the mean over the learner seeds.
        worst[:, position, :] = arrays["worst_group_distortion"].mean(axis=1)
        levels[:, position, :, :] = arrays["group_distortion"].mean(axis=1)
        gaps[:, position] = arrays["seed_difference"].mean(axis=1)
        per_group = arrays["group_distortion"]
        group_gaps[:, position, :] = (
            per_group[:, :, pooled_at, :] - per_group[:, :, multigroup_at, :]
        ).mean(axis=1)
        for name in DIAGNOSTIC_KEYS:
            noise[name][:, position, :] = np.nanmean(arrays[f"label_{name}"], axis=1)

    level = float(confidence_level)
    alpha = 1.0 - level
    worst_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    split_score_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    curve: dict[str, dict[str, Any]] = {}
    group_curve: dict[str, dict[str, dict[str, Any]]] = {}
    gap_curve: dict[str, dict[str, Any]] = {}
    group_gap_curve: dict[str, dict[str, dict[str, Any]]] = {}

    for index, cell in enumerate(cells):
        base = {"cell": cell.label, "protocol": cell.protocol, "beta": cell.beta_text}
        curve[cell.label] = {}
        group_curve[cell.label] = {}
        for method_index, method in enumerate(SWEEP_METHODS):
            stats = corrected(worst[index, :, method_index], test_sizes, train_sizes, level)
            row = {**base, "method": method, **stats}
            curve[cell.label][method] = row
            worst_rows.append(row)
            group_curve[cell.label][method] = {}
            for group_index, group in enumerate(groups):
                gstats = corrected(
                    levels[index, :, method_index, group_index], test_sizes, train_sizes, level
                )
                grow = {**base, "method": method, "group": group, **gstats}
                group_curve[cell.label][method][group] = grow
                level_rows.append(grow)
        gap_stats = corrected(gaps[index], test_sizes, train_sizes, level)
        gap_row = {**base, "group": "worst_group", **gap_stats,
                   "p_value_holm": None, "holm_reject_at_alpha": None}
        gap_curve[cell.label] = gap_row
        gap_rows.append(gap_row)
        per_group_stats = [
            corrected(group_gaps[index, :, k], test_sizes, train_sizes, level)
            for k in range(n_groups)
        ]
        raw_p = [1.0 if row["p_value_two_sided"] is None else float(row["p_value_two_sided"])
                 for row in per_group_stats]
        adjusted = holm_adjusted(raw_p)
        group_gap_curve[cell.label] = {}
        for k, group in enumerate(groups):
            stats = per_group_stats[k]
            row = {
                **base, "group": group, **stats,
                "p_value_holm": None if stats["p_value_two_sided"] is None else float(adjusted[k]),
                "holm_reject_at_alpha": (
                    None if stats["p_value_two_sided"] is None else bool(adjusted[k] < alpha)
                ),
            }
            group_gap_curve[cell.label][group] = row
            gap_rows.append(row)
        for position, key in enumerate(order):
            split_score_rows.append({
                **base, "repeat": key[0], "fold": key[1],
                "n_train": int(train_sizes[position]), "n_test": int(test_sizes[position]),
                "split_score": float(gaps[index, position]),
                "mean_worst_pooled": float(worst[index, position, pooled_at]),
                "mean_worst_multigroup": float(worst[index, position, multigroup_at]),
                "mean_worst_uniform": float(
                    worst[index, position, SWEEP_METHODS.index(UNIFORM_ARM)]
                ),
            })
        for arm_index, arm in enumerate(ARMS):
            untied = float(np.nanmean(noise["n_untied"][index, :, arm_index]))
            ties = float(np.nanmean(noise["n_ties"][index, :, arm_index]))
            total = untied + ties
            noise_rows.append({
                **base, "arm": arm,
                "mean_flips": float(np.nanmean(noise["flips"][index, :, arm_index])),
                "mean_n_untied": untied,
                "mean_n_ties": ties,
                "tie_fraction": float(ties / total) if total > 0 else None,
                "mean_flip_rate": (
                    None if not np.isfinite(np.nanmean(noise["flip_rate"][index, :, arm_index]))
                    else float(np.nanmean(noise["flip_rate"][index, :, arm_index]))
                ),
                "mean_expected_error": (
                    None
                    if not np.isfinite(np.nanmean(noise["expected_error"][index, :, arm_index]))
                    else float(np.nanmean(noise["expected_error"][index, :, arm_index]))
                ),
                "flip_rate_denominator": "strict utility differences only; exact ties excluded",
            })

    # ---- active minus passive, paired at the split level, per arm per beta.
    by_key = {(cell.protocol, cell.beta_text): index for index, cell in enumerate(cells)}
    contrast_rows: list[dict[str, Any]] = []
    contrasts: dict[str, dict[str, Any]] = {}
    for beta_text in finite_protocol_contrast_betas(cells):
        passive_index = by_key[(PASSIVE, beta_text)]
        active_index = by_key[(ACTIVE, beta_text)]
        contrasts[beta_text] = {}
        for method_index, method in enumerate(SWEEP_METHODS):
            # Paired split scores first, corrected afterwards.  CI endpoints are
            # never subtracted and marginal SEs are never combined.
            paired = worst[active_index, :, method_index] - worst[passive_index, :, method_index]
            row = {
                "beta": beta_text, "method": method,
                "contrast": "active_minus_passive",
                **corrected(paired, test_sizes, train_sizes, level),
            }
            contrasts[beta_text][method] = row
            contrast_rows.append(row)
        paired_gap = gaps[active_index] - gaps[passive_index]
        row = {
            "beta": beta_text, "method": "paired_gap",
            "contrast": "active_minus_passive",
            **corrected(paired_gap, test_sizes, train_sizes, level),
        }
        contrasts[beta_text]["paired_gap"] = row
        contrast_rows.append(row)

    base_columns = ["cell", "protocol", "beta", *INTERVAL_KEYS]
    pd.DataFrame(worst_rows)[
        ["cell", "protocol", "beta", "method", *INTERVAL_KEYS]
    ].to_csv(reduce_dir / "cell_worst_group.csv", index=False)
    pd.DataFrame(level_rows)[
        ["cell", "protocol", "beta", "method", "group", *INTERVAL_KEYS]
    ].to_csv(reduce_dir / "cell_group_levels.csv", index=False)
    pd.DataFrame(gap_rows)[
        ["cell", "protocol", "beta", "group", *INTERVAL_KEYS,
         "p_value_holm", "holm_reject_at_alpha"]
    ].to_csv(reduce_dir / "cell_gaps.csv", index=False)
    pd.DataFrame(split_score_rows).to_csv(reduce_dir / "split_scores.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(reduce_dir / "cell_label_noise.csv", index=False)
    contrast_frame = pd.DataFrame(
        contrast_rows, columns=["beta", "method", "contrast", *INTERVAL_KEYS]
    )
    contrast_frame.to_csv(reduce_dir / "protocol_contrasts.csv", index=False)
    _write_json(reduce_dir / "protocol_contrasts.json", {
        "contrast": "active_minus_passive",
        "sign_note": "a positive contrast means HIGHER distortion under active sampling",
        "reduction": (
            "paired split scores c_s = mean_k(W_active) - mean_k(W_passive) are formed first and the "
            "Nadeau-Bengio correction is applied to those; CI endpoints are never subtracted and "
            "marginal standard errors are never combined as if independent"
        ),
        "betas": list(finite_protocol_contrast_betas(cells)),
        "interval_label": INTERVAL_LABEL,
        "rows": contrast_rows,
        "by_beta": contrasts,
    })
    for name, rows in (("cell_worst_group", worst_rows), ("cell_group_levels", level_rows),
                       ("cell_gaps", gap_rows), ("cell_label_noise", noise_rows)):
        _write_json(reduce_dir / f"{name}.json", {"rows": rows})

    anchor = next(cell for cell in cells if cell.is_anchor)
    inference = {
        "cells": [cell.as_mapping() for cell in cells],
        "cell_labels": labels,
        "protocols": sorted({cell.protocol for cell in cells}),
        "betas": sorted({cell.beta_text for cell in cells}, key=lambda t: (t == "inf", t)),
        "anchor_cell": anchor.label,
        "groups": list(groups),
        "methods": list(SWEEP_METHODS),
        "online_arms": list(ARMS),
        "benchmark_arms": [UNIFORM_ARM],
        "n_splits": len(order),
        "n_learner_seeds": len(learner_seeds),
        "confidence_level": level,
        "evaluation_tau": float(EVALUATION_TAU),
        "quantities": {
            "worst_group": "mean_over_splits( mean_seed( max_group(D_arm) ) )",
            "group_level": "mean_over_splits( mean_seed( D_arm[group] ) )",
            "gap": "mean_over_splits( mean_seed( max_group(D_pooled) - max_group(D_multigroup) ) )",
            "group_gap": "mean_over_splits( mean_seed( D_pooled[group] - D_multigroup[group] ) )",
            "protocol_contrast":
                "mean_over_splits( mean_seed(W_active) - mean_seed(W_passive) ) at one beta",
        },
        "positive_favors": "multigroup",
        "method": "nadeau_bengio_corrected_repeated_cv_t",
        "interval_label": INTERVAL_LABEL,
        "rng_coupling_note": RNG_COUPLING_NOTE,
        "score_bound_note": dict(SCORE_BOUND_NOTE),
        "uniform_note": (
            "the untrained uniform benchmark is not fitted and ignores the comparison log, so it is "
            "identical in every cell; the run asserts that per split rather than assuming it"
        ),
        "curve": curve,
        "group_curve": group_curve,
        "gaps": gap_curve,
        "group_gaps": group_gap_curve,
        "protocol_contrasts": contrasts,
        "label_noise": noise_rows,
        "status": gap_curve[anchor.label]["status"],
    }
    _write_json(reduce_dir / "inference.json", inference)
    del base_columns
    return inference


# ------------------------------------------------------------ anchor gate


def frozen_group_order(design: Mapping[str, Any]) -> list[str]:
    """The selected group family, in order, from either dataset's design manifest.

    The Coffee routes record analyst-supplied ``definitions``; the persona routes
    record the discovery-``selected`` family.  Both are the group axis of the
    saved arrays, so either is accepted -- but an absent one is not silently
    read as "no groups".
    """

    block = design.get("groups", {})
    definitions = block.get("definitions")
    if definitions:
        return [str(item["name"]) for item in definitions]
    selected = block.get("selected")
    if selected:
        return [str(name) for name in selected]
    raise RuntimeError(
        "the frozen design manifest records neither groups.definitions nor groups.selected, so "
        "the group axis of its saved arrays cannot be verified"
    )


def _load_frozen_split(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "split.npz", allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def check_anchor_against_frozen_run(
    *,
    output_dir: Path,
    frozen_run: Path,
    base_protocol_id: str,
    base_config_digest: str,
    data_sha256: str,
    groups: Sequence[str],
    learner_seeds: Sequence[int],
    repeats: int,
    folds: int,
    cells: Sequence[SweepCell],
    split_summaries: Sequence[Mapping[str, Any]],
    inference: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the refitted anchor cell with the frozen source run, array by array.

    This is a **production gate**.  A missing artifact or any mismatch raises:
    nothing is skipped, no other frozen run is substituted, and equality is never
    loosened to make it pass.  The anchor's own arrays are recomputed inside this
    sweep's namespace -- the frozen arrays are read only to compare.
    """

    frozen_run = Path(frozen_run)
    design_path = frozen_run / "design" / "design.json"
    inference_path = frozen_run / "reduce" / "inference.json"
    for path in (design_path, inference_path):
        if not path.is_file():
            raise RuntimeError(
                f"the anchor check requires the frozen source run, but {path} is missing. "
                "Point --frozen-run at a completed run of the source route; the gate is never skipped."
            )
    design = json.loads(design_path.read_text(encoding="utf-8"))
    frozen_inference = json.loads(inference_path.read_text(encoding="utf-8"))

    identity: dict[str, Any] = {}
    problems: list[str] = []

    def _expect(name: str, observed: Any, expected: Any) -> None:
        identity[name] = {"frozen": observed, "sweep": expected, "agrees": observed == expected}
        if observed != expected:
            problems.append(f"{name}: frozen {observed!r} != sweep {expected!r}")

    _expect("protocol_id", design.get("protocol_id"), base_protocol_id)
    _expect("config_sha256", design.get("config_sha256"), base_config_digest)
    _expect("data_sha256", design.get("data", {}).get("sha256"), data_sha256)
    _expect("groups", frozen_group_order(design), list(groups))
    _expect("learner_seeds", [int(v) for v in design.get("learner_seeds", [])],
            [int(v) for v in learner_seeds])
    _expect("repeats", int(design.get("cross_validation", {}).get("repeats", -1)), int(repeats))
    _expect("folds", int(design.get("cross_validation", {}).get("folds", -1)), int(folds))
    discovery = design.get("discovery", {})
    identity["discovery"] = {
        "seed": discovery.get("seed"), "k": discovery.get("k"),
        "n_analysis": discovery.get("n_analysis"),
        "discovery_ids_sha256": discovery.get("discovery_ids_sha256"),
    }

    frozen_splits = {
        (int(row["repeat"]), int(row["fold"])): row
        for row in design.get("cross_validation", {}).get("splits", [])
    }
    anchor = next(cell for cell in cells if cell.is_anchor)
    anchor_index = [cell.label for cell in cells].index(anchor.label)
    seen = {(int(row["repeat"]), int(row["fold"])): row for row in split_summaries}
    planned = sorted({(r, f) for r in range(repeats) for f in range(folds)})

    compared = 0
    mismatches: list[dict[str, Any]] = []
    for key in planned:
        frozen_dir = frozen_run / "splits" / f"r{key[0]:02d}_f{key[1]:02d}"
        if not (frozen_dir / "split.npz").is_file():
            raise RuntimeError(
                f"the frozen source run is missing split {key} ({frozen_dir}); the anchor gate "
                "requires all 100 splits and is never partially satisfied"
            )
        frozen_manifest = json.loads((frozen_dir / "split.json").read_text(encoding="utf-8"))
        mine = seen[key]
        for name in ("train_ids_sha256", "test_ids_sha256", "n_train", "n_test"):
            if frozen_manifest.get(name) != mine.get(name):
                problems.append(
                    f"split {key} {name}: frozen {frozen_manifest.get(name)!r} != "
                    f"sweep {mine.get(name)!r}"
                )
        plan_row = frozen_splits.get(key, {})
        if plan_row and plan_row.get("train_ids_sha256") != mine.get("train_ids_sha256"):
            problems.append(f"split {key} fold membership disagrees with the frozen design manifest")
        frozen_arrays = _load_frozen_split(frozen_dir)
        ours = anchor_arrays(split_arrays(Path(output_dir), *key), anchor_index)
        for key_name, frozen_name in ANCHOR_ARRAY_MAP.items():
            if frozen_name not in frozen_arrays:
                continue
            left, right = ours[key_name], frozen_arrays[frozen_name]
            if left.shape != right.shape or not np.array_equal(left, right):
                difference = (
                    None if left.shape != right.shape or left.dtype.kind not in "fiu"
                    else float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                )
                mismatches.append({
                    "repeat": key[0], "fold": key[1], "array": frozen_name,
                    "sweep_shape": list(left.shape), "frozen_shape": list(right.shape),
                    "max_abs_difference": difference,
                })
            compared += 1

    gap = inference["gaps"][anchor.label]
    headline_keys = ("estimate", "corrected_se", "ci_lower", "ci_upper", "t_statistic",
                     "p_value_two_sided", "degrees_of_freedom")
    headline: dict[str, Any] = {}
    for name in headline_keys:
        frozen_value = frozen_inference.get(name)
        sweep_value = gap.get(name)
        agrees = (
            frozen_value == sweep_value
            if frozen_value is None or sweep_value is None
            else float(frozen_value) == float(sweep_value)
        )
        headline[name] = {
            "frozen": frozen_value, "sweep": sweep_value, "agrees": bool(agrees),
            "absolute_difference": (
                None if frozen_value is None or sweep_value is None
                else abs(float(frozen_value) - float(sweep_value))
            ),
        }
        if not agrees:
            problems.append(f"headline {name}: frozen {frozen_value!r} != sweep {sweep_value!r}")
    if int(frozen_inference.get("n_scores", -1)) != int(gap.get("n_splits", -2)):
        problems.append("the frozen run and the anchor reduce a different number of split scores")

    report = {
        "checked": True,
        "frozen_run": str(frozen_run),
        "anchor_cell": anchor.label,
        "base_protocol_id": base_protocol_id,
        "identity": identity,
        "splits_compared": len(planned),
        "arrays_compared": compared,
        "arrays_compared_per_split": sorted(ANCHOR_ARRAY_MAP.values()),
        "excluded_from_equality": [
            "the untrained uniform benchmark axis (the frozen run never fitted it)",
            "timings, seed records and route-specific metadata",
        ],
        "array_mismatches": mismatches,
        "headline": headline,
        "problems": problems,
        "agrees": not problems and not mismatches,
        "environment": {
            "note": (
                "exact reproduction is expected because the anchor is the same computation on the "
                "same inputs; any residual difference here would be an environment or "
                "implementation discrepancy and is reported, never absorbed"
            ),
        },
    }
    return report


# ---------------------------------------------------------------- summary


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _interval_text(row: Mapping[str, Any]) -> str:
    if row.get("ci_lower") is None:
        return "undefined"
    return f"[{_fmt(row['ci_lower'])}, {_fmt(row['ci_upper'])}]"


def write_summary(
    output_dir: Path, summary: Mapping[str, Any], *, title: str, standing_note: Sequence[str],
) -> None:
    """The human-readable run record, in the tau sweeps' shape."""

    inference = summary.get("inference", {})
    info = summary["run_info"]
    labels = inference.get("cell_labels", [])
    lines = [
        f"# {title}", "",
        f"**Status:** {info['status']}", "",
        *[f"> {line}" for line in standing_note], "",
        f"Started {info.get('started_at')}; finished {info.get('finished_at')}.", "",
        "## Design", "",
        f"- Source design: `{info.get('base_protocol_id')}` — same data, discovery reservation, "
        f"cv seed, fold plan, group family, learner seeds, `tau = {inference.get('learner_tau', 0.5)}` "
        f"and evaluation ball {inference.get('evaluation_tau')}.",
        f"- Cells: {', '.join(f'`{label}`' for label in labels)}",
        f"- Anchor: `{inference.get('anchor_cell')}` — the source design's own feedback model, "
        "refitted here and compared array by array with the frozen run.",
        f"- Arms: {', '.join(f'`{m}`' for m in inference.get('methods', []))} "
        "(the uniform benchmark needs no fit).",
        "- The respondent order is the passive log's own permutation and is shared by every cell, "
        "active included.", "",
        "## Worst-group distortion by cell", "",
        "`mean_over_splits( mean_seed( max_group(D) ) )` (lower is better)", "",
        "| protocol | beta | pooled | 95% corrected interval | multigroup | 95% corrected interval "
        "| uniform |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for label in labels:
        cells = inference["curve"][label]
        head = cells["pooled"]
        row = [f"| `{head['protocol']}` | `{head['beta']}` "]
        for arm in ARMS:
            cell = cells[arm]
            row.append(f"| {_fmt(cell['estimate'])} | {_interval_text(cell)} ")
        row.append(f"| {_fmt(cells[UNIFORM_ARM]['estimate'])} ")
        lines.append("".join(row) + "|")

    lines += [
        "", "## Paired pooled minus multigroup gap by cell", "",
        f"`{inference.get('quantities', {}).get('gap', '')}` (positive favors multigroup)", "",
        "| protocol | beta | estimate | corrected SE | 95% corrected interval | p |",
        "|---|---|---:|---:|---|---:|",
    ]
    for label in labels:
        cell = inference["gaps"][label]
        p = cell["p_value_two_sided"]
        lines.append(
            f"| `{cell['protocol']}` | `{cell['beta']}` | {_fmt(cell['estimate'])} | "
            f"{_fmt(cell['corrected_se'])} | {_interval_text(cell)} | "
            f"{'n/a' if p is None else f'{p:.3e}'} |"
        )

    contrasts = inference.get("protocol_contrasts", {})
    if contrasts:
        lines += [
            "", "## Active minus passive, paired at the split level", "",
            "Positive means **higher** distortion under active sampling.", "",
            "| beta | quantity | estimate | 95% corrected interval | p |",
            "|---|---|---:|---|---:|",
        ]
        for beta_text, rows in contrasts.items():
            for method, row in rows.items():
                p = row["p_value_two_sided"]
                lines.append(
                    f"| `{beta_text}` | `{method}` | {_fmt(row['estimate'])} | "
                    f"{_interval_text(row)} | {'n/a' if p is None else f'{p:.3e}'} |"
                )

    lines += ["", "## Label noise on each cell's own queried pairs", "",
              "Flip rates count disagreement with `sign(u_i - u_j)` on **strictly** ordered pairs "
              "only; exact ties are fair coins even at `beta = inf` and never enter the "
              "denominator.", "",
              "| protocol | beta | arm | mean flips | non-tied | ties | tie fraction | flip rate | "
              "E[error] |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in inference.get("label_noise", []):
        lines.append(
            f"| `{row['protocol']}` | `{row['beta']}` | `{row['arm']}` | "
            f"{_fmt(row['mean_flips'], 1)} | {_fmt(row['mean_n_untied'], 1)} | "
            f"{_fmt(row['mean_n_ties'], 1)} | {_fmt(row['tie_fraction'])} | "
            f"{_fmt(row['mean_flip_rate'])} | {_fmt(row['mean_expected_error'])} |"
        )

    anchor = summary.get("anchor_check", {})
    if anchor:
        lines += [
            "", "## Anchor check (production gate)", "",
            f"- Frozen source run: `{anchor.get('frozen_run')}`",
            f"- Splits compared: {anchor.get('splits_compared')}; arrays compared: "
            f"{anchor.get('arrays_compared')}",
            f"- Agrees: **{anchor.get('agrees')}**",
            f"- Machine-readable report: `reduce/anchor_check.json`",
        ]

    for key in ("interval_label", "rng_coupling_note", "uniform_note"):
        text = inference.get(key)
        if text:
            lines += ["", text[0].upper() + text[1:]]
    if inference.get("score_bound_note"):
        lines += [
            "", "Score bound used to normalize rewards, per protocol (the learner's existing, "
            "unmodified choice): "
            + "; ".join(f"`{k}` = {v}" for k, v in inference["score_bound_note"].items())
            + ". Read active minus passive as the effect of the implemented protocol, its native "
              "normalization included, not of the queried pairs alone.",
        ]
    lines += ["", "## Where the outputs are", ""]
    for label, relative in summary.get("outputs", {}).items():
        lines.append(f"- {label}: `{relative}`")
    (Path(output_dir) / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================== Coffee variant

PROTOCOL_ID = "coffee-crossfit-v4-protocol-beta-sweep"
IMPLEMENTATION_ID = "coffee-protocol-beta-sweep-2026-09-20.1"
MODE = "crossfit-10x10-v4-protocol-beta"
BASE_ROUTE = ROUTE_V4_10X10

TITLE = "Held-out distortion by sampling protocol and Bradley-Terry beta (Coffee v4)"
STANDING_NOTE = (
    "This sweep uses the `coffee-repeated-cv-v4-10x10` design. Its per-cell intervals are marginal",
    "Nadeau-Bengio corrected repeated-CV intervals — not simultaneous over the grid,",
    "the arms or the groups, and not the manuscript's bootstrap bands. Because the v4 analysis",
    "complement is a different 90% of the survey, these numbers are reported beside the v4 route's",
    "own numbers only and are **never** compared with `coffee-repeated-cv-v1`, `-v2-10x10` or",
    "`-v3-10x10`.",
)


@dataclass(frozen=True)
class ProtocolBetaSweepConfig:
    """The frozen source configuration plus this sweep's own, separately validated axes.

    The wrapper keeps the source route's validated configuration object exactly
    as it is -- that validator still rejects a finite ``beta`` and a non-passive
    protocol, and is never handed either.  The sweep axes live here and are
    validated here.
    """

    base: CoffeeCrossfitConfig
    cells: tuple[SweepCell, ...]
    evaluation_tau: float = EVALUATION_TAU

    def validate(self) -> "ProtocolBetaSweepConfig":
        object.__setattr__(self, "cells", validate_cells(self.cells))
        if self.base.route is not BASE_ROUTE:
            raise ValueError(
                f"this sweep runs on the {BASE_ROUTE.protocol_id!r} design but the configuration "
                f"declares {self.base.route.protocol_id!r}"
            )
        if self.base.comparisons_per_respondent != 1:
            raise ValueError("this sweep requires one comparison per training respondent")
        if not np.isclose(self.evaluation_tau, EVALUATION_TAU):
            raise ValueError(
                f"the evaluation ball is held at tau = {EVALUATION_TAU}; beta changes the training "
                "labels, not the held-out target"
            )
        if not np.isclose(self.base.tau, EVALUATION_TAU):
            raise ValueError("the source design's learner radius tau = 0.5 is held fixed here")
        return self

    @property
    def cell_labels(self) -> list[str]:
        return [cell.label for cell in self.cells]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "base_protocol_id": self.base.route.protocol_id,
            "base_config_sha256": self.base.digest(),
            "cells": [cell.as_mapping() for cell in self.cells],
            "anchor_cell": next(c.label for c in self.cells if c.is_anchor),
            "evaluation_tau": float(self.evaluation_tau),
            "learner_tau": float(self.base.tau),
            "groups": list(self.base.groups),
            "learner_seeds": list(self.base.learner_seeds),
            "arms": list(ARMS),
            "methods": list(SWEEP_METHODS),
            "repeats": self.base.repeats,
            "folds": self.base.folds,
            "n_splits": self.base.n_splits,
            "rng_convention": {
                "training_seed": "SeedSequence([cv_seed, repeat, fold, learner_seed]).generate_state(1)",
                "active_learner": ACTIVE_RNG_DERIVATION,
                "passive_relabelling": PASSIVE_RELABEL_RNG_DERIVATION,
                "note": RNG_COUPLING_NOTE,
            },
            "score_bound": dict(SCORE_BOUND_NOTE),
            "policy_fits": self.base.n_splits * len(self.base.learner_seeds)
                           * len(ARMS) * len(self.cells),
        }

    def digest(self) -> str:
        payload = json.dumps(self.as_mapping(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def compute_split(
    config: ProtocolBetaSweepConfig, inputs: Any, repeat: int, fold: int
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    from empirical_testing.experiments.coffee_crossfit import _check_split_groups

    base = config.base
    dataset, reservation, plan = inputs.dataset, inputs.reservation, inputs.plan
    train_pos, test_pos = plan.split_positions(repeat, fold)
    train_rows = reservation.analysis_indices[train_pos]
    test_rows = reservation.analysis_indices[test_pos]
    if np.intersect1d(train_rows, test_rows).size:
        raise AssertionError("a respondent appears in both the training set and the test fold")
    if np.intersect1d(np.concatenate((train_rows, test_rows)),
                      reservation.discovery_indices).size:
        raise AssertionError("a reserved discovery respondent entered a cross-validation split")
    train_view = subset_dense_dataset(dataset, train_rows)
    test_view = subset_dense_dataset(dataset, test_rows)
    _check_split_groups(base, train_view, test_view, repeat, fold)

    summary, arrays, active_logs = compute_split_cells(
        algorithm=base.algorithm, cv_seed=base.cv_seed, learner_seeds=base.learner_seeds,
        comparisons_per_respondent=base.comparisons_per_respondent, tau=base.tau,
        groups=base.groups, cells=config.cells, train_view=train_view, test_view=test_view,
        repeat=repeat, fold=fold,
    )
    ids = np.asarray(dataset.respondent_ids).astype(str)
    summary.update({
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "base_protocol_id": base.route.protocol_id,
        "config_sha256": config.digest(),
        "base_config_sha256": base.digest(),
        "data_sha256": inputs.data_sha256,
        "n_train": int(train_rows.size), "n_test": int(test_rows.size),
        "train_ids_sha256": sorted_id_sha256(ids[train_rows]),
        "test_ids_sha256": sorted_id_sha256(ids[test_rows]),
        "discovery_ids_sha256": sorted_id_sha256(ids[reservation.discovery_indices]),
    })
    return summary, arrays, active_logs


REUSE_KEYS = (
    "protocol_id", "implementation_id", "base_protocol_id", "config_sha256", "base_config_sha256",
    "data_sha256", "repeat", "fold", "train_ids_sha256", "test_ids_sha256",
    "discovery_ids_sha256", "groups", "learner_seeds", "cell_labels", "methods",
    "evaluation_tau", "learner_tau",
)


def run_split(
    config: ProtocolBetaSweepConfig, inputs: Any, output_dir: Path, repeat: int, fold: int
) -> dict[str, Any]:
    """Compute one split unless an identical one is already saved; publish atomically."""

    base = config.base
    directory = split_dir(output_dir, repeat, fold)
    ids = np.asarray(inputs.dataset.respondent_ids).astype(str)
    train_pos, test_pos = inputs.plan.split_positions(repeat, fold)
    expected = {
        "protocol_id": PROTOCOL_ID, "implementation_id": IMPLEMENTATION_ID,
        "base_protocol_id": base.route.protocol_id,
        "config_sha256": config.digest(), "base_config_sha256": base.digest(),
        "data_sha256": inputs.data_sha256, "repeat": int(repeat), "fold": int(fold),
        "train_ids_sha256": sorted_id_sha256(ids[inputs.reservation.analysis_indices[train_pos]]),
        "test_ids_sha256": sorted_id_sha256(ids[inputs.reservation.analysis_indices[test_pos]]),
        "discovery_ids_sha256": sorted_id_sha256(ids[inputs.reservation.discovery_indices]),
        "groups": list(base.groups), "learner_seeds": list(base.learner_seeds),
        "cell_labels": config.cell_labels, "methods": list(SWEEP_METHODS),
        "evaluation_tau": float(config.evaluation_tau), "learner_tau": float(base.tau),
    }
    reused = saved_split(
        directory, expected, n_cells=len(config.cells), n_seeds=len(base.learner_seeds),
        n_groups=len(base.groups),
    )
    if reused is not None:
        reused["reused"] = True
        return reused
    summary, arrays, active_logs = compute_split(config, inputs, repeat, fold)
    publish_split(directory, summary, arrays, active_logs)
    summary["reused"] = False
    return summary


_WORKER_STATE: dict[str, Any] = {}


def _init_worker(config_mapping: Mapping[str, Any], cell_spec: Sequence[Mapping[str, Any]],
                 project_root: str, output_dir: str, expected_data_sha256: str | None) -> None:
    if _WORKER_STATE.get("output_dir") == output_dir:
        return
    base = CoffeeCrossfitConfig.from_mapping(config_mapping)
    config = ProtocolBetaSweepConfig(
        base=base,
        cells=tuple(SweepCell(item["protocol"], parse_beta(item["beta"])) for item in cell_spec),
    ).validate()
    inputs = load_crossfit_inputs(
        base, project_root=Path(project_root), expected_data_sha256=expected_data_sha256
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(config=config, inputs=inputs, output_dir=output_dir)


def _run_split_task(task: tuple[int, int]) -> dict[str, Any]:
    repeat, fold = task
    return run_split(
        _WORKER_STATE["config"], _WORKER_STATE["inputs"], Path(_WORKER_STATE["output_dir"]),
        repeat, fold,
    )


@dataclass
class SweepResult:
    exit_code: int
    output_dir: Path
    summary: dict[str, Any]


def run_protocol_beta_sweep(
    *,
    project_root: Path,
    output_dir: Path,
    frozen_run: Path,
    config_path: Path | None = None,
    config: CoffeeCrossfitConfig | None = None,
    cells: Sequence[SweepCell] | None = None,
    expected_data_sha256: str | None = None,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> SweepResult:
    """Run the whole protocol x beta grid into ``output_dir``, resuming saved splits."""

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = config or load_crossfit_config(
        config_path or project_root / "configs" / BASE_ROUTE.config_name
    )
    sweep = ProtocolBetaSweepConfig(
        base=base, cells=tuple(cells) if cells is not None else default_cells()
    ).validate()

    try:
        from coffee_expanded.manifests import runtime_versions

        versions = runtime_versions()
    except Exception:  # pragma: no cover - the fallback is environment-dependent
        import sys as _sys

        versions = {"python": _sys.version.split()[0]}

    info: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID, "mode": MODE, "reportable": True, "status": "running",
        "implementation_id": IMPLEMENTATION_ID, "base_protocol_id": base.route.protocol_id,
        "started_at": _now(), "workers": int(workers), "package_versions": versions,
        "cells": [cell.as_mapping() for cell in sweep.cells],
        "config_sha256": sweep.digest(), "frozen_run": str(frozen_run),
    }
    summary: dict[str, Any] = {"run_info": info, "outputs": {
        "run record": "run_info.json", "log": "run.log",
        "design manifest": "design/design.json",
        "per-split results": "splits/rNN_fMM/ (split.json, split.npz, comparisons.npz)",
        "tables": "reduce/ (cell_worst_group, cell_group_levels, cell_gaps, split_scores, "
                  "cell_label_noise, protocol_contrasts; CSV + JSON)",
        "inference record": "reduce/inference.json",
        "anchor check": "reduce/anchor_check.json",
    }}

    def save_info() -> None:
        _write_json(output_dir / "run_info.json", info)

    save_info()
    try:
        from empirical_testing.experiments.coffee_crossfit import design_manifest, validate_design

        inputs = load_crossfit_inputs(
            base, project_root=project_root, expected_data_sha256=expected_data_sha256
        )
        validate_design(base, inputs)
        manifest = design_manifest(base, inputs)
        manifest["protocol_beta_sweep"] = sweep.as_mapping()
        _write_json(output_dir / "design" / "design.json", manifest)
        n_fits = sweep.as_mapping()["policy_fits"]
        log(
            f"protocol x beta sweep: {len(inputs.reservation.discovery_indices)} discovery "
            f"respondents excluded; {inputs.reservation.n_analysis} remain for "
            f"{base.n_splits} splits x {len(sweep.cells)} cells x {len(base.learner_seeds)} seeds "
            f"x {len(ARMS)} arms = {n_fits} adaptive fits"
        )
        log("protocol x beta sweep: cells " + ", ".join(sweep.cell_labels))

        tasks = [(r, f) for r in range(base.repeats) for f in range(base.folds)]
        worker_args = (base.as_mapping(), [cell.as_mapping() for cell in sweep.cells],
                       str(project_root), str(output_dir), expected_data_sha256)
        started = time.perf_counter()

        def on_progress(done: int, total: int, task: tuple[int, int], seconds: float) -> None:
            log(f"protocol x beta split repeat={task[0]} fold={task[1]} done in {seconds:.1f} s "
                f"({done}/{total})")

        split_summaries, report = ordered_map(
            _run_split_task, tasks, workers=int(workers), initializer=_init_worker,
            initargs=worker_args, on_progress=on_progress, log=log,
        )
        info["splits_seconds"] = round(time.perf_counter() - started, 2)
        info["splits_reused"] = int(sum(1 for row in split_summaries if row.get("reused")))
        info["parallelism"] = report.summary()
        log(f"protocol x beta sweep: {len(split_summaries)} splits complete "
            f"({info['splits_reused']} reused); {report.summary()}")
        save_info()

        log("protocol x beta sweep: reducing to per-cell corrected intervals")
        inference = reduce_cells(
            output_dir=output_dir, cells=sweep.cells, groups=base.groups,
            learner_seeds=base.learner_seeds, repeats=base.repeats, folds=base.folds,
            confidence_level=base.confidence_level, split_summaries=split_summaries,
        )
        inference["protocol_id"] = PROTOCOL_ID
        inference["implementation_id"] = IMPLEMENTATION_ID
        inference["base_protocol_id"] = base.route.protocol_id
        inference["learner_tau"] = float(base.tau)
        _write_json(output_dir / "reduce" / "inference.json", inference)
        summary["inference"] = inference

        log("protocol x beta sweep: anchor check against the frozen source run")
        anchor = check_anchor_against_frozen_run(
            output_dir=output_dir, frozen_run=frozen_run,
            base_protocol_id=base.route.protocol_id, base_config_digest=base.digest(),
            data_sha256=inputs.data_sha256, groups=base.groups,
            learner_seeds=base.learner_seeds, repeats=base.repeats, folds=base.folds,
            cells=sweep.cells, split_summaries=split_summaries, inference=inference,
        )
        _write_json(output_dir / "reduce" / "anchor_check.json", anchor)
        summary["anchor_check"] = anchor
        info["anchor_check"] = {"agrees": anchor["agrees"],
                                "splits_compared": anchor["splits_compared"],
                                "arrays_compared": anchor["arrays_compared"]}
        if not anchor["agrees"]:
            raise RuntimeError(
                "the anchor cell does not reproduce the frozen source run: "
                f"{anchor['problems'][:3]} {anchor['array_mismatches'][:3]}. The anchor is the "
                "source design's own feedback model refitted here, so this is a defect, not a "
                "finding; the run fails rather than loosening the comparison."
            )
        log(f"protocol x beta sweep: anchor reproduces {frozen_run} exactly "
            f"({anchor['arrays_compared']} arrays over {anchor['splits_compared']} splits)")

        info["status"] = "completed"
        info["finished_at"] = _now()
        save_info()
        write_summary(output_dir, summary, title=TITLE, standing_note=STANDING_NOTE)
        _write_json(output_dir / "summary.json", summary)
        exit_code = 0 if inference["status"] == NB_STATUS_OK else 3
    except Exception:
        info["status"] = "failed"
        info["finished_at"] = _now()
        save_info()
        raise
    return SweepResult(exit_code=exit_code, output_dir=output_dir, summary=summary)
