"""Discovery reservations, repeated k-fold partitions, and Nadeau-Bengio intervals.

These routines operate on respondent counts and score arrays without file I/O
or policy fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "DiscoveryReservation",
    "RepeatedFoldPlan",
    "NadeauBengioResult",
    "NB_STATUS_OK",
    "NB_STATUS_DEGENERATE",
    "NB_STATUS_INSUFFICIENT",
    "NB_STATUS_NONFINITE",
    "discovery_reservation",
    "stratified_discovery_reservation",
    "repeated_fold_plan",
    "split_learner_seed",
    "nadeau_bengio",
]

NB_STATUS_OK = "ok"
NB_STATUS_DEGENERATE = "degenerate_zero_variance"
NB_STATUS_INSUFFICIENT = "insufficient_split_scores"
NB_STATUS_NONFINITE = "nonfinite_split_scores"


@dataclass(frozen=True)
class DiscoveryReservation:
    """The reserved discovery sample and the analysis complement.

    ``discovery_indices`` and ``analysis_indices`` are row positions into the
    eligible dataset *in the order the adapter produced it* (source CSV row
    order after the complete-profile filter).  They are sorted ascending; the
    unsorted draw order is kept in ``draw_order`` so the exact ``choice`` call
    can be audited.
    """

    n_rows: int
    fraction: float
    seed: int
    draw_order: np.ndarray
    discovery_indices: np.ndarray
    analysis_indices: np.ndarray

    @property
    def n_discovery(self) -> int:
        return int(self.discovery_indices.size)

    @property
    def n_analysis(self) -> int:
        return int(self.analysis_indices.size)


def discovery_reservation(n_rows: int, *, fraction: float = 0.10, seed: int = 20260916) -> DiscoveryReservation:
    """Reserve ``ceil(fraction * n_rows)`` eligible rows as the discovery sample.

    The draw is exactly the one that was specified for this study::

        np.random.default_rng(seed).choice(n_rows, k, replace=False)

    with two conventions this function makes explicit, because the instruction
    alone does not pin them down:

    * ``n_rows`` counts eligible rows **after** the complete-profile filter, in
      the source CSV row order the adapter retains.  IDs are not sorted first.
    * ``k`` rounds **up**: ``ceil(0.10 * 3744) == 375``.

    The generator is used for this one call and for nothing else, so the draw
    does not depend on any other random work in the run.
    """

    n_rows = int(n_rows)
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("the discovery fraction must lie strictly between zero and one")
    k = math.ceil(float(fraction) * n_rows)
    if not 0 < k < n_rows:
        raise ValueError("the discovery fraction leaves no analysis respondents")
    draw = np.random.default_rng(int(seed)).choice(n_rows, k, replace=False)
    is_discovery = np.zeros(n_rows, dtype=bool)
    is_discovery[draw] = True
    discovery_indices = np.flatnonzero(is_discovery)
    analysis_indices = np.flatnonzero(~is_discovery)
    if discovery_indices.size != k:
        raise AssertionError("the discovery draw did not return distinct rows")
    return DiscoveryReservation(
        n_rows=n_rows, fraction=float(fraction), seed=int(seed),
        draw_order=np.asarray(draw, dtype=np.int64),
        discovery_indices=discovery_indices.astype(np.int64),
        analysis_indices=analysis_indices.astype(np.int64),
    )


def stratified_discovery_reservation(
    strata: Sequence[int] | np.ndarray, *, fraction: float = 0.10, seed: int
) -> DiscoveryReservation:
    """Reserve ``ceil(fraction * n_rows)`` rows, balanced across ``strata``.

    Used where the rows carry a designed factorial structure that an unstratified
    draw would disturb -- the synthetic persona panel has exactly 125
    pseudo-respondents in each of its 16 cells, and a simple random 10% draw
    would leave the cells unequal for no reason.

    The allocation depends only on the target size and the number of strata::

        base  = n_discovery // n_strata          (12 for 200 over 16 cells)
        extra = n_discovery %  n_strata          (8 cells get one more)

    and the ``extra`` strata that receive the additional row are themselves
    drawn from the declared seed, ``default_rng(SeedSequence([seed, 0]))``.
    Each stratum's own members are then drawn with its own generator,
    ``default_rng(SeedSequence([seed, 1, stratum]))``, so one stratum's draw
    never depends on another's size or on the row order between strata.

    Strata are visited in ascending label order, and ``draw_order`` records the
    concatenated per-stratum draws in that order.  ``discovery_indices`` and
    ``analysis_indices`` are sorted row positions, exactly as in
    :func:`discovery_reservation`.
    """

    labels = np.asarray(strata)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("strata must be a nonempty one-dimensional array of labels")
    n_rows = int(labels.size)
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("the discovery fraction must lie strictly between zero and one")
    k = math.ceil(float(fraction) * n_rows)
    if not 0 < k < n_rows:
        raise ValueError("the discovery fraction leaves no analysis respondents")
    unique = np.unique(labels)
    n_strata = int(unique.size)
    if k < n_strata:
        raise ValueError("the discovery sample is smaller than the number of strata")
    base, extra = divmod(k, n_strata)
    bonus_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    bonus = set(np.sort(bonus_rng.choice(n_strata, extra, replace=False)).tolist()) if extra else set()
    draws: list[np.ndarray] = []
    for position, label in enumerate(unique):
        members = np.flatnonzero(labels == label)
        take = base + (1 if position in bonus else 0)
        if take > members.size:
            raise ValueError(
                f"stratum {label!r} has {members.size} rows but the balanced allocation needs {take}"
            )
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1, int(position)]))
        draws.append(members[rng.choice(members.size, take, replace=False)])
    draw_order = np.concatenate(draws).astype(np.int64)
    if draw_order.size != k or np.unique(draw_order).size != k:
        raise AssertionError("the stratified draw did not return k distinct rows")
    is_discovery = np.zeros(n_rows, dtype=bool)
    is_discovery[draw_order] = True
    return DiscoveryReservation(
        n_rows=n_rows, fraction=float(fraction), seed=int(seed), draw_order=draw_order,
        discovery_indices=np.flatnonzero(is_discovery).astype(np.int64),
        analysis_indices=np.flatnonzero(~is_discovery).astype(np.int64),
    )


@dataclass(frozen=True)
class RepeatedFoldPlan:
    """``fold_of[r, i]`` is the fold that analysis position ``i`` is tested in on repeat ``r``.

    Every analysis respondent is tested exactly once per repeat and trains on
    the other ``folds - 1`` parts of that repeat.  Discovery respondents are
    absent by construction: the plan only ever indexes the analysis complement.
    """

    seed: int
    repeats: int
    folds: int
    fold_of: np.ndarray  # (repeats, n_analysis), int64 in [0, folds)

    @property
    def n_analysis(self) -> int:
        return int(self.fold_of.shape[1])

    def split_positions(self, repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(train_positions, test_positions)`` into the analysis complement."""

        if not 0 <= int(repeat) < self.repeats:
            raise ValueError("repeat is outside the plan")
        if not 0 <= int(fold) < self.folds:
            raise ValueError("fold is outside the plan")
        assignment = self.fold_of[int(repeat)]
        return np.flatnonzero(assignment != int(fold)), np.flatnonzero(assignment == int(fold))

    def splits(self):
        for repeat in range(self.repeats):
            for fold in range(self.folds):
                yield repeat, fold


def repeated_fold_plan(n_analysis: int, *, repeats: int = 10, folds: int = 5, seed: int) -> RepeatedFoldPlan:
    """Build ``repeats`` independent balanced ``folds``-way partitions.

    Each repeat draws its own permutation from ``default_rng(SeedSequence([seed,
    repeat]))`` and cuts it into ``folds`` near-equal contiguous blocks with
    ``numpy.array_split``, so fold sizes differ by at most one respondent.  The
    derivation uses only the declared seed and the repeat index; it never uses
    Python's randomized ``hash()`` and never depends on task scheduling, so the
    plan is identical whether the run is serial, parallel, or resumed.
    """

    n_analysis = int(n_analysis)
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    if folds < 2:
        raise ValueError("folds must be at least two")
    if n_analysis < folds:
        raise ValueError("fewer analysis respondents than folds")
    fold_of = np.empty((int(repeats), n_analysis), dtype=np.int64)
    for repeat in range(int(repeats)):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(repeat)]))
        permutation = rng.permutation(n_analysis)
        for fold, block in enumerate(np.array_split(permutation, int(folds))):
            fold_of[repeat, block] = fold
    return RepeatedFoldPlan(seed=int(seed), repeats=int(repeats), folds=int(folds), fold_of=fold_of)


def split_learner_seed(cv_seed: int, repeat: int, fold: int, learner_seed: int) -> int:
    """A reproducible 32-bit training seed for one (repeat, fold, learner seed).

    Both arms of a split receive this one seed, so they share the comparison log
    exactly as the single-holdout protocol requires.  ``SeedSequence`` mixes the
    four declared identities; nothing here reads the clock, the process, or the
    order in which splits happen to be dispatched.
    """

    state = np.random.SeedSequence(
        [int(cv_seed), int(repeat), int(fold), int(learner_seed)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


@dataclass(frozen=True)
class NadeauBengioResult:
    """The corrected repeated-CV t interval and test for one score vector.

    This is the conventional Nadeau-Bengio approximation for repeated
    cross-validation, not an exact independence or coverage theorem.  The
    ``m`` split scores overlap heavily (each respondent trains in most of
    them), and the correction widens the naive interval to acknowledge that;
    reusing the same data is not new data.  Coverage has not been established
    for this particular worst-group ratio statistic.
    """

    status: str
    n_scores: int
    estimate: float
    variance: float
    test_train_ratio: float
    variance_factor: float
    corrected_se: float
    degrees_of_freedom: int
    confidence_level: float
    ci_lower: float | None
    ci_upper: float | None
    t_statistic: float | None
    p_value_two_sided: float | None
    naive_se: float | None
    naive_ci_lower: float | None
    naive_ci_upper: float | None

    def as_mapping(self) -> dict:
        payload = dict(vars(self))
        payload["naive_interval_label"] = (
            "diagnostic only, not for reporting: treats the overlapping split scores as "
            "independent and therefore understates the uncertainty"
        )
        return payload


def nadeau_bengio(
    scores: Sequence[float] | np.ndarray,
    test_sizes: Sequence[int] | np.ndarray,
    train_sizes: Sequence[int] | np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> NadeauBengioResult:
    """Corrected interval and two-sided test for paired repeated-CV split scores.

    ``scores`` holds one score per split (50 in the production design: ten
    repeats of five folds), *not* one per learner seed and not one per repeat.
    ``test_sizes``/``train_sizes`` are that split's respondent counts; their
    mean ratio is the variance correction term, so the one-person differences
    between fold sizes are handled exactly rather than assumed to be ``1/4``.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    test_sizes = np.asarray(test_sizes, dtype=np.float64).reshape(-1)
    train_sizes = np.asarray(train_sizes, dtype=np.float64).reshape(-1)
    if test_sizes.shape != scores.shape or train_sizes.shape != scores.shape:
        raise ValueError("test_sizes and train_sizes must align with scores")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    m = int(scores.size)

    def _degenerate(status: str, *, estimate: float, variance: float, ratio: float) -> NadeauBengioResult:
        return NadeauBengioResult(
            status=status, n_scores=m, estimate=estimate, variance=variance,
            test_train_ratio=ratio, variance_factor=float("nan"), corrected_se=float("nan"),
            degrees_of_freedom=max(m - 1, 0), confidence_level=float(confidence_level),
            ci_lower=None, ci_upper=None, t_statistic=None, p_value_two_sided=None,
            naive_se=None, naive_ci_lower=None, naive_ci_upper=None,
        )

    if m < 2:
        estimate = float(scores[0]) if m == 1 else float("nan")
        return _degenerate(NB_STATUS_INSUFFICIENT, estimate=estimate, variance=float("nan"), ratio=float("nan"))
    if not np.all(np.isfinite(scores)):
        return _degenerate(NB_STATUS_NONFINITE, estimate=float("nan"), variance=float("nan"), ratio=float("nan"))
    if (np.any(train_sizes <= 0.0) or np.any(test_sizes <= 0.0)
            or not np.all(np.isfinite(train_sizes)) or not np.all(np.isfinite(test_sizes))):
        raise ValueError("every split must have positive finite train and test sizes")

    from scipy.stats import t as student_t

    estimate = float(scores.mean())
    variance = float(scores.var(ddof=1))
    ratio = float(np.mean(test_sizes / train_sizes))
    if variance == 0.0:
        # A zero-width interval here would assert certainty the design does not
        # supply, so the test and interval are reported as undefined instead.
        return _degenerate(NB_STATUS_DEGENERATE, estimate=estimate, variance=variance, ratio=ratio)
    factor = 1.0 / m + ratio
    corrected_se = float(math.sqrt(factor * variance))
    df = m - 1
    critical = float(student_t.ppf(0.5 + float(confidence_level) / 2.0, df))
    t_statistic = estimate / corrected_se
    p_value = float(2.0 * student_t.sf(abs(t_statistic), df))
    naive_se = float(math.sqrt(variance / m))
    return NadeauBengioResult(
        status=NB_STATUS_OK, n_scores=m, estimate=estimate, variance=variance,
        test_train_ratio=ratio, variance_factor=float(factor), corrected_se=corrected_se,
        degrees_of_freedom=df, confidence_level=float(confidence_level),
        ci_lower=estimate - critical * corrected_se, ci_upper=estimate + critical * corrected_se,
        t_statistic=float(t_statistic), p_value_two_sided=p_value,
        naive_se=naive_se, naive_ci_lower=estimate - critical * naive_se,
        naive_ci_upper=estimate + critical * naive_se,
    )
