"""Evaluation and uncertainty for the Coffee primary's one fixed holdout.

This module deliberately has no training, split construction, file I/O, or
cross-validation concepts.  It receives *one already separated test
partition* and frozen policies, then implements the estimand in
``experiment/coffee-primary/EXPERIMENT_SPEC.md``:

``mean_s(max_k D[pooled,k,s] - max_k D[multigroup,k,s])``.

The welfare comparator is one fixed lottery per test group, hence it is fit
once and shared by every method and learner seed.  The signed exploitability
enemy is instead fit once for each (seed, method, group).  Both use the trusted
KL tilt and are noncontextual: a returned ``q`` is an action lottery, never an
array indexed by a respondent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from empirical_testing.reporting.metrics import (
    RATIO_DEFINED,
    RATIO_EMPTY,
    RATIO_POSITIVE_OVER_ZERO,
    batched_kl_tilt,
    margin_matrix,
    ratio_with_status,
)


PRIMARY_METHODS = ("pooled", "multigroup")
CONDITIONAL_INTERVAL_LABEL = (
    "descriptive percentile interval conditional on frozen policies, split, "
    "groups, and design"
)


@dataclass(frozen=True)
class SingleHoldoutEvaluation:
    """All evaluation quantities for one test partition.

    Array axes are ``(seed, method, group, action)`` unless documented
    otherwise.  ``welfare_q`` has no seed/method axis by design: the same
    group-specific empirical attainable-welfare comparator is used for every
    policy.  Empty groups retain a complete grid with NaNs and ``empty``
    statuses so they can never silently become a favorable maximum.
    """

    seed_ids: tuple[int, ...]
    method_names: tuple[str, ...]
    group_names: tuple[str, ...]
    group_counts: np.ndarray  # (K,)
    welfare_q: np.ndarray  # (K, A), shared across methods and seeds
    welfare_optimum: np.ndarray  # (K,)
    welfare_q_status: np.ndarray  # (K,)
    welfare: np.ndarray  # (S, M, K)
    distortion: np.ndarray  # (S, M, K)
    distortion_status: np.ndarray  # (S, M, K)
    enemy_q: np.ndarray  # (S, M, K, A), method/seed/group specific
    signed_exploitability: np.ndarray  # (S, M, K)
    enemy_q_status: np.ndarray  # (S, M, K)
    worst_group_distortion: np.ndarray  # (S, M)
    worst_group_index: np.ndarray  # (S, M), -1 when no defined ratio
    per_seed_pooled_minus_multigroup: np.ndarray  # (S,)
    primary_estimate: float
    paired_seed_mcse: float


@dataclass(frozen=True)
class SingleHoldoutBootstrap:
    """Conditional respondent-bootstrap result for the frozen policies.

    Every row of ``multiplicities`` sums to the complete test-partition size.
    The same row is supplied to every group, seed, and method in its replicate;
    each replicate calls :func:`evaluate_single_holdout`, thereby re-fitting q
    and re-selecting method-specific worst groups before the paired reduction.
    """

    point_estimate: float
    paired_seed_mcse: float
    replicate_statistics: np.ndarray
    multiplicities: np.ndarray
    percentile_interval: tuple[float, float]
    confidence_level: float
    interval_label: str = CONDITIONAL_INTERVAL_LABEL


def _as_policy_cube(
    policies: Mapping[str, np.ndarray] | np.ndarray,
    *,
    n_test: int,
    n_actions: int,
    method_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Normalize frozen policies to ``(S, M, N, A)`` without fitting them."""

    if isinstance(policies, Mapping):
        names = tuple(policies) if method_names is None else tuple(method_names)
        if not names or set(names) != set(policies):
            raise ValueError("method_names must name exactly the policy mapping keys")
        values: list[np.ndarray] = []
        n_seeds: int | None = None
        for name in names:
            value = np.asarray(policies[name], dtype=np.float64)
            if value.ndim == 2:
                value = value[None, :, :]
            if value.ndim != 3 or value.shape[1:] != (n_test, n_actions):
                raise ValueError(
                    f"policy {name!r} must have shape (n_seeds, n_test, n_actions) "
                    "or (n_test, n_actions)"
                )
            if n_seeds is None:
                n_seeds = value.shape[0]
            elif value.shape[0] != n_seeds:
                raise ValueError("every method must have the same number of learner seeds")
            values.append(value)
        cube = np.stack(values, axis=1)
    else:
        cube = np.asarray(policies, dtype=np.float64)
        if cube.ndim != 4 or cube.shape[2:] != (n_test, n_actions):
            raise ValueError("policy array must have shape (n_seeds, n_methods, n_test, n_actions)")
        if method_names is None:
            raise ValueError("method_names is required when policies is an array")
        names = tuple(method_names)
        if len(names) != cube.shape[1] or len(set(names)) != len(names):
            raise ValueError("method_names must be unique and match the policy method axis")
    if not np.all(np.isfinite(cube)) or np.any(cube < 0.0):
        raise ValueError("policies must be finite nonnegative lotteries")
    if not np.allclose(cube.sum(axis=-1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("every frozen policy row must sum to one")
    return cube, names


def _validate_inputs(
    utilities: np.ndarray,
    group_membership: np.ndarray,
    reference: np.ndarray | None,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(utilities, dtype=np.float64)
    groups = np.asarray(group_membership)
    if u.ndim != 2 or u.shape[0] == 0 or u.shape[1] < 2 or not np.all(np.isfinite(u)):
        raise ValueError("utilities must be a finite nonempty (n_test, n_actions) array")
    if groups.ndim != 2 or groups.shape[0] != u.shape[0] or groups.shape[1] == 0:
        raise ValueError("group_membership must have shape (n_test, n_groups)")
    if groups.dtype != np.bool_:
        raise ValueError("group_membership must be boolean; missing membership is False")
    ref = np.full(u.shape[1], 1.0 / u.shape[1]) if reference is None else np.asarray(reference, dtype=np.float64)
    if ref.shape != (u.shape[1],) or not np.all(np.isfinite(ref)) or np.any(ref <= 0.0):
        raise ValueError("reference must be a finite, full-support action lottery")
    if not np.isclose(ref.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("reference must sum to one")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and nonnegative")
    return u, groups, ref


def _solve_q_batch(
    values: np.ndarray, reference: np.ndarray, tau: float, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batch trusted fixed-lottery solves, failing closed on any bad row."""

    result = batched_kl_tilt(values, reference, tau, tie_scale=counts.astype(np.float64))
    if not np.all(result.converged) or not np.all(result.feasible):
        raise RuntimeError("trusted KL optimizer did not return feasible converged comparators")
    return result.solution, result.status.astype(object)


def _worst_defined(distortion: np.ndarray, statuses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Method-specific max over groups for each seed, excluding undefined cells."""

    # A positive benchmark over zero achieved welfare is a valid ``+inf``
    # distortion, not an undefined cell.  It must bind a worst-group maximum.
    defined = (statuses == RATIO_DEFINED) | (statuses == RATIO_POSITIVE_OVER_ZERO)
    candidates = np.where(defined, distortion, -np.inf)
    indexes = candidates.argmax(axis=-1)
    maxima = candidates.max(axis=-1)
    empty = maxima == -np.inf
    maxima[empty] = np.nan
    indexes = indexes.astype(np.int64)
    indexes[empty] = -1
    return maxima, indexes


def evaluate_single_holdout(
    utilities: np.ndarray,
    group_membership: np.ndarray,
    policies: Mapping[str, np.ndarray] | np.ndarray,
    *,
    group_names: Sequence[str],
    method_names: Sequence[str] | None = None,
    seed_ids: Sequence[int] | None = None,
    reference: np.ndarray | None = None,
    tau: float = 0.5,
    beta: float = float("inf"),
    multiplicities: np.ndarray | None = None,
) -> SingleHoldoutEvaluation:
    """Score frozen policies on exactly one held-out partition.

    ``multiplicities`` is an optional whole-respondent resample vector.  It is
    intentionally a single vector rather than a per-method/group input; this
    makes a bootstrap caller structurally share its draw across all paired and
    overlapping quantities.  It never changes a policy.
    """

    u, groups, ref = _validate_inputs(utilities, group_membership, reference, float(tau))
    cube, names = _as_policy_cube(policies, n_test=u.shape[0], n_actions=u.shape[1], method_names=method_names)
    group_names = tuple(group_names)
    if len(group_names) != groups.shape[1] or len(set(group_names)) != len(group_names):
        raise ValueError("group_names must be unique and match the group axis")
    if not set(PRIMARY_METHODS).issubset(names):
        raise ValueError("methods must include the primary paired arms 'pooled' and 'multigroup'")
    ids = tuple(range(cube.shape[0])) if seed_ids is None else tuple(int(x) for x in seed_ids)
    if len(ids) != cube.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("seed_ids must be unique and match the policy seed axis")
    weights = np.ones(u.shape[0], dtype=np.int64) if multiplicities is None else np.asarray(multiplicities, dtype=np.int64)
    if weights.shape != (u.shape[0],) or np.any(weights < 0):
        raise ValueError("multiplicities must be a nonnegative vector over the test respondents")
    if int(weights.sum()) == 0:
        raise ValueError("multiplicities must select at least one test respondent")
    if not np.isfinite(beta) and not np.isposinf(beta):
        raise ValueError("beta must be positive or +infinity")
    if not np.isposinf(beta) and beta <= 0.0:
        raise ValueError("beta must be positive or +infinity")

    s_count, m_count, _, a_count = cube.shape
    k_count = groups.shape[1]
    # This (K, N) matrix is the one shared bootstrap multiplicity vector
    # intersected with every group.  It is subsequently used for *all* methods
    # and seeds, so overlap and pairing are retained structurally.
    group_weights = groups.T.astype(np.int64) * weights[None, :]
    counts = group_weights.sum(axis=1, dtype=np.int64)
    active_groups = counts > 0
    welfare_q = np.full((k_count, a_count), np.nan)
    welfare_optimum = np.full(k_count, np.nan)
    welfare_q_status = np.full(k_count, RATIO_EMPTY, dtype=object)
    welfare_values = np.full((s_count, m_count, k_count), np.nan)
    distortion = np.full_like(welfare_values, np.nan)
    distortion_status = np.full(welfare_values.shape, RATIO_EMPTY, dtype=object)
    enemy_q = np.full((s_count, m_count, k_count, a_count), np.nan)
    enemy_q_status = np.full(welfare_values.shape, RATIO_EMPTY, dtype=object)
    signed_exploitability = np.full_like(welfare_values, np.nan)

    group_mean_u = np.full((k_count, a_count), np.nan)
    if np.any(active_groups):
        group_mean_u[active_groups] = (
            np.einsum("kn,na->ka", group_weights[active_groups], u, optimize=True)
            / counts[active_groups, None]
        )
        q, status = _solve_q_batch(
            group_mean_u[active_groups], ref, float(tau), counts[active_groups]
        )
        welfare_q[active_groups] = q
        welfare_q_status[active_groups] = status
        welfare_optimum[active_groups] = np.einsum(
            "ka,ka->k", q, group_mean_u[active_groups], optimize=True
        )

    # The competitor/enemy q uses the frozen deployed policies and is therefore
    # intentionally distinct for each seed, method, and group.  Do all policy
    # scoring and all active enemy solves in two dense batches; this turns the
    # bootstrap's former O(S*M*K) Python solver calls into one solver call.
    margins = margin_matrix(u, float(beta))
    respondent_welfare = np.einsum("smna,na->smn", cube, u, optimize=True)
    welfare_values = np.einsum("kn,smn->smk", group_weights, respondent_welfare, optimize=True)
    welfare_values = np.divide(
        welfare_values, counts[None, None, :],
        out=np.full_like(welfare_values, np.nan), where=active_groups[None, None, :],
    )
    respondent_deployed_margin = np.einsum("smna,nab->smnb", cube, margins, optimize=True)
    deployed_mean = np.einsum("kn,smnb->smkb", group_weights, respondent_deployed_margin, optimize=True)
    deployed_mean = np.divide(
        deployed_mean, counts[None, None, :, None],
        out=np.full_like(deployed_mean, np.nan), where=active_groups[None, None, :, None],
    )
    active_enemy = np.broadcast_to(active_groups, (s_count, m_count, k_count))
    flat_active = active_enemy.reshape(-1)
    if np.any(flat_active):
        flat_values = (-deployed_mean).reshape(-1, a_count)
        flat_counts = np.broadcast_to(counts, (s_count, m_count, k_count)).reshape(-1)
        q, status = _solve_q_batch(flat_values[flat_active], ref, float(tau), flat_counts[flat_active])
        enemy_q.reshape(-1, a_count)[flat_active] = q
        enemy_q_status.reshape(-1)[flat_active] = status
    signed_exploitability = np.einsum(
        "smka,smka->smk", enemy_q, -deployed_mean, optimize=True
    )
    # Preserve the established ratio-status vocabulary rather than encoding an
    # ad-hoc vectorized variant of its zero/empty/nonfinite branches.
    for s, m, k in np.ndindex(s_count, m_count, k_count):
        ratio, status = ratio_with_status(
            welfare_optimum[k], welfare_values[s, m, k], count=int(counts[k])
        )
        distortion[s, m, k] = ratio
        distortion_status[s, m, k] = status

    worst, worst_index = _worst_defined(distortion, distortion_status)
    pooled_index, multigroup_index = names.index("pooled"), names.index("multigroup")
    contrast = worst[:, pooled_index] - worst[:, multigroup_index]
    if not np.all(np.isfinite(contrast)):
        raise RuntimeError("the primary contrast is undefined because a primary arm has no defined group distortion")
    primary_estimate = float(contrast.mean())
    mcse = float(contrast.std(ddof=1) / np.sqrt(contrast.size)) if contrast.size > 1 else float("nan")
    return SingleHoldoutEvaluation(
        seed_ids=ids,
        method_names=names,
        group_names=group_names,
        group_counts=counts,
        welfare_q=welfare_q,
        welfare_optimum=welfare_optimum,
        welfare_q_status=welfare_q_status,
        welfare=welfare_values,
        distortion=distortion,
        distortion_status=distortion_status,
        enemy_q=enemy_q,
        signed_exploitability=signed_exploitability,
        enemy_q_status=enemy_q_status,
        worst_group_distortion=worst,
        worst_group_index=worst_index,
        per_seed_pooled_minus_multigroup=contrast,
        primary_estimate=primary_estimate,
        paired_seed_mcse=mcse,
    )


def bootstrap_single_holdout(
    utilities: np.ndarray,
    group_membership: np.ndarray,
    policies: Mapping[str, np.ndarray] | np.ndarray,
    *,
    group_names: Sequence[str],
    replicates: int = 2000,
    bootstrap_seed: int = 8675309,
    confidence_level: float = 0.95,
    method_names: Sequence[str] | None = None,
    seed_ids: Sequence[int] | None = None,
    reference: np.ndarray | None = None,
    tau: float = 0.5,
    beta: float = float("inf"),
) -> SingleHoldoutBootstrap:
    """Bootstrap the conditional test-partition uncertainty of the primary statistic.

    Policies are passed unchanged to every replicate.  The one joint draw is
    converted to respondent multiplicities and sent to the evaluator, which
    re-fits both types of q and repeats the binding-group selection inside the
    replicate as required by the frozen protocol.
    """

    if int(replicates) != replicates or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    u, groups, _ = _validate_inputs(utilities, group_membership, reference, float(tau))
    # Validate and freeze the policy representation once.  The evaluator will
    # validate the cube again only through inexpensive shape/probability checks.
    cube, names = _as_policy_cube(policies, n_test=u.shape[0], n_actions=u.shape[1], method_names=method_names)
    point = evaluate_single_holdout(
        u, groups, cube, group_names=group_names, method_names=names, seed_ids=seed_ids,
        reference=reference, tau=tau, beta=beta,
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    multiplicities = np.empty((int(replicates), u.shape[0]), dtype=np.int64)
    statistics = np.empty(int(replicates), dtype=np.float64)
    for b in range(int(replicates)):
        draw = rng.integers(0, u.shape[0], size=u.shape[0])
        multiplicities[b] = np.bincount(draw, minlength=u.shape[0])
        replicate = evaluate_single_holdout(
            u, groups, cube, group_names=group_names, method_names=names, seed_ids=point.seed_ids,
            reference=reference, tau=tau, beta=beta, multiplicities=multiplicities[b],
        )
        statistics[b] = replicate.primary_estimate
    alpha = (1.0 - float(confidence_level)) / 2.0
    interval = tuple(float(x) for x in np.quantile(statistics, [alpha, 1.0 - alpha]))
    return SingleHoldoutBootstrap(
        point_estimate=point.primary_estimate,
        paired_seed_mcse=point.paired_seed_mcse,
        replicate_statistics=statistics,
        multiplicities=multiplicities,
        percentile_interval=interval,
        confidence_level=float(confidence_level),
    )


def build_single_holdout_figures_tables(
    *,
    evaluation_dir: Path,
    bootstrap_dir: Path,
    output_dir: Path,
    config: object,
) -> None:
    """Write labelled point-estimate tables and figures for the primary.

    This deliberately consumes only completed evaluation/bootstrap artifacts;
    it cannot train a policy, alter a split, or select another analysis.  A
    figures intentionally contain point estimates only: the test bootstrap
    interval belongs to the predeclared *primary contrast*, not to each plotted
    bar, and no unrequested bar-level uncertainty is invented here.
    """

    evaluation_path = Path(evaluation_dir) / "single_holdout_evaluation.npz"
    bootstrap_path = Path(bootstrap_dir) / "single_holdout_bootstrap.npz"
    if not evaluation_path.is_file() or not bootstrap_path.is_file():
        raise RuntimeError("figures/tables requires completed single-holdout evaluation and bootstrap artifacts")
    with np.load(evaluation_path, allow_pickle=False) as evaluation, np.load(bootstrap_path, allow_pickle=False) as bootstrap:
        contrast = np.asarray(evaluation["per_seed_pooled_minus_multigroup"], dtype=np.float64)
        distortion = np.asarray(evaluation["distortion"], dtype=np.float64)
        worst_group_distortion = np.asarray(evaluation["worst_group_distortion"], dtype=np.float64)
        replicates = np.asarray(bootstrap["replicate_statistics"], dtype=np.float64)
    if (
        contrast.ndim != 1 or contrast.size == 0 or replicates.ndim != 1 or replicates.size == 0
        or distortion.ndim != 3 or worst_group_distortion.ndim != 2
    ):
        raise RuntimeError("single-holdout result artifacts have invalid primary-statistic arrays")
    methods = tuple(str(x) for x in getattr(config, "arms", ())) + tuple(
        str(x) for x in getattr(config, "secondary_arms", ())
    )
    groups = tuple(str(x) for x in getattr(config, "groups", ()))
    if distortion.shape[1:] != (len(methods), len(groups)):
        raise RuntimeError("evaluation artifact axes do not match the frozen primary methods/groups")
    if worst_group_distortion.shape != distortion.shape[:2]:
        raise RuntimeError("worst-group artifact axes do not match per-group distortions")
    point = float(contrast.mean())
    mcse = float(contrast.std(ddof=1) / np.sqrt(contrast.size)) if contrast.size > 1 else float("nan")
    confidence_level = float(getattr(config, "bootstrap_confidence_level", 0.95))
    if not 0.0 < confidence_level < 1.0:
        raise RuntimeError("primary config has an invalid bootstrap confidence level")
    alpha = (1.0 - confidence_level) / 2.0
    interval = [float(x) for x in np.quantile(replicates, [alpha, 1.0 - alpha])]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    table = target / "primary_summary.csv"
    table.write_text(
        "statistic,estimate,lower,upper,learner_seed_mcse,label\n"
        f"paired_pooled_minus_multigroup_worst_group_empirical_distortion,{point:.17g},"
        f"{interval[0]:.17g},{interval[1]:.17g},{mcse:.17g},"
        f'"{CONDITIONAL_INTERVAL_LABEL}"\n',
        encoding="utf-8",
    )
    (target / "primary_estimand.json").write_text(
        json.dumps(
            {
                "estimand": "mean_seed(max_group(D_pooled) - max_group(D_multigroup))",
                "positive_favors": "multigroup",
                "bootstrap_confidence_level": confidence_level,
                "bootstrap_percentile_interval": interval,
                "learner_seed_mcse": mcse,
                "bootstrap_label": CONDITIONAL_INTERVAL_LABEL,
                "study_status": "exploratory_mechanically_held_out_rerun",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    # Keep plotting import local so evaluation/bootstrapping remains usable in
    # compute-only environments. Force the noninteractive backend before
    # creating a figure rather than relying on process defaults.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
        # Do not convert undefined or infinite ratios into made-up finite bars.
        finite = np.where(np.isfinite(values), values, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(finite, axis=axis)

    title_suffix = "Held-out rerun — point estimates only"
    method_labels = [name.replace("_", " ") for name in methods]
    colors = ["#4477AA", "#228833", "#CCBB44", "#AA3377"]

    worst_mean = _finite_mean(worst_group_distortion, axis=0)
    fig, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    bars = axis.bar(np.arange(len(methods)), worst_mean, color=colors[:len(methods)])
    axis.set_xticks(np.arange(len(methods)), method_labels)
    axis.set_ylabel("Worst-group empirical distortion")
    axis.set_title(title_suffix)
    axis.text(
        0.5, -0.22, "No bar-level intervals; the primary paired-bootstrap interval is reported in primary_summary.csv.",
        ha="center", va="top", transform=axis.transAxes, fontsize=8,
    )
    for bar, value in zip(bars, worst_mean):
        if np.isfinite(value):
            axis.annotate(f"{value:.3g}", (bar.get_x() + bar.get_width() / 2, value),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(target / f"worst_group_distortion.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Mean across learner seeds is a descriptive point estimate for each arm
    # and frozen group; it does not replace the prescribed max-then-paired
    # primary reduction represented above.
    group_mean = _finite_mean(distortion, axis=0)  # (M, K)
    fig, axis = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    x = np.arange(len(groups))
    width = 0.8 / len(methods)
    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2.0) * width
        axis.bar(x + offset, group_mean[method_index], width, label=method.replace("_", " "),
                 color=colors[method_index % len(colors)])
    axis.set_xticks(x, groups, rotation=24, ha="right")
    axis.set_ylabel("Empirical distortion (mean over learner seeds)")
    axis.set_title(title_suffix + "; secondary per-group display")
    axis.legend(title="Frozen policy arm")
    axis.text(
        0.5, -0.32, CONDITIONAL_INTERVAL_LABEL + "; no per-group intervals shown.",
        ha="center", va="top", transform=axis.transAxes, fontsize=8,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(target / f"per_group_distortion.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
