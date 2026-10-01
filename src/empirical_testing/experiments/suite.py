"""Comparison identities, grids, and execution policy.

The four methods are multigroup GLHF, pooled GLHF, passive NLHF, and uniform.
The beta grid fixes learner/evaluation radii at 0.5. The radius sweep varies
learner radii over 0.05, 0.2, 0.35, 0.5, 0.75, and 1.0, holding evaluation at 0.5.
Shared radius-0.5 cells are reused across the two comparisons.

reuse-required validates completed beta artifacts; fresh trains the beta grid
from ratings. Historical schema identifiers remain in saved metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from empirical_testing.methods import ALLOWED_METHODS, FITTED_METHODS, canonical, require_allowed

#: Configuration/output-schema version stamped on everything suite writes.
SCHEMA_VERSION = "v10"

#: The NLHF solver baseline suite production fits use.  Imported rather than
#: re-spelled so the two cannot drift.
from empirical_testing.algorithms.nlhf import (  # noqa: E402
    NLHF_BASELINE_VERSION_REFERENCE_START as NLHF_BASELINE,
)

PASSIVE = "passive"
ACTIVE = "active"

#: The beta grid, in reporting order.  ``inf`` is the deterministic
#: score-order oracle and is kept off the finite log axis in every figure.
BETA_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, math.inf)

#: The betas at which the active protocol is also run.
ACTIVE_BETAS: tuple[float, ...] = (4.0, math.inf)

#: The learner radii of the active tau ablation.
TAU_GRID: tuple[float, ...] = (0.05, 0.2, 0.35, 0.5, 0.75, 1.0)

#: The evaluation radius, fixed at every point of every condition.  A moving
#: evaluation ball would change what the y-axis means.
EVALUATION_TAU = 0.5

#: The learner radius the beta condition is run at, and the one point of the
#: tau grid that is shared with it.
BETA_CONDITION_TAU = 0.5

#: The label parameter the tau ablation is run at.
TAU_CONDITION_BETA = math.inf

#: The four methods, in reporting order.
METHODS: tuple[str, ...] = ("multigroup", "pooled", "pooled_nlhf", "uniform")

#: Display names for tables and figures.
DISPLAY_NAMES: Mapping[str, str] = dict(ALLOWED_METHODS)

#: How each method is trained, stated explicitly wherever it is reported.
#: NLHF is passive-trained in *every* condition, the active ones included.
TRAINING_PROTOCOL: Mapping[str, str] = {
    "multigroup": "span-adaptive learner, per-group memberships",
    "pooled": "span-adaptive learner, pooled memberships",
    "pooled_nlhf": "passive-trained batch solver (never fitted on an active log)",
    "uniform": "not fitted",
}


class ExecutionPolicyError(RuntimeError):
    """The requested execution policy cannot be satisfied."""


class ReuseBlockedError(ExecutionPolicyError):
    """Imported beta artifacts are missing or inconsistent.

    Raised instead of retraining: a blocker is reported, never repaired by
    silently refitting the beta grid.
    """


# --------------------------------------------------------------- suite identity

@dataclass(frozen=True)
class Suite:
    """The suite identities and destinations for one dataset."""

    dataset: str
    suite_id: str
    beta_protocol_id: str
    tau_protocol_id: str
    beta_directory: str
    tau_directory: str
    suite_directory: str
    #: Relative to the standalone ``EmpiricalTesting/`` root.
    figure_directory: str
    #: The historical source design this condition sits on.  Recorded as a
    #: historical id; never rewritten to suggest a different design or fresh
    #: training.
    source_design_id: str
    source_beta_run_id: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "suite_id": self.suite_id,
            "beta_protocol_id": self.beta_protocol_id,
            "tau_protocol_id": self.tau_protocol_id,
            "beta_directory": self.beta_directory,
            "tau_directory": self.tau_directory,
            "suite_directory": self.suite_directory,
            "figure_directory": self.figure_directory,
            "historical_ids": {
                "source_design_id": self.source_design_id,
                "source_beta_run_id": self.source_beta_run_id,
            },
            "nlhf_baseline": NLHF_BASELINE,
            "methods": list(METHODS),
            "method_training_protocol": dict(TRAINING_PROTOCOL),
            "evaluation_tau": EVALUATION_TAU,
        }


SUITES: Mapping[str, Suite] = {
    "coffee": Suite(
        dataset="coffee",
        suite_id="coffee-comparison-suite-v10",
        beta_protocol_id="coffee-protocol-beta-v10",
        tau_protocol_id="coffee-active-tau-v10",
        beta_directory="results/coffee/beta-source",
        tau_directory="results/coffee/radius-rerun",
        suite_directory="results/coffee/comparison-rerun",
        figure_directory="figures/coffee",
        source_design_id="coffee-repeated-cv-v4-10x10",
        source_beta_run_id="coffee-crossfit-v4-protocol-beta-sweep",
    ),
    "persona": Suite(
        dataset="persona",
        suite_id="persona-comparison-suite-v10",
        beta_protocol_id="persona-protocol-beta-v10",
        tau_protocol_id="persona-active-tau-v10",
        beta_directory="results/persona/beta-source",
        tau_directory="results/persona/radius-rerun",
        suite_directory="results/persona/comparison-rerun",
        figure_directory="figures/persona",
        source_design_id="persona-repeated-cv-v2-fixed-six-answer",
        source_beta_run_id="persona-fixed-v8-protocol-beta-sweep",
    ),
}

MODES = ("comparison-suite", "protocol-beta", "learning-curves", "panel-permutations")
EXECUTION_POLICIES = ("reuse-required", "fresh")


# ------------------------------------------------------------------- the grids

@dataclass(frozen=True)
class Condition:
    """One reportable cell: a condition plus the method trained inside it."""

    condition: str          # "beta" or "tau"
    #: How this method was **trained**.  For NLHF this is always ``passive``,
    #: in every condition, because NLHF is not an active learner in this study.
    protocol: str
    beta: float
    learner_tau: float
    method: str
    #: How this cell is obtained: fitted here, imported, or shared with another
    #: cell that already provides exactly this result.
    provenance: str
    #: When ``provenance == "shared"``, the cell whose result this one *is*.
    shared_with: str | None = None
    #: Where this result is *displayed*, when that differs from where it was
    #: trained.  The passive NLHF benchmark is shown beside the active
    #: protocol; that is a position in a table, not a claim about training, and
    #: keeping the two apart is what stops a reused passive result from being
    #: labelled an active fit.
    displayed_beside: str | None = None

    @property
    def training_protocol_name(self) -> str:
        return self.protocol

    @property
    def label(self) -> str:
        position = self.displayed_beside or self.protocol
        if self.condition == "beta":
            return f"beta:{position}@{_beta_text(self.beta)}:{self.method}"
        return f"tau:{position}@{self.learner_tau:g}:{self.method}"

    @property
    def is_fit(self) -> bool:
        return self.provenance == "fit"

    def as_mapping(self) -> dict[str, Any]:
        payload = {
            "cell": self.label, "condition": self.condition,
            # ``protocol`` is the *training* protocol throughout, so a reader
            # never has to infer it from where the number is printed.
            "protocol": self.protocol,
            "sampling_protocol_trained": self.protocol,
            "beta": _beta_text(self.beta), "learner_tau": self.learner_tau,
            "evaluation_tau": EVALUATION_TAU, "method": self.method,
            "display_name": DISPLAY_NAMES[self.method],
            "training_protocol": TRAINING_PROTOCOL[self.method],
            "provenance": self.provenance,
        }
        if self.shared_with:
            payload["shared_with"] = self.shared_with
        if self.displayed_beside:
            payload["displayed_beside_protocol"] = self.displayed_beside
            payload["display_note"] = (
                f"the {self.protocol}-trained benchmark, shown beside the "
                f"{self.displayed_beside} protocol; not an "
                f"{self.displayed_beside} fit"
            )
        return payload


def _beta_text(beta: float) -> str:
    return "inf" if math.isinf(beta) else f"{beta:g}"


def beta_condition_cells() -> tuple[Condition, ...]:
    """Every reportable cell of the beta/protocol condition.

    Passive GLHF at every beta; active GLHF only at 4 and infinity; NLHF
    passive-trained at every beta; uniform everywhere and never fitted.
    """

    cells: list[Condition] = []
    for beta in BETA_GRID:
        for method in ("multigroup", "pooled"):
            cells.append(Condition("beta", PASSIVE, beta, BETA_CONDITION_TAU, method,
                                   provenance="imported"))
        # NLHF is the passive-trained benchmark, fitted once per beta and shown
        # beside the active protocol as well -- counted once, here.
        cells.append(Condition("beta", PASSIVE, beta, BETA_CONDITION_TAU, "pooled_nlhf",
                               provenance="fit"))
        cells.append(Condition("beta", PASSIVE, beta, BETA_CONDITION_TAU, "uniform",
                               provenance="evaluate_only"))
        if beta in ACTIVE_BETAS:
            for method in ("multigroup", "pooled"):
                cells.append(Condition("beta", ACTIVE, beta, BETA_CONDITION_TAU, method,
                                       provenance="imported"))
            # Beside the active protocol the NLHF column is the *same* passive
            # fit, not a second one and not an "active NLHF".
            cells.append(Condition(
                "beta", PASSIVE, beta, BETA_CONDITION_TAU, "pooled_nlhf",
                provenance="shared",
                shared_with=f"beta:{PASSIVE}@{_beta_text(beta)}:pooled_nlhf",
                displayed_beside=ACTIVE,
            ))
            cells.append(Condition("beta", ACTIVE, beta, BETA_CONDITION_TAU, "uniform",
                                   provenance="evaluate_only"))
    return tuple(cells)


def tau_condition_cells() -> tuple[Condition, ...]:
    """Every reportable cell of the active tau ablation.

    At ``tau = 0.5`` both fitted families are shared rather than refitted: the
    GLHF arms are the completed active/``beta = inf`` beta cell, and the NLHF
    benchmark is the passive ``beta = inf`` fit.
    """

    shared_beta = _beta_text(TAU_CONDITION_BETA)
    cells: list[Condition] = []
    for tau in TAU_GRID:
        shared = tau == BETA_CONDITION_TAU
        for method in ("multigroup", "pooled"):
            cells.append(Condition(
                "tau", ACTIVE, TAU_CONDITION_BETA, tau, method,
                provenance="shared" if shared else "fit",
                shared_with=f"beta:{ACTIVE}@{shared_beta}:{method}" if shared else None,
            ))
        cells.append(Condition(
            "tau", PASSIVE, TAU_CONDITION_BETA, tau, "pooled_nlhf",
            provenance="shared" if shared else "fit",
            shared_with=f"beta:{PASSIVE}@{shared_beta}:pooled_nlhf" if shared else None,
            displayed_beside=ACTIVE,
        ))
        cells.append(Condition("tau", PASSIVE, TAU_CONDITION_BETA, tau, "uniform",
                               provenance="evaluate_only", displayed_beside=ACTIVE))
    return tuple(cells)


def all_cells() -> tuple[Condition, ...]:
    return beta_condition_cells() + tau_condition_cells()


# ----------------------------------------------------------- the work inventory

@dataclass
class FitPlan:
    """How many fits each condition needs, and which need none."""

    n_splits: int
    n_seeds: int
    cells: tuple[Condition, ...] = field(default_factory=all_cells)

    @property
    def units(self) -> int:
        return self.n_splits * self.n_seeds

    def fits_by_kind(self) -> dict[str, int]:
        """New fits this plan schedules, split by what is being fitted."""

        counts = {"glhf_beta": 0, "glhf_tau": 0, "nlhf_beta": 0, "nlhf_tau": 0}
        for cell in self.cells:
            if not cell.is_fit:
                continue
            family = "nlhf" if cell.method == "pooled_nlhf" else "glhf"
            counts[f"{family}_{cell.condition}"] += self.units
        return counts

    def total_new_fits(self) -> int:
        return sum(self.fits_by_kind().values())

    def as_mapping(self) -> dict[str, Any]:
        by_kind = self.fits_by_kind()
        return {
            "version": SCHEMA_VERSION,
            "splits": self.n_splits, "seeds": self.n_seeds, "units": self.units,
            "cells": [cell.as_mapping() for cell in self.cells],
            "new_fits_by_kind": by_kind,
            "new_fits_total": sum(by_kind.values()),
            "beta_glhf_fits": by_kind["glhf_beta"],
            "shared_cells": [c.label for c in self.cells if c.provenance == "shared"],
            "imported_cells": [c.label for c in self.cells if c.provenance == "imported"],
            "evaluate_only_cells": [c.label for c in self.cells if c.provenance == "evaluate_only"],
            "methods": list(METHODS),
            "note": (
                "Shared cells are displayed in a second condition but counted once. "
                "No NLHF benchmark is counted twice for appearing beside both protocols."
            ),
        }


def production_plan(n_splits: int = 100, n_seeds: int = 50) -> FitPlan:
    return FitPlan(n_splits=n_splits, n_seeds=n_seeds)


# --------------------------------------------------------------- the reuse gate

class BetaFitGuard:
    """Makes a beta GLHF fit impossible under ``reuse-required``.

    The requirement is not "we intend not to retrain the beta grid" but "no
    beta GLHF fitting call happens, anchor validation included".  An intention
    is not testable and costs an hour when it is wrong; this is both.

    Use as a context manager around any execution that must not retrain the
    beta grid.  It counts what it is told about and raises on the first
    violation, so the failure names the cell rather than appearing as an
    unexplained runtime.
    """

    def __init__(self, policy: str) -> None:
        if policy not in EXECUTION_POLICIES:
            raise ExecutionPolicyError(
                f"unknown execution policy {policy!r}; expected one of {EXECUTION_POLICIES}"
            )
        self.policy = policy
        self.calls: list[str] = []

    @property
    def forbids_beta_glhf(self) -> bool:
        return self.policy == "reuse-required"

    def record_fit(self, cell: Condition | str, *, method: str | None = None,
                   condition: str | None = None) -> None:
        """Declare a fit about to happen.  Raises if the policy forbids it."""

        if isinstance(cell, Condition):
            label, method, condition = cell.label, cell.method, cell.condition
        else:
            label = str(cell)
        resolved = canonical(method) if method else None
        if resolved == "uniform":
            raise ExecutionPolicyError(
                f"{label}: the uniform reference is never fitted; it is evaluated only"
            )
        if resolved is not None and resolved not in FITTED_METHODS:
            raise ExecutionPolicyError(f"{label}: {method!r} is not a fittable method")
        if self.forbids_beta_glhf and condition == "beta" and resolved != "pooled_nlhf":
            raise ExecutionPolicyError(
                f"{label}: execution policy 'reuse-required' forbids beta GLHF fitting. "
                f"The completed beta GLHF results are imported, never retrained -- "
                f"including to validate an anchor, which is done by comparing saved "
                f"arrays directly."
            )
        self.calls.append(label)

    def __enter__(self) -> "BetaFitGuard":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def resolve_beta_source(project_root: Path, suite: Suite) -> Path:
    """Locate the single canonical imported beta run."""

    return Path(project_root) / suite.beta_directory / "source"


def check_reuse_available(project_root: Path, suite: Suite) -> dict[str, Any]:
    """Whether the validated beta artifacts suite needs are actually present.

    Reports a blocker; never repairs one by retraining.  The caller under
    ``reuse-required`` must treat a negative result as fatal.
    """

    from empirical_testing.experiments.provenance import (
        IMPORT_MANIFEST, SOURCE_RUNS, verify_source,
    )

    imported = resolve_beta_source(project_root, suite)
    beta_dir = imported.parent
    manifest = beta_dir / IMPORT_MANIFEST
    status: dict[str, Any] = {
        "suite": suite.suite_id, "beta_directory": suite.beta_directory,
        "imported_source": str(imported), "manifest": str(manifest),
        "available": False, "blockers": [],
    }
    if not manifest.is_file():
        status["blockers"].append(f"no suite import manifest at {manifest}")
    if not imported.is_dir():
        status["blockers"].append(f"no imported beta run at {imported}")
    if status["blockers"]:
        return status

    report = verify_source(imported, SOURCE_RUNS[suite.dataset])
    status["validation"] = report.as_json()
    if not report.ok:
        status["blockers"].extend(report.failures)
        return status
    status["available"] = True
    return status


def require_reuse(project_root: Path, suite: Suite) -> dict[str, Any]:
    """``check_reuse_available``, but a blocker is fatal."""

    status = check_reuse_available(project_root, suite)
    if not status["available"]:
        raise ReuseBlockedError(
            f"{suite.dataset.title()}: execution policy 'reuse-required' needs validated "
            f"completed beta GLHF artifacts, and they are missing or inconsistent:\n  - "
            + "\n  - ".join(str(b) for b in status["blockers"])
            + "\nThis is reported as a blocker; the beta grid is not retrained to repair it."
        )
    return status


# ------------------------------------------------------------------- dry running

def dry_run_inventory(suite: Suite, *, policy: str = "reuse-required",
                      n_splits: int = 100, n_seeds: int = 50) -> dict[str, Any]:
    """What a run would do, without doing any of it.

    ``fresh`` includes training the beta GLHF grid from the input ratings.
    """

    if policy not in EXECUTION_POLICIES:
        raise ExecutionPolicyError(
            f"unknown execution policy {policy!r}; expected one of {EXECUTION_POLICIES}"
        )
    cells = all_cells()
    if policy == "fresh":
        cells = tuple(replace(cell, provenance="fit") if cell.provenance == "imported"
                      else cell for cell in cells)
    plan = FitPlan(n_splits=n_splits, n_seeds=n_seeds, cells=cells)
    cells = plan.cells
    methods = sorted({c.method for c in cells})
    require_allowed(methods, where=f"{suite.dataset.title()} dry-run inventory")
    inventory = {
        **suite.as_mapping(),
        "execution": policy,
        "unsupported_execution": False,
        "plan": plan.as_mapping(),
        "method_inventory": methods,
        "conditions": {
            "beta": {
                "learner_tau": BETA_CONDITION_TAU, "evaluation_tau": EVALUATION_TAU,
                "betas": [_beta_text(b) for b in BETA_GRID],
                "active_betas": [_beta_text(b) for b in ACTIVE_BETAS],
            },
            "tau": {
                "betas": [_beta_text(TAU_CONDITION_BETA)],
                "learner_taus": list(TAU_GRID), "evaluation_tau": EVALUATION_TAU,
                "glhf_sampling": ACTIVE, "nlhf_training": PASSIVE,
            },
        },
    }
    if policy == "reuse-required":
        assert inventory["plan"]["beta_glhf_fits"] == 0, (
            "reuse-required must schedule zero beta GLHF fits"
        )
    return inventory
