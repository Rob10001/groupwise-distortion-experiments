"""Shared Coffee data loading, cross-validation, fitting, and reduction.

A discovery reservation is excluded from all analysis folds. Each split fits
pooled and multigroup policies on training respondents and evaluates them on
the held-out fold. Inference uses the mean of split-level worst-group contrasts
and a Nadeau-Bengio corrected interval.

Route records retain the identifiers required to read saved configurations.
The public comparison workflow is implemented in experiments.run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import traceback
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from empirical_testing.core import make_uniform_passive_log, subset_dense_dataset
from empirical_testing.core.holdout import sorted_id_sha256
from empirical_testing.core.repeated_cv import (
    NB_STATUS_OK,
    DiscoveryReservation,
    RepeatedFoldPlan,
    discovery_reservation,
    nadeau_bengio,
    repeated_fold_plan,
    split_learner_seed,
)
from empirical_testing.datasets import load_dense_dataset
from empirical_testing.parallel import ordered_map
from empirical_testing.reporting.metrics import RATIO_DEFINED

ARMS = ("pooled", "multigroup")
MEMBERSHIP_ALIGNMENT = "log_respondent_indices"
#: Under active self-play there is no pre-built log to index through, so the
#: learner reads one training row per round in training-row order and the
#: multigroup contexts are that same order.  Recorded per seed so a reader can
#: tell which alignment a saved unit used without inferring it from the route.
ROW_ORDER_ALIGNMENT = "training_row_order"

#: The two feedback protocols a route may declare in its ``protocol`` key.
#: ``PASSIVE_PROTOCOL`` is the established one: one uniformly sampled unordered
#: pair per training respondent, drawn once per (split, learner seed) and
#: replayed identically into both arms with ``protocol="logged"``.
PASSIVE_PROTOCOL = "passive_uniform_unordered_pair"
#: ``ACTIVE_PROTOCOL`` hands pair selection to the learner's own self-play
#: protocol (``protocol="active"``): each round the learner sees its current
#: policy, draws a pair from it, and reads that training respondent's label.
#: The two arms therefore *cannot* share a log -- each arm's comparisons depend
#: on its own policy -- so a route declaring this protocol records
#: ``shared_log_across_arms: False`` and saves each arm's drawn log.
ACTIVE_PROTOCOL = "active_span_adaptive_self_play"
FEEDBACK_PROTOCOLS = (PASSIVE_PROTOCOL, ACTIVE_PROTOCOL)


def parse_label_beta(text: str) -> float:
    """The numeric Bradley-Terry label parameter a route's ``beta`` string names.

    ``"inf"`` is the deterministic score-order oracle (the strict ordering, with
    a seeded fair coin on exact ties).  A finite value draws each label from
    ``sigmoid(beta * (u_a - u_b))``, the manuscript's Bradley-Terry assumption.
    The string is what the configuration declares and what the digest covers;
    this is the only place it becomes a number.
    """

    if str(text) == "inf":
        return math.inf
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"beta must be 'inf' or a finite positive number, not {text!r}")
    return value


@dataclass(frozen=True)
class CrossfitRoute:
    """Everything that distinguishes one frozen repeated-CV route from another.

    A route owns its protocol id, its implementation id, its group family and
    the prose its summary carries.  Two routes therefore produce different
    configuration digests and different saved-split identities, so a unit
    written by one is never reused by the other.  Adding a route cannot change
    an existing one: the tuples below are the whole definition.
    """

    protocol_id: str
    implementation_id: str
    #: Historical route label and output-namespace stem; not a public CLI mode.
    #: It is recorded in ``run_info.json`` and is what stops a run of one route
    #: from resuming into a directory written by another.
    mode: str
    config_name: str
    study_status: str
    groups: tuple[str, ...]
    group_definitions: tuple[tuple[str, str | None, tuple[str, ...]], ...]
    title: str
    #: Short design note repeated in the run summary.
    standing_note: tuple[str, ...]
    #: Free-text notes folded into the design manifest's ``groups`` block.
    group_notes: Mapping[str, str]
    #: Whether the reduction also writes the per-group paired-gap table,
    #: its Holm-adjusted decisions and its forest figure.
    per_group_gaps: bool = False
    #: The feedback protocol this route's configuration must declare, and which
    #: :func:`fit_split_policies` dispatches on.  Declared per route so the four
    #: passive routes keep their guard exactly as it was: a configuration whose
    #: ``protocol`` key does not equal its route's value is refused.
    feedback_protocol: str = PASSIVE_PROTOCOL
    #: The Bradley-Terry label parameter this route's configuration must declare,
    #: as the string the config file carries and the digest covers.
    label_beta: str = "inf"

    @property
    def beta_value(self) -> float:
        """The numeric label parameter, ``+inf`` for the score-order oracle."""

        return parse_label_beta(self.label_beta)

    @property
    def active_feedback(self) -> bool:
        """Whether the learner chooses its own pairs (and so owns its own log)."""

        return self.feedback_protocol == ACTIVE_PROTOCOL


#: ``coffee-repeated-cv-v1``: the original 10 x 5-fold route.  Frozen.  Its
#: implementation id is bumped only when a change would make an already-saved
#: split incomparable with a freshly computed one; a saved split whose manifest
#: names a different value is recomputed rather than reused.
ROUTE_V1 = CrossfitRoute(
    protocol_id="coffee-repeated-cv-v1",
    implementation_id="coffee-crossfit-2026-09-16.1",
    mode="crossfit",
    config_name="coffee_crossfit.json",
    study_status="new_exploratory_analysis_with_mechanically_separated_discovery_sample",
    groups=(
        "universal", "spend:<20", "age:45+", "gender:female",
        "race:hispanic-latino", "work:from-home",
    ),
    group_definitions=(
        ("universal", None, ("ALL",)),
        ("spend:<20", "total_spend", ("<$20",)),
        ("age:45+", "age", ("45-54 years old", "55-64 years old", ">65 years old")),
        ("gender:female", "gender", ("Female",)),
        ("race:hispanic-latino", "ethnicity_race", ("Hispanic/Latino",)),
        ("work:from-home", "wfh", ("I primarily work from home",)),
    ),
    title="Coffee repeated cross-validation",
    standing_note=(
        "A reserved 10% discovery sample separates group-selection rows from analysis rows.",
        "The group family was supplied by the analyst, not selected by a search in this run.",
    ),
    group_notes={
        # Reproduced verbatim from the accepted v1 run's design manifest.
        "work_from_home_interpretation":
            "the survey's explicit 'I primarily work from home' answer only; "
            "'I do a mix of both' is excluded",
    },
)

#: ``coffee-repeated-cv-v2-10x10`` changes both the fold design (10 x 10 instead
#: of 10 x 5) and the subgroup family relative to v1, so the estimates are not
#: directly comparable. The frozen data, discovery draw and learner are shared.
ROUTE_V2_10X10 = CrossfitRoute(
    protocol_id="coffee-repeated-cv-v2-10x10",
    implementation_id="coffee-crossfit-10x10-2026-09-16.1",
    mode="crossfit-10x10",
    config_name="coffee_crossfit_10x10.json",
    study_status="post_hoc_exploratory_sensitivity_analysis_not_confirmatory",
    groups=(
        "universal", "age:45+", "gender:female",
        "most-willing:$10-$15", "expertise:8+", "wfh:in-person",
    ),
    group_definitions=(
        ("universal", None, ("ALL",)),
        ("age:45+", "age", ("45-54 years old", "55-64 years old", ">65 years old")),
        ("gender:female", "gender", ("Female",)),
        ("most-willing:$10-$15", "most_willing", ("$10-$15",)),
        ("expertise:8+", "expertise", ("8", "9", "10")),
        ("wfh:in-person", "wfh", ("I primarily work in person",)),
    ),
    title="Coffee repeated cross-validation, 10x10 sensitivity analysis",
    standing_note=(
        "This route changes both the fold design (10 x 10 rather than 10 x 5) and the subgroup family",
        "relative to `coffee-repeated-cv-v1`; a difference cannot be attributed to either change alone.",
        "The discovery reservation, frozen data and learner are shared with v1. The group family was",
        "supplied by the analyst, not selected by a search in this run.",
    ),
    group_notes={
        "imputation":
            "group membership is read from the source column and is never imputed, "
            "defaulted or back-filled from any other field",
        "wfh_in_person_interpretation":
            "`wfh:in-person` is the survey's explicit 'I primarily work in person' answer only; both "
            "'I primarily work from home' and the hybrid 'I do a mix of both' are excluded",
        "expertise_interpretation":
            "`expertise:8+` is a self-rated expertise of 8, 9 or 10; the column is coerced to a number once, so "
            "'8', 8 and '8.0' agree, and a missing or nonnumeric rating is a nonmember",
        "most_willing_interpretation":
            "`most-willing:$10-$15` is the survey's single '$10-$15' willingness-to-pay answer; "
            "adjacent categories such as "
            "'$8-$10' and '$15-$20' are nonmembers",
    },
    per_group_gaps=True,
)

#: ``coffee-repeated-cv-v3-10x10`` keeps the v2 fold design (10 x 10) and swaps
#: two subgroup members: ``most-willing:$10-$15`` becomes ``spend:$40-$60`` and
#: the top-of-scale ``expertise:8+`` becomes the bottom-of-scale
#: ``expertise:1-3``. The family was supplied by the analyst. The 90% analysis
#: sample, frozen data, discovery draw and learner are shared with v1 and v2.
ROUTE_V3_10X10 = CrossfitRoute(
    protocol_id="coffee-repeated-cv-v3-10x10",
    implementation_id="coffee-crossfit-10x10-v3-2026-09-16.1",
    mode="crossfit-10x10-v3",
    config_name="coffee_crossfit_10x10_v3.json",
    study_status="post_hoc_exploratory_sensitivity_analysis_not_confirmatory",
    groups=(
        "universal", "age:45+", "gender:female",
        "expertise:1-3", "spend:$40-$60", "wfh:in-person",
    ),
    group_definitions=(
        ("universal", None, ("ALL",)),
        ("age:45+", "age", ("45-54 years old", "55-64 years old", ">65 years old")),
        ("gender:female", "gender", ("Female",)),
        ("expertise:1-3", "expertise", ("1", "2", "3")),
        ("spend:$40-$60", "total_spend", ("$40-$60",)),
        ("wfh:in-person", "wfh", ("I primarily work in person",)),
    ),
    title="Coffee repeated cross-validation, 10x10 sensitivity analysis (low-expertise family)",
    standing_note=(
        "This route shares the v2 fold design (10 x 10) and the same 90% analysis sample.",
        "Intervals condition on the supplied group family.",
        "It differs from v2 in two group members: `spend:$40-$60` in place of",
        "`most-willing:$10-$15`, and the bottom-of-scale `expertise:1-3` in place of `expertise:8+`. The",
        "discovery reservation, the frozen data and every learner detail are the same as v1 and v2; the",
        "group family was supplied by the analyst, not selected by a search in this run.",
    ),
    group_notes={
        "imputation":
            "group membership is read from the source column and is never imputed, "
            "defaulted or back-filled from any other field",
        "wfh_in_person_interpretation":
            "`wfh:in-person` is the survey's explicit 'I primarily work in person' answer only; both "
            "'I primarily work from home' and the hybrid 'I do a mix of both' are excluded",
        "expertise_interpretation":
            "`expertise:1-3` is a self-rated expertise of 1, 2 or 3; the column is coerced to a number "
            "once, so '2', 2 and '2.0' agree, and a missing or nonnumeric rating is a nonmember rather "
            "than a low rating",
        "total_spend_interpretation":
            "`spend:$40-$60` is the survey's single '$40-$60' total_spend answer; adjacent categories "
            "such as '$20-$40' and '$60-$80' are nonmembers",
    },
    per_group_gaps=True,
)

#: ``coffee-repeated-cv-v4-10x10`` uses its own discovery reservation.
#: It is not a variant of v2 or v3: it rests on its **own** frozen discovery
#: reservation, drawn with seed ``1667498008`` and recorded in
#: ``experiment/coffee-crossfit/DISCOVERY-10X10-V4.md``, and on its **own**
#: cross-validation seed generated from OS entropy. The subgroup family was
#: supplied by the analyst from the discovery sample's demographic screen.
#: The fold design (10 x 10), the
#: learner, the seeds and the corrected inference are the established ones.
#:
#: This route reports its own numbers only.  It deliberately makes no
#: substantive comparison with v1, v2 or v3: its analysis complement is a
#: different 90% of the same survey, so a difference between routes would
#: confound the reservation, the group family and ordinary sampling variation.
ROUTE_V4_10X10 = CrossfitRoute(
    protocol_id="coffee-repeated-cv-v4-10x10",
    implementation_id="coffee-crossfit-10x10-v4-2026-09-16.1",
    mode="crossfit-10x10-v4",
    config_name="coffee_crossfit_10x10_v4.json",
    study_status="post_hoc_exploratory_sensitivity_analysis_not_confirmatory",
    groups=(
        "universal", "age:45+", "gender:female",
        "expertise:1-4", "children:reported-1+", "work:from-home",
    ),
    group_definitions=(
        ("universal", None, ("ALL",)),
        ("age:45+", "age", ("45-54 years old", "55-64 years old", ">65 years old")),
        ("gender:female", "gender", ("Female",)),
        ("expertise:1-4", "expertise", ("1", "2", "3", "4")),
        ("children:reported-1+", "number_children", ("1", "2", "3", "More than 3")),
        ("work:from-home", "wfh", ("I primarily work from home",)),
    ),
    title="Coffee repeated cross-validation, 10x10 v4 (seed-1667498008 reservation)",
    standing_note=(
        "This route has its own frozen 10% discovery reservation (seed 1667498008) and cross-validation",
        "seed. The subgroup family was supplied by the analyst using only the discovery sample; no",
        "discovery search, ranking or selection gate runs here. Discovery rows are excluded from analysis.",
        "A different reservation means a different 90% analysis sample. Comparisons with earlier Coffee",
        "routes would confound the reservation, group family and ordinary sampling variation.",
    ),
    group_notes={
        "imputation":
            "group membership is read from the source column and is never imputed, "
            "defaulted or back-filled from any other field",
        "expertise_interpretation":
            "`expertise:1-4` is a self-rated expertise of 1, 2, 3 or 4; the column is coerced to a "
            "number once, so '4', 4 and '4.0' agree, and a missing or nonnumeric rating is a "
            "nonmember rather than a low rating",
        "children_interpretation":
            "`children:reported-1+` is a reported `number_children` of 1, 2, 3 or 'More than 3'; a "
            "missing child count is a nonmember and is never read as zero",
        "work_from_home_interpretation":
            "`work:from-home` is the survey's explicit 'I primarily work from home' answer only; "
            "the hybrid 'I do a mix of both' and the in-person answer are excluded",
    },
    per_group_gaps=True,
)

#: ``coffee-repeated-cv-v5-10x10-active-beta3``: the ``coffee-repeated-cv-v4``
#: design with **two** feedback changes and nothing else.
#:
#: 1. **Active sampling.** Pairs come from the learner's own self-play protocol
#:    (``protocol="active"``) instead of a uniform passive log replayed with
#:    ``protocol="logged"``.  The learner sees its current policy each round and
#:    draws one comparison per training respondent, in training-row order.
#: 2. **Finite Bradley-Terry labels, ``beta = 3``.** Labels are drawn from
#:    ``Pr(a > b) = sigmoid(3 * (u_a - u_b))`` at generation time, instead of the
#:    deterministic score-order oracle ``beta = inf``.
#:
#: Everything else is v4's, byte for byte: the same discovery reservation (seed
#: ``1667498008``), the same cross-validation seed ``3183923047``, the same
#: 10 x 10 fold plan, the same analyst-supplied group family, the same 50
#: learner seeds, ``tau = 0.5``, a uniform reference, all-pre-round averaging
#: and the same corrected inference.
#:
#: Because active sampling makes each arm's comparisons a function of that arm's
#: own policy, the two arms **cannot** share one log here.  Common random
#: numbers therefore align only the label coin, not the pairs.  The route
#: records ``shared_log_across_arms: False`` and saves both arms' drawn logs.
#:
#: Comparison policy: these numbers may be reported beside
#: ``coffee-repeated-cv-v4-10x10``'s as a *same design, different feedback
#: protocol* sensitivity, and never against v1, v2 or v3, whose analysis
#: complement is a different 90% of the survey.
ROUTE_V5_10X10_ACTIVE_BETA3 = CrossfitRoute(
    protocol_id="coffee-repeated-cv-v5-10x10-active-beta3",
    implementation_id="coffee-crossfit-10x10-v5-active-beta3-2026-09-20.1",
    mode="crossfit-10x10-v5-active-beta3",
    config_name="coffee_crossfit_10x10_v5_active_beta3.json",
    study_status="post_hoc_exploratory_sensitivity_analysis_not_confirmatory",
    groups=ROUTE_V4_10X10.groups,
    group_definitions=ROUTE_V4_10X10.group_definitions,
    title="Coffee repeated cross-validation, 10x10 v5 (active sampling, beta = 3)",
    standing_note=(
        "This route uses the `coffee-repeated-cv-v4-10x10` design with two changes to the feedback protocol:",
        "pairs are chosen by the learner's own active self-play instead of a uniform passive log, and",
        "labels are drawn from a finite Bradley-Terry model with `beta = 3` instead of the",
        "deterministic score-order oracle. The discovery reservation, the cross-validation seed, the",
        "fold plan, the group family, the learner seeds, `tau` and the inference are v4's, unchanged.",
        "Under active sampling the two arms **cannot** share one comparison log: each arm's pairs",
        "depend on its own policy, so common random numbers align the label coin but not the pairs,",
        "and the pairing between arms is weaker than in the passive routes. These numbers may be read",
        "beside v4's as a same-design, different-feedback-protocol sensitivity. They are **not**",
        "compared with `coffee-repeated-cv-v1`, `-v2-10x10` or `-v3-10x10`, whose analysis complement",
        "is a different 90% of the same survey.",
    ),
    group_notes=dict(ROUTE_V4_10X10.group_notes),
    per_group_gaps=True,
    feedback_protocol=ACTIVE_PROTOCOL,
    label_beta="3",
)

ROUTES: dict[str, CrossfitRoute] = {
    route.protocol_id: route
    for route in (
        ROUTE_V1, ROUTE_V2_10X10, ROUTE_V3_10X10, ROUTE_V4_10X10,
        ROUTE_V5_10X10_ACTIVE_BETA3,
    )
}

# Backwards-compatible v1 aliases.  Existing imports, tests and saved manifests
# keep meaning exactly what they meant before the routes were introduced.
PROTOCOL_ID = ROUTE_V1.protocol_id
IMPLEMENTATION_ID = ROUTE_V1.implementation_id
CROSSFIT_GROUPS = ROUTE_V1.groups
CROSSFIT_GROUP_DEFINITIONS = ROUTE_V1.group_definitions
CROSSFIT_GROUPS_10X10 = ROUTE_V2_10X10.groups
CROSSFIT_GROUPS_10X10_V3 = ROUTE_V3_10X10.groups
CROSSFIT_GROUPS_10X10_V4 = ROUTE_V4_10X10.groups
#: The v5 active/beta=3 route reuses the v4 family unchanged; named separately
#: so a reader does not have to know that to follow the route.
CROSSFIT_GROUPS_10X10_V5_ACTIVE_BETA3 = ROUTE_V5_10X10_ACTIVE_BETA3.groups
CONFIG_KEYS = frozenset({
    "protocol_id", "study_status", "dataset", "data_path", "split", "discovery_seed",
    "discovery_fraction", "discovery_rounding", "discovery_draw", "discovery_row_order",
    "cv_seed", "folds", "repeats", "confidence_level", "group_selection", "groups",
    "group_definitions", "algorithm", "arms", "learner_seeds", "protocol",
    "comparisons_per_respondent", "reference", "tau", "output_averaging", "tie_rule",
    "beta", "estimand", "inference",
})
#: Keys a configuration *may* carry.  They are not required, so the four frozen
#: configs stay complete and valid exactly as written, and an absent optional
#: key contributes nothing to :meth:`CoffeeCrossfitConfig.as_mapping` -- so no
#: existing route's configuration digest moves.
OPTIONAL_CONFIG_KEYS = frozenset({"changes_from_source_route"})
ESTIMAND_NAME = "mean_split_of_paired_pooled_minus_multigroup_worst_group_empirical_distortion"
ESTIMAND_FORMULA = "mean_over_splits( mean_seed( max_group(D_pooled) - max_group(D_multigroup) ) )"


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class CoffeeCrossfitConfig:
    """Validated configuration for the repeated-CV route.

    Each route validates its own frozen design and configuration digest.
    """

    route: CrossfitRoute
    dataset: str
    data_path: str
    discovery_seed: int
    discovery_fraction: float
    cv_seed: int
    folds: int
    repeats: int
    confidence_level: float
    groups: tuple[str, ...]
    group_definitions: tuple[tuple[str, str | None, tuple[str, ...]], ...]
    algorithm: str
    arms: tuple[str, ...]
    learner_seeds: tuple[int, ...]
    protocol: str
    comparisons_per_respondent: int
    reference: str
    tau: float
    output_averaging: str
    tie_rule: str
    #: Optional one-line note naming the route this one was copied from and what
    #: was changed.  Documentation only: it never changes what is computed, but
    #: it is inside the configuration digest so it cannot be edited silently.
    changes_from_source_route: str = ""

    @property
    def beta(self) -> float:
        """The numeric Bradley-Terry label parameter, owned by the route.

        Derived rather than stored: :meth:`from_mapping` has already refused a
        configuration whose ``beta`` key disagrees with ``route.label_beta``, so
        the route is the single source of truth and no configuration can carry a
        label parameter its route does not declare.
        """
        return self.route.beta_value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CoffeeCrossfitConfig":
        unknown = sorted(set(raw) - CONFIG_KEYS - OPTIONAL_CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unrecognized crossfit configuration keys: {unknown}")
        missing = sorted(CONFIG_KEYS - set(raw))
        if missing:
            raise ValueError(f"crossfit configuration is missing keys: {missing}")
        route = ROUTES.get(str(raw["protocol_id"]))
        if route is None:
            raise ValueError(
                f"unknown crossfit protocol_id {raw['protocol_id']!r}; "
                f"known routes: {sorted(ROUTES)}"
            )
        if str(raw["study_status"]) != route.study_status:
            raise ValueError(
                f"{route.protocol_id} requires study_status={route.study_status!r}"
            )
        definitions = tuple(
            (str(item["name"]), None if item.get("field") is None else str(item["field"]),
             tuple(str(value) for value in item.get("values", ())))
            for item in raw["group_definitions"]
        )
        config = cls(
            route=route,
            dataset=str(raw["dataset"]),
            data_path=str(raw["data_path"]),
            discovery_seed=int(raw["discovery_seed"]),
            discovery_fraction=float(raw["discovery_fraction"]),
            cv_seed=int(raw["cv_seed"]),
            folds=int(raw["folds"]),
            repeats=int(raw["repeats"]),
            confidence_level=float(raw["confidence_level"]),
            groups=tuple(str(item) for item in raw["groups"]),
            group_definitions=definitions,
            algorithm=str(raw["algorithm"]),
            arms=tuple(str(item) for item in raw["arms"]),
            learner_seeds=tuple(int(value) for value in raw["learner_seeds"]),
            protocol=str(raw["protocol"]),
            comparisons_per_respondent=int(raw["comparisons_per_respondent"]),
            reference=str(raw["reference"]),
            tau=float(raw["tau"]),
            output_averaging=str(raw["output_averaging"]),
            tie_rule=str(raw["tie_rule"]),
            changes_from_source_route=str(raw.get("changes_from_source_route", "")),
        )
        if str(raw["split"]) != "repeated_k_fold_cross_validation":
            raise ValueError("crossfit requires split='repeated_k_fold_cross_validation'")
        if str(raw["discovery_rounding"]) != "ceiling":
            raise ValueError("the discovery sample size rounds up; discovery_rounding must be 'ceiling'")
        if str(raw["group_selection"]) != "user_specified_not_automatically_discovered":
            raise ValueError("the group family is user-specified; no discovery search is performed")
        # Per route, so the four passive routes reject a finite beta exactly as
        # they did before the active route existed.
        if str(raw["beta"]) != route.label_beta:
            raise ValueError(
                f"{route.protocol_id} declares beta={route.label_beta!r}"
                + (" (deterministic score-order feedback)" if route.label_beta == "inf"
                   else " (finite Bradley-Terry labels)")
                + f", not {str(raw['beta'])!r}"
            )
        return config.validate()

    def validate(self) -> "CoffeeCrossfitConfig":
        if self.beta <= 0 or (not np.isfinite(self.beta) and not np.isposinf(self.beta)):
            raise ValueError("beta must be positive or +infinity")
        if self.dataset != "coffee":
            raise ValueError("crossfit requires dataset='coffee'")
        if self.groups != tuple(name for name, _, _ in self.group_definitions):
            raise ValueError("groups and group_definitions must list the same names in the same order")
        if self.groups != self.route.groups or self.group_definitions != self.route.group_definitions:
            raise ValueError(
                f"{self.route.protocol_id} groups must be exactly the specified six-group family "
                f"{self.route.groups} with the declared source columns and values"
            )
        if self.arms != ARMS:
            raise ValueError("crossfit trains only the matched pooled and multigroup arms")
        if self.algorithm != "groupwise_self_play_span_adaptive":
            raise ValueError("crossfit requires the span-adaptive learner")
        if self.route.feedback_protocol not in FEEDBACK_PROTOCOLS:
            raise ValueError(
                f"{self.route.protocol_id} declares an unknown feedback protocol "
                f"{self.route.feedback_protocol!r}; known={list(FEEDBACK_PROTOCOLS)}"
            )
        if self.protocol != self.route.feedback_protocol or self.comparisons_per_respondent != 1:
            raise ValueError(
                f"{self.route.protocol_id} requires one "
                f"{self.route.feedback_protocol} comparison per training respondent"
            )
        # Rejected here rather than at fit time: the numeric value has to exist
        # before any policy is trained.
        parse_label_beta(self.route.label_beta)
        if self.reference != "uniform" or self.output_averaging != "all_pre_round_policy_maps":
            raise ValueError("crossfit requires a uniform reference and all pre-round policy averaging")
        if self.tie_rule != "fair_seeded_coin" or not np.isclose(self.tau, 0.5):
            raise ValueError("crossfit requires fair seeded ties and tau=0.5")
        if self.folds < 2 or self.repeats < 1:
            raise ValueError("crossfit requires at least two folds and one repeat")
        if not 0.0 < self.discovery_fraction < 1.0:
            raise ValueError("discovery_fraction must lie strictly between zero and one")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not self.learner_seeds or len(set(self.learner_seeds)) != len(self.learner_seeds):
            raise ValueError("learner_seeds must be a nonempty list of distinct integers")
        return self

    @property
    def n_splits(self) -> int:
        return self.repeats * self.folds

    def as_mapping(self) -> dict[str, Any]:
        return {
            "protocol_id": self.route.protocol_id,
            "study_status": self.route.study_status,
            "dataset": self.dataset, "data_path": self.data_path,
            "split": "repeated_k_fold_cross_validation",
            "discovery_seed": self.discovery_seed, "discovery_fraction": self.discovery_fraction,
            "discovery_rounding": "ceiling",
            "discovery_draw": "numpy.default_rng(discovery_seed).choice(n_eligible, k, replace=False)",
            "discovery_row_order": "source_csv_row_order_after_complete_profile_filter",
            "cv_seed": self.cv_seed, "folds": self.folds, "repeats": self.repeats,
            "confidence_level": self.confidence_level,
            "group_selection": "user_specified_not_automatically_discovered",
            "groups": list(self.groups),
            "group_definitions": [
                {"name": name, "field": field, "values": list(values)}
                for name, field, values in self.group_definitions
            ],
            "algorithm": self.algorithm, "arms": list(self.arms),
            "learner_seeds": list(self.learner_seeds), "protocol": self.protocol,
            "comparisons_per_respondent": self.comparisons_per_respondent,
            "reference": self.reference, "tau": self.tau,
            "output_averaging": self.output_averaging, "tie_rule": self.tie_rule,
            "beta": self.route.label_beta, "estimand": ESTIMAND_FORMULA,
            "inference": "nadeau_bengio_corrected_repeated_cv_t",
            # Emitted only when present, so the four frozen routes' digests are
            # exactly what they were.
            **({"changes_from_source_route": self.changes_from_source_route}
               if self.changes_from_source_route else {}),
        }

    def digest(self) -> str:
        payload = json.dumps(self.as_mapping(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_crossfit_config(path: Path) -> CoffeeCrossfitConfig:
    return CoffeeCrossfitConfig.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def tiny_config(config: CoffeeCrossfitConfig, *, seeds: int = 2, repeats: int = 2,
                folds: int = 2) -> CoffeeCrossfitConfig:
    """A reduced design for synthetic tests only.

    Production defaults are never changed by this helper, and no supported
    CLI command calls it: it exists so a test can drive the whole
    pipeline on a few dozen synthetic respondents.
    """

    return replace(config, learner_seeds=config.learner_seeds[:seeds], repeats=repeats, folds=folds).validate()


# ---------------------------------------------------------------------- inputs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CrossfitInputs:
    """The eligible dataset, its discovery reservation and its repeated-fold plan."""

    dataset: Any
    reservation: DiscoveryReservation
    plan: RepeatedFoldPlan
    data_path: Path
    data_sha256: str

    @property
    def analysis_ids(self) -> np.ndarray:
        return np.asarray(self.dataset.respondent_ids).astype(str)[self.reservation.analysis_indices]

    @property
    def discovery_ids(self) -> np.ndarray:
        return np.asarray(self.dataset.respondent_ids).astype(str)[self.reservation.discovery_indices]


def load_crossfit_inputs(
    config: CoffeeCrossfitConfig, *, project_root: Path, expected_data_sha256: str | None = None,
) -> CrossfitInputs:
    """Load the survey, reserve the discovery sample, and build the fold plan.

    The raw CSV is never modified: the discovery sample is a set of row indices
    held in memory and written to the run's design manifest.
    """

    data_path = Path(config.data_path)
    if not data_path.is_absolute():
        data_path = Path(project_root) / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"Coffee survey data is missing: {data_path}")
    data_sha256 = _sha256_file(data_path)
    if expected_data_sha256 is not None and data_sha256 != expected_data_sha256:
        from empirical_testing.experiments.coffee_primary import describe_data_hash_mismatch

        raise RuntimeError(describe_data_hash_mismatch(data_path, expected_data_sha256))
    dataset = load_dense_dataset(config.dataset, data_path, config.groups)
    ids = np.asarray(dataset.respondent_ids).astype(str)
    if len(set(ids)) != len(ids):
        raise RuntimeError("the eligible Coffee respondents do not have unique IDs")
    reservation = discovery_reservation(
        len(ids), fraction=config.discovery_fraction, seed=config.discovery_seed
    )
    plan = repeated_fold_plan(
        reservation.n_analysis, repeats=config.repeats, folds=config.folds, seed=config.cv_seed
    )
    return CrossfitInputs(
        dataset=dataset, reservation=reservation, plan=plan,
        data_path=data_path, data_sha256=data_sha256,
    )


def feedback_block(config: Any) -> dict[str, Any]:
    """How comparisons are chosen and labelled, for the design manifest.

    Shared by the Coffee and Persona routes, which is why it takes any config
    exposing ``route``, ``protocol``, ``comparisons_per_respondent``,
    ``reference``, ``output_averaging`` and ``tie_rule``.
    """

    route = config.route
    active = route.active_feedback
    return {
        "protocol": config.protocol,
        "pair_selection": (
            "the learner's own active self-play: each round it sees its current policy, samples the "
            "first alternative from that policy and the second uniformly from the rest, and reads "
            "that training respondent's label"
            if active else
            "one uniformly sampled unordered pair per training respondent, drawn once per "
            "(split, learner seed) and replayed identically into both arms"
        ),
        "comparisons_per_respondent": config.comparisons_per_respondent,
        "row_order": (
            "one comparison per training respondent, consumed in training-row order"
            if active else
            "the saved passive log's own permuted row order"
        ),
        "beta": route.label_beta,
        "labels": (
            "deterministic score order, with a seeded fair coin on exact utility ties"
            if route.label_beta == "inf" else
            f"drawn at generation time from Pr(a > b) = sigmoid({route.label_beta} * (u_a - u_b)), "
            "the Bradley-Terry label model"
        ),
        "shared_comparison_log_across_arms": not active,
        "pairing_between_arms": (
            "weaker than in the passive routes: both arms start from the same training seed, so the "
            "common random numbers align the label coin, but each arm's pairs are a function of its "
            "own policy and therefore differ between arms"
            if active else
            "the two arms train on the identical comparison log, so they are paired on the "
            "comparisons themselves"
        ),
        "training_membership_alignment": ROW_ORDER_ALIGNMENT if active else MEMBERSHIP_ALIGNMENT,
        "drawn_logs_saved": (
            "splits/rNN_fMM/comparisons.npz, one (n_seeds, n_train, 4) block per arm "
            "(first, second, sign, propensity)"
            if active else "not saved; the shared log is a deterministic function of the training seed"
        ),
        "held_out_utilities": (
            "never read during fitting; the test fold contributes group memberships as prediction "
            "contexts only"
        ),
        "reference": config.reference,
        "output_averaging": config.output_averaging,
        "tie_rule": config.tie_rule,
        "welfare_comparator": "refit within each test fold, shared across arms and learner seeds",
    }


def design_manifest(config: CoffeeCrossfitConfig, inputs: CrossfitInputs) -> dict[str, Any]:
    """The compact, auditable record of what was reserved and how splits were cut."""

    reservation, plan = inputs.reservation, inputs.plan
    ids = np.asarray(inputs.dataset.respondent_ids).astype(str)
    split_rows = []
    for repeat, fold in plan.splits():
        train_pos, test_pos = plan.split_positions(repeat, fold)
        train_ids = ids[reservation.analysis_indices[train_pos]]
        test_ids = ids[reservation.analysis_indices[test_pos]]
        split_rows.append({
            "repeat": int(repeat), "fold": int(fold),
            "n_train": int(train_pos.size), "n_test": int(test_pos.size),
            "train_ids_sha256": sorted_id_sha256(train_ids),
            "test_ids_sha256": sorted_id_sha256(test_ids),
        })
    group_counts = {
        name: int(np.asarray(inputs.dataset.groups)[reservation.analysis_indices, index].sum())
        for index, name in enumerate(config.groups)
    }
    return {
        "protocol_id": config.route.protocol_id,
        "implementation_id": config.route.implementation_id,
        "config_sha256": config.digest(),
        "data": {
            "path": str(inputs.data_path),
            "sha256": inputs.data_sha256,
            "eligible_rows": int(len(ids)),
            "eligibility": "complete personal-preference profiles only (unchanged)",
            "utility_map": "(rating - 1) / 4",
            "physical_file_modified": False,
        },
        "discovery": {
            "seed": reservation.seed,
            "fraction": reservation.fraction,
            "rounding": "ceiling",
            "draw": "numpy.default_rng(seed).choice(n_eligible, k, replace=False)",
            "row_order": "source CSV row order retained by CoffeeAdapter after the eligibility filter",
            "k": reservation.n_discovery,
            "n_analysis": reservation.n_analysis,
            "discovery_ids_sha256": sorted_id_sha256(ids[reservation.discovery_indices]),
            "analysis_ids_sha256": sorted_id_sha256(ids[reservation.analysis_indices]),
            "excluded_from": "every cross-validation training set and every test fold",
        },
        "groups": {
            "selection": "user-specified; no discovery ranking or search was run",
            "definitions": [
                {"name": name, "field": field, "values": list(values),
                 "analysis_members": group_counts[name]}
                for name, field, values in config.group_definitions
            ],
            "missing_values": "missing or unrecognized values mean subgroup nonmembership",
            **dict(config.route.group_notes),
        },
        "cross_validation": {
            "seed": plan.seed, "repeats": plan.repeats, "folds": plan.folds,
            "n_splits": plan.repeats * plan.folds,
            "assignment": "default_rng(SeedSequence([cv_seed, repeat])).permutation then numpy.array_split into folds",
            "training_seed_derivation": "SeedSequence([cv_seed, repeat, fold, learner_seed]).generate_state(1)",
            "splits": split_rows,
        },
        "learner": {
            "algorithm": config.algorithm,
            "tau": config.tau,
            **feedback_block(config),
        },
        "learner_seeds": list(config.learner_seeds),
        # Recorded before any policy is fitted, alongside the seeds, so the
        # design manifest names every fitted unit: split x seed x arm.
        "arms": list(config.arms),
        "policy_fits": plan.repeats * plan.folds * len(config.learner_seeds) * len(config.arms),
    }


def validate_design(config: CoffeeCrossfitConfig, inputs: CrossfitInputs) -> None:
    """Structural checks that must hold before any policy is fitted."""

    reservation, plan = inputs.reservation, inputs.plan
    ids = np.asarray(inputs.dataset.respondent_ids).astype(str)
    discovery = set(ids[reservation.discovery_indices].tolist())
    analysis = set(ids[reservation.analysis_indices].tolist())
    if discovery & analysis:
        raise RuntimeError("the discovery sample overlaps the analysis complement")
    if len(discovery) + len(analysis) != len(ids):
        raise RuntimeError("the discovery reservation does not partition the eligible respondents")
    for repeat in range(plan.repeats):
        counts = np.bincount(plan.fold_of[repeat], minlength=plan.folds)
        if counts.sum() != reservation.n_analysis or np.any(counts == 0):
            raise RuntimeError(f"repeat {repeat} does not test every analysis respondent exactly once")
        if int(counts.max() - counts.min()) > 1:
            raise RuntimeError(f"repeat {repeat} folds are not balanced to within one respondent")
    groups = np.asarray(inputs.dataset.groups)[reservation.analysis_indices]
    empty = [name for index, name in enumerate(config.groups) if groups[:, index].sum() == 0]
    if empty:
        raise RuntimeError(f"selected groups have no analysis members: {empty}")


# ------------------------------------------------------------- one split


def _check_training_only_utilities(
    utilities: np.ndarray, train_view: Any, test_view: Any, n_train: int
) -> None:
    """Refuse to hand a learner anything but this split's *training* utilities.

    The active protocol is the first path in this route that passes utilities
    into the fitter at all, so the held-out invariant is asserted explicitly
    rather than left implicit in the call site: the array must be exactly the
    training view's own block, and it must not be the test view's.  The test
    fold contributes group memberships as prediction contexts and nothing else.
    """

    expected = np.asarray(train_view.utilities, dtype=np.float64)
    if utilities.shape != expected.shape or utilities.shape[0] != n_train:
        raise AssertionError(
            f"the learner was handed utilities of shape {utilities.shape}, not this split's "
            f"{expected.shape} training block"
        )
    if not np.array_equal(utilities, expected):
        raise AssertionError("the learner was handed utilities that are not this split's training block")
    test_utilities = np.asarray(test_view.utilities, dtype=np.float64)
    if utilities.shape == test_utilities.shape and np.array_equal(utilities, test_utilities):
        raise AssertionError("held-out test-fold utilities reached the fitter")


def fit_split_policies(
    config: CoffeeCrossfitConfig, train_view: Any, test_view: Any, *, repeat: int, fold: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Fit both arms once per learner seed on this split's training respondents.

    Two feedback protocols, dispatched on ``config.route.feedback_protocol``:

    ``PASSIVE_PROTOCOL``
        the established path, unchanged.  One uniform unordered-pair log is
        built per learner seed with deterministic labels and replayed
        identically into both arms with ``protocol="logged"``, so the arms are
        paired on the comparisons themselves.  Multigroup training contexts are
        indexed through ``log.respondent_indices`` (the KI-1 fix), never taken
        in training-row order.

    ``ACTIVE_PROTOCOL``
        the learner chooses its own pairs (``protocol="active"``): each round it
        sees its current policy, draws a pair from it, and reads that training
        respondent's label from the Bradley-Terry model at the route's ``beta``.
        One comparison per training respondent, consumed in **training-row
        order** -- so the multigroup contexts are the training rows themselves
        and there is no log to align through.  The two arms therefore cannot
        share a log: each arm's comparisons are a function of that arm's own
        policy.  Both arms still start from the same ``training_seed``, so the
        common random numbers align the label coin but **not** the pairs; the
        per-seed record says ``shared_log_across_arms: False`` and both arms'
        drawn logs are returned for saving.

    Held-out utilities are never read here under either protocol: the test fold
    contributes only respondent IDs and group memberships, as prediction
    contexts.  The active path asserts that explicitly -- see
    :func:`_check_training_only_utilities`.
    """

    from empirical_testing.algorithms.registry import fit_online_learner

    n_train = int(train_view.utilities.shape[0])
    n_actions = int(train_view.utilities.shape[1])
    test_groups = np.asarray(test_view.groups)
    n_test = int(test_groups.shape[0])
    reference = np.full(n_actions, 1.0 / n_actions)
    pooled_train_groups = np.ones((n_train, 1), dtype=np.int8)
    pooled_test_groups = np.ones((n_test, 1), dtype=np.int8)
    train_groups = np.asarray(train_view.groups)
    active = config.route.active_feedback
    beta = config.route.beta_value
    stacked: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    drawn_logs: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    records: list[dict[str, Any]] = []
    for learner_seed in config.learner_seeds:
        training_seed = split_learner_seed(config.cv_seed, repeat, fold, learner_seed)
        if active:
            train_utilities = np.asarray(train_view.utilities, dtype=np.float64)
            _check_training_only_utilities(train_utilities, train_view, test_view, n_train)
            for arm, groups, prediction_groups in (
                ("pooled", pooled_train_groups, pooled_test_groups),
                ("multigroup", train_groups, test_groups),
            ):
                model = fit_online_learner(
                    config.algorithm, groups=groups, utilities=train_utilities,
                    beta=beta, reference=reference, tau=config.tau,
                    protocol="active", seed=training_seed,
                )
                if not np.array_equal(model.snapshot_rounds, np.arange(n_train, dtype=np.int64)):
                    raise AssertionError("the fitted policy is not the all-round pre-round average")
                drawn = np.asarray(model.drawn_comparisons, dtype=np.float64)
                if drawn.shape != (n_train, 4) or not np.all(np.isfinite(drawn)):
                    raise AssertionError(
                        f"the active learner recorded {drawn.shape} comparisons, not one per each of "
                        f"the {n_train} training respondents"
                    )
                if not np.all(np.isin(drawn[:, 2], (-1.0, 1.0))):
                    raise AssertionError("a drawn label is not a strict +-1 winner")
                if not np.all(drawn[:, 3] > 0.0):
                    raise AssertionError("a drawn pair has nonpositive propensity")
                drawn_logs[arm].append(drawn)
                stacked[arm].append(np.asarray(model.predict(prediction_groups), dtype=np.float64))
            records.append({
                "learner_seed": int(learner_seed),
                "training_seed": int(training_seed),
                "n_comparisons": int(n_train),
                # Each arm drew its own log from its own policy; only the label
                # coin is shared, because both arms start from this seed.
                "shared_log_across_arms": False,
                "training_membership_alignment": ROW_ORDER_ALIGNMENT,
                "feedback_protocol": config.route.feedback_protocol,
                "beta": config.route.label_beta,
            })
            continue
        log_rows = make_uniform_passive_log(
            train_view, np.arange(n_train, dtype=np.int64), seed=training_seed,
            comparisons_per_respondent=config.comparisons_per_respondent, beta=beta,
        )
        if not np.all(np.isin(log_rows.respondent_indices, np.arange(n_train))):
            raise AssertionError("the comparison log escaped this split's training view")
        aligned_train_groups = train_groups[log_rows.respondent_indices]
        for arm, groups, prediction_groups in (
            ("pooled", pooled_train_groups, pooled_test_groups),
            ("multigroup", aligned_train_groups, test_groups),
        ):
            model = fit_online_learner(
                config.algorithm, groups=groups, utilities=None, reference=reference,
                tau=config.tau, protocol="logged", seed=training_seed,
                pair_indices=log_rows.pairs, signs=log_rows.signs, propensities=log_rows.propensities,
            )
            if not np.array_equal(model.snapshot_rounds, np.arange(len(log_rows.signs), dtype=np.int64)):
                raise AssertionError("the fitted policy is not the all-round pre-round average")
            stacked[arm].append(np.asarray(model.predict(prediction_groups), dtype=np.float64))
        records.append({
            "learner_seed": int(learner_seed),
            "training_seed": int(training_seed),
            "n_comparisons": int(len(log_rows.signs)),
            "shared_log_across_arms": True,
            "training_membership_alignment": MEMBERSHIP_ALIGNMENT,
        })
    policies = {arm: np.stack(rows, axis=0) for arm, rows in stacked.items()}
    # (n_seeds, n_train, 4) per arm: first, second, sign, propensity.  Empty
    # under the passive protocol, where the one shared log is a deterministic
    # function of the training seed and is regenerated rather than stored.
    comparisons = (
        {arm: np.stack(rows, axis=0) for arm, rows in drawn_logs.items()} if active else {}
    )
    return policies, records, comparisons


def _check_split_groups(config: CoffeeCrossfitConfig, train_view: Any, test_view: Any,
                        repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    train_counts = np.asarray(train_view.groups).sum(axis=0).astype(np.int64)
    test_counts = np.asarray(test_view.groups).sum(axis=0).astype(np.int64)
    offenders = [
        f"{name} (train {int(train_counts[index])}, test {int(test_counts[index])})"
        for index, name in enumerate(config.groups)
        if train_counts[index] == 0 or test_counts[index] == 0
    ]
    if offenders:
        raise RuntimeError(
            f"split repeat={repeat} fold={fold} has empty selected groups: {offenders}. "
            "The design requires every selected group to be present in both the training set and "
            "the test fold; groups are not dropped and splits are not redrawn."
        )
    return train_counts, test_counts


def _check_distortion_complete(config: CoffeeCrossfitConfig, evaluation: Any, repeat: int, fold: int) -> None:
    status = np.asarray(evaluation.distortion_status, dtype=object)
    values = np.asarray(evaluation.distortion, dtype=np.float64)
    bad = (status != RATIO_DEFINED) | ~np.isfinite(values)
    if not np.any(bad):
        return
    offenders = []
    for s, m, k in zip(*np.nonzero(bad)):
        offenders.append(
            f"seed={evaluation.seed_ids[s]} arm={evaluation.method_names[m]} "
            f"group={config.groups[k]} status={status[s, m, k]} value={values[s, m, k]}"
        )
    raise RuntimeError(
        f"split repeat={repeat} fold={fold} has {len(offenders)} undefined or nonfinite distortion "
        f"values for the selected group family, e.g. {offenders[:5]}. The design requires a defined "
        "finite value for every selected group; values are not clipped, groups are not dropped, and "
        "the split is not discarded."
    )


def compute_split(
    config: CoffeeCrossfitConfig, inputs: CrossfitInputs, repeat: int, fold: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit, evaluate and reduce one split.

    Returns its summary, its evaluation arrays, and -- under the active
    protocol only -- each arm's drawn comparison log, which is saved beside the
    evaluation arrays so the run is auditable.  ``split.npz`` keeps exactly the
    keys :func:`_expected_array_shapes` declares, so the logs go in their own
    bundle rather than widening that one.
    """

    from empirical_testing.reporting import evaluate_single_holdout

    dataset, reservation, plan = inputs.dataset, inputs.reservation, inputs.plan
    train_pos, test_pos = plan.split_positions(repeat, fold)
    train_rows = reservation.analysis_indices[train_pos]
    test_rows = reservation.analysis_indices[test_pos]
    if np.intersect1d(train_rows, test_rows).size:
        raise AssertionError("a respondent appears in both the training set and the test fold")
    if np.intersect1d(np.concatenate((train_rows, test_rows)), reservation.discovery_indices).size:
        raise AssertionError("a reserved discovery respondent entered a cross-validation split")
    train_view = subset_dense_dataset(dataset, train_rows)
    test_view = subset_dense_dataset(dataset, test_rows)
    train_counts, test_counts = _check_split_groups(config, train_view, test_view, repeat, fold)

    started = time.perf_counter()
    policies, seed_records, comparisons = fit_split_policies(
        config, train_view, test_view, repeat=repeat, fold=fold
    )
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    # Held-out utilities are first read here, after every policy for this split
    # is fitted. The welfare comparator is refit on this test fold and shared
    # across both arms and all learner seeds, as in the single-holdout route.
    evaluation = evaluate_single_holdout(
        test_view.utilities, np.asarray(test_view.groups).astype(bool), dict(policies),
        group_names=config.groups, method_names=ARMS, seed_ids=config.learner_seeds, tau=config.tau,
    )
    evaluate_seconds = time.perf_counter() - started
    _check_distortion_complete(config, evaluation, repeat, fold)

    seed_difference = np.asarray(evaluation.per_seed_pooled_minus_multigroup, dtype=np.float64)
    if seed_difference.shape != (len(config.learner_seeds),) or not np.all(np.isfinite(seed_difference)):
        raise RuntimeError(f"split repeat={repeat} fold={fold} produced an incomplete per-seed contrast")
    ids = np.asarray(dataset.respondent_ids).astype(str)
    summary = {
        "protocol_id": config.route.protocol_id,
        "implementation_id": config.route.implementation_id,
        "config_sha256": config.digest(),
        "data_sha256": inputs.data_sha256,
        "repeat": int(repeat), "fold": int(fold),
        "n_train": int(train_rows.size), "n_test": int(test_rows.size),
        "train_ids_sha256": sorted_id_sha256(ids[train_rows]),
        "test_ids_sha256": sorted_id_sha256(ids[test_rows]),
        "discovery_ids_sha256": sorted_id_sha256(ids[reservation.discovery_indices]),
        "groups": list(config.groups),
        "learner_seeds": list(config.learner_seeds),
        "arms": list(ARMS),
        "feedback_protocol": config.route.feedback_protocol,
        "beta": config.route.label_beta,
        # False under active sampling: each arm's pairs come from its own policy.
        "shared_log_across_arms": not config.route.active_feedback,
        # d[r, f]: the learner-seed mean of the paired worst-group contrast.
        "split_score": float(seed_difference.mean()),
        "split_score_seed_sd": float(seed_difference.std(ddof=1)) if seed_difference.size > 1 else float("nan"),
        "test_outcomes_used_for_training": False,
        "seed_records": seed_records,
        "fit_seconds": round(fit_seconds, 3),
        "evaluate_seconds": round(evaluate_seconds, 3),
    }
    arrays = {
        "distortion": np.asarray(evaluation.distortion, dtype=np.float64),
        "distortion_status": np.asarray(evaluation.distortion_status).astype(str),
        "welfare": np.asarray(evaluation.welfare, dtype=np.float64),
        "welfare_optimum": np.asarray(evaluation.welfare_optimum, dtype=np.float64),
        "worst_group_distortion": np.asarray(evaluation.worst_group_distortion, dtype=np.float64),
        "worst_group_index": np.asarray(evaluation.worst_group_index, dtype=np.int64),
        "seed_difference": seed_difference,
        "train_group_counts": train_counts,
        "test_group_counts": test_counts,
    }
    if comparisons:
        summary["drawn_comparisons"] = {
            "file": "comparisons.npz",
            "columns": ["first", "second", "sign", "propensity"],
            "shape_per_arm": [len(config.learner_seeds), int(train_rows.size), 4],
            "note": (
                "one comparison per training respondent per learner seed, in training-row order, "
                "drawn by each arm's own active self-play from its own policy"
            ),
        }
    return summary, arrays, comparisons


# -------------------------------------------------------- split persistence


def split_dir(output_dir: Path, repeat: int, fold: int) -> Path:
    return Path(output_dir) / "splits" / f"r{int(repeat):02d}_f{int(fold):02d}"


_REUSE_KEYS = (
    "protocol_id", "implementation_id", "config_sha256", "data_sha256", "repeat", "fold",
    "train_ids_sha256", "test_ids_sha256", "discovery_ids_sha256", "groups", "learner_seeds", "arms",
)


#: Every array a completed split must carry, with the shape it must have, given
#: ``(n_seeds, n_arms, n_groups)``.  A saved unit missing one, or carrying one
#: of the wrong shape, is recomputed rather than trusted.
def _expected_array_shapes(n_seeds: int, n_arms: int, n_groups: int) -> dict[str, tuple[int, ...]]:
    return {
        "distortion": (n_seeds, n_arms, n_groups),
        "distortion_status": (n_seeds, n_arms, n_groups),
        "welfare": (n_seeds, n_arms, n_groups),
        # One attainable-welfare comparator per group, refit on the test fold
        # and shared across every arm and learner seed -- hence no seed axis.
        "welfare_optimum": (n_groups,),
        "worst_group_distortion": (n_seeds, n_arms),
        "worst_group_index": (n_seeds, n_arms),
        "seed_difference": (n_seeds,),
        "train_group_counts": (n_groups,),
        "test_group_counts": (n_groups,),
    }


def _validated_arrays(directory: Path, expected: Mapping[str, Any]) -> bool:
    """Load the saved NPZ and check it is a complete, usable split.

    The manifest says what a unit *claims* to be; this reads what it actually
    holds.  A truncated, partially written or stale bundle fails here and the
    split is recomputed, which is why resume cannot silently analyse a corrupt
    unit.
    """

    shapes = _expected_array_shapes(
        len(expected["learner_seeds"]), len(expected["arms"]), len(expected["groups"])
    )
    try:
        with np.load(Path(directory) / "split.npz", allow_pickle=False) as bundle:
            if set(bundle.files) != set(shapes):
                return False
            for key, shape in shapes.items():
                array = np.asarray(bundle[key])
                if array.shape != shape:
                    return False
                if key == "distortion_status":
                    if not np.all(array.astype(str) == RATIO_DEFINED):
                        return False
                elif array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                    return False
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        # A half-written or corrupt bundle is not an error to raise: it is a
        # unit that cannot be reused, so the split is simply recomputed.
        return False
    return True


def _saved_split(
    directory: Path, expected: Mapping[str, Any], *, require_comparisons: bool = False,
) -> dict[str, Any] | None:
    """Reuse a saved split only when its recorded identity matches exactly.

    Directory existence alone is never enough: the manifest must name the same
    protocol, implementation, configuration, source data, selected groups, split
    identity and seeds, *and* the saved arrays must load with the expected keys,
    shapes, finite values and defined statuses.  An active route additionally
    requires the drawn-comparison bundle its manifest promises, so a unit saved
    before the logs were written is recomputed rather than reused without them.
    """

    manifest = Path(directory) / "split.json"
    if not manifest.is_file() or not (Path(directory) / "split.npz").is_file():
        return None
    if require_comparisons and not (Path(directory) / "comparisons.npz").is_file():
        return None
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if any(recorded.get(key) != expected[key] for key in _REUSE_KEYS):
        return None
    if not _validated_arrays(directory, expected):
        return None
    return recorded


def _publish(staging: Path, destination: Path, attempts: int = 20) -> None:
    for attempt in range(attempts):
        try:
            os.replace(staging, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def run_split(
    config: CoffeeCrossfitConfig, inputs: CrossfitInputs, output_dir: Path, repeat: int, fold: int,
) -> dict[str, Any]:
    """Compute one split unless an identical one is already saved; save atomically."""

    directory = split_dir(output_dir, repeat, fold)
    ids = np.asarray(inputs.dataset.respondent_ids).astype(str)
    train_pos, test_pos = inputs.plan.split_positions(repeat, fold)
    expected = {
        "protocol_id": config.route.protocol_id, "implementation_id": config.route.implementation_id,
        "config_sha256": config.digest(), "data_sha256": inputs.data_sha256,
        "repeat": int(repeat), "fold": int(fold),
        "train_ids_sha256": sorted_id_sha256(ids[inputs.reservation.analysis_indices[train_pos]]),
        "test_ids_sha256": sorted_id_sha256(ids[inputs.reservation.analysis_indices[test_pos]]),
        "discovery_ids_sha256": sorted_id_sha256(ids[inputs.reservation.discovery_indices]),
        "groups": list(config.groups), "learner_seeds": list(config.learner_seeds), "arms": list(ARMS),
    }
    reused = _saved_split(
        directory, expected, require_comparisons=config.route.active_feedback
    )
    if reused is not None:
        reused["reused"] = True
        return reused
    summary, arrays, comparisons = compute_split(config, inputs, repeat, fold)
    staging = directory.parent / f".partial-{directory.name}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    if directory.exists():
        shutil.rmtree(directory)
    staging.mkdir()
    np.savez_compressed(staging / "split.npz", **arrays)
    if comparisons:
        # Saved before split.json, so a unit is never marked complete without
        # the logs its manifest promises.
        np.savez_compressed(staging / "comparisons.npz", **comparisons)
    # split.json is written last: it is what marks the unit complete.
    _write_json(staging / "split.json", summary)
    _publish(staging, directory)
    summary["reused"] = False
    return summary


# ------------------------------------------------- worker-process plumbing

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(config_mapping: Mapping[str, Any], project_root: str, output_dir: str,
                 expected_data_sha256: str | None) -> None:
    """Materialize the read-only inputs once per process (Windows spawns children)."""

    if _WORKER_STATE.get("output_dir") == output_dir:
        return
    config = CoffeeCrossfitConfig.from_mapping(config_mapping)
    inputs = load_crossfit_inputs(
        config, project_root=Path(project_root), expected_data_sha256=expected_data_sha256
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(config=config, inputs=inputs, output_dir=output_dir)


def _run_split_task(task: tuple[int, int]) -> dict[str, Any]:
    repeat, fold = task
    return run_split(
        _WORKER_STATE["config"], _WORKER_STATE["inputs"], Path(_WORKER_STATE["output_dir"]), repeat, fold
    )


# ------------------------------------------------------------------ reduce


def _split_arrays(output_dir: Path, repeat: int, fold: int) -> dict[str, np.ndarray]:
    with np.load(split_dir(output_dir, repeat, fold) / "split.npz", allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def reduce_splits(
    config: CoffeeCrossfitConfig, output_dir: Path, split_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the per-value tables and compute the corrected interval and test.

    Inference requires every planned split and every planned learner seed: a
    missing unit is an error, never a smaller sample quietly analysed.
    """

    import pandas as pd

    output_dir = Path(output_dir)
    reduce_dir = output_dir / "reduce"
    reduce_dir.mkdir(parents=True, exist_ok=True)
    expected_splits = {
        (repeat, fold) for repeat in range(config.repeats) for fold in range(config.folds)
    }
    seen = {(int(row["repeat"]), int(row["fold"])): row for row in split_summaries}
    missing = sorted(expected_splits - set(seen))
    if missing:
        raise RuntimeError(
            f"{len(missing)} planned splits have no result, e.g. {missing[:5]}; "
            "the corrected interval requires all planned split scores"
        )

    split_rows, seed_rows, group_rows = [], [], []
    # gap_scores[k] collects, for group k, one number per split: the
    # learner-seed mean of that group's own paired pooled-minus-multigroup
    # distortion.  It is the same reduction as the headline score with the
    # max over groups removed, so it is a per-group companion, never a
    # decomposition of the worst-group estimand.
    gap_scores: list[list[float]] = [[] for _ in config.groups]
    # level_scores[arm][group]: the same one-number-per-split reduction applied
    # to each arm's own distortion, without differencing.  Levels, unlike the
    # paired gaps, discard the within-split pairing, so their intervals cannot
    # be read as a test of the difference -- see ``per_group_levels``.
    level_scores: list[list[list[float]]] = [[[] for _ in config.groups] for _ in ARMS]
    for repeat in range(config.repeats):
        for fold in range(config.folds):
            summary = seen[(repeat, fold)]
            arrays = _split_arrays(output_dir, repeat, fold)
            seed_difference = np.asarray(arrays["seed_difference"], dtype=np.float64)
            if seed_difference.size != len(config.learner_seeds):
                raise RuntimeError(
                    f"split repeat={repeat} fold={fold} has {seed_difference.size} seed scores, "
                    f"expected {len(config.learner_seeds)}"
                )
            worst = np.asarray(arrays["worst_group_distortion"], dtype=np.float64)
            distortion = np.asarray(arrays["distortion"], dtype=np.float64)
            status = np.asarray(arrays["distortion_status"]).astype(str)
            welfare = np.asarray(arrays["welfare"], dtype=np.float64)
            test_counts = np.asarray(arrays["test_group_counts"], dtype=np.int64)
            train_counts = np.asarray(arrays["train_group_counts"], dtype=np.int64)
            # Reduction order: max over groups within each (seed, arm), then the
            # difference between arms, then the mean over learner seeds. Only
            # then does a split contribute one number to the t statistic.
            split_score = float(seed_difference.mean())
            for k in range(len(config.groups)):
                gap_scores[k].append(float((distortion[:, 0, k] - distortion[:, 1, k]).mean()))
                for m in range(len(ARMS)):
                    level_scores[m][k].append(float(distortion[:, m, k].mean()))
            if not math.isclose(split_score, float(summary["split_score"]), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"split repeat={repeat} fold={fold} disagrees with its saved split score")
            split_rows.append({
                "repeat": repeat, "fold": fold,
                "n_train": int(summary["n_train"]), "n_test": int(summary["n_test"]),
                "n_learner_seeds": int(seed_difference.size),
                "split_score": split_score,
                "seed_sd": float(seed_difference.std(ddof=1)) if seed_difference.size > 1 else float("nan"),
                "mean_worst_pooled": float(worst[:, 0].mean()),
                "mean_worst_multigroup": float(worst[:, 1].mean()),
                "test_train_ratio": float(summary["n_test"]) / float(summary["n_train"]),
            })
            for s, learner_seed in enumerate(config.learner_seeds):
                seed_rows.append({
                    "repeat": repeat, "fold": fold, "learner_seed": int(learner_seed),
                    "worst_group_distortion_pooled": float(worst[s, 0]),
                    "worst_group_distortion_multigroup": float(worst[s, 1]),
                    "worst_group_pooled": config.groups[int(arrays["worst_group_index"][s, 0])],
                    "worst_group_multigroup": config.groups[int(arrays["worst_group_index"][s, 1])],
                    "seed_difference": float(seed_difference[s]),
                })
                for m, arm in enumerate(ARMS):
                    for k, group in enumerate(config.groups):
                        group_rows.append({
                            "repeat": repeat, "fold": fold, "learner_seed": int(learner_seed),
                            "arm": arm, "group": group,
                            "distortion": float(distortion[s, m, k]),
                            "distortion_status": str(status[s, m, k]),
                            "welfare": float(welfare[s, m, k]),
                            "train_members": int(train_counts[k]),
                            "test_members": int(test_counts[k]),
                        })

    split_table = pd.DataFrame(split_rows)
    split_table.to_csv(reduce_dir / "split_scores.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(reduce_dir / "seed_scores.csv", index=False)
    pd.DataFrame(group_rows).to_csv(reduce_dir / "group_values.csv", index=False)

    result = nadeau_bengio(
        split_table["split_score"].to_numpy(), split_table["n_test"].to_numpy(),
        split_table["n_train"].to_numpy(), confidence_level=config.confidence_level,
    )
    inference = {
        "estimand": ESTIMAND_NAME,
        "formula": ESTIMAND_FORMULA,
        "positive_favors": "multigroup",
        "estimand_note": (
            "the mean of fold-specific worst-group contrasts; the max over groups and the welfare "
            "ratio are nonlinear, so this is a different estimand from a maximum or ratio taken "
            "after pooling all out-of-fold respondents"
        ),
        "method": "nadeau_bengio_corrected_repeated_cv_t",
        "method_note": (
            "the conventional corrected repeated-CV approximation, not an exact independence or "
            "coverage theorem; the correction widens the interval relative to treating the "
            "overlapping fold scores as independent, and coverage has not been established for "
            "this worst-group ratio statistic"
        ),
        "n_repeats": config.repeats, "n_folds": config.folds,
        "n_splits": int(len(split_table)), "n_learner_seeds": len(config.learner_seeds),
        "n_policy_fits": int(len(split_table) * len(config.learner_seeds) * len(ARMS)),
        "discovery_excluded": True,
        **result.as_mapping(),
    }
    _write_json(reduce_dir / "inference.json", inference)
    _write_figure(reduce_dir, split_table["split_score"].to_numpy(), inference)
    if config.route.per_group_gaps:
        gaps = per_group_gaps(config, split_table, gap_scores)
        pd.DataFrame(gaps["groups"]).to_csv(reduce_dir / "group_gaps.csv", index=False)
        _write_json(reduce_dir / "group_gaps.json", gaps)
        _write_gap_figure(reduce_dir, gaps)
        inference["per_group_gaps"] = gaps

        levels = per_group_levels(config, split_table, level_scores)
        pd.DataFrame(levels["cells"]).to_csv(reduce_dir / "group_levels.csv", index=False)
        _write_json(reduce_dir / "group_levels.json", levels)
        _write_level_figure(reduce_dir, levels, gaps)
        inference["per_group_levels"] = levels
    return inference


# ------------------------------------------------- per-group paired gaps


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, monotone and capped at one.

    Holm controls the family-wise error rate of the *decisions*.  It says
    nothing about interval widths: adjusting p-values does not turn ordinary
    marginal intervals into simultaneous ones.
    """

    values = [float(value) for value in p_values]
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def per_group_gaps(
    config: "CoffeeCrossfitConfig", split_table: Any, gap_scores: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Corrected intervals for each group's own paired gap, plus Holm decisions.

    Each group gets the same Nadeau-Bengio calculation as the headline, applied
    to that group's own split scores.  The reported intervals are ordinary
    **marginal** corrected 95% intervals: they are *not* multiplicity-adjusted
    and must not be described as Holm-adjusted intervals.  The Holm column
    adjusts the p-values only, so an interval that excludes zero while the
    Holm-adjusted p-value exceeds 0.05 is not a contradiction.
    """

    test_sizes = split_table["n_test"].to_numpy()
    train_sizes = split_table["n_train"].to_numpy()
    results = [
        nadeau_bengio(np.asarray(scores, dtype=np.float64), test_sizes, train_sizes,
                      confidence_level=config.confidence_level)
        for scores in gap_scores
    ]
    # An undefined p-value cannot be ranked; it is carried as 1.0 for the Holm
    # step-down and reported as undefined in the table.
    raw_p = [1.0 if r.p_value_two_sided is None else float(r.p_value_two_sided) for r in results]
    adjusted = holm_adjusted(raw_p)
    alpha = 1.0 - float(config.confidence_level)
    rows = []
    for index, (name, result) in enumerate(zip(config.groups, results)):
        rows.append({
            "group": name,
            "status": result.status,
            "n_splits": result.n_scores,
            "estimate": result.estimate,
            "corrected_se": result.corrected_se,
            "degrees_of_freedom": result.degrees_of_freedom,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "t_statistic": result.t_statistic,
            "p_value_two_sided": result.p_value_two_sided,
            "p_value_holm": None if result.p_value_two_sided is None else adjusted[index],
            "holm_reject_at_alpha": (
                None if result.p_value_two_sided is None else bool(adjusted[index] < alpha)
            ),
        })
    return {
        "estimand": "mean_over_splits( mean_seed( D_pooled[group] - D_multigroup[group] ) )",
        "positive_favors": "multigroup",
        "method": "nadeau_bengio_corrected_repeated_cv_t_per_group",
        "confidence_level": config.confidence_level,
        "alpha": alpha,
        "interval_type": "marginal_corrected_95_percent_not_simultaneous",
        "interval_label": (
            "ordinary marginal Nadeau-Bengio corrected 95% intervals, one group at a time; these are "
            "NOT Holm-adjusted intervals and NOT simultaneous over the group family, because no "
            "multiplicity-adjusted interval was computed"
        ),
        "multiplicity": "holm_step_down_on_p_values_only",
        "multiplicity_label": (
            f"the Holm adjustment covers the {len(config.groups)} p-value decisions only; an interval "
            "that excludes zero while its Holm-adjusted p-value exceeds alpha is not a contradiction"
        ),
        "companion_note": (
            "these per-group gaps drop the max over groups, so they are a companion to the headline "
            "worst-group estimand, not a decomposition of it"
        ),
        "groups": rows,
    }


def _plain_label(text: str) -> str:
    """Escape ``$`` so matplotlib renders it literally.

    Group names such as ``most-willing:$10-$15`` contain a matched pair of
    dollar signs, which matplotlib's mathtext would otherwise read as inline
    math and render as ``most-willing:10-15`` in italics.
    """

    return str(text).replace("$", r"\$")


def per_group_levels(
    config: "CoffeeCrossfitConfig", split_table: Any, level_scores: Sequence[Sequence[Sequence[float]]],
) -> dict[str, Any]:
    """Corrected intervals for each arm's own per-group distortion **level**.

    Same reduction and same Nadeau-Bengio correction as everywhere else in this
    route: one seed-averaged number per split, then the corrected interval over
    the splits.  These are *levels*, not contrasts, and the two differ in what
    they can support:

    * the intervals here are **marginal** per (arm, group) cell.  They are not
      simultaneous over the twelve cells and are not adjusted for multiplicity;
    * because the two arms are fitted on the same split from the same shared
      comparison log, their level estimates are **paired and positively
      correlated**.  Overlapping level intervals therefore do **not** imply the
      arms are indistinguishable.  The paired question is answered only by
      ``per_group_gaps``, whose interval is computed on the within-split
      difference and is the one to read for a comparison.
    """

    test_sizes = split_table["n_test"].to_numpy()
    train_sizes = split_table["n_train"].to_numpy()
    cells = []
    for arm_index, arm in enumerate(ARMS):
        for group_index, group in enumerate(config.groups):
            result = nadeau_bengio(
                np.asarray(level_scores[arm_index][group_index], dtype=np.float64),
                test_sizes, train_sizes, confidence_level=config.confidence_level,
            )
            cells.append({
                "arm": arm,
                "group": group,
                "status": result.status,
                "n_splits": result.n_scores,
                "estimate": result.estimate,
                "corrected_se": result.corrected_se,
                "degrees_of_freedom": result.degrees_of_freedom,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
            })
    return {
        "quantity": "mean_over_splits( mean_seed( D_arm[group] ) )",
        "lower_is_better": True,
        "method": "nadeau_bengio_corrected_repeated_cv_t_per_arm_and_group",
        "confidence_level": config.confidence_level,
        "interval_type": "marginal_corrected_95_percent_not_simultaneous",
        "interval_label": (
            "ordinary marginal Nadeau-Bengio corrected 95% intervals, one (arm, group) cell at a "
            "time; they are not simultaneous over the cells and are not multiplicity-adjusted"
        ),
        "pairing_warning": (
            "the two arms share each split and each seed's comparison log, so their level estimates "
            "are paired and positively correlated; overlap between a pooled interval and a "
            "multigroup interval does NOT mean the arms are indistinguishable, and the difference "
            "of the two levels is not the paired gap -- read group_gaps for the comparison"
        ),
        "cells": cells,
    }


def _write_level_figure(reduce_dir: Path, levels: Mapping[str, Any],
                        gaps: Mapping[str, Any] | None = None) -> None:
    """Per-group distortion levels for both arms with their corrected intervals."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_arm: dict[str, list[Mapping[str, Any]]] = {}
        for cell in levels["cells"]:
            if cell.get("ci_lower") is not None:
                by_arm.setdefault(cell["arm"], []).append(cell)
        if not by_arm:
            return
        group_names = [cell["group"] for cell in by_arm[ARMS[0]]]
        positions = np.arange(len(group_names))[::-1].astype(float)
        colors = {"pooled": "#b23a48", "multigroup": "#2f6f8f"}
        offsets = {"pooled": 0.16, "multigroup": -0.16}

        figure, axes = plt.subplots(figsize=(7.2, 0.62 * len(group_names) + 2.2))
        for arm, cells in by_arm.items():
            y = positions + offsets.get(arm, 0.0)
            estimates = np.array([cell["estimate"] for cell in cells])
            lower = estimates - np.array([cell["ci_lower"] for cell in cells])
            upper = np.array([cell["ci_upper"] for cell in cells]) - estimates
            axes.errorbar(
                estimates, y, xerr=np.vstack([lower, upper]), fmt="o", markersize=5,
                color=colors.get(arm, "#666666"), ecolor=colors.get(arm, "#666666"),
                elinewidth=2.0, capsize=3, label=arm,
            )
        for position in positions[1::2]:
            axes.axhspan(position - 0.5, position + 0.5, color="#000000", alpha=0.035, zorder=0)
        axes.set_yticks(positions)
        axes.set_yticklabels([_plain_label(name) for name in group_names], fontsize=9)
        axes.set_ylim(positions.min() - 0.6, positions.max() + 0.6)
        axes.set_xlabel("empirical distortion (lower is better)")
        axes.set_title(
            "Per-group distortion by arm\n"
            f"marginal corrected 95% intervals over {levels['cells'][0]['n_splits']} splits",
            fontsize=10,
        )
        axes.legend(frameon=False, fontsize=9, loc="best")
        # The arms are paired within each split, so overlap here is not a test.
        caption = (
            "Intervals are marginal and not simultaneous. The arms are paired within each split,\n"
            "so overlapping intervals do not mean the arms are indistinguishable"
        )
        if gaps is not None:
            caption += "; see group_gaps for the paired comparison"
        figure.text(0.01, 0.01, caption + ".", fontsize=7.5, color="#444444", va="bottom")
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        for extension in ("png", "pdf"):
            figure.savefig(reduce_dir / f"group_levels.{extension}", dpi=200)
        plt.close(figure)
    except Exception:  # a plot is a convenience, not a result
        pass


def _write_gap_figure(reduce_dir: Path, gaps: Mapping[str, Any]) -> None:
    """A small forest plot of the marginal per-group intervals; never fatal."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [row for row in gaps["groups"] if row.get("ci_lower") is not None]
        if not rows:
            return
        figure, axes = plt.subplots(figsize=(6.4, 0.5 * len(rows) + 1.8))
        positions = np.arange(len(rows))[::-1]
        for position, row in zip(positions, rows):
            axes.plot([row["ci_lower"], row["ci_upper"]], [position, position],
                      color="#7aa6c2", linewidth=2.4, solid_capstyle="butt")
            axes.plot([row["estimate"]], [position], marker="o", color="#b23a48", markersize=5)
        axes.axvline(0.0, color="#444444", linewidth=1.0, linestyle=":")
        axes.set_yticks(positions)
        axes.set_yticklabels([_plain_label(row["group"]) for row in rows], fontsize=8)
        axes.set_xlabel("per-group paired gap: pooled minus multigroup distortion")
        axes.set_title("Marginal corrected 95% intervals (not simultaneous)", fontsize=9)
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(reduce_dir / f"group_gaps.{extension}", dpi=200)
        plt.close(figure)
    except Exception:  # a plot is a convenience, not a result
        pass


def _write_figure(reduce_dir: Path, scores: np.ndarray, inference: Mapping[str, Any]) -> None:
    """One small contrast figure; a failure here never fails a completed run."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(6.0, 3.4))
        axes.hist(scores, bins=min(20, max(5, len(scores) // 2)), color="#7aa6c2", edgecolor="white")
        axes.axvline(0.0, color="#444444", linewidth=1.0, linestyle=":")
        axes.axvline(float(inference["estimate"]), color="#b23a48", linewidth=1.6, label="estimate")
        if inference.get("ci_lower") is not None:
            axes.axvspan(float(inference["ci_lower"]), float(inference["ci_upper"]),
                         color="#b23a48", alpha=0.12, label="95% corrected interval")
        axes.set_xlabel("split score: mean-over-seeds pooled minus multigroup worst-group distortion")
        axes.set_ylabel("splits")
        axes.set_title(f"{inference['n_splits']} repeated-CV split scores")
        axes.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(reduce_dir / f"split_scores.{extension}", dpi=200)
        plt.close(figure)
    except Exception:  # a plot is a convenience, not a result
        pass


# ---------------------------------------------------------------- workflow


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sentence(text: str) -> str:
    """Capitalize the first letter only; ``str.capitalize`` would lowercase 'CV'."""

    return (text[:1].upper() + text[1:] + ".") if text else ""


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "undefined"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{float(value):.{digits}f}"


def _fmt_p(value: Any) -> str:
    """P-values span many orders of magnitude; fixed decimals hide the small ones."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "undefined"
    value = float(value)
    return f"{value:.6f}" if value >= 1e-4 else f"{value:.3e}"


def write_summary(output_dir: Path, summary: Mapping[str, Any]) -> None:
    output_dir = Path(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    info = summary["run_info"]
    design = summary.get("design", {})
    inference = summary.get("inference")
    route = ROUTES.get(str(info.get("protocol_id")), ROUTE_V1)
    lines = [
        f"# {route.title} (`{route.protocol_id}`)", "",
        f"**Status:** {info['status']}" + ("" if info["status"] == "completed" else f" — {info.get('error', '')}"),
        "",
        *[f"> {line}" for line in route.standing_note],
        "",
        f"Started {info.get('started_at')}; finished {info.get('finished_at', 'n/a')}.",
        f"Python {info['package_versions'].get('python')}, workers={info.get('workers')}.",
        "",
    ]
    if design:
        discovery, cv = design["discovery"], design["cross_validation"]
        lines += [
            "## Design", "",
            f"- Eligible respondents: **{design['data']['eligible_rows']}** (complete personal-preference profiles).",
            f"- Discovery sample reserved and excluded: **{discovery['k']}** "
            f"(`default_rng({discovery['seed']}).choice(n, k, replace=False)`, "
            f"ceiling of {discovery['fraction']:.2f} n, source CSV row order after the eligibility filter).",
            f"- Cross-validation population: **{discovery['n_analysis']}** respondents; the physical CSV is unchanged.",
            f"- Repeated CV: **{cv['repeats']} repeats x {cv['folds']} folds = {cv['n_splits']} splits**, "
            f"cv_seed {cv['seed']}.",
            "- Selected groups (user-specified): "
            + ", ".join(f"`{item['name']}` ({item['analysis_members']})" for item in design["groups"]["definitions"]),
            *[f"- {_sentence(design['groups'][key])}" for key in route.group_notes],
            "- Missing or unrecognized values mean nonmembership; membership is never imputed.",
            "",
        ]
        learner = design.get("learner")
        if learner:
            lines += [
                "## Feedback protocol", "",
                f"- Protocol: `{learner['protocol']}`, "
                f"{learner['comparisons_per_respondent']} comparison per training respondent.",
                f"- Pair selection: {_sentence(learner['pair_selection'])}",
                f"- Row order: {_sentence(learner['row_order'])}",
                f"- Labels: `beta = {learner['beta']}` — {_sentence(learner['labels'])}",
                f"- One comparison log shared identically across the paired arms: "
                f"**{learner['shared_comparison_log_across_arms']}**.",
                f"- Pairing between arms: {_sentence(learner['pairing_between_arms'])}",
                f"- Drawn logs: {_sentence(learner['drawn_logs_saved'])}",
                f"- Held-out utilities: {_sentence(learner['held_out_utilities'])}",
                "",
            ]
    if inference:
        status = inference["status"]
        lines += [
            "## Headline", "",
            f"`{inference['formula']}` (positive favors {inference['positive_favors']})", "",
            "| estimate | corrected SE | df | 95% corrected interval | t | two-sided p | splits | seeds |",
            "|---:|---:|---:|---|---:|---:|---:|---:|",
            f"| {_fmt(inference['estimate'])} | {_fmt(inference['corrected_se'])} | "
            f"{inference['degrees_of_freedom']} | "
            + (f"[{_fmt(inference['ci_lower'])}, {_fmt(inference['ci_upper'])}]"
               if inference.get("ci_lower") is not None else "undefined")
            + f" | {_fmt(inference['t_statistic'])} | {_fmt_p(inference['p_value_two_sided'])} | "
            f"{inference['n_splits']} | {inference['n_learner_seeds']} |",
            "",
            f"Split-score variance {inference['variance']:.6g}; mean test/train ratio "
            f"{_fmt(inference['test_train_ratio'], 6)}; variance factor "
            f"{_fmt(inference['variance_factor'], 6)} = 1/{inference['n_splits']} + ratio.",
            "",
            _sentence(inference["estimand_note"]),
            "",
            _sentence(inference["method_note"]),
            "",
        ]
        if status != NB_STATUS_OK:
            lines += [f"**Inference status: `{status}`.** The test and interval are reported as undefined.", ""]
        if inference.get("naive_ci_lower") is not None:
            lines += [
                f"Diagnostic only, **not for reporting**: the uncorrected interval that treats the "
                f"{inference['n_splits']} overlapping split scores as independent is "
                f"[{_fmt(inference['naive_ci_lower'])}, {_fmt(inference['naive_ci_upper'])}] "
                f"(SE {_fmt(inference['naive_se'])}). It understates the uncertainty.", "",
            ]
        lines += [
            "This route does not reproduce, and is not comparable with, the accepted fixed-holdout result in",
            "`results/coffee-primary/run-20260915-aligned`: the group family, the training sets and the",
            "estimand all differ.", "",
        ]
        if route is ROUTE_V5_10X10_ACTIVE_BETA3:
            lines += [
                "These numbers may be reported **beside `coffee-repeated-cv-v4-10x10`'s** as a",
                "*same design, different feedback protocol* sensitivity: v4 is this design with a uniform",
                "passive log and deterministic labels, and the two differ in exactly those two things. They",
                "must **not** be compared with `coffee-repeated-cv-v1`, `-v2-10x10` or `-v3-10x10`, whose",
                "analysis complement is a different 90% of the same survey. Note also that active sampling",
                "removes the shared comparison log, so the between-arm pairing here is weaker than v4's and",
                "the split-score variance is not directly attributable to the label model alone.", "",
            ]
        if route is ROUTE_V2_10X10:
            lines += [
                "It is not directly comparable with `coffee-repeated-cv-v1`: both the fold design and",
                "the subgroup family changed.", "",
            ]
        gaps = inference.get("per_group_gaps")
        if gaps:
            lines += [
                "## Per-group paired gaps", "",
                f"`{gaps['estimand']}` (positive favors {gaps['positive_favors']})", "",
                "| group | estimate | corrected SE | 95% corrected interval (marginal) | p | Holm p | "
                "Holm rejects |",
                "|---|---:|---:|---|---:|---:|:--:|",
            ]
            for row in gaps["groups"]:
                interval = (
                    f"[{_fmt(row['ci_lower'])}, {_fmt(row['ci_upper'])}]"
                    if row.get("ci_lower") is not None else "undefined"
                )
                holm = "n/a" if row["holm_reject_at_alpha"] is None else (
                    "yes" if row["holm_reject_at_alpha"] else "no"
                )
                lines.append(
                    f"| `{row['group']}` | {_fmt(row['estimate'])} | {_fmt(row['corrected_se'])} | "
                    f"{interval} | {_fmt_p(row['p_value_two_sided'])} | {_fmt_p(row['p_value_holm'])} | {holm} |"
                )
            lines += [
                "",
                _sentence(gaps["interval_label"]),
                "",
                _sentence(gaps["multiplicity_label"]),
                "",
                _sentence(gaps["companion_note"]),
                "",
            ]
        levels = inference.get("per_group_levels")
        if levels:
            by_group: dict[str, dict[str, Any]] = {}
            for cell in levels["cells"]:
                by_group.setdefault(cell["group"], {})[cell["arm"]] = cell
            lines += [
                "## Per-group distortion by arm", "",
                f"`{levels['quantity']}` (lower is better)", "",
                "| group | pooled | 95% corrected interval | multigroup | 95% corrected interval |",
                "|---|---:|---|---:|---|",
            ]
            for group, cells in by_group.items():
                row = [f"| `{group}` "]
                for arm in ARMS:
                    cell = cells.get(arm, {})
                    interval = (
                        f"[{_fmt(cell.get('ci_lower'))}, {_fmt(cell.get('ci_upper'))}]"
                        if cell.get("ci_lower") is not None else "undefined"
                    )
                    row.append(f"| {_fmt(cell.get('estimate'))} | {interval} ")
                lines.append("".join(row) + "|")
            lines += [
                "",
                _sentence(levels["interval_label"]),
                "",
                "**" + _sentence(levels["pairing_warning"]) + "**",
                "",
                "Figure: `reduce/group_levels.png` / `.pdf`.",
                "",
            ]
    lines += ["## Where the outputs are", ""]
    for label, relative in summary.get("outputs", {}).items():
        lines.append(f"- {label}: `{relative}`")
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class CrossfitResult:
    exit_code: int
    output_dir: Path
    summary: dict[str, Any]


def run_crossfit(
    *,
    project_root: Path,
    output_dir: Path,
    config_path: Path | None = None,
    config: CoffeeCrossfitConfig | None = None,
    route: CrossfitRoute | None = None,
    expected_data_sha256: str | None = None,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> CrossfitResult:
    """Run the whole repeated-CV route into ``output_dir``, resuming saved splits."""

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config is None:
        config = load_crossfit_config(
            config_path or project_root / "configs" / (route or ROUTE_V1).config_name
        )
    if route is not None and config.route is not route:
        raise ValueError(
            f"this run was started as {route.protocol_id!r} but the configuration declares "
            f"{config.route.protocol_id!r}"
        )

    try:
        from coffee_expanded.manifests import runtime_versions

        versions = runtime_versions()
    except Exception:
        import sys as _sys

        versions = {"python": _sys.version.split()[0]}

    info: dict[str, Any] = {
        "protocol_id": config.route.protocol_id, "mode": config.route.mode,
        "reportable": True, "status": "running",
        "implementation_id": config.route.implementation_id, "started_at": _now(), "workers": int(workers),
        "package_versions": versions,
    }
    summary: dict[str, Any] = {"run_info": info, "outputs": {
        "run record": "run_info.json", "log": "run.log",
        "design manifest": "design/design.json", "per-split results": "splits/",
        "tables": "reduce/ (split_scores.csv, seed_scores.csv, group_values.csv, inference.json)",
    }}
    if config.route.active_feedback:
        summary["outputs"]["per-arm drawn comparison logs"] = (
            "splits/rNN_fMM/comparisons.npz (first, second, sign, propensity)"
        )
    if config.route.per_group_gaps:
        summary["outputs"]["per-group paired gaps"] = "reduce/group_gaps.csv, reduce/group_gaps.json"
        summary["outputs"]["per-group distortion levels by arm"] = (
            "reduce/group_levels.csv, reduce/group_levels.json, reduce/group_levels.png"
        )
    info_path = output_dir / "run_info.json"

    def save_info() -> None:
        info_path.write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")

    save_info()
    exit_code = 0
    try:
        log(f"crossfit: loading inputs and reserving the {config.discovery_fraction:.0%} discovery sample")
        inputs = load_crossfit_inputs(
            config, project_root=project_root, expected_data_sha256=expected_data_sha256
        )
        validate_design(config, inputs)
        manifest = design_manifest(config, inputs)
        # The design is written before any policy is fitted, so a change to the
        # groups or the discovery identity invalidates resume rather than being
        # silently mixed into saved units.
        _write_json(output_dir / "design" / "design.json", manifest)
        np.savez_compressed(
            output_dir / "design" / "discovery.npz",
            discovery_indices=inputs.reservation.discovery_indices,
            analysis_indices=inputs.reservation.analysis_indices,
            draw_order=inputs.reservation.draw_order,
            eligible_respondent_ids=np.asarray(inputs.dataset.respondent_ids).astype(str),
            fold_of=inputs.plan.fold_of,
        )
        info["design"] = {
            "eligible_rows": manifest["data"]["eligible_rows"],
            "discovery_k": manifest["discovery"]["k"],
            "n_analysis": manifest["discovery"]["n_analysis"],
            "n_splits": manifest["cross_validation"]["n_splits"],
            "learner_seeds": len(config.learner_seeds),
            "policy_fits": manifest["cross_validation"]["n_splits"] * len(config.learner_seeds) * len(ARMS),
        }
        summary["design"] = manifest
        save_info()
        log(f"crossfit: {manifest['discovery']['k']} discovery respondents excluded; "
            f"{manifest['discovery']['n_analysis']} remain for {manifest['cross_validation']['n_splits']} splits "
            f"({info['design']['policy_fits']} policy fits)")

        tasks = [(repeat, fold) for repeat, fold in inputs.plan.splits()]
        _init_worker(config.as_mapping(), str(project_root), str(output_dir), expected_data_sha256)
        _WORKER_STATE.update(config=config, inputs=inputs, output_dir=str(output_dir))
        started = time.perf_counter()

        def on_progress(done: int, total: int, task: tuple[int, int], seconds: float) -> None:
            log(f"crossfit split repeat={task[0]} fold={task[1]} done in {seconds:.1f} s ({done}/{total})")

        split_summaries, report = ordered_map(
            _run_split_task, tasks, workers=int(workers),
            initializer=_init_worker,
            initargs=(config.as_mapping(), str(project_root), str(output_dir), expected_data_sha256),
            on_progress=on_progress, log=log,
        )
        info["splits_seconds"] = round(time.perf_counter() - started, 2)
        info["splits_reused"] = int(sum(1 for row in split_summaries if row.get("reused")))
        info["parallelism"] = report.summary()
        log(f"crossfit: {len(split_summaries)} splits complete ({info['splits_reused']} reused); {report.summary()}")
        save_info()

        log("crossfit: reducing to split scores and computing the corrected interval")
        inference = reduce_splits(config, output_dir, split_summaries)
        summary["inference"] = inference
        info["inference_status"] = inference["status"]
        info["estimate"] = inference["estimate"]
        if inference["status"] != NB_STATUS_OK:
            log(f"crossfit: inference status {inference['status']}; the test and interval are undefined")
        else:
            log(f"crossfit: estimate {inference['estimate']:.6f}, corrected SE {inference['corrected_se']:.6f}, "
                f"95% CI [{inference['ci_lower']:.6f}, {inference['ci_upper']:.6f}], "
                f"p={inference['p_value_two_sided']:.6g}")
        info["status"] = "completed"
        info["finished_at"] = _now()
    except BaseException as error:
        info["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        info["finished_at"] = _now()
        info["error"] = f"{type(error).__name__}: {error}"
        info["traceback"] = traceback.format_exc(limit=8)
        log(f"FAILED: {type(error).__name__}: {error}")
        exit_code = 1
        save_info()
        write_summary(output_dir, summary)
        if isinstance(error, KeyboardInterrupt):
            raise
        return CrossfitResult(exit_code, output_dir, summary)
    save_info()
    write_summary(output_dir, summary)
    log(f"summary: {output_dir / 'SUMMARY.md'}")
    return CrossfitResult(exit_code, output_dir, summary)
