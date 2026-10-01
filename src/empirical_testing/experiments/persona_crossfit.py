"""Persona panel construction, discovery group selection, and cross-validation.

Answer draws are independently permuted within each persona cell to form
artificial respondents. A cell-stratified discovery sample selects the group
family once, before cross-validation, and is excluded from all analysis folds.
Fitting and corrected inference use the shared Coffee routines.

Panel sensitivity repeats construction and group selection for each panel seed.
Intervals are conditional on the selected groups and artificial alignment."""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from empirical_testing.core.holdout import sorted_id_sha256
from empirical_testing.core.repeated_cv import (
    NB_STATUS_OK,
    repeated_fold_plan,
    stratified_discovery_reservation,
)
from empirical_testing.datasets import load_dense_dataset
from empirical_testing.datasets.persona import (
    UNIVERSAL,
    panel_record,
    permutation_table,
)
from empirical_testing.experiments.coffee_crossfit import (
    ACTIVE_PROTOCOL,
    ARMS,
    FEEDBACK_PROTOCOLS,
    PASSIVE_PROTOCOL,
    CrossfitInputs,
    feedback_block,
    parse_label_beta,
    reduce_splits,
    run_split,
)
from empirical_testing.parallel import ordered_map

ESTIMAND_NAME = "mean_split_of_paired_pooled_minus_multigroup_worst_group_empirical_distortion"
ESTIMAND_FORMULA = "mean_over_splits( mean_seed( max_group(D_pooled) - max_group(D_multigroup) ) )"

# Predeclared selection rules.  Every one of these is fixed before any score is
# read; the run fails rather than resolving an unanticipated case at analysis
# time.

#: The number words the statistic identifier spells out.  ``C(5,2) = 10`` gives
#: v1's ``..._ten_mean_pairwise_margins_...`` and ``C(6,2) = 15`` gives v2's
#: ``..._fifteen_...``: the identifier is generated, not hard-coded, so adding
#: an answer changes it in step with the statistic it names.
_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "twenty_one", "twenty_two", "twenty_three", "twenty_four",
    "twenty_five", "twenty_six", "twenty_seven", "twenty_eight", "twenty_nine", "thirty",
)


def n_pairwise_margins(n_answers: int) -> int:
    """``choose(n_answers, 2)``: how many ordered-pair margins a summary has."""

    n = int(n_answers)
    if n < 2:
        raise ValueError("a pairwise margin summary needs at least two answers")
    return n * (n - 1) // 2


def selection_statistic(n_answers: int) -> str:
    """The predeclared selection-statistic identifier for ``n_answers`` answers.

    The statistic itself is action-count agnostic -- it is the Pearson
    correlation between a candidate's ``choose(A, 2)`` mean pairwise margins and
    the discovery population's -- but the *identifier* spells the count out, so
    a configuration cannot silently claim a ten-margin statistic over a
    six-answer collection.  :func:`PersonaCrossfitConfig.declared_n_answers`
    inverts this, and the run cross-checks it against the loaded data.
    """

    count = n_pairwise_margins(n_answers)
    if count >= len(_NUMBER_WORDS):
        raise ValueError(f"no declared statistic identifier for {count} pairwise margins")
    return f"pearson_correlation_of_{_NUMBER_WORDS[count]}_mean_pairwise_margins_with_discovery_population"


#: Every identifier a configuration may declare, mapped back to its answer
#: count.  Membership is checked when the configuration is read; the count is
#: checked against the data before any policy is fitted.
KNOWN_SELECTION_STATISTICS: dict[str, int] = {
    selection_statistic(answers): answers
    for answers in range(2, 2 + len(_NUMBER_WORDS))
    if n_pairwise_margins(answers) < len(_NUMBER_WORDS)
}
#: v1's identifier, kept as a named constant because its frozen configuration
#: and its published protocol both carry this exact string.
SELECTION_STATISTIC = selection_statistic(5)
#: Four candidates from the low end of the ranking -- those whose preference
#: ordering departs *most* from the discovery population's -- plus one ordinary
#: candidate from the middle of the ranking, so the family is not composed
#: entirely of outliers and the multigroup arm is also asked to serve a group
#: that behaves like the population.  "Low" means the low end of the observed
#: range and is not by itself a claim about sign: each candidate's correlation is
#: reported with its sign, and the fifth slot is the median rather than the
#: maximum because a maximum-correlation group is close to indistinguishable
#: from ``universal``.
SELECTION_RULE = "lowest_four_plus_median_correlation"
SELECTION_TIE_BREAK = (
    "low slots: correlation ascending; median slot: the lower median of the remaining eligible "
    "candidates in ascending correlation; high slots (unused here): correlation descending; then "
    "discovery support descending, then candidate file order; slots are filled low, then median, "
    "then high, and a candidate can fill only one slot"
)
SELECTION_UNDEFINED = "ineligible_never_selected_and_recorded_as_undefined"
SELECTION_COMPLEMENTS = "complementary_groups_may_both_be_selected"


@dataclass(frozen=True)
class PersonaRoute:
    """Everything that distinguishes one frozen persona repeated-CV route.

    A route owns a protocol id, an implementation id, a historical route label, a
    configuration file, a **protocol-hash file**, a result namespace and a
    figure namespace.  Two routes therefore cannot read each other's frozen
    hashes, write into each other's result directories or overwrite each
    other's figures, and a route pointed at the other's collection fails the
    snapshot hash check rather than analysing it.
    """

    protocol_id: str
    implementation_id: str
    mode: str
    config_name: str
    study_status: str
    title: str
    standing_note: tuple[str, ...]
    #: The ``experiment/persona-eval/`` file holding this route's frozen input
    #: hashes.  Route-specific: v1 must not be checked against v2's snapshot.
    protocol_file: str = "analysis_protocol.json"
    #: Where this route's figures belong, under ``figures/``.
    figure_namespace: str = "persona_crossfit"
    per_group_gaps: bool = True
    #: The feedback protocol this route's configuration must declare.  Declared
    #: per route so the two passive routes keep their guard exactly as it was.
    feedback_protocol: str = PASSIVE_PROTOCOL
    #: The Bradley-Terry label parameter the configuration must declare, as the
    #: string the config file carries and the digest covers.
    label_beta: str = "inf"

    @property
    def beta_value(self) -> float:
        """The numeric label parameter, ``+inf`` for the score-order oracle."""

        return parse_label_beta(self.label_beta)

    @property
    def active_feedback(self) -> bool:
        """Whether the learner chooses its own pairs (and so owns its own log)."""

        return self.feedback_protocol == ACTIVE_PROTOCOL


ROUTE_V1 = PersonaRoute(
    protocol_id="persona-repeated-cv-v1",
    implementation_id="persona-crossfit-2026-09-16.1",
    mode="persona-crossfit",
    config_name="persona_crossfit.json",
    protocol_file="analysis_protocol.json",
    figure_namespace="persona_crossfit",
    study_status="exploratory_analysis_of_synthetic_model_ratings_with_discovery_selected_groups",
    title="Persona synthetic-panel repeated cross-validation",
    standing_note=(
        "**Synthetic benchmark.** The ratings are model outputs produced under persona prompts by a",
        "single evaluator model on one question. A persona bitstring identifies a demographic *cell*,",
        "not a person, and nothing here observes or predicts human demographic preferences.",
        "The five answers were rated in separate API calls, so no five-answer vector was ever jointly",
        "elicited: pseudo-respondents are built by independently permuting each answer's draws within a",
        "cell under the recorded `panel_seed`, and that alignment is an artifact of the construction.",
        "The group family was selected on the reserved 10% discovery sample, so the reported interval is",
        "conditional on that selection and carries none of its uncertainty.",
    ),
)

ROUTE_V2 = PersonaRoute(
    protocol_id="persona-repeated-cv-v2-fixed-six-answer",
    implementation_id="persona-crossfit-fixed-v8-2026-09-16.1",
    mode="persona-crossfit-fixed-v8",
    config_name="persona_crossfit_fixed_v8.json",
    protocol_file="analysis_protocol_fixed_v8.json",
    figure_namespace="persona_crossfit_fixed_v8",
    study_status="exploratory_analysis_of_synthetic_model_ratings_with_discovery_selected_groups",
    title="Persona fixed-order six-answer synthetic-panel repeated cross-validation",
    standing_note=(
        "**Synthetic benchmark.** The ratings are model outputs produced under persona prompts by a",
        "single evaluator model on one question. A persona bitstring identifies a demographic *cell*,",
        "not a person, and nothing here observes or predicts human demographic preferences.",
        "The six answers were rated in separate API calls, so no six-answer vector was ever jointly",
        "elicited: pseudo-respondents are built by independently permuting each answer's draws within a",
        "cell under the recorded `panel_seed`, and that alignment is an artifact of the construction.",
        "The group family was selected on the reserved 10% discovery sample, so the reported interval is",
        "conditional on that selection and carries none of its uncertainty.",
        "**Demographic-line position was held constant** in every request of this collection. That makes",
        "line position a fixed condition, not a variable: its causal effect is *not* identified here.",
        "This collection also differs from `persona-repeated-cv-v1` in prompt wording, answer wording,",
        "answer count, negative controls and draws per cell/answer, so a v1-v2 difference is descriptive",
        "only and attributes to nothing in particular. The reportable inferential comparison is pooled",
        "versus multigroup **within** this collection.",
    ),
)

#: ``persona-repeated-cv-v3-fixed-six-answer-active-beta3``: the
#: ``persona-repeated-cv-v2-fixed-six-answer`` route over the *same* frozen
#: collection, with two changes to the **feedback protocol** and nothing else:
#: pairs come from the learner's own active self-play instead of a uniform
#: passive log, and labels are drawn from ``sigmoid(3 * (u_a - u_b))`` instead of
#: the deterministic score-order oracle.  The collection, the panel seed, the
#: discovery seed and reservation, the candidate universe and selection rule,
#: the cv seed, the 10 x 10 fold plan, the 50 learner seeds, ``tau``, the
#: uniform reference and the corrected inference are all v2's, unchanged -- and
#: because the selection rule reads only the discovery sample's utilities, which
#: the feedback protocol does not touch, the selected family is v2's too.
#:
#: Under active sampling the two arms cannot share a comparison log, so the
#: route records ``shared_log_across_arms: False`` and saves both arms' drawn
#: logs.  Its numbers may be read beside v2's as a *same design, different
#: feedback protocol* sensitivity, and never against ``persona-repeated-cv-v1``,
#: which is a different collection.
ROUTE_V3_ACTIVE_BETA3 = PersonaRoute(
    protocol_id="persona-repeated-cv-v3-fixed-six-answer-active-beta3",
    implementation_id="persona-crossfit-fixed-v8-active-beta3-2026-09-20.1",
    mode="persona-crossfit-fixed-v8-active-beta3",
    config_name="persona_crossfit_fixed_v8_active_beta3.json",
    protocol_file="analysis_protocol_fixed_v8_active_beta3.json",
    figure_namespace="persona_crossfit_fixed_v8_active_beta3",
    study_status="exploratory_analysis_of_synthetic_model_ratings_with_discovery_selected_groups",
    title=(
        "Persona fixed-order six-answer repeated cross-validation "
        "(active sampling, beta = 3)"
    ),
    standing_note=(
        "**Synthetic benchmark.** The ratings are model outputs produced under persona prompts by a",
        "single evaluator model on one question. A persona bitstring identifies a demographic *cell*,",
        "not a person, and nothing here observes or predicts human demographic preferences.",
        "The six answers were rated in separate API calls, so no six-answer vector was ever jointly",
        "elicited: pseudo-respondents are built by independently permuting each answer's draws within a",
        "cell under the recorded `panel_seed`, and that alignment is an artifact of the construction.",
        "The group family was selected on the reserved 10% discovery sample, so the reported interval is",
        "conditional on that selection and carries none of its uncertainty.",
        "This route is",
        "`persona-repeated-cv-v2-fixed-six-answer` over the same frozen collection with exactly two",
        "changes to the *feedback protocol*: pairs are chosen by the learner's own active self-play",
        "instead of a uniform passive log, and labels are drawn from a finite Bradley-Terry model with",
        "`beta = 3` instead of the deterministic score-order oracle. Under active sampling the two arms",
        "**cannot** share one comparison log: each arm's pairs depend on its own policy, so common",
        "random numbers align the label coin but not the pairs, and the between-arm pairing is weaker",
        "than in the passive route. These numbers may be read beside v2's as a same-design,",
        "different-feedback-protocol sensitivity. They are **not** compared with",
        "`persona-repeated-cv-v1`, which is a different collection entirely.",
        "**Demographic-line position was held constant** in every request of this collection, so its",
        "causal effect is not identified here. The reportable inferential comparison is pooled versus",
        "multigroup **within** this collection.",
    ),
    feedback_protocol=ACTIVE_PROTOCOL,
    label_beta="3",
)

ROUTES: dict[str, PersonaRoute] = {
    route.protocol_id: route
    for route in (ROUTE_V1, ROUTE_V2, ROUTE_V3_ACTIVE_BETA3)
}

CONFIG_KEYS = frozenset({
    "protocol_id", "study_status", "dataset", "data_path", "study_path", "panel_seed",
    "panel_alignment", "split", "discovery_seed", "discovery_fraction", "discovery_rounding",
    "discovery_stratification", "discovery_draw", "cv_seed", "folds", "repeats",
    "confidence_level", "group_selection", "candidate_groups", "n_selected_subgroups",
    "n_low_correlation", "n_median_correlation", "n_high_correlation", "min_discovery_support", "selection_statistic", "selection_rule", "selection_tie_break",
    "selection_undefined", "selection_complements", "algorithm", "arms", "learner_seeds",
    "protocol", "comparisons_per_respondent", "reference", "tau", "output_averaging",
    "tie_rule", "beta", "estimand", "inference",
})
#: Keys a configuration *may* carry.  Not required, so the two frozen configs
#: stay complete and valid exactly as written, and an absent optional key
#: contributes nothing to ``as_mapping`` -- so no existing digest moves.
OPTIONAL_CONFIG_KEYS = frozenset({"changes_from_source_route"})


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class PersonaCrossfitConfig:
    """Validated configuration for one persona repeated-CV route.

    ``groups`` is empty in the configuration file and stays empty until
    :func:`select_groups` has run on the discovery sample; the selected family
    is then frozen into a copy with :meth:`with_groups`.  Nothing downstream
    accepts a config whose groups are still empty, so a policy can never be
    fitted against an unselected family.
    """

    route: PersonaRoute
    dataset: str
    data_path: str
    study_path: str
    panel_seed: int
    discovery_seed: int
    discovery_fraction: float
    cv_seed: int
    folds: int
    repeats: int
    confidence_level: float
    candidate_groups: tuple[str, ...]
    n_selected_subgroups: int
    n_low_correlation: int
    n_median_correlation: int
    n_high_correlation: int
    min_discovery_support: int
    #: The predeclared statistic identifier, which spells out
    #: ``choose(n_answers, 2)``.  See :func:`selection_statistic`.
    selection_statistic: str
    algorithm: str
    arms: tuple[str, ...]
    learner_seeds: tuple[int, ...]
    protocol: str
    comparisons_per_respondent: int
    reference: str
    tau: float
    output_averaging: str
    tie_rule: str
    groups: tuple[str, ...] = ()
    #: Optional one-line note naming the route this one was copied from and what
    #: was changed.  Documentation only, but inside the configuration digest so
    #: it cannot be edited silently.
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
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PersonaCrossfitConfig":
        unknown = sorted(set(raw) - CONFIG_KEYS - OPTIONAL_CONFIG_KEYS - {"groups"})
        if unknown:
            raise ValueError(f"unrecognized persona crossfit configuration keys: {unknown}")
        missing = sorted(CONFIG_KEYS - set(raw))
        if missing:
            raise ValueError(f"persona crossfit configuration is missing keys: {missing}")
        route = ROUTES.get(str(raw["protocol_id"]))
        if route is None:
            raise ValueError(
                f"unknown persona protocol_id {raw['protocol_id']!r}; known routes: {sorted(ROUTES)}"
            )
        if str(raw["study_status"]) != route.study_status:
            raise ValueError(f"{route.protocol_id} requires study_status={route.study_status!r}")
        for key, expected in (
            ("split", "repeated_k_fold_cross_validation"),
            ("discovery_rounding", "ceiling"),
            ("discovery_stratification", "persona_cell_balanced_base_plus_seeded_remainder"),
            ("group_selection", "discovery_sample_only_frozen_before_any_policy_fit"),
            ("selection_rule", SELECTION_RULE),
            ("selection_tie_break", SELECTION_TIE_BREAK),
            ("selection_undefined", SELECTION_UNDEFINED),
            ("selection_complements", SELECTION_COMPLEMENTS),
        ):
            if str(raw[key]) != expected:
                raise ValueError(f"persona crossfit requires {key}={expected!r}")
        # Per route, so the two passive routes reject a finite beta exactly as
        # they did before the active route existed.
        if str(raw["beta"]) != route.label_beta:
            raise ValueError(
                f"{route.protocol_id} declares beta={route.label_beta!r}"
                + (" (deterministic score-order feedback)" if route.label_beta == "inf"
                   else " (finite Bradley-Terry labels)")
                + f", not {str(raw['beta'])!r}"
            )
        config = cls(
            route=route,
            dataset=str(raw["dataset"]),
            data_path=str(raw["data_path"]),
            study_path=str(raw["study_path"]),
            panel_seed=int(raw["panel_seed"]),
            discovery_seed=int(raw["discovery_seed"]),
            discovery_fraction=float(raw["discovery_fraction"]),
            cv_seed=int(raw["cv_seed"]),
            folds=int(raw["folds"]),
            repeats=int(raw["repeats"]),
            confidence_level=float(raw["confidence_level"]),
            candidate_groups=tuple(str(item) for item in raw["candidate_groups"]),
            n_selected_subgroups=int(raw["n_selected_subgroups"]),
            n_low_correlation=int(raw["n_low_correlation"]),
            n_median_correlation=int(raw["n_median_correlation"]),
            n_high_correlation=int(raw["n_high_correlation"]),
            min_discovery_support=int(raw["min_discovery_support"]),
            selection_statistic=str(raw["selection_statistic"]),
            algorithm=str(raw["algorithm"]),
            arms=tuple(str(item) for item in raw["arms"]),
            learner_seeds=tuple(int(value) for value in raw["learner_seeds"]),
            protocol=str(raw["protocol"]),
            comparisons_per_respondent=int(raw["comparisons_per_respondent"]),
            reference=str(raw["reference"]),
            tau=float(raw["tau"]),
            output_averaging=str(raw["output_averaging"]),
            tie_rule=str(raw["tie_rule"]),
            groups=tuple(str(item) for item in raw.get("groups", ())),
            changes_from_source_route=str(raw.get("changes_from_source_route", "")),
        )
        return config.validate()

    def validate(self) -> "PersonaCrossfitConfig":
        if self.beta <= 0 or (not np.isfinite(self.beta) and not np.isposinf(self.beta)):
            raise ValueError("beta must be positive or +infinity")
        if self.dataset != "persona":
            raise ValueError("this route requires dataset='persona'")
        if not self.candidate_groups or self.candidate_groups[0] != UNIVERSAL:
            raise ValueError("the candidate universe must list universal first")
        if len(set(self.candidate_groups)) != len(self.candidate_groups):
            raise ValueError("the candidate universe has duplicate group names")
        if not 1 <= self.n_selected_subgroups <= len(self.candidate_groups) - 1:
            raise ValueError("n_selected_subgroups must fit inside the candidate universe")
        slots = (self.n_low_correlation, self.n_median_correlation, self.n_high_correlation)
        if any(count < 0 for count in slots):
            raise ValueError("the per-slot selection counts cannot be negative")
        if sum(slots) != self.n_selected_subgroups:
            raise ValueError(
                "n_low_correlation + n_median_correlation + n_high_correlation must equal "
                f"n_selected_subgroups ({' + '.join(str(count) for count in slots)} != "
                f"{self.n_selected_subgroups})"
            )
        if self.min_discovery_support < 1:
            raise ValueError("min_discovery_support must be at least one respondent")
        if self.selection_statistic not in KNOWN_SELECTION_STATISTICS:
            raise ValueError(
                f"unrecognized selection_statistic {self.selection_statistic!r}; it must name the "
                "Pearson correlation of choose(n_answers, 2) mean pairwise margins with the discovery "
                f"population, e.g. {selection_statistic(5)!r} for five answers or "
                f"{selection_statistic(6)!r} for six"
            )
        if self.arms != ARMS:
            raise ValueError("this route trains only the matched pooled and multigroup arms")
        if self.algorithm != "groupwise_self_play_span_adaptive":
            raise ValueError("this route requires the span-adaptive learner")
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
        parse_label_beta(self.route.label_beta)
        if self.reference != "uniform" or self.output_averaging != "all_pre_round_policy_maps":
            raise ValueError("this route requires a uniform reference and all pre-round policy averaging")
        if self.tie_rule != "fair_seeded_coin" or not np.isclose(self.tau, 0.5):
            raise ValueError("this route requires fair seeded ties and tau=0.5")
        if self.folds < 2 or self.repeats < 1:
            raise ValueError("this route requires at least two folds and one repeat")
        if not 0.0 < self.discovery_fraction < 1.0:
            raise ValueError("discovery_fraction must lie strictly between zero and one")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not self.learner_seeds or len(set(self.learner_seeds)) != len(self.learner_seeds):
            raise ValueError("learner_seeds must be a nonempty list of distinct integers")
        if self.groups:
            if self.groups[0] != UNIVERSAL:
                raise ValueError("the selected family must list universal first")
            if len(self.groups) != self.n_selected_subgroups + 1:
                raise ValueError(
                    f"the selected family must be universal plus {self.n_selected_subgroups} subgroups"
                )
            unknown = sorted(set(self.groups) - set(self.candidate_groups))
            if unknown:
                raise ValueError(f"selected groups outside the candidate universe: {unknown}")
            if len(set(self.groups)) != len(self.groups):
                raise ValueError("the selected family repeats a group")
        return self

    def with_groups(self, groups: Sequence[str]) -> "PersonaCrossfitConfig":
        """Freeze a discovery-selected family into a copy of this config."""

        if self.groups:
            raise ValueError("the group family is already frozen for this configuration")
        return replace(self, groups=tuple(str(name) for name in groups)).validate()

    @property
    def declared_n_answers(self) -> int:
        """The answer count this configuration's selection statistic declares."""

        return KNOWN_SELECTION_STATISTICS[self.selection_statistic]

    @property
    def n_pairwise_margins(self) -> int:
        """``choose(declared_n_answers, 2)``: the size of a preference summary."""

        return n_pairwise_margins(self.declared_n_answers)

    def check_answer_count(self, action_names: Sequence[str]) -> None:
        """Refuse a collection whose answer count is not the declared one.

        The numerical selection machinery is action-count agnostic, so without
        this check a configuration could claim a ten-margin statistic while
        silently correlating fifteen margins.  The declaration and the data have
        to agree before any group is selected or any policy fitted.
        """

        observed = len(tuple(action_names))
        if observed != self.declared_n_answers:
            raise RuntimeError(
                f"{self.route.protocol_id} declares selection_statistic="
                f"{self.selection_statistic!r}, i.e. {self.declared_n_answers} answers and "
                f"{self.n_pairwise_margins} pairwise margins, but the collection at "
                f"{self.data_path} has {observed} answers "
                f"({n_pairwise_margins(observed)} margins). The protocol string and the data must "
                "agree; nothing is analysed against a collection the protocol does not describe."
            )

    @property
    def dataset_options(self) -> dict[str, Any]:
        return {"study_path": self.study_path, "panel_seed": self.panel_seed}

    @property
    def n_splits(self) -> int:
        return self.repeats * self.folds

    def as_mapping(self) -> dict[str, Any]:
        return {
            "protocol_id": self.route.protocol_id,
            "study_status": self.route.study_status,
            "dataset": self.dataset,
            "data_path": self.data_path,
            "study_path": self.study_path,
            "panel_seed": self.panel_seed,
            "panel_alignment": (
                "within each persona cell, each answer's draws are independently permuted with "
                "default_rng(SeedSequence([panel_seed, cell_index, answer_index])) and aligned"
            ),
            "split": "repeated_k_fold_cross_validation",
            "discovery_seed": self.discovery_seed,
            "discovery_fraction": self.discovery_fraction,
            "discovery_rounding": "ceiling",
            "discovery_stratification": "persona_cell_balanced_base_plus_seeded_remainder",
            "discovery_draw": (
                "per cell, default_rng(SeedSequence([discovery_seed, 1, cell])).choice(members, take, "
                "replace=False); the cells taking the extra respondent come from "
                "default_rng(SeedSequence([discovery_seed, 0]))"
            ),
            "cv_seed": self.cv_seed,
            "folds": self.folds,
            "repeats": self.repeats,
            "confidence_level": self.confidence_level,
            "group_selection": "discovery_sample_only_frozen_before_any_policy_fit",
            "candidate_groups": list(self.candidate_groups),
            "n_selected_subgroups": self.n_selected_subgroups,
            "n_low_correlation": self.n_low_correlation,
            "n_median_correlation": self.n_median_correlation,
            "n_high_correlation": self.n_high_correlation,
            "min_discovery_support": self.min_discovery_support,
            "selection_statistic": self.selection_statistic,
            "selection_rule": SELECTION_RULE,
            "selection_tie_break": SELECTION_TIE_BREAK,
            "selection_undefined": SELECTION_UNDEFINED,
            "selection_complements": SELECTION_COMPLEMENTS,
            "groups": list(self.groups),
            "algorithm": self.algorithm,
            "arms": list(self.arms),
            "learner_seeds": list(self.learner_seeds),
            "protocol": self.protocol,
            "comparisons_per_respondent": self.comparisons_per_respondent,
            "reference": self.reference,
            "tau": self.tau,
            "output_averaging": self.output_averaging,
            "tie_rule": self.tie_rule,
            "beta": self.route.label_beta,
            "estimand": ESTIMAND_FORMULA,
            "inference": "nadeau_bengio_corrected_repeated_cv_t",
            # Emitted only when present, so the two frozen routes' digests are
            # exactly what they were.
            **({"changes_from_source_route": self.changes_from_source_route}
               if self.changes_from_source_route else {}),
        }

    def digest(self) -> str:
        payload = json.dumps(self.as_mapping(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_persona_config(path: Path) -> PersonaCrossfitConfig:
    return PersonaCrossfitConfig.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def tiny_config(config: PersonaCrossfitConfig, *, seeds: int = 2, repeats: int = 2,
                folds: int = 2) -> PersonaCrossfitConfig:
    """A reduced design for tests only; no production path calls this."""

    return replace(
        config, learner_seeds=config.learner_seeds[:seeds], repeats=repeats, folds=folds
    ).validate()


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


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def load_persona_inputs(
    config: PersonaCrossfitConfig,
    *,
    project_root: Path,
    group_names: Sequence[str] | None = None,
    expected_data_sha256: str | None = None,
    expected_study_sha256: str | None = None,
) -> CrossfitInputs:
    """Build the panel, reserve the stratified discovery sample, cut the folds.

    ``group_names`` defaults to the whole candidate universe, which is what the
    discovery stage needs; once the family is selected the caller passes the
    frozen family so the loaded dataset carries exactly those columns in that
    order.  The source CSV is never modified: the reservation is a set of row
    positions held in memory and written to the design manifest.
    """

    data_path = _resolve(project_root, config.data_path)
    study_path = _resolve(project_root, config.study_path)
    for path, label in ((data_path, "persona results CSV"), (study_path, "persona study file")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    data_sha256 = _sha256_file(data_path)
    study_sha256 = _sha256_file(study_path)
    for actual, expected, label in (
        (data_sha256, expected_data_sha256, "persona results CSV"),
        (study_sha256, expected_study_sha256, "persona study file"),
    ):
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"the {label} does not match the frozen snapshot this design was written against.\n"
                f"  expected sha256 {expected}\n  actual   sha256 {actual}\n"
                "The discovery draw and the panel alignment are only meaningful against those exact "
                "bytes in that exact row order; nothing is analysed against a different snapshot."
            )
    requested = tuple(group_names) if group_names is not None else config.candidate_groups
    dataset = load_dense_dataset(config.dataset, data_path, requested, config.dataset_options)
    # The earliest point at which the declared protocol string and the actual
    # collection can be compared, and before anything is selected or fitted.
    config.check_answer_count(dataset.action_names)
    ids = np.asarray(dataset.respondent_ids).astype(str)
    if len(set(ids)) != len(ids):
        raise RuntimeError("the constructed pseudo-respondents do not have unique IDs")
    reservation = stratified_discovery_reservation(
        np.asarray(dataset.auxiliary["cell_index"], dtype=np.int64),
        fraction=config.discovery_fraction,
        seed=config.discovery_seed,
    )
    plan = repeated_fold_plan(
        reservation.n_analysis, repeats=config.repeats, folds=config.folds, seed=config.cv_seed
    )
    return CrossfitInputs(
        dataset=dataset, reservation=reservation, plan=plan,
        data_path=data_path, data_sha256=f"{data_sha256}+study:{study_sha256}",
    )


# ------------------------------------------------------------ group selection


def margin_vector(utilities: np.ndarray) -> np.ndarray:
    """The ``C(A,2)`` mean pairwise margins ``mean U(a) - mean U(b)`` for ``a < b``.

    Pairs are enumerated in ascending ``(a, b)`` order, the same order for every
    group, so the correlation below compares like with like.
    """

    values = np.asarray(utilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("margins need a nonempty (n, A>=2) utility matrix")
    means = values.mean(axis=0)
    n_actions = values.shape[1]
    return np.asarray(
        [means[a] - means[b] for a in range(n_actions) for b in range(a + 1, n_actions)],
        dtype=np.float64,
    )


def margin_correlation(group_margins: np.ndarray, population_margins: np.ndarray) -> float | None:
    """Pearson correlation of two margin vectors, or ``None`` when undefined.

    It is undefined exactly when one of the vectors is constant (zero variance,
    so no correlation exists) or when a value is nonfinite.  The predeclared
    handling is to make that candidate **ineligible**: it is never selected and
    is recorded as undefined, rather than being coerced to zero, dropped
    silently or replaced by another statistic.
    """

    a = np.asarray(group_margins, dtype=np.float64)
    b = np.asarray(population_margins, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("margin vectors must have the same shape and at least two entries")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    if np.isclose(a.std(), 0.0) or np.isclose(b.std(), 0.0):
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return None if not math.isfinite(value) else value


@dataclass(frozen=True)
class GroupSelection:
    """The discovery-only selection and the full ranking that produced it."""

    selected: tuple[str, ...]
    low_correlation: tuple[str, ...]
    median_correlation: tuple[str, ...]
    high_correlation: tuple[str, ...]
    candidates: tuple[dict[str, Any], ...]
    population_margins: tuple[float, ...]
    action_names: tuple[str, ...]
    n_discovery: int

    def _sign_note(self) -> str:
        """What the observed correlation range actually was, with its sign.

        Derived rather than asserted: on the first persona collection every
        candidate correlated positively with the discovery population, but that
        is a property of a collection and not of the rule, so the note reports
        the range this run saw instead of repeating a claim from another one.
        """

        values = [row["correlation"] for row in self.candidates
                  if row["group"] != UNIVERSAL and row["correlation"] is not None]
        if not values:
            observed = "no candidate had a defined correlation"
        else:
            negative = sum(1 for value in values if value < 0.0)
            observed = (
                f"the observed range is {min(values):.6f} to {max(values):.6f}, and "
                + ("every candidate correlates positively with the population"
                   if negative == 0 else
                   f"{negative} of {len(values)} candidates correlate negatively with it")
            )
        return (
            "'low' is the low end of the observed correlation range, not by itself a claim about "
            f"sign; each candidate's correlation is reported with its sign. Here {observed}. The "
            "ordinary slot is the median rather than the maximum because a maximum-correlation "
            "group is close to indistinguishable from universal"
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "selected_groups": list(self.selected),
            "selected_low_correlation": list(self.low_correlation),
            "selected_median_correlation": list(self.median_correlation),
            "selected_high_correlation": list(self.high_correlation),
            "n_discovery_respondents": self.n_discovery,
            "n_answers": len(self.action_names),
            "n_pairwise_margins": n_pairwise_margins(len(self.action_names)),
            "statistic": selection_statistic(len(self.action_names)),
            "rule": SELECTION_RULE,
            "rule_intent": (
                "four groups from the low end plus one ordinary median group, so the family is not "
                "composed entirely of outliers and the multigroup arm is also asked to serve a group "
                "that behaves like the population"
            ),
            "low_end_sign_note": self._sign_note(),
            "discovery_only": (
                "every number used to rank, filter and select comes from the reserved discovery "
                "sample; no analysis respondent, fold or fitted policy is read"
            ),
            "not_tuned_to_the_result": (
                "the slot counts are fixed by the analysis configuration"
            ),
            "tie_break": SELECTION_TIE_BREAK,
            "undefined_handling": SELECTION_UNDEFINED,
            "complements": SELECTION_COMPLEMENTS,
            "universal": "always included and never counted as a selected subgroup",
            "pair_order": [
                f"{self.action_names[a]}>{self.action_names[b]}"
                for a in range(len(self.action_names))
                for b in range(a + 1, len(self.action_names))
            ],
            "discovery_population_margins": list(self.population_margins),
            "candidates": [dict(row) for row in self.candidates],
            "selection_frozen": (
                "run once on the discovery sample before any policy fit; not repeated inside a CV "
                "repeat and never recomputed on analysis respondents"
            ),
            "uncertainty_note": (
                "the headline interval is conditional on this family and carries no uncertainty from "
                "having selected it"
            ),
        }


def select_groups(
    utilities: np.ndarray,
    memberships: np.ndarray,
    candidate_names: Sequence[str],
    *,
    n_low: int,
    n_median: int = 0,
    n_high: int = 0,
    min_support: int,
    action_names: Sequence[str],
) -> GroupSelection:
    """Select subgroups by where they sit in the margin-correlation ranking.

    ``utilities`` and ``memberships`` are the **discovery** respondents only:
    every number this function reads comes from the reserved 10% sample, and
    nothing about the ranking, the eligibility or the slots looks at an analysis
    respondent, a cross-validation fold or any fitted policy.

    For each candidate subgroup the ``choose(A, 2)`` mean pairwise margins are
    correlated with the discovery population's -- ten margins over v1's five
    answers, fifteen over v2's six, and the code counts rather than assumes.
    The family then takes

    * ``n_low`` candidates with the *lowest* correlation -- the groups whose
      preference ordering departs most from the population's;
    * ``n_median`` ordinary candidates from the *middle* of what remains (the
      lower median in ascending correlation), so the family is not composed
      entirely of outliers;
    * ``n_high`` candidates with the *highest* correlation.

    "Low" means the low end of the observed range and is not by itself a claim
    about sign; each candidate's correlation is reported with its sign, and the
    selection record states the range actually observed.  A "high" slot would
    pick a group nearly indistinguishable from ``universal`` whenever the
    candidates correlate strongly with the population, which is why the ordinary
    slot is the median rather than the maximum.

    The rules this function applies are all predeclared: ``universal`` is always
    included and never competes; a candidate with fewer than ``min_support``
    discovery members, or an undefined correlation, is ineligible and can never
    be selected; the slots are filled low, then median, then high, a candidate
    can fill only one slot, and ties break by correlation (ascending for the low
    and median slots, descending for the high slots), then discovery support
    descending, then the candidate's position in the frozen candidate list; and
    two complementary groups may both be selected.  Too few eligible candidates
    is an error, not an occasion to select fewer groups.
    """

    values = np.asarray(utilities, dtype=np.float64)
    members = np.asarray(memberships)
    names = tuple(str(name) for name in candidate_names)
    if members.shape != (values.shape[0], len(names)):
        raise ValueError("memberships must be (n_discovery, n_candidates)")
    if not names or names[0] != UNIVERSAL:
        raise ValueError("the candidate list must start with universal")
    population = margin_vector(values)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        support = int(np.asarray(members[:, index]).astype(bool).sum())
        if name == UNIVERSAL:
            rows.append({
                "group": name, "candidate_index": index, "discovery_support": support,
                "margins": list(population), "correlation": 1.0, "eligible": False,
                "status": "universal_always_included", "selected": True, "rank": None,
            })
            continue
        eligible = support >= int(min_support)
        margins = margin_vector(values[np.asarray(members[:, index]).astype(bool)]) if support else None
        correlation = None if margins is None else margin_correlation(margins, population)
        if not eligible:
            status = f"ineligible_support_below_{int(min_support)}"
        elif correlation is None:
            status = "ineligible_undefined_correlation"
            eligible = False
        else:
            status = "eligible"
        rows.append({
            "group": name, "candidate_index": index, "discovery_support": support,
            "margins": None if margins is None else list(margins),
            "correlation": correlation, "eligible": eligible, "status": status,
            "selected": False, "rank": None,
        })

    n_low, n_median, n_high = int(n_low), int(n_median), int(n_high)
    wanted = n_low + n_median + n_high
    contenders = [row for row in rows if row["eligible"]]
    if len(contenders) < wanted:
        raise RuntimeError(
            f"only {len(contenders)} of {len(names) - 1} candidate subgroups are eligible, but the "
            f"design selects {n_low} + {n_median} + {n_high}; the family is not quietly shrunk and "
            "no ineligible candidate is substituted"
        )
    ascending = sorted(
        contenders,
        key=lambda row: (row["correlation"], -row["discovery_support"], row["candidate_index"]),
    )
    for rank, row in enumerate(ascending):
        row["rank"] = rank
    # Slots are filled low, then median, then high, each from what is still
    # unclaimed, so no candidate can occupy two slots however the counts are set.
    low = ascending[:n_low]
    remaining = ascending[n_low:]
    median: list[dict[str, Any]] = []
    for _ in range(n_median):
        # The lower median of the remaining ascending order: an ordinary
        # candidate, neither the most nor the least population-like.
        median.append(remaining.pop((len(remaining) - 1) // 2))
    descending = sorted(
        remaining,
        key=lambda row: (-row["correlation"], -row["discovery_support"], row["candidate_index"]),
    )
    high = descending[:n_high]
    for group_rows, end in ((low, "low_correlation"), (median, "median_correlation"),
                            (high, "high_correlation")):
        for row in group_rows:
            row["selected"] = True
            row["selected_end"] = end
    for row in rows:
        row.setdefault("selected_end", "universal" if row["group"] == UNIVERSAL else None)
    selected = (
        (UNIVERSAL,)
        + tuple(row["group"] for row in low)
        + tuple(row["group"] for row in median)
        + tuple(row["group"] for row in high)
    )
    return GroupSelection(
        selected=selected,
        low_correlation=tuple(row["group"] for row in low),
        median_correlation=tuple(row["group"] for row in median),
        high_correlation=tuple(row["group"] for row in high),
        candidates=tuple(rows),
        population_margins=tuple(float(value) for value in population),
        action_names=tuple(str(name) for name in action_names),
        n_discovery=int(values.shape[0]),
    )


def select_groups_from_inputs(
    config: PersonaCrossfitConfig, inputs: CrossfitInputs
) -> GroupSelection:
    """Run :func:`select_groups` on the reserved discovery respondents only."""

    dataset = inputs.dataset
    if tuple(dataset.group_names) != config.candidate_groups:
        raise ValueError(
            "group selection needs the dataset loaded with the whole candidate universe, in order"
        )
    config.check_answer_count(dataset.action_names)
    rows = inputs.reservation.discovery_indices
    return select_groups(
        np.asarray(dataset.utilities)[rows],
        np.asarray(dataset.groups)[rows],
        config.candidate_groups,
        n_low=config.n_low_correlation,
        n_median=config.n_median_correlation,
        n_high=config.n_high_correlation,
        min_support=config.min_discovery_support,
        action_names=dataset.action_names,
    )


# ----------------------------------------------------------------- the design


def design_manifest(
    config: PersonaCrossfitConfig,
    inputs: CrossfitInputs,
    selection: GroupSelection,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """The compact, auditable record of what was built, reserved and selected."""

    reservation, plan = inputs.reservation, inputs.plan
    ids = np.asarray(inputs.dataset.respondent_ids).astype(str)
    cells = np.asarray(inputs.dataset.auxiliary["cell_bits"]).astype(str)
    split_rows = []
    for repeat, fold in plan.splits():
        train_pos, test_pos = plan.split_positions(repeat, fold)
        split_rows.append({
            "repeat": int(repeat), "fold": int(fold),
            "n_train": int(train_pos.size), "n_test": int(test_pos.size),
            "train_ids_sha256": sorted_id_sha256(ids[reservation.analysis_indices[train_pos]]),
            "test_ids_sha256": sorted_id_sha256(ids[reservation.analysis_indices[test_pos]]),
        })
    unique_cells = sorted(set(cells.tolist()))
    discovery_cells = cells[reservation.discovery_indices]
    analysis_cells = cells[reservation.analysis_indices]
    group_counts = {
        name: int(np.asarray(inputs.dataset.groups)[reservation.analysis_indices, index].sum())
        for index, name in enumerate(inputs.dataset.group_names)
    }
    panel = panel_record(
        _resolve(project_root, config.data_path), _resolve(project_root, config.study_path),
        config.panel_seed, inputs.dataset,
    )
    study = json.loads(_resolve(project_root, config.study_path).read_text(encoding="utf-8"))
    return {
        "protocol_id": config.route.protocol_id,
        "implementation_id": config.route.implementation_id,
        "config_sha256": config.digest(),
        "collection": {
            "study_id": str(study.get("study_id", "")),
            "answers": list(inputs.dataset.action_names),
            "n_answers": len(inputs.dataset.action_names),
            "n_pairwise_margins": n_pairwise_margins(len(inputs.dataset.action_names)),
            "demographic_order": panel["demographic_order"],
            "separately_elicited": (
                f"each of the {len(inputs.dataset.action_names)} answers was rated in its own API "
                "call; no multi-answer vector was ever jointly elicited"
            ),
        },
        "data": {
            "path": str(inputs.data_path),
            "sha256": inputs.data_sha256,
            "ratings": int(len(unique_cells) * len(inputs.dataset.action_names)
                           * panel["draws_per_cell_answer"]),
            "physical_file_modified": False,
            **panel,
        },
        "panel": {
            "pseudo_respondents": int(ids.size),
            "cells": len(unique_cells),
            "per_cell": panel["draws_per_cell_answer"],
            "permutation_seeds": permutation_table(
                config.panel_seed, unique_cells, inputs.dataset.action_names,
                panel["draws_per_cell_answer"],
            ),
            "independence_assumption": (
                "the answers were evaluated in separate API calls, so the aligned vectors assume "
                "independence across answer-specific rating draws within a cell"
            ),
            "n_answers": len(inputs.dataset.action_names),
            "not_individuals": (
                f"a pseudo-respondent is not {len(inputs.dataset.action_names)} ratings from one "
                "simulated individual, and a persona bitstring identifies a demographic cell rather "
                "than a person"
            ),
        },
        "discovery": {
            "seed": reservation.seed,
            "fraction": reservation.fraction,
            "rounding": "ceiling",
            "stratification": "balanced by persona cell; the remainder cells are drawn from the seed",
            "k": reservation.n_discovery,
            "n_analysis": reservation.n_analysis,
            "per_cell_discovery": {cell: int((discovery_cells == cell).sum()) for cell in unique_cells},
            "per_cell_analysis": {cell: int((analysis_cells == cell).sum()) for cell in unique_cells},
            "discovery_ids_sha256": sorted_id_sha256(ids[reservation.discovery_indices]),
            "analysis_ids_sha256": sorted_id_sha256(ids[reservation.analysis_indices]),
            "excluded_from": (
                "policy training, every cross-validation test fold, hyperparameter selection and the "
                "headline estimate"
            ),
        },
        "groups": {
            "selection": "selected on the discovery sample only, then frozen",
            "candidate_universe": list(config.candidate_groups),
            "selected": list(config.groups),
            "analysis_members": group_counts,
            "membership": (
                "a deterministic function of the persona bitstring; nothing is imputed or back-filled"
            ),
            **selection.as_mapping(),
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
        "arms": list(ARMS),
        "policy_fits": plan.repeats * plan.folds * len(config.learner_seeds) * len(ARMS),
    }


def validate_design(config: PersonaCrossfitConfig, inputs: CrossfitInputs) -> None:
    """Structural checks that must hold before any policy is fitted."""

    if not config.groups:
        raise RuntimeError("the group family has not been selected yet")
    config.check_answer_count(inputs.dataset.action_names)
    if tuple(inputs.dataset.group_names) != config.groups:
        raise RuntimeError("the dataset's group columns are not the frozen selected family, in order")
    reservation, plan = inputs.reservation, inputs.plan
    ids = np.asarray(inputs.dataset.respondent_ids).astype(str)
    discovery = set(ids[reservation.discovery_indices].tolist())
    analysis = set(ids[reservation.analysis_indices].tolist())
    if discovery & analysis:
        raise RuntimeError("the discovery sample overlaps the analysis complement")
    if len(discovery) + len(analysis) != len(ids):
        raise RuntimeError("the discovery reservation does not partition the pseudo-respondents")
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


# ------------------------------------------------- worker-process plumbing

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(config_mapping: Mapping[str, Any], project_root: str, output_dir: str,
                 expected_data_sha256: str | None, expected_study_sha256: str | None) -> None:
    """Rebuild the read-only inputs once per process (Windows spawns children).

    The worker receives the configuration **after** group selection, so it
    never repeats the selection and cannot select a different family: it loads
    the panel with exactly the frozen columns it is given.
    """

    if _WORKER_STATE.get("output_dir") == output_dir:
        return
    config = PersonaCrossfitConfig.from_mapping(config_mapping)
    if not config.groups:
        raise RuntimeError("a worker was started before the group family was frozen")
    inputs = load_persona_inputs(
        config, project_root=Path(project_root), group_names=config.groups,
        expected_data_sha256=expected_data_sha256, expected_study_sha256=expected_study_sha256,
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(config=config, inputs=inputs, output_dir=output_dir)


def _run_split_task(task: tuple[int, int]) -> dict[str, Any]:
    repeat, fold = task
    return run_split(
        _WORKER_STATE["config"], _WORKER_STATE["inputs"], Path(_WORKER_STATE["output_dir"]), *task
    )


# ---------------------------------------------------------------- reporting


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sentence(text: str) -> str:
    return (text[:1].upper() + text[1:] + ".") if text else ""


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "undefined"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{float(value):.{digits}f}"


def _fmt_p(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "undefined"
    value = float(value)
    return f"{value:.6f}" if value >= 1e-4 else f"{value:.3e}"


def write_summary(output_dir: Path, summary: Mapping[str, Any]) -> None:
    """``summary.json`` plus a readable ``SUMMARY.md`` for the route."""

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
        f"**Status:** {info['status']}"
        + ("" if info["status"] == "completed" else f" — {info.get('error', '')}"),
        "",
        *[f"> {line}" for line in route.standing_note],
        "",
        f"Started {info.get('started_at')}; finished {info.get('finished_at', 'n/a')}.",
        f"Python {info['package_versions'].get('python')}, workers={info.get('workers')}.",
        "",
    ]
    if design:
        panel, discovery, groups, cv = (
            design["panel"], design["discovery"], design["groups"], design["cross_validation"],
        )
        collection = design.get("collection", {})
        n_answers = collection.get("n_answers", panel.get("n_answers"))
        lines += [
            "## Panel", "",
            f"- {design['data']['ratings']} model-generated ratings -> **{panel['pseudo_respondents']}** "
            f"pseudo-respondents ({panel['cells']} cells x {n_answers} separately elicited answers x "
            f"{panel['per_cell']} draws).",
            f"- Collection `{collection.get('study_id', 'n/a')}`; "
            f"{_sentence(collection.get('demographic_order', {}).get('note', 'demographic-line position not recorded'))}",
            f"- Alignment: {design['data']['alignment']}; `panel_seed={design['data']['panel_seed']}`.",
            f"- Utilities: `{design['data']['utility_map']}`.",
            f"- {_sentence(panel['independence_assumption'])}",
            f"- {_sentence(panel['not_individuals'])}",
            "",
            "## Design", "",
            f"- Discovery reserved and excluded: **{discovery['k']}** of "
            f"{panel['pseudo_respondents']} (stratified by persona cell, ceiling of "
            f"{discovery['fraction']:.2f} n, seed {discovery['seed']}).",
            f"- Cross-validation population: **{discovery['n_analysis']}** pseudo-respondents.",
            f"- Discovery respondents are excluded from {discovery['excluded_from']}.",
            f"- Repeated CV: **{cv['repeats']} repeats x {cv['folds']} folds = {cv['n_splits']} splits**, "
            f"cv_seed {cv['seed']}.",
            f"- Candidate universe ({len(groups['candidate_universe'])}): "
            + ", ".join(f"`{name}`" for name in groups["candidate_universe"]),
            "- Selected on discovery only: "
            + ", ".join(
                f"`{name}` ({groups['analysis_members'][name]} analysis members)"
                for name in groups["selected"]
            ),
            "- Slots: "
            + "; ".join(
                f"{label} " + (", ".join(f"`{name}`" for name in groups[key]) or "none")
                for label, key in (
                    ("lowest correlation:", "selected_low_correlation"),
                    ("median correlation:", "selected_median_correlation"),
                    ("highest correlation:", "selected_high_correlation"),
                )
            )
            + ".",
            f"- Selection statistic: `{groups['statistic']}`, rule `{groups['rule']}`.",
            f"- {_sentence(groups['rule_intent'])}",
            f"- {_sentence(groups['low_end_sign_note'])}",
            f"- {_sentence(groups['discovery_only'])}",
            f"- {_sentence(groups['not_tuned_to_the_result'])}",
            f"- {_sentence(groups['selection_frozen'])}",
            "",
            "| candidate | discovery support | correlation with population margins | rank | status | "
            "selected (end) |",
            "|---|---:|---:|---:|---|---|",
        ]
        for row in sorted(
            groups["candidates"], key=lambda item: (item["rank"] is None, item["rank"])
        ):
            end = row.get("selected_end")
            lines.append(
                f"| `{row['group']}` | {row['discovery_support']} | "
                f"{_fmt(row['correlation'], 6)} | "
                f"{'n/a' if row['rank'] is None else row['rank']} | {row['status']} | "
                + (f"yes ({end})" if row["selected"] and end != "universal" else
                   ("always included" if end == "universal" else "no"))
                + " |"
            )
        lines += ["", f"_{_sentence(groups['uncertainty_note'])}_", ""]
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
            if not learner["shared_comparison_log_across_arms"]:
                lines += [
                    "These numbers may be reported **beside "
                    "`persona-repeated-cv-v2-fixed-six-answer`'s** as a *same design, different feedback",
                    "protocol* sensitivity: that route is this design with a uniform passive log and",
                    "deterministic labels, over the same frozen collection, and the two differ in exactly",
                    "those two things. They must **not** be compared with `persona-repeated-cv-v1`, which is",
                    "a different collection. Note that active sampling removes the shared comparison log, so",
                    "the between-arm pairing here is weaker than the passive route's.", "",
                ]
    if inference:
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
            f"There are **{inference['n_splits']} inferential scores**, not "
            f"{inference['n_splits'] * inference['n_learner_seeds']}: the seed-level contrasts are "
            "averaged within a split before the test, because they are not independent observations.",
            "",
            _sentence(inference["estimand_note"]),
            "",
            _sentence(inference["method_note"]),
            "",
        ]
        if inference["status"] != NB_STATUS_OK:
            lines += [
                f"**Inference status: `{inference['status']}`.** The test and interval are undefined.", "",
            ]
        if inference.get("naive_ci_lower") is not None:
            lines += [
                f"Diagnostic only, **not for reporting**: the uncorrected interval that treats the "
                f"{inference['n_splits']} overlapping split scores as independent is "
                f"[{_fmt(inference['naive_ci_lower'])}, {_fmt(inference['naive_ci_upper'])}] "
                f"(SE {_fmt(inference['naive_se'])}). It understates the uncertainty.", "",
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
                    f"{interval} | {_fmt_p(row['p_value_two_sided'])} | {_fmt_p(row['p_value_holm'])} | "
                    f"{holm} |"
                )
            lines += [
                "", _sentence(gaps["interval_label"]), "",
                _sentence(gaps["multiplicity_label"]), "",
                _sentence(gaps["companion_note"]), "",
            ]
        lines += [
            "## What this interval is conditional on", "",
            "- the discovery-selected group family (its selection uncertainty is **not** included);",
            "- the artificial answer-rating alignment and its `panel_seed`"
            + (f" ({design['data']['panel_seed']})" if design else "") + ";",
            "- the evaluator model and the prompt;",
            "- the collection's demographic-line condition;",
            "- the candidate-group universe;",
            "- the fixed hyperparameters.",
            "",
            "It does not establish anything about human demographic preferences. Because a "
            "pseudo-respondent's scores were never jointly elicited, read it alongside the "
            "prespecified sensitivity analysis over several `panel_seed` values (separate runs, "
            "separate output directories).",
            "",
        ]
        order = (design.get("collection", {}) or {}).get("demographic_order", {})
        if order.get("fixed"):
            lines += [
                "Demographic-line position was **held constant** across this collection, so it is a "
                "fixed condition of the design rather than something this analysis varies: its causal "
                "effect is not identified here. A comparison with a collection that varied it would "
                "also differ in prompt wording, answer wording, answer count, negative controls, draws "
                "per cell/answer and the random API samples, and is therefore descriptive only.",
                "",
            ]
    lines += ["## Where the outputs are", ""]
    for label, relative in summary.get("outputs", {}).items():
        lines.append(f"- {label}: `{relative}`")
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class PersonaCrossfitResult:
    exit_code: int
    output_dir: Path
    summary: dict[str, Any]


def run_persona_crossfit(
    *,
    project_root: Path,
    output_dir: Path,
    config_path: Path | None = None,
    config: PersonaCrossfitConfig | None = None,
    route: PersonaRoute | None = None,
    panel_seed: int | None = None,
    expected_data_sha256: str | None = None,
    expected_study_sha256: str | None = None,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> PersonaCrossfitResult:
    """Run the whole route into ``output_dir``, resuming saved splits.

    ``panel_seed`` overrides the configured one, which is how the prespecified
    panel-construction sensitivity analysis is run: one separate output
    directory per seed.  A different panel seed changes the configuration
    digest, so a saved split from another seed is recomputed rather than reused.
    """

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config is None:
        config = load_persona_config(
            config_path or project_root / "configs" / (route or ROUTE_V1).config_name
        )
    if route is not None and config.route is not route:
        raise ValueError(
            f"this run was started as {route.protocol_id!r} but the configuration declares "
            f"{config.route.protocol_id!r}"
        )
    if panel_seed is not None and int(panel_seed) != config.panel_seed:
        config = replace(config, panel_seed=int(panel_seed)).validate()

    try:
        from coffee_expanded.manifests import runtime_versions

        versions = runtime_versions()
    except Exception:
        import sys as _sys

        versions = {"python": _sys.version.split()[0]}

    info: dict[str, Any] = {
        "protocol_id": config.route.protocol_id, "mode": config.route.mode,
        "reportable": True, "status": "running",
        "implementation_id": config.route.implementation_id,
        "panel_seed": config.panel_seed,
        "prespecified_config_sha256": config.digest(),
        "started_at": _now(), "workers": int(workers), "package_versions": versions,
    }
    summary: dict[str, Any] = {"run_info": info, "outputs": {
        "run record": "run_info.json", "log": "run.log",
        "design manifest": "design/design.json",
        "group selection": "design/group_selection.json",
        "per-split results": "splits/",
        "tables": "reduce/ (split_scores.csv, seed_scores.csv, group_values.csv, inference.json)",
        "per-group paired gaps": "reduce/group_gaps.csv, reduce/group_gaps.json",
        "per-group distortion levels by arm": (
            "reduce/group_levels.csv, reduce/group_levels.json, reduce/group_levels.png"
        ),
        **({"per-arm drawn comparison logs":
            "splits/rNN_fMM/comparisons.npz (first, second, sign, propensity)"}
           if config.route.active_feedback else {}),
    }}
    info_path = output_dir / "run_info.json"

    def save_info() -> None:
        info_path.write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")

    save_info()
    exit_code = 0
    try:
        log(f"persona: building the panel (panel_seed={config.panel_seed}) and reserving the "
            f"{config.discovery_fraction:.0%} stratified discovery sample")
        discovery_inputs = load_persona_inputs(
            config, project_root=project_root, group_names=config.candidate_groups,
            expected_data_sha256=expected_data_sha256, expected_study_sha256=expected_study_sha256,
        )
        log(f"persona: selecting {config.n_selected_subgroups} subgroups from "
            f"{len(config.candidate_groups) - 1} candidates, using the "
            f"{discovery_inputs.reservation.n_discovery} discovery respondents only")
        selection = select_groups_from_inputs(config, discovery_inputs)
        _write_json(output_dir / "design" / "group_selection.json", selection.as_mapping())
        config = config.with_groups(selection.selected)
        info["selected_groups"] = list(config.groups)
        info["analysis_config_sha256"] = config.digest()
        log("persona: group family frozen: " + ", ".join(config.groups))

        inputs = load_persona_inputs(
            config, project_root=project_root, group_names=config.groups,
            expected_data_sha256=expected_data_sha256, expected_study_sha256=expected_study_sha256,
        )
        if not np.array_equal(
            inputs.reservation.discovery_indices, discovery_inputs.reservation.discovery_indices
        ):
            raise AssertionError("the discovery reservation changed when the family was frozen")
        validate_design(config, inputs)
        manifest = design_manifest(config, inputs, selection, project_root=project_root)
        _write_json(output_dir / "design" / "design.json", manifest)
        np.savez_compressed(
            output_dir / "design" / "discovery.npz",
            discovery_indices=inputs.reservation.discovery_indices,
            analysis_indices=inputs.reservation.analysis_indices,
            draw_order=inputs.reservation.draw_order,
            pseudo_respondent_ids=np.asarray(inputs.dataset.respondent_ids).astype(str),
            cell_index=np.asarray(inputs.dataset.auxiliary["cell_index"], dtype=np.int64),
            fold_of=inputs.plan.fold_of,
        )
        info["design"] = {
            "pseudo_respondents": manifest["panel"]["pseudo_respondents"],
            "discovery_k": manifest["discovery"]["k"],
            "n_analysis": manifest["discovery"]["n_analysis"],
            "n_splits": manifest["cross_validation"]["n_splits"],
            "learner_seeds": len(config.learner_seeds),
            "policy_fits": manifest["policy_fits"],
        }
        summary["design"] = manifest
        save_info()
        log(f"persona: {manifest['discovery']['k']} discovery respondents excluded; "
            f"{manifest['discovery']['n_analysis']} remain for "
            f"{manifest['cross_validation']['n_splits']} splits ({manifest['policy_fits']} policy fits)")

        tasks = [(repeat, fold) for repeat, fold in inputs.plan.splits()]
        worker_args = (
            config.as_mapping(), str(project_root), str(output_dir),
            expected_data_sha256, expected_study_sha256,
        )
        _init_worker(*worker_args)
        _WORKER_STATE.update(config=config, inputs=inputs, output_dir=str(output_dir))
        started = time.perf_counter()

        def on_progress(done: int, total: int, task: tuple[int, int], seconds: float) -> None:
            log(f"persona split repeat={task[0]} fold={task[1]} done in {seconds:.1f} s ({done}/{total})")

        split_summaries, report = ordered_map(
            _run_split_task, tasks, workers=int(workers),
            initializer=_init_worker, initargs=worker_args, on_progress=on_progress, log=log,
        )
        info["splits_seconds"] = round(time.perf_counter() - started, 2)
        info["splits_reused"] = int(sum(1 for row in split_summaries if row.get("reused")))
        info["parallelism"] = report.summary()
        log(f"persona: {len(split_summaries)} splits complete ({info['splits_reused']} reused); "
            f"{report.summary()}")
        save_info()

        log("persona: reducing to split scores and computing the corrected interval")
        inference = reduce_splits(config, output_dir, split_summaries)
        summary["inference"] = inference
        info["inference_status"] = inference["status"]
        info["estimate"] = inference["estimate"]
        if inference["status"] != NB_STATUS_OK:
            log(f"persona: inference status {inference['status']}; test and interval are undefined")
        else:
            log(f"persona: estimate {inference['estimate']:.6f}, corrected SE "
                f"{inference['corrected_se']:.6f}, 95% CI [{inference['ci_lower']:.6f}, "
                f"{inference['ci_upper']:.6f}], p={inference['p_value_two_sided']:.6g}")
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
        return PersonaCrossfitResult(exit_code, output_dir, summary)
    save_info()
    write_summary(output_dir, summary)
    log(f"summary: {output_dir / 'SUMMARY.md'}")
    return PersonaCrossfitResult(exit_code, output_dir, summary)
