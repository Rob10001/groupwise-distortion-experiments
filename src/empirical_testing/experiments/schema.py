"""The suite reduction output schema.

Everything suite reports is written through these column sets, so the reduction
that produces a table and the figure that consumes it cannot disagree about
what a column means.  Each table carries the method's **training protocol**
explicitly, because the whole point of the study is that two columns printed
side by side were not trained the same way.

Conventions that hold in every table:

* ``learner_tau`` is what the tau ablation varies.  ``evaluation_tau`` is
  ``0.5`` in every row of the ``fixed_0.5`` regime, and equal to ``learner_tau``
  in every row of the ``moving`` regime.  The two regimes are **different
  estimands**: they are written into separate directories, every row carries
  ``evaluation_regime``, and their rows are never mixed in one table or one
  figure.
* ``beta`` serializes infinity as the string ``"inf"``, never as a float, and
  figures keep it off the finite log axis.
* ``protocol`` is always the protocol a method was **trained** under.  A result
  displayed beside the other protocol carries ``displayed_beside_protocol`` as
  well; the two are never conflated.
* Intervals are **marginal** Nadeau-Bengio corrected repeated-CV intervals.
  Holm adjusts *decisions*, not interval widths, so an interval excluding zero
  while its adjusted p exceeds 0.05 is not a contradiction.
"""

from __future__ import annotations

from typing import Mapping

#: Columns identifying which cell a row belongs to.  Present in every table.
CELL_COLUMNS: tuple[str, ...] = (
    "condition",              # "beta" or "tau"
    "evaluation_regime",      # "fixed_0.5" or "moving" -- a separate estimand
    "cell",                   # the cell label, e.g. "passive@inf" or "tau@0.35"
    "protocol",               # training protocol: "passive" or "active"
    "beta",                   # "0.5" ... "4", or "inf"
    "learner_tau",
    "evaluation_tau",
)

#: Columns identifying a method and how it was trained.
METHOD_COLUMNS: tuple[str, ...] = (
    "method",                 # internal identity
    "display_name",           # "Multigroup GLHF", ...
    "training_protocol",      # prose, stated in tables and figures
    "provenance",             # "fit" | "imported" | "shared" | "evaluate_only"
    "shared_with",            # the cell this result *is*, when shared
    "displayed_beside_protocol",
)

#: The paired-inference columns, identical wherever an estimate is reported.
INFERENCE_COLUMNS: tuple[str, ...] = (
    "estimate",
    "corrected_se",
    "df",                     # 99
    "ci_low", "ci_high",
    "t", "p_value",
    "p_holm",                 # adjusted decision, within the stated family
    "holm_family",            # what the adjustment ranged over
    "n_splits", "n_seeds",
    "variance_factor",
    "mean_test_train_ratio",
)

#: Provenance and diagnostics carried alongside every reduction.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "suite_id", "protocol_id", "implementation_id", "config_sha256",
    "source_design_id", "nlhf_baseline", "schema_version",
)

#: NLHF solver diagnostics, reported per cell.
SOLVER_COLUMNS: tuple[str, ...] = (
    "nlhf_selection", "nlhf_optimizer_calls", "nlhf_initialization",
    "nlhf_gradient_mode", "nlhf_solver_success_count", "nlhf_solver_failure_count",
    "nlhf_max_exploitability", "nlhf_exploitability_gate", "nlhf_max_kl",
)

#: The tables suite writes into ``reduce/``, and their columns.
TABLES: Mapping[str, tuple[str, ...]] = {
    # Four-method levels: worst-group distortion per condition and method.
    "method_levels": CELL_COLUMNS + METHOD_COLUMNS + (
        "worst_group_distortion", "worst_group_se", "welfare", "welfare_optimum",
    ) + PROVENANCE_COLUMNS,
    # The same, per group rather than maximized over groups.
    "method_group_levels": CELL_COLUMNS + METHOD_COLUMNS + (
        "group", "distortion", "distortion_se", "n_test_group",
    ) + PROVENANCE_COLUMNS,
    # All six pairwise contrasts within each condition, worst-group.
    "method_contrasts": CELL_COLUMNS + (
        "contrast", "minuend", "subtrahend",
        "minuend_training_protocol", "subtrahend_training_protocol",
        "positive_favors",
    ) + INFERENCE_COLUMNS + PROVENANCE_COLUMNS,
    # The per-group counterparts of those contrasts.
    "method_group_contrasts": CELL_COLUMNS + (
        "contrast", "minuend", "subtrahend", "group",
    ) + INFERENCE_COLUMNS + PROVENANCE_COLUMNS,
    # Active minus passive GLHF, at beta 4 and infinity only.
    "protocol_contrasts": CELL_COLUMNS + (
        "method", "display_name", "contrast", "positive_favors",
    ) + INFERENCE_COLUMNS + PROVENANCE_COLUMNS,
    "protocol_group_contrasts": CELL_COLUMNS + (
        "method", "display_name", "contrast", "group",
    ) + INFERENCE_COLUMNS + PROVENANCE_COLUMNS,
    # One row per split, the scores the t statistic is computed from.
    "split_scores": CELL_COLUMNS + (
        "repeat", "fold", "contrast", "split_score", "n_train", "n_test",
    ),
    "solver_diagnostics": CELL_COLUMNS + ("method",) + SOLVER_COLUMNS,
    "coverage": CELL_COLUMNS + METHOD_COLUMNS + (
        "expected_units", "completed_units", "complete",
    ),
}

#: The six pairwise method contrasts, in reporting order.  Holm ranges over
#: exactly these six within each condition.
PAIRWISE_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("pooled", "multigroup"),
    ("pooled_nlhf", "multigroup"),
    ("uniform", "multigroup"),
    ("pooled_nlhf", "pooled"),
    ("uniform", "pooled"),
    ("uniform", "pooled_nlhf"),
)

#: Holm family sizes, stated so a reader knows what an adjusted p ranged over.
HOLM_FAMILIES: Mapping[str, str] = {
    "worst_group": "the six pairwise method comparisons within one condition",
    "per_group": "the selected groups, separately for each contrast and condition",
}

SCHEMA_VERSION = "v10"

#: The figures suite writes, each as PDF, PNG and the CSV actually plotted.
FIGURES: Mapping[str, str] = {
    "beta_curves": (
        "Worst-group distortion vs beta for the four passive methods, with "
        "active GLHF shown only at beta 4 and infinity. Infinity is separated "
        "from the finite log axis."
    ),
    "beta_gaps": "Competitor minus multigroup-GLHF gaps across the beta grid.",
    "protocol_contrasts": (
        "Active minus passive GLHF, at beta 4 and infinity, for both GLHF methods."
    ),
    "tau_curves": (
        "Active GLHF tau curves with the passive-trained NLHF benchmark and the "
        "uniform reference. Tau is the learner radius; evaluation stays at 0.5."
    ),
    "tau_gaps": "Competitor minus multigroup-GLHF gaps across the tau grid.",
    "tau_group_panels": "Per-group panels of the tau curves.",
}
