# Persona panel-permutation analysis

**Status:** completed

> **Synthetic benchmark.** The ratings are model outputs produced under persona prompts by a
> single evaluator model on one question. A persona bitstring identifies a demographic *cell*,
> not a person, and nothing here observes or predicts human demographic preferences.
> The six answers were rated in separate API calls, so no six-answer vector was ever jointly
> elicited: pseudo-respondents are built by independently permuting each answer's draws within a
> cell under the recorded `panel_seed`, and that alignment is an artifact of the construction.
> The group family was selected on the reserved 10% discovery sample, so the reported interval is
> conditional on that selection and carries none of its uncertainty.
> **Demographic-line position was held constant** in every request of this collection. That makes
> line position a fixed condition, not a variable: its causal effect is *not* identified here.
> This collection differs from earlier Persona collections in prompt wording, answer wording,
> answer count, negative controls and draws per cell/answer, so a between-collection difference is descriptive
> only and attributes to nothing in particular. The reportable inferential comparison is pooled
> versus multigroup **within** this collection.

Started 2026-09-16T22:14:56+00:00; finished 2026-09-16T22:14:59+00:00.
Python 3.13.11, workers=4.

## Panel

- 9984 model-generated ratings -> **1664** pseudo-respondents (16 cells x 6 separately elicited answers x 104 draws).
- Persona collection; Demographic-line position was held constant at '00011011' in every request, so it is not a source of within-collection variation and its causal effect is not identifiable here.
- Alignment: within each cell, each answer's draws are independently permuted with default_rng(SeedSequence([panel_seed, cell_index, answer_index])) and read side by side; `panel_seed=20260917`.
- Utilities: `(score - 1) / 9`.
- The answers were evaluated in separate API calls, so the aligned vectors assume independence across answer-specific rating draws within a cell.
- A pseudo-respondent is not 6 ratings from one simulated individual, and a persona bitstring identifies a demographic cell rather than a person.

## Design

- Discovery reserved and excluded: **167** of 1664 (stratified by persona cell, ceiling of 0.10 n, seed 20260918).
- Cross-validation population: **1497** pseudo-respondents.
- Discovery respondents are excluded from policy training, every cross-validation test fold, hyperparameter selection and the headline estimate.
- Repeated CV: **10 repeats x 10 folds = 100 splits**, cv_seed 20260919.
- Candidate universe (9): `universal`, `age_22_32`, `age_62_72`, `education_high_school_or_less`, `education_university`, `technical_background_not_technical`, `technical_background_very_technical`, `writing_direct`, `writing_expressive`
- Selected on discovery only: `universal` (1497 analysis members), `technical_background_not_technical` (750 analysis members), `writing_expressive` (747 analysis members), `writing_direct` (750 analysis members), `technical_background_very_technical` (747 analysis members), `education_university` (748 analysis members)
- Slots: lowest correlation: `technical_background_not_technical`, `writing_expressive`, `writing_direct`, `technical_background_very_technical`; median correlation: `education_university`; highest correlation: none.
- Selection statistic: `pearson_correlation_of_fifteen_mean_pairwise_margins_with_discovery_population`, rule `lowest_four_plus_median_correlation`.
- Four groups from the low end plus one ordinary median group, so the family is not composed entirely of outliers and the multigroup arm is also asked to serve a group that behaves like the population.
- 'low' is the low end of the observed correlation range, not by itself a claim about sign; each candidate's correlation is reported with its sign. Here the observed range is -0.110205 to 0.997857, and 1 of 8 candidates correlate negatively with it. The ordinary slot is the median rather than the maximum because a maximum-correlation group is close to indistinguishable from universal.
- Every number used to rank, filter and select comes from the reserved discovery sample; no analysis respondent, fold or fitted policy is read.
- Run once on the discovery sample before any policy fit; not repeated inside a CV repeat and never recomputed on analysis respondents.

| candidate | discovery support | correlation with population margins | rank | status | selected (end) |
|---|---:|---:|---:|---|---|
| `technical_background_not_technical` | 82 | -0.110205 | 0 | eligible | yes (low_correlation) |
| `writing_expressive` | 85 | 0.497507 | 1 | eligible | yes (low_correlation) |
| `writing_direct` | 82 | 0.642240 | 2 | eligible | yes (low_correlation) |
| `technical_background_very_technical` | 85 | 0.772604 | 3 | eligible | yes (low_correlation) |
| `education_high_school_or_less` | 83 | 0.979749 | 4 | eligible | no |
| `education_university` | 84 | 0.985086 | 5 | eligible | yes (median_correlation) |
| `age_22_32` | 82 | 0.997674 | 6 | eligible | no |
| `age_62_72` | 85 | 0.997857 | 7 | eligible | no |
| `universal` | 167 | 1.000000 | n/a | universal_always_included | always included |

_The headline interval is conditional on this family and carries no uncertainty from having selected it._

## Headline

`mean_over_splits( mean_seed( max_group(D_pooled) - max_group(D_multigroup) ) )` (positive favors multigroup)

| estimate | corrected SE | df | 95% corrected interval | t | two-sided p | splits | seeds |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.2765 | 0.0047 | 99 | [0.2672, 0.2858] | 58.9616 | 6.353e-79 | 100 | 50 |

Split-score variance 0.000181559; mean test/train ratio 0.111111; variance factor 0.121111 = 1/100 + ratio.

There are **100 inferential scores**, not 5000: the seed-level contrasts are averaged within a split before the test, because they are not independent observations.

The mean of fold-specific worst-group contrasts; the max over groups and the welfare ratio are nonlinear, so this is a different estimand from a maximum or ratio taken after pooling all out-of-fold respondents.

The conventional corrected repeated-CV approximation, not an exact independence or coverage theorem; the correction widens the interval relative to treating the overlapping fold scores as independent, and coverage has not been established for this worst-group ratio statistic.

Diagnostic only, **not for reporting**: the uncorrected interval that treats the 100 overlapping split scores as independent is [0.2738, 0.2792] (SE 0.0013). It understates the uncertainty.

## Per-group paired gaps

`mean_over_splits( mean_seed( D_pooled[group] - D_multigroup[group] ) )` (positive favors multigroup)

| group | estimate | corrected SE | 95% corrected interval (marginal) | p | Holm p | Holm rejects |
|---|---:|---:|---|---:|---:|:--:|
| `universal` | 0.1704 | 0.0023 | [0.1659, 0.1750] | 9.955e-89 | 5.973e-88 | yes |
| `technical_background_not_technical` | 0.2223 | 0.0060 | [0.2104, 0.2343] | 1.045e-59 | 1.045e-59 | yes |
| `writing_expressive` | 0.1581 | 0.0041 | [0.1499, 0.1662] | 2.729e-61 | 5.457e-61 | yes |
| `writing_direct` | 0.2552 | 0.0041 | [0.2470, 0.2633] | 3.053e-81 | 1.527e-80 | yes |
| `technical_background_very_technical` | 0.2017 | 0.0052 | [0.1914, 0.2120] | 1.152e-61 | 3.455e-61 | yes |
| `education_university` | 0.1800 | 0.0033 | [0.1734, 0.1866] | 2.920e-75 | 1.168e-74 | yes |

Ordinary marginal Nadeau-Bengio corrected 95% intervals, one group at a time; these are NOT Holm-adjusted intervals and NOT simultaneous over the group family, because no multiplicity-adjusted interval was computed.

The Holm adjustment covers the 6 p-value decisions only; an interval that excludes zero while its Holm-adjusted p-value exceeds alpha is not a contradiction.

These per-group gaps drop the max over groups, so they are a companion to the headline worst-group estimand, not a decomposition of it.

## What this interval is conditional on

- the discovery-selected group family (its selection uncertainty is **not** included);
- the artificial answer-rating alignment and its `panel_seed` (20260917);
- the evaluator model and the prompt;
- the collection's demographic-line condition;
- the candidate-group universe;
- the fixed hyperparameters.

It does not establish anything about human demographic preferences. Because a pseudo-respondent's scores were never jointly elicited, read it alongside the sensitivity analysis over several `panel_seed` values (separate runs, separate output directories).

Demographic-line position was **held constant** across this collection, so it is a fixed condition of the design rather than something this analysis varies: its causal effect is not identified here. A comparison with a collection that varied it would also differ in prompt wording, answer wording, answer count, negative controls, draws per cell/answer and the random API samples, and is therefore descriptive only.

## Where the outputs are

- run record: `run_info.json`
- design manifest: `design/design.json`
- group selection: `design/group_selection.json`
- per-split results: `splits/`
- tables: `reduce/ (split_scores.csv, seed_scores.csv, group_values.csv, inference.json)`
- per-group paired gaps: `reduce/group_gaps.csv, reduce/group_gaps.json`
- per-group distortion levels by arm: `reduce/group_levels.csv, reduce/group_levels.json, reduce/group_levels.png`
