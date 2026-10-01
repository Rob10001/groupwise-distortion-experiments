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

Started 2026-09-16T22:14:47+00:00; finished 2026-09-16T22:14:50+00:00.
Python 3.13.11, workers=4.

## Panel

- 9984 model-generated ratings -> **1664** pseudo-respondents (16 cells x 6 separately elicited answers x 104 draws).
- Persona collection; Demographic-line position was held constant at '00011011' in every request, so it is not a source of within-collection variation and its causal effect is not identifiable here.
- Alignment: within each cell, each answer's draws are independently permuted with default_rng(SeedSequence([panel_seed, cell_index, answer_index])) and read side by side; `panel_seed=20260916`.
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
- 'low' is the low end of the observed correlation range, not by itself a claim about sign; each candidate's correlation is reported with its sign. Here the observed range is -0.115209 to 0.996389, and 1 of 8 candidates correlate negatively with it. The ordinary slot is the median rather than the maximum because a maximum-correlation group is close to indistinguishable from universal.
- Every number used to rank, filter and select comes from the reserved discovery sample; no analysis respondent, fold or fitted policy is read.
- Run once on the discovery sample before any policy fit; not repeated inside a CV repeat and never recomputed on analysis respondents.

| candidate | discovery support | correlation with population margins | rank | status | selected (end) |
|---|---:|---:|---:|---|---|
| `technical_background_not_technical` | 82 | -0.115209 | 0 | eligible | yes (low_correlation) |
| `writing_expressive` | 85 | 0.510912 | 1 | eligible | yes (low_correlation) |
| `writing_direct` | 82 | 0.642752 | 2 | eligible | yes (low_correlation) |
| `technical_background_very_technical` | 85 | 0.777251 | 3 | eligible | yes (low_correlation) |
| `education_high_school_or_less` | 83 | 0.982818 | 4 | eligible | no |
| `education_university` | 84 | 0.987819 | 5 | eligible | yes (median_correlation) |
| `age_22_32` | 82 | 0.996328 | 6 | eligible | no |
| `age_62_72` | 85 | 0.996389 | 7 | eligible | no |
| `universal` | 167 | 1.000000 | n/a | universal_always_included | always included |

_The headline interval is conditional on this family and carries no uncertainty from having selected it._

## Headline

`mean_over_splits( mean_seed( max_group(D_pooled) - max_group(D_multigroup) ) )` (positive favors multigroup)

| estimate | corrected SE | df | 95% corrected interval | t | two-sided p | splits | seeds |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.2760 | 0.0048 | 99 | [0.2665, 0.2856] | 57.4909 | 7.219e-78 | 100 | 50 |

Split-score variance 0.000190361; mean test/train ratio 0.111111; variance factor 0.121111 = 1/100 + ratio.

There are **100 inferential scores**, not 5000: the seed-level contrasts are averaged within a split before the test, because they are not independent observations.

The mean of fold-specific worst-group contrasts; the max over groups and the welfare ratio are nonlinear, so this is a different estimand from a maximum or ratio taken after pooling all out-of-fold respondents.

The conventional corrected repeated-CV approximation, not an exact independence or coverage theorem; the correction widens the interval relative to treating the overlapping fold scores as independent, and coverage has not been established for this worst-group ratio statistic.

Diagnostic only, **not for reporting**: the uncorrected interval that treats the 100 overlapping split scores as independent is [0.2733, 0.2788] (SE 0.0014). It understates the uncertainty.

## Per-group paired gaps

`mean_over_splits( mean_seed( D_pooled[group] - D_multigroup[group] ) )` (positive favors multigroup)

| group | estimate | corrected SE | 95% corrected interval (marginal) | p | Holm p | Holm rejects |
|---|---:|---:|---|---:|---:|:--:|
| `universal` | 0.1704 | 0.0023 | [0.1659, 0.1749] | 3.841e-89 | 2.305e-88 | yes |
| `technical_background_not_technical` | 0.2227 | 0.0059 | [0.2109, 0.2345] | 2.788e-60 | 3.839e-60 | yes |
| `writing_expressive` | 0.1588 | 0.0042 | [0.1506, 0.1671] | 4.038e-61 | 1.211e-60 | yes |
| `writing_direct` | 0.2541 | 0.0050 | [0.2442, 0.2639] | 7.547e-73 | 3.019e-72 | yes |
| `technical_background_very_technical` | 0.2015 | 0.0054 | [0.1909, 0.2122] | 1.920e-60 | 3.839e-60 | yes |
| `education_university` | 0.1804 | 0.0031 | [0.1743, 0.1865] | 1.279e-78 | 6.394e-78 | yes |

Ordinary marginal Nadeau-Bengio corrected 95% intervals, one group at a time; these are NOT Holm-adjusted intervals and NOT simultaneous over the group family, because no multiplicity-adjusted interval was computed.

The Holm adjustment covers the 6 p-value decisions only; an interval that excludes zero while its Holm-adjusted p-value exceeds alpha is not a contradiction.

These per-group gaps drop the max over groups, so they are a companion to the headline worst-group estimand, not a decomposition of it.

## What this interval is conditional on

- the discovery-selected group family (its selection uncertainty is **not** included);
- the artificial answer-rating alignment and its `panel_seed` (20260916);
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
