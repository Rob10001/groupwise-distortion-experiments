# Held-out distortion by sampling protocol and Bradley-Terry beta (Persona)

**Status:** completed

> **Sampling-protocol and label-model sensitivity sweep.**
> The panel, the 10% stratified discovery reservation, the
> discovery-selected group family, the fold plan, the learner seeds, `tau = 0.5` and the
> evaluation ball at 0.5 are all inherited from the frozen Persona six-answer design;
> only the sampling protocol and `beta` move.
> **Synthetic benchmark.** The ratings are model outputs produced under persona prompts on a
> collection of six separately elicited answers with demographic-line position held constant;
> a persona bitstring identifies a demographic cell, not a person.
> The per-cell intervals are marginal Nadeau-Bengio corrected repeated-CV intervals.
> They are **not simultaneous** over the grid, arms, or groups.

Started 2026-09-21T00:00:00+00:00; finished 2026-09-21T00:30:13+00:00.

## Design

- Source design: the frozen Persona six-answer design — same data, discovery reservation, cv seed, fold plan, group family, learner seeds, `tau = 0.5` and evaluation ball 0.5.
- Cells: `passive@0.5`, `passive@1`, `passive@2`, `passive@4`, `passive@inf`, `active@4`, `active@inf`
- Anchor: `passive@inf` — the source design's own feedback model, refitted here and compared array by array with the frozen run.
- Arms: `pooled`, `multigroup`, `uniform` (the uniform benchmark needs no fit).
- The respondent order is the passive log's own permutation and is shared by every cell, active included.

## Worst-group distortion by cell

`mean_over_splits( mean_seed( max_group(D) ) )` (lower is better)

| protocol | beta | pooled | 95% corrected interval | multigroup | 95% corrected interval | uniform |
|---|---|---:|---|---:|---|---:|
| `passive` | `0.5` | 1.65361 | [1.63465, 1.67258] | 1.53004 | [1.51482, 1.54525] | 1.73569 |
| `passive` | `1` | 1.52433 | [1.50843, 1.54022] | 1.32833 | [1.31779, 1.33888] | 1.73569 |
| `passive` | `2` | 1.40533 | [1.39220, 1.41846] | 1.16547 | [1.15999, 1.17095] | 1.73569 |
| `passive` | `4` | 1.34602 | [1.33724, 1.35481] | 1.10258 | [1.09905, 1.10610] | 1.73569 |
| `passive` | `inf` | 1.37463 | [1.36047, 1.38878] | 1.09858 | [1.08749, 1.10967] | 1.73569 |
| `active` | `4` | 1.31804 | [1.30827, 1.32780] | 1.05946 | [1.05631, 1.06260] | 1.73569 |
| `active` | `inf` | 1.35013 | [1.33877, 1.36150] | 1.06946 | [1.05673, 1.08220] | 1.73569 |

## Paired pooled minus multigroup gap by cell

`mean_over_splits( mean_seed( max_group(D_pooled) - max_group(D_multigroup) ) )` (positive favors multigroup)

| protocol | beta | estimate | corrected SE | 95% corrected interval | p |
|---|---|---:|---:|---|---:|
| `passive` | `0.5` | 0.12358 | 0.00436 | [0.11494, 0.13222] | 2.389e-49 |
| `passive` | `1` | 0.19599 | 0.00493 | [0.18621, 0.20578] | 1.121e-62 |
| `passive` | `2` | 0.23986 | 0.00493 | [0.23007, 0.24965] | 6.542e-71 |
| `passive` | `4` | 0.24345 | 0.00385 | [0.23581, 0.25109] | 7.668e-82 |
| `passive` | `inf` | 0.27605 | 0.00480 | [0.26652, 0.28557] | 7.219e-78 |
| `active` | `4` | 0.25858 | 0.00511 | [0.24843, 0.26873] | 1.592e-72 |
| `active` | `inf` | 0.28067 | 0.00538 | [0.26999, 0.29134] | 8.027e-74 |

## Active minus passive, paired at the split level

Positive means **higher** distortion under active sampling.

| beta | quantity | estimate | 95% corrected interval | p |
|---|---|---:|---|---:|
| `4` | `pooled` | -0.02799 | [-0.03824, -0.01773] | 4.288e-07 |
| `4` | `multigroup` | -0.04312 | [-0.04563, -0.04061] | 1.870e-56 |
| `4` | `uniform` | 0.00000 | undefined | n/a |
| `4` | `paired_gap` | 0.01513 | [0.00533, 0.02493] | 2.815e-03 |
| `inf` | `pooled` | -0.02449 | [-0.03663, -0.01236] | 1.195e-04 |
| `inf` | `multigroup` | -0.02912 | [-0.03274, -0.02549] | 4.199e-29 |
| `inf` | `uniform` | 0.00000 | undefined | n/a |
| `inf` | `paired_gap` | 0.00462 | [-0.00708, 0.01632] | 4.349e-01 |

## Label noise on each cell's own queried pairs

Flip rates count disagreement with `sign(u_i - u_j)` on **strictly** ordered pairs only; exact ties are fair coins even at `beta = inf` and never enter the denominator.

| protocol | beta | arm | mean flips | non-tied | ties | tie fraction | flip rate | E[error] |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `passive` | `0.5` | `pooled` | 451.2 | 1038.0 | 309.3 | 0.22955 | 0.43462 | 0.43468 |
| `passive` | `0.5` | `multigroup` | 451.2 | 1038.0 | 309.3 | 0.22955 | 0.43462 | 0.43468 |
| `passive` | `1` | `pooled` | 387.3 | 1038.0 | 309.3 | 0.22955 | 0.37312 | 0.37307 |
| `passive` | `1` | `multigroup` | 387.3 | 1038.0 | 309.3 | 0.22955 | 0.37312 | 0.37307 |
| `passive` | `2` | `pooled` | 281.1 | 1038.0 | 309.3 | 0.22955 | 0.27083 | 0.27075 |
| `passive` | `2` | `multigroup` | 281.1 | 1038.0 | 309.3 | 0.22955 | 0.27083 | 0.27075 |
| `passive` | `4` | `pooled` | 159.5 | 1038.0 | 309.3 | 0.22955 | 0.15369 | 0.15360 |
| `passive` | `4` | `multigroup` | 159.5 | 1038.0 | 309.3 | 0.22955 | 0.15369 | 0.15360 |
| `passive` | `inf` | `pooled` | 0.0 | 1038.0 | 309.3 | 0.22955 | 0.00000 | 0.00000 |
| `passive` | `inf` | `multigroup` | 0.0 | 1038.0 | 309.3 | 0.22955 | 0.00000 | 0.00000 |
| `active` | `4` | `pooled` | 174.6 | 1127.5 | 219.8 | 0.16312 | 0.15484 | 0.15475 |
| `active` | `4` | `multigroup` | 153.1 | 1098.8 | 248.5 | 0.18441 | 0.13934 | 0.13947 |
| `active` | `inf` | `pooled` | 0.0 | 1119.1 | 228.2 | 0.16939 | 0.00000 | 0.00000 |
| `active` | `inf` | `multigroup` | 0.0 | 1101.7 | 245.6 | 0.18230 | 0.00000 | 0.00000 |

## Anchor check (production gate)

- Frozen source run: `results/persona/panel-permutations/base/` (package-root-relative).
- Splits compared: 100; arrays compared: 900
- Agrees: **True**
- Machine-readable report: `reduce/anchor_check.json`

Every cell is a marginal Nadeau-Bengio corrected repeated-CV interval over the same splits: 100 split scores, each the learner-seed mean, variance factor 1/100 + mean(n_test/n_train) and df = 99. The intervals are NOT simultaneous over the cells, the protocols, the arms or the groups, and they are NOT the manuscript's respondent bootstrap bands. Holm adjustment applies to the per-group p-values within one cell and nowhere else.

Within a split and a learner seed every passive cell shares one log and one vector of common random numbers, so the passive beta curve is a pure label-noise effect. The active cells draw their pairs and their labels from the learner's own spawned generator, which is a different stream from the passive relabelling stream: the same outer seed therefore does NOT give the two protocols identical label coins, and no such alignment is claimed. What pairs the protocol contrast is the matched split and the matched learner seed.

The untrained uniform benchmark is not fitted and ignores the comparison log, so it is identical in every cell; the run asserts that per split rather than assuming it

Score bound used to normalize rewards, per protocol (the learner's existing, unmodified choice): `active` = K - 1 (one_expert._prepare_training_inputs, protocol='active'); `passive` = max(1 / propensity) = K (K - 1) / 2 for a uniform log (protocol='logged'). Read active minus passive as the effect of the implemented protocol, its native normalization included, not of the queried pairs alone.

## Where the outputs are

- run record: `run_info.json`
- design manifest: `design/design.json`
- group selection: `design/group_selection.json`
- per-split results: `splits/rNN_fMM/ (split.json, split.npz, comparisons.npz)`
- tables: `reduce/ (cell_worst_group, cell_group_levels, cell_gaps, split_scores, cell_label_noise, protocol_contrasts; CSV + JSON)`
- inference record: `reduce/inference.json`
- anchor check: `reduce/anchor_check.json`
