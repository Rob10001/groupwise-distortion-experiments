# Held-out distortion by sampling protocol and Bradley-Terry beta (Coffee)

**Status:** completed

> Uses the frozen Coffee analysis design. The per-cell intervals are marginal
> Nadeau-Bengio corrected repeated-CV intervals, not simultaneous over the grid,
> arms, or groups. All cells use the same discovery reservation and analysis population.

Started 2026-09-20T23:08:22+00:00; finished 2026-09-20T23:59:49+00:00.

## Design

- Source design: Coffee frozen analysis design — same data, discovery reservation, cv seed, fold plan, group family, learner seeds, `tau = 0.5` and evaluation ball 0.5.
- Cells: `passive@0.5`, `passive@1`, `passive@2`, `passive@4`, `passive@inf`, `active@4`, `active@inf`
- Anchor: `passive@inf` — the source design's own feedback model, refitted here and compared array by array with the frozen run.
- Arms: `pooled`, `multigroup`, `uniform` (the uniform benchmark needs no fit).
- The respondent order is the passive log's own permutation and is shared by every cell, active included.

## Worst-group distortion by cell

`mean_over_splits( mean_seed( max_group(D) ) )` (lower is better)

| protocol | beta | pooled | 95% corrected interval | multigroup | 95% corrected interval | uniform |
|---|---|---:|---|---:|---|---:|
| `passive` | `0.5` | 1.14298 | [1.12044, 1.16553] | 1.12470 | [1.10602, 1.14337] | 1.12168 |
| `passive` | `1` | 1.15927 | [1.13116, 1.18737] | 1.12281 | [1.10294, 1.14267] | 1.12168 |
| `passive` | `2` | 1.18778 | [1.15156, 1.22401] | 1.11636 | [1.09636, 1.13636] | 1.12168 |
| `passive` | `4` | 1.21971 | [1.17458, 1.26484] | 1.10451 | [1.08560, 1.12341] | 1.12168 |
| `passive` | `inf` | 1.24553 | [1.19400, 1.29707] | 1.09133 | [1.07509, 1.10757] | 1.12168 |
| `active` | `4` | 1.22696 | [1.17995, 1.27398] | 1.09326 | [1.07724, 1.10927] | 1.12168 |
| `active` | `inf` | 1.25278 | [1.19935, 1.30622] | 1.07955 | [1.06543, 1.09367] | 1.12168 |

## Paired pooled minus multigroup gap by cell

`mean_over_splits( mean_seed( max_group(D_pooled) - max_group(D_multigroup) ) )` (positive favors multigroup)

| protocol | beta | estimate | corrected SE | 95% corrected interval | p |
|---|---|---:|---:|---|---:|
| `passive` | `0.5` | 0.01829 | 0.00240 | [0.01352, 0.02305] | 1.565e-11 |
| `passive` | `1` | 0.03646 | 0.00476 | [0.02701, 0.04591] | 1.321e-11 |
| `passive` | `2` | 0.07143 | 0.00901 | [0.05356, 0.08930] | 3.363e-12 |
| `passive` | `4` | 0.11520 | 0.01449 | [0.08644, 0.14396] | 3.084e-12 |
| `passive` | `inf` | 0.15420 | 0.01953 | [0.11545, 0.19296] | 4.029e-12 |
| `active` | `4` | 0.13371 | 0.01707 | [0.09983, 0.16758] | 5.481e-12 |
| `active` | `inf` | 0.17324 | 0.02213 | [0.12933, 0.21715] | 5.591e-12 |

## Active minus passive, paired at the split level

Positive means **higher** distortion under active sampling.

| beta | quantity | estimate | 95% corrected interval | p |
|---|---|---:|---|---:|
| `4` | `pooled` | 0.00726 | [0.00251, 0.01200] | 3.089e-03 |
| `4` | `multigroup` | -0.01125 | [-0.01664, -0.00586] | 7.230e-05 |
| `4` | `uniform` | 0.00000 | undefined | n/a |
| `4` | `paired_gap` | 0.01851 | [0.01142, 0.02560] | 1.176e-06 |
| `inf` | `pooled` | 0.00725 | [0.00402, 0.01049] | 2.287e-05 |
| `inf` | `multigroup` | -0.01178 | [-0.01648, -0.00709] | 2.680e-06 |
| `inf` | `uniform` | 0.00000 | undefined | n/a |
| `inf` | `paired_gap` | 0.01904 | [0.01237, 0.02570] | 1.449e-07 |

## Label noise on each cell's own queried pairs

Flip rates count disagreement with `sign(u_i - u_j)` on **strictly** ordered pairs only; exact ties are fair coins even at `beta = inf` and never enter the denominator.

| protocol | beta | arm | mean flips | non-tied | ties | tie fraction | flip rate | E[error] |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `passive` | `0.5` | `pooled` | 1176.9 | 2645.5 | 386.6 | 0.12749 | 0.44486 | 0.44486 |
| `passive` | `0.5` | `multigroup` | 1176.9 | 2645.5 | 386.6 | 0.12749 | 0.44486 | 0.44486 |
| `passive` | `1` | `pooled` | 1037.3 | 2645.5 | 386.6 | 0.12749 | 0.39210 | 0.39204 |
| `passive` | `1` | `multigroup` | 1037.3 | 2645.5 | 386.6 | 0.12749 | 0.39210 | 0.39204 |
| `passive` | `2` | `pooled` | 792.8 | 2645.5 | 386.6 | 0.12749 | 0.29968 | 0.29957 |
| `passive` | `2` | `multigroup` | 792.8 | 2645.5 | 386.6 | 0.12749 | 0.29968 | 0.29957 |
| `passive` | `4` | `pooled` | 459.6 | 2645.5 | 386.6 | 0.12749 | 0.17372 | 0.17362 |
| `passive` | `4` | `multigroup` | 459.6 | 2645.5 | 386.6 | 0.12749 | 0.17372 | 0.17362 |
| `passive` | `inf` | `pooled` | 0.0 | 2645.5 | 386.6 | 0.12749 | 0.00000 | 0.00000 |
| `passive` | `inf` | `multigroup` | 0.0 | 2645.5 | 386.6 | 0.12749 | 0.00000 | 0.00000 |
| `active` | `4` | `pooled` | 452.6 | 2652.4 | 379.7 | 0.12522 | 0.17065 | 0.17069 |
| `active` | `4` | `multigroup` | 458.0 | 2648.3 | 383.8 | 0.12658 | 0.17293 | 0.17295 |
| `active` | `inf` | `pooled` | 0.0 | 2655.6 | 376.5 | 0.12417 | 0.00000 | 0.00000 |
| `active` | `inf` | `multigroup` | 0.0 | 2650.1 | 382.0 | 0.12599 | 0.00000 | 0.00000 |

## Anchor check (production gate)

- Frozen anchor: archived Coffee source run; its historical identity is recorded in `reduce/anchor_check.json`.
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
- per-split results: `splits/rNN_fMM/ (split.json, split.npz, comparisons.npz)`
- tables: `reduce/ (cell_worst_group, cell_group_levels, cell_gaps, split_scores, cell_label_noise, protocol_contrasts; CSV + JSON)`
- inference record: `reduce/inference.json`
- anchor check: `reduce/anchor_check.json`
