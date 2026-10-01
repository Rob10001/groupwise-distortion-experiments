# Frozen comparison figures — Coffee

These plots use the frozen comparison reductions in `../comparison/reduce/`. Their current location is `results/coffee/figures/` within the standalone package.

Every figure is written three ways: **PDF** (for the manuscript), **PNG** (for quick viewing) and **CSV** — the CSV holds exactly the points that were drawn, so a reader can check the picture against the numbers.

## Evaluation regimes

- **`fixed_0.5`** — **Primary.** Every policy scored at evaluation radius 0.50, so all learner radii are held to one robustness standard and the vertical axis means the same thing at every point.

- **`moving`** — **Diagnostic, a different estimand.** Each policy scored at an evaluation radius equal to its own learner radius. The uncertainty set moves with the policy, so the uniform reference is *not* constant across radii. Do not read this as the cross-radius comparison.

The two regimes are **different estimands**. They live in separate reduction directories, every row carries `evaluation_regime`, and the regime is named in each figure's title, axis label, caption and filename. They agree exactly at the shared radius `tau = 0.5`, which is reused rather than recomputed.

## Figures

### `coffee_beta_curves_fixed_0.5`

Worst-group distortion against the label parameter beta, for the four methods. Passive GLHF at every beta; active GLHF only at beta 4 and infinity. Infinity is drawn in its own panel, off the finite log axis.

- files: `coffee_beta_curves_fixed_0.5.pdf`, `coffee_beta_curves_fixed_0.5.png`, `coffee_beta_curves_fixed_0.5.csv`
- evaluation regime: `fixed_0.5`
- drawn from: `../comparison/reduce/beta/method_levels.csv`
- contents: 24 plotted points; regime: fixed_0.5; 4 methods

### `coffee_beta_gaps_fixed_0.5`

Competitor-minus-groupwise-GLHF gaps across the beta grid, with corrected intervals.

- files: `coffee_beta_gaps_fixed_0.5.pdf`, `coffee_beta_gaps_fixed_0.5.png`, `coffee_beta_gaps_fixed_0.5.csv`
- evaluation regime: `fixed_0.5`
- drawn from: `../comparison/reduce/beta/method_contrasts.csv`
- contents: 21 plotted points; regime: fixed_0.5; 3 methods

### `coffee_protocol_contrasts_fixed_0.5`

Active-minus-passive GLHF at beta 4 and infinity, for both GLHF methods. NLHF has no active counterpart and so appears in no contrast.

- files: `coffee_protocol_contrasts_fixed_0.5.pdf`, `coffee_protocol_contrasts_fixed_0.5.png`, `coffee_protocol_contrasts_fixed_0.5.csv`
- evaluation regime: `fixed_0.5`
- drawn from: `../comparison/reduce/beta/protocol_contrasts.csv`
- contents: 4 plotted points; regime: fixed_0.5; 2 methods

### `coffee_tau_curves_fixed_0.5`

Worst-group distortion against the learner radius, active GLHF with the passive-trained NLHF benchmark and the uniform reference.

- files: `coffee_tau_curves_fixed_0.5.pdf`, `coffee_tau_curves_fixed_0.5.png`, `coffee_tau_curves_fixed_0.5.csv`
- evaluation regime: `fixed_0.5`
- drawn from: `../comparison/reduce/fixed_0.5/method_levels.csv`
- contents: 24 plotted points; regime: fixed_0.5; 4 methods

### `coffee_tau_curves_moving`

Worst-group distortion against the learner radius, active GLHF with the passive-trained NLHF benchmark and the uniform reference.

- files: `coffee_tau_curves_moving.pdf`, `coffee_tau_curves_moving.png`, `coffee_tau_curves_moving.csv`
- evaluation regime: `moving`
- drawn from: `../comparison/reduce/moving/method_levels.csv`
- contents: 24 plotted points; regime: moving; 4 methods

### `coffee_tau_gaps_fixed_0.5`

Competitor-minus-groupwise-GLHF gaps across the learner-radius grid.

- files: `coffee_tau_gaps_fixed_0.5.pdf`, `coffee_tau_gaps_fixed_0.5.png`, `coffee_tau_gaps_fixed_0.5.csv`
- evaluation regime: `fixed_0.5`
- drawn from: `../comparison/reduce/fixed_0.5/method_contrasts.csv`
- contents: 18 plotted points; regime: fixed_0.5; 3 methods

### `coffee_tau_gaps_moving`

Competitor-minus-groupwise-GLHF gaps across the learner-radius grid.

- files: `coffee_tau_gaps_moving.pdf`, `coffee_tau_gaps_moving.png`, `coffee_tau_gaps_moving.csv`
- evaluation regime: `moving`
- drawn from: `../comparison/reduce/moving/method_contrasts.csv`
- contents: 18 plotted points; regime: moving; 3 methods

### `coffee_tau_group_panels_fixed_0.5`

Per-group distortion against the learner radius, one panel per group, shared vertical axis. Drawn without intervals on purpose.

- files: `coffee_tau_group_panels_fixed_0.5.pdf`, `coffee_tau_group_panels_fixed_0.5.png`, `coffee_tau_group_panels_fixed_0.5.csv`
- evaluation regime: `fixed_0.5`
- drawn from: `../comparison/reduce/fixed_0.5/method_group_levels.csv`
- contents: 144 plotted points; regime: fixed_0.5; 4 methods; 6 groups

### `coffee_tau_group_panels_moving`

Per-group distortion against the learner radius, one panel per group, shared vertical axis. Drawn without intervals on purpose.

- files: `coffee_tau_group_panels_moving.pdf`, `coffee_tau_group_panels_moving.png`, `coffee_tau_group_panels_moving.csv`
- evaluation regime: `moving`
- drawn from: `../comparison/reduce/moving/method_group_levels.csv`
- contents: 144 plotted points; regime: moving; 4 methods; 6 groups

## Reading the intervals

Bands and bars are **Nadeau-Bengio corrected repeated-CV 95% intervals** over 100 split scores with `df = 99`, marginal per cell. They are *not* simultaneous over a grid. Holm adjusts the **decisions** of the six pairwise method comparisons within each condition; it does not narrow an interval, so an interval excluding zero while its adjusted p exceeds 0.05 is not a contradiction.

## Method labels

Every series names how it was **trained**. NLHF is passive-trained in every condition of this study, the active ones included: where it is shown beside an active column it is labelled the passive-trained benchmark, and it is never drawn or described as an active learner. The uniform reference is labelled as not fitted.
