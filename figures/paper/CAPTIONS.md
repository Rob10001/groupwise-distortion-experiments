# Empirical figures

Production uses reductions from 100 CV splits and 50 learner seeds per split. Group bars compare active-trained multigroup and pooled GLHF at β=∞, τL=τE=0.5 with passive-trained batch NLHF and the untrained uniform reference. The rightmost cluster is the mean of the within-run maximum over selected groups. Bars extend from one, where welfare equals that of each group's best fixed feasible lottery; lower distortion means higher welfare. Whiskers show approximate 95% confidence intervals for each mean, corrected for overlapping cross-validation folds (Nadeau–Bengio over split-level scores).

Radius panels use active-trained GLHF over six learner radii, passive-trained batch NLHF at each radius, and a fixed evaluation radius τE=0.5. The paired group gap figure shows active pooled minus active Multigroup GLHF at the anchor; positive values favor Multigroup GLHF. The combined beta figure plots the five passive settings for all four methods and active GLHF at β=4 and ∞ only; the x-axis is categorical. Shaded bands are marginal corrected 95% intervals.

The plotted_rows.csv sidecar contains the exact source rows used for new plots. This script reads completed results; it performs no training.
