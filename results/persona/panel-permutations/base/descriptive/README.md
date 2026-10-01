# Persona collection: descriptive checks

Secondary diagnostics described in `data/persona/protocol.json`. They are not inputs to the headline estimand, group family, or inference.

Collection: `data/persona/` (paths are relative to the standalone package root).

Study: Persona synthetic evaluation.

Win rates are alignment-free: `P(a > b) + 0.5 P(a = b)` over all ordered pairs of a cell's independent draws, so they do not depend on `panel_seed` and are not built on a pseudo-respondent.

A negative control scoring highly is a finding about this synthetic evaluator, not a reason to drop the answer or rerun the collection. Nothing here is filtered or dropped.

- `ratings_by_answer.csv`
- `ratings_by_answer_cell.csv`
- `control_ranks_and_win_rates.csv`
- `correct_vs_incorrect.csv`
- `answer_contrasts.csv`
- `action_frequency.csv`
