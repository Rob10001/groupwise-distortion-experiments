# Coffee comparison suite

- status: **completed**
- reportable: True
- wall: 2650.6 s at 6 workers
- methods: multigroup, pooled, pooled_nlhf, uniform
- evaluation regimes: fixed_0.5, moving

## Fits

| family | count |
|---|---:|
| glhf_beta | 0 |
| glhf_tau | 50,000 |
| nlhf_beta | 25,000 |
| nlhf_tau | 25,000 |
| total | 100,000 |

## NLHF solver

- fits: 50,000
- optimizer reported non-convergence: 209 (verified inside the gate; not failed fits)
- max |exploitability|: 1.001e-05 against a gate of 2e-05
- every fit a single reference start: True

## Validation

- group_family_matches_imported: `True`
- group_family: `['universal', 'age:45+', 'gender:female', 'expertise:1-4', 'children:reported-1+', 'work:from-home']`
- comparator_exact_on_every_split: `True`
- splits_checked: `100`
- shared_tau_identity: `{'shared_tau': 0.5, 'regimes_agree_exactly_at_shared_tau': True, 'mismatched_splits': [], 'splits_compared': 100}`
- imported_source: `results/coffee/beta-source/source`

## Estimands

The `fixed_0.5` and `moving` reductions are **different estimands** and are
written into separate directories. `fixed_0.5` scores every policy at
evaluation tau = 0.5, so all radii are held to one robustness standard.
`moving` scores each policy at its own learner radius, so the evaluation
uncertainty set moves with the policy. Do not read a moving-tau curve as the
fixed-tau comparison.
