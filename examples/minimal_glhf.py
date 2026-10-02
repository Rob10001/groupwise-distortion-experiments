"""Learn from a toy passive comparison log, then serve new group memberships.

Run from the repository root: python examples/minimal_glhf.py
No datasets, API keys, file outputs, or cardinal-utility inputs are needed.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from empirical_testing.algorithms.span_adaptive import fit_groupwise_span_adaptive


def fit_example():
    rng = np.random.default_rng(7)
    n_users, n_alternatives = 240, 3

    # Shape (users, groups). The universal group covers every user; the two
    # other groups overlap. Column order must be the same at prediction time.
    groups = np.column_stack([
        np.ones(n_users, dtype=np.int8),
        rng.integers(0, 2, size=(n_users, 2), dtype=np.int8),
    ])

    # One uniformly selected unordered pair per user, independent of groups.
    # Alternatives are numbered 0, 1, 2. Both pair orientations are accepted.
    possible_pairs = np.array([[0, 1], [0, 2], [1, 2]])
    pairs = possible_pairs[rng.integers(0, len(possible_pairs), size=n_users)]
    propensities = np.full(n_users, 1.0 / len(possible_pairs))

    # Toy feedback: group 2 favors alternative 2; otherwise group 1 favors 1,
    # and the remaining users favor 0. Nonfavorite pairs are fair coin flips.
    # +1 means the first queried alternative won; -1 means the second won.
    favorite = np.where(groups[:, 2], 2, np.where(groups[:, 1], 1, 0))
    signs = rng.choice([-1.0, 1.0], size=n_users)
    signs[pairs[:, 0] == favorite] = 1.0
    signs[pairs[:, 1] == favorite] = -1.0

    # The learner sees only memberships and the comparison log above.
    model = fit_groupwise_span_adaptive(
        groups=groups,
        pair_indices=pairs,
        signs=signs,
        propensities=propensities,
        reference=np.full(n_alternatives, 1.0 / n_alternatives),
        tau=0.5,
        protocol="logged",
        kernel_backend="python",
    )
    new_groups = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    return model, new_groups


def main():
    model, new_groups = fit_example()
    print("Group memberships -> response probabilities [0, 1, 2]")
    for membership, lottery in zip(new_groups, model.predict(new_groups)):
        print(f"{membership.tolist()} -> {np.round(lottery, 4).tolist()}")


if __name__ == "__main__":
    main()
