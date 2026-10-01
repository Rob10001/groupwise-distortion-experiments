"""Focused checks for full-fit prefix deployment reconstruction."""

import numpy as np
import pytest
from empirical_testing.algorithms.one_expert import AlignmentLotteryPolicy
from empirical_testing.experiments.trajectory import (
    checkpoint_rounds,
    predict_prefixes,
)


def _model():
    return AlignmentLotteryPolicy(
        _outer_log_mass_snapshots=np.log(
            [[1.0, 3.0], [4.0, 1.0], [1.0, 2.0]]
        ),
        _inner_policy_snapshots=np.array(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.8, 0.2], [0.3, 0.7]],
                [[0.7, 0.3], [0.4, 0.6]],
            ]
        ),
        snapshot_weights=np.full(3, 1.0 / 3),
        snapshot_rounds=np.arange(3),
        reference=np.array([0.5, 0.5]),
        tau=1.0,
    )


def test_prefixes_match_manual_mixtures_and_full_predict():
    model = _model()
    groups = np.array([[1, 1], [1, 0], [0, 1], [0, 0], [1, 1]])
    checkpoints = [3, 0, 2, 1, 2]
    actual = predict_prefixes(model, groups, checkpoints)
    np.testing.assert_allclose(actual[0], model.predict(groups), atol=2e-15)
    np.testing.assert_array_equal(actual[1], np.full((len(groups), 2), 0.5))
    np.testing.assert_array_equal(actual[2], actual[4])
    manual = []
    for t in range(2):
        rows = []
        for row in groups:
            selected = row.astype(bool)
            if not selected.any():
                rows.append(model.reference)
            else:
                mass = np.exp(model._outer_log_mass_snapshots[t, selected])
                rows.append(mass @ model._inner_policy_snapshots[t, selected] / mass.sum())
        manual.append(rows)
    np.testing.assert_allclose(actual[2], np.mean(manual, axis=0), atol=2e-15)
    np.testing.assert_allclose(actual[3], manual[0], atol=2e-15)


def test_checkpoint_validation_and_fraction_mapping():
    np.testing.assert_array_equal(checkpoint_rounds(10, [1, 0.5, 0, 0.5]), [0, 5, 10])
    with pytest.raises(ValueError):
        checkpoint_rounds(0, [0.5])
    with pytest.raises(ValueError):
        checkpoint_rounds(3, [1.1])
    for invalid in ([-1], [4], [1.5], [True]):
        with pytest.raises(ValueError):
            predict_prefixes(_model(), [[1, 1]], invalid)
    with pytest.raises(ValueError):
        predict_prefixes(_model(), [[1, 1, 0]], [1])
