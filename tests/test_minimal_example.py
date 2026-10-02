"""Keep the reader's quick-start example executable and feasible."""

import importlib.util
from pathlib import Path

import numpy as np

from empirical_testing.algorithms.one_expert import kl_divergence


def test_minimal_example_learns_from_one_logged_comparison_per_user(capsys):
    path = Path(__file__).resolve().parents[1] / "examples/minimal_glhf.py"
    spec = importlib.util.spec_from_file_location("minimal_glhf", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    model, groups = example.fit_example()
    lotteries = model.predict(groups)
    assert lotteries.shape == (4, 3)
    assert np.all(lotteries >= 0.0)
    np.testing.assert_allclose(lotteries.sum(axis=1), 1.0)
    assert all(kl_divergence(p, model.reference) <= model.tau + 1e-9 for p in lotteries)
    assert model.drawn_comparisons.shape == (240, 4)
    np.testing.assert_allclose(model.drawn_comparisons[:, 3], 1.0 / 3.0)
    np.testing.assert_array_equal(model.snapshot_rounds, np.arange(240))
    example.main()
    assert "[1, 1, 1] ->" in capsys.readouterr().out
