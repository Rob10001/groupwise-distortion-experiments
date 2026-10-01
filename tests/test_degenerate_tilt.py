"""Regression tests for D-304: the three latent `_kl_tilt` / `batched_kl_tilt`
defects that previously degenerated silently instead of raising.

1. An unnormalized ``reference`` was never checked to sum to one.
2. A blown-up bracket search silently returned ``reference.copy()``, which is
   indistinguishable from a genuine zero-radius result.
3. Finite ``values`` whose spread (``max - min``) overflows to a non-finite
   float64 silently degenerated instead of being refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from empirical_testing.reporting import metrics
from empirical_testing.reporting.metrics import (
    KLTiltNonconvergenceError,
    _kl_tilt,
    batched_kl_tilt,
)


def test_scalar_tilt_rejects_a_reference_that_does_not_sum_to_one():
    reference = np.array([0.5, 0.6])  # sums to 1.1
    values = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="sum to one"):
        _kl_tilt(reference, values, 0.1)


def test_batched_tilt_rejects_a_reference_that_does_not_sum_to_one():
    reference = np.array([0.5, 0.6])  # sums to 1.1
    values_batch = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="sum to one"):
        batched_kl_tilt(values_batch, reference, 0.1)


def test_scalar_tilt_raises_on_bracket_nonconvergence(monkeypatch):
    """A bracket search that blows past `_TILT_MAX_SCALE` must raise, not
    silently return the reference (that outcome is indistinguishable from a
    genuine zero-radius result). Monkeypatch the cap down so an otherwise
    perfectly ordinary near-tie case (the one from the `_kl_tilt` docstring)
    blows the bracket on the first doubling.
    """

    monkeypatch.setattr(metrics, "_TILT_MAX_SCALE", 1.5)
    reference = np.array([0.5, 0.5])
    values = np.array([0.5, 0.500001])
    with pytest.raises(KLTiltNonconvergenceError):
        _kl_tilt(reference, values, 0.1)


def test_scalar_tilt_rejects_finite_values_with_overflowing_spread():
    reference = np.array([0.5, 0.5])
    values = np.array([-1e308, 1e308])
    assert np.all(np.isfinite(values))
    with np.errstate(over="ignore"):
        assert not np.isfinite(values.max() - values.min())
    with pytest.warns(RuntimeWarning, match="overflow"), pytest.raises(ValueError, match="spread"):
        _kl_tilt(reference, values, 0.1)
