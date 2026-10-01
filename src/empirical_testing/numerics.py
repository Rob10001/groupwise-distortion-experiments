"""Small float64 kernels shared by the learners and the reporting layer.

This module deliberately imports nothing from the rest of the package.  It sits
below ``algorithms`` and ``reporting`` so either can use it without creating an
import cycle, and so the kernels can be tested against their general-purpose
counterparts in isolation.

Everything here is a *specialization*, never an approximation: each function
reproduces the exact floating-point operations of the library routine it
replaces on the inputs it accepts, and delegates the cases it does not accept
to that routine.  Nothing here loosens a tolerance, changes a dtype or enables
fast-math relaxations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp as _scipy_logsumexp

FloatArray = NDArray[np.float64]

#: ``np.finfo(np.float64).tiny`` performs a ``finfo`` lookup on every access and
#: the hot loops read it tens of thousands of times per fit.
TINY = np.finfo(np.float64).tiny

__all__ = ["TINY", "log_sum_exp"]


def log_sum_exp(values: FloatArray) -> np.float64:
    """``log(sum(exp(values)))`` for the small one-dimensional vectors used here.

    Contract
    --------
    ``values``
        A one-dimensional ``float64`` array with at least one entry.  In this
        codebase that is one entry per group (``K``) or one per alternative
        (``m``); nothing assumes a particular length, and in particular nothing
        hard-codes Coffee's four actions.  A singleton is supported and returns
        its only element (up to the ``log1p(0) + log(1)`` that the algorithm
        below adds, both exactly zero).
    result
        A ``float64`` scalar.

    Finiteness
        Inputs whose maximum is finite -- including vectors with some ``-inf``
        entries, which contribute exactly zero -- take the fast path.  Every
        other case (any ``+inf``, any ``NaN``, or an all-``-inf`` vector, all of
        which make the maximum non-finite) is delegated to
        :func:`scipy.special.logsumexp` so its edge-case behaviour, not a
        reimplementation of it, decides the answer.

    Why this exists
    ---------------
    :func:`scipy.special.logsumexp` is array-API generic.  On vectors of length
    two to a few dozen its promotion and dispatch cost an order of magnitude
    more than the reduction itself, and profiling the span-adaptive Coffee fit
    attributed roughly three quarters of its run time to those calls.

    Why the arithmetic is written this way
    --------------------------------------
    Not as the textbook ``max(x) + log(sum(exp(x - max(x))))``.  SciPy pulls the
    maximal entries out of the sum and finishes with ``log1p``, which is more
    accurate when every remaining term is negligible, and this function
    reproduces that same sequence of operations::

        a_max = max(values)                   # the shift
        m     = #{a in values : a == a_max}   # multiplicity of the maximum
        s     = sum(exp(values - a_max))      # maximal entries contribute 0
        out   = log1p(s / m) + log(m) + a_max

    The two forms are algebraically the same: the ``m`` maximal entries are
    withheld from ``s`` and reinstated through ``log(m)``, so
    ``exp(out) = m * (1 + s/m) = m + s``, the required sum; and the shift by
    ``a_max`` is a common factor pulled through the exponential, which is exact
    because it is applied before the sum and undone after it.

    Following SciPy's operation order rather than the textbook identity makes
    this function **bit-identical** to the routine it replaces on every input it
    accepts.  That is not a cosmetic property.  Deployed policies feed discrete,
    seeded draws under the active protocol, and archived runs are validated by
    file digests; a one-ulp difference could flip a sampled comparison or
    invalidate an archive.  Equality removes that whole class of risk instead of
    bounding it.
    """

    a_max = np.max(values)
    if not np.isfinite(a_max):
        return np.float64(_scipy_logsumexp(values))
    at_maximum = values == a_max
    multiplicity = np.float64(np.count_nonzero(at_maximum))
    shifted = np.exp(np.where(at_maximum, -np.inf, values) - a_max)
    total = np.sum(shifted)
    if total != 0.0:
        total = total / multiplicity
    return np.log1p(total) + np.log(multiplicity) + a_max
