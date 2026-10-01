"""Independent numerical reference for the span-adaptive learner.

This package is an oracle.  It must not import the optimized implementation.
"""

from empirical_testing.reference.span_adaptive_reference import (
    SpanAdaptiveReferenceResult,
    fit_span_adaptive_reference,
    predict_span_adaptive_reference,
)

__all__ = [
    "SpanAdaptiveReferenceResult",
    "fit_span_adaptive_reference",
    "predict_span_adaptive_reference",
]
