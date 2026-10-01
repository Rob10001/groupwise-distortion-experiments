"""Backend-selection layer for numerical implementations of the learner."""

from empirical_testing.backends.interface import (
    Backend,
    BackendStatus,
    BackendUnavailableError,
    SpanAdaptiveFit,
    list_backends,
    resolve_backend,
)

__all__ = [
    "Backend",
    "BackendStatus",
    "BackendUnavailableError",
    "SpanAdaptiveFit",
    "list_backends",
    "resolve_backend",
]
