"""Backend adapters for comparing production and reference span-adaptive learners.

Available backends are ``cpu-reference`` (the independent NumPy/SciPy oracle)
and ``cpu-production`` (the production span-adaptive learner). Unknown names
raise ``BackendUnavailableError`` rather than falling back to another backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class BackendUnavailableError(LookupError):
    """An explicit backend request could not be honoured.  No fallback occurred."""


@dataclass(frozen=True)
class BackendStatus:
    """Description of one named CPU backend."""

    backend_id: str
    required_dependencies: tuple[str, ...]
    device_description: str
    precision: str


@dataclass
class SpanAdaptiveFit:
    """Scientific outputs shared across backends for differential comparison."""

    averaged_policy: FloatArray
    outer_log_mass_snapshots: FloatArray
    inner_policy_snapshots: FloatArray
    snapshot_weights: FloatArray
    Q: FloatArray
    C: FloatArray
    V: FloatArray
    reference: FloatArray
    tau: float
    drawn_comparisons: FloatArray
    _predict: Optional[Callable[[ArrayLike], FloatArray]] = field(default=None, repr=False)

    def predict(self, groups: ArrayLike) -> FloatArray:
        if self._predict is None:
            raise RuntimeError("this fit has no predict callable attached")
        return self._predict(groups)


class Backend(ABC):
    """Operations the project dispatches through a resolved backend."""

    @property
    @abstractmethod
    def status(self) -> BackendStatus:
        """Description of this backend."""

    @property
    @abstractmethod
    def loaded_module(self) -> ModuleType:
        """The implementation module whose ``__file__`` is recorded in provenance."""

    @property
    def backend_id(self) -> str:
        return self.status.backend_id

    @property
    def loaded_module_path(self) -> str:
        path = getattr(self.loaded_module, "__file__", None)
        if not path:
            raise RuntimeError(f"backend {self.backend_id!r} has no loadable module path")
        return str(path)

    def provenance(self) -> dict[str, str]:
        """Run-record fields: resolved backend id and the loaded module path."""

        return {
            "backend_id": self.backend_id,
            "module_path": self.loaded_module_path,
        }

    @abstractmethod
    def fit_span_adaptive(
        self,
        groups: ArrayLike,
        utilities: Optional[ArrayLike] = None,
        beta: float = 1.0,
        reference: Optional[ArrayLike] = None,
        tau: float = 0.0,
        protocol: str = "passive",
        seed: int = 0,
        pair_indices: Optional[ArrayLike] = None,
        signs: Optional[ArrayLike] = None,
        propensities: Optional[ArrayLike] = None,
    ) -> SpanAdaptiveFit:
        """Fit the span-adaptive learner on this backend."""

    @abstractmethod
    def predict(self, fitted: SpanAdaptiveFit, groups: ArrayLike) -> FloatArray:
        """Evaluate the fitted averaged policy on ``groups``."""

class _CpuReferenceBackend(Backend):
    """Independent oracle.  Loads only the reference module."""

    def __init__(self) -> None:
        import empirical_testing.reference.span_adaptive_reference as module

        self._module = module

    @property
    def status(self) -> BackendStatus:
        return BackendStatus(
            backend_id="cpu-reference",
            required_dependencies=("numpy", "scipy"),
            device_description="host CPU, independent NumPy/SciPy reference",
            precision="float64",
        )

    @property
    def loaded_module(self) -> ModuleType:
        return self._module

    def fit_span_adaptive(
        self,
        groups: ArrayLike,
        utilities: Optional[ArrayLike] = None,
        beta: float = 1.0,
        reference: Optional[ArrayLike] = None,
        tau: float = 0.0,
        protocol: str = "passive",
        seed: int = 0,
        pair_indices: Optional[ArrayLike] = None,
        signs: Optional[ArrayLike] = None,
        propensities: Optional[ArrayLike] = None,
    ) -> SpanAdaptiveFit:
        result = self._module.fit_span_adaptive_reference(
            groups,
            utilities=utilities,
            beta=beta,
            reference=reference,
            tau=tau,
            protocol=protocol,
            seed=seed,
            pair_indices=pair_indices,
            signs=signs,
            propensities=propensities,
        )
        return SpanAdaptiveFit(
            averaged_policy=result.averaged_policy,
            outer_log_mass_snapshots=result.outer_log_mass_snapshots,
            inner_policy_snapshots=result.inner_policy_snapshots,
            snapshot_weights=result.snapshot_weights,
            Q=result.Q,
            C=result.C,
            V=result.V,
            reference=result.reference,
            tau=result.tau,
            drawn_comparisons=result.drawn_comparisons,
            _predict=result.predict,
        )

    def predict(self, fitted: SpanAdaptiveFit, groups: ArrayLike) -> FloatArray:
        return np.asarray(fitted.predict(groups), dtype=np.float64)

class _CpuProductionBackend(Backend):
    """Existing production path.  Imports ``span_adaptive`` only here."""

    def __init__(self) -> None:
        import empirical_testing.algorithms.span_adaptive as module

        self._module = module

    @property
    def status(self) -> BackendStatus:
        return BackendStatus(
            backend_id="cpu-production",
            required_dependencies=("numpy", "scipy"),
            device_description="host CPU, existing span_adaptive implementation",
            precision="float64",
        )

    @property
    def loaded_module(self) -> ModuleType:
        return self._module

    def fit_span_adaptive(
        self,
        groups: ArrayLike,
        utilities: Optional[ArrayLike] = None,
        beta: float = 1.0,
        reference: Optional[ArrayLike] = None,
        tau: float = 0.0,
        protocol: str = "passive",
        seed: int = 0,
        pair_indices: Optional[ArrayLike] = None,
        signs: Optional[ArrayLike] = None,
        propensities: Optional[ArrayLike] = None,
    ) -> SpanAdaptiveFit:
        model = self._module.fit_groupwise_span_adaptive(
            groups,
            utilities=utilities,
            beta=beta,
            reference=reference,
            tau=tau,
            protocol=protocol,
            seed=seed,
            pair_indices=pair_indices,
            signs=signs,
            propensities=propensities,
        )
        averaged = model.predict(groups)
        return SpanAdaptiveFit(
            averaged_policy=np.asarray(averaged, dtype=np.float64),
            outer_log_mass_snapshots=np.stack(model._outer_log_mass_snapshots, axis=0),
            inner_policy_snapshots=np.stack(model._inner_policy_snapshots, axis=0),
            snapshot_weights=np.asarray(model.snapshot_weights, dtype=np.float64),
            Q=np.asarray(model.outer_second_moments, dtype=np.float64),
            C=np.asarray(model.centered_inner_second_moments, dtype=np.float64),
            V=np.asarray(model.inner_second_moments, dtype=np.float64),
            reference=np.asarray(model.reference, dtype=np.float64),
            tau=float(model.tau),
            drawn_comparisons=np.asarray(model.drawn_comparisons, dtype=np.float64),
            _predict=model.predict,
        )

    def predict(self, fitted: SpanAdaptiveFit, groups: ArrayLike) -> FloatArray:
        return np.asarray(fitted.predict(groups), dtype=np.float64)

_REGISTRY: dict[str, Callable[[], Backend]] = {
    "cpu-reference": _CpuReferenceBackend,
    "cpu-production": _CpuProductionBackend,
}


def list_backends() -> tuple[str, ...]:
    """Names of the available CPU implementations."""
    return tuple(_REGISTRY)


def resolve_backend(name: str) -> Backend:
    """Return the named backend, or raise. Never substitutes another backend."""
    if name not in _REGISTRY:
        raise BackendUnavailableError(
            f"Unknown backend {name!r}. Available backends: {list(list_backends())}. "
            "An explicit backend request never falls back to a different backend."
        )
    return _REGISTRY[name]()
