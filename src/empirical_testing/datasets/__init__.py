"""Dataset adapter registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from empirical_testing.core.types import DensePreferenceDataset
from empirical_testing.datasets.base import DenseDatasetAdapter
from empirical_testing.datasets.coffee import CoffeeAdapter
from empirical_testing.datasets.persona import PersonaAdapter
_DENSE_ADAPTERS: dict[str, DenseDatasetAdapter] = {
    CoffeeAdapter.name: CoffeeAdapter(),
    PersonaAdapter.name: PersonaAdapter(),
}


def load_dense_dataset(
    name: str,
    path: Path,
    group_names: Sequence[str],
    options: Mapping[str, Any] | None = None,
) -> DensePreferenceDataset:
    try:
        adapter = _DENSE_ADAPTERS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown dense dataset {name!r}; available={tuple(_DENSE_ADAPTERS)}"
        ) from error
    return adapter.load(path, group_names, dict(options or {}))


__all__ = ["load_dense_dataset"]
