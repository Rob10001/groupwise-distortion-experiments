"""Interfaces implemented by dataset-specific preparation adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from empirical_testing.core.types import DensePreferenceDataset


class DenseDatasetAdapter(Protocol):
    name: str

    def load(
        self, path: Path, group_names: Sequence[str], options: Mapping[str, Any]
    ) -> DensePreferenceDataset:
        """Load, prepare, and validate one dense-profile dataset.

        ``options`` is the config's ``dataset_options`` mapping (possibly empty);
        an adapter must reject keys it does not understand.
        """
