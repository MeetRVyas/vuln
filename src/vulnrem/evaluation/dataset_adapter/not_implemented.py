from __future__ import annotations

from vulnrem.core.errors import DatasetError
from vulnrem.evaluation.dataset_adapter.base import DatasetAdapter
from vulnrem.evaluation.schemas import CanonicalRecord


class NotImplementedDatasetAdapter(DatasetAdapter):
    """Registered so the adapter registry documents which named public
    datasets (BigVul, CVEFixes, CrossVul, DiverseVul, PrimeVul -- see the
    project plan) this build has *chosen* a place for but not yet wired up.
    Adding one is: implement ``load()`` following
    :class:`~vulnrem.evaluation.dataset_adapter.generic_json.GenericJsonLinesAdapter`
    as a template, then replace this entry in the registry."""

    def __init__(self, name: str, **_ignored_config):
        self._name = name

    def load(self) -> list[CanonicalRecord]:
        raise DatasetError(
            f"{self._name!r} is a registered extension point with no adapter implementation "
            "in this build. See vulnrem.evaluation.dataset_adapter.not_implemented."
        )
