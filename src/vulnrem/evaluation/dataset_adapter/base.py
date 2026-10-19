"""The dataset-ingestion abstraction.

The actual dataset's format was not yet decided when this was built (see
the project plan), so every consumer downstream -- the splitter, the
fine-tuning data prep, the evaluation harness -- depends only on
``list[CanonicalRecord]``. Pointing the project at a real dataset later is
"write one :class:`DatasetAdapter` for its actual format" (or configure
:class:`~vulnrem.evaluation.dataset_adapter.generic_json.
GenericJsonLinesAdapter`'s field map, if it's already close to the
canonical shape), not a change to any other module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vulnrem.evaluation.schemas import CanonicalRecord


class DatasetAdapter(ABC):
    @abstractmethod
    def load(self) -> list[CanonicalRecord]: ...
