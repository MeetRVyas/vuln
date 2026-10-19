from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vulnrem.core.enums import Language


class CanonicalRecord(BaseModel):
    """One labeled vulnerability example, in the schema every
    :class:`~vulnrem.evaluation.dataset_adapter.base.DatasetAdapter` must
    produce. The raw dataset's actual on-disk format (CSV columns, JSON
    keys, whatever BigVul/CVEFixes/DiverseVul/PrimeVul happen to use) is
    entirely absorbed by the adapter layer -- nothing downstream (the
    splitter, the fine-tuning data prep, the evaluation harness) ever sees
    it."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    language: Language
    cwe_id: str
    vulnerable_code: str
    fixed_code: str | None = None
    description: str = ""
    source_dataset: str = ""
