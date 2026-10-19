"""Default adapter: one JSON object per line, with a configurable field-name
map for whatever key names the real dataset actually uses (e.g. a dataset
that calls the vulnerable snippet ``"func_before"`` instead of
``vulnerable_code`` needs only ``field_map={"vulnerable_code": "func_before"}``,
not a new adapter class).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vulnrem.core.enums import Language
from vulnrem.core.errors import DatasetSchemaError
from vulnrem.evaluation.dataset_adapter.base import DatasetAdapter
from vulnrem.evaluation.schemas import CanonicalRecord

logger = logging.getLogger(__name__)

_DEFAULT_FIELD_MAP = {
    "record_id": "record_id",
    "language": "language",
    "cwe_id": "cwe_id",
    "vulnerable_code": "vulnerable_code",
    "fixed_code": "fixed_code",
    "description": "description",
}


class GenericJsonLinesAdapter(DatasetAdapter):
    def __init__(
        self,
        path: Path | str,
        field_map: dict[str, str] | None = None,
        source_dataset_name: str = "custom",
        strict: bool = False,
    ):
        self._path = Path(path)
        self._field_map = {**_DEFAULT_FIELD_MAP, **(field_map or {})}
        self._source_dataset_name = source_dataset_name
        self._strict = strict

    def load(self) -> list[CanonicalRecord]:
        records: list[CanonicalRecord] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(self._to_canonical(json.loads(line)))
                except (json.JSONDecodeError, DatasetSchemaError, ValueError) as exc:
                    if self._strict:
                        raise DatasetSchemaError(f"{self._path}:{line_number}: {exc}") from exc
                    logger.warning(
                        "dataset_record_skipped",
                        extra={"path": str(self._path), "line": line_number, "error": str(exc)},
                    )
        return records

    def _to_canonical(self, raw: dict) -> CanonicalRecord:
        def _get(field: str):
            return raw.get(self._field_map[field])

        language_raw = _get("language")
        try:
            language = Language(str(language_raw).lower())
        except ValueError as exc:
            raise DatasetSchemaError(f"Unknown language {language_raw!r}") from exc

        record_id = _get("record_id")
        vulnerable_code = _get("vulnerable_code")
        cwe_id = _get("cwe_id")
        if not record_id or not vulnerable_code or not cwe_id:
            raise DatasetSchemaError(
                f"Missing required field(s) among record_id/vulnerable_code/cwe_id in {raw!r}"
            )

        return CanonicalRecord(
            record_id=str(record_id),
            language=language,
            cwe_id=str(cwe_id),
            vulnerable_code=str(vulnerable_code),
            fixed_code=_get("fixed_code"),
            description=str(_get("description") or ""),
            source_dataset=self._source_dataset_name,
        )
