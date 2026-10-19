from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PreprocessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_source: str
    warnings: list[str] = []


def preprocess_source(source: str) -> PreprocessResult:
    warnings: list[str] = []
    normalized = source

    if normalized.startswith("\ufeff"):
        normalized = normalized[1:]
        warnings.append("Stripped a UTF-8 byte-order mark from the file.")

    if "\r\n" in normalized:
        normalized = normalized.replace("\r\n", "\n")
        warnings.append("Normalized CRLF line endings to LF.")
    elif "\r" in normalized:
        normalized = normalized.replace("\r", "\n")
        warnings.append("Normalized legacy CR line endings to LF.")

    return PreprocessResult(normalized_source=normalized, warnings=warnings)
