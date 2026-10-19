"""Structured application logging.

Every node logs INFO-level lifecycle events (a stage started/finished, a
Judge decision, a retry) and DEBUG-level detail (full prompts/diffs) through
the standard :mod:`logging` module with an ``extra={...}`` dict of
structured fields -- see any module under :mod:`vulnrem.pipeline.nodes` for
the pattern. This module turns that into either human-readable lines (dev)
or one JSON object per line (prod, easy to ship to a log aggregator),
consistently redacting secret-shaped substrings and truncating any
individual field so a full LLM completion or sandbox capture never floods
the log.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from vulnrem.config.schemas import LoggingConfig
from vulnrem.observability.secret_redaction import SecretRedactor

_RESERVED_LOG_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


def _sanitize_value(value: Any, redactor: SecretRedactor | None, max_chars: int) -> Any:
    if isinstance(value, str):
        text = redactor.redact(value) if redactor is not None else value
        if len(text) > max_chars:
            text = text[:max_chars] + f"...<truncated {len(text) - max_chars} chars>"
        return text
    if isinstance(value, dict):
        return {k: _sanitize_value(v, redactor, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(v, redactor, max_chars) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def __init__(self, redact_secrets: bool, max_code_chars: int):
        super().__init__()
        self._redactor = SecretRedactor() if redact_secrets else None
        self._max_chars = max_code_chars

    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_LOG_RECORD_ATTRS}
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **extras,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        payload = _sanitize_value(payload, self._redactor, self._max_chars)
        return json.dumps(payload, default=str)


class RedactingPlainFormatter(logging.Formatter):
    """Human-readable formatter for local dev, with the same redaction
    applied to the rendered message."""

    def __init__(self, redact_secrets: bool, max_code_chars: int):
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        self._redactor = SecretRedactor() if redact_secrets else None
        self._max_chars = max_code_chars

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if self._redactor is not None:
            rendered = self._redactor.redact(rendered)
        if len(rendered) > self._max_chars * 4:  # plain lines carry more than one field; looser cap
            rendered = rendered[: self._max_chars * 4] + "...<truncated>"
        return rendered


def configure_logging(config: LoggingConfig, logger_name: str = "vulnrem") -> None:
    handler = logging.StreamHandler()
    if config.json_format:
        handler.setFormatter(JsonFormatter(config.redact_secrets, config.max_code_chars_in_logs))
    else:
        handler.setFormatter(RedactingPlainFormatter(config.redact_secrets, config.max_code_chars_in_logs))

    logger = logging.getLogger(logger_name)
    logger.handlers = [handler]
    logger.setLevel(config.level)
    logger.propagate = False
