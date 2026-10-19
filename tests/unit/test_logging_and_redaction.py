from __future__ import annotations

import json
import logging

from vulnrem.config.schemas import LoggingConfig
from vulnrem.observability.logging_config import JsonFormatter, configure_logging
from vulnrem.observability.secret_redaction import SecretRedactor


def test_secret_redactor_masks_aws_key():
    text = "found credential AKIAABCDEFGHIJKLMNOP in the diff"
    redacted = SecretRedactor().redact(text)
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "[REDACTED]" in redacted


def test_secret_redactor_masks_generic_api_key_assignment():
    text = 'API_KEY = "sk-abcdef1234567890"'
    redacted = SecretRedactor().redact(text)
    assert "sk-abcdef1234567890" not in redacted


def test_secret_redactor_masks_private_key_block():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIBAAK...\n-----END RSA PRIVATE KEY-----"
    redacted = SecretRedactor().redact(text)
    assert "MIIBAAK" not in redacted


def test_secret_redactor_leaves_ordinary_text_untouched():
    text = "def add(a, b):\n    return a + b"
    assert SecretRedactor().redact(text) == text


def test_json_formatter_includes_extra_fields_and_redacts():
    formatter = JsonFormatter(redact_secrets=True, max_code_chars=2000)
    record = logging.LogRecord(
        name="vulnrem.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="patch_generated",
        args=(),
        exc_info=None,
    )
    record.finding_id = "abc123"
    record.secret_leak = "password = 'hunter2_definitely_a_password'"

    payload = json.loads(formatter.format(record))
    assert payload["message"] == "patch_generated"
    assert payload["finding_id"] == "abc123"
    assert "hunter2_definitely_a_password" not in payload["secret_leak"]


def test_json_formatter_truncates_long_fields():
    formatter = JsonFormatter(redact_secrets=False, max_code_chars=50)
    record = logging.LogRecord(
        name="vulnrem.test", level=logging.DEBUG, pathname=__file__, lineno=1, msg="x", args=(), exc_info=None
    )
    record.big_field = "y" * 500
    payload = json.loads(formatter.format(record))
    assert len(payload["big_field"]) < 500
    assert "truncated" in payload["big_field"]


def test_configure_logging_sets_level_and_does_not_propagate():
    configure_logging(LoggingConfig(level="DEBUG", json_format=True), logger_name="vulnrem.test.configure")
    logger = logging.getLogger("vulnrem.test.configure")
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert len(logger.handlers) == 1
