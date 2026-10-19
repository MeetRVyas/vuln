"""Pragmatic, pattern-based secret redaction for anything that might end up
in a log line (a Patcher-generated diff, a sandbox stdout/stderr capture,
...). This is deliberately a fast, dependency-free regex pass over common
secret *shapes* -- not a full secret-scanning SAST run -- kept separate from
:mod:`vulnrem.sast` so the logging path never has to construct a
:class:`~vulnrem.sast.base.SASTTool` (which needs a real file on disk and an
external binary) just to redact a log line. A fuller implementation could
delegate to Bandit's hardcoded-credential rule instead; this class is the
extension point for that swap.
"""

from __future__ import annotations

import re

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
    re.compile(
        r"-----BEGIN(?: RSA| EC| OPENSSH)? PRIVATE KEY-----.*?-----END(?: RSA| EC| OPENSSH)? PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub personal access tokens
]

_REPLACEMENT = "[REDACTED]"


class SecretRedactor:
    def redact(self, text: str) -> str:
        redacted = text
        for pattern in _PATTERNS:
            redacted = pattern.sub(_REPLACEMENT, redacted)
        return redacted
