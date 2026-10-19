"""Shared enumerations.

Centralised here (rather than duplicated per-module) so that adding a new
language or severity level is a one-line change that every layer picks up
consistently.
"""

from __future__ import annotations

from enum import StrEnum


class Language(StrEnum):
    """Target languages for the pipeline.

    Only PYTHON has a registered :class:`~vulnrem.languages.base.LanguageProfile`
    in this build. C and JAVA are declared here (and referenced in config /
    SAST tool mappings) so that adding them later is additive: register a
    profile + SAST tool config, nothing else in the pipeline changes.
    """

    PYTHON = "python"
    C = "c"
    JAVA = "java"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_loose_string(cls, value: str | None) -> Severity:
        """Best-effort mapping from the many different severity spellings
        that SAST tools use (opengrep: INFO/WARNING/ERROR, bandit:
        LOW/MEDIUM/HIGH, ...)."""
        if not value:
            return cls.MEDIUM
        v = value.strip().lower()
        mapping = {
            "info": cls.LOW,
            "informational": cls.LOW,
            "low": cls.LOW,
            "warning": cls.MEDIUM,
            "medium": cls.MEDIUM,
            "moderate": cls.MEDIUM,
            "error": cls.HIGH,
            "high": cls.HIGH,
            "critical": cls.CRITICAL,
            "blocker": cls.CRITICAL,
        }
        return mapping.get(v, cls.MEDIUM)


class AgentRole(StrEnum):
    """Every distinct LLM-calling role in the pipeline. Used as the routing
    key for role -> model/adapter mappings in provider configuration, and as
    the routing key for MLflow tagging."""

    TRIAGE = "triage"
    LOCALIZER = "localizer"
    PATCHER = "patcher"
    TEST_WRITER = "test_writer"
    VERIFIER = "verifier"  # deterministic in this build; reserved for future LLM-assisted checks
    JUDGE = "judge"
    REPRODUCER = "reproducer"


class RetryCategory(StrEnum):
    """Judge's classification of *why* a candidate patch was rejected. Drives
    the retry policy's branching logic (see :mod:`vulnrem.pipeline.retry_policy`)."""

    SYNTAX_OR_PLUMBING = "syntax_or_plumbing"
    REGRESSION_FAILURE = "regression_failure"
    VULNERABILITY_STILL_PRESENT = "vulnerability_still_present"
    SUSPICIOUS_OR_DEGENERATE = "suspicious_or_degenerate"
    MULTI_FILE_SCOPE_NEEDED = "multi_file_scope_needed"
    NONE = "none"  # accepted -- no retry needed


class FindingStatus(StrEnum):
    """Terminal (and intermediate) status of a single finding as it moves
    through the graph. Used both for the dashboard and for evaluation
    metrics."""

    PENDING = "pending"
    TRIAGED_FALSE_POSITIVE = "triaged_false_positive"
    IN_PROGRESS = "in_progress"
    ACCEPTED = "accepted"
    UNRESOLVED_RETRIES_EXHAUSTED = "unresolved_retries_exhausted"
    UNRESOLVED_ESCALATION_UNAVAILABLE = "unresolved_escalation_unavailable"
    ERROR = "error"
