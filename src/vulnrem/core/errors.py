"""Shared exception hierarchy.

Every subsystem (LLM providers, sandboxes, SAST tools, dataset adapters, ...)
raises exceptions rooted in :class:`VulnRemError` so calling code -- most
importantly the pipeline nodes -- can catch failures at a predictable level
without needing to know which concrete implementation raised them. This is
what lets us swap implementations (e.g. a vLLM provider for a Google
provider) without touching any error-handling code downstream.
"""

from __future__ import annotations


class VulnRemError(Exception):
    """Base class for every exception raised by this project."""


# --------------------------------------------------------------------------- #
# LLM layer
# --------------------------------------------------------------------------- #
class LLMError(VulnRemError):
    """Base class for LLM provider failures."""


class LLMConnectionError(LLMError):
    """The provider could not be reached (network, DNS, connection refused)."""


class LLMResponseError(LLMError):
    """The provider responded, but the response was unusable (HTTP error,
    empty completion, malformed payload, etc.)."""


class LLMStructuredOutputError(LLMError):
    """A structured/JSON response could not be parsed into the requested
    schema, even after the provider's internal retries were exhausted."""


class LLMConfigurationError(LLMError):
    """The provider was misconfigured (missing credential, unknown role/model
    mapping, unknown provider name in the registry, ...)."""


# --------------------------------------------------------------------------- #
# Sandbox layer
# --------------------------------------------------------------------------- #
class SandboxError(VulnRemError):
    """Base class for sandboxed-execution failures."""


class SandboxUnavailableError(SandboxError):
    """The sandbox backend (e.g. the Docker daemon) is not reachable from
    this environment."""


class SandboxTimeoutError(SandboxError):
    """Execution was killed after exceeding its wall-clock timeout."""


# --------------------------------------------------------------------------- #
# SAST layer
# --------------------------------------------------------------------------- #
class SASTError(VulnRemError):
    """Base class for static-analysis tool failures."""


class SASTToolNotAvailableError(SASTError):
    """The underlying binary (opengrep, bandit, ...) is not installed or not
    on PATH."""


class SASTOutputParseError(SASTError):
    """The tool ran but its output could not be parsed."""


# --------------------------------------------------------------------------- #
# Pipeline / orchestration layer
# --------------------------------------------------------------------------- #
class PipelineError(VulnRemError):
    """Base class for orchestration-level failures."""


class EscalationUnavailableError(PipelineError):
    """Raised (and caught) internally when a finding is routed to an
    escalation path (Debugger, repo-scale agent) that has no working
    implementation yet in this build tier. The retry policy treats this as
    a signal to gracefully degrade rather than crash the run."""

    def __init__(self, escalation_name: str, message: str = ""):
        self.escalation_name = escalation_name
        super().__init__(
            message or f"Escalation path '{escalation_name}' is not implemented in this build."
        )


# --------------------------------------------------------------------------- #
# Configuration / dataset layer
# --------------------------------------------------------------------------- #
class ConfigError(VulnRemError):
    """Base class for configuration-loading failures."""


class DatasetError(VulnRemError):
    """Base class for dataset ingestion / splitting failures."""


class DatasetSchemaError(DatasetError):
    """A dataset record did not conform to the canonical schema."""


class DatasetLeakageError(DatasetError):
    """A record appeared in both the fine-tuning and evaluation splits."""
