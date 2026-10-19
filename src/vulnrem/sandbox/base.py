"""The code-execution abstraction.

Every place the pipeline needs to *run* candidate code or generated tests
(the Test Writer's self-validation step, the Verifier's functional and
security oracles) goes through this interface, never through a raw
``subprocess`` call in a pipeline node. That is what lets the production
deployment use real Docker-container isolation while local development and
CI use a lighter, dependency-free fallback, with zero changes to any
oracle or node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vulnrem.sandbox.schemas import ExecutionRequest, ExecutionResult


class Sandbox(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """``"docker"`` or ``"subprocess"`` -- recorded on every oracle
        result / MLflow log line so evaluation runs can be compared
        apples-to-apples even if the backend differed."""

    @abstractmethod
    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Write ``request.files`` into a fresh, isolated workspace and
        execute ``request.command`` there under the given resource/time
        limits. Must never raise for an ordinary non-zero exit or timeout --
        those are reported *in* the returned :class:`ExecutionResult`.
        Raising is reserved for backend-level failures (see
        :class:`~vulnrem.core.errors.SandboxUnavailableError`)."""

    @abstractmethod
    def health_check(self) -> bool:
        """Whether this backend is actually usable right now (e.g. the
        Docker daemon is reachable)."""
