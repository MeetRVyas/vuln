"""The SAST-tool abstraction.

The project plan is explicit that this system should not reinvent static
analysis: every entry here wraps one existing, mature tool (Opengrep,
Bandit, ...) as a subprocess and normalizes its output into
:class:`~vulnrem.sast.schemas.RawFinding`. The only custom logic is in
:mod:`vulnrem.pipeline.nodes.aggregator`, which merges the output of
however many tools are registered for a language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from vulnrem.sast.schemas import RawFinding


class SASTTool(ABC):
    @property
    @abstractmethod
    def tool_name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the underlying binary is installed/importable right now.
        Used by the tool registry to skip (with a warning) a configured but
        missing tool rather than fail the whole scan."""

    @abstractmethod
    def scan(self, target_path: Path) -> list[RawFinding]:
        """Run the tool against a single file (or directory, for tools that
        require one) and return normalized findings."""
