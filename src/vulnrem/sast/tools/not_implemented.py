"""Documented extension point.

cppcheck + clang static analyzer (C) and SpotBugs (Java) are the resolved
toolchain decisions (see project plan) for languages this build does not
yet implement (only Python is wired up -- see
:mod:`vulnrem.languages.registry`). Rather than leaving no trace of them,
each gets a real, registered :class:`~vulnrem.sast.base.SASTTool` entry that
clearly raises :class:`~vulnrem.core.errors.SASTToolNotAvailableError` on
use, so:

* the SAST registry's language -> tool-name -> factory mapping already
  documents the intended toolchain for C/Java, and
* implementing one of these later is "replace this class with a real
  subprocess wrapper" (following :class:`~vulnrem.sast.tools.opengrep.OpengrepTool`
  as a template) -- no registry or pipeline change needed.
"""

from __future__ import annotations

from pathlib import Path

from vulnrem.core.errors import SASTToolNotAvailableError
from vulnrem.sast.base import SASTTool
from vulnrem.sast.schemas import RawFinding


class NotImplementedSASTTool(SASTTool):
    def __init__(self, name: str, **_ignored_config):
        self._name = name

    @property
    def tool_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return False

    def scan(self, target_path: Path) -> list[RawFinding]:
        raise SASTToolNotAvailableError(
            f"{self._name!r} is a registered extension point but has no implementation in "
            "this build (Python is the only fully-wired language). See "
            "vulnrem.sast.tools.not_implemented for how to add it."
        )
