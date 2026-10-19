"""Dynamic sanitizer checking (ASan/UBSan-style) is a C-language concept
with no Python equivalent -- Python's runtime already prevents the memory-
safety bug classes sanitizers exist to catch. This interface exists purely
so the Verifier's report always has a ``sanitizer`` field with a real,
typed status, and so that adding C support later means writing one real
:class:`SanitizerChecker` (wrapping ASan/UBSan build+run) and registering it
in the (future) C ``LanguageProfile`` -- no Verifier change required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vulnrem.pipeline.state import SanitizerStatus


class SanitizerChecker(ABC):
    @abstractmethod
    def check(self, patched_source: str) -> SanitizerStatus: ...


class NotApplicableSanitizerChecker(SanitizerChecker):
    """Used for every language whose runtime has no sanitizer concept
    (Python, in this build)."""

    def check(self, patched_source: str) -> SanitizerStatus:
        return "not_applicable"
