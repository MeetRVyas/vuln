"""The regression-checking abstraction.

Locked in per the project plan: plain test-suite-based checking is the
default and only implementation in this build, but it lives behind this
interface so a coverage/behavior-fingerprinting implementation (comparing
execution traces or coverage maps between original and patched code, not
just named test outcomes) can be added later as a second
:class:`RegressionChecker`, selected by config, with zero changes to the
Verifier node that calls it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vulnrem.languages.base import LanguageProfile
from vulnrem.pipeline.state import OracleResult
from vulnrem.sandbox.base import Sandbox


class RegressionChecker(ABC):
    @abstractmethod
    def check(
        self,
        sandbox: Sandbox,
        language_profile: LanguageProfile,
        patched_source: str,
        test_source: str,
        regression_test_names: list[str],
        timeout_s: float,
        memory_limit_mb: int,
        cpu_limit_seconds: float,
    ) -> OracleResult: ...
