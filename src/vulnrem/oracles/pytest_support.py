"""Runs a pytest file inside a :class:`~vulnrem.sandbox.base.Sandbox` and
parses per-test PASSED/FAILED/SKIPPED outcomes out of ``-v`` output.

Deliberately does *not* rely on any pytest output-format plugin
(``pytest-json-report`` etc.) -- the Sandbox abstraction only returns
stdout/stderr/exit code (not arbitrary files written during the run, which
would be awkward to retrieve uniformly from a removed Docker container), so
parsing the verbose per-test summary lines pytest already prints is the
simplest thing that works identically for both the subprocess and Docker
backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vulnrem.languages.base import LanguageProfile
from vulnrem.sandbox.base import Sandbox
from vulnrem.sandbox.schemas import ExecutionRequest, ExecutionResult

# Matches lines like:
#   tests/test_target.py::test_poc_sql_injection PASSED        [ 33%]
_OUTCOME_LINE = re.compile(
    r"^(?P<nodeid>\S+::\S+)\s+(?P<outcome>PASSED|FAILED|SKIPPED|ERROR)\b", re.MULTILINE
)


def parse_pytest_verbose_output(stdout: str) -> dict[str, str]:
    """Maps bare test function name (not the full nodeid) -> outcome
    string. If a name were to repeat (e.g. parametrized tests -- not used
    in this build's generated batteries), the last occurrence wins."""
    outcomes: dict[str, str] = {}
    for match in _OUTCOME_LINE.finditer(stdout):
        short_name = match.group("nodeid").split("::")[-1]
        outcomes[short_name] = match.group("outcome")
    return outcomes


@dataclass
class PytestRunResult:
    execution: ExecutionResult
    outcomes: dict[str, str]

    def outcome_for(self, test_name: str) -> str | None:
        return self.outcomes.get(test_name)

    def passed(self, test_name: str) -> bool:
        return self.outcome_for(test_name) == "PASSED"

    def all_passed(self, test_names: list[str]) -> bool:
        return bool(test_names) and all(self.passed(name) for name in test_names)


def test_file_name_for(language_profile: LanguageProfile) -> str:
    return f"test_{language_profile.source_file_name()}"


def run_pytest(
    sandbox: Sandbox,
    language_profile: LanguageProfile,
    source_code: str,
    test_code: str,
    timeout_s: float,
    memory_limit_mb: int,
    cpu_limit_seconds: float,
) -> PytestRunResult:
    source_name = language_profile.source_file_name()
    test_name = test_file_name_for(language_profile)
    request = ExecutionRequest(
        language=language_profile.language,
        files={source_name: source_code, test_name: test_code},
        command=language_profile.test_command(test_name),
        timeout_s=timeout_s,
        memory_limit_mb=memory_limit_mb,
        cpu_limit_seconds=cpu_limit_seconds,
    )
    execution = sandbox.run(request)
    outcomes = parse_pytest_verbose_output(execution.stdout)
    return PytestRunResult(execution=execution, outcomes=outcomes)
