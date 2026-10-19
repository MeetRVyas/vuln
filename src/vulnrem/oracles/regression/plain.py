from __future__ import annotations

from vulnrem.languages.base import LanguageProfile
from vulnrem.oracles.pytest_support import run_pytest
from vulnrem.oracles.regression.base import RegressionChecker
from vulnrem.pipeline.state import OracleResult
from vulnrem.sandbox.base import Sandbox


class PlainRegressionChecker(RegressionChecker):
    """Runs the regression-tagged tests from the Test-Writer's battery
    against the patched source and requires all of them to pass. No
    coverage or execution-trace comparison -- see the module docstring in
    :mod:`vulnrem.oracles.regression.base` for the extension point this
    intentionally leaves open."""

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
    ) -> OracleResult:
        if not regression_test_names:
            return OracleResult(passed=True, detail="No regression tests were generated for this finding.")

        run_result = run_pytest(
            sandbox=sandbox,
            language_profile=language_profile,
            source_code=patched_source,
            test_code=test_source,
            timeout_s=timeout_s,
            memory_limit_mb=memory_limit_mb,
            cpu_limit_seconds=cpu_limit_seconds,
        )

        if run_result.execution.timed_out:
            return OracleResult(passed=False, detail="Regression test run timed out.")

        failing = [name for name in regression_test_names if not run_result.passed(name)]
        if failing:
            return OracleResult(passed=False, detail=f"Regression tests failed: {', '.join(failing)}")
        return OracleResult(passed=True, detail=f"All {len(regression_test_names)} regression tests passed.")
