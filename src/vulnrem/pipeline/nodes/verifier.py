from __future__ import annotations

import logging

from vulnrem.languages.base import LanguageProfile
from vulnrem.oracles.pytest_support import run_pytest
from vulnrem.oracles.regression.base import RegressionChecker
from vulnrem.oracles.sanitizer import SanitizerChecker
from vulnrem.oracles.security import SecurityRescanChecker
from vulnrem.pipeline.state import (
    FunctionalOracleResult,
    OracleResult,
    PipelineState,
    SecurityOracleResult,
    VerifierReport,
)
from vulnrem.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class VerifierNode:
    """Deliberately has no LLM dependency: everything here is either a
    cheap static check (syntax) or an execution the Sandbox actually ran.
    This is what makes the dual-oracle result trustworthy input for the
    Judge, rather than another model's opinion."""

    def __init__(
        self,
        sandbox: Sandbox,
        language_profile: LanguageProfile,
        regression_checker: RegressionChecker,
        security_rescan_checker: SecurityRescanChecker,
        sanitizer_checker: SanitizerChecker,
        timeout_s: float,
        memory_limit_mb: int,
        cpu_limit_seconds: float,
    ):
        self._sandbox = sandbox
        self._language_profile = language_profile
        self._regression_checker = regression_checker
        self._security_rescan_checker = security_rescan_checker
        self._sanitizer_checker = sanitizer_checker
        self._timeout_s = timeout_s
        self._memory_limit_mb = memory_limit_mb
        self._cpu_limit_seconds = cpu_limit_seconds

    def __call__(self, state: PipelineState) -> dict:
        assert state.patch_candidate is not None
        assert state.test_battery is not None
        patched_source = state.patch_candidate.patched_source
        battery = state.test_battery

        syntax = self._language_profile.syntax_check(patched_source)
        if not syntax.is_valid:
            skipped = OracleResult(passed=False, detail="Skipped: patched source has a syntax error.")
            report = VerifierReport(
                functional=FunctionalOracleResult(
                    regression=OracleResult(
                        passed=False, detail=f"Syntax error in patched source: {syntax.error_message}"
                    ),
                    edge_case=skipped,
                ),
                security=SecurityOracleResult(poc_no_longer_triggers=skipped, sast_rescan_clean=skipped),
                sanitizer=self._sanitizer_checker.check(patched_source),
            )
            return {"verifier_report": report}

        regression_result = self._regression_checker.check(
            sandbox=self._sandbox,
            language_profile=self._language_profile,
            patched_source=patched_source,
            test_source=battery.combined_source,
            regression_test_names=battery.names_of_kind("regression"),
            timeout_s=self._timeout_s,
            memory_limit_mb=self._memory_limit_mb,
            cpu_limit_seconds=self._cpu_limit_seconds,
        )

        edge_case_names = battery.names_of_kind("edge_case")
        poc_names = battery.names_of_kind("poc")

        run_result = run_pytest(
            sandbox=self._sandbox,
            language_profile=self._language_profile,
            source_code=patched_source,
            test_code=battery.combined_source,
            timeout_s=self._timeout_s,
            memory_limit_mb=self._memory_limit_mb,
            cpu_limit_seconds=self._cpu_limit_seconds,
        )

        if edge_case_names:
            edge_case_result = OracleResult(
                passed=run_result.all_passed(edge_case_names),
                detail=f"{len(edge_case_names)} edge-case test(s) checked.",
            )
        else:
            edge_case_result = OracleResult(passed=True, detail="No edge-case tests were generated.")

        if poc_names:
            poc_result = OracleResult(
                passed=run_result.all_passed(poc_names),
                detail=f"PoC test(s) {poc_names}: "
                f"{'no longer trigger the vulnerability' if run_result.all_passed(poc_names) else 'still trigger it'}.",
            )
        else:
            # Unlike regression/edge-case, an *absent* PoC test must never
            # vacuously pass -- that would let a finding "succeed" purely
            # because Test-Writer failed to produce a working exploit test.
            poc_result = OracleResult(
                passed=False,
                detail="No self-validated PoC test is available; the security oracle cannot "
                "confirm the vulnerability was actually fixed.",
            )

        sast_result = self._security_rescan_checker.check(
            patched_source, self._language_profile.source_file_name(), state.finding
        )

        report = VerifierReport(
            functional=FunctionalOracleResult(regression=regression_result, edge_case=edge_case_result),
            security=SecurityOracleResult(poc_no_longer_triggers=poc_result, sast_rescan_clean=sast_result),
            sanitizer=self._sanitizer_checker.check(patched_source),
            raw_stdout=run_result.execution.stdout,
            raw_stderr=run_result.execution.stderr,
        )

        logger.info(
            "verification_complete",
            extra={
                "finding_id": state.finding.finding_id,
                "functional_pass": report.functional.passed,
                "security_pass": report.security.passed,
            },
        )
        return {"verifier_report": report}
