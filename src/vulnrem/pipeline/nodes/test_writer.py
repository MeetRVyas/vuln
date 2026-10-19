from __future__ import annotations

import logging

from vulnrem.languages.base import LanguageProfile
from vulnrem.llm.base import LLMProvider
from vulnrem.oracles.pytest_support import run_pytest
from vulnrem.pipeline.prompts import test_writer as test_writer_prompt
from vulnrem.pipeline.prompts.schemas import TestWriterOutput
from vulnrem.pipeline.state import PipelineState, TestBattery, TestCase
from vulnrem.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class TestWriterNode:
    """Self-validation contract: a test case only survives into the
    battery the Verifier will later trust if running it against the
    *original* (unpatched) source produces the outcome the LLM itself
    claimed to expect (``expected_result_on_original``). This is what
    prevents a hallucinated or subtly-wrong test from silently becoming the
    ground truth the whole retry loop is judged against."""

    def __init__(
        self,
        llm: LLMProvider,
        sandbox: Sandbox,
        language_profile: LanguageProfile,
        timeout_s: float,
        memory_limit_mb: int,
        cpu_limit_seconds: float,
        max_self_validation_attempts: int = 2,
    ):
        self._llm = llm
        self._sandbox = sandbox
        self._language_profile = language_profile
        self._timeout_s = timeout_s
        self._memory_limit_mb = memory_limit_mb
        self._cpu_limit_seconds = cpu_limit_seconds
        self._max_self_validation_attempts = max_self_validation_attempts

    def __call__(self, state: PipelineState) -> dict:
        assert state.localization_context is not None
        assert state.reproduction is not None

        module_name = self._language_profile.source_file_name().rsplit(".", 1)[0]
        request = test_writer_prompt.build_request(
            finding=state.finding,
            original_source=state.original_source,
            localization_context=state.localization_context,
            reproduction=state.reproduction,
            module_name=module_name,
        )

        validated: list[TestCase] = []
        discarded: list[str] = []
        attempts_used = 0
        for attempt in range(1, self._max_self_validation_attempts + 1):
            attempts_used = attempt
            llm_output = self._llm.generate_structured(request, TestWriterOutput)
            validated, discarded = self._self_validate(state.original_source, llm_output.test_cases)
            has_poc = any(tc.kind == "poc" for tc in validated)
            if has_poc:
                break
            logger.warning(
                "test_writer_self_validation_missing_poc_retrying",
                extra={"finding_id": state.finding.finding_id, "attempt": attempt, "discarded": discarded},
            )

        combined_source = self._assemble(validated)

        logger.info(
            "test_battery_generated",
            extra={
                "finding_id": state.finding.finding_id,
                "kept": len(validated),
                "discarded": len(discarded),
                "self_validation_attempts": attempts_used,
            },
        )

        return {
            "test_battery": TestBattery(
                test_cases=validated,
                combined_source=combined_source,
                self_validation_attempts=attempts_used,
                discarded_case_names=discarded,
            )
        }

    def _self_validate(
        self, original_source: str, proposed: list[TestCase]
    ) -> tuple[list[TestCase], list[str]]:
        if not proposed:
            return [], []
        combined = self._assemble(proposed)
        run_result = run_pytest(
            sandbox=self._sandbox,
            language_profile=self._language_profile,
            source_code=original_source,
            test_code=combined,
            timeout_s=self._timeout_s,
            memory_limit_mb=self._memory_limit_mb,
            cpu_limit_seconds=self._cpu_limit_seconds,
        )
        validated: list[TestCase] = []
        discarded: list[str] = []
        for case in proposed:
            expected_pass = case.expected_result_on_original == "pass"
            actual_pass = run_result.passed(case.name)
            if actual_pass == expected_pass:
                validated.append(case)
            else:
                discarded.append(case.name)
        return validated, discarded

    @staticmethod
    def _assemble(test_cases: list[TestCase]) -> str:
        header = "import pytest\n\n"
        return header + "\n\n".join(tc.source for tc in test_cases)
