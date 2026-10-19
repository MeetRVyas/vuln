from __future__ import annotations

import logging

from vulnrem.languages.base import CWESpec, LanguageProfile
from vulnrem.llm.base import LLMProvider
from vulnrem.pipeline.prompts import reproducer as reproducer_prompt
from vulnrem.pipeline.prompts.schemas import ReproducerOutput
from vulnrem.pipeline.state import PipelineState, ReproductionResult
from vulnrem.sandbox.base import Sandbox
from vulnrem.sandbox.schemas import ExecutionRequest

logger = logging.getLogger(__name__)

_POC_FILE_NAME = "reproduce.py"


class ReproducerNode:
    def __init__(
        self,
        llm: LLMProvider,
        sandbox: Sandbox,
        language_profile: LanguageProfile,
        cwe_lookup: dict[str, CWESpec],
        timeout_s: float,
        memory_limit_mb: int,
        cpu_limit_seconds: float,
    ):
        self._llm = llm
        self._sandbox = sandbox
        self._language_profile = language_profile
        self._cwe_lookup = cwe_lookup
        self._timeout_s = timeout_s
        self._memory_limit_mb = memory_limit_mb
        self._cpu_limit_seconds = cpu_limit_seconds

    def __call__(self, state: PipelineState) -> dict:
        source_name = self._language_profile.source_file_name()
        cwe_spec = self._cwe_lookup.get(state.finding.cwe_id) if state.finding.cwe_id else None

        request = reproducer_prompt.build_request(state.finding, source_name, state.original_source, cwe_spec)
        llm_output = self._llm.generate_structured(request, ReproducerOutput)

        execution = self._sandbox.run(
            ExecutionRequest(
                language=state.language,
                files={source_name: state.original_source, _POC_FILE_NAME: llm_output.poc_script},
                command=["python3", _POC_FILE_NAME],
                timeout_s=self._timeout_s,
                memory_limit_mb=self._memory_limit_mb,
                cpu_limit_seconds=self._cpu_limit_seconds,
            )
        )

        confirmed = (
            execution.exit_code == 0
            and reproducer_prompt.POC_CONFIRMED_MARKER in execution.stdout
            and not execution.timed_out
        )

        logger.info(
            "reproduction_attempted",
            extra={"finding_id": state.finding.finding_id, "confirmed": confirmed},
        )

        return {
            "reproduction": ReproductionResult(
                confirmed=confirmed,
                poc_script=llm_output.poc_script,
                explanation=llm_output.explanation,
                raw_stdout=execution.stdout,
                raw_stderr=execution.stderr,
            )
        }
