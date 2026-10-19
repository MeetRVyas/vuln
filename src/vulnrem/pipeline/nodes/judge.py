from __future__ import annotations

import logging

from vulnrem.llm.base import LLMProvider
from vulnrem.pipeline.prompts import judge as judge_prompt
from vulnrem.pipeline.state import AttemptRecord, JudgeVerdict, PipelineState

logger = logging.getLogger(__name__)


class JudgeNode:
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def __call__(self, state: PipelineState) -> dict:
        assert state.patch_candidate is not None
        assert state.verifier_report is not None

        request = judge_prompt.build_request(state.finding, state.patch_candidate, state.verifier_report)
        verdict = self._llm.generate_structured(request, JudgeVerdict)

        record = AttemptRecord(
            attempt_number=state.current_attempt_number(),
            patch_candidate=state.patch_candidate,
            verifier_report=state.verifier_report,
            judge_verdict=verdict,
        )

        logger.info(
            "judge_verdict",
            extra={
                "finding_id": state.finding.finding_id,
                "attempt_number": record.attempt_number,
                "decision": verdict.decision,
                "category": verdict.category.value,
            },
        )
        return {"judge_verdict": verdict, "attempt_history": [record]}
