from __future__ import annotations

import logging

from vulnrem.languages.base import CWESpec
from vulnrem.llm.base import LLMProvider
from vulnrem.pipeline.nodes._util import line_window
from vulnrem.pipeline.prompts import triage as triage_prompt
from vulnrem.pipeline.prompts.schemas import TriageVerdict
from vulnrem.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class TriageNode:
    def __init__(self, llm: LLMProvider, cwe_lookup: dict[str, CWESpec]):
        self._llm = llm
        self._cwe_lookup = cwe_lookup

    def __call__(self, state: PipelineState) -> dict:
        cwe_spec = self._cwe_lookup.get(state.finding.cwe_id) if state.finding.cwe_id else None
        snippet = line_window(state.original_source, state.finding.line_start, state.finding.line_end)
        request = triage_prompt.build_request(state.finding, snippet, cwe_spec)
        verdict = self._llm.generate_structured(request, TriageVerdict)

        logger.info(
            "triage_complete",
            extra={
                "finding_id": state.finding.finding_id,
                "is_true_positive": verdict.is_true_positive,
                "confidence": verdict.confidence,
            },
        )
        return {"triage_confirmed": verdict.is_true_positive, "triage_rationale": verdict.rationale}
