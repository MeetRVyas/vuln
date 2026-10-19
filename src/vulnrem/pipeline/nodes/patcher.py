from __future__ import annotations

import difflib
import logging

from vulnrem.languages.base import CWESpec
from vulnrem.llm.base import LLMProvider
from vulnrem.pipeline.prompts import patcher as patcher_prompt
from vulnrem.pipeline.prompts.schemas import PatcherOutput
from vulnrem.pipeline.state import PatchCandidate, PipelineState

logger = logging.getLogger(__name__)


class PatcherNode:
    def __init__(self, llm: LLMProvider, cwe_lookup: dict[str, CWESpec]):
        self._llm = llm
        self._cwe_lookup = cwe_lookup

    def __call__(self, state: PipelineState) -> dict:
        assert state.localization_context is not None, "PatcherNode requires localization_context"
        cwe_spec = self._cwe_lookup.get(state.finding.cwe_id) if state.finding.cwe_id else None

        request = patcher_prompt.build_request(
            finding=state.finding,
            original_source=state.original_source,
            localization_context=state.localization_context,
            cwe_spec=cwe_spec,
            attempt_history=state.attempt_history,
        )
        llm_output = self._llm.generate_structured(request, PatcherOutput)

        diff = "".join(
            difflib.unified_diff(
                state.original_source.splitlines(keepends=True),
                llm_output.patched_source.splitlines(keepends=True),
                fromfile=f"a/{state.file_path}",
                tofile=f"b/{state.file_path}",
            )
        )

        logger.info(
            "patch_generated",
            extra={
                "finding_id": state.finding.finding_id,
                "attempt_number": state.current_attempt_number(),
                "diff_lines": diff.count("\n"),
            },
        )

        return {
            "patch_candidate": PatchCandidate(
                patched_source=llm_output.patched_source,
                explanation=llm_output.explanation,
                diff_unified=diff,
            )
        }
