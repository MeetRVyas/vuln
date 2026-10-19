from __future__ import annotations

import logging

from vulnrem.languages.base import LanguageProfile
from vulnrem.pipeline.state import PipelineState
from vulnrem.search.base import RepoSearchTool

logger = logging.getLogger(__name__)

_MAX_SUPPLEMENTARY_MATCHES = 2


class LocalizerNode:
    """No LLM call -- purely deterministic. Uses the language's AST slicer
    as the primary mechanism and a same-file text search
    (:class:`~vulnrem.search.base.RepoSearchTool`) as a supplementary pass
    when the AST slice found no enclosing definition (module-level code),
    so the Patcher still gets *some* nearby context to work with."""

    def __init__(self, language_profile: LanguageProfile, search_tool_factory):
        self._language_profile = language_profile
        # A factory (not a fixed instance) because the search tool is
        # scoped to one file's *current* content, which differs per state
        # even though the LocalizerNode instance itself is shared/reused.
        self._search_tool_factory: type[RepoSearchTool] = search_tool_factory

    def __call__(self, state: PipelineState) -> dict:
        context = self._language_profile.slice_context(
            state.original_source, state.finding.line_start, state.finding.line_end
        )

        if context.enclosing_symbol_name is None:
            search_tool = self._search_tool_factory(state.file_path, state.original_source)
            window = context.primary_snippet
            extra_matches = search_tool.search_text(window.strip().splitlines()[0].strip()[:40]) if window else []
            if extra_matches:
                extra_snippets = [m.snippet for m in extra_matches[:_MAX_SUPPLEMENTARY_MATCHES]]
                context = context.model_copy(
                    update={
                        "related_snippets": [*context.related_snippets, *extra_snippets],
                        "notes": "Finding is at module level; added nearby text matches as extra context.",
                    }
                )

        logger.info(
            "localization_complete",
            extra={
                "finding_id": state.finding.finding_id,
                "enclosing_symbol": context.enclosing_symbol_name,
                "related_snippet_count": len(context.related_snippets),
            },
        )
        return {"localization_context": context}
