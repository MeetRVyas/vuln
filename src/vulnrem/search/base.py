"""The repo-search abstraction.

This build only ever processes one file at a time (per the project's
explicit scope decision), so :class:`~vulnrem.search.single_file.
SingleFileSearchTool` is the only implementation -- it searches within the
one file already in memory and nothing else. The interface is written as
if a real multi-file search existed, so that adding repo-scale search later
is: implement a second :class:`RepoSearchTool` (e.g. backed by ripgrep plus
an index), inject it wherever ``SingleFileSearchTool`` is constructed
today, and nothing in :mod:`vulnrem.pipeline.nodes.localizer` changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    line_start: int
    line_end: int
    snippet: str


class RepoSearchTool(ABC):
    @abstractmethod
    def search_symbol(self, name: str) -> list[SearchResult]:
        """Every definition or reference of ``name`` within the tool's
        scope."""

    @abstractmethod
    def search_text(self, query: str, context_lines: int = 1) -> list[SearchResult]:
        """Plain substring search with ``context_lines`` of surrounding
        context on each side of a match."""
