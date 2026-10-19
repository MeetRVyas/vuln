from __future__ import annotations

import ast

from vulnrem.search.base import RepoSearchTool, SearchResult


class SingleFileSearchTool(RepoSearchTool):
    def __init__(self, file_path: str, content: str):
        self._file_path = file_path
        self._content = content
        self._lines = content.splitlines()

    def search_symbol(self, name: str) -> list[SearchResult]:
        try:
            tree = ast.parse(self._content)
        except SyntaxError:
            return self.search_text(name)

        results: list[SearchResult] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                end = getattr(node, "end_lineno", None) or node.lineno
                snippet = "\n".join(self._lines[node.lineno - 1 : end])
                results.append(
                    SearchResult(file_path=self._file_path, line_start=node.lineno, line_end=end, snippet=snippet)
                )
        return results

    def search_text(self, query: str, context_lines: int = 1) -> list[SearchResult]:
        results: list[SearchResult] = []
        for idx, line in enumerate(self._lines):
            if query in line:
                start = max(0, idx - context_lines)
                end = min(len(self._lines), idx + context_lines + 1)
                snippet = "\n".join(self._lines[start:end])
                results.append(
                    SearchResult(file_path=self._file_path, line_start=start + 1, line_end=end, snippet=snippet)
                )
        return results
