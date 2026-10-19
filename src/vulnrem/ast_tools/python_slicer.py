"""Best-effort, same-file-only static slicing for Python source.

This is deliberately simple: find the smallest enclosing function/class
around a finding's line range, then pull in the source of any other
*top-level* function/class in the same file that the enclosing definition
calls by name. No import resolution, no cross-file following -- that is
exactly the "repo-scale" capability the project plan defers (see
:mod:`vulnrem.search.base` for the extension point).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

_MAX_RELATED_SNIPPETS = 3


@dataclass
class _Definition:
    name: str
    node: ast.AST
    start_line: int
    end_line: int


class PythonSliceError(ValueError):
    """Raised when the source cannot be parsed at all."""


def _parse(source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise PythonSliceError(f"source does not parse: {exc}") from exc


def _end_line(node: ast.AST, fallback: int) -> int:
    return getattr(node, "end_lineno", None) or fallback


def _top_level_definitions(tree: ast.Module) -> dict[str, _Definition]:
    defs: dict[str, _Definition] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name] = _Definition(
                name=node.name,
                node=node,
                start_line=node.lineno,
                end_line=_end_line(node, node.lineno),
            )
    return defs


def find_enclosing_definition(
    tree: ast.Module, line_start: int, line_end: int
) -> ast.AST | None:
    """Smallest FunctionDef/AsyncFunctionDef/ClassDef whose line range fully
    contains [line_start, line_end]. Falls back to None (caller treats the
    whole module as the snippet) if nothing encloses it -- e.g. a
    module-level statement."""
    best: ast.AST | None = None
    best_span = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_start = node.lineno
            node_end = _end_line(node, node_start)
            if node_start <= line_start and line_end <= node_end:
                span = node_end - node_start
                if best_span is None or span < best_span:
                    best, best_span = node, span
    return best


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@dataclass
class SliceResult:
    primary_snippet: str
    primary_start_line: int
    primary_end_line: int
    enclosing_symbol_name: str | None
    related_snippets: list[str]


def slice_python_source(source: str, line_start: int, line_end: int) -> SliceResult:
    """Main entry point used by :class:`~vulnrem.languages.python_profile.
    PythonLanguageProfile`."""
    tree = _parse(source)
    lines = source.splitlines()

    enclosing = find_enclosing_definition(tree, line_start, line_end)

    if enclosing is not None:
        primary_start = enclosing.lineno
        primary_end = _end_line(enclosing, primary_start)
        symbol_name = getattr(enclosing, "name", None)
    else:
        # No enclosing def -- fall back to the whole module (single-file
        # scope means this is still bounded and small in practice).
        primary_start, primary_end = 1, len(lines)
        symbol_name = None

    primary_snippet = "\n".join(lines[primary_start - 1 : primary_end])

    related_snippets: list[str] = []
    if enclosing is not None:
        top_level = _top_level_definitions(tree)
        top_level.pop(symbol_name, None)  # don't include itself
        called = _called_names(enclosing)
        for name in called:
            definition = top_level.get(name)
            if definition is None:
                continue
            snippet = "\n".join(
                lines[definition.start_line - 1 : definition.end_line]
            )
            related_snippets.append(f"# related definition: {name}\n{snippet}")
            if len(related_snippets) >= _MAX_RELATED_SNIPPETS:
                break

    return SliceResult(
        primary_snippet=primary_snippet,
        primary_start_line=primary_start,
        primary_end_line=primary_end,
        enclosing_symbol_name=symbol_name,
        related_snippets=related_snippets,
    )


def check_syntax(source: str) -> tuple[bool, str | None]:
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError as exc:
        return False, str(exc)
    return True, None
