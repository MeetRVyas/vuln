from __future__ import annotations

from vulnrem.ast_tools.python_slicer import check_syntax, slice_python_source
from vulnrem.languages.base import (
    LanguageProfile,
    LanguageSpec,
    LocalizationContext,
    SyntaxCheckResult,
)


class PythonLanguageProfile(LanguageProfile):
    def __init__(self, spec: LanguageSpec | None = None):
        super().__init__(spec or _default_spec())

    def slice_context(self, source: str, line_start: int, line_end: int) -> LocalizationContext:
        result = slice_python_source(source, line_start, line_end)
        return LocalizationContext(
            primary_snippet=result.primary_snippet,
            primary_start_line=result.primary_start_line,
            primary_end_line=result.primary_end_line,
            enclosing_symbol_name=result.enclosing_symbol_name,
            related_snippets=result.related_snippets,
        )

    def syntax_check(self, source: str) -> SyntaxCheckResult:
        is_valid, error = check_syntax(source)
        return SyntaxCheckResult(is_valid=is_valid, error_message=error)

    def test_command(self, test_file_relpath: str) -> list[str]:
        # -v (not -q) is required: the Verifier and Test-Writer self-validation
        # both parse PASSED/FAILED per test name from this verbose output
        # (see vulnrem.oracles.pytest_support). -p no:cacheprovider avoids
        # writing a .pytest_cache/ directory in the throwaway sandbox workspace.
        return ["python3", "-m", "pytest", "-v", "--no-header", "-p", "no:cacheprovider", test_file_relpath]

    def source_file_name(self, base_name: str = "target") -> str:
        return f"{base_name}.py"


def _default_spec() -> LanguageSpec:
    """Fallback used only when no config directory is available (e.g. ad-hoc
    unit tests). Production code paths should load the real spec via
    :func:`vulnrem.languages.base.load_language_spec` and pass it in."""
    from vulnrem.core.enums import Language

    return LanguageSpec(
        language=Language.PYTHON,
        file_extensions=[".py"],
        test_framework="pytest",
        cwes=[],
    )
