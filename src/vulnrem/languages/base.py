"""Language abstraction.

Everything that differs between Python, C, and Java (file extensions, how to
slice source into localization context, how to run a test file, how to
syntax-check a candidate patch) lives behind :class:`LanguageProfile`. The
rest of the pipeline (nodes, orchestrator, oracles) only ever depends on this
interface, never on a concrete language.

Only Python is implemented in this build (see
:mod:`vulnrem.languages.python_profile`). Adding C or Java means: write a
``configs/languages/<lang>.yaml``, write one ``LanguageProfile``
implementation, and add one line to
:data:`vulnrem.languages.registry.LANGUAGE_PROFILE_REGISTRY`. No pipeline
node needs to change -- this is the concrete instance of the "config/
registry over scattered branching" principle for language-specific code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from vulnrem.core.enums import Language
from vulnrem.core.errors import ConfigError


class CWESpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # e.g. "CWE-89"
    name: str
    description: str


class LanguageSpec(BaseModel):
    """Parsed contents of ``configs/languages/<language>.yaml``."""

    model_config = ConfigDict(extra="forbid")

    language: Language
    file_extensions: list[str]
    test_framework: str
    cwes: list[CWESpec]
    debugger_status: Literal["not_implemented"] = "not_implemented"
    sanitizer_status: Literal["not_applicable", "not_implemented", "available"] = "not_applicable"

    def cwe_ids(self) -> list[str]:
        return [c.id for c in self.cwes]


def load_language_spec(language: Language, config_dir: Path | str | None = None) -> LanguageSpec:
    directory = Path(config_dir) if config_dir else Path(__file__).resolve().parents[3] / "configs"
    path = directory / "languages" / f"{language.value}.yaml"
    if not path.is_file():
        raise ConfigError(
            f"No language spec found for {language.value!r} at {path}. "
            "Adding a new language requires a configs/languages/<name>.yaml file."
        )
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw["language"] = language.value
    return LanguageSpec.model_validate(raw)


class SyntaxCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    error_message: str | None = None


class LocalizationContext(BaseModel):
    """What the Localizer hands to the Patcher/Test-Writer: the finding's
    enclosing function/class, plus best-effort same-file callees/callers.
    Deliberately single-file in this build -- see
    :mod:`vulnrem.search.single_file` for why, and how to extend it."""

    model_config = ConfigDict(extra="forbid")

    primary_snippet: str
    primary_start_line: int
    primary_end_line: int
    enclosing_symbol_name: str | None = None
    related_snippets: list[str] = []
    notes: str = ""


class LanguageProfile(ABC):
    """One concrete implementation per supported language."""

    def __init__(self, spec: LanguageSpec):
        self.spec = spec

    @property
    def language(self) -> Language:
        return self.spec.language

    @abstractmethod
    def slice_context(
        self, source: str, line_start: int, line_end: int
    ) -> LocalizationContext:
        """Extract the localization context around the given 1-indexed,
        inclusive line range, using same-file static analysis only."""

    @abstractmethod
    def syntax_check(self, source: str) -> SyntaxCheckResult:
        """Cheap, no-execution check that source parses. Used as the fast
        first gate in the Verifier / retry policy (a syntax error is a
        `RetryCategory.SYNTAX_OR_PLUMBING`, not a real functional failure)."""

    @abstractmethod
    def test_command(self, test_file_relpath: str) -> list[str]:
        """The argv to run inside the Sandbox to execute one test file."""

    @abstractmethod
    def source_file_name(self, base_name: str = "target") -> str:
        """Conventional file name to write a source snippet under, e.g.
        ``target.py``."""
