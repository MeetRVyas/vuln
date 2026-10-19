from __future__ import annotations

from vulnrem.core.enums import AgentRole
from vulnrem.languages.base import CWESpec, LocalizationContext
from vulnrem.llm.schemas import LLMRequest, Message
from vulnrem.pipeline.prompts.common import render_attempt_history, render_cwe_context
from vulnrem.pipeline.state import AttemptRecord, Finding

SYSTEM_PROMPT = """You are the Patcher agent. You fix exactly one flagged vulnerability in a \
single source file. You do not have access to any tools -- you see the full file, the \
specific vulnerable region, and (on a retry) exactly what went wrong last time, and you \
return a complete corrected version of the file.

Rules:
- Preserve all unrelated behavior, formatting, and functionality. Change only what is \
necessary to eliminate the flagged vulnerability class.
- Prefer the language/framework's standard safe idiom for this vulnerability class \
(e.g. parameterized queries for SQL injection, `shlex`/argument-list subprocess calls \
instead of shell=True for command injection, path normalization plus an allow-listed base \
directory for path traversal, environment/config injection instead of a literal secret for \
hard-coded credentials) rather than a narrow special-case check that only defeats the exact \
flagged pattern.
- Do not introduce new dependencies unless the standard library has no reasonable answer.
- Return the ENTIRE file content, not a diff or a partial snippet -- the whole file must \
still run correctly on its own.
"""


def build_request(
    finding: Finding,
    original_source: str,
    localization_context: LocalizationContext,
    cwe_spec: CWESpec | None,
    attempt_history: list[AttemptRecord],
) -> LLMRequest:
    cwe_context = (
        render_cwe_context(cwe_spec.id, cwe_spec.name, cwe_spec.description)
        if cwe_spec
        else render_cwe_context(finding.cwe_id, "", "")
    )
    related = "\n\n".join(localization_context.related_snippets) or "(none)"

    user_content = f"""File: `{finding.file_path}`

Full original source:
```
{original_source}
```

Flagged vulnerability, lines {finding.line_start}-{finding.line_end}:
{cwe_context}

The vulnerable code sits inside this enclosing definition
(`{localization_context.enclosing_symbol_name or "module level"}`):
```
{localization_context.primary_snippet}
```

Other same-file definitions it calls, for context:
{related}

Retry history:
{render_attempt_history(attempt_history)}

Return the complete corrected file."""

    return LLMRequest(
        role=AgentRole.PATCHER,
        temperature=0.2,
        max_output_tokens=4096,
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ],
    )
