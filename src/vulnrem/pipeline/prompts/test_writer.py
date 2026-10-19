from __future__ import annotations

from vulnrem.core.enums import AgentRole
from vulnrem.languages.base import LocalizationContext
from vulnrem.llm.schemas import LLMRequest, Message
from vulnrem.pipeline.state import Finding, ReproductionResult

SYSTEM_PROMPT = """You are the Test-Writer agent. You produce a battery of pytest test \
functions for one flagged-and-confirmed vulnerability, to be used both to verify a future \
patch and to guard against regressions.

Your test functions will be concatenated together into a single file alongside a plain \
`import pytest` at the top -- there is no other shared top-level import block. So each test \
function must do any imports it needs (including importing names from the target module, \
which is importable by its module name without the `.py` extension) INSIDE its own function \
body, not at module level.

Produce exactly three kinds of test, each a complete, self-contained pytest test function \
(just the `def test_...(): ...` block, with its own imports inside the body):

1. kind="poc" (exactly one): a pytest version of the already-confirmed exploit. It must \
assert that the vulnerable behavior does NOT happen (e.g. `assert not rows_leaked`, \
`assert not extra_command_ran`). Because the vulnerability is real, this assertion FAILS \
against the original file -- that is expected and required (expected_result_on_original="fail"). \
After a correct patch, this same test should PASS.

2. kind="regression" (one or more): tests of the function's normal, legitimate behavior \
that has nothing to do with the vulnerability (e.g. a valid, non-malicious call returns the \
expected result). These must PASS against the original file (expected_result_on_original="pass") \
and must keep passing after any legitimate patch.

3. kind="edge_case" (one or more): tests that would catch a *degenerate* "fix" that merely \
special-cases the exact exploit string rather than genuinely closing the vulnerability class \
(e.g. a slightly different malicious payload, or a legitimate input that superficially \
resembles the malicious one). These must PASS against the original file \
(expected_result_on_original="pass") in the sense that they describe correct/expected \
behavior the fix must not break -- they are not expected to detect the vulnerability \
themselves, just to trap an overly narrow patch.

Give every test function a unique, descriptive name starting with `test_`.
"""


def build_request(
    finding: Finding,
    original_source: str,
    localization_context: LocalizationContext,
    reproduction: ReproductionResult,
    module_name: str,
) -> LLMRequest:
    user_content = f"""File: `{finding.file_path}` (importable as module `{module_name}`)

Full original source:
```
{original_source}
```

Vulnerable region (lines {finding.line_start}-{finding.line_end}), inside \
`{localization_context.enclosing_symbol_name or "module level"}`:
```
{localization_context.primary_snippet}
```

The Reproducer already confirmed this vulnerability with the following script and explanation \
-- adapt its logic into your kind="poc" pytest test rather than re-deriving it from scratch:

Reproducer's script:
```
{reproduction.poc_script}
```

Reproducer's explanation: {reproduction.explanation}

Write the full test battery now."""

    return LLMRequest(
        role=AgentRole.TEST_WRITER,
        temperature=0.2,
        max_output_tokens=4096,
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ],
    )
