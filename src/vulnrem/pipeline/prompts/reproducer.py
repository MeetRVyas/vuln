from __future__ import annotations

from vulnrem.core.enums import AgentRole
from vulnrem.languages.base import CWESpec
from vulnrem.llm.schemas import LLMRequest, Message
from vulnrem.pipeline.prompts.common import render_cwe_context
from vulnrem.pipeline.state import Finding

# The exact marker-string contract the Reproducer's script must follow so
# the node can mechanically decide "confirmed" vs "not confirmed" from
# sandboxed stdout/exit-code alone, without any further LLM judgment.
POC_CONFIRMED_MARKER = "VULN_CONFIRMED"
POC_NOT_CONFIRMED_MARKER = "VULN_NOT_CONFIRMED"

SYSTEM_PROMPT = f"""You are the Reproducer agent. Your job is to write a small, self-contained \
Python script that DYNAMICALLY demonstrates a specific flagged vulnerability by actually \
triggering it against the original (unpatched) source file, which will be placed alongside \
your script as `{{source_file_name}}` in the same directory.

Your script must:
- import the vulnerable function(s)/class(es) it needs directly from that file,
- construct a concrete malicious input that exploits the flagged issue,
- call the vulnerable code with that input,
- print exactly the line `{POC_CONFIRMED_MARKER}` and exit with status 0 if the exploit \
succeeded (i.e. you observed the vulnerable behavior: e.g. an SQL injection returned rows \
it should not have, a command injection ran an extra command, a path traversal read a file \
outside the intended directory, a hard-coded credential was usable),
- otherwise print exactly the line `{POC_NOT_CONFIRMED_MARKER}` and exit with status 1.

Do not use any network access. Do not import anything that is not in the Python standard \
library plus whatever the source file itself already imports. Keep the script under 40 lines.
"""


def build_request(
    finding: Finding, source_file_name: str, original_source: str, cwe_spec: CWESpec | None
) -> LLMRequest:
    cwe_context = (
        render_cwe_context(cwe_spec.id, cwe_spec.name, cwe_spec.description)
        if cwe_spec
        else render_cwe_context(finding.cwe_id, "", "")
    )
    user_content = f"""Vulnerability to reproduce, in `{source_file_name}`, \
around lines {finding.line_start}-{finding.line_end}:

{cwe_context}

Static analysis message(s): {"; ".join(tf.message for tf in finding.tool_findings)}

Source code:
```python
{original_source}
```

Write the reproduction script now."""

    system = SYSTEM_PROMPT.format(source_file_name=source_file_name)
    return LLMRequest(
        role=AgentRole.REPRODUCER,
        temperature=0.2,
        max_output_tokens=4096,
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=user_content),
        ],
    )
