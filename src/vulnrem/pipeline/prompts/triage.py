from __future__ import annotations

from vulnrem.core.enums import AgentRole
from vulnrem.languages.base import CWESpec
from vulnrem.llm.schemas import LLMRequest, Message
from vulnrem.pipeline.prompts.common import render_cwe_context
from vulnrem.pipeline.state import Finding

SYSTEM_PROMPT = """You are the Triage agent in an automated vulnerability-remediation \
pipeline. A static-analysis tool has flagged a possible security issue. Your only job is \
to decide, from reading the code, whether this is plausibly a real, exploitable issue \
worth spending further (expensive) analysis on, or a false positive.

Be conservative about dismissing findings: static analyzers have relatively low false \
positive rates on the CWE categories you will see, and a missed true positive is worse \
than a wasted downstream check. Only mark something a false positive when the code \
clearly neutralizes the risk (e.g. the "untrusted" input is actually a hardcoded \
constant, or is already passed through a bound parameter / parameterized API rather \
than string-concatenated).
"""


def build_request(finding: Finding, source_snippet: str, cwe_spec: CWESpec | None) -> LLMRequest:
    cwe_context = (
        render_cwe_context(cwe_spec.id, cwe_spec.name, cwe_spec.description)
        if cwe_spec
        else render_cwe_context(finding.cwe_id, "", "")
    )
    tool_summary = "; ".join(
        f"{tf.tool} ({tf.rule_id}): {tf.message}" for tf in finding.tool_findings
    )
    user_content = f"""Static analysis flagged the following in `{finding.file_path}`, \
lines {finding.line_start}-{finding.line_end}:

Tool finding(s): {tool_summary}

Vulnerability class:
{cwe_context}

Code (with surrounding context):
```
{source_snippet}
```

Decide whether this is a real, plausibly-exploitable issue or a false positive."""

    return LLMRequest(
        role=AgentRole.TRIAGE,
        temperature=0.0,
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ],
    )
