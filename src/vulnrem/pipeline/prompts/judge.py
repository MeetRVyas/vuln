from __future__ import annotations

from vulnrem.core.enums import AgentRole
from vulnrem.llm.schemas import LLMRequest, Message
from vulnrem.pipeline.state import Finding, PatchCandidate, VerifierReport

SYSTEM_PROMPT = """You are the Judge agent -- the final decision-maker for one patch attempt. \
You do not execute anything yourself; you read the diff and the Verifier's dual-oracle report \
(functional oracle: did existing/regression/edge-case behavior survive; security oracle: is \
the vulnerability actually gone, both dynamically via the PoC test and statically via a fresh \
scan) and decide.

Return one of:
- decision="accept": both oracles passed AND the fix is a genuine, non-degenerate solution \
(not just special-casing the exact PoC input) -- category="none".
- decision="retry" with the category that best explains the failure:
  - "syntax_or_plumbing": the patch didn't even run correctly (import error, syntax error, \
wrong file structure) -- a cheap, mechanical problem, not a reasoning failure.
  - "regression_failure": the fix broke previously-correct, unrelated behavior.
  - "vulnerability_still_present": the security oracle failed -- the vulnerability is still \
exploitable or the static rescan still flags it.
  - "suspicious_or_degenerate": both oracles technically passed, but the patch looks like it \
only defeats the specific PoC/edge-case inputs rather than the underlying vulnerability class \
(e.g. hardcoding a check against the exact malicious string) -- reject it anyway.
  - "multi_file_scope_needed": a correct fix is not possible within this single file (e.g. the \
real fix belongs in a caller, a config file, or a different module).

Be skeptical of a patch that passes both oracles for the wrong reason. When in doubt between \
"accept" and "suspicious_or_degenerate", re-read whether the fix generalizes to the vulnerability \
class described, not just the specific test inputs.
"""


def build_request(
    finding: Finding, patch_candidate: PatchCandidate, verifier_report: VerifierReport
) -> LLMRequest:
    report = verifier_report
    report_summary = f"""Functional oracle: {"PASS" if report.functional.passed else "FAIL"}
  regression: {"pass" if report.functional.regression.passed else "FAIL"} -- {report.functional.regression.detail}
  edge_case:  {"pass" if report.functional.edge_case.passed else "FAIL"} -- {report.functional.edge_case.detail}
Security oracle: {"PASS" if report.security.passed else "FAIL"}
  dynamic PoC no longer triggers: {"pass" if report.security.poc_no_longer_triggers.passed else "FAIL"} -- \
{report.security.poc_no_longer_triggers.detail}
  static rescan clean:            {"pass" if report.security.sast_rescan_clean.passed else "FAIL"} -- \
{report.security.sast_rescan_clean.detail}
Sanitizer: {report.sanitizer}"""

    user_content = f"""Finding: {finding.cwe_id} in `{finding.file_path}`, \
lines {finding.line_start}-{finding.line_end}.

Patch explanation given by the Patcher: {patch_candidate.explanation}

Diff:
```diff
{patch_candidate.diff_unified}
```

Verifier report:
{report_summary}

Decide: accept or retry (with category)."""

    return LLMRequest(
        role=AgentRole.JUDGE,
        temperature=0.0,
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ],
    )
