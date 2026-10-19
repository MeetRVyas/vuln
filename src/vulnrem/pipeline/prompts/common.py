from __future__ import annotations

from vulnrem.pipeline.state import AttemptRecord


def render_cwe_context(cwe_id: str | None, cwe_name: str, cwe_description: str) -> str:
    if not cwe_id:
        return "No specific CWE was identified; treat this as a general security concern flagged by static analysis."
    return f"{cwe_id} -- {cwe_name}\n{cwe_description}"


def render_attempt_history(attempt_history: list[AttemptRecord], max_attempts_shown: int = 3) -> str:
    """Renders previous attempts as concrete feedback for the next Patcher
    call: the actual previous diff and the actual failure detail, not a bare
    'attempt failed, try again.' Only the most recent ``max_attempts_shown``
    are included to bound prompt size."""
    if not attempt_history:
        return "This is the first attempt -- there is no prior history."

    blocks: list[str] = []
    for record in attempt_history[-max_attempts_shown:]:
        lines = [f"--- Attempt {record.attempt_number} ---"]
        if record.error:
            lines.append(f"Attempt errored before verification: {record.error}")
        if record.patch_candidate:
            lines.append("Previous patch:")
            lines.append(record.patch_candidate.diff_unified or record.patch_candidate.patched_source)
        if record.verifier_report:
            report = record.verifier_report
            lines.append(
                f"Verifier result: functional={report.functional.passed}, security={report.security.passed}"
            )
            if not report.functional.regression.passed:
                lines.append(f"  regression failure: {report.functional.regression.detail}")
            if not report.functional.edge_case.passed:
                lines.append(f"  edge-case failure: {report.functional.edge_case.detail}")
            if not report.security.poc_no_longer_triggers.passed:
                lines.append(f"  vulnerability still exploitable: {report.security.poc_no_longer_triggers.detail}")
            if not report.security.sast_rescan_clean.passed:
                lines.append(f"  static rescan still flags it: {report.security.sast_rescan_clean.detail}")
        if record.judge_verdict:
            lines.append(
                f"Judge: decision={record.judge_verdict.decision}, "
                f"category={record.judge_verdict.category.value}, "
                f"rationale={record.judge_verdict.rationale}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
