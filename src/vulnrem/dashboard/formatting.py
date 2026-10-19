from __future__ import annotations

from vulnrem.core.enums import FindingStatus
from vulnrem.observability.secret_redaction import SecretRedactor
from vulnrem.pipeline.runner import FileCheckReport

_STATUS_LABELS = {
    FindingStatus.ACCEPTED: "✅ Fixed",
    FindingStatus.TRIAGED_FALSE_POSITIVE: "⚪ Not a real issue (triaged/reproduction)",
    FindingStatus.UNRESOLVED_RETRIES_EXHAUSTED: "🟠 Could not fix (retries exhausted)",
    FindingStatus.UNRESOLVED_ESCALATION_UNAVAILABLE: "🟠 Needs a capability this build doesn't have yet",
    FindingStatus.ERROR: "🔴 Internal error",
    FindingStatus.PENDING: "⏳ Pending",
    FindingStatus.IN_PROGRESS: "⏳ In progress",
}


def format_report_markdown(report: FileCheckReport) -> str:
    redactor = SecretRedactor()
    lines = [f"### `{report.file_path}` -- {len(report.findings)} finding(s)"]

    if report.preprocess_warnings:
        lines.append("_" + "; ".join(report.preprocess_warnings) + "_")

    if not report.findings:
        lines.append("\nNo findings from the configured static-analysis tools. \U0001f389")
        return "\n".join(lines)

    for result in report.findings:
        finding, state = result.finding, result.final_state
        label = _STATUS_LABELS.get(state.status, state.status.value)
        detail = state.status_detail
        if state.status == FindingStatus.ERROR and "429 RESOURCE_EXHAUSTED" in detail:
            label = "🟠 Cloud quota exhausted"
            detail = "Gemini request quota reached. Try again after the quota resets or use a key with available quota."
        elif state.status == FindingStatus.ERROR and "503 UNAVAILABLE" in detail:
            label = "🟠 Cloud provider unavailable"
            detail = "Gemini is temporarily unavailable. Try again shortly."
        lines.append(
            f"\n---\n#### {finding.cwe_id or 'Unclassified'} -- lines "
            f"{finding.line_start}-{finding.line_end}\n**{label}**"
        )
        if detail:
            lines.append(f"> {redactor.redact(detail)}")

        if state.status == FindingStatus.ACCEPTED and state.patch_candidate is not None:
            diff = redactor.redact(state.patch_candidate.diff_unified)
            lines.append(f"\n**Applied fix:**\n```diff\n{diff}\n```")
            lines.append(f"\n_{redactor.redact(state.patch_candidate.explanation)}_")

        if state.attempt_history:
            lines.append(f"\n_Attempts made: {len(state.attempt_history)}_")

    return "\n".join(lines)
