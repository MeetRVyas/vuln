from __future__ import annotations

from vulnrem.core.enums import FindingStatus, Language, Severity
from vulnrem.dashboard.formatting import format_report_markdown
from vulnrem.pipeline.runner import FileCheckReport, FindingResult
from vulnrem.pipeline.state import Finding, PatchCandidate, PipelineState


def _finding() -> Finding:
    return Finding(
        finding_id="f1", language=Language.PYTHON, file_path="x.py", line_start=3, line_end=3,
        cwe_id="CWE-89", severity=Severity.HIGH, tool_findings=[],
    )


def test_format_report_with_no_findings():
    report = FileCheckReport(file_path="clean.py", language=Language.PYTHON, preprocess_warnings=[], findings=[])
    text = format_report_markdown(report)
    assert "No findings" in text


def test_format_report_shows_accepted_diff_and_redacts_secrets():
    finding = _finding()
    state = PipelineState(
        language=Language.PYTHON,
        file_path="x.py",
        original_source="code",
        finding=finding,
        status=FindingStatus.ACCEPTED,
        patch_candidate=PatchCandidate(
            patched_source="fixed",
            explanation="Removed hardcoded key AKIAABCDEFGHIJKLMNOP",
            diff_unified="-old\n+new AKIAABCDEFGHIJKLMNOP",
        ),
    )
    report = FileCheckReport(
        file_path="x.py", language=Language.PYTHON, preprocess_warnings=[],
        findings=[FindingResult(finding=finding, final_state=state)],
    )
    text = format_report_markdown(report)
    assert "CWE-89" in text
    assert "Fixed" in text
    assert "AKIAABCDEFGHIJKLMNOP" not in text
    assert "[REDACTED]" in text


def test_format_report_explains_cloud_quota_error():
    finding = _finding()
    state = PipelineState(
        language=Language.PYTHON,
        file_path="x.py",
        original_source="code",
        finding=finding,
        status=FindingStatus.ERROR,
        status_detail="Google provider request failed: 429 RESOURCE_EXHAUSTED. quota details",
    )
    report = FileCheckReport(
        file_path="x.py",
        language=Language.PYTHON,
        preprocess_warnings=[],
        findings=[FindingResult(finding=finding, final_state=state)],
    )

    text = format_report_markdown(report)

    assert "Cloud quota exhausted" in text
    assert "Internal error" not in text
    assert "quota details" not in text
