from __future__ import annotations

from vulnrem.core.enums import Language, Severity
from vulnrem.pipeline.aggregator import aggregate_findings
from vulnrem.sast.schemas import RawFinding


def _raw(tool, rule_id, cwe_id, file_path="target.py", line_start=10, line_end=10, severity=Severity.MEDIUM):
    return RawFinding(
        tool=tool,
        rule_id=rule_id,
        cwe_id=cwe_id,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        message="msg",
        severity=severity,
    )


def test_overlapping_findings_from_two_tools_merge_into_one():
    raws = [
        _raw("opengrep", "sql-injection", "CWE-89", line_start=10, line_end=10, severity=Severity.HIGH),
        _raw("bandit", "B608", "CWE-89", line_start=10, line_end=10, severity=Severity.MEDIUM),
    ]
    findings = aggregate_findings(raws, Language.PYTHON)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.cwe_id == "CWE-89"
    assert finding.severity == Severity.HIGH  # max of the two
    assert set(finding.contributing_tools()) == {"opengrep", "bandit"}


def test_distant_findings_stay_separate():
    raws = [
        _raw("opengrep", "sql-injection", "CWE-89", line_start=10, line_end=10),
        _raw("bandit", "B105", "CWE-798", line_start=50, line_end=50),
    ]
    findings = aggregate_findings(raws, Language.PYTHON)
    assert len(findings) == 2


def test_findings_within_proximity_but_not_overlapping_still_merge():
    raws = [
        _raw("opengrep", "rule-a", "CWE-89", line_start=10, line_end=10),
        _raw("bandit", "rule-b", "CWE-89", line_start=12, line_end=12),  # 2 lines away, default proximity=2
    ]
    findings = aggregate_findings(raws, Language.PYTHON)
    assert len(findings) == 1


def test_exact_duplicate_from_same_tool_is_deduped():
    raws = [
        _raw("bandit", "B608", "CWE-89", line_start=10, line_end=10),
        _raw("bandit", "B608", "CWE-89", line_start=10, line_end=10),
    ]
    findings = aggregate_findings(raws, Language.PYTHON)
    assert len(findings) == 1
    assert len(findings[0].tool_findings) == 1


def test_dominant_cwe_picked_on_disagreement():
    raws = [
        _raw("opengrep", "rule-a", "CWE-89", line_start=10, line_end=10),
        _raw("bandit", "rule-b", "CWE-89", line_start=10, line_end=10),
        _raw("some-other-tool", "rule-c", "CWE-78", line_start=10, line_end=10),
    ]
    findings = aggregate_findings(raws, Language.PYTHON)
    assert len(findings) == 1
    assert findings[0].cwe_id == "CWE-89"  # majority vote


def test_finding_with_no_cwe_from_any_tool_has_none_cwe():
    raws = [_raw("opengrep", "generic-rule", None, line_start=10, line_end=10)]
    findings = aggregate_findings(raws, Language.PYTHON)
    assert findings[0].cwe_id is None


def test_finding_ids_are_deterministic_across_runs():
    raws = [_raw("opengrep", "rule-a", "CWE-89", line_start=10, line_end=10)]
    first = aggregate_findings(raws, Language.PYTHON)[0].finding_id
    second = aggregate_findings(raws, Language.PYTHON)[0].finding_id
    assert first == second


def test_different_files_kept_independent():
    raws = [
        _raw("opengrep", "rule-a", "CWE-89", file_path="a.py", line_start=10, line_end=10),
        _raw("opengrep", "rule-a", "CWE-89", file_path="b.py", line_start=10, line_end=10),
    ]
    findings = aggregate_findings(raws, Language.PYTHON)
    assert len(findings) == 2
