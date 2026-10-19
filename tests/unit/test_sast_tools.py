from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from vulnrem.core.enums import Severity
from vulnrem.core.errors import SASTOutputParseError, SASTToolNotAvailableError
from vulnrem.sast.tools.bandit import BanditTool
from vulnrem.sast.tools.opengrep import OpengrepTool


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout=stdout, stderr=stderr)


OPENGREP_PAYLOAD = {
    "results": [
        {
            "check_id": "python.lang.security.audit.sql-injection",
            "path": "target.py",
            "start": {"line": 10, "col": 5},
            "end": {"line": 10, "col": 40},
            "extra": {
                "message": "Untrusted input in SQL query",
                "severity": "ERROR",
                "metadata": {"cwe": ["CWE-89: Improper Neutralization of Special Elements"]},
            },
        }
    ],
    "errors": [],
}


def test_opengrep_parses_results_into_raw_findings():
    tool = OpengrepTool(runner=lambda *a, **kw: _completed(stdout=json.dumps(OPENGREP_PAYLOAD)))
    findings = tool.scan(Path("target.py"))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.tool == "opengrep"
    assert finding.cwe_id == "CWE-89"
    assert finding.file_path == "target.py"
    assert finding.line_start == 10
    assert finding.severity == Severity.HIGH


def test_opengrep_raises_on_missing_binary():
    def _raise_not_found(*a, **kw):
        raise FileNotFoundError("opengrep")

    tool = OpengrepTool(runner=_raise_not_found)
    with pytest.raises(SASTToolNotAvailableError):
        tool.scan(Path("target.py"))


def test_opengrep_raises_on_unparsable_output():
    tool = OpengrepTool(runner=lambda *a, **kw: _completed(stdout="not json", stderr="uh oh"))
    with pytest.raises(SASTOutputParseError):
        tool.scan(Path("target.py"))


def test_opengrep_handles_empty_results():
    tool = OpengrepTool(runner=lambda *a, **kw: _completed(stdout=json.dumps({"results": []})))
    assert tool.scan(Path("target.py")) == []


BANDIT_PAYLOAD = {
    "results": [
        {
            "test_id": "B608",
            "filename": "target.py",
            "line_number": 12,
            "line_range": [12],
            "issue_text": "Possible SQL injection vector through string-based query construction.",
            "issue_severity": "MEDIUM",
            "issue_confidence": "HIGH",
        },
        {
            "test_id": "B105",
            "filename": "target.py",
            "line_number": 3,
            "line_range": [3],
            "issue_text": "Possible hardcoded password",
            "issue_severity": "LOW",
            "issue_confidence": "MEDIUM",
            "issue_cwe": {"id": 798, "link": "https://cwe.mitre.org/data/definitions/798.html"},
        },
    ],
    "errors": [],
}


def test_bandit_parses_results_and_uses_native_cwe_field_when_present():
    tool = BanditTool(runner=lambda *a, **kw: _completed(stdout=json.dumps(BANDIT_PAYLOAD)))
    findings = tool.scan(Path("target.py"))

    assert len(findings) == 2
    sql_finding = next(f for f in findings if f.rule_id == "B608")
    assert sql_finding.cwe_id == "CWE-89"  # from fallback table (no issue_cwe on this fixture entry)

    hardcoded_finding = next(f for f in findings if f.rule_id == "B105")
    assert hardcoded_finding.cwe_id == "CWE-798"  # from native issue_cwe field


def test_bandit_raises_on_missing_binary():
    def _raise_not_found(*a, **kw):
        raise FileNotFoundError("bandit")

    tool = BanditTool(runner=_raise_not_found)
    with pytest.raises(SASTToolNotAvailableError):
        tool.scan(Path("target.py"))


def test_bandit_nonzero_exit_with_findings_is_not_an_error():
    # Bandit exits 1 when it finds issues -- this must NOT raise.
    tool = BanditTool(runner=lambda *a, **kw: _completed(stdout=json.dumps(BANDIT_PAYLOAD), returncode=1))
    findings = tool.scan(Path("target.py"))
    assert len(findings) == 2
