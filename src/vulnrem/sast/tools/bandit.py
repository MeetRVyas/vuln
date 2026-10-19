"""Wraps the ``bandit`` CLI -- Python-specific static analysis depth
alongside Opengrep's cross-language rules (per the project plan's decision
to run both).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from vulnrem.core.enums import Severity
from vulnrem.core.errors import SASTError, SASTOutputParseError, SASTToolNotAvailableError
from vulnrem.sast.base import SASTTool
from vulnrem.sast.schemas import RawFinding

_Runner = Callable[..., subprocess.CompletedProcess]

# Bandit's `issue_cwe` field is present in modern releases but not all rule
# plugins populate it; this is the fallback for the starter CWE set this
# project targets first (see configs/languages/python.yaml).
_TEST_ID_TO_CWE_FALLBACK: dict[str, str] = {
    "B608": "CWE-89",  # hardcoded_sql_expressions
    "B105": "CWE-798",  # hardcoded_password_string
    "B106": "CWE-798",  # hardcoded_password_funcarg
    "B107": "CWE-798",  # hardcoded_password_default
    "B602": "CWE-78",  # subprocess_popen_with_shell_equals_true
    "B603": "CWE-78",
    "B604": "CWE-78",
    "B605": "CWE-78",  # start_process_with_a_shell
    "B607": "CWE-78",  # start_process_with_partial_path
    "B609": "CWE-78",  # linux_commands_wildcard_injection
    "B108": "CWE-22",  # hardcoded_tmp_directory (loosely related; documented as approximate)
}


class BanditTool(SASTTool):
    def __init__(
        self,
        timeout_s: float = 30.0,
        extra_args: list[str] | None = None,
        runner: _Runner | None = None,
    ):
        self._timeout_s = timeout_s
        self._extra_args = extra_args or []
        self._runner: _Runner = runner or subprocess.run

    @property
    def tool_name(self) -> str:
        return "bandit"

    def is_available(self) -> bool:
        return shutil.which("bandit") is not None

    def _build_command(self, target_path: Path) -> list[str]:
        cmd = ["bandit", "-f", "json", "-q"]
        cmd.extend(self._extra_args)
        cmd.extend(["-r", str(target_path)])
        return cmd

    def scan(self, target_path: Path) -> list[RawFinding]:
        cmd = self._build_command(target_path)
        try:
            proc = self._runner(cmd, capture_output=True, text=True, timeout=self._timeout_s)
        except FileNotFoundError as exc:
            raise SASTToolNotAvailableError("bandit binary not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise SASTError(f"bandit timed out after {self._timeout_s}s") from exc

        # Bandit exits non-zero when findings exist; that's expected, not a
        # tool failure. Only an unparsable payload is treated as an error.
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise SASTOutputParseError(
                f"Could not parse bandit JSON output: {exc}\nstderr (truncated): {proc.stderr[:500]}"
            ) from exc

        return [self._to_raw_finding(result) for result in payload.get("results", [])]

    @staticmethod
    def _to_raw_finding(result: dict) -> RawFinding:
        test_id = result.get("test_id", "unknown-rule")
        cwe = result.get("issue_cwe")
        if isinstance(cwe, dict) and cwe.get("id") is not None:
            cwe_id = f"CWE-{cwe['id']}"
        else:
            cwe_id = _TEST_ID_TO_CWE_FALLBACK.get(test_id)

        line_range = result.get("line_range") or [result.get("line_number", 0)]

        return RawFinding(
            tool="bandit",
            rule_id=test_id,
            cwe_id=cwe_id,
            file_path=result.get("filename", ""),
            line_start=min(line_range) if line_range else result.get("line_number", 0),
            line_end=max(line_range) if line_range else result.get("line_number", 0),
            message=result.get("issue_text", ""),
            severity=Severity.from_loose_string(result.get("issue_severity")),
            confidence=result.get("issue_confidence"),
        )
