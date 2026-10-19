"""Wraps the ``opengrep`` CLI (an LGPL-2.1, Semgrep-rule-compatible fork
that -- unlike Semgrep's post-relicensing Community Edition -- keeps full
taint analysis, which matters for the injection-style CWEs this project
targets first). See docs/SETUP_DEPLOYMENT.md for installation.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from vulnrem.core.enums import Severity
from vulnrem.core.errors import SASTError, SASTOutputParseError, SASTToolNotAvailableError
from vulnrem.sast.base import SASTTool
from vulnrem.sast.schemas import RawFinding

_CWE_PATTERN = re.compile(r"CWE-\d+")

_Runner = Callable[..., subprocess.CompletedProcess]


class OpengrepTool(SASTTool):
    def __init__(
        self,
        timeout_s: float = 60.0,
        extra_args: list[str] | None = None,
        runner: _Runner | None = None,
    ):
        self._timeout_s = timeout_s
        self._extra_args = extra_args or []
        self._runner: _Runner = runner or subprocess.run

    @property
    def tool_name(self) -> str:
        return "opengrep"

    def is_available(self) -> bool:
        return shutil.which("opengrep") is not None

    def _build_command(self, target_path: Path) -> list[str]:
        cmd = ["opengrep", "scan", "--json", "--quiet"]
        cmd.extend(self._extra_args or ["--config", "auto"])
        cmd.append(str(target_path))
        return cmd

    def scan(self, target_path: Path) -> list[RawFinding]:
        cmd = self._build_command(target_path)
        try:
            proc = self._runner(cmd, capture_output=True, text=True, timeout=self._timeout_s)
        except FileNotFoundError as exc:
            raise SASTToolNotAvailableError("opengrep binary not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise SASTError(f"opengrep timed out after {self._timeout_s}s") from exc

        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise SASTOutputParseError(
                f"Could not parse opengrep JSON output: {exc}\nstderr (truncated): {proc.stderr[:500]}"
            ) from exc

        return [self._to_raw_finding(result) for result in payload.get("results", [])]

    @staticmethod
    def _to_raw_finding(result: dict) -> RawFinding:
        extra = result.get("extra", {})
        metadata = extra.get("metadata", {})
        cwe_field = metadata.get("cwe")
        cwe_id = None
        if cwe_field:
            candidate = cwe_field[0] if isinstance(cwe_field, list) else cwe_field
            match = _CWE_PATTERN.search(str(candidate))
            cwe_id = match.group(0) if match else None

        return RawFinding(
            tool="opengrep",
            rule_id=result.get("check_id", "unknown-rule"),
            cwe_id=cwe_id,
            file_path=result.get("path", ""),
            line_start=result.get("start", {}).get("line", 0),
            line_end=result.get("end", {}).get("line", result.get("start", {}).get("line", 0)),
            message=extra.get("message", ""),
            severity=Severity.from_loose_string(extra.get("severity")),
            confidence=metadata.get("confidence"),
        )
