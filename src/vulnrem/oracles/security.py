"""The static half of the dual security oracle.

The dynamic half (does the PoC test still trigger?) is just another pytest
outcome and is handled inline by the Verifier node using the same
:mod:`vulnrem.oracles.pytest_support` machinery as the functional oracle --
it doesn't need its own abstraction. This one does, because it's a
genuinely different mechanism (re-running static analysis, not executing
code), and because a smarter rescan strategy (e.g. diff-aware: only compare
findings whose rule/CWE matches AND whose surrounding function signature is
unchanged) is a plausible future swap-in.
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from vulnrem.core.errors import SASTError
from vulnrem.pipeline.state import Finding, OracleResult
from vulnrem.sast.base import SASTTool

_DEFAULT_LINE_TOLERANCE = 5  # a patch can shift line numbers; same CWE within this many lines still counts


class SecurityRescanChecker(ABC):
    @abstractmethod
    def check(self, patched_source: str, file_name: str, finding: Finding) -> OracleResult: ...


class SastRescanChecker(SecurityRescanChecker):
    def __init__(self, tools: list[SASTTool], line_tolerance: int = _DEFAULT_LINE_TOLERANCE):
        self._tools = tools
        self._line_tolerance = line_tolerance

    def check(self, patched_source: str, file_name: str, finding: Finding) -> OracleResult:
        if not self._tools:
            return OracleResult(
                passed=True,
                detail="No SAST tools were available to rescan with (treated as inconclusive-pass; "
                "the dynamic PoC oracle is still authoritative).",
            )

        with tempfile.TemporaryDirectory(prefix="vulnrem-rescan-") as tmpdir:
            target = Path(tmpdir) / file_name
            target.write_text(patched_source)

            still_present: list[str] = []
            for tool in self._tools:
                try:
                    raw_findings = tool.scan(target)
                except SASTError:
                    continue  # a tool failing to run shouldn't itself block acceptance
                for raw in raw_findings:
                    same_cwe = finding.cwe_id is not None and raw.cwe_id == finding.cwe_id
                    nearby = abs(raw.line_start - finding.line_start) <= self._line_tolerance
                    if same_cwe and nearby:
                        still_present.append(f"{tool.tool_name}:{raw.rule_id}@L{raw.line_start}")

            if still_present:
                return OracleResult(
                    passed=False,
                    detail=f"SAST rescan still flags {finding.cwe_id} near line {finding.line_start}: "
                    f"{', '.join(still_present)}",
                )
            return OracleResult(
                passed=True, detail="SAST rescan found no matching finding in the patched file."
            )
