"""The shared contract every pipeline node reads from and writes to.

:class:`PipelineState` is the LangGraph state for the *per-finding*
remediation graph (see :mod:`vulnrem.pipeline.graph`): one file can produce
several findings (SAST + aggregation happens once, outside the graph -- see
:mod:`vulnrem.pipeline.runner`), and each finding gets its own independent
run of this state machine.

Every node function has the signature ``(state: PipelineState) -> dict``
and returns only the fields it changed; LangGraph merges that into the
state using each field's reducer (plain overwrite by default, ``operator.add``
for the accumulating ``attempt_history`` list -- see the field definition
below for why that one is different).
"""

from __future__ import annotations

import hashlib
import operator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from vulnrem.core.enums import FindingStatus, Language, RetryCategory, Severity
from vulnrem.languages.base import LocalizationContext
from vulnrem.sast.schemas import RawFinding

# --------------------------------------------------------------------------- #
# Findings (post-aggregation)
# --------------------------------------------------------------------------- #


class Finding(BaseModel):
    """One de-duplicated finding, potentially corroborated by more than one
    SAST tool. Produced by :mod:`vulnrem.pipeline.nodes.aggregator`."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    language: Language
    file_path: str
    line_start: int
    line_end: int
    cwe_id: str | None
    severity: Severity
    tool_findings: list[RawFinding]

    def contributing_tools(self) -> list[str]:
        return sorted({f.tool for f in self.tool_findings})

    @staticmethod
    def make_id(file_path: str, line_start: int, cwe_id: str | None) -> str:
        digest = hashlib.sha256(f"{file_path}:{line_start}:{cwe_id or ''}".encode()).hexdigest()
        return digest[:12]


# --------------------------------------------------------------------------- #
# Patcher output
# --------------------------------------------------------------------------- #


class PatchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patched_source: str
    explanation: str
    diff_unified: str = ""  # computed via difflib after the fact -- never requested from the LLM


# --------------------------------------------------------------------------- #
# Reproducer output
# --------------------------------------------------------------------------- #


class ReproductionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed: bool
    poc_script: str
    explanation: str
    raw_stdout: str = ""
    raw_stderr: str = ""


# --------------------------------------------------------------------------- #
# Test Writer output
# --------------------------------------------------------------------------- #

TestKind = Literal["poc", "regression", "edge_case"]


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # bare pytest function name, e.g. "test_poc_sql_injection"
    kind: TestKind
    source: str  # the full `def test_...(): ...` source block
    expected_result_on_original: Literal["fail", "pass"]


class TestBattery(BaseModel):
    """The validated set of tests handed to the Verifier. Only tests that
    matched their ``expected_result_on_original`` after self-validation (see
    :mod:`vulnrem.pipeline.nodes.test_writer`) survive into
    ``test_cases`` -- the rest are recorded in ``discarded_case_names`` for
    observability, never silently dropped."""

    model_config = ConfigDict(extra="forbid")

    test_cases: list[TestCase] = Field(default_factory=list)
    combined_source: str = ""  # assembled single pytest file (imports + all surviving cases)
    self_validation_attempts: int = 0
    discarded_case_names: list[str] = Field(default_factory=list)

    def names_of_kind(self, kind: TestKind) -> list[str]:
        return [tc.name for tc in self.test_cases if tc.kind == kind]


# --------------------------------------------------------------------------- #
# Verifier output -- the dual oracle
# --------------------------------------------------------------------------- #


class OracleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    detail: str = ""


class FunctionalOracleResult(BaseModel):
    """'Did the patch preserve intended behavior?' -- pre-existing/generated
    regression tests plus the edge-case traps must all still pass."""

    model_config = ConfigDict(extra="forbid")

    regression: OracleResult
    edge_case: OracleResult

    @property
    def passed(self) -> bool:
        return self.regression.passed and self.edge_case.passed


class SecurityOracleResult(BaseModel):
    """'Is the vulnerability actually gone?' -- both the dynamic PoC test
    and a fresh SAST rescan of the patched file must agree."""

    model_config = ConfigDict(extra="forbid")

    poc_no_longer_triggers: OracleResult
    sast_rescan_clean: OracleResult

    @property
    def passed(self) -> bool:
        return self.poc_no_longer_triggers.passed and self.sast_rescan_clean.passed


SanitizerStatus = Literal["not_applicable", "not_implemented", "passed", "failed"]


class VerifierReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functional: FunctionalOracleResult
    security: SecurityOracleResult
    sanitizer: SanitizerStatus = "not_applicable"
    raw_stdout: str = ""
    raw_stderr: str = ""

    @property
    def overall_pass(self) -> bool:
        return self.functional.passed and self.security.passed


# --------------------------------------------------------------------------- #
# Judge output
# --------------------------------------------------------------------------- #


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "retry"]
    category: RetryCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


# --------------------------------------------------------------------------- #
# Episodic memory across retries
# --------------------------------------------------------------------------- #


class AttemptRecord(BaseModel):
    """One full pass through Patcher -> Verifier -> Judge. The *entire*
    record (not just pass/fail) is fed back into the next Patcher prompt --
    a Reflexion-style retry gets the previous diff and the previous
    failure's actual trace, not just 'that attempt failed, try again.'"""

    model_config = ConfigDict(extra="forbid")

    attempt_number: int
    patch_candidate: PatchCandidate | None = None
    verifier_report: VerifierReport | None = None
    judge_verdict: JudgeVerdict | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# The graph state itself
# --------------------------------------------------------------------------- #


class PipelineState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # -- fixed inputs, set once before the graph runs -- #
    language: Language
    file_path: str
    original_source: str
    finding: Finding

    # -- populated as the graph progresses -- #
    triage_confirmed: bool | None = None
    triage_rationale: str = ""

    reproduction: ReproductionResult | None = None

    localization_context: LocalizationContext | None = None

    patch_candidate: PatchCandidate | None = None
    test_battery: TestBattery | None = None
    verifier_report: VerifierReport | None = None
    judge_verdict: JudgeVerdict | None = None

    attempt_history: Annotated[list[AttemptRecord], operator.add] = Field(default_factory=list)

    status: FindingStatus = FindingStatus.PENDING
    status_detail: str = ""

    def current_attempt_number(self) -> int:
        return len(self.attempt_history) + 1
