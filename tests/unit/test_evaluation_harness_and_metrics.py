from __future__ import annotations

from vulnrem.core.enums import FindingStatus, Language, Severity
from vulnrem.evaluation.harness import EvaluationHarness
from vulnrem.evaluation.metrics import (
    EvalCaseResult,
    MetricWeights,
    compute_breakdown_by_cwe,
    compute_metrics,
)
from vulnrem.evaluation.schemas import CanonicalRecord
from vulnrem.pipeline.runner import FileCheckReport, FindingResult
from vulnrem.pipeline.state import Finding, PipelineState, ReproductionResult

# --------------------------------------------------------------------------- #
# metrics.compute_metrics / compute_breakdown_by_cwe
# --------------------------------------------------------------------------- #


def _case(record_id, cwe, detected, remediated, sast_detected=True) -> EvalCaseResult:
    return EvalCaseResult(
        record_id=record_id,
        language=Language.PYTHON,
        expected_cwe_id=cwe,
        sast_detected=sast_detected,
        detected=detected,
        remediated=remediated,
    )


def test_compute_metrics_basic_rates():
    results = [
        _case("r1", "CWE-89", detected=True, remediated=True),
        _case("r2", "CWE-89", detected=True, remediated=False),
        _case("r3", "CWE-89", detected=False, remediated=False, sast_detected=False),
        _case("r4", "CWE-89", detected=False, remediated=False, sast_detected=False),
    ]
    metrics = compute_metrics(results)
    assert metrics.total_eval_cases == 4
    assert metrics.detection_recall == 0.5
    assert metrics.remediation_recall == 0.25
    assert metrics.sast_detection_rate == 0.5


def test_compute_metrics_weighted_score_uses_custom_weights():
    results = [_case("r1", "CWE-89", detected=True, remediated=True)]
    metrics = compute_metrics(results, weights=MetricWeights(detection=0.1, remediation=0.9))
    assert metrics.weighted_overall_score == 0.1 * 1.0 + 0.9 * 1.0


def test_compute_metrics_empty_results_do_not_divide_by_zero():
    metrics = compute_metrics([])
    assert metrics.total_eval_cases == 0
    assert metrics.detection_recall == 0.0
    assert metrics.remediation_recall == 0.0


def test_compute_metrics_extra_metric_fns_are_included():
    results = [_case("r1", "CWE-89", detected=True, remediated=True)]
    metrics = compute_metrics(results, extra_metric_fns={"case_count_doubled": lambda rs: len(rs) * 2})
    assert metrics.additional_metrics["case_count_doubled"] == 2


def test_breakdown_by_cwe_groups_correctly():
    results = [
        _case("r1", "CWE-89", detected=True, remediated=True),
        _case("r2", "CWE-78", detected=False, remediated=False, sast_detected=False),
    ]
    breakdown = compute_breakdown_by_cwe(results)
    assert breakdown["CWE-89"].detection_recall == 1.0
    assert breakdown["CWE-78"].detection_recall == 0.0


# --------------------------------------------------------------------------- #
# EvaluationHarness
# --------------------------------------------------------------------------- #


def _finding(cwe_id: str | None) -> Finding:
    return Finding(
        finding_id="f1",
        language=Language.PYTHON,
        file_path="x.py",
        line_start=1,
        line_end=1,
        cwe_id=cwe_id,
        severity=Severity.HIGH,
        tool_findings=[],
    )


def _state(finding: Finding, status: FindingStatus, reproduction_confirmed: bool | None) -> PipelineState:
    reproduction = (
        ReproductionResult(confirmed=reproduction_confirmed, poc_script="", explanation="")
        if reproduction_confirmed is not None
        else None
    )
    return PipelineState(
        language=Language.PYTHON,
        file_path="x.py",
        original_source="code",
        finding=finding,
        reproduction=reproduction,
        status=status,
    )


class _FakeRunner:
    def __init__(self, report: FileCheckReport):
        self._report = report

    def check_file(self, file_path: str, source: str) -> FileCheckReport:
        return self._report


def test_harness_marks_detected_and_remediated_when_accepted():
    finding = _finding("CWE-89")
    state = _state(finding, FindingStatus.ACCEPTED, reproduction_confirmed=True)
    report = FileCheckReport(
        file_path="r1.py", language=Language.PYTHON, preprocess_warnings=[],
        findings=[FindingResult(finding=finding, final_state=state)],
    )
    harness = EvaluationHarness(runner_factory=lambda lang: _FakeRunner(report))
    record = CanonicalRecord(record_id="r1", language=Language.PYTHON, cwe_id="CWE-89", vulnerable_code="code")

    results = harness.run([record])

    assert results[0].sast_detected is True
    assert results[0].detected is True
    assert results[0].remediated is True


def test_harness_marks_undetected_when_no_findings_at_all():
    report = FileCheckReport(file_path="r1.py", language=Language.PYTHON, preprocess_warnings=[], findings=[])
    harness = EvaluationHarness(runner_factory=lambda lang: _FakeRunner(report))
    record = CanonicalRecord(record_id="r1", language=Language.PYTHON, cwe_id="CWE-89", vulnerable_code="code")

    results = harness.run([record])

    assert results[0].sast_detected is False
    assert results[0].detected is False
    assert results[0].remediated is False


def test_harness_detected_but_not_remediated_when_retries_exhausted():
    finding = _finding("CWE-89")
    state = _state(finding, FindingStatus.UNRESOLVED_RETRIES_EXHAUSTED, reproduction_confirmed=True)
    report = FileCheckReport(
        file_path="r1.py", language=Language.PYTHON, preprocess_warnings=[],
        findings=[FindingResult(finding=finding, final_state=state)],
    )
    harness = EvaluationHarness(runner_factory=lambda lang: _FakeRunner(report))
    record = CanonicalRecord(record_id="r1", language=Language.PYTHON, cwe_id="CWE-89", vulnerable_code="code")

    results = harness.run([record])

    assert results[0].detected is True
    assert results[0].remediated is False


def test_harness_falls_back_to_any_finding_when_cwe_label_mismatches():
    # aggregation could tag a finding with a slightly different CWE than the
    # dataset's label for the same real issue -- the harness should not
    # manufacture a false "missed" result purely from a labeling mismatch.
    finding = _finding("CWE-78")  # dataset label says CWE-89
    state = _state(finding, FindingStatus.ACCEPTED, reproduction_confirmed=True)
    report = FileCheckReport(
        file_path="r1.py", language=Language.PYTHON, preprocess_warnings=[],
        findings=[FindingResult(finding=finding, final_state=state)],
    )
    harness = EvaluationHarness(runner_factory=lambda lang: _FakeRunner(report))
    record = CanonicalRecord(record_id="r1", language=Language.PYTHON, cwe_id="CWE-89", vulnerable_code="code")

    results = harness.run([record])

    assert results[0].detected is True
    assert results[0].remediated is True
