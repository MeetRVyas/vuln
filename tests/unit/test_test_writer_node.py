from __future__ import annotations

from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.enums import AgentRole, Language, Severity
from vulnrem.languages.python_profile import PythonLanguageProfile
from vulnrem.llm.providers.mock import MockLLMProvider
from vulnrem.pipeline.nodes.test_writer import TestWriterNode
from vulnrem.pipeline.prompts.schemas import TestWriterOutput
from vulnrem.pipeline.state import Finding, PipelineState, ReproductionResult, TestCase
from vulnrem.sandbox.subprocess_sandbox import SubprocessSandbox

_SOURCE = """def double(x):
    return x * 2
"""


def _finding() -> Finding:
    return Finding(
        finding_id="abc123",
        language=Language.PYTHON,
        file_path="target.py",
        line_start=1,
        line_end=2,
        cwe_id=None,
        severity=Severity.LOW,
        tool_findings=[],
    )


def _base_state() -> PipelineState:
    profile = PythonLanguageProfile()
    context = profile.slice_context(_SOURCE, 1, 2)
    return PipelineState(
        language=Language.PYTHON,
        file_path="target.py",
        original_source=_SOURCE,
        finding=_finding(),
        localization_context=context,
        reproduction=ReproductionResult(confirmed=True, poc_script="", explanation="n/a"),
    )


def _node(mock_llm: MockLLMProvider, max_attempts: int = 2) -> TestWriterNode:
    return TestWriterNode(
        llm=mock_llm,
        sandbox=SubprocessSandbox(),
        language_profile=PythonLanguageProfile(),
        timeout_s=10,
        memory_limit_mb=256,
        cpu_limit_seconds=10,
        max_self_validation_attempts=max_attempts,
    )


_GOOD_POC = TestCase(
    name="test_poc_good",
    kind="poc",
    source="""def test_poc_good():
    from target import double
    assert double(2) != 4
""",
    expected_result_on_original="fail",  # double(2) IS 4, so this correctly fails on original
)

_BAD_CASE_WRONG_EXPECTATION = TestCase(
    name="test_regression_wrong_expectation",
    kind="regression",
    source="""def test_regression_wrong_expectation():
    from target import double
    assert double(2) == 999
""",
    expected_result_on_original="pass",  # actually FAILS on original -> should be discarded
)

_GOOD_REGRESSION = TestCase(
    name="test_regression_good",
    kind="regression",
    source="""def test_regression_good():
    from target import double
    assert double(2) == 4
""",
    expected_result_on_original="pass",
)


def test_self_validation_discards_mismatched_case_and_keeps_correct_ones():
    mock_llm = MockLLMProvider(models=RoleModelMap(default="mock-model"))
    mock_llm.enqueue_structured(
        AgentRole.TEST_WRITER,
        TestWriterOutput(test_cases=[_GOOD_POC, _BAD_CASE_WRONG_EXPECTATION, _GOOD_REGRESSION]),
    )

    result = _node(mock_llm)(_base_state())
    battery = result["test_battery"]

    assert battery.discarded_case_names == ["test_regression_wrong_expectation"]
    assert {tc.name for tc in battery.test_cases} == {"test_poc_good", "test_regression_good"}
    assert battery.self_validation_attempts == 1


def test_retries_when_no_poc_survives_first_attempt():
    mock_llm = MockLLMProvider(models=RoleModelMap(default="mock-model"))
    bad_poc = TestCase(
        name="test_poc_bad",
        kind="poc",
        source="""def test_poc_bad():
    from target import double
    assert double(2) == 4  # wrong: this PASSES on original, but poc must FAIL on original
""",
        expected_result_on_original="fail",
    )
    # First attempt: only a broken poc (gets discarded -> no poc survives).
    mock_llm.enqueue_structured(AgentRole.TEST_WRITER, TestWriterOutput(test_cases=[bad_poc, _GOOD_REGRESSION]))
    # Second attempt: a correct poc.
    mock_llm.enqueue_structured(
        AgentRole.TEST_WRITER, TestWriterOutput(test_cases=[_GOOD_POC, _GOOD_REGRESSION])
    )

    result = _node(mock_llm, max_attempts=2)(_base_state())
    battery = result["test_battery"]

    assert battery.self_validation_attempts == 2
    assert "test_poc_good" in {tc.name for tc in battery.test_cases}
    assert any(tc.kind == "poc" for tc in battery.test_cases)
