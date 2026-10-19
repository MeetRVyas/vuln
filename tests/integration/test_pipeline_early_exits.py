from __future__ import annotations

import shutil

import pytest

from vulnrem.config.loader import load_config
from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.enums import AgentRole, FindingStatus, Language
from vulnrem.llm.providers.mock import MockLLMProvider
from vulnrem.pipeline.prompts.schemas import ReproducerOutput
from vulnrem.pipeline.runner import build_file_check_runner
from vulnrem.sandbox.subprocess_sandbox import SubprocessSandbox

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("bandit") is None, reason="bandit not installed"),
]


def _runner(config_dir, mock_llm):
    app_config = load_config(env="dev", config_dir=config_dir)
    return build_file_check_runner(
        app_config=app_config,
        language=Language.PYTHON,
        config_dir=str(config_dir),
        llm=mock_llm,
        sandbox=SubprocessSandbox(),
    )


def test_triage_false_positive_short_circuits_before_reproducer(config_dir, sql_injection_source):
    mock_llm = MockLLMProvider(models=RoleModelMap(default="mock-model"))
    mock_llm.enqueue_structured(
        AgentRole.TRIAGE,
        {
            "is_true_positive": False,
            "confidence": 0.8,
            "rationale": "This particular call site only ever receives a hardcoded constant.",
        },
    )
    # Deliberately nothing queued for REPRODUCER/etc -- if the graph tried to
    # call them, MockLLMProvider would raise, failing the test.

    report = _runner(config_dir, mock_llm).check_file("sql_injection.py", sql_injection_source)
    final_state = report.findings[0].final_state

    assert final_state.status == FindingStatus.TRIAGED_FALSE_POSITIVE
    assert "hardcoded constant" in final_state.status_detail
    assert len(mock_llm.call_log) == 1  # only Triage ran


def test_unconfirmed_reproduction_short_circuits_before_localizer(config_dir, sql_injection_source):
    mock_llm = MockLLMProvider(models=RoleModelMap(default="mock-model"))
    mock_llm.enqueue_structured(
        AgentRole.TRIAGE, {"is_true_positive": True, "confidence": 0.7, "rationale": "Looks concerning."}
    )
    mock_llm.enqueue_structured(
        AgentRole.REPRODUCER,
        ReproducerOutput(
            poc_script="print('VULN_NOT_CONFIRMED')",
            explanation="Could not construct an input that actually triggers the issue.",
        ),
    )

    report = _runner(config_dir, mock_llm).check_file("sql_injection.py", sql_injection_source)
    final_state = report.findings[0].final_state

    assert final_state.status == FindingStatus.TRIAGED_FALSE_POSITIVE
    assert "did not confirm" in final_state.status_detail
    assert len(mock_llm.call_log) == 2  # triage + reproducer only
