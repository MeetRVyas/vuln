from __future__ import annotations

import pytest
from pydantic import BaseModel

from vulnrem.core.enums import AgentRole
from vulnrem.core.errors import LLMStructuredOutputError
from vulnrem.llm.schemas import LLMRequest, Message


class _Verdict(BaseModel):
    decision: str
    confidence: float


def _request(role: AgentRole) -> LLMRequest:
    return LLMRequest(role=role, messages=[Message(role="user", content="hello")])


def test_generate_returns_default_text_when_nothing_queued(mock_llm):
    response = mock_llm.generate(_request(AgentRole.TRIAGE))
    assert response.text == "mock response"
    assert response.provider_name == "mock"
    assert len(mock_llm.call_log) == 1


def test_generate_drains_queued_text_in_order(mock_llm):
    mock_llm.enqueue_text(AgentRole.PATCHER, "first")
    mock_llm.enqueue_text(AgentRole.PATCHER, "second")

    first = mock_llm.generate(_request(AgentRole.PATCHER))
    second = mock_llm.generate(_request(AgentRole.PATCHER))
    third = mock_llm.generate(_request(AgentRole.PATCHER))

    assert [first.text, second.text, third.text] == ["first", "second", "mock response"]


def test_queues_are_isolated_per_role(mock_llm):
    mock_llm.enqueue_text(AgentRole.PATCHER, "patcher text")
    response = mock_llm.generate(_request(AgentRole.JUDGE))
    assert response.text == "mock response"  # judge queue untouched


def test_generate_structured_returns_canned_instance(mock_llm):
    mock_llm.enqueue_structured(AgentRole.JUDGE, _Verdict(decision="accept", confidence=0.9))
    result = mock_llm.generate_structured(_request(AgentRole.JUDGE), _Verdict)
    assert result.decision == "accept"
    assert result.confidence == 0.9


def test_generate_structured_accepts_raw_dict(mock_llm):
    mock_llm.enqueue_structured(AgentRole.JUDGE, {"decision": "retry", "confidence": 0.4})
    result = mock_llm.generate_structured(_request(AgentRole.JUDGE), _Verdict)
    assert isinstance(result, _Verdict)
    assert result.decision == "retry"


def test_generate_structured_raises_when_nothing_queued(mock_llm):
    with pytest.raises(LLMStructuredOutputError):
        mock_llm.generate_structured(_request(AgentRole.JUDGE), _Verdict)


def test_health_check_always_true(mock_llm):
    assert mock_llm.health_check() is True
