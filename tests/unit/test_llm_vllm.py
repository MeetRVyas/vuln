from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import BaseModel

from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.enums import AgentRole
from vulnrem.core.errors import LLMConnectionError, LLMResponseError
from vulnrem.llm.providers.vllm import VLLMProvider
from vulnrem.llm.schemas import LLMRequest, Message

BASE_URL = "http://localhost:8000/v1"


class _Verdict(BaseModel):
    decision: str
    confidence: float


@pytest.fixture
def provider() -> VLLMProvider:
    models = RoleModelMap(default="qwen3-coder-base", per_role={AgentRole.PATCHER: "patcher-python-lora"})
    return VLLMProvider(models=models, base_url=BASE_URL, timeout_s=5, max_retries=1)


def _chat_completion_payload(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 34},
    }


@respx.mock
def test_generate_resolves_role_specific_adapter_and_parses_response(provider):
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_payload("patched code here"))
    )
    request = LLMRequest(role=AgentRole.PATCHER, messages=[Message(role="user", content="fix this")])

    response = provider.generate(request)

    assert response.text == "patched code here"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 34
    sent_body = route.calls.last.request.content
    assert b'"patcher-python-lora"' in sent_body  # role -> adapter routing


@respx.mock
def test_generate_uses_default_model_for_unmapped_role(provider):
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_payload("triage result"))
    )
    request = LLMRequest(role=AgentRole.TRIAGE, messages=[Message(role="user", content="triage this")])

    response = provider.generate(request)

    assert response.model_used == "qwen3-coder-base"


@respx.mock
def test_generate_retries_on_5xx_then_succeeds(provider):
    route = respx.post(f"{BASE_URL}/chat/completions")
    route.side_effect = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(200, json=_chat_completion_payload("ok on retry")),
    ]
    request = LLMRequest(role=AgentRole.JUDGE, messages=[Message(role="user", content="judge this")])

    response = provider.generate(request)

    assert response.text == "ok on retry"
    assert route.call_count == 2


@respx.mock
def test_generate_raises_llm_response_error_on_4xx(provider):
    respx.post(f"{BASE_URL}/chat/completions").mock(return_value=httpx.Response(400, json={"error": "bad request"}))
    request = LLMRequest(role=AgentRole.JUDGE, messages=[Message(role="user", content="judge this")])

    with pytest.raises(LLMResponseError):
        provider.generate(request)


@respx.mock
def test_generate_raises_connection_error_when_unreachable(provider):
    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    request = LLMRequest(role=AgentRole.JUDGE, messages=[Message(role="user", content="judge this")])

    with pytest.raises(LLMConnectionError):
        provider.generate(request)


@respx.mock
def test_generate_structured_uses_guided_json_and_parses(provider):
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_payload('{"decision": "accept", "confidence": 0.8}'))
    )
    request = LLMRequest(role=AgentRole.JUDGE, messages=[Message(role="user", content="judge this")])

    result = provider.generate_structured(request, _Verdict)

    assert result.decision == "accept"
    sent_body = route.calls.last.request.content
    assert b"guided_json" in sent_body


@respx.mock
def test_generate_structured_falls_back_when_guided_json_rejected(provider):
    route = respx.post(f"{BASE_URL}/chat/completions")
    route.side_effect = [
        httpx.Response(400, json={"error": "guided_json not supported"}),
        httpx.Response(200, json=_chat_completion_payload('{"decision": "retry", "confidence": 0.3}')),
    ]
    request = LLMRequest(role=AgentRole.JUDGE, messages=[Message(role="user", content="judge this")])

    result = provider.generate_structured(request, _Verdict)

    assert result.decision == "retry"
    assert route.call_count == 2


@respx.mock
def test_health_check_true_when_models_endpoint_ok(provider):
    respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    assert provider.health_check() is True


@respx.mock
def test_health_check_false_when_unreachable(provider):
    respx.get(f"{BASE_URL}/models").mock(side_effect=httpx.ConnectError("refused"))
    assert provider.health_check() is False
