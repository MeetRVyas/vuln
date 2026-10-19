from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, ConfigDict

from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.enums import AgentRole
from vulnrem.core.errors import LLMConnectionError, LLMResponseError
from vulnrem.llm.providers.google import GoogleGeminiProvider
from vulnrem.llm.schemas import LLMRequest, Message


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    confidence: float


class _FakeModelsAPI:
    def __init__(self, response=None, exception=None, list_ok=True):
        self._response = response
        self._exception = exception
        self._list_ok = list_ok
        self.calls: list[dict] = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._exception is not None:
            raise self._exception
        return self._response

    def list(self):
        if not self._list_ok:
            raise RuntimeError("unreachable")
        return iter([object()])


class _FakeClient:
    def __init__(self, models_api: _FakeModelsAPI):
        self.models = models_api


def _provider(models_api: _FakeModelsAPI) -> GoogleGeminiProvider:
    return GoogleGeminiProvider(
        models=RoleModelMap(default="gemini-2.5-flash", per_role={AgentRole.JUDGE: "gemini-2.5-pro"}),
        api_key_env_var="UNUSED_BECAUSE_CLIENT_INJECTED",
        client=_FakeClient(models_api),
    )


def test_generate_maps_roles_to_configured_models_and_returns_text():
    fake_response = SimpleNamespace(
        text="here is the fix",
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
    )
    models_api = _FakeModelsAPI(response=fake_response)
    provider = _provider(models_api)

    request = LLMRequest(
        role=AgentRole.JUDGE,
        messages=[
            Message(role="system", content="You are the Judge."),
            Message(role="user", content="Here is the verifier report."),
        ],
    )
    response = provider.generate(request)

    assert response.text == "here is the fix"
    assert response.model_used == "gemini-2.5-pro"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20
    assert models_api.calls[0]["config"].system_instruction == "You are the Judge."


def test_generate_uses_default_model_for_unmapped_role():
    fake_response = SimpleNamespace(text="triaged", usage_metadata=None)
    provider = _provider(_FakeModelsAPI(response=fake_response))

    request = LLMRequest(role=AgentRole.TRIAGE, messages=[Message(role="user", content="hi")])
    response = provider.generate(request)

    assert response.model_used == "gemini-2.5-flash"


def test_generate_raises_response_error_on_empty_completion():
    fake_response = SimpleNamespace(text="", usage_metadata=None)
    provider = _provider(_FakeModelsAPI(response=fake_response))

    with pytest.raises(LLMResponseError):
        provider.generate(LLMRequest(role=AgentRole.TRIAGE, messages=[Message(role="user", content="hi")]))


def test_generate_wraps_connection_like_exceptions():
    provider = _provider(_FakeModelsAPI(exception=TimeoutError("connection timed out")))

    with pytest.raises(LLMConnectionError):
        provider.generate(LLMRequest(role=AgentRole.TRIAGE, messages=[Message(role="user", content="hi")]))


def test_generate_wraps_other_exceptions_as_response_error():
    provider = _provider(_FakeModelsAPI(exception=ValueError("model not found")))

    with pytest.raises(LLMResponseError):
        provider.generate(LLMRequest(role=AgentRole.TRIAGE, messages=[Message(role="user", content="hi")]))


def test_generate_structured_uses_native_parsed_field():
    fake_response = SimpleNamespace(parsed=_Verdict(decision="accept", confidence=0.95), text="{}")
    models_api = _FakeModelsAPI(response=fake_response)
    provider = _provider(models_api)

    result = provider.generate_structured(
        LLMRequest(role=AgentRole.JUDGE, messages=[Message(role="user", content="judge")]), _Verdict
    )

    assert result.decision == "accept"
    assert result.confidence == 0.95
    config = models_api.calls[0]["config"]
    assert config.response_schema is None
    assert config.response_json_schema == _Verdict.model_json_schema()


def test_generate_structured_falls_back_to_text_when_parsed_missing():
    fake_response = SimpleNamespace(parsed=None, text='{"decision": "retry", "confidence": 0.2}')
    provider = _provider(_FakeModelsAPI(response=fake_response))

    result = provider.generate_structured(
        LLMRequest(role=AgentRole.JUDGE, messages=[Message(role="user", content="judge")]), _Verdict
    )

    assert result.decision == "retry"


def test_generate_structured_retries_temporary_server_error(monkeypatch):
    attempts = []
    models_api = _FakeModelsAPI()

    def generate_content(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise ServerError(503, {"error": {"status": "UNAVAILABLE", "message": "busy"}})
        return SimpleNamespace(parsed=_Verdict(decision="accept", confidence=0.9), text="{}")

    monkeypatch.setattr(models_api, "generate_content", generate_content)
    monkeypatch.setattr("vulnrem.llm.providers.google.time.sleep", lambda _seconds: None)
    provider = _provider(models_api)

    result = provider.generate_structured(
        LLMRequest(role=AgentRole.TRIAGE, messages=[Message(role="user", content="triage")]), _Verdict
    )

    assert result.decision == "accept"
    assert len(attempts) == 2


def test_generate_structured_does_not_retry_quota_error(monkeypatch):
    attempts = []
    models_api = _FakeModelsAPI()

    def generate_content(**kwargs):
        attempts.append(kwargs)
        raise ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}})

    monkeypatch.setattr(models_api, "generate_content", generate_content)
    provider = _provider(models_api)

    with pytest.raises(LLMResponseError, match="429"):
        provider.generate_structured(
            LLMRequest(role=AgentRole.TRIAGE, messages=[Message(role="user", content="triage")]), _Verdict
        )

    assert len(attempts) == 1


def test_health_check_true_when_list_succeeds():
    provider = _provider(_FakeModelsAPI(response=None, list_ok=True))
    assert provider.health_check() is True


def test_health_check_false_when_list_fails():
    provider = _provider(_FakeModelsAPI(response=None, list_ok=False))
    assert provider.health_check() is False
