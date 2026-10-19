"""Local LLM provider: talks to a vLLM server's OpenAI-compatible HTTP API.

This is the production backend: vLLM serves the shared Qwen3-Coder base
model plus one LoRA adapter per role that has one (see
``docs/SETUP_DEPLOYMENT.md`` for how the server is started with
``--enable-lora``). Role -> model/adapter-name routing is entirely
config-driven (:class:`~vulnrem.config.schemas.RoleModelMap`): the HTTP
request's ``"model"`` field is simply whichever adapter (or the base model)
is configured for the request's role, exactly as vLLM's multi-LoRA serving
expects.

Structured output is requested via vLLM's ``guided_json`` extra body param.
If the server rejects it (older/newer build with a different structured-
output mechanism), we transparently fall back to prompting for JSON and
parsing/validating the free-text response, so this provider keeps working
across vLLM versions at the cost of a slightly less reliable structured
response in that fallback path.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.errors import LLMConnectionError, LLMResponseError
from vulnrem.llm.base import LLMProvider, SchemaT
from vulnrem.llm.schemas import LLMRequest, LLMResponse, TokenUsage


class VLLMProvider(LLMProvider):
    def __init__(
        self,
        models: RoleModelMap,
        base_url: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ):
        super().__init__(models)
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._client = http_client or httpx.Client(timeout=timeout_s)

    @property
    def provider_name(self) -> str:
        return "vllm"

    @staticmethod
    def _messages_payload(request: LLMRequest) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in request.messages]

    def _post_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        request: LLMRequest,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if extra_body:
            body.update(extra_body)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(f"{self._base_url}/chat/completions", json=body)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as exc:
                last_exc = exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < self._max_retries:
                    last_exc = exc
                    continue
                raise LLMResponseError(
                    f"vLLM returned HTTP {exc.response.status_code} for model={model}: "
                    f"{exc.response.text[:300]}"
                ) from exc
            except httpx.HTTPError as exc:
                last_exc = exc
        raise LLMConnectionError(f"Could not reach vLLM server at {self._base_url}: {last_exc}")

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = self._resolve_model(request.role)
        start = time.perf_counter()
        payload = self._post_chat_completion(model, self._messages_payload(request), request)
        latency_ms = (time.perf_counter() - start) * 1000
        return self._response_from_payload(payload, model, latency_ms)

    def generate_structured(self, request: LLMRequest, schema: type[SchemaT]) -> SchemaT:
        model = self._resolve_model(request.role)
        messages = self._messages_payload(request)
        try:
            payload = self._post_chat_completion(
                model, messages, request, extra_body={"guided_json": schema.model_json_schema()}
            )
            text = payload["choices"][0]["message"]["content"]
            return self._extract_and_validate_json(text, schema)
        except LLMResponseError:
            # Server doesn't support guided_json (version mismatch) --
            # degrade gracefully to instruction-based JSON prompting.
            fallback_messages = messages + [
                {"role": "user", "content": self._structured_output_instruction(schema)}
            ]
            payload = self._post_chat_completion(model, fallback_messages, request)
            text = payload["choices"][0]["message"]["content"]
            return self._extract_and_validate_json(text, schema)

    def health_check(self) -> bool:
        try:
            response = self._client.get(f"{self._base_url}/models", timeout=5)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _response_from_payload(self, payload: dict[str, Any], model: str, latency_ms: float) -> LLMResponse:
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMResponseError(f"Malformed vLLM response payload: {payload}") from exc
        usage = payload.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            provider_name=self.provider_name,
            model_used=model,
            latency_ms=latency_ms,
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            ),
        )
