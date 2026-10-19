"""Cloud LLM provider backed by Google's Gemini API.

Exists so development and testing can happen without a GPU / local vLLM
server (per the project's requirement that the LLM layer be swappable
between the fine-tuned local model and a cloud model). The API key is never
read from config directly: :class:`GoogleGeminiProvider` is handed the name
of an environment variable (``api_key_env_var`` in
:class:`~vulnrem.config.schemas.LLMProviderConfig`) and reads the real value
itself, at construction time, from ``os.environ``.

The concrete ``genai.Client`` (or any object with a matching
``.models.generate_content`` surface) can be injected via ``client=`` --
this is what the unit tests use to exercise the request/response mapping
without a real network call or real credential.
"""

from __future__ import annotations

import os
import time
from typing import Any

from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.errors import LLMConfigurationError, LLMConnectionError, LLMResponseError
from vulnrem.llm.base import LLMProvider, SchemaT
from vulnrem.llm.schemas import LLMRequest, LLMResponse, TokenUsage


class GoogleGeminiProvider(LLMProvider):
    def __init__(
        self,
        models: RoleModelMap,
        api_key_env_var: str,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ):
        super().__init__(models)
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client = client if client is not None else self._build_client(api_key_env_var, timeout_s)

    @staticmethod
    def _build_client(api_key_env_var: str, timeout_s: float) -> Any:
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise LLMConfigurationError(
                f"Environment variable {api_key_env_var!r} is not set; the Google provider "
                "needs it to hold a valid Gemini API key. See docs/SETUP_DEPLOYMENT.md."
            )
        from google import genai
        from google.genai import types

        try:
            return genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
            )
        except (TypeError, AttributeError):
            # Defensive fallback in case the installed SDK version's
            # HttpOptions signature has changed -- correctness of the
            # request itself matters more than honoring the timeout knob.
            return genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "google"

    def _generate_content_with_retries(self, **kwargs: Any) -> Any:
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.models.generate_content(**kwargs)
            except Exception as exc:
                if getattr(exc, "code", None) not in {500, 502, 503, 504} or attempt == self._max_retries:
                    raise
                time.sleep(2**attempt)

    def _to_contents_and_system(self, request: LLMRequest) -> tuple[list[Any], str | None]:
        from google.genai import types

        system_parts = [m.content for m in request.messages if m.role == "system"]
        system_instruction = "\n\n".join(system_parts) or None

        contents = []
        for message in request.messages:
            if message.role == "system":
                continue
            gemini_role = "model" if message.role == "assistant" else "user"
            contents.append(types.Content(role=gemini_role, parts=[types.Part(text=message.content)]))
        return contents, system_instruction

    def generate(self, request: LLMRequest) -> LLMResponse:
        from google.genai import types

        contents, system_instruction = self._to_contents_and_system(request)
        model = self._resolve_model(request.role)
        start = time.perf_counter()
        try:
            response = self._generate_content_with_retries(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - SDK exception hierarchy varies by version
            raise self._wrap_exception(exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        text = getattr(response, "text", None)
        if not text:
            raise LLMResponseError(f"Google provider returned an empty completion for model={model}")

        usage_meta = getattr(response, "usage_metadata", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", None),
            output_tokens=getattr(usage_meta, "candidates_token_count", None),
        )
        return LLMResponse(
            text=text, provider_name=self.provider_name, model_used=model, latency_ms=latency_ms, usage=usage
        )

    def generate_structured(self, request: LLMRequest, schema: type[SchemaT]) -> SchemaT:
        from google.genai import types

        contents, system_instruction = self._to_contents_and_system(request)
        model = self._resolve_model(request.role)
        try:
            response = self._generate_content_with_retries(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                    response_mime_type="application/json",
                    response_json_schema=schema.model_json_schema(),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_exception(exc) from exc

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        # Native parsing didn't succeed (e.g. installed SDK version doesn't
        # populate `.parsed` for this schema shape) -- fall back to the
        # shared extract-and-validate helper against the raw text.
        text = getattr(response, "text", "") or ""
        return self._extract_and_validate_json(text, schema)

    def health_check(self) -> bool:
        try:
            models_pager = self._client.models.list()
            next(iter(models_pager), None)
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _wrap_exception(exc: Exception) -> Exception:
        message = str(exc)
        transient_markers = ("connection", "timeout", "timed out", "dns", "unreachable")
        if any(marker in message.lower() for marker in transient_markers):
            return LLMConnectionError(f"Google provider connection failure: {message}")
        return LLMResponseError(f"Google provider request failed: {message}")
