"""The LLM abstraction.

Every concrete backend (a real local vLLM server serving the fine-tuned
Qwen3-Coder + per-role LoRA adapters, Google's Gemini API for cloud
dev/testing, or an in-memory mock for unit tests / CI) implements this same
three-method contract. Every pipeline node depends only on
:class:`LLMProvider` -- never on a concrete class -- so which backend is
actually running is a single config value
(``llm.active_provider`` in ``configs/{dev,prod}.yaml``), not a code change.
This is the Dependency Inversion half of the requirement; the Open/Closed
half is :mod:`vulnrem.llm.registry`, which lets a brand new provider be
added without touching any existing provider or any pipeline node.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.enums import AgentRole
from vulnrem.core.errors import LLMStructuredOutputError
from vulnrem.llm.schemas import LLMRequest, LLMResponse

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMProvider(ABC):
    """Required interface every LLM backend must implement.

    Subclasses receive a :class:`~vulnrem.config.schemas.RoleModelMap` at
    construction time and should use :meth:`_resolve_model` to turn an
    ``AgentRole`` into a concrete model/adapter name -- this is what lets a
    single provider serve every pipeline role (base model for most roles,
    fine-tuned LoRA adapter for the Patcher/Test-Writer) purely through
    configuration.
    """

    def __init__(self, models: RoleModelMap):
        self._models = models

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short, stable identifier (``"vllm"``, ``"google"``, ``"mock"``)
        used in logs, MLflow tags, and error messages."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Free-text completion."""

    @abstractmethod
    def generate_structured(self, request: LLMRequest, schema: type[SchemaT]) -> SchemaT:
        """Completion validated/parsed into ``schema``. Implementations
        should prefer the provider's native structured/guided-decoding
        feature when one exists (Gemini's ``response_schema``, vLLM's
        ``guided_json``) and fall back to prompt-plus-parse (see
        :meth:`_extract_and_validate_json`) otherwise. Must raise
        :class:`~vulnrem.core.errors.LLMStructuredOutputError` -- never
        return a semantically-empty/default instance -- if the response
        cannot be made to conform, so callers can treat this as a genuine
        failure (e.g. a `syntax_or_plumbing` retry) rather than a silent
        wrong answer."""

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap reachability check (no content generation). Used by the
        dashboard's startup checks and by tests to skip provider-specific
        suites when the backend isn't reachable."""

    # -- shared helpers available to every subclass -------------------- #

    def _resolve_model(self, role: AgentRole) -> str:
        return self._models.resolve(role)

    @staticmethod
    def _structured_output_instruction(schema: type[BaseModel]) -> str:
        """Appended to the prompt for providers with no native structured-
        output mode. Native-capable providers (Google, vLLM's guided
        decoding) don't need this -- they pass the schema directly."""
        return (
            "\n\nRespond with ONLY a single JSON object that conforms to this "
            "JSON Schema, and nothing else (no prose, no markdown fences):\n"
            f"{json.dumps(schema.model_json_schema())}"
        )

    @staticmethod
    def _extract_and_validate_json(text: str, schema: type[SchemaT]) -> SchemaT:
        """Best-effort extraction of a JSON object from free text (handles
        the common case of a model wrapping it in a ```json fence anyway),
        then strict Pydantic validation."""
        candidates = [text.strip()]
        fence_match = _JSON_FENCE_RE.search(text)
        if fence_match:
            candidates.insert(0, fence_match.group(1).strip())

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                return schema.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                continue

        raise LLMStructuredOutputError(
            f"Could not parse a {schema.__name__} from model output: {last_error}\n"
            f"Raw output (truncated): {text[:500]!r}"
        )
