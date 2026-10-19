"""In-memory mock provider.

Used as the default provider in ``configs/dev.yaml`` (so the whole
repository is runnable and testable with zero external dependencies -- no
GPU, no API key, no network) and directly by unit/integration tests, which
enqueue canned responses per role to script an exact pipeline scenario
(e.g. "Judge rejects once with `regression_failure`, then accepts").
"""

from __future__ import annotations

import time
from collections import defaultdict

from pydantic import BaseModel

from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.enums import AgentRole
from vulnrem.core.errors import LLMStructuredOutputError
from vulnrem.llm.base import LLMProvider, SchemaT
from vulnrem.llm.schemas import LLMRequest, LLMResponse, TokenUsage


class MockLLMProvider(LLMProvider):
    def __init__(
        self,
        models: RoleModelMap | None = None,
        default_text: str = "mock response",
    ):
        super().__init__(models or RoleModelMap(default="mock-model"))
        self._default_text = default_text
        self._text_queues: dict[AgentRole, list[str]] = defaultdict(list)
        self._structured_queues: dict[AgentRole, list[BaseModel | dict]] = defaultdict(list)
        self.call_log: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def enqueue_text(self, role: AgentRole, text: str) -> None:
        self._text_queues[role].append(text)

    def enqueue_structured(self, role: AgentRole, instance: BaseModel | dict) -> None:
        self._structured_queues[role].append(instance)

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_log.append(request)
        start = time.perf_counter()
        queue = self._text_queues[request.role]
        text = queue.pop(0) if queue else self._default_text
        latency_ms = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text=text,
            provider_name=self.provider_name,
            model_used=self._resolve_model(request.role),
            latency_ms=latency_ms,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )

    def generate_structured(self, request: LLMRequest, schema: type[SchemaT]) -> SchemaT:
        self.call_log.append(request)
        queue = self._structured_queues[request.role]
        if not queue:
            raise LLMStructuredOutputError(
                f"MockLLMProvider has no canned structured response queued for "
                f"role={request.role.value}. Call enqueue_structured(role, ...) in your "
                f"test/fixture setup before invoking the pipeline."
            )
        item = queue.pop(0)
        if isinstance(item, schema):
            return item
        return schema.model_validate(item)

    def health_check(self) -> bool:
        return True
