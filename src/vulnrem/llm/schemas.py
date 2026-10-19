from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vulnrem.core.enums import AgentRole


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """Provider-agnostic request. ``role`` is the routing key each provider
    uses to pick a concrete model/adapter (see
    :class:`vulnrem.config.schemas.RoleModelMap`) -- callers never name a
    model directly, which is what makes swapping providers a config change."""

    model_config = ConfigDict(extra="forbid")

    role: AgentRole
    messages: list[Message]
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(2048, ge=1, le=32768)
    # Free-form correlation data (finding_id, attempt_number, ...) threaded
    # through to logs/MLflow. Never treated as part of the prompt.
    metadata: dict[str, str] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    provider_name: str
    model_used: str
    latency_ms: float
    usage: TokenUsage = Field(default_factory=TokenUsage)
