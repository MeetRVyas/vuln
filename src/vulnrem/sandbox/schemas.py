from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vulnrem.core.enums import Language


class ExecutionRequest(BaseModel):
    """Everything a :class:`~vulnrem.sandbox.base.Sandbox` needs to run one
    isolated command. ``files`` are written into a fresh, empty workspace
    before ``command`` is executed with that workspace as its cwd -- this is
    intentionally the *only* way code reaches the sandbox; no ambient
    filesystem access is implied."""

    model_config = ConfigDict(extra="forbid")

    language: Language
    files: dict[str, str] = Field(
        description="relative-path -> file content, written into a fresh workspace"
    )
    command: list[str]
    timeout_s: float = 20.0
    memory_limit_mb: int = 512
    cpu_limit_seconds: float = 15.0
    network_disabled: bool = True
    env: dict[str, str] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    resource_limit_exceeded: bool = False
    duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.resource_limit_exceeded
