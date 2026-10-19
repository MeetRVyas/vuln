"""Typed configuration schema.

Every runtime knob in the system is declared here as a Pydantic model, so
that:

* a malformed YAML file fails fast, at startup, with a clear validation
  error -- not three pipeline stages later;
* every module that needs a setting gets a fully-typed object instead of
  poking around in a raw dict;
* adding a new provider / sandbox backend / language is "add a new variant
  to the relevant ``kind`` literal + config block", not a schema rewrite.

Secrets (API keys) are never modelled as plain string *values* here -- only
as the *name of the environment variable* that holds them
(``api_key_env_var``). The loader resolves that indirection at load time
(see :mod:`vulnrem.config.loader`). This means config YAML files are safe to
commit to version control.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vulnrem.core.enums import AgentRole, Language

# --------------------------------------------------------------------------- #
# Hardware profiles
# --------------------------------------------------------------------------- #


class HardwareProfile(BaseModel):
    """One entry in ``hardware_profiles.yaml``. Describes how the local
    (vLLM) serving stack should be configured for a class of GPU."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    min_vram_gb: float
    max_vram_gb: float | None = None
    quantization: Literal["awq", "gptq", "int4", "int8", "fp16", "bf16", "none"] = "awq"
    gpu_memory_utilization: float = Field(0.90, gt=0, le=1.0)
    max_concurrent_requests: int = Field(4, ge=1)
    max_model_len: int = Field(8192, ge=512)
    kv_cache_dtype: Literal["auto", "fp8", "fp16"] = "auto"
    notes: str = ""


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Explicit override always wins. If unset, the resolver in
    # `vulnrem.config.hardware` tries auto-detection and falls back to
    # `fallback_profile`.
    override_profile: str | None = None
    fallback_profile: str = "small"
    profiles: dict[str, HardwareProfile]

    @model_validator(mode="after")
    def _validate_names(self) -> HardwareConfig:
        if self.override_profile and self.override_profile not in self.profiles:
            raise ValueError(
                f"hardware.override_profile={self.override_profile!r} is not a key in "
                f"hardware.profiles ({sorted(self.profiles)})"
            )
        if self.fallback_profile not in self.profiles:
            raise ValueError(
                f"hardware.fallback_profile={self.fallback_profile!r} is not a key in "
                f"hardware.profiles ({sorted(self.profiles)})"
            )
        return self


# --------------------------------------------------------------------------- #
# LLM providers
# --------------------------------------------------------------------------- #


class RoleModelMap(BaseModel):
    """Maps each pipeline role to the concrete model/adapter name a provider
    should use for it. ``default`` is used for any role not explicitly
    listed."""

    model_config = ConfigDict(extra="forbid")

    default: str
    per_role: dict[AgentRole, str] = Field(default_factory=dict)

    def resolve(self, role: AgentRole) -> str:
        return self.per_role.get(role, self.default)


class LLMProviderConfig(BaseModel):
    """One provider entry under ``llm.providers``. ``kind`` selects which
    concrete :class:`~vulnrem.llm.base.LLMProvider` implementation the
    registry instantiates -- see :mod:`vulnrem.llm.registry`."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mock", "google", "vllm"]
    enabled: bool = True
    base_url: str | None = None  # required for "vllm"
    api_key_env_var: str | None = None  # required for "google"
    timeout_s: float = 60.0
    max_retries: int = 2
    models: RoleModelMap
    extra: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_kind_requirements(self) -> LLMProviderConfig:
        if self.kind == "vllm" and not self.base_url:
            raise ValueError("llm provider of kind 'vllm' requires 'base_url'")
        if self.kind == "google" and not self.api_key_env_var:
            raise ValueError("llm provider of kind 'google' requires 'api_key_env_var'")
        return self


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_provider: str
    providers: dict[str, LLMProviderConfig]

    @model_validator(mode="after")
    def _validate_active(self) -> LLMConfig:
        if self.active_provider not in self.providers:
            raise ValueError(
                f"llm.active_provider={self.active_provider!r} is not a key in "
                f"llm.providers ({sorted(self.providers)})"
            )
        return self


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implementation: Literal["subprocess", "docker"] = "subprocess"
    timeout_s: float = 30.0
    memory_limit_mb: int = 512
    cpu_limit_seconds: float = 20.0
    network_disabled: bool = True
    docker_images: dict[Language, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# SAST
# --------------------------------------------------------------------------- #


class SASTToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    timeout_s: float = 60.0
    extra_args: list[str] = Field(default_factory=list)


class SASTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools_by_language: dict[Language, list[SASTToolConfig]]


# --------------------------------------------------------------------------- #
# Retry / escalation policy
# --------------------------------------------------------------------------- #


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_normal_retries: int = Field(4, ge=0, le=10)
    max_escalation_attempts: int = Field(2, ge=0, le=5)
    max_test_writer_self_validation_attempts: int = Field(2, ge=1, le=5)


# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implementation: Literal["mlflow", "in_memory", "null"] = "in_memory"
    # `file:` MLflow tracking stores are in maintenance mode as of MLflow 3.x
    # (mlflow.exceptions.MlflowException unless MLFLOW_ALLOW_FILE_STORE=true)
    # -- sqlite is the supported no-server default instead.
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "vulnrem-dev"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = True
    redact_secrets: bool = True
    max_code_chars_in_logs: int = Field(
        2000, description="Full code/model output is never dumped past this length into logs."
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    held_out_fraction: float = Field(0.2, gt=0, lt=1)
    random_seed: int = 42
    min_cases_per_stratum: int = Field(
        1, ge=1, description="Minimum records per (language, cwe) stratum kept in eval split"
    )
    manifest_path: str = "data/splits/manifest.json"


# --------------------------------------------------------------------------- #
# Top-level application config
# --------------------------------------------------------------------------- #


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: Literal["dev", "prod"] = "dev"
    enabled_languages: list[Language] = Field(default_factory=lambda: [Language.PYTHON])

    llm: LLMConfig
    sandbox: SandboxConfig
    sast: SASTConfig
    hardware: HardwareConfig
    retry: RetryConfig
    tracking: TrackingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
