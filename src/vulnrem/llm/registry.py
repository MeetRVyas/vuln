"""Provider registry: the Open/Closed half of the LLM abstraction.

:mod:`vulnrem.llm.base` gives every provider a common interface (Dependency
Inversion). This module lets a new provider be *added* -- a fourth cloud
vendor, say -- by writing one factory function and registering it with
:func:`register_llm_provider`, without editing this file's existing
entries, :mod:`vulnrem.llm.base`, or any pipeline node. The active provider
for a given deployment is selected purely by the ``llm.active_provider``
string in ``configs/{dev,prod}.yaml``.
"""

from __future__ import annotations

from collections.abc import Callable

from vulnrem.config.schemas import LLMConfig, LLMProviderConfig
from vulnrem.core.errors import LLMConfigurationError
from vulnrem.llm.base import LLMProvider
from vulnrem.llm.providers.google import GoogleGeminiProvider
from vulnrem.llm.providers.mock import MockLLMProvider
from vulnrem.llm.providers.vllm import VLLMProvider

_ProviderFactory = Callable[[LLMProviderConfig], LLMProvider]

LLM_PROVIDER_REGISTRY: dict[str, _ProviderFactory] = {}


def register_llm_provider(kind: str) -> Callable[[_ProviderFactory], _ProviderFactory]:
    def decorator(factory: _ProviderFactory) -> _ProviderFactory:
        LLM_PROVIDER_REGISTRY[kind] = factory
        return factory

    return decorator


@register_llm_provider("mock")
def _build_mock(config: LLMProviderConfig) -> LLMProvider:
    return MockLLMProvider(models=config.models)


@register_llm_provider("google")
def _build_google(config: LLMProviderConfig) -> LLMProvider:
    assert config.api_key_env_var is not None  # enforced by LLMProviderConfig validator
    return GoogleGeminiProvider(
        models=config.models,
        api_key_env_var=config.api_key_env_var,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )


@register_llm_provider("vllm")
def _build_vllm(config: LLMProviderConfig) -> LLMProvider:
    assert config.base_url is not None  # enforced by LLMProviderConfig validator
    return VLLMProvider(
        models=config.models,
        base_url=config.base_url,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )


def build_llm_provider(config: LLMProviderConfig) -> LLMProvider:
    factory = LLM_PROVIDER_REGISTRY.get(config.kind)
    if factory is None:
        raise LLMConfigurationError(
            f"No LLM provider factory registered for kind={config.kind!r}. "
            f"Registered kinds: {sorted(LLM_PROVIDER_REGISTRY)}"
        )
    return factory(config)


def get_active_provider(llm_config: LLMConfig) -> LLMProvider:
    active_config = llm_config.providers[llm_config.active_provider]
    if not active_config.enabled:
        raise LLMConfigurationError(
            f"llm.active_provider={llm_config.active_provider!r} is configured but disabled "
            "(enabled: false)."
        )
    return build_llm_provider(active_config)
