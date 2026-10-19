from __future__ import annotations

from collections.abc import Callable

from vulnrem.config.schemas import SandboxConfig
from vulnrem.core.errors import ConfigError
from vulnrem.sandbox.base import Sandbox
from vulnrem.sandbox.docker_sandbox import DockerSandbox
from vulnrem.sandbox.subprocess_sandbox import SubprocessSandbox

_SandboxFactory = Callable[[SandboxConfig], Sandbox]

SANDBOX_REGISTRY: dict[str, _SandboxFactory] = {}


def register_sandbox(implementation: str) -> Callable[[_SandboxFactory], _SandboxFactory]:
    def decorator(factory: _SandboxFactory) -> _SandboxFactory:
        SANDBOX_REGISTRY[implementation] = factory
        return factory

    return decorator


@register_sandbox("subprocess")
def _build_subprocess(_config: SandboxConfig) -> Sandbox:
    return SubprocessSandbox()


@register_sandbox("docker")
def _build_docker(config: SandboxConfig) -> Sandbox:
    return DockerSandbox(image_by_language=config.docker_images)


def build_sandbox(config: SandboxConfig) -> Sandbox:
    factory = SANDBOX_REGISTRY.get(config.implementation)
    if factory is None:
        raise ConfigError(
            f"No sandbox factory registered for implementation={config.implementation!r}. "
            f"Registered: {sorted(SANDBOX_REGISTRY)}"
        )
    return factory(config)
