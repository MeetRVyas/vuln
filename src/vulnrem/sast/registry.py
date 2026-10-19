from __future__ import annotations

import logging
from collections.abc import Callable

from vulnrem.config.schemas import SASTConfig, SASTToolConfig
from vulnrem.core.enums import Language
from vulnrem.core.errors import ConfigError
from vulnrem.sast.base import SASTTool
from vulnrem.sast.tools.bandit import BanditTool
from vulnrem.sast.tools.not_implemented import NotImplementedSASTTool
from vulnrem.sast.tools.opengrep import OpengrepTool

logger = logging.getLogger(__name__)

_ToolFactory = Callable[[SASTToolConfig], SASTTool]

SAST_TOOL_REGISTRY: dict[str, _ToolFactory] = {}


def register_sast_tool(name: str) -> Callable[[_ToolFactory], _ToolFactory]:
    def decorator(factory: _ToolFactory) -> _ToolFactory:
        SAST_TOOL_REGISTRY[name] = factory
        return factory

    return decorator


@register_sast_tool("opengrep")
def _build_opengrep(config: SASTToolConfig) -> SASTTool:
    return OpengrepTool(timeout_s=config.timeout_s, extra_args=config.extra_args)


@register_sast_tool("bandit")
def _build_bandit(config: SASTToolConfig) -> SASTTool:
    return BanditTool(timeout_s=config.timeout_s, extra_args=config.extra_args)


@register_sast_tool("cppcheck")
def _build_cppcheck(_config: SASTToolConfig) -> SASTTool:
    return NotImplementedSASTTool("cppcheck")


@register_sast_tool("clang-static-analyzer")
def _build_clang_static_analyzer(_config: SASTToolConfig) -> SASTTool:
    return NotImplementedSASTTool("clang-static-analyzer")


@register_sast_tool("spotbugs")
def _build_spotbugs(_config: SASTToolConfig) -> SASTTool:
    return NotImplementedSASTTool("spotbugs")


def build_tools_for_language(language: Language, sast_config: SASTConfig) -> list[SASTTool]:
    """Instantiates every *enabled* tool configured for ``language``,
    skipping (with a warning, not a hard failure) any whose binary isn't
    actually installed -- a missing optional scanner shouldn't take down
    the whole pipeline."""
    tool_configs = sast_config.tools_by_language.get(language, [])
    tools: list[SASTTool] = []
    for tool_config in tool_configs:
        if not tool_config.enabled:
            continue
        factory = SAST_TOOL_REGISTRY.get(tool_config.name)
        if factory is None:
            raise ConfigError(
                f"No SAST tool factory registered for name={tool_config.name!r}. "
                f"Registered: {sorted(SAST_TOOL_REGISTRY)}"
            )
        tool = factory(tool_config)
        if not tool.is_available():
            logger.warning(
                "sast_tool_unavailable_skipping",
                extra={"tool": tool_config.name, "language": language.value},
            )
            continue
        tools.append(tool)
    return tools
