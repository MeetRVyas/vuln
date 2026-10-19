"""Loads and validates the application configuration.

Resolution order for *which* config to load:

1. the ``env`` argument passed explicitly by the caller, else
2. the ``VULNREM_ENV`` environment variable, else
3. ``"dev"``.

Resolution order for *where* to load it from:

1. the ``config_dir`` argument passed explicitly, else
2. the ``VULNREM_CONFIG_DIR`` environment variable, else
3. ``<repo-root>/configs``.

The env-specific file (``dev.yaml`` / ``prod.yaml``) is the primary document.
``hardware_profiles.yaml`` is merged into its ``hardware.profiles`` key so
that hardware profiles are shared between dev and prod rather than
duplicated.

String values anywhere in the YAML may reference environment variables with
``${VAR_NAME}`` or ``${VAR_NAME:-default}`` syntax; this is resolved before
validation. This is *not* how secrets are supplied (see
``LLMProviderConfig.api_key_env_var`` -- providers read the credential
themselves, at call time, from the named variable) -- it exists for ordinary
deployment-varying values such as a vLLM base URL.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from vulnrem.config.schemas import AppConfig
from vulnrem.core.errors import ConfigError

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        def _sub(match: re.Match[str]) -> str:
            var_name, _, default = match.groups()
            return os.environ.get(var_name, default if default is not None else "")

        return _VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value


def _default_config_dir() -> Path:
    env_override = os.environ.get("VULNREM_CONFIG_DIR")
    if env_override:
        return Path(env_override)
    # repo-root/configs, derived from this file's location:
    # src/vulnrem/config/loader.py -> repo-root is 3 parents up from "config"
    return Path(__file__).resolve().parents[3] / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at the top level of {path}, got {type(data)}")
    return data


def load_raw_config(env: str | None = None, config_dir: Path | str | None = None) -> dict[str, Any]:
    """Loads and merges the raw (pre-validation) configuration dict. Exposed
    separately from :func:`load_config` so tests can inspect/mutate the dict
    before validation."""
    resolved_env = env or os.environ.get("VULNREM_ENV", "dev")
    if resolved_env not in ("dev", "prod"):
        raise ConfigError(f"env must be 'dev' or 'prod', got {resolved_env!r}")

    directory = Path(config_dir) if config_dir else _default_config_dir()

    base = _read_yaml(directory / f"{resolved_env}.yaml")
    hardware_profiles = _read_yaml(directory / "hardware_profiles.yaml")

    merged = dict(base)
    hardware_section = dict(merged.get("hardware", {}))
    profiles = hardware_profiles.get("profiles", {})
    # The YAML keys the profile by name already; backfill the `name` field
    # on each entry so authors don't have to repeat it.
    profiles = {
        key: {"name": key, **value} if "name" not in value else value
        for key, value in profiles.items()
    }
    hardware_section["profiles"] = profiles
    merged["hardware"] = hardware_section
    merged["env"] = resolved_env

    return _resolve_env_placeholders(merged)


def load_config(env: str | None = None, config_dir: Path | str | None = None) -> AppConfig:
    """Loads, resolves, and validates the full application configuration."""
    raw = load_raw_config(env=env, config_dir=config_dir)
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc


@lru_cache(maxsize=4)
def get_cached_config(env: str | None = None) -> AppConfig:
    """Process-wide cached accessor. Most application code (dashboard, CLI)
    should use this rather than re-parsing YAML on every call; tests and the
    evaluation harness typically call :func:`load_config` directly so they
    can point at fixture config directories without cache interference."""
    return load_config(env=env)
