from __future__ import annotations

import pytest

from vulnrem.config.loader import load_config
from vulnrem.core.errors import ConfigError


def test_dev_config_loads_and_validates(config_dir):
    cfg = load_config(env="dev", config_dir=config_dir)
    assert cfg.env == "dev"
    assert cfg.llm.active_provider == "google"
    assert set(cfg.hardware.profiles) == {"small", "medium", "large"}
    assert cfg.hardware.profiles["medium"].quantization == "int8"


def test_prod_config_loads_and_validates(config_dir):
    cfg = load_config(env="prod", config_dir=config_dir)
    assert cfg.llm.active_provider == "vllm"
    assert cfg.sandbox.implementation == "docker"
    assert cfg.tracking.implementation == "mlflow"


def test_unknown_env_rejected(config_dir):
    with pytest.raises(ConfigError):
        load_config(env="staging", config_dir=config_dir)


def test_missing_config_dir_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(env="dev", config_dir=tmp_path)


def test_env_var_placeholder_resolution(config_dir, monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://my-custom-host:9000/v1")
    cfg = load_config(env="dev", config_dir=config_dir)
    assert cfg.llm.providers["vllm"].base_url == "http://my-custom-host:9000/v1"


def test_env_var_placeholder_falls_back_to_default_when_unset(config_dir, monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    cfg = load_config(env="dev", config_dir=config_dir)
    assert cfg.llm.providers["vllm"].base_url == "http://localhost:8000/v1"


def test_prod_config_supports_hf_spaces_topology_via_env_vars(config_dir, monkeypatch):
    monkeypatch.setenv("VULNREM_LLM_PROVIDER", "google")
    monkeypatch.setenv("VULNREM_SANDBOX_IMPL", "subprocess")
    cfg = load_config(env="prod", config_dir=config_dir)
    assert cfg.llm.active_provider == "google"
    assert cfg.sandbox.implementation == "subprocess"
