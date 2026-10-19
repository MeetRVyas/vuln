from __future__ import annotations

from pathlib import Path

import pytest

from vulnrem.config.schemas import RoleModelMap
from vulnrem.llm.providers.mock import MockLLMProvider

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider(models=RoleModelMap(default="mock-model"))


@pytest.fixture
def sql_injection_source() -> str:
    return (FIXTURES_DIR / "vulnerable_samples" / "sql_injection.py").read_text()
