from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vulnrem.core.enums import AgentRole, Language

# Standard attention + MLP projection module names for Qwen-family models --
# what QLoRA actually adapts. Kept as a default rather than hardcoded so a
# different base model's module names are a one-line config override, not a
# code change.
_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoRAHyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: list(_DEFAULT_TARGET_MODULES))

    learning_rate: float = 1.5e-4
    epochs: int = 2
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    seed: int = 42


class FineTuningJobConfig(BaseModel):
    """Everything :mod:`vulnrem.finetuning.train_qlora` needs for one
    adapter (one role x one language)."""

    model_config = ConfigDict(extra="forbid")

    role: AgentRole
    language: Language
    base_model: str  # Hugging Face repo id, e.g. "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    output_adapter_dir: Path
    train_dataset_path: Path  # JSONL produced by vulnrem.finetuning.data_prep
    eval_dataset_path: Path | None = None
    manifest_path: Path  # for the dataset_split_manifest_hash logged alongside this run

    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    lora: LoRAHyperparameters = Field(default_factory=LoRAHyperparameters)

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "vulnrem-finetune"
