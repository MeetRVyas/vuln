from __future__ import annotations

from vulnrem.finetuning.train_qlora import load_job_config


def test_load_job_config_parses_yaml(tmp_path):
    config_path = tmp_path / "job.yaml"
    config_path.write_text(
        """
role: patcher
language: python
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
output_adapter_dir: /data/adapters/patcher-python
train_dataset_path: /data/train/patcher_python.jsonl
manifest_path: /data/splits/manifest.json
quantization: 4bit
lora:
  rank: 8
  alpha: 16
  learning_rate: 0.0002
"""
    )
    config = load_job_config(config_path)

    assert config.role.value == "patcher"
    assert config.language.value == "python"
    assert config.lora.rank == 8
    assert config.lora.alpha == 16
    assert config.quantization == "4bit"
