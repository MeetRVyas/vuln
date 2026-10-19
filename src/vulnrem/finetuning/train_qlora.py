"""QLoRA fine-tuning for one (role, language) LoRA adapter.

Requires the ``finetune`` optional dependency group
(``uv sync --extra finetune``: torch, transformers, peft, bitsandbytes, trl,
datasets, accelerate) and a CUDA GPU -- neither is available in the
environment this repository was built in, so this script is real,
production-intent code that has **not** been executed here. All its
imports are deferred inside :func:`run_training` so importing this module
(e.g. from a test that only calls :func:`load_job_config`) never requires
torch to be installed. Before running for real, check the installed
``transformers``/``trl``/``peft`` versions' current docs -- these libraries'
APIs move quickly and this was written against a particular point-in-time
understanding of them (see docs/SETUP_DEPLOYMENT.md).

Training examples come from :mod:`vulnrem.finetuning.data_prep`, which
builds each example by calling the *real* runtime prompt-builder for the
target role, so the fine-tuned adapter is trained on exactly the prompt
shape it will be served with.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from vulnrem.evaluation.splitter import load_manifest
from vulnrem.finetuning.config import FineTuningJobConfig
from vulnrem.tracking.mlflow_tracker import MlflowTracker
from vulnrem.tracking.schemas import TrainingRunRecord


def load_job_config(path: Path | str) -> FineTuningJobConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return FineTuningJobConfig.model_validate(raw)


def run_training(config: FineTuningJobConfig) -> TrainingRunRecord:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    start = time.perf_counter()

    quantization_config = None
    if config.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=str(config.train_dataset_path), split="train")
    eval_dataset = (
        load_dataset("json", data_files=str(config.eval_dataset_path), split="train")
        if config.eval_dataset_path is not None
        else None
    )

    def _format(example: dict) -> str:
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    sft_config = SFTConfig(
        output_dir=str(config.output_adapter_dir),
        num_train_epochs=config.lora.epochs,
        per_device_train_batch_size=config.lora.per_device_batch_size,
        gradient_accumulation_steps=config.lora.gradient_accumulation_steps,
        learning_rate=config.lora.learning_rate,
        warmup_ratio=config.lora.warmup_ratio,
        weight_decay=config.lora.weight_decay,
        max_length=config.lora.max_seq_length,
        seed=config.lora.seed,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],  # logged to MLflow ourselves via ExperimentTracker below, not the trainer's built-in integration
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=_format,
    )

    train_result = trainer.train()
    trainer.save_model(str(config.output_adapter_dir))

    eval_metrics: dict[str, float] = {}
    if eval_dataset is not None:
        eval_metrics = {
            f"eval_{key}": value for key, value in trainer.evaluate().items() if isinstance(value, int | float)
        }

    duration_s = time.perf_counter() - start
    manifest = load_manifest(config.manifest_path)

    record = TrainingRunRecord(
        role=config.role.value,
        language=config.language.value,
        base_model=config.base_model,
        lora_rank=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        learning_rate=config.lora.learning_rate,
        epochs=config.lora.epochs,
        per_device_batch_size=config.lora.per_device_batch_size,
        gradient_accumulation_steps=config.lora.gradient_accumulation_steps,
        dataset_split_manifest_hash=manifest.dataset_checksum,
        train_examples=len(train_dataset),
        eval_examples=len(eval_dataset) if eval_dataset is not None else 0,
        final_train_loss=getattr(train_result, "training_loss", None),
        eval_metrics=eval_metrics,
        duration_s=duration_s,
    )

    tracker = MlflowTracker(
        tracking_uri=config.mlflow_tracking_uri, experiment_name=config.mlflow_experiment_name
    )
    tracker.log_training_run(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune one role/language adapter.")
    parser.add_argument("job_config", type=Path, help="Path to a FineTuningJobConfig YAML file.")
    args = parser.parse_args()

    config = load_job_config(args.job_config)
    record = run_training(config)
    print(f"Finished: final_train_loss={record.final_train_loss} duration_s={record.duration_s:.1f}")


if __name__ == "__main__":
    main()
