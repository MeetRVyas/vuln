# Setup & Deployment Guide

This covers everything needed *outside* the codebase itself: none of it was runnable in
the environment this repository was built in (no GPU, no Docker daemon, no vLLM), so
treat the exact commands below as a well-informed starting point -- check current tool
docs before running them, especially version-pinned flags.

## 1. Prerequisites

| Requirement | Needed for |
|---|---|
| Python 3.12+ and [`uv`](https://docs.astral.sh/uv/) | everything |
| A CUDA GPU (see the VRAM table below) + recent NVIDIA driver + CUDA 12.x | serving the local model, fine-tuning |
| Docker Engine + NVIDIA Container Toolkit | the self-hosted deployment topology, the Docker sandbox backend, the vLLM container |
| A Google Gemini API key | the cloud LLM provider (dev/testing, or Hugging Face Spaces deployment) |

## 2. Local install (no Docker, no GPU) -- for development

```bash
uv sync --extra dashboard --extra tracking --group dev
uv run pytest                      # the whole test suite runs with zero external services
uv run vulnrem hardware-profile    # falls back to the "small" profile without a GPU
```

The default `configs/dev.yaml` uses the `mock` LLM provider and the `subprocess` sandbox,
so the pipeline is fully exercisable without any of the infrastructure below. To try it
against a real model without a local GPU:

```bash
export GOOGLE_API_KEY=...
# then edit configs/dev.yaml: llm.active_provider: google
uv run vulnrem scan path/to/file.py
```

## 3. Hardware profiles and quantization

`configs/hardware_profiles.yaml` defines three tiers; `vulnrem hardware-profile` resolves
which one applies (explicit `hardware.override_profile` > `nvidia-smi`-based
auto-detection > `hardware.fallback_profile`). Use the resolved profile's `quantization`
and `gpu_memory_utilization` values when constructing the vLLM launch command below.

| Profile | VRAM | Quantization | Notes |
|---|---|---|---|
| `small` | ~16-24GB (RTX 4080/4090) | AWQ (4-bit) | Higher latency, smaller context |
| `medium` | ~24-48GB (A6000, L40S) | INT8 | The project's originally-assumed class |
| `large` | 80GB+ (A100/H100) | BF16 | Minimal quantization, best throughput |

## 4. Serving the base model with vLLM (multi-LoRA)

```bash
pip install vllm   # or use the vllm/vllm-openai Docker image, see docker-compose.yml

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --enable-lora \
  --lora-modules patcher-python-lora=/models/adapters/patcher-python \
                 test-writer-python-lora=/models/adapters/test-writer-python \
  --quantization awq \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

`VLLMProvider` (`src/vulnrem/llm/providers/vllm.py`) talks to this server's
OpenAI-compatible `/chat/completions` endpoint at `VLLM_BASE_URL` (default
`http://localhost:8000/v1`), and routes each pipeline role to a model/adapter name purely
via `configs/{env}.yaml`'s `llm.providers.vllm.models` map -- adding a fine-tuned adapter
for a new role is a config change (plus the `--lora-modules` flag above), not a code
change. Check vLLM's current docs for the exact multi-LoRA flag names/behavior for the
version you install -- this moves between releases.

## 5. Fine-tuning a LoRA adapter

```bash
uv sync --extra finetune   # torch, transformers, peft, bitsandbytes, trl, datasets, accelerate

# 1. Split the dataset (see docs/DATASET_SCHEMA.md for the expected format)
uv run vulnrem dataset-split data/my_dataset.jsonl --manifest-out data/splits/manifest.json

# 2. Build Patcher training examples from the train split only
uv run vulnrem finetune-data-prep data/my_dataset.jsonl \
  --manifest data/splits/manifest.json --output data/train/patcher_python.jsonl

# 3. Write a job config (see FineTuningJobConfig in src/vulnrem/finetuning/config.py)
cat > job.yaml <<'EOF'
role: patcher
language: python
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
output_adapter_dir: models/adapters/patcher-python
train_dataset_path: data/train/patcher_python.jsonl
manifest_path: data/splits/manifest.json
quantization: 4bit
lora:
  rank: 16
  alpha: 32
  learning_rate: 0.00015
  epochs: 2
EOF

uv run python -m vulnrem.finetuning.train_qlora job.yaml
```

`train_qlora.py` has never been executed in this build environment (no GPU/torch here) --
its imports are deferred specifically so the rest of the codebase never needs `torch`
installed, but before a real run, sanity-check the installed `transformers`/`trl`/`peft`
versions' current API against the script; these libraries move quickly.

Repeat for `role: test_writer` once you have a dataset of confirmed PoC scripts per
finding (see the docstring on `build_test_writer_training_example` in
`src/vulnrem/finetuning/data_prep.py` for exactly what that needs).

## 6. Running the evaluation harness

```bash
uv run vulnrem eval-run data/my_dataset.jsonl \
  --manifest data/splits/manifest.json \
  --output-json outputs/eval_report.json \
  --output-markdown outputs/eval_report.md
```

This runs every held-out record through the *real* `FileCheckRunner` (whatever
`llm.active_provider`/`sandbox.implementation` your active config selects) and reports
detection recall, remediation recall, and per-CWE/per-language breakdowns.

## 7. Experiment tracking (MLflow)

> **Version note**: MLflow 3.x put the `file:` tracking-store backend into maintenance
> mode (it now raises `MlflowException` unless `MLFLOW_ALLOW_FILE_STORE=true` is set).
> This project defaults to a SQLite backend (`sqlite:///mlflow.db`) instead, which needs
> no separate server.

```bash
# View runs locally:
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# Or point at a real MLflow tracking server instead:
export MLFLOW_TRACKING_URI=https://your-mlflow-server
```

Set `tracking.implementation: mlflow` in `configs/{env}.yaml` (the dev default is
`in_memory`, for zero-setup local runs and tests).

## 8. Secrets

Every credential is read from an environment variable named *by config*
(`llm.providers.google.api_key_env_var`, currently `GOOGLE_API_KEY`) -- config YAML files
never contain a secret value and are safe to commit. Copy `.env.example` to `.env` (or set
these in your deployment platform's secret manager) and never commit the real `.env`.

## 9. Deployment topology A: self-hosted, `docker compose`

Full stack, real Docker sandboxing, your own GPU:

```bash
docker build -f docker/Dockerfile.sandbox-python -t vulnrem-sandbox-python:latest .
mkdir -p models/adapters data
cd docker && docker compose up --build
```

This starts three services (`docker/docker-compose.yml`): `vllm` (the model server),
`dashboard` (the Gradio app, `sandbox.implementation=docker`, with the host's Docker
socket bind-mounted in so it can start sibling `vulnrem-sandbox-python` containers), and
`mlflow-ui`. If the dashboard container can't reach the Docker socket, it's almost always
a permissions issue -- either run the container as a user in the host's `docker` group, or
(simplest for a single-user box) run the whole stack as root.

## 10. Deployment topology B: Hugging Face Spaces

A Space's container has **no Docker socket, no privileged mode, and (outside a paid GPU
Space) no GPU** -- it cannot start further Docker containers, so `sandbox.implementation`
must be `subprocess` here, and the LLM provider must be something reachable over plain
HTTPS (`google`, or a `vllm` server you host elsewhere and point `VLLM_BASE_URL` at).

1. Create a **Docker**-type Space and point it at `docker/Dockerfile.dashboard` (it
   already listens on port 7860, which Docker-type Spaces expect).
2. Set these as Space secrets/variables:
   ```
   VULNREM_ENV=prod
   VULNREM_LLM_PROVIDER=google
   VULNREM_SANDBOX_IMPL=subprocess
   GOOGLE_API_KEY=<your key>
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db   # or omit tracking entirely by leaving tracking.implementation at in_memory
   ```
3. Push. `configs/prod.yaml` already reads both `VULNREM_LLM_PROVIDER` and
   `VULNREM_SANDBOX_IMPL` (defaulting to `vllm`/`docker` for topology A above) -- no config
   file edit is needed to switch topologies, only these environment variables.

The dashboard's core logic (`vulnrem.dashboard.service.PipelineService`) and the pipeline
itself are identical between the two topologies; only which concrete `LLMProvider` and
`Sandbox` get constructed changes.
