# vulnrem

An automated code-vulnerability detection & remediation pipeline: static analysis (Opengrep +
Bandit) finds a candidate issue in a single source file, and an LLM-driven multi-agent
loop (Triage -> Reproducer -> Localizer -> Test-Writer -> Patcher -> Verifier -> Judge)
dynamically confirms it, patches it, and verifies the fix against a dual oracle before
accepting it -- orchestrated per finding by a LangGraph state machine.

Python is the only language wired up end-to-end in this build (C and Java are named,
scoped, and have a documented extension point, but no implementation -- see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). Everything is designed to run against
either a local fine-tuned model (served via vLLM) or a cloud model (Google Gemini), and
against either a real Docker sandbox or a local-dev subprocess fallback, chosen entirely
by configuration -- see that same doc for the full "why."

## Quickstart (no GPU, no Docker, no API key required)

```bash
uv sync --extra dashboard --extra tracking --group dev
uv run pytest                    # full test suite -- exercises real Bandit, real subprocess
                                  # sandboxing, real pytest parsing; only LLM calls are mocked
uv run vulnrem hardware-profile  # resolves to the "small" fallback profile without a GPU
```

To actually check a file, you need a working LLM provider. The fastest path without a
GPU is Google's API:

```bash
export GOOGLE_API_KEY=...
# configs/dev.yaml: change llm.active_provider to `google`
uv run vulnrem scan path/to/file.py --json
```

Or launch the dashboard:

```bash
uv run python -m vulnrem.dashboard.app
```

## Project layout

```
src/vulnrem/      the package (see docs/ARCHITECTURE.md for the full module map)
configs/          dev.yaml / prod.yaml + hardware_profiles.yaml + per-language specs
docker/           Dockerfiles + docker-compose.yml for the self-hosted deployment
tests/            unit/ + integration/ (see docs/ARCHITECTURE.md's "Testing strategy")
docs/             ARCHITECTURE.md, SETUP_DEPLOYMENT.md, DATASET_SCHEMA.md
```

## Documentation

- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) -- the SOLID rationale, the module map, every
  extension point, and the scope cuts made for this build tier.
- [**SETUP_DEPLOYMENT.md**](docs/SETUP_DEPLOYMENT.md) -- GPU/CUDA, vLLM, Docker, fine-tuning,
  MLflow, secrets, and both deployment topologies (self-hosted docker-compose vs. Hugging
  Face Spaces).
- [**DATASET_SCHEMA.md**](docs/DATASET_SCHEMA.md) -- the canonical dataset record shape and
  how to point the project at a real dataset.

## Status

This is a complete MVP at the Strong-minor-project tier, not a prototype: the full SRP
agent roster (including the Reproducer and Judge), categorized retry/escalation, the
dual-oracle Verifier, and Docker-based sandboxing are all implemented, not stubbed --
see `docs/ARCHITECTURE.md`'s "Deliberate scope cuts" section for exactly what is
intentionally out of scope and why, and what each extension point needs to add it later.
