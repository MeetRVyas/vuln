# Architecture

This is the Strong-minor-project-tier MVP (per the project plan's §3.4 tiers) of an
automated vulnerability-detection-and-remediation pipeline: static analysis finds a
candidate issue, an LLM-driven multi-agent loop confirms it, localizes it, patches it,
writes tests for it, verifies the patch against a dual oracle, and a Judge decides
accept/retry/give-up -- all orchestrated per finding by a small LangGraph state machine.

The guiding constraint throughout: **every external capability (an LLM, a sandboxed
execution environment, a static analyzer, an experiment tracker) sits behind an
abstract interface with a config-driven registry choosing the concrete implementation.**
That is what makes "swap the fine-tuned local model for Gemini during development" or
"add C support later" changes to configuration and one new class, not a refactor.

## Module map

```
config/         AppConfig schema + YAML loader (dev.yaml/prod.yaml) + hardware profile resolution
core/           Shared enums (Language, AgentRole, RetryCategory, FindingStatus) and the exception hierarchy
llm/            LLMProvider(ABC) + mock/google/vllm implementations + registry
sandbox/        Sandbox(ABC) + subprocess/docker implementations + registry
sast/           SASTTool(ABC) + opengrep/bandit wrappers + registry
languages/      LanguageProfile(ABC) + the Python implementation + registry
ast_tools/      Python AST-based same-file slicer (used by the Python LanguageProfile)
search/         RepoSearchTool(ABC) + the single-file implementation
oracles/        RegressionChecker(ABC)+plain impl, the SAST-rescan security oracle, the sanitizer stub
pipeline/       Finding aggregation, the per-finding LangGraph, prompts, nodes, retry policy, the file-level runner
evaluation/     DatasetAdapter(ABC), the stratified splitter, the harness (runs the real pipeline), metrics/report
finetuning/     LoRA config, training-data prep (reuses the real prompt builders), the QLoRA training script
tracking/       ExperimentTracker(ABC) + MLflow/in-memory/null implementations + registry
observability/  Structured JSON/plain logging + secret redaction
dashboard/      Gradio UI + a thin PipelineService seam to the pipeline
cli/            `vulnrem` command-line entrypoint
```

## Two levels of orchestration

A single-file check is **two** layers, not one:

1. **The outer runner** (`pipeline/runner.py`, `pipeline/aggregator.py`,
   `pipeline/preprocess.py`) is plain, deterministic Python: normalize the file, run every
   registered SAST tool for the language, merge overlapping raw findings into
   deduplicated `Finding`s. No LLM call happens here.
2. **The per-finding graph** (`pipeline/graph.py`) is a compiled LangGraph
   `StateGraph[PipelineState]`, one independent run per `Finding`:

   ```
   START -> triage -> [false positive?] --------------------> finalize -> END
                    -> reproducer -> [not confirmed?] -------> finalize -> END
                                   -> localizer -> test_writer -> patcher
                                      -> verifier -> judge -> [retry?] -> patcher (loop)
                                                             -> finalize -> END
   ```

Why split it this way: a file can contain several unrelated findings, and each one's
retry loop is independent. Modeling that fan-out as plain Python (a `for` loop over
`Finding`s, each invoking its own compiled graph) is far easier to reason about, log, and
unit-test than a single graph handling both the aggregation fan-out and the per-finding
retry loop with LangGraph's `Send` map-reduce API. The trade-off: findings within one
file are checked sequentially, not concurrently. Parallelizing that loop (or running
Patcher and Test-Writer as true concurrent branches within one finding, which their data
dependencies would actually allow -- see the docstring in `graph.py`) is a
non-architectural follow-up, not a redesign.

## The Reproducer, and why Triage isn't the only filter

Two independent false-positive filters run before any patch is attempted:

* **Triage** is a cheap, static LLM judgment call ("does this look real from reading the
  code?"). It exists to skip obviously-benign findings without spending a sandbox
  execution on them.
* **Reproducer** actually writes and *runs* a minimal PoC against the original code,
  inside the Sandbox. Only a finding that was dynamically confirmed --  a real exploit
  actually observed, not inferred -- proceeds to Localizer/Patcher.

The Reproducer's confirmed PoC is also **reused** by the Test-Writer (which converts it
into the battery's one `kind="poc"` pytest test) instead of being independently
re-derived, so the test the Verifier's security oracle ultimately trusts is grounded in
something that was actually observed to work, not merely an LLM's second guess at what a
PoC should look like.

## Test-Writer self-validation

Every test case the Test-Writer proposes carries an `expected_result_on_original`
("fail" for the PoC -- the vulnerability should trigger against the unpatched file --
"pass" for regression/edge-case tests). Before anything is handed to the Verifier, each
case is actually run against the original source; only cases whose real outcome matches
what the model claimed survive into the battery (`TestBattery.discarded_case_names`
records the rest). If zero PoC tests survive, the whole battery is regenerated (up to
`retry.max_test_writer_self_validation_attempts`) -- an un-survivable PoC would silently
make the security oracle vacuously fail forever otherwise. This is what makes the retry
loop's ground truth trustworthy rather than "whatever the model said it expects."

## The dual oracle and the locked-in definition of "remediated"

`VerifierReport` has exactly two top-level oracles, matching the project's locked-in
success definition:

* **Functional oracle** (`regression` + `edge_case` sub-results): did the patch preserve
  everything the patch wasn't supposed to touch?
* **Security oracle** (`poc_no_longer_triggers` + `sast_rescan_clean`): is the
  vulnerability actually gone, both dynamically (the PoC test now passes) and statically
  (a fresh SAST scan of the patched file no longer flags the same CWE near the same
  line)?

A finding counts as **remediated** only when both oracles pass *and* the Judge accepts
(`FindingRunRecord.remediated` in `tracking/schemas.py` is the single place this
three-part definition lives) -- never a bare "status == accepted", and never one oracle
standing in for the other.

## Categorized retry / escalation (`pipeline/retry_policy.py`)

The Judge classifies every non-accept verdict into one of five categories. Two of them
route to something this Strong-tier MVP does not implement (both are Ambitious-tier, per
the project plan's §3.4 phasing) and are handled by *graceful degradation* rather than a
missing feature crashing the run:

| Judge category | Would normally escalate to | What this build actually does |
|---|---|---|
| `syntax_or_plumbing` | -- (cheap, mechanical) | retry, counted against `max_normal_retries` |
| `regression_failure` | -- | retry, counted against `max_normal_retries` |
| `suspicious_or_degenerate` | -- | retry, counted against `max_normal_retries` |
| `vulnerability_still_present` | **Debugger-in-the-loop** (instrumented root-cause trace) | retry anyway, using only the Judge's rationale as signal, counted against a *separate*, smaller `max_escalation_attempts` budget |
| `multi_file_scope_needed` | **Repo-scale agent** (edits >1 file) | immediately unresolved -- a single-file Patcher cannot fix this by construction, so there is no degraded retry |

## Extension points (what's a config/registry change vs. what needs new code)

| Capability | Extend by | Where |
|---|---|---|
| New LLM provider | Register a factory | `llm/registry.py` |
| New sandbox backend | Register a factory | `sandbox/registry.py` |
| New SAST tool | Register a factory + language mapping | `sast/registry.py` |
| C or Java support | Implement `LanguageProfile` + `configs/languages/<lang>.yaml` | `languages/registry.py` |
| Repo-wide search (vs. single-file) | Implement `RepoSearchTool` | `search/base.py` |
| Debugger-in-the-loop | Implement a `DebuggerPort`-style class the `vulnerability_still_present` route calls before the graceful-degradation fallback | `pipeline/retry_policy.py`, `pipeline/graph.py` |
| Repo-scale (multi-file) agent | Same shape as the Debugger extension point, for `multi_file_scope_needed` | `pipeline/retry_policy.py`, `pipeline/graph.py` |
| Coverage/behavior-fingerprint regression checking | Implement `RegressionChecker` | `oracles/regression/base.py` |
| A named public dataset (BigVul, CVEFixes, ...) | Implement `DatasetAdapter` | `evaluation/dataset_adapter/registry.py` |
| A new evaluation metric | Pass an `extra_metric_fns` callable | `evaluation/metrics.py` |
| A new experiment-tracking backend | Register a factory | `tracking/registry.py` |

## Deliberate scope cuts (this build vs. what's designed-for)

* **Single-file only.** Repo-wide search, cross-file localization, and multi-file
  patching are explicitly out of scope (per the project's own instruction) but every
  seam that would need to change is the single `RepoSearchTool` extension point above,
  plus the `multi_file_scope_needed` route already existing in the retry policy.
* **Python only.** C and Java are named in the SAST toolchain (`sast/tools/not_implemented.py`
  registers `cppcheck`, `clang-static-analyzer`, and `spotbugs` as documented,
  not-yet-implemented entries) and in `core/enums.Language`, but have no `LanguageProfile`.
* **Debugger-in-the-loop and the repo-scale agent are Ambitious-tier** (see the project
  plan's §3.4) and are not implemented -- see the retry-policy table above for exactly
  how their absence is handled instead of silently breaking the pipeline.
* **Patcher and Test-Writer run sequentially**, not as parallel graph branches, for MVP
  simplicity (see `graph.py`'s docstring) even though their data dependencies would allow
  true parallelism.
* **Sanitizer checking is `not_applicable`** for Python (it's a C-language dynamic-analysis
  concept -- ASan/UBSan); `oracles/sanitizer.py` is the extension point for when C support
  is added.

## Testing strategy

* Every provider/sandbox/tool class supports dependency injection of its underlying
  client/subprocess-runner specifically so it can be unit-tested against a fake without a
  real network call, Docker daemon, or installed binary (`tests/unit/test_llm_google.py`,
  `test_sandbox_docker.py`, `test_sast_tools.py`).
* `tests/integration/test_pipeline_end_to_end.py` and `test_pipeline_early_exits.py` run
  the *entire* compiled graph for real -- real Bandit scan, real subprocess Sandbox
  execution, real pytest parsing -- with only the LLM calls scripted via `MockLLMProvider`,
  including a scenario that exercises the Judge -> Patcher retry loop twice before
  accepting.
* `tests/unit/test_tracking.py` writes a real run to a local SQLite-backed MLflow store
  and reads it back via `MlflowClient`, not just asserting that a mock was called.
