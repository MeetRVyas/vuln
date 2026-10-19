"""What gets tracked, and why each field earns its place.

The instrumentation principle followed here: log what is materially needed
to reproduce, evaluate, compare, debug, and audit a run -- not everything
that happens to be lying around. Full, unbounded code/model output belongs
in structured application logs (see :mod:`vulnrem.observability.logging_config`,
which also truncates/redacts it there), not in MLflow params/metrics, which
are the layer meant to stay small enough to compare across hundreds of runs
at a glance.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FindingRunRecord(BaseModel):
    """One row per finding processed by the per-finding graph -- the unit
    the evaluation harness's recall metrics are computed over."""

    model_config = ConfigDict(extra="forbid")

    file_path: str
    language: str
    finding_id: str
    cwe_id: str | None
    severity: str
    contributing_tools: list[str]

    llm_provider: str
    llm_active_model_role_default: str = ""  # the provider's `default` model, for quick filtering

    triage_confirmed: bool | None
    reproduction_confirmed: bool | None

    final_status: str
    status_detail: str = ""

    attempt_count: int
    attempt_categories: list[str] = Field(default_factory=list)  # Judge category per attempt, in order
    final_functional_pass: bool | None = None
    final_security_pass: bool | None = None

    test_cases_kept: int = 0
    test_cases_discarded: int = 0

    total_duration_s: float = 0.0
    extra_tags: dict[str, str] = Field(default_factory=dict)

    @property
    def remediated(self) -> bool:
        """The locked-in definition of "success": Judge ACCEPT + functional
        oracle PASS + security oracle PASS, all three -- not just a
        terminal status of ACCEPTED (kept here as a derived property so the
        evaluation harness and the tracker agree on one definition)."""
        return (
            self.final_status == "accepted"
            and bool(self.final_functional_pass)
            and bool(self.final_security_pass)
        )


class TrainingRunRecord(BaseModel):
    """One row per fine-tuning run (one per role+language LoRA adapter)."""

    model_config = ConfigDict(extra="forbid")

    role: str
    language: str
    base_model: str

    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int

    dataset_split_manifest_hash: str
    train_examples: int
    eval_examples: int

    final_train_loss: float | None = None
    eval_metrics: dict[str, float] = Field(default_factory=dict)

    duration_s: float = 0.0
    extra_tags: dict[str, str] = Field(default_factory=dict)
