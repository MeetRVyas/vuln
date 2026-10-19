"""Schemas passed to :meth:`~vulnrem.llm.base.LLMProvider.generate_structured`.

Kept separate from :mod:`vulnrem.pipeline.state` because these describe
exactly what we ask the *model* to return -- not everything a node
persists into ``PipelineState`` (e.g. the Patcher persists a computed
unified diff alongside the model's output, but never asks the model to
produce one). Where a state schema already matches an LLM output 1:1
(``JudgeVerdict``, ``TestCase``) we reuse it directly instead of
duplicating an identical class.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vulnrem.pipeline.state import TestCase


class TriageVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_true_positive: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ReproducerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poc_script: str
    explanation: str


class PatcherOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patched_source: str
    explanation: str


class TestWriterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_cases: list[TestCase]
