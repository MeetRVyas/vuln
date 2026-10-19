"""Metrics computed over a set of :class:`EvalCaseResult` -- one per
held-out dataset record, produced by :mod:`vulnrem.evaluation.harness`.

Kept deliberately extensible per the project plan: ``compute_metrics``
accepts arbitrary ``extra_metric_fns`` so a new metric is a new function
passed in at call time, not a change to this module's schema, and
:func:`compute_breakdown_by_cwe` reuses the exact same computation per
stratum rather than a separate implementation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from vulnrem.core.enums import Language


class EvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    language: Language
    expected_cwe_id: str

    sast_detected: bool  # did any SAST tool flag *anything* in this file
    detected: bool  # did Triage+Reproducer dynamically confirm a real finding
    remediated: bool  # did it reach FindingStatus.ACCEPTED with both oracles passing

    final_status: str | None = None
    notes: str = ""


class MetricWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detection: float = 0.4
    remediation: float = 0.6


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_eval_cases: int
    sast_detected_count: int
    detected_count: int
    remediated_count: int

    sast_detection_rate: float
    detection_recall: float
    remediation_recall: float
    weighted_overall_score: float

    additional_metrics: dict[str, float] = Field(default_factory=dict)


def _rate(numerator: int, total: int) -> float:
    return (numerator / total) if total else 0.0


def compute_metrics(
    results: list[EvalCaseResult],
    weights: MetricWeights | None = None,
    extra_metric_fns: dict[str, Callable[[list[EvalCaseResult]], float]] | None = None,
) -> EvaluationMetrics:
    total = len(results)
    weights = weights or MetricWeights()

    sast_detected = sum(1 for r in results if r.sast_detected)
    detected = sum(1 for r in results if r.detected)
    remediated = sum(1 for r in results if r.remediated)

    detection_recall = _rate(detected, total)
    remediation_recall = _rate(remediated, total)

    additional_metrics = {name: fn(results) for name, fn in (extra_metric_fns or {}).items()}

    return EvaluationMetrics(
        total_eval_cases=total,
        sast_detected_count=sast_detected,
        detected_count=detected,
        remediated_count=remediated,
        sast_detection_rate=_rate(sast_detected, total),
        detection_recall=detection_recall,
        remediation_recall=remediation_recall,
        weighted_overall_score=weights.detection * detection_recall + weights.remediation * remediation_recall,
        additional_metrics=additional_metrics,
    )


def compute_breakdown_by_cwe(
    results: list[EvalCaseResult], weights: MetricWeights | None = None
) -> dict[str, EvaluationMetrics]:
    by_cwe: dict[str, list[EvalCaseResult]] = defaultdict(list)
    for result in results:
        by_cwe[result.expected_cwe_id].append(result)
    return {cwe: compute_metrics(group, weights) for cwe, group in by_cwe.items()}


def compute_breakdown_by_language(
    results: list[EvalCaseResult], weights: MetricWeights | None = None
) -> dict[Language, EvaluationMetrics]:
    by_language: dict[Language, list[EvalCaseResult]] = defaultdict(list)
    for result in results:
        by_language[result.language].append(result)
    return {language: compute_metrics(group, weights) for language, group in by_language.items()}
