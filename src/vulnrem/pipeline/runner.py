"""The single-file check entry point.

Deliberately *not* part of the LangGraph state machine: static-analysis
scanning and aggregation happen once per file, deterministically, with no
LLM involved, before fanning out into one independent per-finding graph run
each (see :mod:`vulnrem.pipeline.graph`). Keeping this fan-out as plain
Python rather than a LangGraph ``Send``-based map-reduce keeps a multi-
finding file easy to reason about and debug -- each finding's whole
history is a self-contained, separately-invokable graph run.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vulnrem.config.schemas import AppConfig
from vulnrem.core.enums import FindingStatus, Language
from vulnrem.core.errors import SASTError
from vulnrem.languages.registry import get_language_profile
from vulnrem.llm.registry import get_active_provider
from vulnrem.oracles.regression.plain import PlainRegressionChecker
from vulnrem.oracles.sanitizer import NotApplicableSanitizerChecker
from vulnrem.oracles.security import SastRescanChecker
from vulnrem.pipeline.aggregator import aggregate_findings
from vulnrem.pipeline.graph import PipelineDependencies, build_pipeline_graph
from vulnrem.pipeline.preprocess import preprocess_source
from vulnrem.pipeline.retry_policy import RetryPolicy
from vulnrem.pipeline.state import Finding, PipelineState
from vulnrem.sandbox.registry import build_sandbox
from vulnrem.sast.registry import build_tools_for_language
from vulnrem.tracking.base import ExperimentTracker
from vulnrem.tracking.registry import build_tracker
from vulnrem.tracking.schemas import FindingRunRecord

logger = logging.getLogger(__name__)


class FindingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    finding: Finding
    final_state: PipelineState


class FileCheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    file_path: str
    language: Language
    preprocess_warnings: list[str]
    findings: list[FindingResult]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.findings:
            key = result.final_state.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts


class FileCheckRunner:
    def __init__(
        self,
        language: Language,
        app_config: AppConfig,
        config_dir: str | None = None,
        llm=None,
        sandbox=None,
        tracker: ExperimentTracker | None = None,
    ):
        self._language = language
        self._language_profile = get_language_profile(language, config_dir=config_dir)
        self._sast_tools = build_tools_for_language(language, app_config.sast)
        cwe_lookup = {c.id: c for c in self._language_profile.spec.cwes}

        # `llm=`/`sandbox=`/`tracker=` let a caller (tests, or a long-lived
        # dashboard process amortizing provider construction across many
        # file checks) inject already-built instances instead of going
        # through the config-driven registry every time.
        llm = llm if llm is not None else get_active_provider(app_config.llm)
        sandbox = sandbox if sandbox is not None else build_sandbox(app_config.sandbox)
        self._tracker = tracker if tracker is not None else build_tracker(app_config.tracking)
        self._llm_provider_name = llm.provider_name
        retry_policy = RetryPolicy(app_config.retry)

        deps = PipelineDependencies(
            llm=llm,
            sandbox=sandbox,
            language_profile=self._language_profile,
            regression_checker=PlainRegressionChecker(),
            security_rescan_checker=SastRescanChecker(self._sast_tools),
            sanitizer_checker=NotApplicableSanitizerChecker(),
            retry_policy=retry_policy,
            sandbox_timeout_s=app_config.sandbox.timeout_s,
            sandbox_memory_limit_mb=app_config.sandbox.memory_limit_mb,
            sandbox_cpu_limit_seconds=app_config.sandbox.cpu_limit_seconds,
            test_writer_max_self_validation_attempts=app_config.retry.max_test_writer_self_validation_attempts,
            cwe_lookup=cwe_lookup,
        )
        self._graph = build_pipeline_graph(deps)

    def check_file(self, file_path: str, source: str) -> FileCheckReport:
        preprocessed = preprocess_source(source)
        raw_findings = self._run_sast(file_path, preprocessed.normalized_source)
        findings = aggregate_findings(raw_findings, self._language)

        logger.info(
            "file_check_started",
            extra={"file_path": file_path, "language": self._language.value, "finding_count": len(findings)},
        )

        results: list[FindingResult] = []
        for finding in findings:
            initial_state = PipelineState(
                language=self._language,
                file_path=file_path,
                original_source=preprocessed.normalized_source,
                finding=finding,
            )
            start = time.perf_counter()
            try:
                raw_result = self._graph.invoke(initial_state)
                final_state = PipelineState.model_validate(raw_result)
            except Exception as exc:  # noqa: BLE001 - one finding's crash must not sink the whole file
                logger.exception("finding_graph_run_failed", extra={"finding_id": finding.finding_id})
                final_state = initial_state.model_copy(
                    update={"status": FindingStatus.ERROR, "status_detail": str(exc)}
                )
            duration_s = time.perf_counter() - start

            self._track_finding_run(file_path, finding, final_state, duration_s)
            results.append(FindingResult(finding=finding, final_state=final_state))

        return FileCheckReport(
            file_path=file_path,
            language=self._language,
            preprocess_warnings=preprocessed.warnings,
            findings=results,
        )

    def _track_finding_run(
        self, file_path: str, finding: Finding, final_state: PipelineState, duration_s: float
    ) -> None:
        report = final_state.verifier_report
        try:
            record = FindingRunRecord(
                file_path=file_path,
                language=self._language.value,
                finding_id=finding.finding_id,
                cwe_id=finding.cwe_id,
                severity=finding.severity.value,
                contributing_tools=finding.contributing_tools(),
                llm_provider=self._llm_provider_name,
                triage_confirmed=final_state.triage_confirmed,
                reproduction_confirmed=(
                    final_state.reproduction.confirmed if final_state.reproduction else None
                ),
                final_status=final_state.status.value,
                status_detail=final_state.status_detail,
                attempt_count=len(final_state.attempt_history),
                attempt_categories=[
                    a.judge_verdict.category.value for a in final_state.attempt_history if a.judge_verdict
                ],
                final_functional_pass=report.functional.passed if report else None,
                final_security_pass=report.security.passed if report else None,
                test_cases_kept=len(final_state.test_battery.test_cases) if final_state.test_battery else 0,
                test_cases_discarded=(
                    len(final_state.test_battery.discarded_case_names) if final_state.test_battery else 0
                ),
                total_duration_s=duration_s,
            )
            self._tracker.log_finding_run(record)
        except Exception:  # noqa: BLE001 - tracking must never break the actual pipeline result
            logger.exception("tracking_log_failed", extra={"finding_id": finding.finding_id})

    def _run_sast(self, file_path: str, normalized_source: str) -> list:
        raw_findings = []
        with tempfile.TemporaryDirectory(prefix="vulnrem-sast-scan-") as tmpdir:
            target = Path(tmpdir) / Path(file_path).name
            target.write_text(normalized_source)
            for tool in self._sast_tools:
                try:
                    found = tool.scan(target)
                except SASTError:
                    logger.warning("sast_scan_failed", extra={"tool": tool.tool_name, "file_path": file_path})
                    continue
                # Report findings against the caller's logical file_path, not
                # the throwaway temp path we had to scan on disk.
                raw_findings.extend(rf.model_copy(update={"file_path": file_path}) for rf in found)
        return raw_findings


def build_file_check_runner(
    app_config: AppConfig,
    language: Language,
    config_dir: str | None = None,
    llm=None,
    sandbox=None,
    tracker: ExperimentTracker | None = None,
) -> FileCheckRunner:
    return FileCheckRunner(
        language=language,
        app_config=app_config,
        config_dir=config_dir,
        llm=llm,
        sandbox=sandbox,
        tracker=tracker,
    )
