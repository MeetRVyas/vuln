"""Runs the held-out evaluation set through the *real* production pipeline
(:class:`~vulnrem.pipeline.runner.FileCheckRunner`) -- there is no separate
"evaluation-only" code path, so a reported recall number always reflects
exactly the pipeline that would run in the dashboard.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from vulnrem.core.enums import FindingStatus, Language
from vulnrem.evaluation.metrics import EvalCaseResult
from vulnrem.evaluation.schemas import CanonicalRecord
from vulnrem.pipeline.runner import FileCheckRunner

logger = logging.getLogger(__name__)


class EvaluationHarness:
    def __init__(self, runner_factory: Callable[[Language], FileCheckRunner]):
        self._runner_factory = runner_factory
        self._runners: dict[Language, FileCheckRunner] = {}

    def _runner_for(self, language: Language) -> FileCheckRunner:
        if language not in self._runners:
            self._runners[language] = self._runner_factory(language)
        return self._runners[language]

    def run(self, eval_records: list[CanonicalRecord]) -> list[EvalCaseResult]:
        results: list[EvalCaseResult] = []
        for record in eval_records:
            results.append(self._run_one(record))
        return results

    def _run_one(self, record: CanonicalRecord) -> EvalCaseResult:
        try:
            runner = self._runner_for(record.language)
            report = runner.check_file(f"{record.record_id}.py", record.vulnerable_code)
        except Exception as exc:  # noqa: BLE001 - one bad eval case must not abort the whole run
            logger.exception("eval_case_failed", extra={"record_id": record.record_id})
            return EvalCaseResult(
                record_id=record.record_id,
                language=record.language,
                expected_cwe_id=record.cwe_id,
                sast_detected=False,
                detected=False,
                remediated=False,
                notes=f"harness error: {exc}",
            )

        sast_detected = len(report.findings) > 0
        # Prefer findings whose aggregated CWE matches the label; if none
        # matched exactly (a tool's CWE mapping can legitimately differ from
        # the dataset's label for the same real issue) fall back to
        # whatever was found, so a labeling mismatch doesn't manufacture a
        # false "missed" result for a genuinely-surfaced finding.
        matching = [fr for fr in report.findings if fr.finding.cwe_id == record.cwe_id]
        candidates = matching or report.findings

        detected = any(
            fr.final_state.reproduction is not None and fr.final_state.reproduction.confirmed
            for fr in candidates
        )
        remediated = any(fr.final_state.status == FindingStatus.ACCEPTED for fr in candidates)
        final_status = candidates[0].final_state.status.value if candidates else None

        return EvalCaseResult(
            record_id=record.record_id,
            language=record.language,
            expected_cwe_id=record.cwe_id,
            sast_detected=sast_detected,
            detected=detected,
            remediated=remediated,
            final_status=final_status,
        )
