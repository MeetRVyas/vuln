from __future__ import annotations

from vulnrem.tracking.base import ExperimentTracker
from vulnrem.tracking.schemas import FindingRunRecord, TrainingRunRecord


class InMemoryTracker(ExperimentTracker):
    def __init__(self):
        self.finding_runs: list[FindingRunRecord] = []
        self.training_runs: list[TrainingRunRecord] = []

    def log_finding_run(self, record: FindingRunRecord) -> None:
        self.finding_runs.append(record)

    def log_training_run(self, record: TrainingRunRecord) -> None:
        self.training_runs.append(record)
