from __future__ import annotations

from vulnrem.tracking.base import ExperimentTracker
from vulnrem.tracking.schemas import FindingRunRecord, TrainingRunRecord


class NullTracker(ExperimentTracker):
    def log_finding_run(self, record: FindingRunRecord) -> None:
        pass

    def log_training_run(self, record: TrainingRunRecord) -> None:
        pass
