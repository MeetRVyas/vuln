from __future__ import annotations

from abc import ABC, abstractmethod

from vulnrem.tracking.schemas import FindingRunRecord, TrainingRunRecord


class ExperimentTracker(ABC):
    @abstractmethod
    def log_finding_run(self, record: FindingRunRecord) -> None: ...

    @abstractmethod
    def log_training_run(self, record: TrainingRunRecord) -> None: ...

    def close(self) -> None:  # noqa: B027 - intentionally concrete, not abstract: most backends need no teardown
        """Optional hook for backends that hold a live connection/handle.
        Default no-op so most implementations don't need to override it."""
