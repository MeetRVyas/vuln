"""MLflow-backed tracker.

Uses ``mlflow.tracking.MlflowClient`` (the object-oriented API) rather than
the fluent ``mlflow.start_run()`` global-state API, so that constructing a
tracker never mutates process-wide active-run state -- important since the
dashboard may process many files/findings against the same long-lived
tracker instance.
"""

from __future__ import annotations

from typing import Any

from vulnrem.tracking.base import ExperimentTracker
from vulnrem.tracking.schemas import FindingRunRecord, TrainingRunRecord


class MlflowTracker(ExperimentTracker):
    def __init__(self, tracking_uri: str, experiment_name: str, client: Any | None = None):
        if client is not None:
            self._client = client
        else:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(tracking_uri=tracking_uri)
        self._experiment_id = self._get_or_create_experiment(experiment_name)

    def _get_or_create_experiment(self, name: str) -> str:
        experiment = self._client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        return self._client.create_experiment(name)

    def log_finding_run(self, record: FindingRunRecord) -> None:
        run = self._client.create_run(self._experiment_id, run_name=f"finding-{record.finding_id}")
        run_id = run.info.run_id

        tags = {
            "kind": "finding_run",
            "language": record.language,
            "cwe_id": record.cwe_id or "unknown",
            "final_status": record.final_status,
            "llm_provider": record.llm_provider,
            "contributing_tools": ",".join(record.contributing_tools),
            "attempt_categories": ",".join(record.attempt_categories),
            **record.extra_tags,
        }
        for key, value in tags.items():
            self._client.set_tag(run_id, key, value)

        params = {
            "file_path": record.file_path,
            "severity": record.severity,
            "llm_model_default": record.llm_active_model_role_default,
        }
        for key, value in params.items():
            self._client.log_param(run_id, key, value)

        metrics: dict[str, float] = {
            "attempt_count": float(record.attempt_count),
            "test_cases_kept": float(record.test_cases_kept),
            "test_cases_discarded": float(record.test_cases_discarded),
            "total_duration_s": record.total_duration_s,
            "remediated": 1.0 if record.remediated else 0.0,
        }
        if record.final_functional_pass is not None:
            metrics["functional_pass"] = 1.0 if record.final_functional_pass else 0.0
        if record.final_security_pass is not None:
            metrics["security_pass"] = 1.0 if record.final_security_pass else 0.0
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, value)

        self._client.set_terminated(run_id, status="FINISHED")

    def log_training_run(self, record: TrainingRunRecord) -> None:
        run = self._client.create_run(
            self._experiment_id, run_name=f"train-{record.role}-{record.language}"
        )
        run_id = run.info.run_id

        tags = {"kind": "training_run", "role": record.role, "language": record.language, **record.extra_tags}
        for key, value in tags.items():
            self._client.set_tag(run_id, key, value)

        params = {
            "base_model": record.base_model,
            "lora_rank": record.lora_rank,
            "lora_alpha": record.lora_alpha,
            "lora_dropout": record.lora_dropout,
            "learning_rate": record.learning_rate,
            "epochs": record.epochs,
            "per_device_batch_size": record.per_device_batch_size,
            "gradient_accumulation_steps": record.gradient_accumulation_steps,
            "dataset_split_manifest_hash": record.dataset_split_manifest_hash,
        }
        for key, value in params.items():
            self._client.log_param(run_id, key, value)

        metrics: dict[str, float] = {
            "train_examples": float(record.train_examples),
            "eval_examples": float(record.eval_examples),
            "duration_s": record.duration_s,
        }
        if record.final_train_loss is not None:
            metrics["final_train_loss"] = record.final_train_loss
        metrics.update(record.eval_metrics)
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, value)

        self._client.set_terminated(run_id, status="FINISHED")
