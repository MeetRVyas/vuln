from __future__ import annotations

from vulnrem.tracking.in_memory import InMemoryTracker
from vulnrem.tracking.mlflow_tracker import MlflowTracker
from vulnrem.tracking.null import NullTracker
from vulnrem.tracking.registry import build_tracker
from vulnrem.tracking.schemas import FindingRunRecord, TrainingRunRecord


def _finding_record(**overrides) -> FindingRunRecord:
    defaults = dict(
        file_path="target.py",
        language="python",
        finding_id="abc123",
        cwe_id="CWE-89",
        severity="high",
        contributing_tools=["opengrep", "bandit"],
        llm_provider="mock",
        llm_active_model_role_default="mock-model",
        triage_confirmed=True,
        reproduction_confirmed=True,
        final_status="accepted",
        status_detail="",
        attempt_count=1,
        attempt_categories=["none"],
        final_functional_pass=True,
        final_security_pass=True,
        test_cases_kept=3,
        test_cases_discarded=0,
        total_duration_s=1.23,
    )
    defaults.update(overrides)
    return FindingRunRecord(**defaults)


def _training_record(**overrides) -> TrainingRunRecord:
    defaults = dict(
        role="patcher",
        language="python",
        base_model="qwen3-coder-30b-a3b-instruct",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        learning_rate=1.5e-4,
        epochs=2,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        dataset_split_manifest_hash="deadbeef",
        train_examples=800,
        eval_examples=200,
        final_train_loss=0.42,
        eval_metrics={"detection_recall": 0.9},
        duration_s=3600.0,
    )
    defaults.update(overrides)
    return TrainingRunRecord(**defaults)


def test_in_memory_tracker_records_both_kinds():
    tracker = InMemoryTracker()
    tracker.log_finding_run(_finding_record())
    tracker.log_training_run(_training_record())

    assert len(tracker.finding_runs) == 1
    assert len(tracker.training_runs) == 1
    assert tracker.finding_runs[0].remediated is True


def test_finding_record_remediated_requires_all_three_conditions():
    accepted_but_functional_failed = _finding_record(final_functional_pass=False)
    assert accepted_but_functional_failed.remediated is False

    not_accepted = _finding_record(final_status="unresolved_retries_exhausted")
    assert not_accepted.remediated is False

    fully_remediated = _finding_record()
    assert fully_remediated.remediated is True


def test_null_tracker_is_a_no_op():
    tracker = NullTracker()
    tracker.log_finding_run(_finding_record())
    tracker.log_training_run(_training_record())  # must not raise


def test_registry_builds_in_memory_and_null():
    from vulnrem.config.schemas import TrackingConfig

    assert isinstance(build_tracker(TrackingConfig(implementation="in_memory")), InMemoryTracker)
    assert isinstance(build_tracker(TrackingConfig(implementation="null")), NullTracker)


def test_mlflow_tracker_writes_a_real_run_to_a_local_sqlite_store(tmp_path):
    # MLflow 3.x put the `file:` backend into maintenance mode -- sqlite is
    # the supported no-server local backend (see configs/dev.yaml).
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MlflowTracker(tracking_uri=tracking_uri, experiment_name="vulnrem-test")

    record = _finding_record()
    tracker.log_finding_run(record)

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("vulnrem-test")
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert run.data.tags["cwe_id"] == "CWE-89"
    assert run.data.tags["final_status"] == "accepted"
    assert run.data.params["file_path"] == "target.py"
    assert run.data.metrics["remediated"] == 1.0
    assert run.data.metrics["attempt_count"] == 1.0
    assert run.info.status == "FINISHED"


def test_mlflow_tracker_writes_a_training_run(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MlflowTracker(tracking_uri=tracking_uri, experiment_name="vulnrem-train-test")

    tracker.log_training_run(_training_record())

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("vulnrem-train-test")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert run.data.params["lora_rank"] == "16"
    assert run.data.metrics["final_train_loss"] == 0.42
    assert run.data.metrics["detection_recall"] == 0.9
