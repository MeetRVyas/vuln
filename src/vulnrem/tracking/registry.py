from __future__ import annotations

from collections.abc import Callable

from vulnrem.config.schemas import TrackingConfig
from vulnrem.core.errors import ConfigError
from vulnrem.tracking.base import ExperimentTracker
from vulnrem.tracking.in_memory import InMemoryTracker
from vulnrem.tracking.mlflow_tracker import MlflowTracker
from vulnrem.tracking.null import NullTracker

_TrackerFactory = Callable[[TrackingConfig], ExperimentTracker]

TRACKER_REGISTRY: dict[str, _TrackerFactory] = {}


def register_tracker(implementation: str) -> Callable[[_TrackerFactory], _TrackerFactory]:
    def decorator(factory: _TrackerFactory) -> _TrackerFactory:
        TRACKER_REGISTRY[implementation] = factory
        return factory

    return decorator


@register_tracker("in_memory")
def _build_in_memory(_config: TrackingConfig) -> ExperimentTracker:
    return InMemoryTracker()


@register_tracker("null")
def _build_null(_config: TrackingConfig) -> ExperimentTracker:
    return NullTracker()


@register_tracker("mlflow")
def _build_mlflow(config: TrackingConfig) -> ExperimentTracker:
    return MlflowTracker(tracking_uri=config.tracking_uri, experiment_name=config.experiment_name)


def build_tracker(config: TrackingConfig) -> ExperimentTracker:
    factory = TRACKER_REGISTRY.get(config.implementation)
    if factory is None:
        raise ConfigError(
            f"No tracker factory registered for implementation={config.implementation!r}. "
            f"Registered: {sorted(TRACKER_REGISTRY)}"
        )
    return factory(config)
