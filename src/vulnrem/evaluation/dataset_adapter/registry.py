from __future__ import annotations

from collections.abc import Callable

from vulnrem.core.errors import ConfigError
from vulnrem.evaluation.dataset_adapter.base import DatasetAdapter
from vulnrem.evaluation.dataset_adapter.generic_json import GenericJsonLinesAdapter
from vulnrem.evaluation.dataset_adapter.not_implemented import NotImplementedDatasetAdapter

_AdapterFactory = Callable[..., DatasetAdapter]

DATASET_ADAPTER_REGISTRY: dict[str, _AdapterFactory] = {
    "generic_jsonl": lambda **kwargs: GenericJsonLinesAdapter(**kwargs),
    "bigvul": lambda **kwargs: NotImplementedDatasetAdapter("bigvul", **kwargs),
    "cvefixes": lambda **kwargs: NotImplementedDatasetAdapter("cvefixes", **kwargs),
    "crossvul": lambda **kwargs: NotImplementedDatasetAdapter("crossvul", **kwargs),
    "diversevul": lambda **kwargs: NotImplementedDatasetAdapter("diversevul", **kwargs),
    "primevul": lambda **kwargs: NotImplementedDatasetAdapter("primevul", **kwargs),
}


def build_dataset_adapter(name: str, **kwargs) -> DatasetAdapter:
    factory = DATASET_ADAPTER_REGISTRY.get(name)
    if factory is None:
        raise ConfigError(
            f"No dataset adapter registered for name={name!r}. Registered: "
            f"{sorted(DATASET_ADAPTER_REGISTRY)}"
        )
    return factory(**kwargs)
