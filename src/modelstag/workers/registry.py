"""Worker type registry."""

from typing import Dict, List, Type

from modelstag.core.types import ModelType
from modelstag.workers.base import BaseModelWorker


class WorkerRegistry:
    """Registry mapping model types to worker classes."""

    _registry: Dict[ModelType, Type[BaseModelWorker]] = {}

    @classmethod
    def register(cls, model_type: ModelType):
        """Decorator to register a worker class."""

        def decorator(worker_class: Type[BaseModelWorker]):
            cls._registry[model_type] = worker_class
            return worker_class

        return decorator

    @classmethod
    def get(cls, model_type: ModelType) -> Type[BaseModelWorker]:
        """Get worker class for model type."""
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._registry[model_type]

    @classmethod
    def registered_types(cls) -> List[ModelType]:
        """Get list of registered model types."""
        return list(cls._registry.keys())


def get_worker_class(model_type: ModelType) -> Type[BaseModelWorker]:
    """Get worker class for model type."""
    return WorkerRegistry.get(model_type)
