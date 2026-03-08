"""Custom exceptions for ModelStag."""


class ModelStagError(Exception):
    """Base exception for ModelStag."""
    pass


class ModelNotFoundError(ModelStagError):
    """Model not found in configuration."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model not found: {model_name}")


class ModelNotRunningError(ModelStagError):
    """Model worker is not running."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model not running: {model_name}")


class ModelAlreadyRunningError(ModelStagError):
    """Model worker is already running."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model already running: {model_name}")


class WorkerError(ModelStagError):
    """Error in worker process."""
    pass


class IPCError(ModelStagError):
    """Error in inter-process communication."""
    pass


class InferenceError(ModelStagError):
    """Error during inference."""
    pass
