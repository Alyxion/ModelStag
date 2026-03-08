"""Model management endpoints."""

from fastapi import APIRouter, Request, HTTPException

from modelstag.api.schemas.responses import ModelInfo, ModelsListResponse
from modelstag.core.exceptions import (
    ModelNotFoundError,
    ModelNotRunningError,
    ModelAlreadyRunningError,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsListResponse)
def list_models(request: Request) -> ModelsListResponse:
    """List all configured models with their status."""
    manager = request.app.state.manager
    settings = request.app.state.settings

    statuses = manager.get_all_statuses()
    models = []

    for name, status in statuses.items():
        config = settings.get_model(name)
        models.append(
            ModelInfo(
                name=name,
                status=status["status"],
                type=status["type"],
                startup=status["startup"],
                output_formats=config.output_formats if config else [],
                error=status.get("error"),
            )
        )

    return ModelsListResponse(models=models)


@router.get("/{model_name}", response_model=ModelInfo)
def get_model(request: Request, model_name: str) -> ModelInfo:
    """Get information about a specific model."""
    manager = request.app.state.manager
    settings = request.app.state.settings

    try:
        statuses = manager.get_all_statuses()
        if model_name not in statuses:
            raise ModelNotFoundError(model_name)

        status = statuses[model_name]
        config = settings.get_model(model_name)

        return ModelInfo(
            name=model_name,
            status=status["status"],
            type=status["type"],
            startup=status["startup"],
            output_formats=config.output_formats if config else [],
            error=status.get("error"),
        )

    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")


@router.post("/{model_name}/start")
def start_model(request: Request, model_name: str):
    """Start a model worker."""
    manager = request.app.state.manager

    try:
        manager.start_worker(model_name)
        return {"status": "started", "model": model_name}

    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    except ModelAlreadyRunningError:
        raise HTTPException(
            status_code=409, detail=f"Model already running: {model_name}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/stop")
def stop_model(request: Request, model_name: str, force: bool = False):
    """Stop a model worker."""
    manager = request.app.state.manager

    try:
        manager.stop_worker(model_name)
        return {"status": "stopped", "model": model_name}

    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    except ModelNotRunningError:
        raise HTTPException(
            status_code=409, detail=f"Model not running: {model_name}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop-all")
def stop_all_models(request: Request, force: bool = False):
    """Stop all running model workers."""
    manager = request.app.state.manager

    try:
        manager.stop_all()
        return {"status": "stopped_all"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
