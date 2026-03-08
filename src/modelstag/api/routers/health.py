"""Health check endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from modelstag.api.schemas.responses import HealthResponse
from modelstag.core.types import ModelStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Overall health check with model statuses."""
    manager = request.app.state.manager
    statuses = manager.get_all_statuses()

    # Determine overall health
    all_ok = all(
        s["status"] in (ModelStatus.RUNNING.value, ModelStatus.STOPPED.value)
        for s in statuses.values()
    )

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        models={name: s["status"] for name, s in statuses.items()},
    )


@router.get("/health/ready")
def ready(request: Request) -> JSONResponse:
    """Kubernetes readiness probe."""
    manager = request.app.state.manager
    statuses = manager.get_all_statuses()

    # Ready if at least one model is running
    any_running = any(
        s["status"] == ModelStatus.RUNNING.value for s in statuses.values()
    )

    if any_running:
        return JSONResponse({"status": "ready"})
    else:
        return JSONResponse({"status": "not_ready"}, status_code=503)


@router.get("/health/live")
def live() -> JSONResponse:
    """Kubernetes liveness probe."""
    return JSONResponse({"status": "alive"})
