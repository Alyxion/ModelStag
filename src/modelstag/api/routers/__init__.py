"""API routers."""

from modelstag.api.routers.health import router as health_router
from modelstag.api.routers.models import router as models_router
from modelstag.api.routers.inference import router as inference_router

__all__ = ["health_router", "models_router", "inference_router"]
