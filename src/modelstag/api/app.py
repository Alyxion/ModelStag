"""FastAPI application with lifespan management."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from modelstag.config.settings import Settings, get_settings
from modelstag.manager.process_manager import ProcessManager
from modelstag.api.routers import health_router, models_router, inference_router

# Paths
STATIC_DIR = Path(__file__).parent / "static"
IMAGES_DIR = Path.cwd() / "images"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting ModelStag API")

    settings = get_settings()
    manager = ProcessManager(settings)

    app.state.settings = settings
    app.state.manager = manager

    # Sync startup
    manager.startup()

    yield

    # Shutdown
    logger.info("Shutting down ModelStag API")
    manager.shutdown()


def create_app(settings: Settings = None) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="ModelStag",
        description="Multi-model hosting system for AI vision models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(inference_router)

    # Serve test UI
    @app.get("/", response_class=HTMLResponse)
    @app.get("/test", response_class=HTMLResponse)
    async def test_page():
        """Serve the test UI page."""
        test_html = STATIC_DIR / "test.html"
        if test_html.exists():
            return FileResponse(test_html)
        return HTMLResponse("<h1>Test page not found</h1>", status_code=404)

    # List available images
    @app.get("/images")
    def list_images():
        """List available sample images."""
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        images = []
        if IMAGES_DIR.exists():
            for f in sorted(IMAGES_DIR.iterdir()):
                if f.suffix.lower() in extensions:
                    images.append(f.name)
        return {"images": images}

    # Serve sample images
    @app.get("/images/{filename}")
    async def get_image(filename: str):
        """Serve sample images."""
        image_path = IMAGES_DIR / filename
        if image_path.exists() and image_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
            return FileResponse(image_path)
        return HTMLResponse("Image not found", status_code=404)

    return app


# Default app instance
app = create_app()
