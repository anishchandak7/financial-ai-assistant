import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.v1.api_router import api_router
from observability.logger import get_custom_logger
from observability.middleware import ObservabilityMiddleware
from rag.vector_store import vector_store

logger = get_custom_logger(name=__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Financial AI Assistant API",
    description="Enterprise Multi-Agent Financial Assistant with ReAct workflows, RAG, and real-time market telemetry.",
    version="2.0.0"
)

# Configure CORS (supports comma-separated origins via ALLOWED_ORIGINS env var)
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
if allowed_origins_env == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Production Observability Middleware (Request/Trace correlation & telemetry)
app.add_middleware(ObservabilityMiddleware)

# Mount modular API routes (preserving /api/... for frontend compatibility)
app.include_router(api_router, prefix="/api")

# Serve built frontend Single-Page Application (SPA) if present
frontend_dist_dirs = [
    PROJECT_ROOT / "webapp" / "dist",
    BACKEND_DIR / "dist",
]
frontend_dist = next((d for d in frontend_dist_dirs if d.is_dir()), None)

if frontend_dist:
    assets_dir = frontend_dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Do not intercept API routes, OpenAPI documentation, or metrics
        if (
            full_path.startswith("api")
            or full_path.startswith("docs")
            or full_path.startswith("redoc")
            or full_path.startswith("openapi.json")
        ):
            raise HTTPException(status_code=404, detail="Endpoint not found")

        candidate_file = frontend_dist / full_path
        if full_path and candidate_file.is_file():
            return FileResponse(candidate_file)

        index_file = frontend_dist / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Frontend index.html not found")
else:
    @app.get("/")
    async def root():
        return {
            "status": "online",
            "service": "Financial AI Assistant API",
            "version": "2.0.0",
            "documentation": "/docs",
            "health_check": "/api/health",
            "message": "Frontend build not detected. Build webapp (npm run build) to serve unified UI."
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Vector Store Status on startup: {vector_store.check_status()}")
    logger.info(f"Starting Financial AI Assistant FastAPI Server on http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=reload, app_dir=str(BACKEND_DIR))

