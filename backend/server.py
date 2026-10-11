import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
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

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Production Observability Middleware (Request/Trace correlation & telemetry)
app.add_middleware(ObservabilityMiddleware)

# Mount modular API routes (preserving /api/... for frontend compatibility)
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Vector Store Status on startup: {vector_store.check_status()}")
    logger.info("Starting Financial AI Assistant FastAPI Server on http://127.0.0.1:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(BACKEND_DIR))
