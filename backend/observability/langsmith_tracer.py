from typing import Optional, Dict, Any
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

def get_langsmith_client() -> Optional[Client]:
    """Returns an authenticated LangSmith Client instance if configured, else None."""
    from core.config import settings
    if not settings.LANGSMITH_API_KEY:
        return None
    try:
        return Client(
            api_key=settings.LANGSMITH_API_KEY,
            api_url=settings.LANGSMITH_ENDPOINT
        )
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith Client: {e}")
        return None

def get_langsmith_tracer() -> Optional[LangChainTracer]:
    """Returns a LangChainTracer configured for the active LangSmith project."""
    from core.config import settings
    if not settings.LANGSMITH_TRACING or not settings.LANGSMITH_API_KEY:
        return None
    try:
        return LangChainTracer(
            project_name=settings.LANGSMITH_PROJECT,
            client=get_langsmith_client()
        )
    except Exception as e:
        logger.warning(f"Failed to create LangChainTracer: {e}")
        return None

def check_langsmith_status() -> Dict[str, Any]:
    """Returns telemetry diagnostic information about the LangSmith configuration."""
    from core.config import settings
    api_key_present = bool(settings.LANGSMITH_API_KEY)
    is_enabled = settings.LANGSMITH_TRACING and api_key_present

    return {
        "enabled": is_enabled,
        "project": settings.LANGSMITH_PROJECT,
        "endpoint": settings.LANGSMITH_ENDPOINT,
        "api_key_configured": api_key_present,
        "message": (
            f"LangSmith is active and logging traces to project '{settings.LANGSMITH_PROJECT}'"
            if is_enabled
            else "LangSmith tracing is disabled. Set LANGSMITH_API_KEY in backend/.env to activate."
        )
    }
