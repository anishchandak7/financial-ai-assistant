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

def log_rag_run_feedback(
    run_id: str,
    faithfulness: float,
    answer_relevance: float,
    context_relevance: float,
    overall_rag_score: Optional[float] = None
):
    """Attaches reference-free evaluation metrics as structured feedback on a LangSmith trace run."""
    client = get_langsmith_client()
    if not client or not run_id:
        return
    try:
        client.create_feedback(run_id, key="faithfulness", score=faithfulness)
        client.create_feedback(run_id, key="answer_relevance", score=answer_relevance)
        client.create_feedback(run_id, key="context_relevance", score=context_relevance)
        if overall_rag_score is not None:
            client.create_feedback(run_id, key="overall_rag_score", score=overall_rag_score)
        logger.info(f"Attached reference-free evaluation feedback to LangSmith run: {run_id}")
    except Exception as e:
        logger.warning(f"Failed to log feedback to LangSmith for run '{run_id}': {e}")

