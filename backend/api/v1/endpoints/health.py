from fastapi import APIRouter, Response
from rag.vector_store import vector_store
from services.market_service import market_service
from observability.metrics import metrics

from observability.langsmith_tracer import check_langsmith_status

router = APIRouter()

@router.get("/health")
async def health_check():
    """Check application health, vector store heartbeat, and LangSmith telemetry status."""
    try:
        vs_status = vector_store.check_status()
        ls_status = check_langsmith_status()
        return {
            "status": "healthy",
            "vector_store_status": vs_status,
            "markets_configured": market_service.markets,
            "langsmith_telemetry": ls_status
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@router.get("/metrics")
async def get_prometheus_metrics():
    """Expose Prometheus-formatted telemetry metrics for scrapers."""
    content = metrics.export_prometheus()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")

@router.get("/metrics/summary")
async def get_metrics_summary():
    """Expose structured JSON metrics summary for internal monitoring."""
    return metrics.get_summary()
