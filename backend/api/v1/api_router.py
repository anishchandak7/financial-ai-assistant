from fastapi import APIRouter
from api.v1.endpoints import chat, market, documents, models, health

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Metrics"])
api_router.include_router(models.router, tags=["Models"])
api_router.include_router(documents.router, tags=["Documents & RAG"])
api_router.include_router(market.router, tags=["Market Intelligence"])
api_router.include_router(chat.router, tags=["Multi-Agent Chat"])
