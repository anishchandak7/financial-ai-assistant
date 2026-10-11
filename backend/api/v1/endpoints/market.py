from fastapi import APIRouter, HTTPException
from api.schemas.market import MarketOverviewResponse
from services.market_service import market_service
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

router = APIRouter()

@router.get("/markets", response_model=MarketOverviewResponse)
async def get_market_overview():
    """Fetch live market summaries and index data for all configured markets."""
    try:
        results = market_service.get_all_markets_overview()
        return MarketOverviewResponse(markets=results)
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
