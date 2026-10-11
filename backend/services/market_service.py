from typing import List, Optional, Tuple, Dict, Any
import yfinance as yf
from core.config import settings
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

class MarketService:
    """Provides market index overview and ticker metrics via Yahoo Finance."""

    def __init__(self) -> None:
        self.markets: List[str] = []
        if settings.MARKETS is not None:
            self.markets = [m.strip() for m in settings.MARKETS.split(",") if m.strip()]

    def get_market_summary(self, market_name: str) -> Optional[Tuple[Any, Any]]:
        """Retrieves market status and ticker summaries."""
        try:
            if not self.markets:
                logger.warning("No market names are configured in MARKETS setting.")
                return None
            if market_name not in self.markets:
                logger.warning(f"Market '{market_name}' not in configured markets list.")
                return None

            market = yf.Market(market=market_name)
            return market.status, market.summary
        except Exception as e:
            logger.error(f"Failed to get market summary for '{market_name}': {e}", exc_info=True)
            raise e

    def get_all_markets_overview(self) -> List[Dict[str, Any]]:
        """Retrieves formatted status and ticker metrics for all configured markets."""
        results = []
        for market in self.markets:
            summary_data = self.get_market_summary(market_name=market)
            if not summary_data:
                continue
            status, summary = summary_data

            if isinstance(status, dict):
                m_name = status.get("name", market.upper())
                m_state = status.get("status", "UNKNOWN").upper()
                m_msg = status.get("message", "No message provided.")
            else:
                m_name = f"{market.upper()} Market"
                m_state = "N/A"
                m_msg = "Market status endpoint not supported for this region."

            tickers_list = []
            if isinstance(summary, dict) and summary:
                for ticker, data in summary.items():
                    if not isinstance(data, dict):
                        continue
                    name = data.get("shortName", ticker)
                    price = data.get("regularMarketPrice", 0.0)
                    change = data.get("regularMarketChange", 0.0)
                    change_percent = data.get("regularMarketChangePercent", 0.0)
                    tickers_list.append({
                        "symbol": ticker,
                        "name": name,
                        "price": price,
                        "change": change,
                        "change_percent": change_percent
                    })

            results.append({
                "market": market,
                "name": m_name,
                "state": m_state,
                "message": m_msg,
                "tickers": tickers_list
            })
        return results

market_service = MarketService()
