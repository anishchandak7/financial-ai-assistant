import yfinance as yf
from core.logger import get_custom_logger
from core.config import settings

logger = get_custom_logger(name=__name__)

class MarketDataService:
    
    def __init__(self) -> None:
        self.markets = []
        if settings.MARKETS is not None:
            self.markets:list = settings.MARKETS.split(",")
            self.markets = [market.strip() for market in self.markets]
    
    def get_market_summary(self, market_name:str):
        """
        Get market summary based on market_name.
        """
        try:
            if not self.markets:
                logger.warning(msg="No market names are configured. Cannot retrieve market summary.")
                return
            if market_name not in self.markets:
                logger.warning(msg=f"{market_name} not present in the configured list. Hence, cannot proceed.")
                return
            market = yf.Market(market=market_name)
            logger.info(msg=f"{market_name} logs:")
            if market.status:
                logger.info(msg="Status Fetched")
            if market.summary:
                logger.info(msg="Summary Fetched")
            return market.status, market.summary
        except Exception as e:
            logger.error(msg="Failed to get market summary", exc_info=True, stack_info=True)
            raise e
            
        

