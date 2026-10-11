from typing import List, Dict, Any
from pydantic import BaseModel

class MarketOverviewResponse(BaseModel):
    markets: List[Dict[str, Any]]
