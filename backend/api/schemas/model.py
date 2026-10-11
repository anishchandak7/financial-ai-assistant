from typing import List, Optional, Any
from pydantic import BaseModel, Field

class SetModelRequest(BaseModel):
    model: str = Field(..., description="Target model name or display name to activate")

class ModelsResponse(BaseModel):
    models: List[Any]
    display_names: List[str]
    active_model: Optional[str]
