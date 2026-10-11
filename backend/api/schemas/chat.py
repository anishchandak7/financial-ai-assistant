from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User question or financial inquiry")
    model: Optional[str] = Field(None, description="Optional LLM model override")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID for tracing")

class ChatResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    session_id: Optional[str] = Field(None, description="Active conversation session ID")
    error: Optional[str] = None
