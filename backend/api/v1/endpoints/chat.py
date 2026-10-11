from fastapi import APIRouter, HTTPException
from api.schemas.chat import ChatRequest, ChatResponse
from services.orchestration_service import orchestration_service
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Executes the adaptive multi-agent workflow on the user prompt."""
    try:
        logger.info(f"API Chat request received: {req.prompt[:100]}")
        result = orchestration_service.execute(
            prompt=req.prompt,
            model_override=req.model
        )
        return ChatResponse(
            success=True,
            result=result
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
