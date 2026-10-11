from fastapi import APIRouter, HTTPException
from api.schemas.model import ModelsResponse, SetModelRequest
from services.model_service import model_service
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

router = APIRouter()

@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Retrieve available LLM models."""
    try:
        models = model_service.get_available_models(details="all")
        display_names = model_service.get_available_models(details="display_name")
        active = model_service.active_display_name
        return ModelsResponse(
            models=models,
            display_names=display_names,
            active_model=active
        )
    except Exception as e:
        logger.error(f"Error fetching models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/set")
async def set_active_model(req: SetModelRequest):
    """Set the active LLM model."""
    try:
        active = model_service.set_active_model(req.model)
        return {"success": True, "active_model": active}
    except Exception as e:
        logger.error(f"Error setting model: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
