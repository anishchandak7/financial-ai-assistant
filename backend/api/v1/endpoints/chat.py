from fastapi import APIRouter, HTTPException
from api.schemas.chat import ChatRequest, ChatResponse
from services.orchestration_service import orchestration_service
from memory.manager import agent_memory_manager
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Executes the adaptive multi-agent workflow on the user prompt with per-agent session memory."""
    try:
        logger.info(f"API Chat request received: {req.prompt[:100]} (Session: {req.session_id})")
        result = orchestration_service.execute(
            prompt=req.prompt,
            model_override=req.model,
            session_id=req.session_id
        )
        return ChatResponse(
            success=True,
            result=result,
            session_id=result.get("session_id")
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/session/reset")
async def reset_session_endpoint(session_id: str):
    """Clears all isolated agent memories for a given conversation session."""
    try:
        cleared = agent_memory_manager.reset_session(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "cleared": cleared,
            "message": f"Session memory {'successfully reset' if cleared else 'not found or already empty'}."
        }
    except Exception as e:
        logger.error(f"Error resetting session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/session/{session_id}/memory")
async def get_session_memory_endpoint(session_id: str):
    """Returns the per-agent memory partitions and working scratchpad for telemetry and inspection."""
    try:
        session = agent_memory_manager.get_or_create_session(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "active_ticker": session.active_ticker,
            "last_active_agent": session.last_active_agent,
            "agents": {
                name: {
                    "turns_count": len(agent_mem.turns),
                    "turns": [t.model_dump() for t in agent_mem.turns],
                    "working_memory": agent_mem.working_memory
                }
                for name, agent_mem in session.agents.items()
            }
        }
    except Exception as e:
        logger.error(f"Error inspecting session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
