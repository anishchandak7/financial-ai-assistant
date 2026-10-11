from typing import Dict, Any, Optional
from agents.orchestrator.state import GraphState
from agents.orchestrator.graph import orchestrator_graph
from services.model_service import model_service
from observability.logger import get_custom_logger, current_trace_id, current_request_id
from observability.tracer import trace_span
from observability.langsmith_tracer import get_langsmith_tracer

logger = get_custom_logger(name=__name__)

class OrchestrationService:
    """Coordinates execution of the multi-agent orchestrator graph with LangSmith telemetry."""

    def __init__(self) -> None:
        self.workflow = orchestrator_graph

    def set_model(self, selected_model: str):
        model_service.set_active_model(selected_model)

    def execute(self, prompt: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Executes the adaptive multi-agent graph with guardrails and LangSmith telemetry."""
        if model_override:
            self.set_model(model_override)

        active_model = model_service.active_model_name
        logger.info(f"Orchestrating user prompt using model: {active_model}")

        state: GraphState = {
            "model_name": active_model,
            "input_query": prompt,
            "agent_output": {},
            "citations": []
        }

        # Build LangSmith configuration and run metadata
        config: Dict[str, Any] = {
            "run_name": "Financial AI Assistant Orchestration",
            "tags": ["financial-assistant", "multi-agent", active_model],
            "metadata": {
                "model_name": active_model,
                "trace_id": current_trace_id.get(),
                "request_id": current_request_id.get()
            }
        }

        ls_tracer = get_langsmith_tracer()
        if ls_tracer:
            config["callbacks"] = [ls_tracer]

        with trace_span("orchestrator.execute", {"model": active_model, "query": prompt[:80]}):
            try:
                result = self.workflow.invoke(input=state, config=config)
                logger.info("Multi-agent workflow execution completed successfully.")
                return result
            except Exception as e:
                logger.error(f"Failed to execute multi-agent workflow: {e}", exc_info=True)
                raise e

orchestration_service = OrchestrationService()
